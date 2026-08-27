"""
Dual-head value/policy net (docs/AGENT_REBUILD_PLAN.md §2.2): an MLP trunk
feeding a value head (softmax over num_players -- P(each seat wins) from
the encoding's observer) and a policy head (4 logits, one per piece_id,
masked to legal actions by the caller).

Two forward paths compute the SAME function from the SAME weights:
- AZNet (torch): autodiff, MPS on the M4 -- used for the training step.
- NumpyAZNet: no torch call/dispatch overhead -- used for search's batched
  leaf evaluation, where inference batches are 50-1000 rows of a small MLP
  and per-call overhead dominates raw FLOPs more than matmul size does.

See parchis/tests/test_net.py::test_numpy_and_torch_forward_agree for the
cross-check that both paths agree to the tolerance Part 3 item 7 asks for
(1e-5). Neither forward() applies softmax/masking -- callers (search.py)
do that via masked_policy_probs/value_probs below, since only the caller
knows which actions are legal.
"""

import numpy as np
import torch
import torch.nn as nn

DEFAULT_HIDDEN_SIZES = (256, 256)
NUM_ACTIONS = 4  # Discrete(4): which piece to move, fixed slot by piece_id
LAYER_NORM_EPS = 1e-5  # torch.nn.LayerNorm's own default -- NumpyAZNet must match exactly


class AZNet(nn.Module):
    """MLP trunk ([256, 256] by default) + ReLU + LayerNorm per layer, then
    two linear heads. See module docstring for why raw logits are
    returned, not probabilities."""

    def __init__(self, input_size, num_players, hidden_sizes=DEFAULT_HIDDEN_SIZES):
        super().__init__()
        self.input_size = input_size
        self.num_players = num_players
        self.hidden_sizes = tuple(hidden_sizes)

        layers = []
        prev = input_size
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h, eps=LAYER_NORM_EPS))
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, NUM_ACTIONS)
        self.value_head = nn.Linear(prev, num_players)

    def forward(self, x):
        """x: (batch, input_size) float32 tensor.
        Returns (policy_logits (batch, 4), value_logits (batch, num_players)),
        both raw (no softmax)."""
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h)

    def numpy_weights(self):
        """Extract this model's CURRENT weights as the plain-array bundle
        NumpyAZNet.from_torch needs. Call again after every training step
        before search's numpy path picks up the update -- this is a
        snapshot, not a live view."""
        linear_layers = [m for m in self.trunk if isinstance(m, nn.Linear)]
        norm_layers = [m for m in self.trunk if isinstance(m, nn.LayerNorm)]
        trunk_layers = tuple(
            (
                linear.weight.detach().cpu().numpy().astype(np.float32),
                linear.bias.detach().cpu().numpy().astype(np.float32),
                norm.weight.detach().cpu().numpy().astype(np.float32),
                norm.bias.detach().cpu().numpy().astype(np.float32),
            )
            for linear, norm in zip(linear_layers, norm_layers)
        )
        return {
            "trunk_layers": trunk_layers,
            "policy_W": self.policy_head.weight.detach().cpu().numpy().astype(np.float32),
            "policy_b": self.policy_head.bias.detach().cpu().numpy().astype(np.float32),
            "value_W": self.value_head.weight.detach().cpu().numpy().astype(np.float32),
            "value_b": self.value_head.bias.detach().cpu().numpy().astype(np.float32),
        }


def _layer_norm(x, gamma, beta, eps=LAYER_NORM_EPS):
    """Per-row LayerNorm over the last axis, matching torch.nn.LayerNorm's
    formula exactly: biased variance (divide by N, not N-1 -- np.var's
    default ddof=0 already matches), same eps."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


class NumpyAZNet:
    """NumPy-only forward path computing the identical function as AZNet,
    from weights extracted via AZNet.numpy_weights(). No torch import
    needed at inference time (see module docstring for why this exists)."""

    def __init__(self, weights):
        self.weights = weights

    @classmethod
    def from_torch(cls, az_net):
        return cls(az_net.numpy_weights())

    def forward(self, x):
        """x: (batch, input_size) array-like, any float dtype.
        Returns (policy_logits (batch, 4), value_logits (batch, num_players))
        as float32 numpy arrays, matching AZNet.forward's contract."""
        h = np.asarray(x, dtype=np.float32)
        if h.ndim == 1:
            h = h[None, :]
        for W, b, gamma, beta in self.weights["trunk_layers"]:
            h = h @ W.T + b
            h = np.maximum(h, 0.0)
            h = _layer_norm(h, gamma, beta)
        policy_logits = h @ self.weights["policy_W"].T + self.weights["policy_b"]
        value_logits = h @ self.weights["value_W"].T + self.weights["value_b"]
        return policy_logits, value_logits


def masked_policy_probs(policy_logits, legal_mask):
    """Softmax over ONLY the legal actions (legal_mask: same shape as
    policy_logits, 1.0/truthy for legal, 0.0/falsy for illegal) -- illegal
    actions get exactly 0.0 probability, never a tiny nonzero leak.
    Broadcasts over a leading batch dimension if present."""
    logits = np.asarray(policy_logits, dtype=np.float32)
    mask = np.asarray(legal_mask, dtype=bool)
    masked_logits = np.where(mask, logits, -np.inf)
    masked_logits = masked_logits - np.max(masked_logits, axis=-1, keepdims=True)
    exp = np.where(mask, np.exp(masked_logits), 0.0)
    total = exp.sum(axis=-1, keepdims=True)
    return exp / total


def value_probs(value_logits):
    """Plain softmax over the last axis -- the value head's whole output IS
    the per-seat win-probability distribution; there's no "illegal seat" to
    mask out."""
    logits = np.asarray(value_logits, dtype=np.float32)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)
