"""
Dual-head value/policy net (docs/AGENT_REBUILD_PLAN.md §2.2): an MLP trunk
feeding a value head (softmax over num_players -- P(each seat wins) from
the encoding's observer) and a policy head (4 logits, one per piece_id,
masked to legal actions by the caller). A third, auxiliary head (Phase
4.1, .claude/plans/twinkly-marinating-hinton.md) was added after three
training-loop-level fixes (escalation retirement, opponent-pool
broadening, rollout-refined targets) ran for 50 rounds combined with no
detectable strength improvement -- see docs/AZ_DESIGN.md's
"Strength-improvement plan" entry for the full experimental record. The
aux head predicts, for each of the mover's own 4 pieces, whether it goes
on to finish by game end (a KataGo-ownership-head analogue: free extra
supervision from games already being generated, no new data-generation
cost) -- see parchis.az.selfplay's module docstring for how the target
is computed, and parchis.az.train for how its loss is combined with the
existing two.

Two forward paths compute the SAME function from the SAME weights:
- AZNet (torch): autodiff, MPS on the M4 -- used for the training step.
- NumpyAZNet: no torch call/dispatch overhead -- used for search's batched
  leaf evaluation, where inference batches are 50-1000 rows of a small MLP
  and per-call overhead dominates raw FLOPs more than matmul size does.
  Deliberately does NOT grow an aux-head path: the aux head only shapes
  training gradients on the shared trunk, search never consults it, so
  NumpyAZNet's forward() keeps its original 2-tuple (policy, value)
  contract unchanged -- see numpy_weights() below.

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
NUM_AUX_TARGETS = 4  # one per own piece_id -- numerically the same as NUM_ACTIONS
                      # (every piece has exactly one associated action in this game),
                      # kept as its own name since the two represent different things.
LAYER_NORM_EPS = 1e-5  # torch.nn.LayerNorm's own default -- NumpyAZNet must match exactly


class AZNet(nn.Module):
    """MLP trunk ([256, 256] by default) + ReLU + LayerNorm per layer, then
    three linear heads (policy, value, aux). See module docstring for why
    raw logits are returned, not probabilities, and for the aux head's
    purpose."""

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
        self.aux_head = nn.Linear(prev, NUM_AUX_TARGETS)

    def forward(self, x):
        """x: (batch, input_size) float32 tensor.
        Returns (policy_logits (batch, 4), value_logits (batch, num_players),
        aux_logits (batch, 4)), all raw (no softmax/sigmoid)."""
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h), self.aux_head(h)

    def load_state_dict_compat(self, state_dict):
        """Loads `state_dict` into this model, tolerating a checkpoint
        saved before the aux head existed (missing 'aux_head.weight'/
        'aux_head.bias') by leaving the aux head at its own fresh random
        init in that one, specific, understood case -- the aux head only
        shapes training gradients (search never consults it), so an old
        checkpoint's warm-start doesn't need to carry it forward; it
        starts learning from the very next round's data instead. This is
        a NARROW migration path for exactly that transition, not a
        blanket strict=False: any OTHER mismatch (a real bug, a genuine
        shape error) still goes through strict=True and raises its usual,
        detailed error, so this can't silently mask something unrelated.
        """
        own_keys = set(self.state_dict().keys())
        aux_keys = {k for k in own_keys if k.startswith("aux_head.")}
        is_exactly_missing_aux_head = set(state_dict.keys()) == own_keys - aux_keys
        if is_exactly_missing_aux_head:
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict)

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
