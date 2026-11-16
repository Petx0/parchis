"""Weight soups: average the weights of AZNet checkpoints from one lineage (PILA-8).

SWA / model soups (Izmailov et al. 2018, arXiv:1803.05407; Wortsman et al. 2022, arXiv:2203.05482):
nets fine-tuned from a shared start often sit in one loss basin, and the average of their weights
lands nearer its centre than any single endpoint. AZNet normalises with LayerNorm, not BatchNorm, so
an averaged state dict needs no statistics recomputed: it is usable as soon as it is written.

  * `average_state_dicts` -- the (weighted) average, refusing ingredients whose keys, shapes or
    dtypes differ.
  * `make_soup`           -- load checkpoints, average, write `<out>.pt` and `config.json` beside it.
  * `shard_losses`        -- value / policy log-loss of one state dict on self-play shards, the
    health check that drops a soup sitting on a loss barrier between its ingredients.
"""
import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# A soup is dropped before any match when its value log-loss is worse than its WORST ingredient's by
# more than this (PILA-8 kill criterion): it sits on a barrier, not inside the basin.
HEALTH_MARGIN = 0.01


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalise_weights(weights, n):
    """Mixing weights summing to 1. None -> uniform. Each weight must be > 0."""
    if weights is None:
        return [1.0 / n] * n
    weights = [float(w) for w in weights]
    if len(weights) != n:
        raise ValueError(f"{len(weights)} weights for {n} ingredients")
    if any(not np.isfinite(w) or w <= 0 for w in weights):
        raise ValueError(f"weights must be finite and > 0, got {weights}")
    total = sum(weights)
    return [w / total for w in weights]


def average_state_dicts(state_dicts, weights=None):
    """Weighted average of state dicts with identical keys, shapes and dtypes (float tensors only).
    Accumulates in float64 and casts back, so the order of the ingredients does not matter."""
    state_dicts = list(state_dicts)
    if len(state_dicts) < 2:
        raise ValueError("a soup needs at least 2 ingredients")
    weights = normalise_weights(weights, len(state_dicts))
    ref = state_dicts[0]
    keys = list(ref.keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != set(keys):
            missing, extra = sorted(set(keys) - set(sd)), sorted(set(sd) - set(keys))
            raise ValueError(f"ingredient {i} keys differ from ingredient 0: missing {missing}, extra {extra}")
        for k in keys:
            if tuple(sd[k].shape) != tuple(ref[k].shape):
                raise ValueError(f"ingredient {i} {k}: shape {tuple(sd[k].shape)} != {tuple(ref[k].shape)}")
            if sd[k].dtype != ref[k].dtype:
                raise ValueError(f"ingredient {i} {k}: dtype {sd[k].dtype} != {ref[k].dtype}")
    out = OrderedDict()
    for k in keys:
        if not torch.is_floating_point(ref[k]):
            raise ValueError(f"{k} is {ref[k].dtype}; only float tensors can be averaged")
        acc = torch.zeros(ref[k].shape, dtype=torch.float64)
        for w, sd in zip(weights, state_dicts):
            acc += w * sd[k].detach().to(torch.float64)
        out[k] = acc.to(ref[k].dtype)
    return out


def make_soup(out_path, ingredient_paths, weights=None, name=None, extra=None):
    """Average the checkpoints at `ingredient_paths` into `out_path` and write `config.json` in the
    same directory listing the ingredients (path + sha256) and their weights. Returns the config."""
    out_path = Path(out_path)
    paths = [Path(p) for p in ingredient_paths]
    sds = [torch.load(p, map_location="cpu") for p in paths]
    w = normalise_weights(weights, len(paths))
    soup = average_state_dicts(sds, w)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup, out_path)
    from parchis.registry.record import rel
    config = dict(
        name=name or out_path.parent.name,
        method="weight average (model soup)",
        weighting="uniform" if weights is None else "explicit",
        ingredients=[dict(path=rel(p), sha256=_sha256(p), weight=wi) for p, wi in zip(paths, w)],
        output=dict(path=rel(out_path), sha256=_sha256(out_path)),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if extra:
        config.update(extra)
    (out_path.parent / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def hidden_sizes_from_state_dict(sd):
    layers = sorted((int(k.split(".")[1]), v.shape[0]) for k, v in sd.items()
                    if k.startswith("trunk.") and k.endswith(".weight") and v.dim() == 2)
    return tuple(width for _, width in layers)


def load_shards(shard_paths):
    """(X, policy_targets, value_targets, policy_weights) of the shards, concatenated."""
    from parchis.az import train
    X, p, v, _aux_t, _aux_w, pw = train._load_and_concat_shards([str(s) for s in shard_paths])
    return X, p, v, pw


def shard_losses(state_dict, data, num_players=2, batch_size=65536):
    """Mean value log-loss (cross-entropy against the shard's value targets, the training value loss)
    and policy log-loss (weighted as in training: afterstate rows carry 0) of one state dict on
    `data` = load_shards(...). Also the value loss on decision rows only (policy weight > 0)."""
    from parchis.az import encoding
    from parchis.az.net import AZNet
    X, p, v, pw = data
    model = AZNet(encoding.encoding_size(num_players), num_players,
                  hidden_sizes=hidden_sizes_from_state_dict(state_dict))
    model.load_state_dict_compat(state_dict)
    model.eval()
    v_sum = pol_sum = vdec_sum = 0.0
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[s:s + batch_size]).float()
            pol_logits, val_logits, _ = model(xb)
            vt = torch.from_numpy(v[s:s + batch_size]).float()
            wb = torch.from_numpy(pw[s:s + batch_size]).float()
            v_rows = torch.nn.functional.cross_entropy(val_logits, vt, reduction="none")
            pt = torch.from_numpy(p[s:s + batch_size])          # class indices or probability rows
            pt = pt.float() if pt.dim() == 2 else pt.long()
            p_rows = torch.nn.functional.cross_entropy(pol_logits, pt, reduction="none")
            v_sum += v_rows.sum().item()
            vdec_sum += (v_rows * (wb > 0)).sum().item()
            pol_sum += (p_rows * wb).sum().item()
    n, n_dec, w_sum = len(X), int((pw > 0).sum()), float(pw.sum())
    return dict(value_loss=v_sum / n, value_loss_decisions=vdec_sum / max(n_dec, 1),
                policy_loss=pol_sum / max(w_sum, 1e-12), n_rows=n, n_decision_rows=n_dec)


def health_verdict(soup_value_loss, ingredient_value_losses, margin=HEALTH_MARGIN):
    """True (keep) unless the soup's value loss is worse than its worst ingredient's by > margin."""
    worst = max(ingredient_value_losses)
    return soup_value_loss <= worst + margin
