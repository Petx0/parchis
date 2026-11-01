"""Raw value-head quality of several nets on one fixed shard (docs/V2_PLAN.md §10.1).

Usage: value_quality.py <shard.npz> name=model.pt [name=model.pt ...] [--n 60000]

Outcome is decoded from the blended target as in round_summary.py (lam = 0.5). Reports log-loss,
Brier, mean bias, 10-bin ECE, prediction std, share of confident predictions, and the mean
prediction gap between eventual winners and losers (resolution).
"""
import sys

import numpy as np
from _common import setup

setup()
from parchis.az import champion_pool
from parchis.az.net import value_probs

argv = list(sys.argv[1:])
n_rows = 60000
if "--n" in argv:
    i = argv.index("--n")
    n_rows = int(argv[i + 1])
    del argv[i:i + 2]
args = argv
shard = args[0]
paths = dict(a.split("=", 1) for a in args[1:])
d = np.load(shard)
X = d["X"]
z = d["value_targets"][:, 0].astype(np.float64)
o = (z > 0.5).astype(float)
idx = np.random.default_rng(0).choice(len(X), min(n_rows, len(X)), replace=False)
X, o = X[idx], o[idx]
print(f"shard={shard} rows={len(X)} mover-win share={o.mean():.3f}")
print("net    logloss  brier   bias(pp)  ECE10  pred_std  |p-.5|>.4%  sep(win-loss)")
for k, p_ in paths.items():
    net = champion_pool.load_numpy_net(p_, 216, 2, (256, 256))
    _, vl = net.forward(X)
    p = value_probs(vl)[:, 0].astype(np.float64)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(o * np.log(pc) + (1 - o) * np.log(1 - pc)).mean()
    br = ((p - o) ** 2).mean()
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum((bins == b).mean() * abs(p[bins == b].mean() - o[bins == b].mean()) for b in range(10) if (bins == b).any())
    print(f"{k:<6} {ll:.4f}  {br:.4f}  {100 * (p.mean() - o.mean()):+6.2f}   {ece:.3f}   {p.std():.3f}    "
          f"{100 * (np.abs(p - .5) > .4).mean():5.1f}      {p[o == 1].mean() - p[o == 0].mean():.3f}")
