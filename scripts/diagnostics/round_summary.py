"""Per-round summary of a self-play run from its shards + metrics (docs/V2_PLAN.md §6/§10).

Usage: round_summary.py <run_dir> [tau_target=0.015]

Decodes outcome and depth-1 root value from the stored blended value target
(z = (1-lam)*outcome + lam*root with lam = 0.5 and outcome in {0,1}: outcome = [z > 0.5],
root = 2z - outcome; truncated games, outcome 0.5, are rare and misread as wins), and the
best-vs-second move gap from the stored policy target (gap = tau_target * (log p1 - log p2)).
Columns: training losses at the last epoch; mover-channel bias/ECE of the root value; move gaps;
P(best) of the acting policy at tau_target (equal to the target's max prob when tau_end ==
tau_target); expected win-probability given away per multi-move decision by that policy; the
candidate-vs-previous telemetry match. Rows with policy weight 0 -- afterstate rows (C3) and, under
deep_targets (C2), cheap decisions whose value target is the outcome alone -- are excluded from every
column and counted separately at the end of the line, so with deep_targets the columns describe the
deep decisions only.
"""
import glob
import json
import os
import sys

import numpy as np

run_dir = sys.argv[1]
TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.015
LAM = json.load(open(os.path.join(run_dir, "config.json"))).get("lam", 0.5)
assert abs(LAM - 0.5) < 1e-9, "decoding assumes lam = 0.5"

print("round  rows/game epochs  val_val val_pol val_aux | win_share bias(pp)  ECE  | gap_mean gap_med <1pp%  P(best) loss/dec(pp) | wr   a/split/b")
for r in sorted(glob.glob(os.path.join(run_dir, "rounds", "round_*"))):
    m = [json.loads(l) for l in open(os.path.join(r, "metrics.jsonl"))]
    last = m[-1]
    p = json.load(open(os.path.join(r, "promotion_result.json")))
    cfg_games = json.load(open(os.path.join(run_dir, "config.json")))["n_games_per_round"]
    zs, gaps, pmax, loss = [], [], [], []
    n = n_after = 0
    for s in sorted(glob.glob(os.path.join(r, "shards", "*.npz"))):
        d = np.load(s)
        # Afterstate rows (C3) carry policy weight 0; all columns below are over decision rows only.
        dec = d["policy_weights"] > 0 if "policy_weights" in d else np.ones(len(d["value_targets"]), bool)
        vt = d["value_targets"][dec, 0].astype(np.float64)
        pt = d["policy_targets"][dec].astype(np.float64)
        n += int(dec.sum())
        n_after += int((~dec).sum())
        zs.append(vt)
        q = pt[(pt > 0).sum(1) > 1]
        lp = np.log(np.where(q > 0, q, 1.0))
        mx = lp.max(1, keepdims=True)
        gap_all = np.where(q > 0, TAU * (mx - lp), 0.0)
        srt = np.sort(q, 1)[:, ::-1]
        gaps.append(TAU * (np.log(srt[:, 0]) - np.log(np.maximum(srt[:, 1], 1e-30))))
        pmax.append(srt[:, 0])
        loss.append((q * gap_all).sum(1))
    z = np.concatenate(zs); g = np.concatenate(gaps); pm = np.concatenate(pmax); L = np.concatenate(loss)
    o = (z > 0.5).astype(float)
    rv = np.where(z > 0.5, 2 * z - 1, 2 * z)
    bins = np.clip((rv * 10).astype(int), 0, 9)
    ece = sum((bins == b).mean() * abs(rv[bins == b].mean() - o[bins == b].mean()) for b in range(10) if (bins == b).any())
    pr = p["pair_record"]
    print(f"{os.path.basename(r)[-2:]:>5}  {n / cfg_games:8.1f} {len(m):>5}   {last['val_value_loss']:.4f} {last['val_policy_loss']:.4f} {last['val_aux_loss']:.4f} | "
          f"{o.mean():.3f}     {100 * (rv.mean() - o.mean()):+5.2f}  {ece:.3f} | {100 * g.mean():6.2f}   {100 * np.median(g):5.2f}  {100 * (g < 0.01).mean():5.1f}   "
          f"{pm.mean():.3f}   {100 * L.mean():.3f}     | {p['win_rate_a']:.3f} {pr['a_better']}/{pr['split']}/{pr['b_better']}"
          + (f"   (+{n_after / cfg_games:.0f} zero-policy-weight rows/game)" if n_after else ""))
