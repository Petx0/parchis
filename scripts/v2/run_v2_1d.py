"""V2.1d (docs/V2_PLAN.md §10.7 next): the V2.1c recipe with deep_fraction 0.5 instead of 0.25 -- does
the C2 gain grow with the share of depth-2 targets? 20 ungated rounds from the V2.0b champion, the same
start and length as V2.1c, so the two endpoints are compared head to head (scripts/v2/h2h.py, 10,000
pairs, Wilson lower bound > 50%). Control = runs/v2_1c_champion/model.pt.

Config = V2.1c (tau_end 0.005, lr 3e-4, no random anchor, heuristic 1/15, deep_targets, afterstate_rows,
8 spawn workers, 8 shards of 750 games) with deep_fraction 0.5 at deep_depth 2. Expected ~3 games/s per
process (~25 on 8 workers), rounds ~6 min. Ladders every 5 rounds against the standard anchors
(heuristic, round 23, V2.0, V2.0b, V2.0c, V2.1c).

Usage: run_v2_1d.py <runs_dir>      (shards: ~2.4 GB/round; use a local non-iCloud directory)
Post hoc: scripts/diagnostics/round_summary.py <runs_dir>/v2_1d   (columns = deep decisions only)
          h2h.py runs/v2_1d_champion/model.pt runs/v2_1c_champion/model.pt <out.json> 10000
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from ladder import ANCHORS, run_ladder

INIT = REPO / "runs/v2_0b_champion/model.pt"

if __name__ == "__main__":
    RUNS = Path(sys.argv[1])
    RUNS.mkdir(parents=True, exist_ok=True)
    from parchis.az import config as C, round_loop
    cfg = C.SelfPlayRoundConfig(
        run_name="v2_1d", num_players=2, n_games_per_round=6000, games_per_shard=750,
        generation_workers=8,
        tau_start=0.03, tau_end=0.005, dirichlet_epsilon=0.0, tau_target=0.015,
        enable_escalation=False, aux_loss_weight=0.2, gate_enabled=False,
        promotion_n_pairs=100,
        learning_rate=3e-4,
        pool_random_anchor=False, pool_heuristic_fraction=1 / 15,
        deep_fraction=0.5, deep_depth=2, deep_targets=True,
        afterstate_rows=True,
    )
    anchors = dict(ANCHORS)
    init = torch.load(INIT, map_location="cpu")
    t0 = time.time()
    for stop in (5, 10, 15, 20):
        round_loop.run_continuous(cfg, runs_dir=str(RUNS), max_rounds=stop, initial_champion_state_dict=init)
        print(f"== {stop} rounds done, {(time.time() - t0) / 60:.0f} min since start", flush=True)
        run_ladder(RUNS / "v2_1d" / "champion.pt", RUNS / f"ladder_round{stop}.json", tag=f"round{stop}",
                   anchors=anchors)
