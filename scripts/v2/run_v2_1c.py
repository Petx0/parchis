"""V2.1c (docs/V2_PLAN.md §10.4 next steps, item 3): C2 deep targets + the chance-resolved afterstate
target, 20 ungated rounds from the V2.0b champion -- the same start and length as the V2.0c control, so
the two endpoints are compared head to head (>= 2,000 pairs, Wilson lower bound > 50%).

Config = V2.0b/V2.0c (tau_end 0.005, lr 3e-4, no random anchor, heuristic 1/15) plus deep_fraction 0.25
at deep_depth 2 with deep_targets=True (cheap decisions: outcome-only value target, policy weight 0;
deep decisions: blended target + policy target) and afterstate_rows=True (a deep decision's afterstate
row bootstraps from the depth-2 value of the played move; a cheap one from the outcome alone).
Generation on 8 spawn workers (C6), 8 shards of 750 games. Ladders every 5 rounds against the tuned
heuristic, the round-23, V2.0 and V2.0b champions and -- when runs/v2_0c_champion/model.pt exists --
the V2.0c control.

Usage: run_v2_1c.py <runs_dir>      (shards: ~2.4 GB/round; use a local non-iCloud directory)
Post hoc: scripts/diagnostics/round_summary.py <runs_dir>/v2_1c   (columns = deep decisions only)
          leaf_and_target_check.py on the result (pre-roll/post-roll consistency, gate < 0.5pp)
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
V2_0C = REPO / "runs/v2_0c_champion/model.pt"

if __name__ == "__main__":
    RUNS = Path(sys.argv[1])
    RUNS.mkdir(parents=True, exist_ok=True)
    from parchis.az import config as C, round_loop
    cfg = C.SelfPlayRoundConfig(
        run_name="v2_1c", num_players=2, n_games_per_round=6000, games_per_shard=750,
        generation_workers=8,
        tau_start=0.03, tau_end=0.005, dirichlet_epsilon=0.0, tau_target=0.015,
        enable_escalation=False, aux_loss_weight=0.2, gate_enabled=False,
        promotion_n_pairs=100,
        learning_rate=3e-4,
        pool_random_anchor=False, pool_heuristic_fraction=1 / 15,
        deep_fraction=0.25, deep_depth=2, deep_targets=True,
        afterstate_rows=True,
    )
    anchors = dict(ANCHORS)
    if V2_0C.exists():
        anchors["v2_0c"] = V2_0C
    init = torch.load(INIT, map_location="cpu")
    t0 = time.time()
    for stop in (5, 10, 15, 20):
        round_loop.run_continuous(cfg, runs_dir=str(RUNS), max_rounds=stop, initial_champion_state_dict=init)
        print(f"== {stop} rounds done, {(time.time() - t0) / 60:.0f} min since start", flush=True)
        run_ladder(RUNS / "v2_1c" / "champion.pt", RUNS / f"ladder_round{stop}.json", tag=f"round{stop}",
                   anchors=anchors)
