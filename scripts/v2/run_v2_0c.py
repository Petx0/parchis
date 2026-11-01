"""V2.0c continuation (docs/V2_PLAN.md §10.4 next steps, item 1): 20 more ungated rounds of the V2.0b
config from the V2.0b champion. Question: is the V2.0b gain a step or a slope? Ladders every 5 rounds
against the tuned heuristic, the round-23, V2.0 and V2.0b champions (1,000 pairs each, depth 1). The
endpoint is the control for V2.1c.

Usage: run_v2_0c.py <runs_dir>      (shards: ~1.2 GB/round; use a local non-iCloud directory)
Post hoc: scripts/diagnostics/round_summary.py <runs_dir>/v2_0c
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from ladder import run_ladder

INIT = REPO / "runs/v2_0b_champion/model.pt"

if __name__ == "__main__":
    # Everything below the guard: the ladder's spawn workers re-import this module.
    RUNS = Path(sys.argv[1])
    RUNS.mkdir(parents=True, exist_ok=True)
    from parchis.az import config as C, round_loop
    cfg = C.SelfPlayRoundConfig(
        run_name="v2_0c", num_players=2, n_games_per_round=6000, games_per_shard=2000,
        tau_start=0.03, tau_end=0.005, dirichlet_epsilon=0.0, tau_target=0.015,
        enable_escalation=False, aux_loss_weight=0.2, deep_fraction=0.0, gate_enabled=False,
        promotion_n_pairs=100,  # telemetry only (gate bypassed); the real read is the ladder
        learning_rate=3e-4,
        pool_random_anchor=False, pool_heuristic_fraction=1 / 15,
    )
    init = torch.load(INIT, map_location="cpu")
    t0 = time.time()
    for stop in (5, 10, 15, 20):
        round_loop.run_continuous(cfg, runs_dir=str(RUNS), max_rounds=stop, initial_champion_state_dict=init)
        print(f"== {stop} rounds done, {(time.time() - t0) / 60:.0f} min since start", flush=True)
        run_ladder(RUNS / "v2_0c" / "champion.pt", RUNS / f"ladder_round{stop}.json", tag=f"round{stop}")
