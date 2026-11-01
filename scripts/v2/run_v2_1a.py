"""V2.1a (docs/V2_PLAN.md §10.3): the V2.0b config plus C3 afterstate rows (`afterstate_rows=True`),
10 ungated rounds from the V2.0 champion; ladders at rounds 5 and 10 against the tuned heuristic, the
round-23 champion, the V2.0 champion and -- when `runs/v2_0b_champion/model.pt` exists -- the V2.0b
control champion. Gate (§10.3): leaf mean abs error < 0.5pp (run leaf_and_target_check.py on the
result) and the latest beats the V2.0 champion with Wilson lower bound > 50%, or gains >= 3pp vs the
heuristic.

Usage: run_v2_1a.py <runs_dir>      (shards: ~2.4 GB/round with afterstate rows; local non-iCloud dir)
Post hoc: scripts/diagnostics/round_summary.py <runs_dir>/v2_1a
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from ladder import ANCHORS, run_ladder

INIT = REPO / "runs/v2_0_champion/model.pt"
V2_0B = REPO / "runs/v2_0b_champion/model.pt"

if __name__ == "__main__":
    RUNS = Path(sys.argv[1])
    RUNS.mkdir(parents=True, exist_ok=True)
    from parchis.az import config as C, round_loop
    cfg = C.SelfPlayRoundConfig(
        run_name="v2_1a", num_players=2, n_games_per_round=6000, games_per_shard=2000,
        tau_start=0.03, tau_end=0.005, dirichlet_epsilon=0.0, tau_target=0.015,
        enable_escalation=False, aux_loss_weight=0.2, deep_fraction=0.0, gate_enabled=False,
        promotion_n_pairs=100,
        learning_rate=3e-4,
        pool_random_anchor=False, pool_heuristic_fraction=1 / 15,
        afterstate_rows=True,
    )
    anchors = dict(ANCHORS)
    if V2_0B.exists():
        anchors["v2_0b"] = V2_0B
    init = torch.load(INIT, map_location="cpu")
    t0 = time.time()
    for stop in (5, 10):
        round_loop.run_continuous(cfg, runs_dir=str(RUNS), max_rounds=stop, initial_champion_state_dict=init)
        print(f"== {stop} rounds done, {(time.time() - t0) / 60:.0f} min since start", flush=True)
        run_ladder(RUNS / "v2_1a" / "champion.pt", RUNS / f"ladder_round{stop}.json", tag=f"round{stop}",
                   anchors=anchors)
