"""Replay-window test (docs/V2_PLAN.md §10.10 correction): does each training step seeing more
independent games help? 20 rounds of the V2.1c recipe continued from the V2.1c champion, changing
only replay_window_rounds. Window 3 is the control (V2.1c's own setting, the V2.0c pattern);
window 10 is the treatment (60,000 games per training step instead of 18,000). Ladders at rounds 10
and 20 (heuristic, v2_0c, v2_1c); the decision is the 10,000-pair h2h.py match.

Usage: run_window.py <window> <runs_dir>
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from ladder import ANCHORS, run_ladder

INIT = REPO / "runs/v2_1c_champion/model.pt"

if __name__ == "__main__":
    window, RUNS = int(sys.argv[1]), Path(sys.argv[2])
    name = f"win{window}"
    RUNS.mkdir(parents=True, exist_ok=True)
    from parchis.az import config as C, round_loop
    cfg = C.SelfPlayRoundConfig(
        run_name=name, num_players=2, n_games_per_round=6000, games_per_shard=750,
        generation_workers=8, replay_window_rounds=window,
        tau_start=0.03, tau_end=0.005, dirichlet_epsilon=0.0, tau_target=0.015,
        enable_escalation=False, aux_loss_weight=0.2, gate_enabled=False,
        promotion_n_pairs=100, learning_rate=3e-4,
        pool_random_anchor=False, pool_heuristic_fraction=1 / 15,
        deep_fraction=0.25, deep_depth=2, deep_targets=True, afterstate_rows=True,
    )
    init = torch.load(INIT, map_location="cpu")
    t0 = time.time()
    for stop in (10, 20):
        round_loop.run_continuous(cfg, runs_dir=str(RUNS), max_rounds=stop, initial_champion_state_dict=init)
        print(f"== {name}: {stop} rounds done, {(time.time() - t0) / 60:.0f} min since start", flush=True)
        out = RUNS / f"ladder_{name}_round{stop}.json"
        if not out.exists():
            run_ladder(RUNS / name / "champion.pt", out, tag=f"{name}_round{stop}",
                       anchors={k: ANCHORS[k] for k in ("heuristic", "v2_0c", "v2_1c")})
    print(f"RUN DONE {name}", flush=True)
