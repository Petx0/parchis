"""C10 capacity A/B (docs/V2_PLAN.md §4 C10, revised 2026-11-02): one arm per invocation.

Arm A = fresh 2x256, arm B = fresh 3x512. Each starts from its c10_precheck.py fit on 24 shards of
V2.1c self-play (runs/c10_precheck/start_<size>.pt) and runs 20 rounds of the V2.1c recipe, unchanged
except hidden_sizes. Start-strength ladder (heuristic, v2_1c) before round 0; ladders at rounds 10
and 20 (heuristic, v2_0c, v2_1c). The decision is the 10,000-pair h2h.py match afterwards.

Usage: run_c10_arm.py A|B <runs_dir>
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from ladder import ANCHORS, run_ladder

ARMS = {"A": ("2x256", (256, 256)), "B": ("3x512", (512, 512, 512))}

if __name__ == "__main__":
    arm, RUNS = sys.argv[1], Path(sys.argv[2])
    size, hidden = ARMS[arm]
    name = f"c10_{arm.lower()}_{size}"
    RUNS.mkdir(parents=True, exist_ok=True)
    start = REPO / f"runs/c10_precheck/start_{size}.pt"
    from parchis.az import config as C, round_loop
    cfg = C.SelfPlayRoundConfig(
        run_name=name, num_players=2, n_games_per_round=6000, games_per_shard=750,
        generation_workers=8, hidden_sizes=tuple(hidden),
        tau_start=0.03, tau_end=0.005, dirichlet_epsilon=0.0, tau_target=0.015,
        enable_escalation=False, aux_loss_weight=0.2, gate_enabled=False,
        promotion_n_pairs=100, learning_rate=3e-4,
        pool_random_anchor=False, pool_heuristic_fraction=1 / 15,
        deep_fraction=0.25, deep_depth=2, deep_targets=True, afterstate_rows=True,
    )
    t0 = time.time()
    if not (RUNS / "ladder_start.json").exists():
        run_ladder(start, RUNS / "ladder_start.json", tag="start",
                   anchors={k: ANCHORS[k] for k in ("heuristic", "v2_1c")})
        print(f"== start ladder done, {(time.time() - t0) / 60:.0f} min", flush=True)
    init = torch.load(start, map_location="cpu")
    for stop in (10, 20):
        round_loop.run_continuous(cfg, runs_dir=str(RUNS), max_rounds=stop, initial_champion_state_dict=init)
        print(f"== {stop} rounds done, {(time.time() - t0) / 60:.0f} min since start", flush=True)
        out = RUNS / f"ladder_round{stop}.json"
        if not out.exists():
            run_ladder(RUNS / name / "champion.pt", out, tag=f"round{stop}",
                       anchors={k: ANCHORS[k] for k in ("heuristic", "v2_0c", "v2_1c")})
    print("ARM DONE", flush=True)
