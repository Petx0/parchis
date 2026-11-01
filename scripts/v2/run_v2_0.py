"""V2.0 go/no-go (docs/V2_PLAN.md section 5): 20 ungated rounds from the round-23 champion with
win-probability-scale temperatures; ladder at rounds 10 and 20. Usage: run_v2_0.py <runs_dir>"""
import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

RUNS = Path(sys.argv[1]); RUNS.mkdir(parents=True, exist_ok=True)
CHAMP23 = REPO / "runs/selfplay_2p_v1_champion/model.pt"
PAIRS, WORKERS = 1000, 8


def _worker(args):
    kind, ckpt, seed, n = args
    sys.path.insert(0, str(REPO))
    from parchis.az import champion_pool
    from parchis.az.agent import NetEvaluator, make_search_agent_factory
    from parchis.agents import heuristic
    from parchis.evaluation import duplicate
    mk = lambda p: make_search_agent_factory(NetEvaluator(champion_pool.load_numpy_net(p, 216, 2, (256, 256))), depth=1)
    other = mk(str(CHAMP23)) if kind == "champ23" else heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS)
    r = duplicate.play_duplicate_match(mk(ckpt), other, n_pairs=n, num_players=2, max_turns=500, seed=seed)
    return r["wins_a"], r["n_games"]


def ladder(tag, ckpt):
    from parchis.evaluation import stats
    out = {}
    for kind in ("champ23", "heuristic"):
        with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_worker, [(kind, str(ckpt), 9000 + i, PAIRS // WORKERS) for i in range(WORKERS)]))
        w, n = sum(r[0] for r in res), sum(r[1] for r in res)
        lo, hi = stats.wilson_score_interval(w, n)
        out[kind] = dict(win_rate=w / n, ci=[lo, hi], n_games=n)
        print(f"LADDER {tag} latest vs {kind}: {w/n:.3f} [{lo:.3f}, {hi:.3f}] n={n}", flush=True)
    (RUNS / f"ladder_{tag}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    from parchis.az import config as C, round_loop
    cfg = C.SelfPlayRoundConfig(
        run_name="v2_0", num_players=2, n_games_per_round=6000, games_per_shard=2000,
        tau_start=0.03, tau_end=0.015, dirichlet_epsilon=0.0, tau_target=0.015,
        enable_escalation=False, aux_loss_weight=0.2, deep_fraction=0.0, gate_enabled=False,
        promotion_n_pairs=100,  # telemetry only (gate bypassed); the real read is the ladder
    )
    init = torch.load(CHAMP23, map_location="cpu")
    for stop in (10, 20):
        t = time.time()
        round_loop.run_continuous(cfg, runs_dir=str(RUNS), max_rounds=stop, initial_champion_state_dict=init)
        print(f"== {stop} rounds done in {(time.time()-t)/60:.0f} min", flush=True)
        ladder(f"round{stop}", RUNS / "v2_0" / "champion.pt")
