"""Head-to-head of one checkpoint against one other, on seeds disjoint from the ladders (70000+i).
Usage: h2h.py <ckpt_a> <ckpt_b> <out.json> [pairs=10000] [luck=default|<model.pt>] [luck_depth=0|1]
(pairs counts duplicate pairs = 2 games each; luck= adds C8 luck-adjusted scoring, see
parchis/evaluation/luck.py -- 'default' uses ladder.LUCK_NET; luck_depth=1 uses the depth-1 search
value as the luck evaluator, about 2x the variance removed at about 2x the per-game cost)"""
import json, sys, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ladder import _worker, pool_luck, format_luck, single_thread_blas, WORKERS, DEPTH, LUCK_NET, REPO

if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if "=" not in a]
    extra = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    a, b, out = positional[0], positional[1], positional[2]
    pairs = int(positional[3]) if len(positional) > 3 else 10000
    luck = extra.get("luck")
    if luck == "default":
        luck = str(LUCK_NET)
    luck_depth = int(extra.get("luck_depth", 0))
    sys.path.insert(0, str(REPO))
    from parchis.evaluation import stats
    t = time.time()
    jobs = [(b, a, 70000 + i, pairs // WORKERS, DEPTH) + ((luck, luck_depth) if luck else ())
            for i in range(WORKERS)]
    with single_thread_blas(), ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(_worker, jobs))
    w, n = sum(r[0] for r in res), sum(r[1] for r in res)
    lo, hi = stats.wilson_score_interval(w, n)
    r = dict(a=a, b=b, win_rate_a=w / n, ci=[lo, hi], n_games=n, depth=DEPTH, seeds="70000+i")
    line = f"H2H a vs b: {w / n:.4f} [{lo:.4f}, {hi:.4f}] n={n}"
    if luck:
        summary = pool_luck(res)
        r.update(luck_net=luck, luck_depth=luck_depth, adjusted_score_a=summary["adjusted_score_a"],
                 adjusted_ci=list(summary["adjusted_score_a_ci"]),
                 raw_ci_groups=list(summary["raw_score_a_ci_groups"]),
                 luck_variance_multiplier=summary["luck_variance_multiplier"],
                 mean_abs_luck=summary["mean_abs_luck"])
        line += " " + format_luck(summary)
    Path(out).write_text(json.dumps(r, indent=2))
    print(f"{line} ({time.time() - t:.0f}s)", flush=True)
    # run record of a's run (docs/EXPERIMENT_REGISTRY.md); never fails the match
    from parchis.registry import record as registry
    registry.safe(lambda: registry.record_eval(registry.eval_entry(
        "h2h", a, b, w / n, [lo, hi], n, seeds=f"70000+i, i<{WORKERS}", depth=DEPTH, source=out,
        wall_clock_s=round(time.time() - t, 1),
        **({"adjusted_score": r["adjusted_score_a"], "adjusted_ci": r["adjusted_ci"]} if luck else {}))))
