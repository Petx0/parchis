"""Head-to-head of one checkpoint against one other, on seeds disjoint from the ladders (70000+i).
Usage: h2h.py <ckpt_a> <ckpt_b> <out.json> [pairs=10000]  (pairs counts duplicate pairs = 2 games each)"""
import json, sys, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ladder import _worker, WORKERS, DEPTH

if __name__ == "__main__":
    a, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
    pairs = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    from ladder import REPO
    sys.path.insert(0, str(REPO))
    from parchis.evaluation import stats
    t = time.time()
    jobs = [(b, a, 70000 + i, pairs // WORKERS, DEPTH) for i in range(WORKERS)]
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(_worker, jobs))
    w, n = sum(r[0] for r in res), sum(r[1] for r in res)
    lo, hi = stats.wilson_score_interval(w, n)
    r = dict(a=a, b=b, win_rate_a=w / n, ci=[lo, hi], n_games=n, depth=DEPTH, seeds="70000+i")
    Path(out).write_text(json.dumps(r, indent=2))
    print(f"H2H a vs b: {w / n:.4f} [{lo:.4f}, {hi:.4f}] n={n} ({time.time() - t:.0f}s)", flush=True)
