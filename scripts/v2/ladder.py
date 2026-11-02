"""Fixed-anchor ladder for V2 runs (docs/V2_PLAN.md §10.2): one checkpoint at depth 1 against each
anchor, 1,000 duplicate pairs per anchor, pooled over 8 spawn workers. Seeds 9000+i match the V2.0
ladders, so numbers against the same anchor are directly comparable across runs.

Importable (run_v2_0b.py) or standalone:
    ladder.py <checkpoint.pt> <out.json> [tag] [pairs]
"""
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# name -> checkpoint path, or None for the tuned heuristic.
ANCHORS = {
    "heuristic": None,
    "champ23": REPO / "runs/selfplay_2p_v1_champion/model.pt",
    "v2_0": REPO / "runs/v2_0_champion/model.pt",
    "v2_0b": REPO / "runs/v2_0b_champion/model.pt",  # best-so-far after §10.4 (V2.0c tied it, §10.6)
    "v2_0c": REPO / "runs/v2_0c_champion/model.pt",  # V2.0b config, 20 more rounds: control for V2.1c
    "v2_1c": REPO / "runs/v2_1c_champion/model.pt",  # best-so-far after §10.7 (C2 deep targets)
}
PAIRS, WORKERS, DEPTH = 1000, 8, 1


def _worker(args):
    anchor_path, ckpt, seed, n, depth = args
    sys.path.insert(0, str(REPO))
    from parchis.agents import heuristic
    from parchis.az import champion_pool
    from parchis.az.agent import NetEvaluator, make_search_agent_factory
    from parchis.evaluation import duplicate

    def net_factory(path):
        return make_search_agent_factory(
            NetEvaluator(champion_pool.load_numpy_net(path, 216, 2, (256, 256))), depth=depth)

    other = (heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS) if anchor_path is None
             else net_factory(anchor_path))
    r = duplicate.play_duplicate_match(net_factory(ckpt), other, n_pairs=n, num_players=2,
                                       max_turns=500, seed=seed)
    return r["wins_a"], r["n_games"]


def run_ladder(ckpt, out_path, tag="", anchors=ANCHORS, pairs=PAIRS, workers=WORKERS, depth=DEPTH):
    from parchis.evaluation import stats
    out = {}
    for name, anchor_path in anchors.items():
        t = time.time()
        jobs = [(None if anchor_path is None else str(anchor_path), str(ckpt), 9000 + i, pairs // workers, depth)
                for i in range(workers)]
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_worker, jobs))
        w, n = sum(r[0] for r in res), sum(r[1] for r in res)
        lo, hi = stats.wilson_score_interval(w, n)
        out[name] = dict(win_rate=w / n, ci=[lo, hi], n_games=n, depth=depth)
        print(f"LADDER {tag} latest vs {name}: {w / n:.3f} [{lo:.3f}, {hi:.3f}] n={n} ({time.time() - t:.0f}s)",
              flush=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    # ladder.py <ckpt> <out.json> [tag] [pairs] [name=model.pt ...]  (extra anchors join the defaults)
    positional = [a for a in sys.argv[1:] if "=" not in a]
    extra = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    anchors = dict(ANCHORS)
    anchors.update({k: Path(v) for k, v in extra.items()})
    run_ladder(positional[0], positional[1], tag=positional[2] if len(positional) > 2 else "",
               pairs=int(positional[3]) if len(positional) > 3 else PAIRS, anchors=anchors)
