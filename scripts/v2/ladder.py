"""Fixed-anchor ladder for V2 runs (docs/V2_PLAN.md §10.2): one checkpoint at depth 1 against each
anchor, 1,000 duplicate pairs per anchor, pooled over 8 spawn workers. Seeds 9000+i match the V2.0
ladders, so numbers against the same anchor are directly comparable across runs.

Importable (run_v2_0b.py) or standalone:
    ladder.py <checkpoint.pt> <out.json> [tag] [pairs]
"""
import os
from contextlib import contextmanager
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
# One BLAS thread per worker: the nets' forwards are tiny (<= a few rows x 256) and OpenBLAS's
# threading makes a 5-6-row forward 10x slower on a busy machine (measured 2026-11-02: 1.7-2.9 ms vs
# 0.2 ms). Same caps as round_loop._THREAD_ENV; set in the parent before the pool spawns so the
# children inherit them, restored afterwards.
_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "VECLIB_MAXIMUM_THREADS": "1"}


@contextmanager
def single_thread_blas():
    saved = {k: os.environ.get(k) for k in _THREAD_ENV}
    os.environ.update(_THREAD_ENV)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
LUCK_NET = REPO / "runs/v2_1c_champion/model.pt"  # C8 reference evaluator; any fixed net is unbiased


def hidden_sizes_of(path):
    """Trunk widths read from a checkpoint's own weights (C10: 2x256 and 3x512 nets share the ladder)."""
    import torch
    sd = torch.load(str(path), map_location="cpu")
    layers = sorted((int(k.split(".")[1]), v.shape[0]) for k, v in sd.items()
                    if k.startswith("trunk.") and k.endswith(".weight") and v.dim() == 2)
    return tuple(width for _, width in layers)


def _worker(args):
    anchor_path, ckpt, seed, n, depth, *rest = args
    luck_path = rest[0] if rest else None       # C8: reference net for luck adjustment, or None
    luck_depth = int(rest[1]) if len(rest) > 1 else 0   # 0 = the net's post-roll value, k = depth-k search value
    sys.path.insert(0, str(REPO))
    from parchis.agents import heuristic
    from parchis.az import champion_pool
    from parchis.az.agent import NetEvaluator, make_search_agent_factory
    from parchis.evaluation import duplicate

    def net_factory(path):
        return make_search_agent_factory(
            NetEvaluator(champion_pool.load_numpy_net(path, 216, 2, hidden_sizes_of(path))), depth=depth)

    other = (heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS) if anchor_path is None
             else net_factory(anchor_path))
    luck_ev = None
    if luck_path:
        from parchis.evaluation.luck import NetLuckEvaluator, SearchLuckEvaluator
        luck_net = NetEvaluator(champion_pool.load_numpy_net(str(luck_path), 216, 2, hidden_sizes_of(luck_path)))
        luck_ev = SearchLuckEvaluator(luck_net, luck_depth) if luck_depth > 0 else NetLuckEvaluator(luck_net)
    r = duplicate.play_duplicate_match(net_factory(ckpt), other, n_pairs=n, num_players=2,
                                       max_turns=500, seed=seed, luck_evaluator=luck_ev)
    extra = ({"raw_groups": r["raw_groups"], "adjusted_groups": r["adjusted_groups"]}
             if luck_ev is not None else {})
    return r["wins_a"], r["n_games"], extra


def pool_luck(results):
    """Concatenate the workers' per-group scores and summarise (C8)."""
    from parchis.evaluation.duplicate import summarise_luck_groups
    raw = [x for r in results for x in r[2]["raw_groups"]]
    adj = [x for r in results for x in r[2]["adjusted_groups"]]
    return summarise_luck_groups(raw, adj)


def format_luck(summary):
    lo, hi = summary["adjusted_score_a_ci"]
    return (f"adj {summary['adjusted_score_a']:.3f} [{lo:.3f}, {hi:.3f}] "
            f"x{summary['luck_variance_multiplier']:.1f}")


def run_ladder(ckpt, out_path, tag="", anchors=ANCHORS, pairs=PAIRS, workers=WORKERS, depth=DEPTH,
               luck=None):
    """luck: None, or a checkpoint path used as the C8 luck evaluator (LUCK_NET for the default);
    then every anchor line also carries the adjusted score, its t-interval and the variance
    multiplier, and the JSON gains 'adjusted', 'adjusted_ci', 'luck_multiplier', 'luck_net'."""
    from parchis.evaluation import stats
    out = {}
    for name, anchor_path in anchors.items():
        t = time.time()
        jobs = [(None if anchor_path is None else str(anchor_path), str(ckpt), 9000 + i, pairs // workers, depth)
                + ((str(luck),) if luck else ())
                for i in range(workers)]
        with single_thread_blas(), ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_worker, jobs))
        w, n = sum(r[0] for r in res), sum(r[1] for r in res)
        lo, hi = stats.wilson_score_interval(w, n)
        out[name] = dict(win_rate=w / n, ci=[lo, hi], n_games=n, depth=depth)
        line = f"LADDER {tag} latest vs {name}: {w / n:.3f} [{lo:.3f}, {hi:.3f}] n={n}"
        if luck:
            summary = pool_luck(res)
            out[name].update(adjusted=summary["adjusted_score_a"], adjusted_ci=list(summary["adjusted_score_a_ci"]),
                             luck_multiplier=summary["luck_variance_multiplier"], luck_net=str(luck))
            line += " " + format_luck(summary)
        print(f"{line} ({time.time() - t:.0f}s)", flush=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    _record_ladder(ckpt, out_path, tag, anchors, out, workers, depth)
    return out


def _record_ladder(ckpt, out_path, tag, anchors, out, workers, depth):
    """Append every anchor line to the run record owning `ckpt` (docs/EXPERIMENT_REGISTRY.md)."""
    from parchis.registry import record as registry

    def write():
        run_dir = registry.find_run_dir(registry.resolve(registry.rel(ckpt)))
        if run_dir is None:
            print(f"[registry] no meta.json owns {ckpt}; ladder not recorded in a run record", flush=True)
            return None

        def mutate(m):
            for name, r in out.items():
                opp = "heuristic_tuned" if anchors[name] is None else anchors[name]
                registry.add_eval(m, registry.eval_entry(
                    "ladder", ckpt, opp, r["win_rate"], r["ci"], r["n_games"], seeds=f"9000+i, i<{workers}",
                    depth=depth, source=out_path, opponent_name=name, tag=tag or None,
                    **{k: r[k] for k in ("adjusted", "adjusted_ci") if k in r}))
        registry.update_run(run_dir, mutate)
    registry.safe(write)


if __name__ == "__main__":
    # ladder.py <ckpt> <out.json> [tag] [pairs] [name=model.pt ...] [luck=default|<model.pt>]
    # (extra anchors join the defaults; luck= turns on C8 luck-adjusted scoring)
    positional = [a for a in sys.argv[1:] if "=" not in a]
    extra = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    luck = extra.pop("luck", None)
    if luck == "default":
        luck = LUCK_NET
    anchors = dict(ANCHORS)
    anchors.update({k: Path(v) for k, v in extra.items()})
    run_ladder(positional[0], positional[1], tag=positional[2] if len(positional) > 2 else "",
               pairs=int(positional[3]) if len(positional) > 3 else PAIRS, anchors=anchors, luck=luck)
