"""Full 2-player round robin over every tracked checkpoint + random + heuristic baselines (PILA-1).

Every unordered pair plays `pairs` duplicate pairs (2 games each) at search depth 1, split into
`chunks` jobs that run on `workers` spawn processes with one-thread BLAS (same harness as
scripts/v2/ladder.py). Pairing k, chunk c uses seed SEED_BASE + 1000*k + c -- disjoint from the
ladder (9000+i) and head-to-head (70000+i) seed ranges. One JSON line per pairing is appended to
<out_dir>/pairings.jsonl in the parchis.evaluation.ratings format, so
`python -m parchis.evaluation.ratings --pairings-path <out_dir>/pairings.jsonl` also reads it.
Re-running skips pairings already present in that file (resumable).

Usage: round_robin.py <out_dir> [pairs=1000] [workers=8] [only=name1,name2,...]
"""
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/v2"))
from ladder import single_thread_blas, hidden_sizes_of  # noqa: E402

SEED_BASE = 120000
DEPTH = 1
MAX_TURNS = 500

# name -> "random" | "heuristic_tuned" | "heuristic_default" | checkpoint path (relative to REPO)
PARTICIPANTS = {
    "random": "random",
    "heuristic_default": "heuristic_default",
    "heuristic_tuned": "heuristic_tuned",
    "bootstrap_v3": "runs/bootstrap_2p_v3/model.pt",
    "bootstrap_v4_large": "runs/bootstrap_2p_v4_large/model.pt",
    "champ23": "runs/selfplay_2p_v1_champion/model.pt",
    "v2_0": "runs/v2_0_champion/model.pt",
    "v2_0b": "runs/v2_0b_champion/model.pt",
    "v2_0c": "runs/v2_0c_champion/model.pt",
    "v2_1a": "runs/v2_1a_champion/model.pt",
    "v2_1c": "runs/v2_1c_champion/model.pt",
    "v2_1d": "runs/v2_1d_champion/model.pt",
    "win3": "runs/window_test/win3/model.pt",
    "win10": "runs/window_test/win10/model.pt",
    "c10_start_2x256": "runs/c10_precheck/start_2x256.pt",
    "c10_start_3x512": "runs/c10_precheck/start_3x512.pt",
}


def _factory(spec):
    from parchis.agents import heuristic
    from parchis.az import champion_pool
    from parchis.az.agent import NetEvaluator, make_search_agent_factory
    from parchis.az.selfplay import random_factory
    if spec == "random":
        return random_factory
    if spec == "heuristic_tuned":
        return heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS)
    if spec == "heuristic_default":
        return heuristic.make_heuristic_agent_factory(heuristic.DEFAULT_WEIGHTS)
    path = REPO / spec
    net = champion_pool.load_numpy_net(str(path), 216, 2, hidden_sizes_of(path))
    return make_search_agent_factory(NetEvaluator(net), depth=DEPTH)


def _job(args):
    spec_a, spec_b, seed, n_pairs = args
    sys.path.insert(0, str(REPO))
    from parchis.evaluation import duplicate
    r = duplicate.play_duplicate_match(_factory(spec_a), _factory(spec_b), n_pairs=n_pairs,
                                       num_players=2, max_turns=MAX_TURNS, seed=seed)
    return r["wins_a"], r["n_games"]


def main():
    positional = [a for a in sys.argv[1:] if "=" not in a]
    extra = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out_dir = Path(positional[0])
    pairs = int(extra.get("pairs", 1000))
    workers = int(extra.get("workers", 8))
    names = extra["only"].split(",") if "only" in extra else list(PARTICIPANTS)
    chunks = workers
    out_dir.mkdir(parents=True, exist_ok=True)
    pairings_path = out_dir / "pairings.jsonl"
    done = set()
    if pairings_path.exists():
        for line in pairings_path.read_text().splitlines():
            d = json.loads(line)
            done.add((d["participant_a"], d["participant_b"]))

    from parchis.evaluation import stats
    all_pairs = list(combinations(list(PARTICIPANTS), 2))  # fixed index -> fixed seeds
    todo = [(k, a, b) for k, (a, b) in enumerate(all_pairs)
            if a in names and b in names and (a, b) not in done]
    print(f"{len(todo)} pairings x {2 * pairs} games, {workers} workers", flush=True)
    t0 = time.time()
    with single_thread_blas(), ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {}
        for k, a, b in todo:
            for c in range(chunks):
                n = pairs // chunks + (1 if c < pairs % chunks else 0)
                f = ex.submit(_job, (PARTICIPANTS[a], PARTICIPANTS[b], SEED_BASE + 1000 * k + c, n))
                futs[f] = (k, a, b)
        acc = {}
        for f in as_completed(futs):
            k, a, b = futs[f]
            w, n = f.result()
            st = acc.setdefault((k, a, b), [0, 0, 0])
            st[0] += w
            st[1] += n
            st[2] += 1
            if st[2] == chunks:
                lo, hi = stats.wilson_score_interval(st[0], st[1])
                rec = dict(timestamp=datetime.now().isoformat(), participant_a=a, participant_b=b,
                           num_players=2, wins_a=st[0], n_games=st[1], win_rate_a=st[0] / st[1],
                           win_rate_a_ci=[lo, hi], depth=DEPTH, max_turns=MAX_TURNS,
                           seeds=f"{SEED_BASE + 1000 * k}+c, c<{chunks}",
                           spec_a=PARTICIPANTS[a], spec_b=PARTICIPANTS[b])
                with pairings_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"[{time.time() - t0:6.0f}s] {a} vs {b}: {st[0] / st[1]:.4f} "
                      f"[{lo:.4f}, {hi:.4f}] n={st[1]}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
