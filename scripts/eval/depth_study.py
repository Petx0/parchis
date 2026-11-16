"""Play-time search depth study on the 2-player champion (PILA-6).

Measures what deeper expectimax (parchis.az.search) buys the champion and what it costs per move:

  d2_vs_d1   V2.1c depth 2 vs V2.1c depth 1                10,000 duplicate pairs
  sib_d1     V2.1c depth 1 vs V2.0c depth 1                10,000 pairs  (same deals as sib_d2)
  sib_d2     V2.1c depth 2 vs V2.0c depth 2                10,000 pairs  (same deals as sib_d1)
  d3_vs_d2   V2.1c depth 3 vs V2.1c depth 2                 2,000 pairs  (skipped if d2_vs_d1 < 52%;
                                                                          stopped at --d3-budget-min)
  timing     ms per decision at depths 1-4, one process, one BLAS thread, on positions sampled from
             V2.1c depth-1 self-play (every 3rd decision)

Seeds come from one fresh block of the evaluation seed registry (docs/EVAL_PROTOCOL.md §5), recorded
as kind "study": stage offsets 0 / 100 / 100 / 200 / 900 inside the block, 40 chunks per match
(chunk c uses seed block_base + offset + c). This is not a promotion run: deeper search on the same
net is a different time budget (EVAL_PROTOCOL §1).

Usage:
    python scripts/eval/depth_study.py --run-id depth_study_v2_1c --issue PILA-6 [--workers 8]
        [--d3-budget-min 67.5] [--positions 500] [--dry-run] [--smoke]

Writes runs/<run_id>/eval/depth_study_b<block>.json after every stage and the run record
runs/<run_id>/meta.json. Takes ~/parchis_compute.lock for the whole run (exit 3 if held).
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/v2"))
from ladder import single_thread_blas, hidden_sizes_of  # noqa: E402
from parchis.evaluation import protocol as P  # noqa: E402

LOCK = Path.home() / "parchis_compute.lock"
REPO_REGISTRY = REPO / "runs/eval_seed_registry.jsonl"
V21C = REPO / "runs/v2_1c_champion/model.pt"
V20C = REPO / "runs/v2_0c_champion/model.pt"
CHUNKS = 40
KILL_RATE = 0.52          # d2_vs_d1 below this skips d3_vs_d2 (plan's kill criterion)
STAGES = [
    # name, (net a, depth a), (net b, depth b), seed offset, duplicate pairs
    ("d2_vs_d1", (V21C, 2), (V21C, 1), 0, 10_000),
    ("sib_d1", (V21C, 1), (V20C, 1), 100, 10_000),
    ("sib_d2", (V21C, 2), (V20C, 2), 100, 10_000),
    ("d3_vs_d2", (V21C, 3), (V21C, 2), 200, 2_000),
]
TIMING_OFFSET = 900
TIMING_DEPTHS = (1, 2, 3, 4)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _net_factory(path, depth):
    from parchis.az import champion_pool
    from parchis.az.agent import NetEvaluator, make_search_agent_factory
    net = champion_pool.load_numpy_net(str(path), 216, P.NUM_PLAYERS, hidden_sizes_of(path))
    return make_search_agent_factory(NetEvaluator(net), depth=depth)


def _job(args):
    pa, da, pb, db, seed, n_pairs = args
    sys.path.insert(0, str(REPO))
    from parchis.evaluation import duplicate
    t = time.time()
    r = duplicate.play_duplicate_match(_net_factory(pa, da), _net_factory(pb, db), n_pairs=n_pairs,
                                       num_players=P.NUM_PLAYERS, max_turns=P.MAX_TURNS, seed=seed)
    draws = sum(w is None for g in r["groups"] for w in g["winners"])
    return r["wins_a"], draws, r["n_games"], time.time() - t


def _timing(args):
    """Time search() at every depth on the same positions, sampled from depth-1 self-play."""
    path, seed, n_positions, depths, every = args
    sys.path.insert(0, str(REPO))
    from parchis.az import champion_pool, search
    from parchis.az.agent import NetEvaluator, make_search_agent_factory
    from parchis.az.turn_context import TurnContextTracker
    from parchis.evaluation import duplicate
    ev = NetEvaluator(champion_pool.load_numpy_net(str(path), 216, P.NUM_PLAYERS, hidden_sizes_of(path)))
    times = {d: [] for d in depths}
    n_legal = []
    calls = [0]

    def timing_factory(game, seat, roll_box):
        tracker = TurnContextTracker()

        def choose_move(legal_moves):
            if not legal_moves:
                tracker.record_move(game, None)
                return None
            roll, pending_bonus, sixes = tracker.context_for(roll_box)
            calls[0] += 1
            if calls[0] % every == 0 and len(n_legal) < n_positions:
                n_legal.append(len(legal_moves))
                for d in depths:
                    t = time.perf_counter()
                    search.search(game, roll=roll, pending_bonus=pending_bonus, consecutive_sixes=sixes,
                                  depth=d, evaluator=ev)
                    times[d].append(time.perf_counter() - t)
            move, _, _ = search.search(game, roll=roll, pending_bonus=pending_bonus,
                                       consecutive_sixes=sixes, depth=1, evaluator=ev)
            tracker.record_move(game, move)
            return move

        return choose_move

    plain = make_search_agent_factory(ev, depth=1)
    i = 0
    while len(n_legal) < n_positions:
        duplicate.play_duplicate_match(timing_factory, plain, n_pairs=1, num_players=P.NUM_PLAYERS,
                                       max_turns=P.MAX_TURNS, seed=seed + i)
        i += 1
    return dict(times=times, n_legal=n_legal, pairs_played=i, seeds=f"{seed}+i, i<{i}")


def _summ_times(ts):
    ms = sorted(1000 * t for t in ts)
    q = lambda f: ms[min(len(ms) - 1, int(f * len(ms)))]  # noqa: E731
    return dict(n=len(ms), mean_ms=statistics.fmean(ms), median_ms=q(0.5), p90_ms=q(0.9), p99_ms=q(0.99),
                max_ms=ms[-1])


def acquire_lock(issue, what, eta_min):
    if LOCK.exists():
        try:
            held = json.loads(LOCK.read_text())
            os.kill(int(held["pid"]), 0)
            return held
        except (ProcessLookupError, ValueError, KeyError, json.JSONDecodeError):
            print(f"stale compute lock overwritten: {LOCK.read_text().strip()}", flush=True)
        except PermissionError:
            return json.loads(LOCK.read_text())
    now = datetime.now(timezone.utc)
    LOCK.write_text(json.dumps(dict(issue=issue, agent="Evaluator", pid=os.getpid(),
                                    started=now.isoformat(timespec="seconds"),
                                    eta=(now + timedelta(minutes=eta_min)).isoformat(timespec="seconds"),
                                    what=what)))
    return None


def release_lock():
    try:
        if json.loads(LOCK.read_text()).get("pid") == os.getpid():
            LOCK.unlink()
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def run_stage(ex, spec, seeds, pairs, budget_s=None):
    """Play one match as len(seeds) chunks. With budget_s, stop submitting/waiting once the stage
    has run that long; returns the counts of the chunks that finished and whether it was cut."""
    (pa, da), (pb, db) = spec
    per = [pairs // len(seeds) + (1 if c < pairs % len(seeds) else 0) for c in range(len(seeds))]
    t0 = time.time()
    futs = {ex.submit(_job, (str(pa), da, str(pb), db, s, k)): c for c, (s, k) in enumerate(zip(seeds, per))}
    acc, done_chunks, cut = [0, 0, 0], [], False
    pending = set(futs)
    while pending:
        timeout = None if budget_s is None else max(0.0, budget_s - (time.time() - t0))
        done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
        for f in done:
            w, d, n, _ = f.result()
            acc[0] += w; acc[1] += d; acc[2] += n
            done_chunks.append(futs[f])
        if budget_s is not None and time.time() - t0 >= budget_s and pending:
            cut = True
            break
    return acc, sorted(done_chunks), cut, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--issue", required=True)
    ap.add_argument("--workers", type=int, default=P.WORKERS)
    ap.add_argument("--d3-budget-min", type=float, default=67.5, help="1.5x the plan's 45 min")
    ap.add_argument("--positions", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, scratch seeds 0..; no registry, no lock")
    a = ap.parse_args()

    from parchis.evaluation import stats  # noqa: F401  (import check)
    from parchis.registry import record as registry
    sha = {"v2_1c": sha256(V21C), "v2_0c": sha256(V20C)}
    champ = json.loads((REPO / "runs/champions.json").read_text())["2p"]["current"]
    assert champ["sha256"] == sha["v2_1c"], "runs/champions.json champion is not runs/v2_1c_champion/model.pt"
    stages = [(n, sa, sb, off, (4 if a.smoke and n != "d3_vs_d2" else 2 if a.smoke else pairs))
              for n, sa, sb, off, pairs in STAGES]
    chunks = 2 if a.smoke else CHUNKS
    positions = 6 if a.smoke else a.positions
    git = registry.git_state()
    if a.smoke:
        block, base = None, 0
        seeds = {n: [off + c for c in range(chunks)] for n, _, _, off, _ in stages}
        tseed = TIMING_OFFSET
    else:
        block = P.next_block(P.read_registry(P.HOME_REGISTRY), P.read_registry(REPO_REGISTRY))
        seeds = {n: P.block_seeds(block, off, chunks) for n, _, _, off, _ in stages}
        tseed = P.block_seeds(block, TIMING_OFFSET, 1)[0]
    print(f"depth study  block {block}  git {git['sha'][:9]}{' DIRTY' if git['dirty'] else ''}", flush=True)
    for n, sa, sb, off, pairs in stages:
        print(f"  {n}: {sa[0].parent.name} d{sa[1]} vs {sb[0].parent.name} d{sb[1]}, {pairs} pairs, "
              f"seeds {seeds[n][0]}+c, c<{chunks}", flush=True)
    if a.dry_run:
        return 0

    run_dir = REPO / "runs" / a.run_id
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (f"depth_study_b{block}.json" if block is not None else "depth_study_smoke.json")
    if not a.smoke:
        held = acquire_lock(a.issue, f"PILA-6 depth study block {block}", eta_min=100)
        if held:
            print(f"compute lock held: {held}", flush=True)
            return 3
        P.append_registry([P.HOME_REGISTRY, REPO_REGISTRY], dict(
            kind="study", block=block, seeds=[P.block_seeds(block, 0, 1)[0], P.block_seeds(block, 0, 1)[0] + P.BLOCK_SIZE - 1],
            run_id=a.run_id, issue=a.issue, what="play-time search depth study: V2.1c d2 vs d1, "
            "V2.1c vs V2.0c at d1 and d2 (same deals), V2.1c d3 vs d2, ms/move timing",
            allocated=datetime.now(timezone.utc).isoformat(timespec="seconds")))
        cfg = dict(stages=[dict(name=n, a=registry.rel(sa[0]), depth_a=sa[1], b=registry.rel(sb[0]), depth_b=sb[1],
                                seed_offset=off, pairs=pairs) for n, sa, sb, off, pairs in stages],
                   chunks=chunks, block=block, workers=a.workers, max_turns=P.MAX_TURNS, kill_rate=KILL_RATE,
                   d3_budget_min=a.d3_budget_min, timing=dict(positions=positions, depths=list(TIMING_DEPTHS),
                                                              every=3, seed=tseed))
        registry.begin_run(run_dir, kind="evaluation", config=cfg, issue=a.issue,
                           title="Play-time search depth on the 2p champion V2.1c: strength vs ms/move",
                           notes="Not a promotion run (EVAL_PROTOCOL §1: deeper search is a different time budget).")
    record = dict(study="PILA-6 play-time depth", issue=a.issue, run_id=a.run_id, block=block,
                  git_sha=git["sha"], git_dirty=git["dirty"], nets={"v2_1c": dict(path=registry.rel(V21C), sha256=sha["v2_1c"]),
                  "v2_0c": dict(path=registry.rel(V20C), sha256=sha["v2_0c"])}, workers=a.workers,
                  max_turns=P.MAX_TURNS, chunks=chunks, results={}, skipped={},
                  started=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def save():
        out.write_text(json.dumps(record, indent=2, default=float))

    try:
        with single_thread_blas(), ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
            for n, sa, sb, off, pairs in stages:
                if n == "d3_vs_d2" and record["results"]["d2_vs_d1"]["win_rate"] < KILL_RATE:
                    record["skipped"][n] = f"kill criterion: d2_vs_d1 < {KILL_RATE:.0%}"
                    save()
                    print(f"{n}: skipped ({record['skipped'][n]})", flush=True)
                    continue
                budget = 60 * a.d3_budget_min if n == "d3_vs_d2" else None
                acc, done, cut, wall = run_stage(ex, (sa, sb), seeds[n], pairs, budget)
                r = P.summarise(*acc)
                r.update(a=f"{sa[0].parent.name} d{sa[1]}", b=f"{sb[0].parent.name} d{sb[1]}", depth_a=sa[1],
                         depth_b=sb[1], pairs_planned=pairs, chunks_done=len(done), chunks_planned=len(seeds[n]),
                         seeds=f"{seeds[n][0]}+c, c in {done if cut else f'0..{len(seeds[n]) - 1}'}",
                         stopped_at_budget=cut, wall_clock_s=round(wall, 1))
                record["results"][n] = r
                save()
                print(f"{n}: {r['a']} vs {r['b']}  {100 * r['win_rate']:.2f}% [{100 * r['ci'][0]:.2f}, "
                      f"{100 * r['ci'][1]:.2f}]  n={r['n_games']} draws={r['draws']}  {wall:.0f}s"
                      f"{'  STOPPED AT BUDGET' if cut else ''}", flush=True)
                if cut:
                    record["skipped"]["after_d3"] = "d3 budget exceeded: stop and ask the owner"
                    save()
                    for p in list(getattr(ex, "_processes", {}).values()):
                        p.terminate()      # our own workers; the pool is unusable after this
                    break
        if "after_d3" not in record["skipped"]:
            t0 = time.time()
            with single_thread_blas(), ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
                tr = ex.submit(_timing, (str(V21C), tseed, positions, TIMING_DEPTHS, 3)).result()
            record["timing"] = dict(per_depth={d: _summ_times(ts) for d, ts in tr["times"].items()},
                                    positions=positions, sampled_every=3, mean_legal_moves=statistics.fmean(tr["n_legal"]),
                                    forced_share=sum(k == 1 for k in tr["n_legal"]) / len(tr["n_legal"]),
                                    seeds=tr["seeds"], wall_clock_s=round(time.time() - t0, 1),
                                    note="one process, one BLAS thread, search() per decision incl. forced moves")
            save()
            for d, s in record["timing"]["per_depth"].items():
                print(f"timing d{d}: mean {s['mean_ms']:.2f} ms  median {s['median_ms']:.2f}  p90 {s['p90_ms']:.2f}"
                      f"  max {s['max_ms']:.1f}  (n={s['n']})", flush=True)
    finally:
        if not a.smoke:
            release_lock()

    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save()
    if not a.smoke:
        def mutate(m):
            for n, r in record["results"].items():
                (pa, _), (pb, _) = next((sa, sb) for nn, sa, sb, _, _ in stages if nn == n)
                registry.add_eval(m, registry.eval_entry(
                    "other", pa, pb, r["win_rate"], r["ci"], r["n_games"], seeds=r["seeds"], depth=r["depth_a"],
                    source=out, tag=f"b{block}:{n}", opponent_depth=r["depth_b"], draws=r["draws"],
                    issue=a.issue, stopped_at_budget=r["stopped_at_budget"]))
            m["status"] = "finished"
            m["finished"] = record["finished"]
            m["selection"] = "n/a: fixed nets (V2.1c champion, V2.0c sibling); no checkpoint chosen"
        registry.safe(registry.update_run, run_dir, mutate)
    print(f"record {out.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
