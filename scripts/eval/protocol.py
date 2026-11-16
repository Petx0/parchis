"""Champion-promotion protocol: the single entry point (docs/EVAL_PROTOCOL.md, PILA-2).

Plays one candidate checkpoint through the fixed match set against the 2-player champion named in
runs/champions.json, on a fresh seed block from the evaluation seed registry, and prints the table
and a PASS / FAIL / PROVISIONAL / INVALID verdict. Writes runs/<run_id>/eval/protocol_b<block>.json.

Usage:
    python scripts/eval/protocol.py <candidate.pt> --run-id <run_id> --issue PILA-N
        [--selection fixed | best-of-N] [--confirmation-of <earlier protocol json>]
        [--workers 8] [--dry-run]

Exit codes: 0 PASS, 1 FAIL / INVALID, 2 PROVISIONAL, 3 compute lock held, 4 bad input.
Takes ~/parchis_compute.lock for the duration (the run is a heavy job: 8 workers).
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/v2"))
from ladder import single_thread_blas, hidden_sizes_of  # noqa: E402
from parchis.evaluation import protocol as P  # noqa: E402

LOCK = Path.home() / "parchis_compute.lock"
REPO_REGISTRY = REPO / "runs/eval_seed_registry.jsonl"
CHAMPIONS = REPO / "runs/champions.json"
ETA_MINUTES = 10


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state():
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())
    return sha, dirty


def _factory(spec, depth):
    from parchis.agents import heuristic
    from parchis.az import champion_pool
    from parchis.az.agent import NetEvaluator, make_search_agent_factory
    from parchis.az.selfplay import random_factory
    if spec == "random":
        return random_factory
    if spec == "heuristic_tuned":
        return heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS)
    path = Path(spec)
    net = champion_pool.load_numpy_net(str(path), 216, P.NUM_PLAYERS, hidden_sizes_of(path))
    return make_search_agent_factory(NetEvaluator(net), depth=depth)


def _job(args):
    spec_a, spec_b, seed, n_pairs, depth = args
    sys.path.insert(0, str(REPO))
    from parchis.evaluation import duplicate
    r = duplicate.play_duplicate_match(_factory(spec_a, depth), _factory(spec_b, depth), n_pairs=n_pairs,
                                       num_players=P.NUM_PLAYERS, max_turns=P.MAX_TURNS, seed=seed)
    draws = sum(w is None for g in r["groups"] for w in g["winners"])
    return r["wins_a"], draws, r["n_games"]


def acquire_lock(issue, what):
    if LOCK.exists():
        try:
            held = json.loads(LOCK.read_text())
            os.kill(int(held["pid"]), 0)
            return held                      # live holder
        except (ProcessLookupError, ValueError, KeyError, json.JSONDecodeError):
            print(f"stale compute lock overwritten: {LOCK.read_text().strip()}", flush=True)
        except PermissionError:
            return json.loads(LOCK.read_text())
    now = datetime.now(timezone.utc)
    LOCK.write_text(json.dumps(dict(issue=issue, agent="Evaluator", pid=os.getpid(),
                                    started=now.isoformat(timespec="seconds"),
                                    eta=(now + timedelta(minutes=ETA_MINUTES)).isoformat(timespec="seconds"),
                                    what=what)))
    return None


def release_lock():
    try:
        if json.loads(LOCK.read_text()).get("pid") == os.getpid():
            LOCK.unlink()
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--issue", required=True)
    ap.add_argument("--selection", default="fixed",
                    help="'fixed' (checkpoint named before any eval) or 'best-of-N' (picked among N)")
    ap.add_argument("--confirmation-of", help="protocol JSON of the PROVISIONAL run this confirms")
    ap.add_argument("--workers", type=int, default=P.WORKERS)
    ap.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan; no games, no seeds burned")
    a = ap.parse_args()

    cand = Path(a.candidate).resolve()
    if not cand.exists():
        sys.exit(print(f"candidate not found: {cand}") or 4)
    champs = json.loads(CHAMPIONS.read_text())["2p"]
    champ, prev = champs["current"], champs["previous"]
    champ_path, prev_path = REPO / champ["path"], REPO / prev["path"]
    if sha256(champ_path) != champ["sha256"]:
        sys.exit(print(f"champion checkpoint {champ_path} does not match the sha256 in {CHAMPIONS}") or 4)
    cand_sha = sha256(cand)
    if cand_sha == champ["sha256"]:
        sys.exit(print("candidate is the champion checkpoint") or 4)
    if a.selection != "fixed" and not a.selection.startswith("best-of-"):
        sys.exit(print("--selection must be 'fixed' or 'best-of-N'") or 4)
    prior = None
    if a.confirmation_of:
        prior = json.loads(Path(a.confirmation_of).read_text())
        if prior["candidate"]["sha256"] != cand_sha or prior["verdict"]["verdict"] != "PROVISIONAL":
            sys.exit(print("--confirmation-of must be a PROVISIONAL run of this same checkpoint") or 4)
    depth = int(champ.get("depth", P.DEFAULT_DEPTH))
    git_sha, dirty = git_state()

    specs = {
        "decisive": (str(cand), str(champ_path)),
        "heuristic": (str(cand), "heuristic_tuned"),
        "champion_heuristic": (str(champ_path), "heuristic_tuned"),
        "random": (str(cand), "random"),
        "previous": (str(cand), str(prev_path)),
    }
    labels = {
        "decisive": f"**candidate vs champion {champ['name']}** (decisive)",
        "heuristic": "candidate vs tuned heuristic",
        "champion_heuristic": f"champion {champ['name']} vs tuned heuristic (guard reference)",
        "random": "candidate vs random",
        "previous": f"candidate vs previous champion {prev['name']} (context)",
    }
    home_reg, repo_reg = P.read_registry(P.HOME_REGISTRY), P.read_registry(REPO_REGISTRY)
    block = P.next_block(home_reg, repo_reg)
    seeds = {name: P.block_seeds(block, off, a.workers) for name, off, _ in P.MATCHES}
    pairs = {name: n for name, _, n in P.MATCHES}
    print(f"candidate {cand} sha256 {cand_sha[:12]}  champion {champ['name']}  depth {depth}  "
          f"block {block} (seeds {seeds['decisive'][0]}..)  git {git_sha[:9]}{' DIRTY' if dirty else ''}",
          flush=True)
    if a.dry_run:
        for name in specs:
            print(f"  {name}: {pairs[name]} pairs, seeds {seeds[name][0]}..{seeds[name][-1]}")
        return 0

    held = acquire_lock(a.issue, f"eval protocol {a.run_id} block {block}")
    if held:
        print(f"compute lock held: {held}", flush=True)
        return 3
    try:
        entry = dict(kind="protocol", block=block, seeds=[P.block_seeds(block, 0, 1)[0], P.block_seeds(block, 0, 1)[0] + P.BLOCK_SIZE - 1],
                     run_id=a.run_id, issue=a.issue, candidate=str(cand.relative_to(REPO)) if cand.is_relative_to(REPO) else str(cand),
                     candidate_sha256=cand_sha, champion=champ["name"], selection=a.selection,
                     allocated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        P.append_registry([P.HOME_REGISTRY, REPO_REGISTRY], entry)
        t0 = time.time()
        acc = {name: [0, 0, 0] for name in specs}
        with single_thread_blas(), ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {}
            for name, (sa, sb) in specs.items():
                n = pairs[name]
                for c, seed in enumerate(seeds[name]):
                    k = n // a.workers + (1 if c < n % a.workers else 0)
                    futs[ex.submit(_job, (sa, sb, seed, k, depth))] = name
            for f in as_completed(futs):
                w, d, n = f.result()
                st = acc[futs[f]]
                st[0] += w; st[1] += d; st[2] += n
        wall = time.time() - t0
    finally:
        release_lock()

    out_dir = REPO / "runs" / a.run_id / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"protocol_b{block}.json"
    # raw counts first, so a failure below cannot lose the games already played
    (out_dir / f"protocol_b{block}.raw.json").write_text(json.dumps(dict(acc=acc, wall_clock_s=wall)))
    results = {name: P.summarise(*acc[name]) for name in specs}
    v = P.verdict(results, a.selection, a.confirmation_of)
    record = dict(protocol="docs/EVAL_PROTOCOL.md v1", run_id=a.run_id, issue=a.issue,
                  candidate=dict(path=entry["candidate"], sha256=cand_sha),
                  champion=champ, previous=prev, depth=depth, num_players=P.NUM_PLAYERS,
                  max_turns=P.MAX_TURNS, workers=a.workers, block=block,
                  seeds={k: f"{s[0]}+c, c<{len(s)}" for k, s in seeds.items()}, pairs=pairs,
                  specs=specs, results=results, verdict=v, selection=a.selection,
                  confirmation_of=a.confirmation_of, git_sha=git_sha, git_dirty=dirty,
                  wall_clock_s=round(wall, 1), finished=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    out.write_text(json.dumps(record, indent=2, default=float))
    print()
    print(P.format_table(results, labels, seeds, depth))
    print()
    print(P.format_checks(v))
    print(f"\nVERDICT: {v['verdict']}   selection: {a.selection}   block {block}   "
          f"wall-clock {wall:.0f}s   record {out.relative_to(REPO)}", flush=True)
    return {"PASS": 0, "PROVISIONAL": 2}.get(v["verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())
