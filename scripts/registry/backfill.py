"""One-time backfill of run records (meta.json) for every run that predates the registry (PILA-7).

Everything is read from files already in runs/ and from the documents named per run; nothing is
re-run. Fields that were never recorded stay null and say so:
  * git.sha is the commit that ADDED the run directory (git.source = "backfill: ..."), not the
    commit that trained it -- the training SHA was not recorded;
  * ladder results at intermediate rounds belong to a champion.pt that was overwritten later, so
    their checkpoint_sha256 is null; only the last ladder is tied to the kept model.pt;
  * per-round wall-clock comes from run.log / driver.log ("round N finished in Xs").

Idempotent: re-running rebuilds the same records (evals are keyed). Usage:
    python scripts/registry/backfill.py [--dry-run]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from parchis.registry import record as R  # noqa: E402

RUNS = REPO / "runs"
ANCHOR_PATHS = {
    "heuristic": "heuristic_tuned",
    "champ23": "runs/selfplay_2p_v1_champion/model.pt",
    "v2_0": "runs/v2_0_champion/model.pt",
    "v2_0b": "runs/v2_0b_champion/model.pt",
    "v2_0c": "runs/v2_0c_champion/model.pt",
    "v2_1c": "runs/v2_1c_champion/model.pt",
}
ENDPOINT_UNGATED = "endpoint: champion after the last round (ungated, every candidate adopted); fixed in advance"

# run_id -> what the files cannot say. `init` names a run_id (its model.pt) or None for random init.
RUNS_TABLE = {
    "bootstrap_2p_v3": dict(
        title="Phase 2 supervised bootstrap on 20k mixed-pool games", created="2026-08-28",
        doc="docs/AZ_DESIGN.md (bootstrap); docs/LEADERBOARD.md §participants", init=None, loop="bootstrap",
        selection="best validation epoch (early stopping)"),
    "bootstrap_2p_v4_large": dict(
        title="Phase 2 supervised bootstrap on 200k games (v3 recipe)", created="2026-08-28",
        doc="docs/AZ_DESIGN.md (bootstrap v4)", init=None, loop="bootstrap",
        selection="best validation epoch (early stopping)"),
    "selfplay_2p_v1_champion": dict(
        title="Phase 3 gated self-play (champ23): 132 rounds, 3 promotions, last at round 23",
        created="2026-08-28", doc="docs/AZ_DESIGN.md; docs/AGENT_REBUILD_PLAN.md", init="bootstrap_2p_v4_large",
        loop="selfplay", rounds=133,
        selection="last promoted champion (round 23) of a gated run; gate = 600-pair Wilson lower bound >= 50%"),
    "v2_0_champion": dict(title="V2.0: ungated recipe from champ23", created="2026-10-31",
                          doc="docs/V2_PLAN.md §10, §10.1", init="selfplay_2p_v1_champion", loop="v2",
                          script="scripts/v2/run_v2_0.py", extra=[("round09_candidate.pt", "candidate", "round 9 candidate")]),
    "v2_0b_champion": dict(title="V2.0b: recipe fixes, 10 rounds from V2.0", created="2026-10-31",
                           doc="docs/V2_PLAN.md §10.4", init="v2_0_champion", loop="v2",
                           script="scripts/v2/run_v2_0b.py", extra=[("round04_candidate.pt", "candidate", "round 4 candidate")]),
    "v2_1a_champion": dict(title="V2.1a: 10 rounds from V2.0", created="2026-10-31", doc="docs/V2_PLAN.md §10.4",
                           init="v2_0_champion", loop="v2", script="scripts/v2/run_v2_1a.py",
                           extra=[("round04_candidate.pt", "candidate", "round 4 candidate")]),
    "v2_0c_champion": dict(title="V2.0c: V2.0b recipe, 20 more rounds (control for V2.1c)", created="2026-11-01",
                           doc="docs/V2_PLAN.md §10.6", init="v2_0b_champion", loop="v2",
                           script="scripts/v2/run_v2_0c.py"),
    "v2_1c_champion": dict(title="V2.1c: C2 deep targets (deep_fraction 0.25)", created="2026-11-02",
                           doc="docs/V2_PLAN.md §10.7", init="v2_0b_champion", loop="v2",
                           script="scripts/v2/run_v2_1c.py"),
    "v2_1d_champion": dict(title="V2.1d: deep_fraction 0.5", created="2026-11-02", doc="docs/V2_PLAN.md §10.9",
                           init="v2_0b_champion", loop="v2", script="scripts/v2/run_v2_1d.py"),
    "window_test/win3": dict(title="Replay-window test, window 3 (control): V2.1c recipe continued 20 rounds",
                             created="2026-11-02", doc="docs/V2_PLAN.md §10.11", init="v2_1c_champion", loop="v2",
                             script="scripts/v2/run_window.py 3", log=("window_test/driver.log", "win3")),
    "window_test/win10": dict(title="Replay-window test, window 10: V2.1c recipe continued 20 rounds",
                              created="2026-11-02", doc="docs/V2_PLAN.md §10.11", init="v2_1c_champion", loop="v2",
                              script="scripts/v2/run_window.py 10", log=("window_test/driver.log", "win10")),
}

# h2h files: (file, a checkpoint, b checkpoint). The window-test files name archive paths for `a`;
# the curated copies in runs/window_test/ are those checkpoints.
H2H = [
    ("v2_1c_champion/h2h_vs_v2_0c.json", "runs/v2_1c_champion/model.pt", "runs/v2_0c_champion/model.pt"),
    ("v2_1d_champion/h2h_vs_v2_0c.json", "runs/v2_1d_champion/model.pt", "runs/v2_0c_champion/model.pt"),
    ("v2_1d_champion/h2h_vs_v2_1c.json", "runs/v2_1d_champion/model.pt", "runs/v2_1c_champion/model.pt"),
    ("window_test/h2h_win3_vs_v2_1c.json", "runs/window_test/win3/model.pt", "runs/v2_1c_champion/model.pt"),
    ("window_test/h2h_win10_vs_v2_1c.json", "runs/window_test/win10/model.pt", "runs/v2_1c_champion/model.pt"),
    ("window_test/h2h_win10_vs_win3.json", "runs/window_test/win10/model.pt", "runs/window_test/win3/model.pt"),
]


def added_commit(path):
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H %cI", "--", str(path)], cwd=REPO,
                         capture_output=True, text=True).stdout.split("\n")
    lines = [line for line in out if line.strip()]
    if not lines:
        return None, None
    sha, date = lines[-1].split()
    return sha, date


def round_seconds(log_path, section=None):
    """{round: seconds} from 'round N finished in Xs' lines; `section` picks one run of a shared log
    (the lines before 'RUN DONE <section>' and after the previous RUN DONE)."""
    if not log_path.exists():
        return {}
    lines = log_path.read_text().splitlines()
    if section:
        blocks, cur = {}, []
        for line in lines:
            cur.append(line)
            m = re.match(r"RUN DONE (\S+)", line)
            if m:
                blocks[m.group(1)], cur = cur, []
        lines = blocks.get(section, [])
    out = {}
    for line in lines:
        m = re.match(r"round (\d+) finished in ([\d.]+)s", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def init_of(run_id):
    if run_id is None:
        return dict(path=None, sha256=None, run_id=None, note="random init")
    p = RUNS / run_id / "model.pt"
    return dict(path=R.rel(p), sha256=R.file_sha256(p), run_id=run_id)


def training_block(run_dir, cfg, spec):
    loop = spec["loop"]
    if loop == "bootstrap":
        metrics = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines() if x.strip()]
        best = min(metrics, key=lambda m: m["val_loss"]) if metrics else {}
        return dict(dataset_games=cfg.get("n_games"), epochs=len(metrics), best_epoch=best.get("epoch"),
                    best_val_loss=best.get("val_loss"), metrics="metrics.jsonl")
    if loop == "selfplay":
        cm = json.loads((run_dir / "champion_meta.json").read_text())
        return dict(rounds_completed=spec["rounds"], promotions=cm["promotions"],
                    selfplay_games=spec["rounds"] * cfg["n_games_per_round"],
                    note="rounds 0-132 (champion_meta.json round index 132; docs say '132 rounds'); per-round "
                         "records not kept in runs/ (archive: ~/parchis_training_archive/selfplay_2p_v1)")
    log = RUNS / spec["log"][0] if "log" in spec else run_dir / "run.log"
    secs = round_seconds(log, spec["log"][1] if "log" in spec else None)
    rounds = []
    for rd in sorted((run_dir / "rounds").glob("round_*")) if (run_dir / "rounds").exists() else []:
        n = int(rd.name.split("_")[1])
        pr = json.loads((rd / "promotion_result.json").read_text()) if (rd / "promotion_result.json").exists() else {}
        rounds.append(dict(round=n, seconds=secs.get(n), promoted=pr.get("promoted"), gate_win_rate=pr.get("win_rate_a"),
                           gate_ci=pr.get("win_rate_a_ci"), gate_games=pr.get("n_games"),
                           generation_seed_first=cfg["generation_seed"] + n * 1000,
                           promotion_seed=cfg["promotion_seed"] + n))
    cm = json.loads((run_dir / "champion_meta.json").read_text())
    n_rounds = len(rounds) or cm["round"] + 1
    wall = sum(s for s in secs.values()) if secs else None
    return dict(rounds=rounds, rounds_completed=n_rounds, promotions=cm["promotions"],
                selfplay_games=n_rounds * cfg["n_games_per_round"],
                wall_clock_s=round(wall, 1) if wall else None,
                wall_clock_note="sum of round times from the run log (generation + training + gate), excludes ladders",
                rounds_dir="rounds/" if rounds else None)


def ladder_evals(run_dir, final_sha, n_rounds, run_id):
    evals = []
    for f in sorted(run_dir.glob("ladder_*.json")):
        data = json.loads(f.read_text())
        m = re.search(r"round(\d+)", f.name)
        at_round = int(m.group(1)) if m else None
        is_final = at_round == n_rounds
        for name, r in data.items():
            pairs = r["n_games"] // 2
            e = R.eval_entry("ladder", run_dir / "model.pt", ANCHOR_PATHS[name], r["win_rate"], r["ci"], r["n_games"],
                             seeds="9000+i, i<8", depth=r.get("depth", 1), source=f, opponent_name=name,
                             tag=f.stem.replace("ladder_", ""), pairs=pairs)
            if not is_final:
                e.update(checkpoint=f"{R.rel(run_dir)}: champion after round {at_round - 1} (not kept)",
                         checkpoint_sha256=None)
            e["recorded"] = None
            evals.append(e)
    return evals


def round_robin_evals(board_dir):
    """Both sides of every pairing of the board, keyed by checkpoint path."""
    by_ckpt = {}
    for line in (board_dir / "pairings.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        for side, ckpt, opp, opp_name in (("a", p["spec_a"], p["spec_b"], p["participant_b"]),
                                          ("b", p["spec_b"], p["spec_a"], p["participant_a"])):
            if "/" not in ckpt:
                continue
            wr = p["win_rate_a"] if side == "a" else 1 - p["win_rate_a"]
            lo, hi = p["win_rate_a_ci"]
            ci = [lo, hi] if side == "a" else [1 - hi, 1 - lo]
            e = R.eval_entry("round_robin", ckpt, opp, wr, ci, p["n_games"], seeds=p["seeds"], depth=p["depth"],
                             source=board_dir / "pairings.jsonl", opponent_name=opp_name, tag=board_dir.name,
                             side=side)
            e["recorded"] = p["timestamp"]
            by_ckpt.setdefault(R.rel(ckpt), []).append(e)
    return by_ckpt


def protocol_evals(path):
    rec = json.loads(path.read_text())
    names = {"decisive": rec["champion"]["name"], "heuristic": "heuristic_tuned", "random": "random",
             "previous": rec["previous"]["name"]}
    out = []
    for match, opp_name in names.items():
        r = rec["results"][match]
        extra = dict(draws=r["draws"], block=rec["block"], protocol=rec["protocol"], selection=rec["selection"])
        if match == "decisive":
            extra.update(verdict=rec["verdict"]["verdict"], checks=rec["verdict"]["checks"],
                         guard_reference=rec["results"]["champion_heuristic"], issue=rec["issue"])
        spec_a, spec_b = rec["specs"][match]
        e = R.eval_entry("protocol", spec_a, spec_b, r["win_rate"], r["ci"], r["n_games"], seeds=rec["seeds"][match],
                         depth=rec["depth"], source=path, opponent_name=opp_name, tag=f"b{rec['block']}:{match}", **extra)
        e["recorded"] = rec["finished"]
        out.append(e)
    return out


def build_training(run_id, spec):
    run_dir = RUNS / run_id
    cfg = json.loads((run_dir / "config.json").read_text())
    sha, date = added_commit(run_dir)
    git = dict(sha=sha, dirty=False, branch=None,
               source=f"backfill: commit that added runs/{run_id} on {date[:10] if date else '?'} (training SHA not recorded)")
    meta = R.new_meta(run_id, "training", config=cfg, title=spec["title"], init=init_of(spec["init"]),
                      script=spec.get("script"), command=spec.get("script"), git=git, created=spec["created"],
                      notes=f"Backfilled 2026-11-15 (PILA-7) from files in runs/{run_id}; history in {spec['doc']}.")
    meta.update(status="finished", backfilled=True, doc=spec["doc"], selection=spec.get("selection", ENDPOINT_UNGATED))
    meta["training"] = training_block(run_dir, cfg, spec)
    R.set_checkpoint(meta, R.checkpoint_entry(run_dir / "model.pt", role="final",
                                              round=meta["training"].get("rounds_completed", 1) - 1
                                              if spec["loop"] != "bootstrap" else None))
    for fname, role, label in spec.get("extra", []):
        R.set_checkpoint(meta, R.checkpoint_entry(run_dir / fname, role=role, label=label, primary=False))
    return meta


def build_c10():
    run_dir = RUNS / "c10_precheck"
    pre = json.loads((run_dir / "precheck.json").read_text())
    sha, date = added_commit(run_dir)
    cfg = dict(script_constants=dict(N_TRAIN=24, N_VAL=3, SEED0=500000), lr=1e-3, weight_decay=1e-4,
               batch_size=4096, value_loss_weight=2.0, aux_loss_weight=0.2, early_stopping_patience=4,
               train_seed=0, data="27 shards x 750 games of V2.1c self-play (V2.1c recipe), "
                                  "~/parchis_training_archive/c10_data", fits={k: {kk: v[kk] for kk in ("size", "n_train_shards", "minutes", "n_params")}
                                                                                for k, v in pre.items()})
    meta = R.new_meta("c10_precheck", "training", config=cfg, title="C10 pre-check: 2x256 vs 3x512 fit on V2.1c self-play",
                      init=dict(path=None, sha256=None, run_id=None, note="random init (both sizes)"),
                      script="scripts/v2/c10_precheck.py", command="scripts/v2/c10_precheck.py",
                      git=dict(sha=sha, dirty=False, branch=None,
                               source=f"backfill: commit that added runs/c10_precheck on {date[:10]} (training SHA not recorded)"),
                      created="2026-11-02",
                      notes="Backfilled 2026-11-15 (PILA-7). Fit losses per size and data size in precheck.json; "
                            "docs/V2_PLAN.md §10.10.")
    meta["seeds"] = dict(values=dict(generation_seed="500000+i, i<27", train_seed=0), base_n=None,
                         policy="legacy: seeds below 10_000_000 overlap evaluation ranges",
                         derivation="shard i of the data uses seed 500000+i")
    meta.update(status="finished", backfilled=True, doc="docs/V2_PLAN.md §10.10",
                selection="the 24-shard fit of each size (the largest data size), fixed in advance")
    meta["training"] = dict(dataset_games=27 * 750, fits=len(pre), wall_clock_s=round(60 * sum(v["minutes"] for v in pre.values()), 1),
                            wall_clock_note="sum of fit times; excludes the 11.8 min data generation")
    for size in ("2x256", "3x512"):
        R.set_checkpoint(meta, R.checkpoint_entry(run_dir / f"start_{size}.pt", role="final", label=size, primary=True))
    return meta


def build_eval_record(run_id, title, created, doc, notes):
    run_dir = RUNS / run_id
    sha, date = added_commit(run_dir)
    meta = R.new_meta(run_id, "evaluation", title=title, created=created,
                      git=dict(sha=sha, dirty=False, branch=None,
                               source=f"backfill: commit that added runs/{run_id} on {date[:10]}"), notes=notes)
    meta.update(status="finished", backfilled=True, doc=doc, seeds=dict(policy="n/a"))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    metas = {rid: build_training(rid, spec) for rid, spec in RUNS_TABLE.items()}
    metas["c10_precheck"] = build_c10()
    lb = build_eval_record("leaderboard_2026-11-15", "2-player round robin of every checkpoint (16 participants)",
                           "2026-11-15", "docs/LEADERBOARD.md",
                           "Each pairing is recorded in the run records of both participants (kind round_robin).")
    lb["evals_summary"] = dict(pairings=120, games=240000, seeds="120000+1000k+c, k<120, c<8",
                               source="runs/leaderboard_2026-11-15/pairings.jsonl",
                               ratings="runs/leaderboard_2026-11-15/leaderboard.json")
    lb["issue"] = "PILA-1"
    metas["leaderboard_2026-11-15"] = lb
    pz = build_eval_record("puzzles_2026-11-03", "Puzzle re-evaluation: V2.1c vs round-23 net at depths 1-5",
                           "2026-11-03", "docs/V2_PLAN.md (puzzles)", "Per-puzzle choices in report.json; not a match result.")
    pz["evals_summary"] = dict(source="runs/puzzles_2026-11-03/report.json", nets=["v2_1c", "round23"])
    metas["puzzles_2026-11-03"] = pz

    # evals: ladders (per run), h2h, round robin (both sides), protocol
    for rid, m in metas.items():
        if m["kind"] == "training" and RUNS_TABLE.get(rid, {}).get("loop") == "v2":
            final = next(c for c in m["checkpoints"] if c["role"] == "final")
            for e in ladder_evals(RUNS / rid, final["sha256"], m["training"]["rounds_completed"], rid):
                R.add_eval(m, e)
    for e_dir, evs in h2h_evals_by_run(metas).items():
        for e in evs:
            R.add_eval(metas[e_dir], e)
    rr = round_robin_evals(RUNS / "leaderboard_2026-11-15")
    for rid, m in metas.items():
        for c in m.get("checkpoints", []):
            for e in rr.get(c["path"], []):
                R.add_eval(m, e)
    for e in protocol_evals(RUNS / "v2_1d_champion/eval/protocol_b1.json"):
        R.add_eval(metas["v2_1d_champion"], e)

    for rid, m in metas.items():
        problems = R.validate(m)
        print(f"{rid}: {len(m['evals'])} evals, {len(m['checkpoints'])} checkpoints"
              + ("" if not problems else f"  [{'; '.join(problems)}]"))
        if not a.dry_run:
            R.save_meta(RUNS / rid, m)
    return 0


def h2h_evals_by_run(metas):
    """h2h results keyed by the run_id owning checkpoint a."""
    out = {}
    for f, a, b in H2H:
        d = json.loads((RUNS / f).read_text())
        rid = str(Path(a).parent.relative_to("runs"))
        e = R.eval_entry("h2h", a, b, d["win_rate_a"], d["ci"], d["n_games"], seeds=d.get("seeds", "70000+i") + ", i<8",
                         depth=d.get("depth", 1), source=RUNS / f,
                         opponent_name=str(Path(b).parent.relative_to("runs")), pairs=d["n_games"] // 2,
                         original_a=d["a"])
        e["recorded"] = None
        out.setdefault(rid, []).append(e)
    return {rid: evs for rid, evs in out.items() if rid in metas}


if __name__ == "__main__":
    sys.exit(main())
