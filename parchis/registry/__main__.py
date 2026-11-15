"""Experiment registry CLI (docs/EXPERIMENT_REGISTRY.md).

    python -m parchis.registry compare [--format md|csv|json] [--sort elo|created|run] [--all]
    python -m parchis.registry show <run_id>
    python -m parchis.registry validate [<run_id> ...]          # exit 1 if a record is incomplete
    python -m parchis.registry alloc-seed --run-id <id> --issue PILA-N
    python -m parchis.registry curate <scratch_run_dir> <run_id> [--issue PILA-N]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from parchis.registry import compare as C
from parchis.registry import record as R


def _runs_root(a):
    return Path(a.root) if a.root else R.REPO / "runs"


def cmd_compare(a):
    print(C.report(_runs_root(a), fmt=a.format, sort=a.sort, include_all=a.all))
    return 0


def _find(root, run_id):
    d = root / run_id
    m = R.load_meta(d)
    if m is None:
        sys.exit(f"no run record {d / R.META_NAME}")
    return d, m


def cmd_show(a):
    root = _runs_root(a)
    d, m = _find(root, a.run_id)
    t = m.get("training", {})
    print(f"{m['run_id']} — {m.get('title') or ''}")
    print(f"kind {m['kind']}  issue {m.get('issue')}  status {m.get('status')}  created {m.get('created')}")
    g = m.get("git", {})
    print(f"git {g.get('sha')} ({g.get('source')}{', dirty' if g.get('dirty') else ''})  script {m.get('script')}")
    print(f"init {json.dumps(m.get('init'))}")
    print(f"seeds {json.dumps(m.get('seeds'))}")
    if t:
        print(f"training: {t.get('rounds_completed')} rounds, {t.get('promotions')} promotions, "
              f"{t.get('selfplay_games') or t.get('dataset_games')} games, {t.get('wall_clock_s')} s")
    print(f"selection: {m.get('selection')}")
    for c in m.get("checkpoints", []):
        print(f"checkpoint [{c['role']}{'/' + c['label'] if c.get('label') else ''}{', primary' if c.get('primary') else ''}] "
              f"{c['path']} sha256 {(c.get('sha256') or 'MISSING')[:12]}")
    if m.get("evals"):
        print()
        print("| Kind | Checkpoint | Opponent | Win rate | Wilson 95% CI | Games | Seeds | Depth | Tag | Source |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for e in m["evals"]:
            print(f"| {e['kind']} | {e['checkpoint']} | {e.get('opponent_name') or e.get('opponent')} | "
                  f"{100 * e['win_rate']:.2f}% | [{100 * e['ci'][0]:.2f}, {100 * e['ci'][1]:.2f}] | "
                  f"{e['n_games']:,} | {e['seeds']} | {e['depth']} | {e.get('tag') or ''}"
                  f"{' ' + e['verdict'] if e.get('verdict') else ''} | {e.get('source') or ''} |")
    if m.get("notes"):
        print(f"\nnotes: {m['notes']}")
    return 0


def cmd_validate(a):
    root = _runs_root(a)
    metas = [_find(root, r)[1] for r in a.run_ids] if a.run_ids else C.discover(root)
    bad = 0
    for m in metas:
        problems = R.validate(m)
        print(f"{'OK  ' if not problems else 'FAIL'} {m['run_id']}" + ("" if not problems else ": " + "; ".join(problems)))
        bad += bool(problems)
    return 1 if bad else 0


def cmd_alloc_seed(a):
    got = R.allocate_training_base(a.run_id, a.issue, note=a.note)
    print(json.dumps(got))
    print(f"use: registry.apply_seed_base(cfg, {got['base']})  (generation_seed = train_seed = "
          f"promotion_seed = {got['base']:,})", file=sys.stderr)
    return 0


# Small, curated files copied from a scratch run dir into runs/<run_id>/ (lab rules: no shards,
# no intermediate candidates).
CURATE_TOP = ("config.json", "champion_meta.json", "promoted_history.json", "recent_history.json")
CURATE_ROUND = ("metrics.jsonl", "promotion_result.json", "done.json")


def cmd_curate(a):
    src = Path(a.src).expanduser().resolve()
    dst = R.REPO / "runs" / a.run_id
    meta = R.load_meta(src)
    if meta is None:
        sys.exit(f"{src} has no {R.META_NAME}; nothing to curate")
    if not (src / "champion.pt").exists():
        sys.exit(f"{src}/champion.pt not found")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "champion.pt", dst / "model.pt")
    for f in CURATE_TOP:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    for rd in sorted((src / "rounds").glob("round_*")) if (src / "rounds").exists() else []:
        for f in CURATE_ROUND:
            if (rd / f).exists():
                (dst / "rounds" / rd.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(rd / f, dst / "rounds" / rd.name / f)
    src_ckpt = next((c for c in meta.get("checkpoints", []) if c.get("role") == "final"), {})
    final = R.checkpoint_entry(dst / "model.pt", role="final", round=src_ckpt.get("round"),
                               source=R.rel(src / "champion.pt"))
    if src_ckpt.get("sha256") and src_ckpt["sha256"] != final["sha256"]:
        print("WARNING: champion.pt changed since the last recorded round", file=sys.stderr)
    meta = R.set_checkpoint(meta, final)
    for e in meta.get("evals", []):          # evals keep their checkpoint sha; point paths at the copy
        if e.get("checkpoint_sha256") == final["sha256"]:
            e["checkpoint"] = final["path"]
    meta.update(run_id=a.run_id, curated=dict(at=R.now(), source_dir=R.rel(src)))
    if a.issue and not meta.get("issue"):
        meta["issue"] = a.issue
    R.save_meta(dst, meta)
    problems = R.validate(meta)
    print(f"curated {R.rel(src)} -> {R.rel(dst)}; record " + ("complete" if not problems else "INCOMPLETE: " + "; ".join(problems)))
    return 0 if not problems else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m parchis.registry", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="runs directory (default: the repo's runs/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare", help="table of every run")
    p.add_argument("--format", choices=("md", "csv", "json"), default="md")
    p.add_argument("--sort", choices=tuple(C.SORTS), default="elo")
    p.add_argument("--all", action="store_true", help="also list records without a ranked checkpoint")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("show", help="one run's full record and every eval")
    p.add_argument("run_id")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("validate", help="check records are complete (handoff gate)")
    p.add_argument("run_ids", nargs="*")
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("alloc-seed", help="allocate the next training seed base 10,000,000*n")
    p.add_argument("--run-id", required=True)
    p.add_argument("--issue", required=True)
    p.add_argument("--note")
    p.set_defaults(fn=cmd_alloc_seed)
    p = sub.add_parser("curate", help="copy a finished scratch run into runs/<run_id>/ with its record")
    p.add_argument("src")
    p.add_argument("run_id")
    p.add_argument("--issue")
    p.set_defaults(fn=cmd_curate)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
