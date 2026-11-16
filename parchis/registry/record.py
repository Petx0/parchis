"""The run record: `<run_dir>/meta.json` (schema `parchis.run/1`, docs/EXPERIMENT_REGISTRY.md).

One file per run, written by the code that produces the run:
  * `begin_run` / `update_run`  -- training loops (parchis.az.round_loop does it for every V2 run);
  * `record_eval`               -- every evaluation script, against the run that owns the checkpoint;
  * `allocate_training_base`    -- the training seed base 10_000_000*n, appended to the seed registry.

Everything here is plain JSON I/O so that writing a record can never cost a training run: callers
in hot paths wrap it with `safe()`, which logs and swallows errors.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "parchis.run/1"
META_NAME = "meta.json"

# Seed registry (append-only, docs/EVAL_PROTOCOL.md §5): repo copy + the Mac's authoritative copy.
REPO_SEED_REGISTRY = REPO / "runs/eval_seed_registry.jsonl"
HOME_SEED_REGISTRY = Path.home() / "parchis_eval_seeds.jsonl"
TRAINING_BASE_UNIT = 10_000_000          # training run n uses seeds base = 10_000_000*n (lab rules)
SEED_KEYS = ("generation_seed", "train_seed", "promotion_seed")

KINDS = ("training", "evaluation", "analysis")
EVAL_KINDS = ("protocol", "h2h", "ladder", "round_robin", "gate", "other")


# --- small helpers ---------------------------------------------------------------------------
def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


BUILTIN_OPPONENTS = ("random", "heuristic_tuned", "heuristic_default")


def rel(path):
    """Repo-relative path if inside the repo, `~/...` if under home, else absolute. Builtin opponent
    specs ('random', 'heuristic_tuned', ...) pass through unchanged. A relative path is taken from
    the cwd, or from the repo root when only that exists."""
    if path is None:
        return None
    s = str(path)
    if s in BUILTIN_OPPONENTS:
        return s
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = REPO / p if (REPO / p).exists() and not p.exists() else Path.cwd() / p
    p = p.resolve()
    for root, prefix in ((REPO, ""), (Path.home(), "~/")):
        try:
            return prefix + str(p.relative_to(root))
        except ValueError:
            pass
    return str(p)


def resolve(path):
    """Inverse of rel(): repo-relative and ~/ paths back to absolute Paths."""
    s = str(path)
    if s.startswith("~/"):
        return Path.home() / s[2:]
    p = Path(s)
    return p if p.is_absolute() else REPO / p


def file_sha256(path):
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_state(cwd=REPO):
    def run(*args):
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    sha = run("rev-parse", "HEAD")
    if sha is None:
        return dict(sha=None, dirty=None, branch=None, source="unavailable")
    dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    return dict(sha=sha, dirty=dirty, branch=run("rev-parse", "--abbrev-ref", "HEAD"), source="recorded")


def _json_default(o):
    if hasattr(o, "item"):                   # numpy / torch scalars
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def load_meta(run_dir):
    p = Path(run_dir) / META_NAME
    return json.loads(p.read_text()) if p.exists() else None


def save_meta(run_dir, meta):
    """Atomic write (tmp file + rename) so a crash never leaves a half-written record."""
    p = Path(run_dir) / META_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=_json_default) + "\n")
    os.replace(tmp, p)
    return p


def safe(fn, *args, **kwargs):
    """Run a registry write without ever breaking the caller (training/eval must not die on it)."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 -- deliberately broad, logged
        print(f"[registry] WARNING: {fn.__name__} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return None


# --- seeds -----------------------------------------------------------------------------------
def describe_seeds(config, registries=None):
    """Seed block of a training config: base n, the derivation, and whether it follows the policy."""
    vals = {k: config.get(k) for k in SEED_KEYS if k in config}
    out = dict(values=vals, base_n=None, policy="n/a")
    if not vals:
        return out
    gen = vals.get("generation_seed")
    if all(isinstance(v, int) and v >= TRAINING_BASE_UNIT and v % TRAINING_BASE_UNIT == 0 for v in vals.values()) \
            and len(set(vals.values())) == 1:
        n = gen // TRAINING_BASE_UNIT
        out.update(base_n=n, policy="ok")
        regs = registries if registries is not None else (HOME_SEED_REGISTRY, REPO_SEED_REGISTRY)
        entries = [e for r in regs for e in _read_jsonl(r) if e.get("kind") == "training" and e.get("n") == n]
        out["registered_to"] = sorted({e.get("run_id") for e in entries}) or None
    else:
        out["policy"] = "legacy: seeds below 10_000_000 overlap evaluation ranges"
    if "generation_seed" in vals:
        out["derivation"] = ("generation = generation_seed + round*1000 + shard; "
                             "promotion = promotion_seed + round; train = train_seed")
    return out


def _read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def allocate_training_base(run_id, issue, registries=None, note=None):
    """Allocate the next training seed base n >= 1 and append it to the seed registry.
    Returns dict(n=..., base=10_000_000*n). Never reuses an n, even from an aborted run."""
    regs = list(registries) if registries is not None else [HOME_SEED_REGISTRY, REPO_SEED_REGISTRY]
    used = [e["n"] for r in regs for e in _read_jsonl(r) if e.get("kind") == "training" and "n" in e]
    n = max(used) + 1 if used else 1
    entry = dict(kind="training", n=n, base=TRAINING_BASE_UNIT * n,
                 range=f"{TRAINING_BASE_UNIT * n}..{TRAINING_BASE_UNIT * (n + 1) - 1}",
                 run_id=run_id, issue=issue, allocated=now())
    if note:
        entry["note"] = note
    line = json.dumps(entry, sort_keys=True) + "\n"
    for p in regs:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(line)
    return dict(n=n, base=TRAINING_BASE_UNIT * n)


def apply_seed_base(cfg, base):
    """Set generation_seed, train_seed and promotion_seed of a config to the allocated base."""
    if base < TRAINING_BASE_UNIT or base % TRAINING_BASE_UNIT:
        raise ValueError(f"training seed base must be a positive multiple of {TRAINING_BASE_UNIT}, got {base}")
    for k in SEED_KEYS:
        if hasattr(cfg, k):
            setattr(cfg, k, base)
    return cfg


# --- run records -----------------------------------------------------------------------------
def checkpoint_entry(path, role="final", label=None, primary=None, **extra):
    """One entry of meta['checkpoints']. `primary` checkpoints are the ones `compare` ranks."""
    e = dict(role=role, path=rel(path), sha256=file_sha256(resolve(rel(path))))
    if label:
        e["label"] = label
    e["primary"] = (role == "final") if primary is None else bool(primary)
    e.update(extra)
    return e


def set_checkpoint(meta, entry):
    """Insert or replace the checkpoint with the same role+label."""
    key = (entry.get("role"), entry.get("label"))
    meta["checkpoints"] = [c for c in meta.get("checkpoints", []) if (c.get("role"), c.get("label")) != key]
    meta["checkpoints"].append(entry)
    return meta


def new_meta(run_id, kind, config=None, issue=None, title=None, init=None, script=None, command=None,
             git=None, created=None, notes=None, seeds=None):
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    config = dict(config or {})
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "title": title,
        "kind": kind,
        "issue": issue,
        "status": "running",
        "created": created or now(),
        "finished": None,
        "git": git or git_state(),
        "script": script,
        "command": command,
        "config": config,
        "init": init,
        "seeds": seeds or describe_seeds(config),
        "checkpoints": [],
        "training": {},
        "selection": None,
        "evals": [],
        "sessions": [],
        "notes": notes,
    }


def init_entry(path):
    """`init` field: the checkpoint a run warm-started from (path + sha256 + owning run if known)."""
    if path is None:
        return None
    r = find_run_dir(resolve(rel(path)))
    m = load_meta(r) if r else None
    return dict(path=rel(path), sha256=file_sha256(resolve(rel(path))), run_id=m["run_id"] if m else None)


def begin_run(run_dir, *, kind, config, run_id=None, issue=None, title=None, init_path=None,
              script=None, command=None, notes=None):
    """Create the record (first start) or register a resumed session (later starts). Every session
    records its own git SHA, so a run resumed on different code says so."""
    run_dir = Path(run_dir)
    git = git_state()
    command = command if command is not None else " ".join([rel(sys.argv[0]) or ""] + sys.argv[1:])
    session = dict(started=now(), git_sha=git["sha"], git_dirty=git["dirty"], command=command, pid=os.getpid())
    meta = load_meta(run_dir)
    if meta is None:
        meta = new_meta(run_id or run_dir.name, kind, config=config, issue=issue, title=title,
                        init=init_entry(init_path), script=script or rel(sys.argv[0]), command=command,
                        git=git, notes=notes)
    else:
        if dict(config) != meta.get("config"):
            session["config_changed"] = True
            meta.setdefault("config_history", []).append(dict(at=now(), config=meta.get("config")))
            meta["config"] = dict(config)
            meta["seeds"] = describe_seeds(meta["config"])
        if issue and not meta.get("issue"):
            meta["issue"] = issue
        if init_path and not meta.get("init"):
            meta["init"] = init_entry(init_path)
        meta["status"] = "running"
    meta.setdefault("sessions", []).append(session)
    save_meta(run_dir, meta)
    return meta


def update_run(run_dir, mutate):
    """Load, apply `mutate(meta)` in place, save. Returns the saved meta (None if no record)."""
    meta = load_meta(run_dir)
    if meta is None:
        return None
    mutate(meta)
    save_meta(run_dir, meta)
    return meta


def find_run_dir(path, max_up=3):
    """The run directory owning a checkpoint or file: the nearest ancestor (up to `max_up` levels)
    holding a meta.json. Stops at the repo's runs/ and the repo root."""
    p = Path(path).expanduser()
    p = p if p.is_dir() else p.parent
    stops = {REPO / "runs", REPO, Path(p.anchor)}
    for _ in range(max_up):
        if p in stops:
            return None
        if (p / META_NAME).exists():
            return p
        p = p.parent
    return None


def eval_entry(kind, checkpoint, opponent, win_rate, ci, n_games, *, seeds, depth, source=None,
               opponent_name=None, tag=None, **extra):
    """Normalise one evaluation result. win_rate/ci are the checkpoint's score (wins + draws/2)."""
    if kind not in EVAL_KINDS:
        raise ValueError(f"eval kind must be one of {EVAL_KINDS}")
    opp_path = resolve(rel(opponent)) if opponent not in (None,) + BUILTIN_OPPONENTS else None
    if opponent_name is None and opp_path is not None:
        owner = find_run_dir(opp_path)
        opponent_name = (load_meta(owner) or {}).get("run_id") if owner else None
    e = dict(kind=kind, checkpoint=rel(checkpoint), checkpoint_sha256=file_sha256(resolve(rel(checkpoint))),
             opponent=rel(opponent) if opp_path else opponent, opponent_name=opponent_name,
             opponent_sha256=file_sha256(opp_path) if opp_path else None,
             win_rate=float(win_rate), ci=[float(ci[0]), float(ci[1])], n_games=int(n_games),
             seeds=seeds, depth=depth, source=rel(source) if source else None, tag=tag, recorded=now())
    e.update(extra)
    return e


def _eval_key(e):
    return (e.get("kind"), e.get("checkpoint_sha256") or e.get("checkpoint"), e.get("opponent"),
            e.get("source"), e.get("tag"))


def add_eval(meta, entry):
    """Insert or replace (same kind, checkpoint, opponent, source, tag) -- re-recording is idempotent."""
    key = _eval_key(entry)
    meta["evals"] = [e for e in meta.get("evals", []) if _eval_key(e) != key] + [entry]
    return meta


def record_eval(entry, run_dir=None):
    """Append an eval_entry() to the record of the run owning entry['checkpoint'].
    Returns the run dir written, or None when the checkpoint belongs to no recorded run."""
    run_dir = Path(run_dir) if run_dir else find_run_dir(resolve(entry["checkpoint"]))
    if run_dir is None or load_meta(run_dir) is None:
        print(f"[registry] no meta.json owns {entry['checkpoint']}; eval not recorded in a run record",
              file=sys.stderr, flush=True)
        return None
    update_run(run_dir, lambda m: add_eval(m, entry))
    return run_dir


REQUIRED = {
    "training": ("run_id", "kind", "git.sha", "config", "seeds", "checkpoints", "training", "selection"),
    "evaluation": ("run_id", "kind", "git.sha"),
    "analysis": ("run_id", "kind", "git.sha"),
}


def validate(meta):
    """List of problems that make a record incomplete for a handoff (empty list = complete)."""
    problems = []
    if meta.get("schema") != SCHEMA:
        problems.append(f"schema is {meta.get('schema')!r}, expected {SCHEMA!r}")
    for field in REQUIRED.get(meta.get("kind"), ("kind",)):
        v = meta
        for part in field.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        if v in (None, "", [], {}):
            problems.append(f"missing {field}")
    if meta.get("kind") == "evaluation" and not (meta.get("evals") or meta.get("evals_summary")):
        problems.append("missing evals (or evals_summary)")
    for c in meta.get("checkpoints", []):
        if not c.get("sha256"):
            problems.append(f"checkpoint {c.get('path')} has no sha256 (file missing?)")
    if meta.get("kind") == "training":
        if meta.get("seeds", {}).get("policy", "").startswith("legacy") and not meta.get("backfilled"):
            problems.append("training seeds below 10_000_000 (use allocate_training_base)")
        if not any(c.get("primary") for c in meta.get("checkpoints", [])):
            problems.append("no primary checkpoint")
    for e in meta.get("evals", []):
        for k in ("win_rate", "ci", "n_games", "seeds", "depth"):
            if e.get(k) in (None, ""):
                problems.append(f"eval {e.get('kind')} vs {e.get('opponent_name') or e.get('opponent')}: missing {k}")
    return problems
