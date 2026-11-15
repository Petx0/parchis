"""Tests for the experiment registry (parchis/registry, docs/EXPERIMENT_REGISTRY.md, PILA-7)."""
import json

import pytest

from parchis.registry import compare as C
from parchis.registry import record as R
from parchis.registry.__main__ import main as cli


def _ckpt(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# --- seeds -----------------------------------------------------------------------------------
def test_allocate_training_base_is_monotonic_and_written_to_every_registry(tmp_path):
    regs = [tmp_path / "home.jsonl", tmp_path / "repo.jsonl"]
    regs[1].write_text(json.dumps({"kind": "protocol", "block": 7}) + "\n")
    a = R.allocate_training_base("run_a", "PILA-9", registries=regs)
    b = R.allocate_training_base("run_b", "PILA-9", registries=regs)
    assert (a["n"], a["base"]) == (1, 10_000_000) and (b["n"], b["base"]) == (2, 20_000_000)
    for r in regs:
        lines = [json.loads(x) for x in r.read_text().splitlines()]
        assert [e["run_id"] for e in lines if e["kind"] == "training"] == ["run_a", "run_b"]
    # an n recorded only in one registry still blocks reuse
    with regs[0].open("a") as fh:
        fh.write(json.dumps({"kind": "training", "n": 9, "run_id": "elsewhere"}) + "\n")
    assert R.allocate_training_base("run_c", "PILA-9", registries=regs)["n"] == 10


def test_apply_seed_base_and_describe_seeds(tmp_path):
    class Cfg:
        generation_seed = train_seed = promotion_seed = 0
    cfg = R.apply_seed_base(Cfg(), 30_000_000)
    assert (cfg.generation_seed, cfg.train_seed, cfg.promotion_seed) == (30_000_000,) * 3
    with pytest.raises(ValueError):
        R.apply_seed_base(Cfg(), 70_000)
    reg = tmp_path / "r.jsonl"
    reg.write_text(json.dumps({"kind": "training", "n": 3, "run_id": "x"}) + "\n")
    ok = R.describe_seeds(dict(generation_seed=30_000_000, train_seed=30_000_000, promotion_seed=30_000_000),
                          registries=[reg])
    assert ok["base_n"] == 3 and ok["policy"] == "ok" and ok["registered_to"] == ["x"]
    legacy = R.describe_seeds(dict(generation_seed=0, train_seed=0, promotion_seed=0), registries=[reg])
    assert legacy["base_n"] is None and legacy["policy"].startswith("legacy")


# --- records ---------------------------------------------------------------------------------
def test_begin_run_creates_then_records_resumed_sessions(tmp_path):
    run_dir = tmp_path / "scratch" / "exp1"
    m = R.begin_run(run_dir, kind="training", config={"lr": 1e-3}, issue="PILA-9", command="drv.py")
    assert m["schema"] == R.SCHEMA and m["run_id"] == "exp1" and m["issue"] == "PILA-9"
    assert m["git"]["source"] == "recorded" and len(m["sessions"]) == 1
    m = R.begin_run(run_dir, kind="training", config={"lr": 1e-3}, command="drv.py")
    assert len(m["sessions"]) == 2 and "config_changed" not in m["sessions"][1]
    m = R.begin_run(run_dir, kind="training", config={"lr": 3e-4}, command="drv.py")
    assert m["sessions"][2]["config_changed"] and m["config"] == {"lr": 3e-4}
    assert m["config_history"][0]["config"] == {"lr": 1e-3}


def test_record_eval_finds_owner_run_and_is_idempotent(tmp_path):
    run_dir = tmp_path / "scratch" / "exp1"
    R.begin_run(run_dir, kind="training", config={}, command="x")
    ck = _ckpt(run_dir / "champion.pt", b"weights")
    other = _ckpt(tmp_path / "other.pt", b"other")
    e = R.eval_entry("h2h", ck, other, 0.52, [0.51, 0.53], 20000, seeds="70000+i", depth=1, source=tmp_path / "o.json")
    assert R.record_eval(e) == run_dir
    assert R.record_eval(dict(e, win_rate=0.53)) == run_dir          # same key: replaced, not duplicated
    evals = R.load_meta(run_dir)["evals"]
    assert len(evals) == 1 and evals[0]["win_rate"] == 0.53
    assert evals[0]["checkpoint_sha256"] == R.file_sha256(ck) and evals[0]["opponent_sha256"] == R.file_sha256(other)
    # builtin opponents stay names; a checkpoint outside any run is reported, not raised
    h = R.eval_entry("ladder", ck, "heuristic_tuned", 0.7, [0.68, 0.72], 2000, seeds="9000+i", depth=1)
    assert h["opponent"] == "heuristic_tuned" and h["opponent_sha256"] is None
    stray = R.eval_entry("h2h", other, ck, 0.5, [0.4, 0.6], 10, seeds="s", depth=1)
    assert R.record_eval(stray) is None


def test_validate_lists_gaps():
    m = R.new_meta("r", "training", config={"generation_seed": 0, "train_seed": 0, "promotion_seed": 0},
                   git=dict(sha="abc", dirty=False, source="recorded"))
    problems = R.validate(m)
    assert "missing checkpoints" in problems and "missing training" in problems
    assert any("seeds below 10_000_000" in p for p in problems)
    m["checkpoints"] = [dict(role="final", path="x.pt", sha256="s", primary=True)]
    m["training"] = {"rounds_completed": 1}
    m["selection"] = "endpoint"
    m["config"] = {"generation_seed": 10_000_000, "train_seed": 10_000_000, "promotion_seed": 10_000_000}
    m["seeds"] = R.describe_seeds(m["config"], registries=[])
    assert R.validate(m) == []


# --- round_loop writes the record --------------------------------------------------------------
def test_run_continuous_writes_and_updates_meta(tmp_path, monkeypatch):
    from parchis.az import round_loop
    from parchis.tests.test_round_loop import _forced_promotion_result, _tiny_cfg
    cfg = _tiny_cfg(tmp_path, run_name="recorded")
    monkeypatch.setattr(round_loop.duplicate, "play_duplicate_match", lambda *a, **k: _forced_promotion_result(True))
    round_loop.run_continuous(cfg, runs_dir=str(tmp_path / "runs"), max_rounds=2, record=dict(issue="PILA-9"))
    run_dir = tmp_path / "runs" / "recorded"
    m = R.load_meta(run_dir)
    assert m["status"] == "stopped" and m["issue"] == "PILA-9" and m["kind"] == "training"
    assert m["training"]["rounds_completed"] == 2 and [r["round"] for r in m["training"]["rounds"]] == [0, 1]
    assert m["training"]["selfplay_games"] == 2 * cfg.n_games_per_round
    final = [c for c in m["checkpoints"] if c["role"] == "final"]
    assert len(final) == 1 and final[0]["sha256"] == R.file_sha256(run_dir / "champion.pt")
    assert m["config"]["run_name"] == "recorded" and len(m["sessions"]) == 1
    round_loop.run_continuous(cfg, runs_dir=str(tmp_path / "runs"), max_rounds=3)
    m = R.load_meta(run_dir)
    assert m["training"]["rounds_completed"] == 3 and len(m["sessions"]) == 2


def test_run_continuous_marks_failed_runs(tmp_path, monkeypatch):
    from parchis.az import round_loop
    from parchis.tests.test_round_loop import _tiny_cfg
    cfg = _tiny_cfg(tmp_path, run_name="crashes")

    def boom(*a, **k):
        raise RuntimeError("gate exploded")
    monkeypatch.setattr(round_loop.duplicate, "play_duplicate_match", boom)
    with pytest.raises(RuntimeError):
        round_loop.run_continuous(cfg, runs_dir=str(tmp_path / "runs"), max_rounds=1)
    m = R.load_meta(tmp_path / "runs" / "crashes")
    assert m["status"] == "failed" and "gate exploded" in m["error"]


# --- compare ---------------------------------------------------------------------------------
def _fixture_runs(tmp_path):
    root = tmp_path / "runs"
    champ = _ckpt(root / "champ" / "model.pt", b"champion")
    cand = _ckpt(root / "cand" / "model.pt", b"candidate")
    old = _ckpt(root / "old" / "model.pt", b"old")
    (root / "champions.json").write_text(json.dumps(
        {"2p": {"current": {"name": "champ", "path": "champ/model.pt", "sha256": R.file_sha256(champ)}}}))
    git = dict(sha="0123456789", dirty=False, source="recorded")
    for name, ck in (("champ", champ), ("cand", cand), ("old", old)):
        m = R.new_meta(name, "training", config={}, git=git, created="2026-11-15")
        R.set_checkpoint(m, R.checkpoint_entry(ck))
        m["training"] = {"rounds_completed": 3, "selfplay_games": 18000}
        R.save_meta(root / name, m)
    rec = lambda kind, a, b, wr, n, **kw: R.eval_entry(kind, a, b, wr, [wr - 0.01, wr + 0.01], n,  # noqa: E731
                                                       seeds="s", depth=1, **kw)
    R.record_eval(rec("ladder", cand, champ, 0.55, 2000), run_dir=root / "cand")
    R.record_eval(rec("h2h", cand, champ, 0.505, 20000), run_dir=root / "cand")
    R.record_eval(rec("protocol", cand, champ, 0.51, 20000, verdict="PASS"), run_dir=root / "cand")
    R.record_eval(rec("ladder", cand, "heuristic_tuned", 0.74, 2000), run_dir=root / "cand")
    # 'old' never played the champion; the champion's record has a result against it
    R.record_eval(rec("h2h", champ, old, 0.60, 20000), run_dir=root / "champ")
    return root


def test_compare_picks_protocol_then_mirrors_opponent_results(tmp_path):
    root = _fixture_runs(tmp_path)
    rows = {r["run"]: r for r in C.rows(C.discover(root), C.load_champion(root))}
    assert rows["champ"]["is_champion"] and rows["champ"]["vs_champion"] is None
    c = rows["cand"]["vs_champion"]
    assert c["kind"] == "protocol" and c["win_rate"] == 0.51 and rows["cand"]["verdict"] == "PASS"
    assert rows["cand"]["vs_heuristic"]["win_rate"] == 0.74
    o = rows["old"]["vs_champion"]
    assert o["inverted"] and o["win_rate"] == pytest.approx(0.40) and o["ci"] == pytest.approx([0.39, 0.41])


def test_compare_cli_renders_every_format(tmp_path, capsys):
    root = _fixture_runs(tmp_path)
    assert cli(["--root", str(root), "compare"]) == 0
    md = capsys.readouterr().out
    assert "| **champ** (champion) |" in md and "51.00% [50.00, 52.00] n=20k P" in md and "40.00%" in md
    assert cli(["--root", str(root), "compare", "--format", "json", "--sort", "created"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {r["run"] for r in data["rows"]} == {"champ", "cand", "old"}
    assert cli(["--root", str(root), "compare", "--format", "csv"]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("run,issue,created")
    assert cli(["--root", str(root), "show", "cand"]) == 0
    assert "| protocol |" in capsys.readouterr().out
    assert cli(["--root", str(root), "validate"]) == 1             # fixtures lack selection + new seeds


def test_backfilled_repo_records_render():
    """The committed records for existing runs load, validate and render (PILA-7 'done when')."""
    root = R.REPO / "runs"
    metas = C.discover(root)
    if not metas:
        pytest.skip("no run records in this checkout")
    assert all(R.validate(m) == [] for m in metas), {m["run_id"]: R.validate(m) for m in metas if R.validate(m)}
    out = C.report(root)
    assert "(champion)" in out and "v2_1d_champion" in out
