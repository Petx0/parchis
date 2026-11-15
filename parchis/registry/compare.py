"""Cross-run comparison: one row per primary checkpoint of every run record under runs/.

Columns come only from the records (meta.json), the champion pointer (runs/champions.json) and the
latest round-robin board (runs/leaderboard_*/leaderboard.json). Every number carries its game
count; a cell is empty when no measured result exists -- nothing is interpolated.
"""
import csv
import io
import json
from pathlib import Path

from parchis.registry import record as R

# Which result fills a cell when several exist: a protocol run first (fresh seeds, the promotion
# rule), then the most games, then the more controlled kind, then a result recorded by this run
# over one mirrored from the opponent's record, then the most recent.
KIND_RANK = {"protocol": 0, "h2h": 1, "round_robin": 2, "ladder": 3, "gate": 4, "other": 5}


def discover(root):
    """All run records under root (skipping per-round directories), sorted by run_id."""
    root = Path(root)
    metas = []
    for p in sorted(root.rglob(R.META_NAME)):
        if "rounds" in p.relative_to(root).parts:
            continue
        m = json.loads(p.read_text())
        m["_dir"] = str(p.parent.relative_to(root))
        metas.append(m)
    return metas


def load_champion(root):
    p = Path(root) / "champions.json"
    if not p.exists():
        return None
    cur = json.loads(p.read_text()).get("2p", {}).get("current")
    return cur


def load_leaderboard(root):
    """(board dir name, {checkpoint sha256 or spec -> rating dict}) of the newest round robin."""
    boards = sorted(Path(root).glob("leaderboard_*/leaderboard.json"))
    if not boards:
        return None, {}
    board = boards[-1]
    ratings = {r["name"]: r for r in json.loads(board.read_text())["ratings"]}
    specs = {}
    pairings = board.parent / "pairings.jsonl"
    if pairings.exists():
        for line in pairings.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                specs[e["participant_a"]] = e["spec_a"]
                specs[e["participant_b"]] = e["spec_b"]
    by_key = {}
    for name, r in ratings.items():
        spec = specs.get(name, name)
        sha = R.file_sha256(R.resolve(spec)) if "/" in spec else None
        by_key[sha or spec] = dict(r, participant=name)
    return board.parent.name, by_key


def _best(evals):
    if not evals:
        return None
    newest_first = sorted(evals, key=lambda e: e.get("recorded") or "", reverse=True)
    return sorted(newest_first, key=lambda e: (e.get("kind") != "protocol", -e.get("n_games", 0),
                                               KIND_RANK.get(e.get("kind"), 9), bool(e.get("inverted"))))[0]


def _invert(e):
    """A result recorded from the other side: score 1-x, CI mirrored (duplicate-pair scores)."""
    lo, hi = e["ci"]
    return dict(e, win_rate=1 - e["win_rate"], ci=[1 - hi, 1 - lo], inverted=True)


def result_vs(metas, sha, opponent_match):
    """Best result of checkpoint `sha` against opponents satisfying opponent_match(eval), looking in
    every record: direct results first, then results recorded by the opponent (inverted)."""
    direct, inverse = [], []
    for m in metas:
        for e in m.get("evals", []):
            if e.get("checkpoint_sha256") == sha and opponent_match(e, side="opponent"):
                direct.append(e)
            elif e.get("opponent_sha256") == sha and opponent_match(e, side="checkpoint"):
                inverse.append(_invert(e))
    return _best(direct + inverse)


def rows(metas, champion=None, board=None, include_all=False):
    champ_sha = champion["sha256"] if champion else None
    board = board or {}
    out = []
    for m in metas:
        primaries = [c for c in m.get("checkpoints", []) if c.get("primary")]
        if not primaries and not include_all:
            continue
        for c in primaries or [{}]:
            sha = c.get("sha256")
            label = m["run_id"] if len(primaries) <= 1 else f"{m['run_id']}:{c.get('label') or c.get('role')}"

            def vs_champion(e, side):
                return (e.get("opponent_sha256") if side == "opponent" else e.get("checkpoint_sha256")) == champ_sha

            def vs_heuristic(e, side):
                return side == "opponent" and e.get("opponent") == "heuristic_tuned"

            protocol = [e for e in m.get("evals", []) if e.get("kind") == "protocol" and e.get("verdict")
                        and e.get("checkpoint_sha256") == sha]
            protocol.sort(key=lambda e: e.get("recorded") or "")
            t = m.get("training", {})
            init = m.get("init") or {}
            rating = board.get(sha) if sha else None
            out.append(dict(
                run=label,
                kind=m.get("kind"),
                issue=m.get("issue"),
                created=(m.get("created") or "")[:10],
                status=m.get("status"),
                init=init.get("run_id") or (Path(init["path"]).parent.name if init.get("path") else
                                            "random" if (init.get("note") or "").startswith("random") else None),
                rounds=t.get("rounds_completed"),
                games=t.get("selfplay_games") or t.get("dataset_games"),
                wall_h=round(t["wall_clock_s"] / 3600, 2) if t.get("wall_clock_s") else None,
                seeds=_seed_label(m.get("seeds", {})),
                git=_git_label(m.get("git", {})),
                checkpoint=c.get("path"),
                is_champion=bool(sha and sha == champ_sha),
                vs_champion=None if sha == champ_sha else result_vs(metas, sha, vs_champion) if sha else None,
                vs_heuristic=result_vs(metas, sha, vs_heuristic) if sha else None,
                elo=rating,
                verdict=protocol[-1]["verdict"] if protocol else None,
                verdict_source=protocol[-1]["source"] if protocol else None,
            ))
    return out


def _seed_label(s):
    if s.get("base_n"):
        return f"n={s['base_n']}"
    if s.get("policy", "").startswith("legacy"):
        return "legacy 0"
    return "—"


def _git_label(g):
    sha = (g.get("sha") or "")[:7]
    if not sha:
        return "—"
    if g.get("source", "recorded") != "recorded":
        return sha + "†"
    return sha + ("*" if g.get("dirty") else "")


def _fmt_result(e):
    if not e:
        return ""
    n = e["n_games"]
    nk = f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)
    mark = {"protocol": "P", "round_robin": "RR", "h2h": "H", "ladder": "L", "gate": "G"}.get(e.get("kind"), "?")
    inv = "ᵢ" if e.get("inverted") else ""
    return (f"{100 * e['win_rate']:.2f}% [{100 * e['ci'][0]:.2f}, {100 * e['ci'][1]:.2f}] "
            f"n={nk} {mark}{inv}")


def _fmt_elo(r):
    if not r:
        return ""
    lo, hi = r["elo_ci"]
    return f"{r['elo']:.0f} [{lo:.0f}, {hi:.0f}]"


SORTS = {
    "elo": lambda r: (-(r["elo"]["elo"]) if r["elo"] else float("inf"), r["run"]),
    "created": lambda r: (r["created"] or "", r["run"]),
    "run": lambda r: r["run"],
}


def render_markdown(rs, champion=None, board_name=None, others=()):
    head = ["Run", "Issue", "Date", "Init", "Rounds", "Self-play games", "Train h", "Seeds", "Git",
            f"vs champion {champion['name'] if champion else ''}".strip(), "vs tuned heuristic",
            f"Elo ({board_name})" if board_name else "Elo", "Verdict"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rs:
        run = f"**{r['run']}** (champion)" if r["is_champion"] else r["run"]
        cells = [run, r["issue"] or "", r["created"], r["init"] or "", _num(r["rounds"]), _num(r["games"]),
                 "" if r["wall_h"] is None else f"{r['wall_h']:.1f}", r["seeds"], r["git"],
                 "—" if r["is_champion"] else _fmt_result(r["vs_champion"]), _fmt_result(r["vs_heuristic"]),
                 _fmt_elo(r["elo"]), r["verdict"] or ""]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Win rates: score of the run's checkpoint, Wilson 95% CI, games. Source: P protocol, "
                 "RR round robin, H 10k-pair h2h, L ladder, G promotion gate; ᵢ = recorded by the "
                 "opponent and mirrored. Git: † = backfilled (commit that added the run, not the "
                 "training commit), * = dirty tree. Seeds: n = training base 10,000,000·n; legacy 0 "
                 "overlaps evaluation ranges. Full records: `python -m parchis.registry show <run>`.")
    if others:
        lines.append("")
        lines.append("Records without a ranked checkpoint: " + ", ".join(
            f"{m['run_id']} ({m.get('kind')})" for m in others) + ".")
    return "\n".join(lines)


def _num(x):
    return "" if x is None else f"{x:,}"


def render_csv(rs):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["run", "issue", "created", "init", "rounds", "selfplay_games", "train_hours", "seeds", "git",
                "vs_champion", "vs_champion_lo", "vs_champion_hi", "vs_champion_games", "vs_champion_source",
                "vs_heuristic", "vs_heuristic_games", "elo", "elo_lo", "elo_hi", "verdict", "checkpoint"])
    for r in rs:
        c, h, e = r["vs_champion"] or {}, r["vs_heuristic"] or {}, r["elo"] or {}
        w.writerow([r["run"], r["issue"], r["created"], r["init"], r["rounds"], r["games"], r["wall_h"],
                    r["seeds"], r["git"], c.get("win_rate"), (c.get("ci") or [None])[0],
                    (c.get("ci") or [None, None])[1], c.get("n_games"), c.get("source"),
                    h.get("win_rate"), h.get("n_games"), e.get("elo"), (e.get("elo_ci") or [None])[0],
                    (e.get("elo_ci") or [None, None])[1], r["verdict"], r["checkpoint"]])
    return buf.getvalue()


def report(root=None, fmt="md", sort="elo", include_all=False):
    root = Path(root) if root else R.REPO / "runs"
    metas = discover(root)
    champion = load_champion(root)
    board_name, board = load_leaderboard(root)
    rs = sorted(rows(metas, champion, board, include_all), key=SORTS[sort])
    if fmt == "json":
        return json.dumps(dict(champion=champion, leaderboard=board_name, rows=rs), indent=2, default=str)
    if fmt == "csv":
        return render_csv(rs)
    ranked = {r["run"].split(":")[0] for r in rs}
    others = [m for m in metas if m["run_id"] not in ranked]
    return render_markdown(rs, champion, board_name, others)
