"""Puzzle report: every legal move's search value on my_puzzles.csv, for the current best net and the
net of the last puzzle run (round-23 champion, docs/AZ_DESIGN.md "Puzzle accuracy vs. search depth",
2026-08-29), at depths 1-5.

Usage: puzzle_report.py <out.json> [max_depth=5] [name=run_dir,...] [csv]
       puzzle_report.py merge <out.json> <part.json> [<part.json> ...]

The optional third argument replaces the default nets (e.g. a single net per invocation, so each part
fits a short foreground job); `merge` joins per-net parts of the same CSV into one report.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from parchis.agents import agent_spec
from parchis.evaluation.puzzles.loader import load_puzzles
from parchis.evaluation.puzzles.runner import decide_with_breakdown

NETS = {"v2_1c": "runs/v2_1c_champion", "round23": "runs/selfplay_2p_v1_champion"}
CSV = REPO / "parchis/evaluation/puzzles/my_puzzles.csv"

def merge(out, parts):
    reports = [json.loads(Path(p).read_text()) for p in parts]
    base = reports[0]
    for r in reports[1:]:
        assert r["csv"] == base["csv"], "parts come from different CSVs"
        base["nets"].update(r["nets"])
        for pa, pb in zip(base["puzzles"], r["puzzles"]):
            assert pa["puzzle_id"] == pb["puzzle_id"]
            pa["results"].update(pb["results"])
    Path(out).write_text(json.dumps(base, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3:])
        sys.exit(0)
    out = Path(sys.argv[1])
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if len(sys.argv) > 3 and sys.argv[3]:
        NETS = dict(item.split("=", 1) for item in sys.argv[3].split(","))
    if len(sys.argv) > 4:
        CSV = (REPO / sys.argv[4]).resolve()
    puzzles = load_puzzles(str(CSV))
    report = {"csv": str(CSV.relative_to(REPO)), "nets": NETS, "puzzles": []}
    for case in puzzles:
        player = case.game.get_current_player()
        entry = {
            "puzzle_id": case.puzzle_id, "category": case.category, "roll": case.roll,
            "consecutive_sixes": case.consecutive_sixes, "pending_bonus": case.pending_bonus,
            "mover": player.color.name if hasattr(player.color, "name") else str(player.color),
            "correct_piece_ids": list(case.correct_piece_ids), "rationale": case.rationale,
            "moves": [{"piece_id": m[0].piece_id, "from": m[0].position, "to": m[1], "kind": m[2]}
                      for m in case.legal_moves],
            "results": {},
        }
        for net, run_dir in NETS.items():
            for depth in range(1, max_depth + 1):
                kind, params, _ = agent_spec.parse_spec(f"checkpoint:{REPO / run_dir}:depth={depth}")
                t0 = time.time()
                _move, d = decide_with_breakdown(kind, params, case)
                values = {pid: vec[case.acting_seat] for pid, vec in d["move_values"].items()}
                entry["results"][f"{net}_d{depth}"] = {
                    "chosen": d["chosen_piece_id"], "values": values,
                    "root_value": d["root_value"][case.acting_seat], "seconds": time.time() - t0,
                }
        report["puzzles"].append(entry)
        print(f"puzzle {case.puzzle_id}: " + " ".join(
            f"{k}={v['chosen']}{'+' if v['chosen'] in case.correct_piece_ids else '-'}"
            for k, v in entry["results"].items()), flush=True)
    out.write_text(json.dumps(report, indent=1))
    print("PUZZLES DONE", flush=True)
