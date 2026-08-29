#!/usr/bin/env python3
"""
Runner for the tactical puzzle suite (docs/AGENT_REBUILD_PLAN.md
Part 5.4): scores any agent (search checkpoint, heuristic, random -- the
same `checkpoint:<run_dir>[:depth=N] | heuristic:tuned|default | random`
grammar `parchis.evaluation.ladder`/`parchis.visualization.play_instrumented_game`
already use, via `parchis.agents.agent_spec`) against a set of
hand-authored puzzles, reporting `puzzle_accuracy` overall and per
category.

Deliberately does NOT go through `agent_spec.build_factory`'s arena-style
factory: that factory builds its own `TurnContextTracker`, inferring
`(roll, pending_bonus, consecutive_sixes)` from a live, multi-turn
`roll_box` -- the wrong model here, since a puzzle is one fully-specified
decision with no turn history to infer from. `decide()` below dispatches
directly to the same underlying calls a real factory would eventually
make (`search.search` / `heuristic.choose_move_with_weights` /
`random.choice`) for exactly the context
`parchis.evaluation.puzzles.loader` already computed and validated.

Usage:
    python -m parchis.evaluation.puzzles --agent heuristic:tuned
    python -m parchis.evaluation.puzzles \\
        --agent checkpoint:runs/selfplay_2p_v1_champion:depth=1 \\
        --csv parchis/evaluation/puzzles/tactical_puzzles.csv \\
        --json-out runs/puzzle_results.json
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from parchis.agents import agent_spec, heuristic
from parchis.az import search as az_search
from parchis.evaluation.puzzles.loader import load_puzzles

DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "tactical_puzzles.csv"


def decide_with_breakdown(kind, params, case, rng=None):
    """Like decide() below, but also returns a `decision` dict in the SAME
    shape parchis.visualization.agentinfo_io saves/loads (see that
    module's schema docstring) -- {"kind", "seat", "chosen_piece_id", and
    either "root_value"/"move_values" (search) or "move_scores"
    (heuristic)}. This lets parchis.visualization.visualizer's
    draw_value_panel render a puzzle's per-move breakdown with ZERO new
    rendering code -- the exact same bar chart it already draws for a real
    game's search/heuristic decisions (see visualize_puzzles.py). A
    "random" decision has no evaluation to show, matching
    draw_value_panel's existing "no agent value data" placeholder for that
    case.

    decide() is defined in terms of this -- its second return value is
    simply the breakdown callers there don't need. Tie-breaking/rng
    semantics for "heuristic" and "random" are copied from
    heuristic.choose_move_with_weights and the original decide()
    respectively, not delegated to them, specifically so the score/move
    actually chosen here is guaranteed to be the SAME one decide() would
    have returned (verified by test_decide_with_breakdown_move_matches_decide)."""
    if kind == "search":
        evaluator, depth = params
        move, move_values, root_value = az_search.search(
            case.game, roll=case.roll, pending_bonus=case.pending_bonus,
            consecutive_sixes=case.consecutive_sixes, depth=depth, evaluator=evaluator,
        )
        decision = {
            "kind": "search", "seat": case.acting_seat,
            "chosen_piece_id": move[0].piece_id if move is not None else None,
            "root_value": [float(v) for v in root_value],
            "move_values": {str(pid): [float(v) for v in vec] for pid, vec in move_values.items()},
        }
        return move, decision

    if kind == "heuristic":
        if not case.legal_moves:
            return None, {"kind": "heuristic", "seat": case.acting_seat, "chosen_piece_id": None}
        player = case.game.get_current_player()
        local_rng = rng or random
        scored = [(heuristic._score_move(case.game, player, move, params), move)
                  for move in case.legal_moves]
        best_score = max(s for s, _m in scored)
        best_moves = [m for s, m in scored if s == best_score]
        move = local_rng.choice(best_moves)
        decision = {
            "kind": "heuristic", "seat": case.acting_seat,
            "chosen_piece_id": move[0].piece_id,
            "move_scores": {str(m[0].piece_id): float(s) for s, m in scored},
        }
        return move, decision

    if kind == "random":
        if not case.legal_moves:
            return None, {"kind": "random", "seat": case.acting_seat, "chosen_piece_id": None}
        move = (rng or random).choice(case.legal_moves)
        return move, {"kind": "random", "seat": case.acting_seat, "chosen_piece_id": move[0].piece_id}

    raise ValueError(f"Unknown agent kind {kind!r} (expected 'search', 'heuristic', or 'random')")


def decide(kind, params, case, rng=None):
    """(kind, params) as returned by agent_spec.parse_spec -> the chosen
    move (piece, new_position, move_type), for exactly the one decision
    `case` (a loader.PuzzleCase) describes."""
    move, _decision = decide_with_breakdown(kind, params, case, rng=rng)
    return move


def decide_from_spec(kind, params, seed=0):
    """Returns a decide_fn(case) -> move closure bound to one (kind,
    params) agent spec, with a private seeded RNG for heuristic tie-breaks
    / random choices -- deterministic across identical runs, matching
    this suite's "fast deterministic regression test" purpose (Part 5.4)."""
    rng = random.Random(seed)
    return lambda case: decide(kind, params, case, rng=rng)


def score_puzzles(decide_fn, puzzles):
    """decide_fn(case) -> move_or_None, called once per puzzle.
    `decide_fn` is a plain callable (not an agent_spec-shaped (kind,
    params) pair) so tests can inject a trivial always-correct/
    always-wrong function with no monkeypatching -- see decide_from_spec
    for the (kind, params) convenience wrapper real callers use.

    Returns:
        {'n_correct', 'n_total', 'accuracy',
         'by_category': {category: {'n_correct', 'n_total', 'accuracy'}},
         'results': [{'puzzle_id', 'category', 'correct',
                      'chosen_piece_id', 'correct_piece_ids', 'rationale'}, ...]}

    A puzzle with more than one accepted answer (loader.PuzzleCase.correct_piece_ids
    has >1 entry) counts as correct iff the chosen piece is ANY one of them.
    """
    n_correct = 0
    by_category = {}
    results = []

    for case in puzzles:
        move = decide_fn(case)
        chosen_piece_id = move[0].piece_id if move is not None else None
        correct = chosen_piece_id in case.correct_piece_ids
        n_correct += int(correct)

        cat = by_category.setdefault(case.category, {"n_correct": 0, "n_total": 0})
        cat["n_total"] += 1
        cat["n_correct"] += int(correct)

        results.append({
            "puzzle_id": case.puzzle_id, "category": case.category, "correct": correct,
            "chosen_piece_id": chosen_piece_id, "correct_piece_ids": list(case.correct_piece_ids),
            "rationale": case.rationale,
        })

    for cat in by_category.values():
        cat["accuracy"] = cat["n_correct"] / cat["n_total"] if cat["n_total"] else 0.0

    n_total = len(puzzles)
    return {
        "n_correct": n_correct, "n_total": n_total,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "by_category": by_category,
        "results": results,
    }


def _print_report(label, result):
    print("=" * 60)
    print("PUZZLE ACCURACY")
    print("=" * 60)
    print(f"  Agent: {label}")
    print(f"  Overall: {result['n_correct']}/{result['n_total']} ({result['accuracy']:.1%})")
    print("-" * 60)
    for category, stats in sorted(result["by_category"].items()):
        print(f"  {category:<25} {stats['n_correct']}/{stats['n_total']} ({stats['accuracy']:.1%})")
    print("=" * 60)


def _save_results_json(result, label, csv_path, json_out):
    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "agent": label, "csv_path": str(csv_path), **result,
        }, f, indent=2)
    print(f"\nResults saved to: {json_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Score an agent's puzzle_accuracy on the tactical puzzle suite "
                     "(docs/AGENT_REBUILD_PLAN.md Part 5.4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--agent", required=True, metavar="SPEC",
                         help="checkpoint:<run_dir>[:depth=N] | heuristic:tuned|default | random")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH),
                         help="Puzzle CSV file or directory of CSVs (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None, metavar="PATH")
    args = parser.parse_args()

    kind, params, label = agent_spec.parse_spec(args.agent)
    puzzles = load_puzzles(args.csv)
    decide_fn = decide_from_spec(kind, params, seed=args.seed)
    result = score_puzzles(decide_fn, puzzles)

    _print_report(label, result)
    if args.json_out:
        _save_results_json(result, label, args.csv, args.json_out)


if __name__ == "__main__":
    main()
