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


def decide(kind, params, case, rng=None):
    """(kind, params) as returned by agent_spec.parse_spec -> the chosen
    move (piece, new_position, move_type), for exactly the one decision
    `case` (a loader.PuzzleCase) describes."""
    if kind == "search":
        evaluator, depth = params
        move, _move_values, _root_value = az_search.search(
            case.game, roll=case.roll, pending_bonus=case.pending_bonus,
            consecutive_sixes=case.consecutive_sixes, depth=depth, evaluator=evaluator,
        )
        return move
    if kind == "heuristic":
        player = case.game.get_current_player()
        return heuristic.choose_move_with_weights(
            case.game, player, case.legal_moves, params, rng=rng,
        )
    if kind == "random":
        if not case.legal_moves:
            return None
        return (rng or random).choice(case.legal_moves)
    raise ValueError(f"Unknown agent kind {kind!r} (expected 'search', 'heuristic', or 'random')")


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
                      'chosen_piece_id', 'correct_piece_id', 'rationale'}, ...]}
    """
    n_correct = 0
    by_category = {}
    results = []

    for case in puzzles:
        move = decide_fn(case)
        chosen_piece_id = move[0].piece_id if move is not None else None
        correct = chosen_piece_id == case.correct_piece_id
        n_correct += int(correct)

        cat = by_category.setdefault(case.category, {"n_correct": 0, "n_total": 0})
        cat["n_total"] += 1
        cat["n_correct"] += int(correct)

        results.append({
            "puzzle_id": case.puzzle_id, "category": case.category, "correct": correct,
            "chosen_piece_id": chosen_piece_id, "correct_piece_id": case.correct_piece_id,
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
