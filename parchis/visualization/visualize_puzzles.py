#!/usr/bin/env python3
"""
Visualize the tactical puzzle suite (docs/AGENT_REBUILD_PLAN.md Part 5.4)
on the real board photo, one puzzle at a time: the position, whose turn it
is, the roll and consecutive_sixes, an agent's own per-move evaluation
(win probability for a search agent, raw score for a heuristic one), the
puzzle's ground-truth correct_piece_id(s) -- a puzzle may have more than
one accepted answer -- and whether the agent's answer actually matches
one of them.

Deliberately reuses parchis.visualization.visualizer.ParchisVisualizer's
existing board-rendering and value-panel machinery (built for live-game
replay -- see visualizer.py's own module docstring) rather than
duplicating any of it: a puzzle is rendered as a single-decision "replay"
of one position, going through the exact same draw_pieces/set_status/
draw_value_panel calls a real game's replay does. The only genuinely new
piece is runner.decide_with_breakdown, which computes the SAME per-move
breakdown a real game's recording factories capture (root_value/
move_values or move_scores), but for one fully-specified puzzle decision
rather than one step of a live TurnContextTracker-driven game -- see that
function's own docstring for why (parchis.evaluation.puzzles.runner's
module docstring already explains this for decide() itself).

Usage:
    python -m parchis.visualization.visualize_puzzles --agent heuristic:tuned
    python -m parchis.visualization.visualize_puzzles \\
        --agent checkpoint:runs/selfplay_2p_v1_champion:depth=1 \\
        --csv parchis/evaluation/puzzles/my_puzzles.csv --puzzle-id p003
    python -m parchis.visualization.visualize_puzzles --agent random \\
        --save-dir runs/puzzle_renders   # headless: one PNG per puzzle, no GUI
"""

import argparse
import random as random_module
from pathlib import Path

import matplotlib.pyplot as plt

from parchis.agents import agent_spec
from parchis.evaluation.puzzles.loader import load_puzzles
from parchis.evaluation.puzzles.runner import DEFAULT_CSV_PATH, decide_with_breakdown
from parchis.visualization.visualizer import ParchisVisualizer

# Two lines of status text (turn/roll/consecutive_sixes, then ground-truth
# vs. the agent's answer) need more room than a live replay's one-liner --
# see ParchisVisualizer.STATUS_STRIP_HEIGHT's own docstring.
PUZZLE_STATUS_STRIP_HEIGHT = 95


def _game_state_from_case(case):
    """{color: [pos0..pos3]} for ParchisVisualizer.draw_pieces, straight
    from a loader.PuzzleCase's own live Game object -- None for a piece
    still in base, else its position (1-76, 76 meaning finished)."""
    return {
        player.color: [None if p.in_base else p.position for p in player.pieces]
        for player in case.game.players
    }


def _roll_description(case):
    """Matches visualizer.replay_game_from_log's own roll/bonus phrasing,
    for a puzzle's already-known (roll, pending_bonus) instead of one read
    off a game log's RollEntry."""
    if case.roll is not None:
        return f"roll={case.roll}"
    bonus_type = case.pending_bonus['type'].replace('_', ' ')
    return f"{bonus_type} (+{case.pending_bonus['squares']} squares)"


def render_puzzle(viz, case, kind, params, rng=None):
    """Draws one puzzle's position + status + value-panel breakdown into
    `viz`'s existing figure/axes (create_board(show_value_panel=True) must
    already have been called on it). Returns a result dict in the same
    shape as one entry of runner.score_puzzles's 'results' list, so the
    CLI's console summary and --save-dir file naming can reuse it."""
    mover = case.game.get_current_player()
    move, decision = decide_with_breakdown(kind, params, case, rng=rng)
    chosen_piece_id = decision.get('chosen_piece_id')
    correct = chosen_piece_id in case.correct_piece_ids
    verdict = "CORRECT" if correct else "INCORRECT"
    correct_label = "/".join(str(pid) for pid in case.correct_piece_ids)

    viz.draw_pieces(_game_state_from_case(case))
    viz.set_status(
        f"Puzzle {case.puzzle_id} [{case.category}] — {mover.color} to move — "
        f"{_roll_description(case)} — consecutive_sixes={case.consecutive_sixes}\n"
        f"Ground truth: piece {correct_label}    Agent chose: piece {chosen_piece_id}    "
        f"→ {verdict}",
        color=viz.COLORS[mover.color],
    )
    player_colors_by_seat = {seat: p.color for seat, p in enumerate(case.game.players)}
    viz.draw_value_panel(decision, player_colors_by_seat, correct_piece_ids=case.correct_piece_ids)

    return {
        "puzzle_id": case.puzzle_id, "category": case.category, "correct": correct,
        "chosen_piece_id": chosen_piece_id, "correct_piece_ids": list(case.correct_piece_ids),
        "rationale": case.rationale,
    }


def visualize_puzzles(puzzles, kind, params, seed=0, save_dir=None, step_by_step=True):
    """Renders every puzzle in `puzzles` in order, either as an interactive
    step-through (default -- one figure, reused across puzzles, press
    ENTER to advance) or, if `save_dir` is given, headlessly to one PNG per
    puzzle (no GUI, no pauses -- useful for reviewing many puzzles at once
    or from a script). Returns the list of per-puzzle result dicts (see
    render_puzzle)."""
    rng = random_module.Random(seed)
    viz = ParchisVisualizer()
    viz.STATUS_STRIP_HEIGHT = PUZZLE_STATUS_STRIP_HEIGHT
    viz.create_board(show_value_panel=True)

    results = []
    for i, case in enumerate(puzzles):
        result = render_puzzle(viz, case, kind, params, rng=rng)
        results.append(result)
        mark = "✓" if result["correct"] else "✗"
        correct_label = "/".join(str(pid) for pid in result["correct_piece_ids"])
        print(f"{mark} [{i + 1}/{len(puzzles)}] {result['puzzle_id']} ({result['category']}): "
              f"agent chose piece {result['chosen_piece_id']}, correct is piece "
              f"{correct_label}")

        plt.pause(0.1)
        if save_dir is not None:
            out_path = Path(save_dir) / f"{result['puzzle_id']}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            viz.save(str(out_path))
        elif step_by_step and i < len(puzzles) - 1:
            user_input = input("Press ENTER for next puzzle (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break

    if save_dir is None:
        viz.show()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the tactical puzzle suite on the real board photo "
                     "(docs/AGENT_REBUILD_PLAN.md Part 5.4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--agent", required=True, metavar="SPEC",
                         help="checkpoint:<run_dir>[:depth=N] | heuristic:tuned|default | random")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH),
                         help="Puzzle CSV file or directory of CSVs (default: %(default)s)")
    parser.add_argument("--puzzle-id", default=None,
                         help="Render only this one puzzle (default: step through every puzzle "
                              "in --csv, in file order)")
    parser.add_argument("--seed", type=int, default=0,
                         help="RNG seed for heuristic tie-breaks / random-agent choices")
    parser.add_argument("--save-dir", default=None, metavar="DIR",
                         help="Save one PNG per puzzle here instead of an interactive step-through")
    args = parser.parse_args()

    kind, params, label = agent_spec.parse_spec(args.agent)
    puzzles = load_puzzles(args.csv)
    if args.puzzle_id is not None:
        puzzles = [p for p in puzzles if p.puzzle_id == args.puzzle_id]
        if not puzzles:
            raise SystemExit(f"No puzzle with puzzle_id={args.puzzle_id!r} in {args.csv}")

    results = visualize_puzzles(puzzles, kind, params, seed=args.seed, save_dir=args.save_dir)

    n_correct = sum(r["correct"] for r in results)
    if results:
        print(f"\n{label}: {n_correct}/{len(results)} puzzles correct ({n_correct / len(results):.1%})")


if __name__ == "__main__":
    main()
