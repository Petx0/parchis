#!/usr/bin/env python3
"""
Tests for parchis/visualization/visualize_puzzles.py: rendering the
tactical puzzle suite (docs/AGENT_REBUILD_PLAN.md Part 5.4) onto the real
board, with an agent's per-move evaluation and the ground-truth answer.
"""

import matplotlib
matplotlib.use('Agg')  # headless, no windows during tests

from pathlib import Path

from parchis.agents import heuristic
from parchis.evaluation.puzzles.loader import load_puzzle_row
from parchis.visualization.visualize_puzzles import (
    _game_state_from_case, _roll_description, render_puzzle, visualize_puzzles,
)
from parchis.visualization.visualizer import ParchisVisualizer

TACTICAL_PUZZLES_CSV = (
    Path(__file__).resolve().parent.parent / "evaluation" / "puzzles" / "tactical_puzzles.csv"
)

HEADER = ("puzzle_id,category,a_piece_0,a_piece_1,a_piece_2,a_piece_3,"
          "b_piece_0,b_piece_1,b_piece_2,b_piece_3,turn,roll,"
          "consecutive_sixes,correct_piece_id,rationale")


def _row(puzzle_id="p1", category="cat", a=(20, 50, 0, 0), b=(24, 0, 0, 0),
         turn="A", roll="4", consecutive_sixes="0", correct_piece_id="0",
         rationale="because"):
    return (f"{puzzle_id},{category},{a[0]},{a[1]},{a[2]},{a[3]},"
            f"{b[0]},{b[1]},{b[2]},{b[3]},{turn},{roll},"
            f"{consecutive_sixes},{correct_piece_id},{rationale}")


def _case(**kwargs):
    row = dict(zip(HEADER.split(","), _row(**kwargs).split(",")))
    return load_puzzle_row(row)


def test_game_state_from_case_matches_csv_positions():
    print("\nTesting _game_state_from_case reflects the puzzle's CSV positions...")
    case = _case(a=(20, 0, 76, 0), b=(24, 5, 0, 0))
    state = _game_state_from_case(case)
    assert state["RED"] == [20, None, 76, None]
    assert state["YELLOW"] == [24, 5, None, None]
    assert set(state) == {"RED", "YELLOW"}
    print(f"✓ game_state={state}")


def test_roll_description_plain_roll_and_bonus():
    print("\nTesting _roll_description for a plain roll and a bonus...")
    plain_case = _case(roll="4")
    assert _roll_description(plain_case) == "roll=4"

    bonus_case = _case(a=(10, 43, 0, 0), b=(0, 0, 0, 0), roll="capture_bonus", correct_piece_id="1")
    assert _roll_description(bonus_case) == "capture bonus (+20 squares)"
    print("✓ plain roll and bonus both describe correctly")


def _render(case, kind="heuristic", params=None):
    params = heuristic.TUNED_WEIGHTS if params is None else params
    viz = ParchisVisualizer()
    viz.create_board(show_value_panel=True)
    result = render_puzzle(viz, case, kind, params)
    return viz, result


def test_render_puzzle_correct_answer_marks_agent_and_correct_together():
    print("\nTesting render_puzzle on a puzzle the agent answers correctly...")
    case = _case()  # known-good capture-priority fixture, correct_piece_id=0
    viz, result = _render(case)
    assert result["correct"] is True
    assert result["chosen_piece_id"] == 0
    assert result["correct_piece_ids"] == [0]
    assert len(viz.value_ax_moves.patches) > 0, "Expected move-score bars to be drawn"
    assert viz.status_artist is not None
    assert "CORRECT" in viz.status_artist.get_text()
    print(f"✓ result={result}")


def test_render_puzzle_wrong_answer_marks_agent_and_correct_separately():
    print("\nTesting render_puzzle on a puzzle a weak/adversarial agent answers wrong...")
    case = _case()  # correct_piece_id=0 (capturing), legal moves are pieces 0 and 1

    # Force a wrong answer directly rather than hunting for an agent/seed
    # that happens to get this fixture wrong -- "random" always has SOME
    # chance of being right, and this fixture was deliberately built so
    # the tuned heuristic gets it right (see test_puzzles.py's docstring
    # for _row's defaults). Monkeypatch-free: just call decide_with_breakdown
    # via a params/kind combo forced to prefer piece 1 -- simplest is to
    # invert the weights so the (deliberately non-tactical) score prefers
    # the plain move over the capture.
    import numpy as np
    from parchis.agents.heuristic import NUM_FEATURES
    inverted_weights = np.zeros(NUM_FEATURES)
    inverted_weights[0] = -1.0  # capture_value: NEGATIVE -- prefer NOT capturing
    inverted_weights[2] = 1.0   # progress_gained: reward the plain move instead

    viz, result = _render(case, kind="heuristic", params=inverted_weights)
    assert result["chosen_piece_id"] == 1
    assert result["correct"] is False
    assert "INCORRECT" in viz.status_artist.get_text()

    bars = viz.value_ax_moves.patches
    assert len(bars) == 2
    # One bar (piece 1, chosen) must be red-edged; one (piece 0, correct)
    # must be green-edged -- see visualizer.ParchisVisualizer.CORRECT_MARKER_COLOR.
    edge_colors = sorted(bar.get_edgecolor() for bar in bars)
    from matplotlib.colors import to_rgba
    assert to_rgba('red') in edge_colors
    assert to_rgba(ParchisVisualizer.CORRECT_MARKER_COLOR) in edge_colors
    print(f"✓ result={result}, distinct red (agent) / green (correct) bar edges confirmed")


def test_render_puzzle_multi_answer_marks_every_correct_piece():
    """A puzzle with correct_piece_id='0/1' (loader.py's multi-answer
    support) must mark BOTH pieces 0 and 1 as correct in the value panel,
    regardless of which one the agent actually chose."""
    print("\nTesting render_puzzle marks every accepted answer, not just one...")
    case = _case(a=(20, 50, 0, 0), b=(24, 0, 0, 0), roll="4", correct_piece_id="0/1")
    assert case.correct_piece_ids == (0, 1)

    viz, result = _render(case)  # tuned heuristic chooses piece 0 here
    assert result["correct"] is True
    assert result["correct_piece_ids"] == [0, 1]

    from matplotlib.colors import to_rgba
    bars = viz.value_ax_moves.patches
    assert len(bars) == 2
    edge_colors = [bar.get_edgecolor() for bar in bars]
    # BOTH bars are accepted answers -- neither should carry the plain
    # 'red' (agent-only-wrong) marker; both should carry the correct-answer
    # color (piece 0 solid, since it's also the agent's chosen pick; piece
    # 1 dashed, since it's correct but not what was played).
    assert to_rgba('red') not in edge_colors
    assert all(c == to_rgba(ParchisVisualizer.CORRECT_MARKER_COLOR) for c in edge_colors)
    print(f"✓ result={result}, both accepted answers marked, no spurious 'agent was wrong' red")


def test_visualize_puzzles_end_to_end_save_dir_smoke_test(tmp_path, monkeypatch):
    """The definitive 'no human needs to watch it' check, mirroring
    test_visualizer.py's own end-to-end smoke test: render every puzzle in
    the real, committed tactical_puzzles.csv fixture, headlessly, start to
    finish, no exceptions -- one PNG per puzzle, no interactive input."""
    print("\nTesting visualize_puzzles end-to-end against tactical_puzzles.csv...")
    from parchis.evaluation.puzzles.loader import load_puzzles
    from parchis.visualization import visualize_puzzles as viz_module
    monkeypatch.setattr(viz_module.plt, "pause", lambda *_a, **_k: None)

    puzzles = load_puzzles(str(TACTICAL_PUZZLES_CSV))
    results = visualize_puzzles(
        puzzles, "heuristic", heuristic.TUNED_WEIGHTS, save_dir=str(tmp_path),
    )

    assert len(results) == len(puzzles)
    for case in puzzles:
        png_path = tmp_path / f"{case.puzzle_id}.png"
        assert png_path.exists(), f"Expected {png_path} to be saved"
        assert png_path.stat().st_size > 0
    print(f"✓ rendered {len(results)} puzzles to {tmp_path}, all PNGs written")


def test_cli_smoke_test(tmp_path, monkeypatch, capsys):
    print("\nTesting the visualize_puzzles CLI runs end to end...")
    import sys

    from parchis.visualization import visualize_puzzles as viz_module
    monkeypatch.setattr(viz_module.plt, "pause", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "argv", [
        "visualize_puzzles", "--agent", "heuristic:tuned",
        "--csv", str(TACTICAL_PUZZLES_CSV), "--save-dir", str(tmp_path),
    ])
    viz_module.main()

    captured = capsys.readouterr()
    assert "puzzles correct" in captured.out
    assert (tmp_path / "p001.png").exists()
    assert (tmp_path / "p002.png").exists()
    print("✓ CLI ran cleanly, saved both puzzles' PNGs")


def test_cli_puzzle_id_filter(tmp_path, monkeypatch, capsys):
    print("\nTesting --puzzle-id renders only the requested puzzle...")
    import sys

    from parchis.visualization import visualize_puzzles as viz_module
    monkeypatch.setattr(viz_module.plt, "pause", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "argv", [
        "visualize_puzzles", "--agent", "random", "--csv", str(TACTICAL_PUZZLES_CSV),
        "--puzzle-id", "p002", "--save-dir", str(tmp_path),
    ])
    viz_module.main()

    assert not (tmp_path / "p001.png").exists()
    assert (tmp_path / "p002.png").exists()
    print("✓ only p002 was rendered")


if __name__ == '__main__':
    print("Most tests in this file need tmp_path/monkeypatch/capsys -- run via pytest.")
