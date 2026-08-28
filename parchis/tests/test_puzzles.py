#!/usr/bin/env python3
"""
Tests for parchis/evaluation/puzzles/ (docs/AGENT_REBUILD_PLAN.md Part
5.4): the tactical puzzle suite's CSV loader and scoring runner.

No fixture files are committed -- every test builds its own tiny CSV
inline via tmp_path, matching this repo's existing convention
(parchis/tests/test_ladder.py does the same for its JSONL).
"""

import pytest

from parchis.evaluation.puzzles import loader, runner
from parchis.game.board import Board


HEADER = ("puzzle_id,category,a_piece_0,a_piece_1,a_piece_2,a_piece_3,"
          "b_piece_0,b_piece_1,b_piece_2,b_piece_3,turn,roll,"
          "consecutive_sixes,correct_piece_id,rationale")


def _row(puzzle_id="p1", category="cat", a=(20, 50, 0, 0), b=(24, 0, 0, 0),
         turn="A", roll="4", consecutive_sixes="0", correct_piece_id="0",
         rationale="because"):
    """One well-formed CSV row string, matching the verified capture_priority
    example (RED piece_0@20 can capture YELLOW piece_0@24 with roll=4;
    RED piece_1@50 is a plain alternative) -- defaults are a KNOWN GOOD
    puzzle, so tests only need to override the field(s) they're probing."""
    return (f"{puzzle_id},{category},{a[0]},{a[1]},{a[2]},{a[3]},"
            f"{b[0]},{b[1]},{b[2]},{b[3]},{turn},{roll},"
            f"{consecutive_sixes},{correct_piece_id},{rationale}")


def _write_csv(path, rows):
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n")


def test_load_puzzle_row_capture_example():
    print("\nTesting a well-formed capture-priority row loads correctly...")
    case = loader.load_puzzle_row(dict(zip(
        HEADER.split(","), _row().split(","),
    )))
    assert case.puzzle_id == "p1"
    assert case.category == "cat"
    assert case.roll == 4 and case.pending_bonus is None
    assert case.consecutive_sixes == 0
    assert case.correct_piece_id == 0
    assert case.game.get_current_player().color == "RED"
    legal_ids = sorted({m[0].piece_id for m in case.legal_moves})
    assert legal_ids == [0, 1], f"Expected exactly pieces 0 and 1 legal, got {legal_ids}"
    print(f"✓ loaded puzzle {case.puzzle_id}, acting_seat={case.acting_seat}, "
          f"legal piece_ids={legal_ids}")


def test_load_puzzles_bonus_row(tmp_path):
    print("\nTesting a bonus-decision row (roll=capture_bonus) loads correctly...")
    csv_path = tmp_path / "puzzles.csv"
    _write_csv(csv_path, [_row(
        puzzle_id="p_bonus", category="bonus_allocation",
        a=(10, 43, 0, 0), b=(0, 0, 0, 0), roll="capture_bonus", correct_piece_id="1",
    )])
    cases = loader.load_puzzles(str(csv_path))
    assert len(cases) == 1
    case = cases[0]
    assert case.roll is None
    assert case.pending_bonus == {"type": "capture_bonus", "squares": 20}
    print(f"✓ pending_bonus={case.pending_bonus}, legal piece_ids="
          f"{sorted({m[0].piece_id for m in case.legal_moves})}")


def test_load_puzzles_finish_bonus_row(tmp_path):
    print("\nTesting a finish_bonus row resolves to the right squares...")
    csv_path = tmp_path / "puzzles.csv"
    _write_csv(csv_path, [_row(
        puzzle_id="p_finish_bonus", a=(10, 43, 0, 0), b=(0, 0, 0, 0),
        roll="finish_bonus", correct_piece_id="1",
    )])
    case = loader.load_puzzles(str(csv_path))[0]
    assert case.pending_bonus == {"type": "finish_bonus", "squares": 10}
    print(f"✓ pending_bonus={case.pending_bonus}")


@pytest.mark.parametrize("csv_value,expect_position,expect_in_base,expect_finished", [
    (0, None, True, False),
    (20, 20, False, False),
    (Board.FINAL_POSITION, Board.FINAL_POSITION, False, True),
])
def test_place_piece_transitions(csv_value, expect_position, expect_in_base, expect_finished):
    print(f"\nTesting _place_piece with csv_value={csv_value}...")
    from parchis.game.game import Game
    game = Game(num_players=2)
    piece = game.players[0].pieces[1]  # starts in base
    loader._place_piece(piece, csv_value, game.board)
    assert piece.position == expect_position
    assert piece.in_base is expect_in_base
    assert piece.finished is expect_finished
    if csv_value != 0:
        assert piece in game.board.get_pieces_at(csv_value)
    print(f"✓ position={piece.position} in_base={piece.in_base} finished={piece.finished}")


def test_place_piece_from_finished_back_to_board():
    """The three-field consistency gotcha this helper exists for: a piece
    that WAS finished, relocated onto the main track, must have finished
    cleared (mark_finished/move_to alone wouldn't do this)."""
    print("\nTesting _place_piece clears a stale finished=True when relocating...")
    from parchis.game.game import Game
    game = Game(num_players=2)
    piece = game.players[0].pieces[1]
    loader._place_piece(piece, Board.FINAL_POSITION, game.board)
    assert piece.finished is True
    loader._place_piece(piece, 30, game.board)
    assert piece.position == 30
    assert piece.finished is False, "finished must be cleared when relocating off the final square"
    assert piece.in_base is False
    print("✓ finished=True correctly cleared after relocating a finished piece back onto the track")


def test_consecutive_sixes_must_be_zero_unless_roll_is_six():
    print("\nTesting consecutive_sixes=1 with a non-six roll is rejected...")
    row = dict(zip(HEADER.split(","), _row(roll="3", consecutive_sixes="1").split(",")))
    with pytest.raises(ValueError, match="consecutive_sixes"):
        loader.load_puzzle_row(row)
    print("✓ raises ValueError")


def test_consecutive_sixes_allowed_with_roll_six():
    print("\nTesting consecutive_sixes=2 with roll=6 is accepted...")
    row = dict(zip(
        HEADER.split(","),
        _row(a=(20, 50, 0, 0), b=(0, 0, 0, 0), roll="6", consecutive_sixes="2",
             correct_piece_id="0").split(","),
    ))
    case = loader.load_puzzle_row(row)
    assert case.roll == 6 and case.consecutive_sixes == 2
    print("✓ accepted")


def test_incorrect_piece_id_not_legal_raises():
    print("\nTesting a correct_piece_id with no legal move for it is rejected...")
    # Piece 2 is in base and roll=4 (not the entry roll), so it has no legal move at all.
    row = dict(zip(HEADER.split(","), _row(correct_piece_id="2").split(",")))
    with pytest.raises(ValueError, match="not among the actual legal moves"):
        loader.load_puzzle_row(row)
    print("✓ raises ValueError")


def test_turn_must_be_a_or_b():
    print("\nTesting an invalid turn value is rejected...")
    row = dict(zip(HEADER.split(","), _row(turn="C").split(",")))
    with pytest.raises(ValueError, match="turn"):
        loader.load_puzzle_row(row)
    print("✓ raises ValueError")


def test_mandatory_entry_shadows_on_board_moves():
    """Regression/documentation test for a real engine rule discovered
    while designing this schema: rolling a 5 (ENTRY_ROLL) with an
    enterable piece in base makes entry MANDATORY -- rules.py returns
    ONLY entry moves, never considering on-board moves at all. A puzzle
    author picking roll=5 with a base piece present must expect this."""
    print("\nTesting mandatory entry on roll=5 shadows on-board moves entirely...")
    # piece_0 on board (would have a normal 5-square move available if
    # entry weren't mandatory), piece_1 alone in base, pieces 2/3 already
    # finished (so get_pieces_in_base() sees exactly one base piece).
    row = dict(zip(
        HEADER.split(","),
        _row(a=(20, 0, 76, 76), b=(0, 0, 0, 0), roll="5", correct_piece_id="1").split(","),
    ))
    case = loader.load_puzzle_row(row)
    legal_ids = sorted({m[0].piece_id for m in case.legal_moves})
    assert legal_ids == [1], (
        f"Expected ONLY piece 1's entry move to be legal (mandatory entry), got {legal_ids}"
    )
    assert all(m[2] == "enter" for m in case.legal_moves)
    print(f"✓ only entry moves are legal on roll=5 when entry is possible: {legal_ids}")


def test_load_puzzles_directory_and_global_uniqueness(tmp_path):
    print("\nTesting load_puzzles reads every CSV in a directory, checking global uniqueness...")
    _write_csv(tmp_path / "a.csv", [_row(puzzle_id="p1")])
    _write_csv(tmp_path / "b.csv", [_row(puzzle_id="p2", category="other")])
    cases = loader.load_puzzles(str(tmp_path))
    assert {c.puzzle_id for c in cases} == {"p1", "p2"}
    print(f"✓ loaded {len(cases)} puzzles from 2 files in a directory")

    _write_csv(tmp_path / "c.csv", [_row(puzzle_id="p1")])  # duplicate of a.csv's p1
    with pytest.raises(ValueError, match="duplicate puzzle_id"):
        loader.load_puzzles(str(tmp_path))
    print("✓ a duplicate puzzle_id across files raises ValueError")


def test_load_puzzles_empty_raises(tmp_path):
    print("\nTesting load_puzzles on an empty puzzle set raises...")
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(HEADER + "\n")
    with pytest.raises(ValueError, match="no puzzles found"):
        loader.load_puzzles(str(csv_path))
    print("✓ raises ValueError")


# --- runner.score_puzzles ---

def _fixed_decide(piece_id):
    """A decide_fn that always picks `piece_id`'s move, if legal for that
    puzzle -- else the first legal move. No monkeypatching needed since
    score_puzzles accepts a plain callable."""
    def decide_fn(case):
        for move in case.legal_moves:
            if move[0].piece_id == piece_id:
                return move
        return case.legal_moves[0] if case.legal_moves else None
    return decide_fn


def test_score_puzzles_all_correct(tmp_path):
    print("\nTesting score_puzzles reports 100% when every answer matches...")
    csv_path = tmp_path / "puzzles.csv"
    _write_csv(csv_path, [_row(puzzle_id="p1", correct_piece_id="0")])
    puzzles = loader.load_puzzles(str(csv_path))
    result = runner.score_puzzles(_fixed_decide(0), puzzles)
    assert result == {
        "n_correct": 1, "n_total": 1, "accuracy": 1.0,
        "by_category": {"cat": {"n_correct": 1, "n_total": 1, "accuracy": 1.0}},
        "results": [{
            "puzzle_id": "p1", "category": "cat", "correct": True,
            "chosen_piece_id": 0, "correct_piece_id": 0, "rationale": "because",
        }],
    }
    print(f"✓ {result['accuracy']:.0%} accuracy")


def test_score_puzzles_all_wrong(tmp_path):
    print("\nTesting score_puzzles reports 0% when every answer is wrong...")
    csv_path = tmp_path / "puzzles.csv"
    _write_csv(csv_path, [_row(puzzle_id="p1", correct_piece_id="0")])
    puzzles = loader.load_puzzles(str(csv_path))
    result = runner.score_puzzles(_fixed_decide(1), puzzles)
    assert result["n_correct"] == 0
    assert result["accuracy"] == 0.0
    assert result["results"][0]["correct"] is False
    assert result["results"][0]["chosen_piece_id"] == 1
    print("✓ 0% accuracy, chosen_piece_id recorded correctly")


def test_score_puzzles_per_category_breakdown(tmp_path):
    print("\nTesting score_puzzles' by_category breakdown with a mixed result...")
    csv_path = tmp_path / "puzzles.csv"
    _write_csv(csv_path, [
        _row(puzzle_id="p1", category="cat_a", correct_piece_id="0"),
        _row(puzzle_id="p2", category="cat_b", correct_piece_id="0"),
    ])
    puzzles = loader.load_puzzles(str(csv_path))
    # Always answer piece 0 -- correct for both here, so instead force a
    # split by making cat_b's puzzle expect piece 1 (illegal in this
    # fixture's board -- so use a fixed decide that answers 1 for cat_b
    # only) -- simplest: decide by category.
    def decide_fn(case):
        target = 0 if case.category == "cat_a" else 1
        for move in case.legal_moves:
            if move[0].piece_id == target:
                return move
        return case.legal_moves[0]

    result = runner.score_puzzles(decide_fn, puzzles)
    assert result["by_category"]["cat_a"]["accuracy"] == 1.0
    assert result["by_category"]["cat_b"]["accuracy"] == 0.0
    assert result["accuracy"] == 0.5
    print(f"✓ by_category={result['by_category']}")


def test_score_puzzles_no_legal_move_counts_as_incorrect():
    print("\nTesting a None decision (no legal move) counts as incorrect, not a crash...")
    row = dict(zip(HEADER.split(","), _row().split(",")))
    case = loader.load_puzzle_row(row)
    result = runner.score_puzzles(lambda c: None, [case])
    assert result["n_correct"] == 0
    assert result["results"][0]["chosen_piece_id"] is None
    print("✓ handled gracefully, chosen_piece_id=None")


def test_decide_search_heuristic_random_dispatch(tmp_path):
    """decide()'s three branches, exercised for real (no mocking) against
    the known-good capture-priority fixture."""
    print("\nTesting decide() dispatches correctly for heuristic/random/search kinds...")
    from parchis.agents import heuristic
    from parchis.az.agent import heuristic_position_evaluator

    row = dict(zip(HEADER.split(","), _row().split(",")))
    case = loader.load_puzzle_row(row)

    heuristic_move = runner.decide("heuristic", heuristic.TUNED_WEIGHTS, case)
    assert heuristic_move[0].piece_id in (0, 1)

    import random
    random_move = runner.decide("random", None, case, rng=random.Random(0))
    assert random_move[0].piece_id in (0, 1)

    search_move = runner.decide("search", (heuristic_position_evaluator, 1), case)
    assert search_move[0].piece_id in (0, 1)

    with pytest.raises(ValueError):
        runner.decide("not_a_real_kind", None, case)
    print(f"✓ heuristic={heuristic_move[0].piece_id} random={random_move[0].piece_id} "
          f"search={search_move[0].piece_id}, unknown kind raises")


def test_cli_smoke_test(tmp_path, monkeypatch, capsys):
    print("\nTesting the CLI runs end to end and (optionally) writes JSON...")
    import sys
    import json as json_module

    csv_path = tmp_path / "puzzles.csv"
    _write_csv(csv_path, [_row(puzzle_id="p1", correct_piece_id="0")])
    json_out = tmp_path / "results.json"

    monkeypatch.setattr(sys, "argv", [
        "puzzles", "--agent", "random", "--csv", str(csv_path),
        "--seed", "0", "--json-out", str(json_out),
    ])
    runner.main()

    captured = capsys.readouterr()
    assert "PUZZLE ACCURACY" in captured.out
    assert json_out.exists()
    data = json_module.loads(json_out.read_text())
    assert data["agent"] == "random"
    assert data["n_total"] == 1
    print(f"✓ CLI ran cleanly, JSON output well-formed: {data['n_correct']}/{data['n_total']}")


if __name__ == '__main__':
    print("Most tests in this file need tmp_path/monkeypatch/capsys -- run via pytest.")
