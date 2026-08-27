#!/usr/bin/env python3
"""
Tests for parchis/az/turn_context.py: the shared bonus-vs-fresh-roll /
six-streak tracker used by both parchis/az/agent.py and
parchis/az/selfplay.py.
"""

from parchis.game.board import Board
from parchis.az.turn_context import TurnContextTracker
from parchis.game.game import Game


def test_fresh_roll_reported_correctly():
    print("\nTesting a fresh (non-bonus) roll is reported correctly...")
    tracker = TurnContextTracker()
    roll, pending_bonus, consecutive_sixes = tracker.context_for({"last_roll": 3})
    assert (roll, pending_bonus, consecutive_sixes) == (3, None, 0)
    print("✓ Fresh roll reported with pending_bonus=None, consecutive_sixes=0")


def test_capture_move_sets_pending_bonus_for_next_call():
    """After recording a capturing move -- with mover having a genuinely
    legal +20 move available afterward -- the NEXT context_for() call
    must report pending_bonus (capture_bonus), roll=None, regardless of
    roll_box's own (stale) value."""
    print("\nTesting a capturing move sets pending_bonus for the next call...")

    game = Game(num_players=2)
    mover = game.get_current_player()
    opponent = next(p for p in game.players if p is not mover)
    my_piece = mover.pieces[0]
    victim = opponent.pieces[0]
    target_pos = 20
    game.board.remove_piece(my_piece)
    my_piece.move_to(target_pos - 3)
    game.board.add_piece(my_piece, target_pos - 3)
    game.board.remove_piece(victim)
    victim.move_to(target_pos)
    game.board.add_piece(victim, target_pos)

    move = (my_piece, target_pos, 'move')
    assert game.would_capture(move) == [victim], "Test setup error: expected a capture"

    tracker = TurnContextTracker()
    tracker.context_for({"last_roll": 3})
    # record_move must see the PRE-move game (its own docstring's contract
    # -- it simulates the move itself via snapshot/execute_move/restore),
    # exactly like the real call order in agent.py/selfplay.py: decide the
    # move, record it, THEN return it for Game.play_turn() to actually
    # execute for real.
    tracker.record_move(game, move)
    game.execute_move(my_piece, target_pos, 'move')  # NOW apply it for real, matching Game.play_turn()

    roll, pending_bonus, consecutive_sixes = tracker.context_for({"last_roll": 3})  # stale roll_box
    assert roll is None, "A bonus decision must never report a roll"
    assert pending_bonus == {'type': 'capture_bonus', 'squares': 20}
    print("✓ Capture correctly sets pending_bonus=capture_bonus, roll=None on the next call")


def test_finish_move_sets_pending_bonus_for_next_call():
    """Same shape as the capture test above, but for a finish bonus --
    mover has a SECOND on-board piece able to use the +10, so the bonus
    is genuinely available (contrast test_finish_bonus_with_no_legal_
    move_is_never_recorded_as_pending below, where it deliberately is
    not)."""
    print("\nTesting a finishing move sets pending_bonus for the next call...")

    game = Game(num_players=2)
    mover = game.get_current_player()
    finishing_piece = mover.pieces[0]
    other_piece = mover.pieces[1]
    game.board.remove_piece(finishing_piece)
    finishing_piece.move_to(74)
    game.board.add_piece(finishing_piece, 74)
    game.board.remove_piece(other_piece)
    other_piece.move_to(5)
    game.board.add_piece(other_piece, 5)

    move = (finishing_piece, Board.FINAL_POSITION, 'finish')

    tracker = TurnContextTracker()
    tracker.context_for({"last_roll": 2})
    tracker.record_move(game, move)  # simulates internally, on the pre-move game
    game.execute_move(finishing_piece, Board.FINAL_POSITION, 'finish')  # apply for real

    roll, pending_bonus, _ = tracker.context_for({"last_roll": 2})
    assert roll is None
    assert pending_bonus == {'type': 'finish_bonus', 'squares': 10}
    print("✓ Finishing correctly sets pending_bonus=finish_bonus, roll=None on the next call")


def test_finish_bonus_with_no_legal_move_is_never_recorded_as_pending():
    """Regression test for the bug documented in turn_context.py's module
    docstring (2nd attempt): a bonus that will genuinely have ZERO legal
    moves (every other piece still in base -- can't enter on a bonus
    value of 10, only a real roll of 5/6) must NEVER be recorded as
    pending in the first place -- record_move() simulates the move itself
    and predicts this, exactly mirroring Game._execute_bonus_move's own
    "zero legal moves -> silently skip, never call choose_move" rule."""
    print("\nTesting a finish bonus with no legal move is never recorded as pending...")

    game = Game(num_players=2)
    mover = game.get_current_player()
    finishing_piece = mover.pieces[0]
    game.board.remove_piece(finishing_piece)
    finishing_piece.move_to(74)
    game.board.add_piece(finishing_piece, 74)

    move = (finishing_piece, Board.FINAL_POSITION, 'finish')

    tracker = TurnContextTracker()
    tracker.context_for({"last_roll": 2})
    tracker.record_move(game, move)

    assert tracker._pending_bonus is None, (
        f"Expected no pending bonus to be recorded (zero legal +10 moves), "
        f"got {tracker._pending_bonus}"
    )
    # The move itself must be unaffected by record_move's internal
    # simulate/restore -- still un-executed, exactly as record_move's own
    # contract promises (the REAL caller executes it afterward).
    assert finishing_piece.position == 74 and not finishing_piece.finished

    # And the board must show what record_move's simulation itself relied
    # on: after really executing the move, zero legal +10 moves exist.
    game.execute_move(finishing_piece, Board.FINAL_POSITION, 'finish')
    assert game.get_legal_moves(mover, 10) == [], (
        "Test setup error: expected the finish bonus to have NO legal move"
    )
    print("✓ record_move never sets a pending bonus that would have zero legal moves")


def test_plain_move_clears_pending_bonus():
    """A non-capturing, non-finishing move must clear any prior bonus
    state, so the call after it is treated as wanting a fresh roll again."""
    print("\nTesting a plain move clears pending_bonus...")

    game = Game(num_players=2)
    mover = game.get_current_player()
    piece = mover.pieces[0]
    move = (piece, piece.position + 3, 'move')
    game.board.remove_piece(piece)
    piece.move_to(piece.position)  # no-op, just ensure board consistency
    game.board.add_piece(piece, piece.position)

    tracker = TurnContextTracker()
    tracker.context_for({"last_roll": 3})
    tracker.record_move(game, move)  # not a capture, not a finish

    roll, pending_bonus, _ = tracker.context_for({"last_roll": 5})
    assert pending_bonus is None
    assert roll == 5
    print("✓ A plain move clears pending_bonus; the next call reports a fresh roll")


def test_six_streak_increments_only_across_fresh_sixes():
    """consecutive_sixes counts the streak INCLUDING the current roll
    (matching Game.play_turn()'s own convention, and search.py's
    consecutive_sixes contract: incremented before the three-sixes check),
    and must increment only across FRESH (non-bonus) rolls of 6, capped at
    2 (a would-be third six is a Game-level penalty, never actually
    offered as a decision -- the cap is just defensive)."""
    print("\nTesting six-streak increments only across fresh sixes...")

    tracker = TurnContextTracker()
    _, _, streak1 = tracker.context_for({"last_roll": 6})
    assert streak1 == 1, "The first roll of 6 already counts as streak 1"
    _, _, streak2 = tracker.context_for({"last_roll": 6})
    assert streak2 == 2
    _, _, streak3 = tracker.context_for({"last_roll": 6})
    assert streak3 == 2, "Capped at 2 -- a real third six never reaches a decision at all"

    tracker2 = TurnContextTracker()
    tracker2.context_for({"last_roll": 6})
    _, _, reset_streak = tracker2.context_for({"last_roll": 3})
    assert reset_streak == 0, "A non-six roll must reset the streak"
    print("✓ Six-streak increments correctly across fresh sixes and resets on a non-six")


def test_no_move_clears_pending_bonus():
    """record_move(game, None) (nothing was legal to choose) must clear
    any pending bonus, matching Game.play_turn()'s own 'no piece moved'
    handling. Does not touch `game` at all -- move=None short-circuits
    record_move before `game` is ever used."""
    print("\nTesting record_move(None) clears pending_bonus...")

    tracker = TurnContextTracker()
    tracker._pending_bonus = {'type': 'capture_bonus', 'squares': 20}  # force prior state
    tracker.record_move(None, None)
    assert tracker._pending_bonus is None
    print("✓ record_move(None) clears any pending bonus")


if __name__ == '__main__':
    test_fresh_roll_reported_correctly()
    test_capture_move_sets_pending_bonus_for_next_call()
    test_finish_move_sets_pending_bonus_for_next_call()
    test_finish_bonus_with_no_legal_move_is_never_recorded_as_pending()
    test_plain_move_clears_pending_bonus()
    test_six_streak_increments_only_across_fresh_sixes()
    test_no_move_clears_pending_bonus()
    print("\nAll turn_context tests passed!")
