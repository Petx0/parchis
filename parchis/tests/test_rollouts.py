#!/usr/bin/env python3
"""
Tests for parchis/az/rollouts.py (.claude/plans/twinkly-marinating-hinton.md
Phase 2.2): rollout-refined value estimation for a fixed decision point.
"""

import random

import numpy as np

from parchis.az import rollouts
from parchis.game.board import Board
from parchis.game.game import Game


def _place(game, player_idx, piece_idx, position):
    """Move one piece to an exact absolute board position, mirroring
    test_search.py's own remove/move_to/add_piece pattern."""
    piece = game.players[player_idx].pieces[piece_idx]
    game.board.remove_piece(piece)
    if position == Board.FINAL_POSITION:
        piece.finished = True
        piece.in_base = False
        piece.position = Board.FINAL_POSITION
        game.board.add_piece(piece, Board.FINAL_POSITION)
    else:
        piece.finished = False
        piece.move_to(position)
        game.board.add_piece(piece, position)


def _send_all_to_base(game, player_idx):
    for piece in game.players[player_idx].pieces:
        game.board.remove_piece(piece)
        piece.send_to_base()


def _build_near_certain_win_game():
    """Seat 0: 3 pieces already finished, 1 piece two squares from
    finishing. Seat 1: every piece still in base. An overwhelming lead
    even for an imperfect (heuristic) rollout policy on both sides."""
    game = Game(num_players=2)
    for piece_idx in range(3):
        _place(game, 0, piece_idx, Board.FINAL_POSITION)
    _place(game, 0, 3, Board.FINAL_POSITION - 2)
    _send_all_to_base(game, 1)
    game.current_player_idx = 0
    return game


def test_estimate_rollout_value_converges_on_near_certain_win():
    print("\nTesting estimate_rollout_value on an overwhelming (near-certain-win) position...")
    game = _build_near_certain_win_game()
    snapshot = game.snapshot()
    rng = random.Random(0)

    value = rollouts.estimate_rollout_value(game, snapshot, mover_seat=0, n_rollouts=40, rng=rng)

    assert value.shape == (2,)
    assert abs(float(value.sum()) - 1.0) < 1e-5, "A mean of one-hot/uniform outcome vectors must sum to 1"
    assert value[0] > 0.85, (
        f"Expected the near-certain winner's own win probability to dominate, got {value}"
    )
    print(f"✓ value={value} (mover's own win probability strongly favored, as expected)")


def test_estimate_rollout_value_never_mutates_the_real_game():
    print("\nTesting estimate_rollout_value leaves the real game and its choose_move hooks untouched...")
    game = _build_near_certain_win_game()
    snapshot_before = game.snapshot()

    sentinel_a = lambda legal_moves: legal_moves[0]
    sentinel_b = lambda legal_moves: legal_moves[0]
    game.players[0].choose_move = sentinel_a
    game.players[1].choose_move = sentinel_b

    rng = random.Random(1)
    rollouts.estimate_rollout_value(game, snapshot_before, mover_seat=0, n_rollouts=10, rng=rng)

    snapshot_after = game.snapshot()
    assert snapshot_after == snapshot_before, "Game state must be restored exactly after rollouts"
    assert game.players[0].choose_move is sentinel_a, (
        "choose_move must be restored -- Game.snapshot()/restore() doesn't cover it, so a rollout "
        "that leaves it overwritten would silently swap the REAL game's remaining turns to the "
        "heuristic for every seat, not just this one rollout"
    )
    assert game.players[1].choose_move is sentinel_b
    print("✓ board state and both seats' choose_move hooks are restored exactly")


def test_estimate_rollout_value_does_not_perturb_the_global_random_state():
    print("\nTesting estimate_rollout_value leaves Python's global random state untouched...")
    game = _build_near_certain_win_game()
    snapshot = game.snapshot()

    random.seed(12345)
    expected_next_values = [random.random() for _ in range(5)]

    random.seed(12345)
    rollouts.estimate_rollout_value(game, snapshot, mover_seat=0, n_rollouts=15, rng=random.Random(7))
    actual_next_values = [random.random() for _ in range(5)]

    assert actual_next_values == expected_next_values, (
        "estimate_rollout_value must not leak into Python's global random state -- "
        "Game.dice.roll() reads from it directly, so any leak would silently corrupt "
        "the REAL game's subsequent dice sequence once control returns to it (see "
        "parchis.search.isolated_random, the same fix parchis/search/mcts.py already "
        "needed for its own simulated rollouts)"
    )
    print("✓ global random state is bit-for-bit unaffected by the rollout call")


def test_estimate_rollout_value_is_reproducible_given_the_same_rng_state():
    print("\nTesting estimate_rollout_value is reproducible from the same rng seed...")
    game = _build_near_certain_win_game()
    snapshot = game.snapshot()

    value_1 = rollouts.estimate_rollout_value(game, snapshot, mover_seat=0, n_rollouts=15,
                                               rng=random.Random(42))
    value_2 = rollouts.estimate_rollout_value(game, snapshot, mover_seat=0, n_rollouts=15,
                                               rng=random.Random(42))

    assert np.array_equal(value_1, value_2), "Same seed must give byte-identical rollout results"
    print(f"✓ value_1={value_1} == value_2={value_2}")


if __name__ == '__main__':
    test_estimate_rollout_value_converges_on_near_certain_win()
    test_estimate_rollout_value_never_mutates_the_real_game()
    test_estimate_rollout_value_does_not_perturb_the_global_random_state()
    test_estimate_rollout_value_is_reproducible_given_the_same_rng_state()
    print("\nAll test_rollouts.py tests passed!")
