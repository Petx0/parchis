#!/usr/bin/env python3
"""
Phase A tests for parchis/search/mcts.py: tree correctness (never picks a
masked-illegal action, concentrates visits on genuinely better moves in a
hand-constructed position), isolation from the real game's random stream,
and that the real Game object handed to search() is never mutated.
"""

import random

import pytest

from parchis.game.game import Game
from parchis.game.board import Board
from parchis.search import mcts
from parchis.search.isolated_random import isolated_random
from parchis.search.heuristic_eval import make_heuristic_evaluate_fn


def _snapshot(game):
    """Cheap fingerprint of board+piece state, for mutation checks."""
    return tuple(
        (piece.color, piece.piece_id, piece.position, piece.in_base, piece.finished)
        for player in game.players for piece in player.pieces
    )


def test_isolated_random_never_leaks_into_global_state():
    print("\nTesting isolated_random restores global random state exactly...")
    random.seed(999)
    before = random.getstate()

    with isolated_random(seed=1):
        for _ in range(50):
            random.random()

    after = random.getstate()
    assert before == after, "isolated_random must restore the exact prior global state"
    print("✓ Global random state unaffected by isolated_random's block")


def test_isolated_random_is_deterministic_given_same_seed():
    print("\nTesting isolated_random is reproducible given the same seed...")
    seq1 = []
    with isolated_random(seed=("x", 42)):
        seq1 = [random.random() for _ in range(10)]

    random.seed(12345)  # perturb global state in between
    random.random()

    seq2 = []
    with isolated_random(seed=("x", 42)):
        seq2 = [random.random() for _ in range(10)]

    assert seq1 == seq2, "Same seed must reproduce the same internal sequence"
    print("✓ isolated_random(seed=X) is reproducible regardless of surrounding global state")


def test_search_never_returns_a_masked_illegal_action():
    print("\nTesting search() only ever returns a legal move...")
    random.seed(0)  # Game() rolls dice internally; pin for reproducibility.
    game = Game(num_players=2)
    player = game.players[0]
    legal_moves = game.get_legal_moves(player, 5)  # roll=5: entry moves available
    assert legal_moves, "Test setup: expected at least one legal move for roll=5"
    legal_piece_ids = {piece.piece_id for piece, _np, _mt in legal_moves}

    evaluate_fn = make_heuristic_evaluate_fn()
    move, root = mcts.search(game, agent_seat=0, legal_moves=legal_moves, dice_roll=5,
                              n_simulations=20, evaluate_fn=evaluate_fn, rng_seed=7)

    assert move is not None
    piece, _new_pos, _move_type = move
    assert piece.piece_id in legal_piece_ids, "search() must never pick a masked-illegal action"
    print(f"✓ search() chose piece {piece.piece_id}, one of legal {legal_piece_ids}")


def test_real_game_state_never_mutated_by_search():
    print("\nTesting search() never mutates the real Game object it's given...")
    random.seed(0)  # Game() rolls dice internally; pin for reproducibility.
    game = Game(num_players=2)
    player = game.players[0]
    legal_moves = game.get_legal_moves(player, 5)
    before = _snapshot(game)

    evaluate_fn = make_heuristic_evaluate_fn()
    mcts.search(game, agent_seat=0, legal_moves=legal_moves, dice_roll=5,
                n_simulations=20, evaluate_fn=evaluate_fn, rng_seed=3)

    after = _snapshot(game)
    assert before == after, "search() must only ever operate on deepcopies, never the real game"
    print("✓ Real game object's piece/board state is bit-identical before and after search()")


def test_search_concentrates_visits_on_a_clearly_better_move():
    print("\nTesting search() concentrates simulations on the clearly better of two moves...")
    # Game(num_players=2) internally rolls dice (via the global `random`
    # module -- see determine_starting_player()) to pick colors/starting
    # player and to rotate self.players accordingly, so leaving this
    # unseeded makes the test's own outcome depend on ambient random state
    # (order-dependent when run alongside other tests -- confirmed while
    # writing this test). Seeding pins a known-good, verified-robust setup
    # (checked across multiple internal search rng_seeds) instead.
    random.seed(1)
    game = Game(num_players=2)
    player = game.players[0]

    # piece_a: 3 squares behind an opponent piece sitting on a non-safe
    # square -- rolling 3 captures it (sends it back to base, a big drop in
    # the OPPONENT's progress -- exactly what make_heuristic_evaluate_fn's
    # agent_progress - opp_progress rewards). piece_b: elsewhere, rolling 3
    # just advances it 3 squares with no capture. Both legal for the same
    # roll=3.
    opponent = game.players[1]
    opponent_piece = opponent.pieces[0]
    game.board.remove_piece(opponent_piece)
    opponent_piece.move_to(40)  # confirmed non-safe main-track square
    game.board.add_piece(opponent_piece, 40)
    assert not Board.is_safe_square(game.board, 40)

    piece_a = player.pieces[0]
    piece_b = player.pieces[1]

    game.board.remove_piece(piece_a)
    piece_a.move_to(37)  # 37 + 3 = 40, lands exactly on opponent_piece
    game.board.add_piece(piece_a, 37)

    game.board.remove_piece(piece_b)
    piece_b.move_to(10)
    game.board.add_piece(piece_b, 10)

    legal_moves = game.get_legal_moves(player, 3)
    legal_ids = {piece.piece_id for piece, _np, _mt in legal_moves}
    assert piece_a.piece_id in legal_ids and piece_b.piece_id in legal_ids, (
        "Test setup: both piece_a (capturing) and piece_b (plain advance) must be legal for roll=3"
    )
    capturing_move = next(m for m in legal_moves if m[0] is piece_a)
    assert capturing_move[1] == 40, "Test setup: piece_a's move must land exactly on the opponent"

    evaluate_fn = make_heuristic_evaluate_fn()
    _move, root = mcts.search(game, agent_seat=0, legal_moves=legal_moves, dice_roll=3,
                               n_simulations=150, evaluate_fn=evaluate_fn, rng_seed=11)

    counts = mcts.visit_counts(root)
    print(f"  visit counts: piece_a(capture)={counts.get(piece_a.piece_id, 0)} "
          f"piece_b(plain)={counts.get(piece_b.piece_id, 0)}")
    assert counts[piece_a.piece_id] > counts[piece_b.piece_id], (
        "Search should allocate more simulations to the clearly better (capturing) move"
    )
    print("✓ Search concentrated more visits on the capturing move than the plain advance")


def test_full_game_completes_without_crashing_when_driven_by_search():
    print("\nTesting a full game driven by search() for one seat completes cleanly...")
    random.seed(0)  # Game() rolls dice internally; pin for reproducibility.
    game = Game(num_players=2)
    agent_seat = 0
    evaluate_fn = make_heuristic_evaluate_fn()
    max_real_turns = 400

    turns = 0
    while not game.game_over and turns < max_real_turns:
        if game.current_player_idx == agent_seat:
            player = game.get_current_player()
            player.choose_move = lambda legal_moves: (
                mcts.search(game, agent_seat, legal_moves,
                             dice_roll=None, n_simulations=10,
                             evaluate_fn=evaluate_fn, rng_seed=turns)[0]
                if legal_moves else None
            )
        # Opponent's choose_move is left untouched -- still the class
        # default (random), never overridden in this test.
        game.play_turn()
        turns += 1

    assert turns < max_real_turns, "Game should finish well within the safety cap"
    print(f"✓ Full game completed in {turns} real turns "
          f"({'winner: ' + game.winner.color if game.winner else 'no winner'})")


if __name__ == '__main__':
    test_isolated_random_never_leaks_into_global_state()
    test_isolated_random_is_deterministic_given_same_seed()
    test_search_never_returns_a_masked_illegal_action()
    test_real_game_state_never_mutated_by_search()
    test_search_concentrates_visits_on_a_clearly_better_move()
    test_full_game_completes_without_crashing_when_driven_by_search()
    print("\nAll MCTS tests passed!")
