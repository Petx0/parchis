#!/usr/bin/env python3
"""
Byte-identity tests for the C6 engine speedups (docs/V2_PLAN.md C6(b)/(c)):
Board's cached blockade set, the closed-form calculate_new_position /
path_crosses_blockade, the single-pass get_legal_moves, and the cached
own-blockade count in the encoder. Each is checked against a reference
implementation -- the pre-C6 code, kept verbatim here -- over random
real-game positions (2, 3 and 4 players), with snapshot/restore interleaved
so the cache's rebuild path is exercised too.
"""

import random

import numpy as np

from parchis.az import encoding
from parchis.game.board import Board
from parchis.game.constants import (ALL_OUT_BONUS_ROLL, BONUS_TURN_ROLL, CAPTURE_BONUS_SQUARES,
                                    ENTRY_ROLL, FINISH_BONUS_SQUARES)
from parchis.game.game import Game

ROLLS = (1, 2, 3, 4, 5, 6, ALL_OUT_BONUS_ROLL, FINISH_BONUS_SQUARES, CAPTURE_BONUS_SQUARES)


# ---------------------------------------------------------------------------
# Reference implementations: RuleEngine as it was before C6, verbatim.
# ---------------------------------------------------------------------------

def _ref_get_blockades(board):
    blockades = set()
    for position in Board.SAFE_SQUARES:
        pieces_at_pos = board.get_pieces_at(position)
        if len(pieces_at_pos) == Board.MAX_PIECES_PER_SQUARE:
            if pieces_at_pos[0].color == pieces_at_pos[1].color:
                blockades.add(position)
    return blockades


def _ref_calculate_new_position(rules, player, current_pos, dice_roll):
    new_position = rules.compute_path(player, current_pos, dice_roll)[-1]
    if new_position > Board.FINAL_POSITION:
        return None
    return new_position


def _ref_path_crosses_blockade(rules, player, start_pos, num_squares, blockades):
    if not blockades:
        return False
    path = rules.compute_path(player, start_pos, num_squares)
    return any(pos in blockades for pos in path)


def _ref_occupancy_count_for_move(board, position, color):
    pieces = board.get_pieces_at(position)
    if position >= Board.HOME_COLUMN_START:
        pieces = [p for p in pieces if p.color == color]
    return len(pieces)


def _ref_get_legal_moves(rules, player, dice_roll):
    board = rules.board
    legal_moves = []
    effective_roll = dice_roll
    if dice_roll == BONUS_TURN_ROLL and len(player.get_pieces_in_base()) == 0:
        effective_roll = ALL_OUT_BONUS_ROLL
    if dice_roll == ENTRY_ROLL:
        pieces_in_base = player.get_pieces_in_base()
        if pieces_in_base:
            pieces_at_start = board.get_pieces_at(player.starting_position)
            own_pieces = [p for p in pieces_at_start if p.color == player.color]
            opponent_pieces = [p for p in pieces_at_start if p.color != player.color]
            can_enter = False
            if len(pieces_at_start) == 0:
                can_enter = True
            elif len(own_pieces) == 1 and len(opponent_pieces) == 0:
                can_enter = True
            elif len(own_pieces) == 0 and len(opponent_pieces) == 1:
                can_enter = True
            elif len(own_pieces) == Board.MAX_PIECES_PER_SQUARE:
                can_enter = False
            elif len(own_pieces) == 1 and len(opponent_pieces) == 1:
                can_enter = True
            elif len(opponent_pieces) == Board.MAX_PIECES_PER_SQUARE:
                can_enter = True
            if can_enter:
                for piece in pieces_in_base:
                    legal_moves.append((piece, player.starting_position, 'enter'))
                return legal_moves
    all_blockades = _ref_get_blockades(board)
    for piece in player.get_pieces_on_board():
        current_pos = piece.position
        if current_pos >= Board.HOME_COLUMN_START:
            new_position = current_pos + effective_roll
            if new_position < Board.FINAL_POSITION:
                if _ref_occupancy_count_for_move(board, new_position, player.color) < Board.MAX_PIECES_PER_SQUARE:
                    legal_moves.append((piece, new_position, 'move'))
            elif new_position == Board.FINAL_POSITION:
                legal_moves.append((piece, new_position, 'finish'))
        else:
            new_position = _ref_calculate_new_position(rules, player, current_pos, effective_roll)
            if new_position is not None:
                if _ref_occupancy_count_for_move(board, new_position, player.color) < Board.MAX_PIECES_PER_SQUARE:
                    if not _ref_path_crosses_blockade(rules, player, current_pos, effective_roll, all_blockades):
                        legal_moves.append((piece, new_position, 'move'))
    if dice_roll == BONUS_TURN_ROLL and all_blockades:
        blockade_opening_moves = []
        for move in legal_moves:
            piece, new_pos, move_type = move
            if piece.position in all_blockades:
                blockade_opening_moves.append(move)
        if blockade_opening_moves:
            legal_moves = blockade_opening_moves
    return legal_moves


def _ref_own_blockades_on_track(player):
    return sum(
        1 for sq in Board.SAFE_SQUARES
        if len([p for p in player.pieces
                if not p.in_base and not p.finished and p.position == sq]) == 2
    ) / 2.0


# ---------------------------------------------------------------------------
# Random real-game positions, with snapshot/restore interleaved.
# ---------------------------------------------------------------------------

def _random_positions(n_states, seed):
    """Yields (game, is_after_restore) over `n_states` positions reached by
    random play; before every ply also executes one random legal move
    under a snapshot and restores it, yielding the post-restore state too
    (same position, cache rebuilt from scratch)."""
    rng = random.Random(seed)
    random.seed(seed)  # Game.play_turn draws dice from the global RNG
    produced = 0
    while produced < n_states:
        game = Game(num_players=rng.choice([2, 3, 4]))
        while not game.game_over and produced < n_states:
            yield game
            produced += 1
            player = game.get_current_player()
            moves = game.get_legal_moves(player, rng.choice(ROLLS))
            if moves:
                snap = game.snapshot()
                piece, new_pos, move_type = rng.choice(moves)
                game.execute_move(piece, new_pos, move_type)
                yield game  # mutated state: cache updated incrementally
                produced += 1
                game.restore(snap)
                yield game  # restored state: cache rebuilt
                produced += 1
            game.play_turn()


def _moves_key(moves):
    return [(piece.color, piece.piece_id, new_pos, move_type) for piece, new_pos, move_type in moves]


def test_blockade_cache_matches_rescan_over_100k_states():
    print("\nTesting Board.blockades against a full rescan over 100,000 states...")
    n = 0
    for game in _random_positions(100_000, seed=2026_11_01):
        assert game.board.blockades == _ref_get_blockades(game.board), f"state {n}"
        assert game.get_blockades() == _ref_get_blockades(game.board)
        n += 1
    assert n == 100_000
    print(f"✓ cached blockade set identical to the rescan on {n} states")


def test_legal_moves_and_path_helpers_are_byte_identical_over_100k_states():
    """Every player, every roll value the engine ever passes (1-7, 10, 20):
    same moves, same order, same tuples; and the two closed-form helpers
    agree with their compute_path originals on every on-board piece."""
    print("\nTesting get_legal_moves / path helpers against the pre-C6 reference over 100,000 states...")
    n_states = n_calls = n_moves = 0
    for game in _random_positions(100_000, seed=7):
        rules = game.rules
        blockades = _ref_get_blockades(game.board)
        for player in game.players:
            for roll in ROLLS:
                new = rules.get_legal_moves(player, roll)
                ref = _ref_get_legal_moves(rules, player, roll)
                assert _moves_key(new) == _moves_key(ref), (
                    f"state {n_states} {player.color} roll {roll}: {new} != {ref}")
                assert all(a[0] is b[0] for a, b in zip(new, ref))
                n_calls += 1
                n_moves += len(new)
            for piece in player.get_pieces_on_board():
                if piece.position >= Board.HOME_COLUMN_START:
                    continue
                for roll in ROLLS:
                    assert (rules.calculate_new_position(player, piece.position, roll)
                            == _ref_calculate_new_position(rules, player, piece.position, roll))
                    assert (rules.path_crosses_blockade(player, piece.position, roll, blockades)
                            == _ref_path_crosses_blockade(rules, player, piece.position, roll, blockades))
        n_states += 1
    print(f"✓ {n_calls} get_legal_moves calls ({n_moves} moves) identical on {n_states} states")


def test_per_seat_own_blockade_count_matches_reference_over_100k_states():
    print("\nTesting the cached own-blockade scalar against the piece-list scan over 100,000 states...")
    n = 0
    for game in _random_positions(100_000, seed=11):
        for player in game.players:
            start = Board.STARTING_POSITIONS[player.color]
            fast = encoding._per_seat_scalars(player, start, game.board)
            slow = encoding._per_seat_scalars(player, start)
            assert np.array_equal(fast, slow), f"state {n} {player.color}: {fast} vs {slow}"
            assert fast[5] == _ref_own_blockades_on_track(player)
        n += 1
    print(f"✓ per-seat scalars identical on {n} states")


if __name__ == '__main__':
    test_blockade_cache_matches_rescan_over_100k_states()
    test_legal_moves_and_path_helpers_are_byte_identical_over_100k_states()
    test_per_seat_own_blockade_count_matches_reference_over_100k_states()
    print("\nAll rules_fast tests passed!")


# ---------------------------------------------------------------------------
# C6(c): capture_and_finish_summary and the encoder's threat scorer.
# ---------------------------------------------------------------------------

def _ref_capture_threat_scores(game, observer_player):
    """parchis/az/encoding.py::_capture_threat_scores as it was before C6, verbatim."""
    hit_counts = {piece.piece_id: 0 for piece in observer_player.pieces}
    for opponent in game.players:
        if opponent is observer_player:
            continue
        moves_by_roll = {v: game.get_legal_moves(opponent, v) for v in range(1, 7)}
        direct_targets_by_roll = {
            v: {p for m in moves for p in game.would_capture(m)}
            for v, moves in moves_by_roll.items()
        }
        capture_rolls = {v for v, targets in direct_targets_by_roll.items() if targets}
        finish_rolls = {
            v for v, moves in moves_by_roll.items()
            if any(move_type == 'finish' for (_p, _pos, move_type) in moves)
        }
        bonus20_targets = set()
        if capture_rolls:
            bonus20_moves = game.get_legal_moves(opponent, CAPTURE_BONUS_SQUARES)
            bonus20_targets = {p for m in bonus20_moves for p in game.would_capture(m)}
        bonus10_targets = set()
        if finish_rolls:
            bonus10_moves = game.get_legal_moves(opponent, FINISH_BONUS_SQUARES)
            bonus10_targets = {p for m in bonus10_moves for p in game.would_capture(m)}
        for piece in observer_player.pieces:
            for v in range(1, 7):
                hit = (
                    piece in direct_targets_by_roll[v]
                    or (v in capture_rolls and piece in bonus20_targets)
                    or (v in finish_rolls and piece in bonus10_targets)
                )
                if hit:
                    hit_counts[piece.piece_id] += 1
    return hit_counts


def test_capture_and_finish_summary_matches_legal_moves_over_100k_states():
    print("\nTesting capture_and_finish_summary against get_legal_moves + would_capture over 100,000 states...")
    n_states = n_calls = n_caps = 0
    for game in _random_positions(100_000, seed=23):
        rules = game.rules
        for player in game.players:
            for roll in ROLLS:
                captured, has_finish = rules.capture_and_finish_summary(player, roll)
                moves = rules.get_legal_moves(player, roll)
                ref_captured = {p for m in moves for p in rules.would_capture(m)}
                ref_finish = any(mt == 'finish' for _p, _np, mt in moves)
                assert captured == ref_captured, f"state {n_states} {player.color} roll {roll}"
                assert has_finish == ref_finish, f"state {n_states} {player.color} roll {roll}"
                n_calls += 1
                n_caps += len(captured)
        n_states += 1
    print(f"✓ {n_calls} summaries identical ({n_caps} captures) on {n_states} states")


def test_capture_threat_scores_match_pre_c6_reference_over_30k_states():
    print("\nTesting _capture_threat_scores against the pre-C6 reference over 30,000 states...")
    n = n_nonzero = 0
    for game in _random_positions(30_000, seed=31):
        for seat, player in enumerate(game.players):
            new = encoding._capture_threat_scores(game, player)
            ref = _ref_capture_threat_scores(game, player)
            assert new == ref, f"state {n} seat {seat}: {new} != {ref}"
            n_nonzero += sum(1 for v in new.values() if v)
        n += 1
    print(f"✓ threat scores identical on {n} states ({n_nonzero} threatened pieces)")
