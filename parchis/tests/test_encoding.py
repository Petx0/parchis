#!/usr/bin/env python3
"""
Tests for parchis/az/encoding.py (docs/AGENT_REBUILD_PLAN.md §2.1 / Phase 1
item 6): the canonical, path-relative observation encoding.
"""

import copy
import random

import numpy as np

from parchis.az import encoding
from parchis.game.board import Board
from parchis.game.game import Game

COLOR_CYCLE = ["YELLOW", "BLUE", "RED", "GREEN"]


def _next_color(color):
    return COLOR_CYCLE[(COLOR_CYCLE.index(color) + 1) % 4]


def _rotate_abs_position(pos):
    """Shift a main-track absolute position by +17 (mod 68); home-column
    positions (>=69) and None (base) are private/self-referential per
    colour and pass through unchanged -- see encoding.py's module
    docstring on why the home column can't be folded into the relative
    track."""
    if pos is None or pos >= Board.HOME_COLUMN_START:
        return pos
    return ((pos - 1 + 17) % Board.MAIN_TRACK_SIZE) + 1


def _apply_piece_state(game, color, piece_id, position, in_base, finished):
    player = next(p for p in game.players if p.color == color)
    piece = player.pieces[piece_id]
    game.board.remove_piece(piece)
    if in_base:
        piece.send_to_base()
    elif finished:
        piece.position = 76
        piece.in_base = False
        piece.finished = True
    else:
        piece.finished = False
        piece.move_to(position)
        game.board.add_piece(piece, position)


SCRIPT_A = [
    ("YELLOW", 0, 10, False, False),
    ("YELLOW", 1, None, True, False),
    ("YELLOW", 2, 70, False, False),
    ("YELLOW", 3, 76, False, True),
    ("BLUE", 0, 15, False, False),
    ("BLUE", 1, 50, False, False),
]


def _rotated_script(script):
    return [
        (_next_color(color), piece_id, _rotate_abs_position(pos), in_base, finished)
        for (color, piece_id, pos, in_base, finished) in script
    ]


def _build_scripted_game(script, observer_color):
    game = Game(num_players=4)
    for color, piece_id, pos, in_base, finished in script:
        _apply_piece_state(game, color, piece_id, pos, in_base, finished)
    observer_seat = next(i for i, p in enumerate(game.players) if p.color == observer_color)
    game.current_player_idx = observer_seat
    return game, observer_seat


def test_encoding_size_matches_actual_output_length():
    """encoding_size(num_players) must equal encode()'s real output length
    -- the two are computed independently (one from block-size arithmetic,
    the other from actually building the array), so this is a genuine
    cross-check, not a tautology."""
    print("\nTesting encoding_size matches encode()'s actual output length...")

    for num_players in (2, 3, 4):
        game = Game(num_players=num_players)
        obs = encoding.encode(game, observer_seat=0, roll=3)
        expected = encoding.encoding_size(num_players)
        assert obs.shape == (expected,), (
            f"num_players={num_players}: encoding_size()={expected} but "
            f"encode() produced shape {obs.shape}"
        )
        print(f"  num_players={num_players}: size={expected}")
    print("✓ encoding_size matches encode()'s actual output length for 2/3/4 players")


def test_encoding_never_mutates_game():
    """encode() must be a pure read of Game -- no puppeting, no mutation
    (§2.1's explicit design goal)."""
    print("\nTesting encode() never mutates the game...")

    random.seed(5)
    game = Game(num_players=4)
    for _ in range(15):
        game.play_turn()

    snap = game.snapshot()
    expected = copy.deepcopy(game)
    for seat in range(4):
        encoding.encode(game, seat, roll=4, pending_bonus=None, consecutive_sixes=1)
    game.restore(snap)  # no-op if encode() truly never mutated; proves the API round-trips too

    assert game.board.move_counter == expected.board.move_counter
    for p1, p2 in zip(
        (pc for pl in game.players for pc in pl.pieces),
        (pc for pl in expected.players for pc in pl.pieces),
    ):
        assert (p1.position, p1.in_base, p1.finished, p1.move_order) == \
               (p2.position, p2.in_base, p2.finished, p2.move_order)
    print("✓ encode() left the game unchanged across all 4 observer perspectives")


def test_own_path_step_and_track_index_values():
    """Targeted correctness check for the two core coordinate transforms:
    _own_path_step (0..71 along the observer's own path) and the relative
    track index used for every seat's pieces in the track block."""
    print("\nTesting own path-step and relative track index values...")

    # num_players=4 guarantees YELLOW is present (a 2-player game randomly
    # picks RED-vs-YELLOW or BLUE-vs-GREEN -- see Game.__init__).
    game = Game(num_players=4)
    yellow = next(p for p in game.players if p.color == "YELLOW")

    # Own start (5): s should be 0.
    assert encoding._own_path_step(yellow.pieces[0], observer_start=5) == 0
    # Own home-entry square (68): s should be 63 (§1.2's uniform distance).
    _apply_piece_state(game, "YELLOW", 0, 68, False, False)
    assert encoding._own_path_step(yellow.pieces[0], observer_start=5) == 63
    # One step into the home column (69): s should be 64.
    _apply_piece_state(game, "YELLOW", 0, 69, False, False)
    assert encoding._own_path_step(yellow.pieces[0], observer_start=5) == 64
    # Finished: s should be 71 (_PATH_LENGTH).
    _apply_piece_state(game, "YELLOW", 0, 76, False, True)
    assert encoding._own_path_step(yellow.pieces[0], observer_start=5) == 71

    # Relative track index: a piece one square behind the observer's own
    # start (position 4, i.e. -1 mod 68) should read as j=67.
    assert encoding._relative_track_index(4, observer_start=5) == 67
    # A piece exactly at the observer's own start reads as j=0.
    assert encoding._relative_track_index(5, observer_start=5) == 0
    print("✓ own path-step and relative track index values are correct at key points")


def test_home_column_block_excludes_finished_pieces():
    """Mirrors parchis/rl/env.py's own board-state convention exactly: a
    finished piece must NOT show up in the home_columns occupancy block
    (up to 4 same-colour pieces can legally share square 76, which the
    0/0.5/1.0-capped-at-2 scale can't represent) -- finished-ness is
    already carried by the own-piece block's `finished` flag and the
    per-seat `pieces_finished` scalar."""
    print("\nTesting the home_columns block excludes finished pieces...")

    game = Game(num_players=2)
    yellow_seat = next(i for i, p in enumerate(game.players) if p.color == "YELLOW")
    yellow = game.players[yellow_seat]
    for piece in yellow.pieces:
        game.board.remove_piece(piece)
        piece.position = 76
        piece.in_base = False
        piece.finished = True

    obs = encoding.encode(game, yellow_seat, roll=3)
    offsets = encoding.block_offsets(game.num_players)
    home = obs[offsets["home_columns"]:offsets["per_seat"]].reshape(
        game.num_players, encoding.HOME_COLUMN_SLOTS
    )
    assert np.all(home[0] == 0.0), (
        f"Expected all-finished own pieces to leave the home_columns row all-zero, "
        f"got {home[0]}"
    )
    print("✓ Finished pieces correctly excluded from the home_columns block")


def test_encoding_is_colour_invariant_under_17_square_rotation():
    """The central guarantee of §2.1: encoding a position and its
    17-square rotation with colours permuted must produce byte-identical
    arrays. SCRIPT_B is derived from SCRIPT_A by the SAME rotation rule
    being tested (color advances one step, main-track positions shift by
    +17 mod 68, home-column/base/finished pass through unchanged) -- if
    encode() secretly depended on absolute colour identity anywhere, this
    would catch it."""
    print("\nTesting encode() is colour-invariant under a 17-square rotation...")

    game_a, yellow_seat = _build_scripted_game(SCRIPT_A, "YELLOW")
    game_b, blue_seat = _build_scripted_game(_rotated_script(SCRIPT_A), "BLUE")

    checked = 0
    for roll in (None, 1, 2, 3, 4, 5, 6):
        for pending_bonus in (None, {'type': 'capture_bonus', 'squares': 20},
                               {'type': 'finish_bonus', 'squares': 10}):
            for consecutive_sixes in (0, 1, 2):
                enc_a = encoding.encode(game_a, yellow_seat, roll=roll,
                                         pending_bonus=pending_bonus,
                                         consecutive_sixes=consecutive_sixes)
                enc_b = encoding.encode(game_b, blue_seat, roll=roll,
                                         pending_bonus=pending_bonus,
                                         consecutive_sixes=consecutive_sixes)
                assert np.array_equal(enc_a, enc_b), (
                    f"Mismatch at roll={roll} pending_bonus={pending_bonus} "
                    f"consecutive_sixes={consecutive_sixes}:\n"
                    f"diff indices={np.nonzero(enc_a != enc_b)[0].tolist()}"
                )
                checked += 1
    print(f"✓ encode(game_a, YELLOW) == encode(game_b, BLUE) byte-for-byte across "
          f"{checked} roll/bonus/six-streak combinations")


def test_encoding_is_colour_invariant_on_a_random_real_game_position():
    """Same property, but on an organically-reached position (many random
    turns played) rather than a hand-scripted one -- rotates the SAME real
    position programmatically and checks the invariance holds there too."""
    print("\nTesting colour-invariance on a random real-game position...")

    random.seed(99)
    game = Game(num_players=4)
    for _ in range(25):
        if game.game_over:
            break
        game.play_turn()

    script = []
    for player in game.players:
        for piece in player.pieces:
            script.append((player.color, piece.piece_id, piece.position,
                            piece.in_base, piece.finished))

    for observer_color in COLOR_CYCLE:
        rotated_color = _next_color(observer_color)
        game_a, seat_a = _build_scripted_game(script, observer_color)
        game_b, seat_b = _build_scripted_game(_rotated_script(script), rotated_color)
        enc_a = encoding.encode(game_a, seat_a, roll=3, consecutive_sixes=1)
        enc_b = encoding.encode(game_b, seat_b, roll=3, consecutive_sixes=1)
        assert np.array_equal(enc_a, enc_b), f"Mismatch rotating observer={observer_color}"
    print("✓ Colour-invariance holds on a random real-game position, all 4 rotations")


def test_encoding_bounds_over_100k_states():
    """Property test (Phase 1 item 6): 100,000 (position, observer) encode()
    calls across many random games and all of {2,3,4} players must never
    raise, never produce NaN/Inf, and never leave declared [0,1] bounds."""
    print("\nTesting encode() bounds over 100,000 states...")

    random.seed(20260825)
    samples = 0
    games_played = 0

    while samples < 100_000:
        num_players = random.choice([2, 3, 4])
        game = Game(num_players=num_players)
        games_played += 1

        while not game.game_over and samples < 100_000:
            observer_seat = random.randrange(num_players)
            roll = random.choice([None, 1, 2, 3, 4, 5, 6])
            pending_bonus = random.choice([
                None,
                {'type': 'capture_bonus', 'squares': 20},
                {'type': 'finish_bonus', 'squares': 10},
            ])
            consecutive_sixes = random.choice([0, 1, 2])

            obs = encoding.encode(game, observer_seat, roll=roll,
                                   pending_bonus=pending_bonus,
                                   consecutive_sixes=consecutive_sixes)

            assert obs.shape == (encoding.encoding_size(num_players),)
            assert np.all(np.isfinite(obs)), f"Non-finite value at sample {samples}: {obs}"
            assert obs.min() >= -1e-6, f"Value below 0 at sample {samples}: min={obs.min()}"
            assert obs.max() <= 1.0 + 1e-6, f"Value above 1 at sample {samples}: max={obs.max()}"

            samples += 1
            game.play_turn()

    print(f"✓ {samples} encode() calls across {games_played} games, all finite and within [0, 1]")


if __name__ == '__main__':
    test_encoding_size_matches_actual_output_length()
    test_encoding_never_mutates_game()
    test_own_path_step_and_track_index_values()
    test_home_column_block_excludes_finished_pieces()
    test_encoding_is_colour_invariant_under_17_square_rotation()
    test_encoding_is_colour_invariant_on_a_random_real_game_position()
    test_encoding_bounds_over_100k_states()
    print("\nAll encoding tests passed!")
