#!/usr/bin/env python3
"""
Tests for Game.snapshot()/restore() -- docs/AGENT_REBUILD_PLAN.md Part 3
item 1, "the single highest-risk piece of new code" in the rebuild plan: any
search built on snapshot/restore instead of copy.deepcopy is only as sound
as this round trip, and a subtle bug here would corrupt search silently
(wrong values, not a crash) rather than loudly. See also §1.1 (measured
deepcopy/snapshot costs) and Part 6 (risk table).

test_snapshot_restore_matches_deepcopy_over_10000_random_positions is the
property test the plan calls for: 10,000 sampled decision points, each
checked byte-identical to an independent copy.deepcopy() taken at the same
instant. The rest are small, deterministic anchors for the specific tricky
cases a random walk might rarely hit (a capture, a finish, game-over/winner)
and for the in-place-mutation guarantee restore()'s design depends on.
"""

import copy
import random

from parchis.game.game import Game


def _state_fingerprint(game):
    """A structural fingerprint of exactly the fields snapshot()/restore()
    are contracted to round-trip: board.positions, every piece's
    position/in_base/finished/move_order, board.move_counter,
    current_player_idx, turn_number, game_over, winner. Keyed by
    (color, piece_id) rather than object identity so a live Game and an
    independent copy.deepcopy() of it (which necessarily allocates new
    Piece/Player objects) can be compared for equal content."""
    positions = {
        position: tuple((p.color, p.piece_id) for p in pieces)
        for position, pieces in game.board.positions.items()
    }
    piece_fields = {
        (piece.color, piece.piece_id):
            (piece.position, piece.in_base, piece.finished, piece.move_order)
        for player in game.players
        for piece in player.pieces
    }
    return (
        positions,
        game.board.move_counter,
        piece_fields,
        game.current_player_idx,
        game.turn_number,
        game.game_over,
        game.winner.color if game.winner is not None else None,
    )


def test_snapshot_restore_matches_deepcopy_over_10000_random_positions():
    """snapshot() -> mutate (a real play_turn()) -> restore() must land back
    on a state whose fingerprint is byte-identical to a copy.deepcopy()
    taken at the same pre-mutation instant. Checked at 10,000 sampled
    decision points spanning many games and all of {2,3,4} players, each
    game allowed to run to completion so early-game (pieces in base),
    mid-game (captures, blockades), and end-game (home column, finishing,
    game_over/winner) states are all exercised."""
    print("\nTesting snapshot/restore matches deepcopy over 10,000 random positions...")

    random.seed(20260825)
    samples = 0
    games_played = 0
    games_completed = 0

    while samples < 10_000:
        num_players = random.choice([2, 3, 4])
        game = Game(num_players=num_players)
        games_played += 1

        while not game.game_over and samples < 10_000:
            snap = game.snapshot()
            expected = copy.deepcopy(game)
            expected_fingerprint = _state_fingerprint(expected)

            game.play_turn()  # real mutation: dice, moves, bonuses, captures, wins
            game.restore(snap)

            actual_fingerprint = _state_fingerprint(game)
            assert actual_fingerprint == expected_fingerprint, (
                f"restore() diverged from deepcopy at sample {samples} "
                f"(num_players={num_players})"
            )
            samples += 1

            # Now actually advance the game for real (fresh dice roll, not
            # the one just undone above) so the next sample in this game is
            # a genuinely different position, not the same turn replayed.
            if samples < 10_000:
                game.play_turn()

        if game.game_over:
            games_completed += 1

    assert games_completed >= 1, (
        "Expected at least one game to run to completion, exercising "
        "game_over/winner in the fingerprint"
    )
    print(f"✓ {samples} snapshot/restore round trips across {games_played} games "
          f"({games_completed} completed) all matched deepcopy exactly")


def test_restore_mutates_pieces_in_place_preserves_identity():
    """restore() must write captured values back onto the SAME Piece/Player
    objects, never replace them -- this identity guarantee is what lets a
    search hold a reference to a piece across a restore() and still see it
    reflect the restored state, and is why snapshot() can skip copying
    piece objects at all."""
    print("\nTesting restore() preserves Piece/Player object identity...")

    game = Game(num_players=2)
    pieces_before = [piece for player in game.players for piece in player.pieces]
    players_before = list(game.players)

    snap = game.snapshot()
    game.play_turn()
    game.restore(snap)

    pieces_after = [piece for player in game.players for piece in player.pieces]
    players_after = list(game.players)

    assert players_after == players_before and all(
        a is b for a, b in zip(players_after, players_before)
    ), "restore() must not replace Player objects"
    assert len(pieces_after) == len(pieces_before) and all(
        a is b for a, b in zip(pieces_after, pieces_before)
    ), "restore() must not replace Piece objects, only mutate their attributes"
    print("✓ restore() mutates existing Piece/Player objects in place")


def test_snapshot_restore_around_capture():
    """Deterministic anchor: a capture sends the captured piece to base
    (position=None, in_base=True) and moves the capturing piece onto the
    target square. restore() must put both pieces, and the target square's
    occupant list, back exactly as they were."""
    print("\nTesting snapshot/restore around a capture...")

    game = Game(num_players=2)
    mover = game.players[0].pieces[0]
    victim = game.players[1].pieces[0]

    mover_start = 40
    victim_pos = 41
    for piece, pos in ((mover, mover_start), (victim, victim_pos)):
        game.board.remove_piece(piece)
        piece.move_to(pos)
        game.board.add_piece(piece, pos)

    snap = game.snapshot()
    expected = copy.deepcopy(game)
    expected_fingerprint = _state_fingerprint(expected)

    move_info = game.execute_move(mover, victim_pos, 'move')
    assert victim in move_info.captured, "Test setup error: expected a capture"
    assert victim.in_base and victim.position is None

    game.restore(snap)

    assert _state_fingerprint(game) == expected_fingerprint
    assert mover.position == mover_start and not mover.in_base
    assert victim.position == victim_pos and not victim.in_base
    assert victim in game.board.get_pieces_at(victim_pos)
    print("✓ snapshot/restore correctly undoes a capture")


def test_snapshot_restore_around_piece_finishing():
    """Deterministic anchor: Board.move_piece special-cases the final
    position -- a finishing piece is marked finished and NOT added to
    board.positions (docs/AGENT_REBUILD_PLAN.md references this special
    case). restore() must reverse that: unset `finished` and put the piece
    back on the board at its pre-move position."""
    print("\nTesting snapshot/restore around a piece finishing...")

    game = Game(num_players=2)
    piece = game.players[0].pieces[0]
    pre_finish_pos = 74
    game.board.remove_piece(piece)
    piece.move_to(pre_finish_pos)
    game.board.add_piece(piece, pre_finish_pos)

    snap = game.snapshot()
    expected = copy.deepcopy(game)
    expected_fingerprint = _state_fingerprint(expected)

    game.execute_move(piece, 76, 'move')
    assert piece.finished and piece.position == 76
    assert 76 not in game.board.positions or piece not in game.board.positions[76]

    game.restore(snap)

    assert _state_fingerprint(game) == expected_fingerprint
    assert not piece.finished
    assert piece.position == pre_finish_pos
    assert piece in game.board.get_pieces_at(pre_finish_pos)
    print("✓ snapshot/restore correctly undoes a piece finishing")


def test_snapshot_restore_around_game_over():
    """Deterministic anchor: game_over/winner must round-trip too, not just
    board/piece state -- restore() from a pre-win snapshot after a
    game-ending move must clear both back to their prior values."""
    print("\nTesting snapshot/restore around a game-ending move...")

    game = Game(num_players=2)
    player = game.players[0]
    for piece in player.pieces[1:]:
        piece.finished = True
        piece.position = 76
    last_piece = player.pieces[0]
    game.board.remove_piece(last_piece)
    pre_win_pos = 74
    last_piece.move_to(pre_win_pos)
    game.board.add_piece(last_piece, pre_win_pos)

    assert not player.has_won()
    snap = game.snapshot()
    expected_game_over = game.game_over
    expected_winner = game.winner

    game.execute_move(last_piece, 76, 'move')
    assert player.has_won()
    game.game_over = True
    game.winner = player

    game.restore(snap)

    assert game.game_over == expected_game_over == False
    assert game.winner is expected_winner is None
    assert not last_piece.finished
    assert last_piece.position == pre_win_pos
    print("✓ snapshot/restore correctly undoes game_over/winner")


def test_restored_game_can_continue_playing():
    """Structural smoke test: board.positions after restore() must be made
    of genuinely mutable lists (Board.add_piece/remove_piece append/remove
    on them), not e.g. the tuples snapshot() uses internally -- otherwise a
    restored game would raise the next time anything moves."""
    print("\nTesting a restored game can continue playing normally...")

    game = Game(num_players=3)
    for _ in range(5):
        snap = game.snapshot()
        game.play_turn()
        game.restore(snap)

    for _ in range(50):
        if game.game_over:
            break
        game.play_turn()

    print(f"✓ Restored game continued for {game.turn_number} turns with no errors")


if __name__ == '__main__':
    test_snapshot_restore_matches_deepcopy_over_10000_random_positions()
    test_restore_mutates_pieces_in_place_preserves_identity()
    test_snapshot_restore_around_capture()
    test_snapshot_restore_around_piece_finishing()
    test_snapshot_restore_around_game_over()
    test_restored_game_can_continue_playing()
    print("\nAll snapshot/restore tests passed!")
