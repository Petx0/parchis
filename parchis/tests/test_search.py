#!/usr/bin/env python3
"""
Tests for parchis/az/search.py (docs/AGENT_REBUILD_PLAN.md §2.3 / Phase 1
item 8): full-width expectimax over decision/chance nodes. Covers exactly
the five correctness properties Part 3 item 8 calls for, plus a
never-mutates-the-real-game check.
"""

import copy
import random

import numpy as np

from parchis.az import encoding, search
from parchis.game.board import Board
from parchis.game.game import Game


def _advance(pos, steps):
    """Modular main-track advance (matches RuleEngine.compute_path for
    paths that don't cross a home-entry point)."""
    return ((pos - 1 + steps) % Board.MAIN_TRACK_SIZE) + 1


def _mean_progress(player):
    return sum(
        1.0 if p.finished else (0.0 if p.in_base else p.position / Board.FINAL_POSITION)
        for p in player.pieces
    ) / 4.0


def _progress_evaluator(game, observer_seat, roll=None, pending_bonus=None, consecutive_sixes=0):
    """Toy evaluator: each seat's own mean progress, nothing else -- used
    to test that depth=1 search reproduces a plain greedy-progress agent."""
    return np.array([_mean_progress(p) for p in game.players], dtype=np.float64)


def test_depth1_progress_value_reproduces_greedy_progress_agent():
    """depth=1 with a value function equal to normalised progress must
    reproduce a greedy-progress agent exactly: at depth=1 the evaluator is
    called immediately after each root candidate move (no chance/bonus
    resolution attempted -- see search._evaluate_immediately), so the
    chosen move must be exactly the one maximising the mover's own
    progress right after moving."""
    print("\nTesting depth=1 with a progress evaluator reproduces greedy-progress...")

    game = Game(num_players=2)
    mover = game.get_current_player()
    piece_a = mover.pieces[0]
    piece_b = mover.pieces[1]
    game.board.remove_piece(piece_a)
    piece_a.move_to(_advance(mover.starting_position, 10))
    game.board.add_piece(piece_a, piece_a.position)
    game.board.remove_piece(piece_b)
    piece_b.move_to(_advance(mover.starting_position, 40))
    game.board.add_piece(piece_b, piece_b.position)

    roll = 3
    mover_seat = game.current_player_idx
    legal_moves = game.get_legal_moves(mover, roll)
    assert len(legal_moves) >= 2, "Test setup error: need >= 2 candidate moves"
    for move in legal_moves:
        assert not game.would_capture(move) and move[2] != 'finish', (
            "Test setup error: no candidate should capture/finish, to keep "
            "the depth=1 evaluation a clean 'immediately after moving' case"
        )

    best_move, move_values, _root_value = search.search(
        game, roll=roll, depth=1, evaluator=_progress_evaluator,
    )

    expected_scores = {}
    for move in legal_moves:
        snap = game.snapshot()
        game.execute_move(*move)
        expected_scores[move[0].piece_id] = _mean_progress(game.players[mover_seat])
        game.restore(snap)
    expected_best = max(expected_scores, key=expected_scores.get)

    assert best_move[0].piece_id == expected_best, (
        f"depth=1 chose piece_id={best_move[0].piece_id}, greedy-progress "
        f"expected piece_id={expected_best} (scores={expected_scores})"
    )
    for piece_id, expected in expected_scores.items():
        assert abs(move_values[piece_id][mover_seat] - expected) < 1e-9, (
            f"piece_id={piece_id}: search value {move_values[piece_id][mover_seat]} "
            f"!= greedy-progress expectation {expected}"
        )
    print("✓ depth=1 with a progress evaluator exactly reproduces greedy-progress choices")


def test_chance_node_equals_bruteforce_mean_over_6_faces():
    """Chance node values must equal a brute-force mean over the 6 faces
    computed independently -- isolates search._chance_node's own
    averaging/enumeration from the rest of the recursion."""
    print("\nTesting chance node values equal a brute-force mean over 6 faces...")

    random.seed(7)
    game = Game(num_players=3)
    for _ in range(8):
        if game.game_over:
            break
        game.play_turn()
    num_players = game.num_players

    def position_sum_evaluator(game, observer_seat, roll=None, pending_bonus=None, consecutive_sixes=0):
        return np.array([
            sum((p.position or 0) for p in player.pieces) for player in game.players
        ], dtype=np.float64)

    chance_value = search._chance_node(
        game, consecutive_sixes=0, second_six_piece=None, second_six_entered_home=False,
        depth=1, evaluator=position_sum_evaluator, num_players=num_players,
    )

    brute_force_total = np.zeros(num_players)
    for face in range(1, 7):
        brute_force_total += search._decision_value(
            game, face, None, 0, 1, position_sum_evaluator, num_players,
        )
    brute_force_mean = brute_force_total / 6.0

    assert np.allclose(chance_value, brute_force_mean, atol=1e-9), (
        f"_chance_node={chance_value} != brute-force mean={brute_force_mean}"
    )
    print("✓ _chance_node matches an independently-computed brute-force mean over 6 faces")


def test_search_result_invariant_to_move_ordering(monkeypatch):
    """depth=d's result must be invariant to move ordering: reversing
    whatever order Game.get_legal_moves happens to return must not change
    the resulting value vector or the chosen move."""
    print("\nTesting search results are invariant to move ordering...")

    # Built directly rather than reached via random play: three of mover's
    # pieces on board at well-separated positions guarantees >= 2 legal,
    # distinct-destination moves for most rolls, and gives the recursive
    # depth=2 search (chance node + second decision layer) real structure
    # to explore on both sides of the comparison.
    game = Game(num_players=3)
    mover = game.get_current_player()
    for piece, offset in zip(mover.pieces, (2, 15, 30)):
        game.board.remove_piece(piece)
        piece.move_to(_advance(mover.starting_position, offset))
        game.board.add_piece(piece, piece.position)

    def _distinct_destinations(moves):
        # Excludes genuine ties (e.g. two base pieces both entering via the
        # same roll land on the identical starting square) -- those are
        # correctly value-tied, so which one max() reports first is not a
        # meaningful ordering-dependence to test for.
        return len(moves) >= 2 and len({m[1] for m in moves}) == len(moves)

    roll = next(
        (r for r in range(1, 7) if _distinct_destinations(game.get_legal_moves(mover, r))),
        None,
    )
    assert roll is not None, "Test setup error: need a roll with >= 2 distinct-destination moves"

    def relative_progress_evaluator(game, observer_seat, roll=None, pending_bonus=None, consecutive_sixes=0):
        # Weighted by (piece_id + 1), not a plain mean: a plain sum/mean is
        # insensitive to WHICH piece contributes a given amount of
        # progress, which makes distinct-but-symmetric move choices
        # accidentally tie in value (a real property of that evaluator,
        # not a search bug -- but not what this test wants to exercise).
        values = np.zeros(game.num_players)
        for seat, player in enumerate(game.players):
            start = Board.STARTING_POSITIONS[player.color]
            values[seat] = sum(
                (p.piece_id + 1) * encoding._relative_piece_progress(p, start)
                for p in player.pieces
            ) / 10.0
        return values

    move1, _values1, root1 = search.search(game, roll=roll, depth=2, evaluator=relative_progress_evaluator)

    original_get_legal_moves = Game.get_legal_moves

    def reversed_get_legal_moves(self, player, dice_roll):
        return list(reversed(original_get_legal_moves(self, player, dice_roll)))

    monkeypatch.setattr(Game, "get_legal_moves", reversed_get_legal_moves)
    move2, _values2, root2 = search.search(game, roll=roll, depth=2, evaluator=relative_progress_evaluator)

    assert np.allclose(root1, root2, atol=1e-9), (
        f"Root value changed under reversed move ordering: {root1} vs {root2}"
    )
    assert move1[0].piece_id == move2[0].piece_id, (
        f"Chosen move changed under reversed move ordering: "
        f"{move1[0].piece_id} vs {move2[0].piece_id}"
    )
    print("✓ search() result (value and chosen move) is invariant to move ordering")


def _biased_evaluator(game, observer_seat, roll=None, pending_bonus=None, consecutive_sixes=0):
    """Deliberately naive: rewards captures achieved SO FAR heavily
    (10 per opponent piece currently in base), plus a small tiebreaker on
    the mover's own furthest-advanced on-board piece. Used to build a
    position where this evaluator's depth=1 view (immediately after the
    FIRST capture, bonus not yet resolved) ties on the dominant term and
    picks the wrong move on the small tiebreaker, while depth=2 (which
    resolves the bonus and finds a SECOND capture) is dominated by the big
    10-per-capture term and picks correctly."""
    values = np.zeros(game.num_players)
    for seat, player in enumerate(game.players):
        others_in_base = sum(
            len(p.get_pieces_in_base()) for i, p in enumerate(game.players) if i != seat
        )
        start = Board.STARTING_POSITIONS[player.color]
        own_progress = [
            encoding._relative_piece_progress(piece, start)
            for piece in player.pieces if not piece.in_base
        ]
        values[seat] = 10.0 * others_in_base + (max(own_progress) if own_progress else 0.0)
    return values


def test_2ply_only_capture_chain_found_at_depth2_missed_at_depth1():
    """A hand-built position where a 2-ply-only tactic exists (piece_A's
    capture triggers a bonus whose OWN 20-square move captures a SECOND
    piece) must be found at depth=2 and missed at depth=1, using an
    evaluator that ties on its dominant term at depth=1 (both candidates
    show exactly one capture immediately) and is decided by a small,
    misleading tiebreaker -- exactly the scenario depth=1's "evaluate
    immediately, don't resolve the bonus" design is expected to miss."""
    print("\nTesting a 2-ply-only capture chain is found at depth=2, missed at depth=1...")

    game = Game(num_players=2)
    mover = game.get_current_player()
    opponent = next(p for p in game.players if p is not mover)
    roll = 3

    piece_a, piece_b = mover.pieces[0], mover.pieces[1]
    victim1, victim2, victim3 = opponent.pieces[0], opponent.pieces[1], opponent.pieces[2]

    # piece_A's chain: captures victim1 on this roll, its OWN 20-bonus then
    # captures victim2.
    a_start = _advance(mover.starting_position, 2)
    v1_pos = _advance(mover.starting_position, 5)     # a_start + roll
    v2_pos = _advance(mover.starting_position, 25)    # v1_pos + 20 (capture bonus)
    # Decoy: piece_B captures victim3 on the SAME roll; its bonus leads nowhere.
    b_start = _advance(mover.starting_position, 45)
    v3_pos = _advance(mover.starting_position, 48)    # b_start + roll

    for piece, pos in ((piece_a, a_start), (piece_b, b_start),
                       (victim1, v1_pos), (victim2, v2_pos), (victim3, v3_pos)):
        game.board.remove_piece(piece)
        piece.move_to(pos)
        game.board.add_piece(piece, pos)

    mover_seat = game.current_player_idx
    legal_moves = game.get_legal_moves(mover, roll)
    move_a = next(m for m in legal_moves if m[0] is piece_a)
    move_b = next(m for m in legal_moves if m[0] is piece_b)
    assert game.would_capture(move_a) == [victim1], "Test setup error: piece_A should capture victim1"
    assert game.would_capture(move_b) == [victim3], "Test setup error: piece_B should capture victim3"

    # Confirm the SECOND capture is genuinely 2-ply-only: verify it exists
    # once piece_A's first move is applied, independent of the search.
    snap = game.snapshot()
    game.execute_move(*move_a)
    bonus_moves = game.get_legal_moves(mover, 20)
    bonus_move = next(m for m in bonus_moves if m[0] is piece_a)
    assert game.would_capture(bonus_move) == [victim2], (
        "Test setup error: piece_A's own 20-bonus should capture victim2"
    )
    game.restore(snap)

    move_d1, values_d1, _ = search.search(game, roll=roll, depth=1, evaluator=_biased_evaluator)
    move_d2, values_d2, _ = search.search(game, roll=roll, depth=2, evaluator=_biased_evaluator)

    print(f"  depth=1 chose piece_id={move_d1[0].piece_id} (values: "
          f"A={values_d1[piece_a.piece_id][mover_seat]:.4f} B={values_d1[piece_b.piece_id][mover_seat]:.4f})")
    print(f"  depth=2 chose piece_id={move_d2[0].piece_id} (values: "
          f"A={values_d2[piece_a.piece_id][mover_seat]:.4f} B={values_d2[piece_b.piece_id][mover_seat]:.4f})")

    assert move_d1[0].piece_id == piece_b.piece_id, (
        "Test setup error (or a regression): depth=1 was expected to be "
        "fooled into preferring the decoy piece_B"
    )
    assert move_d2[0].piece_id == piece_a.piece_id, (
        "depth=2 should find the 2-ply double-capture chain via piece_A"
    )
    print("✓ depth=2 finds the 2-ply-only double-capture chain; depth=1 misses it, as designed")


def test_no_legal_move_chains_do_not_recurse_without_bound():
    """Regression test: exact expectimax must also explore the branch
    where a player facing (say) 3 pieces stuck in base keeps NOT rolling
    the 5 they need, for arbitrarily long -- a real single dice sequence
    always eventually breaks that streak, but exhaustive chance-node
    enumeration does not get that guarantee for free. Before the fix, a
    "no legal move" decision cost zero depth (matching a chance node's own
    free resolution), so this specific branch could recurse until Python's
    call stack overflowed -- confirmed via RecursionError while sizing
    Part 3 item 10's gate. Runs many real search() calls, including at
    depth=3, against many independently-random positions/rolls, as a
    stress reproduction rather than one hand-built case."""
    print("\nTesting 'no legal move' chains don't recurse without bound...")

    random.seed(31)
    game = Game(num_players=2)
    checked = 0
    for _ in range(150):
        if game.game_over:
            game = Game(num_players=2)
        mover = game.get_current_player()
        for roll in range(1, 7):
            legal_moves = game.get_legal_moves(mover, roll)
            for depth in (1, 2, 3):
                # Exercises the exact "many consecutive no-legal-move
                # decisions" shape regardless of whether THIS position
                # happens to have one -- search() must terminate either way.
                search.search(game, roll=roll, depth=depth, evaluator=_progress_evaluator)
                checked += 1
        game.play_turn()
    print(f"✓ {checked} search() calls across depths 1-3 completed without unbounded recursion")


def test_search_never_mutates_the_real_game():
    """search() must never mutate the game it's given -- every exploration
    is applied via execute_move/next_player and undone via
    Game.restore(snapshot). Checked via a full-state fingerprint (mirrors
    parchis/tests/test_snapshot.py's approach) before/after, at every depth."""
    print("\nTesting search() never mutates the real game...")

    def fingerprint(g):
        return (
            {pos: tuple((p.color, p.piece_id) for p in pieces)
             for pos, pieces in g.board.positions.items()},
            g.board.move_counter,
            {(p.color, p.piece_id): (p.position, p.in_base, p.finished, p.move_order)
             for pl in g.players for p in pl.pieces},
            g.current_player_idx, g.turn_number, g.game_over,
            g.winner.color if g.winner is not None else None,
        )

    def relative_progress_evaluator(game, observer_seat, roll=None, pending_bonus=None, consecutive_sixes=0):
        values = np.zeros(game.num_players)
        for seat, player in enumerate(game.players):
            start = Board.STARTING_POSITIONS[player.color]
            values[seat] = sum(encoding._relative_piece_progress(p, start) for p in player.pieces) / 4.0
        return values

    random.seed(23)
    for depth in (1, 2, 3):
        game = Game(num_players=3)
        for _ in range(10):
            if game.game_over:
                break
            game.play_turn()
        if game.game_over:
            continue

        before = fingerprint(game)
        expected = copy.deepcopy(game)
        mover = game.get_current_player()
        roll = next((r for r in range(1, 7) if game.get_legal_moves(mover, r)), None)
        if roll is None:
            continue

        search.search(game, roll=roll, depth=depth, evaluator=relative_progress_evaluator)

        after = fingerprint(game)
        assert after == before, f"depth={depth}: search() mutated the game (fingerprint changed)"
        assert fingerprint(expected) == before  # sanity: `before` itself is a faithful snapshot
    print("✓ search() left the real game byte-identical at depths 1, 2, and 3")


if __name__ == '__main__':
    test_depth1_progress_value_reproduces_greedy_progress_agent()
    test_chance_node_equals_bruteforce_mean_over_6_faces()
    test_search_result_invariant_to_move_ordering()
    test_2ply_only_capture_chain_found_at_depth2_missed_at_depth1()
    test_no_legal_move_chains_do_not_recurse_without_bound()
    test_search_never_mutates_the_real_game()
    print("\nAll search tests passed!")
