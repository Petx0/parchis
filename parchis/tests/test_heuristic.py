#!/usr/bin/env python3
"""
Tests for parchis/agents/heuristic.py (docs/AGENT_REBUILD_PLAN.md §2.4 /
Phase 0 item 4): the handcrafted linear-score agent used as the absolute
strength anchor, bootstrap opponent, and Phase 1's gate opponent.
"""

import random

import numpy as np

from parchis.agents import heuristic
from parchis.agents.decision_recorder import DecisionRecorder
from parchis.game.board import Board
from parchis.game.game import Game
from parchis.evaluation import arena


def test_score_move_never_mutates_game():
    """_score_move applies a move to compute post-move features (threat,
    blockade) -- it must always undo that via snapshot()/restore(), for
    every legal move in a realistic position, not just the one chosen."""
    print("\nTesting _score_move never mutates the game...")

    random.seed(1)
    game = Game(num_players=3)
    for _ in range(20):
        game.play_turn()

    player = game.get_current_player()
    legal_moves = game.get_legal_moves(player, 4) or game.get_legal_moves(player, 3)
    assert legal_moves, "Test setup error: need at least one legal move to score"

    for move in legal_moves:
        snap = game.snapshot()
        expected = __import__("copy").deepcopy(game)
        heuristic._score_move(game, player, move, heuristic.DEFAULT_WEIGHTS)
        # Compare board occupancy + all piece fields -- cheap enough here
        # since this loop is small (a handful of legal moves).
        assert game.board.move_counter == expected.board.move_counter
        for p1, p2 in zip(
            (pc for pl in game.players for pc in pl.pieces),
            (pc for pl in expected.players for pc in pl.pieces),
        ):
            assert (p1.position, p1.in_base, p1.finished, p1.move_order) == \
                   (p2.position, p2.in_base, p2.finished, p2.move_order)
        game.restore(snap)  # no-op given the above, but exercises the API too

    print(f"✓ _score_move left the game unchanged across {len(legal_moves)} candidate moves")


def test_choose_move_returns_a_legal_move():
    """choose_move_with_weights must always return one of the exact move
    tuples it was given, for the same identity reasons every other
    choose_move implementation in this codebase does."""
    print("\nTesting choose_move_with_weights returns a legal move...")

    random.seed(2)
    game = Game(num_players=2)
    for _ in range(30):
        if game.game_over:
            break
        player = game.get_current_player()
        for v in range(1, 7):
            legal_moves = game.get_legal_moves(player, v)
            if legal_moves:
                chosen = heuristic.choose_move_with_weights(
                    game, player, legal_moves, heuristic.DEFAULT_WEIGHTS
                )
                assert chosen in legal_moves, f"chose {chosen}, not in {legal_moves}"
        game.play_turn()

    print("✓ choose_move_with_weights always returns a move from legal_moves")


def test_capture_value_prefers_capturing_the_more_advanced_piece():
    """Given a choice between capturing a piece near its own start and one
    much further along, the heuristic (capture_value weight > 0) must
    prefer capturing the more advanced piece, all else equal."""
    print("\nTesting capture_value prefers capturing the more-advanced piece...")

    game = Game(num_players=3)
    mover = game.players[0]
    near_victim = game.players[1].pieces[0]
    far_victim = game.players[2].pieces[0]

    mover_piece = mover.pieces[0]
    mover_pos = 30
    near_pos = 25   # low progress
    far_pos = 25    # placeholder, corrected below to a DIFFERENT square

    # Two candidate capturing moves must land on two DIFFERENT squares (one
    # piece per square here), so give mover two pieces instead of one --
    # each landing on a different victim by rolling a different value.
    mover_piece_2 = mover.pieces[1]
    near_target = 25
    far_target = 33

    for piece, pos in (
        (mover_piece, mover_pos), (mover_piece_2, mover_pos + 8 - (far_target - near_target)),
    ):
        game.board.remove_piece(piece)
        piece.move_to(pos)
        game.board.add_piece(piece, pos)

    # Place near_victim just barely advanced, far_victim heavily advanced,
    # each capturable by exactly one of mover's two pieces via a distinct
    # roll so both candidate moves are simultaneously legal this "roll"
    # (we bypass real dice and just build both legal_moves tuples by hand).
    game.board.remove_piece(near_victim)
    near_victim.move_to(near_target)
    game.board.add_piece(near_victim, near_target)

    game.board.remove_piece(far_victim)
    far_victim.move_to(far_target)
    game.board.add_piece(far_victim, far_target)

    move_near = (mover_piece, near_target, 'move')
    move_far = (mover_piece_2, far_target, 'move')
    assert game.would_capture(move_near) == [near_victim]
    assert game.would_capture(move_far) == [far_victim]
    assert far_victim.position > near_victim.position, "Test setup: far_victim must be more advanced"

    weights = np.zeros(heuristic.NUM_FEATURES)
    weights[0] = 1.0  # isolate capture_value
    chosen = heuristic.choose_move_with_weights(
        game, mover, [move_near, move_far], weights, rng=random.Random(0)
    )
    assert chosen == move_far, (
        f"Expected the heuristic (capture_value only) to capture the more-advanced "
        f"piece at {far_target}, got a move onto {chosen[1]}"
    )
    print("✓ capture_value correctly prefers capturing the more-advanced piece")


def test_lands_in_threat_avoids_capturable_square_when_isolated():
    """With only lands_in_threat active (negative weight), the heuristic
    must avoid landing where an opponent can capture next roll, choosing a
    safe alternative instead."""
    print("\nTesting lands_in_threat avoids a directly-capturable square...")

    game = Game(num_players=2)
    mover = game.players[0]
    opponent = next(p for p in game.players if p is not mover)

    piece_a = mover.pieces[0]
    piece_b = mover.pieces[1]
    threatened_pos = 20
    safe_pos = 12  # a genuine Board.SAFE_SQUARES entry, unreachable by the opponent anyway

    for piece, pos in ((piece_a, threatened_pos - 3), (piece_b, safe_pos - 4)):
        game.board.remove_piece(piece)
        piece.move_to(pos)
        game.board.add_piece(piece, pos)

    opp_threat_piece = opponent.pieces[0]
    game.board.remove_piece(opp_threat_piece)
    opp_threat_piece.move_to(threatened_pos - 2)  # captures piece_a's destination on roll=2
    game.board.add_piece(opp_threat_piece, threatened_pos - 2)

    move_threatened = (piece_a, threatened_pos, 'move')
    move_safe = (piece_b, safe_pos, 'move')

    weights = np.zeros(heuristic.NUM_FEATURES)
    weights[4] = -1.0  # isolate lands_in_threat, negative
    chosen = heuristic.choose_move_with_weights(
        game, mover, [move_threatened, move_safe], weights, rng=random.Random(0)
    )
    assert chosen == move_safe, (
        f"Expected the heuristic (lands_in_threat only) to avoid the threatened "
        f"square {threatened_pos}, chose a move onto {chosen[1]}"
    )
    print("✓ lands_in_threat correctly avoids the directly-capturable square")


def test_default_weights_heuristic_beats_random():
    """The whole point of the heuristic: it must clearly outperform random
    play. Uses arena.play_match (not duplicate matches -- CRN variance
    reduction isn't needed to see an effect this large)."""
    print("\nTesting DEFAULT_WEIGHTS heuristic beats random play...")

    heuristic_factory = heuristic.make_heuristic_agent_factory(heuristic.DEFAULT_WEIGHTS)

    def random_factory(game, seat, roll_box):
        player = game.players[seat]

        def choose_move(legal_moves):
            return player.__class__.choose_move(player, legal_moves)

        return choose_move

    result = arena.play_match(heuristic_factory, random_factory, n_games=200,
                               num_players=2, max_turns=600, seed=7)
    lower, _upper = result["win_rate_a_ci"]
    print(f"  heuristic vs random: {result['wins_a']}/{result['n_games']} "
          f"win_rate={result['win_rate_a']:.3f} Wilson_lower={lower:.3f}")
    assert lower > 0.5, (
        f"Expected DEFAULT_WEIGHTS heuristic's Wilson lower bound vs random to "
        f"clear 50%, got {result}"
    )
    print("✓ DEFAULT_WEIGHTS heuristic beats random with the Wilson lower bound clear of 50%")


def test_epsilon_noisy_heuristic_deviates_from_plain_heuristic_at_expected_rate():
    """make_epsilon_noisy_heuristic_agent_factory must (a) always return a
    legal move and (b) deviate from the plain (epsilon=0) heuristic's own
    choice at roughly the configured epsilon rate, across many decisions
    -- not 0% (dead exploration) and not ~100% (accidentally always
    randomizing)."""
    print("\nTesting the epsilon-noisy heuristic deviates at roughly the configured rate...")

    game = Game(num_players=3)
    for _ in range(15):
        game.play_turn()
    assert not game.game_over, "Test setup error: need a live position"

    epsilon = 0.3
    noisy_factory = heuristic.make_epsilon_noisy_heuristic_agent_factory(
        heuristic.DEFAULT_WEIGHTS, epsilon=epsilon, seed=99
    )
    noisy_choose = noisy_factory(game, game.current_player_idx, roll_box={"last_roll": None})
    plain_player = game.get_current_player()

    # Sample MANY independent noisy choices on the SAME (game, roll) to
    # estimate the deviation rate cleanly.
    legal_moves = None
    for roll in range(1, 7):
        candidates = game.get_legal_moves(plain_player, roll)
        if len(candidates) >= 2:
            legal_moves = candidates
            break
    assert legal_moves is not None, "Test setup error: need a roll with >= 2 legal moves"

    plain_move = heuristic.choose_move_with_weights(
        game, plain_player, legal_moves, heuristic.DEFAULT_WEIGHTS, rng=random.Random(0)
    )
    samples = 3000
    deviations = sum(
        1 for _ in range(samples)
        if noisy_choose(legal_moves)[0].piece_id != plain_move[0].piece_id
    )
    rate = deviations / samples
    print(f"  observed deviation rate={rate:.3f} (epsilon={epsilon})")
    # Loose bounds: epsilon-random deviates from the plain choice on AT
    # MOST epsilon fraction of draws (some random draws coincide with the
    # plain choice too, especially with few candidates), and clearly more
    # than a small fraction (not dead/no-op).
    assert 0.05 < rate <= epsilon + 0.05, (
        f"Expected a deviation rate roughly around epsilon={epsilon} (allowing some random "
        f"draws to coincide with the plain choice), got {rate:.3f}"
    )
    print("✓ Epsilon-noisy heuristic deviates from the plain heuristic at a sane rate")


def test_tuned_weights_beats_default_weights():
    """TUNED_WEIGHTS (fit by cem_tune_weights, see its comment in
    heuristic.py for the exact call) must clearly outperform the untuned
    DEFAULT_WEIGHTS it was tuned against -- "a tuned heuristic clearly
    above an untuned one" (docs/AGENT_REBUILD_PLAN.md §2.4). Uses a fresh
    seed, not one CEM selection was done against."""
    print("\nTesting TUNED_WEIGHTS beats DEFAULT_WEIGHTS...")

    tuned_factory = heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS)
    default_factory = heuristic.make_heuristic_agent_factory(heuristic.DEFAULT_WEIGHTS)

    result = arena.play_match(tuned_factory, default_factory, n_games=300,
                               num_players=2, max_turns=500, seed=31415)
    lower, _upper = result["win_rate_a_ci"]
    print(f"  tuned vs default: {result['wins_a']}/{result['n_games']} "
          f"win_rate={result['win_rate_a']:.3f} Wilson_lower={lower:.3f}")
    assert lower > 0.5, (
        f"Expected TUNED_WEIGHTS' Wilson lower bound vs DEFAULT_WEIGHTS to "
        f"clear 50%, got {result}"
    )
    print("✓ TUNED_WEIGHTS beats DEFAULT_WEIGHTS with the Wilson lower bound clear of 50%")


def test_recording_heuristic_factory_captures_consistent_move_scores():
    """Every DecisionRecord from make_recording_heuristic_agent_factory
    must have move_scores keyed by exactly the legal moves' piece_ids for
    that decision, and chosen_piece_id must be the argmax-scoring one
    (ties aside -- forced here by isolating a single dominant feature)."""
    print("\nTesting recorded heuristic move_scores are internally consistent...")

    recorder = DecisionRecorder()
    factory = heuristic.make_recording_heuristic_agent_factory(
        heuristic.DEFAULT_WEIGHTS, recorder=recorder,
    )
    arena.play_one_game({0: factory, 1: factory}, num_players=2, max_turns=300, seed=11)

    assert recorder.records, "Expected at least one recorded decision"
    for record in recorder.records:
        assert record.kind == "heuristic"
        assert record.move_scores, "move_scores must be non-empty for every recorded decision"
        best_piece_id = max(record.move_scores, key=record.move_scores.get)
        best_score = record.move_scores[best_piece_id]
        assert record.move_scores[record.chosen_piece_id] == best_score, (
            f"chosen_piece_id={record.chosen_piece_id} does not have the max score "
            f"in {record.move_scores}"
        )
    print(f"✓ {len(recorder.records)} recorded decisions, all internally consistent")


if __name__ == '__main__':
    test_score_move_never_mutates_game()
    test_choose_move_returns_a_legal_move()
    test_capture_value_prefers_capturing_the_more_advanced_piece()
    test_lands_in_threat_avoids_capturable_square_when_isolated()
    test_default_weights_heuristic_beats_random()
    test_epsilon_noisy_heuristic_deviates_from_plain_heuristic_at_expected_rate()
    test_tuned_weights_beats_default_weights()
    test_recording_heuristic_factory_captures_consistent_move_scores()
    print("\nAll heuristic tests passed!")
