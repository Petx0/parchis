#!/usr/bin/env python3
"""
Tests for parchis/az/selfplay.py (docs/AGENT_REBUILD_PLAN.md Part 3 item
11): game generation + training-target construction for Phase 2's
bootstrap. Named test_selfplay_generation.py (not test_selfplay.py) to
avoid colliding with the existing SB3-era parchis/tests/test_selfplay.py.
"""

import numpy as np

from parchis.az import encoding, selfplay


def test_outcomes_are_valid_one_hot_or_draw_vectors():
    """Every recorded example's outcome must be a probability vector: a
    one-hot on the winner, or an exact 1/num_players draw vector."""
    print("\nTesting recorded outcomes are valid one-hot/draw vectors...")

    pool = selfplay.default_pool_factories(noisy_seed=1)
    examples, stats = selfplay.generate_games(pool, n_games=15, num_players=2,
                                               max_turns=500, seed=7)
    assert examples, "Test setup error: expected at least one recorded decision"

    for ex in examples:
        outcome = ex['outcome']
        assert outcome.shape == (2,)
        assert abs(outcome.sum() - 1.0) < 1e-6
        is_one_hot = np.any(np.isclose(outcome, 1.0)) and np.any(np.isclose(outcome, 0.0))
        is_draw = np.allclose(outcome, 0.5)
        assert is_one_hot or is_draw, f"Unexpected outcome vector: {outcome}"

    print(f"✓ All {len(examples)} recorded outcomes across {stats['n_games']} games "
          f"are valid one-hot/draw vectors")


def _implied_absolute_winner(example, num_players):
    """Undo an example's mover-relative outcome encoding back to an
    absolute winner seat (or None for a draw) -- the exact inverse of
    generate_games' own np.roll(absolute_outcome, -mover_seat), used here
    only to CHECK that inverse is self-consistent across a whole game."""
    outcome = example['outcome']
    if np.allclose(outcome, 1.0 / num_players):
        return None
    absolute_outcome = np.roll(outcome, example['mover_seat'])
    return int(np.argmax(absolute_outcome))


def test_outcome_backfill_matches_actual_winner_per_game():
    """The bookkeeping that backfills 'outcome' once each game concludes
    must scope correctly to THAT game's own decisions -- not bleed into
    the next game's -- AND each example's outcome must be correctly
    expressed in ITS OWN mover-relative order (not a single shared vector
    reused verbatim): every example in the same game, when rolled back to
    absolute order using its OWN mover_seat, must imply the SAME absolute
    winner (or all imply a draw)."""
    print("\nTesting outcome backfill is correctly scoped and mover-relative per game...")

    pool = selfplay.default_pool_factories(noisy_seed=2)
    examples, stats = selfplay.generate_games(pool, n_games=25, num_players=2,
                                               max_turns=500, seed=13)

    implied_winner_by_game = {}
    for ex in examples:
        implied = _implied_absolute_winner(ex, num_players=2)
        implied_winner_by_game.setdefault(ex['game_index'], set()).add(implied)

    assert len(implied_winner_by_game) == stats['n_games']
    for game_index, implied_winners in implied_winner_by_game.items():
        assert len(implied_winners) == 1, (
            f"game {game_index}: examples imply inconsistent absolute winners "
            f"{implied_winners} once rolled back by their own mover_seat"
        )

    decisive_games = sum(1 for w in implied_winner_by_game.values() if next(iter(w)) is not None)
    assert decisive_games == sum(stats['n_by_winner_seat'].values())
    assert decisive_games + stats['n_truncated'] == stats['n_games']
    print(f"✓ Outcome backfill correctly scoped and mover-relative: {len(implied_winner_by_game)} "
          f"games, {decisive_games} decisive + {stats['n_truncated']} truncated == {stats['n_games']}")


def test_game_index_covers_every_game_with_a_consistent_implied_winner():
    """Each example's 'game_index' must range over 0..n_games-1 with no
    gaps (needed for a correct game-level train/val/test split downstream,
    parchis.az.train.split_by_game), and, per game, every example's
    mover-relative outcome must roll back to the SAME absolute winner."""
    print("\nTesting game_index covers every game with a consistent implied winner...")

    pool = selfplay.default_pool_factories(noisy_seed=6)
    examples, stats = selfplay.generate_games(pool, n_games=20, num_players=2,
                                               max_turns=500, seed=21)

    implied_winner_by_game = {}
    for ex in examples:
        implied_winner_by_game.setdefault(ex['game_index'], set()).add(
            _implied_absolute_winner(ex, num_players=2)
        )
    assert set(implied_winner_by_game.keys()) == set(range(stats['n_games']))
    for game_index, implied_winners in implied_winner_by_game.items():
        assert len(implied_winners) == 1, (
            f"game {game_index}: inconsistent implied winners {implied_winners}"
        )
    print(f"✓ game_index correctly covers 0..{stats['n_games'] - 1} with a consistent "
          f"implied winner per game")


def test_outcome_is_mover_relative_not_absolute():
    """Precise, hand-verified regression test for the mover-relative
    convention itself (the exact bug found while sizing item 13's gate:
    'outcome' was stored in ABSOLUTE seat order while the encoding it
    trains against is mover-relative, training the net on a mismatched,
    unlearnable pair). At num_players=3, seat 0's own decisions must show
    outcome[0]==1.0 when seat 0 wins; seat 1's own decisions must ALSO
    show outcome[0]==1.0 for that SAME game (their own perspective, "did I
    win"), even though seat 1 is not the winner -- i.e. outcome[0] always
    means "did THIS decision's mover win", never "did seat 0 win"."""
    print("\nTesting outcome is stored mover-relative, not absolute...")

    pool = selfplay.default_pool_factories(noisy_seed=9)
    examples, stats = selfplay.generate_games(pool, n_games=60, num_players=3,
                                               max_turns=500, seed=17)
    decisive = [ex for ex in examples if not np.allclose(ex['outcome'], 1.0 / 3.0)]
    assert decisive, "Test setup error: expected at least one decisive game's decisions"

    # Verify against the independently-recomputed absolute winner, across
    # games spanning all 3 seats as movers.
    by_game = {}
    for ex in examples:
        by_game.setdefault(ex['game_index'], []).append(ex)
    checked_seats = set()
    for game_examples in by_game.values():
        outcome0 = game_examples[0]['outcome']
        if np.allclose(outcome0, 1.0 / 3.0):
            continue
        absolute_winner = int(np.roll(outcome0, game_examples[0]['mover_seat']).argmax())
        for ex in game_examples:
            expected0 = 1.0 if ex['mover_seat'] == absolute_winner else 0.0
            assert np.isclose(ex['outcome'][0], expected0), (
                f"mover_seat={ex['mover_seat']} absolute_winner={absolute_winner}: "
                f"expected outcome[0]={expected0}, got {ex['outcome'][0]}"
            )
            checked_seats.add(ex['mover_seat'])
    assert checked_seats == {0, 1, 2}, (
        f"Expected to check decisions from all 3 seats as mover, only saw {checked_seats}"
    )
    print(f"✓ outcome[0] correctly means 'did THIS decision's mover win', "
          f"verified across movers {sorted(checked_seats)}")


def test_truncated_game_scored_as_draw_not_zero():
    """Part 4's explicit truncation rule: hitting max_turns without a
    winner must score every decision from that game as a draw vector
    (1/num_players each), never silently as zero."""
    print("\nTesting a truncated game is scored as a draw, not zero...")

    pool = selfplay.default_pool_factories(noisy_seed=3)
    # max_turns=2 makes truncation essentially certain (a real game needs
    # ~140+ turns on average -- see docs/AGENT_REBUILD_PLAN.md §1.1).
    examples, stats = selfplay.generate_games(pool, n_games=5, num_players=2,
                                               max_turns=2, seed=1)

    assert stats['n_truncated'] == 5, (
        f"Expected all 5 games to truncate at max_turns=2, got n_truncated={stats['n_truncated']}"
    )
    assert examples, "Test setup error: expected at least one recorded decision before truncation"
    for ex in examples:
        assert np.allclose(ex['outcome'], 0.5), (
            f"Truncated game's decisions must be scored as a draw vector, got {ex['outcome']}"
        )
    print(f"✓ All {len(examples)} decisions from {stats['n_truncated']} truncated games "
          f"scored as an exact draw vector")


def test_generate_games_never_confuses_bonus_with_a_fresh_roll(monkeypatch):
    """Same regression shape as test_agent.py's: instruments
    encoding.encode itself (selfplay's recording layer is the ONLY caller
    during generate_games) and confirms every call resolving a bonus has
    pending_bonus set and roll=None, across real games with real bonus
    chains -- the §1.4 bug this whole design avoids by construction."""
    print("\nTesting selfplay generation never confuses a bonus with a fresh roll...")

    calls = []
    original_encode = encoding.encode

    def recording_encode(game, observer_seat, roll=None, pending_bonus=None, consecutive_sixes=0):
        calls.append((roll, pending_bonus))
        return original_encode(game, observer_seat, roll=roll, pending_bonus=pending_bonus,
                                consecutive_sixes=consecutive_sixes)

    monkeypatch.setattr(encoding, "encode", recording_encode)
    monkeypatch.setattr("parchis.az.selfplay.encoding.encode", recording_encode)

    pool = selfplay.default_pool_factories(noisy_seed=4)
    _examples, stats = selfplay.generate_games(pool, n_games=40, num_players=2,
                                                max_turns=400, seed=42)

    bonus_calls = 0
    for roll, pending_bonus in calls:
        if pending_bonus is not None:
            bonus_calls += 1
            assert roll is None, f"Bonus decision incorrectly carried roll={roll}"
        else:
            assert roll is not None, "A non-bonus decision must have a real roll"

    assert bonus_calls > 0, "Expected at least one bonus decision across 40 games"
    assert len(calls) == stats['n_decisions']
    print(f"✓ {bonus_calls}/{len(calls)} decisions were bonus decisions, all correctly "
          f"had roll=None and a real pending_bonus")


def test_examples_to_arrays_shapes_and_values():
    """examples_to_arrays must produce dense arrays with the right shapes
    and dtypes, whose rows match the source examples exactly."""
    print("\nTesting examples_to_arrays produces correctly-shaped arrays...")

    pool = selfplay.default_pool_factories(noisy_seed=5)
    examples, _stats = selfplay.generate_games(pool, n_games=10, num_players=2,
                                                max_turns=400, seed=3)
    X, policy_targets, value_targets = selfplay.examples_to_arrays(examples, num_players=2)

    n = len(examples)
    assert X.shape == (n, encoding.encoding_size(2))
    assert policy_targets.shape == (n,)
    assert value_targets.shape == (n, 2)
    assert X.dtype == np.float32 and value_targets.dtype == np.float32
    assert policy_targets.dtype == np.int64

    for i in (0, n // 2, n - 1):
        assert np.array_equal(X[i], examples[i]['encoding'])
        assert policy_targets[i] == examples[i]['chosen_piece_id']
        assert np.array_equal(value_targets[i], examples[i]['outcome'])
        assert 0 <= policy_targets[i] <= 3
    print(f"✓ examples_to_arrays correctly stacked {n} examples")


if __name__ == '__main__':
    test_outcomes_are_valid_one_hot_or_draw_vectors()
    test_outcome_backfill_matches_actual_winner_per_game()
    test_game_index_covers_every_game_with_a_consistent_implied_winner()
    test_outcome_is_mover_relative_not_absolute()
    test_truncated_game_scored_as_draw_not_zero()
    test_generate_games_never_confuses_bonus_with_a_fresh_roll()
    test_examples_to_arrays_shapes_and_values()
    print("\nAll selfplay generation tests passed!")
