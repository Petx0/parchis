#!/usr/bin/env python3
"""
Tests for parchis/evaluation/duplicate.py (docs/AGENT_REBUILD_PLAN.md §5.1 /
Phase 0 item 3): common-random-number (duplicate) match evaluation.
"""

from parchis.agents import heuristic
from parchis.evaluation import arena, duplicate


def _random_factory(game, seat, roll_box):
    player = game.players[seat]
    return lambda legal_moves: player.__class__.choose_move(player, legal_moves)


def test_duplicate_group_self_play_always_wins_exactly_once():
    """The core correctness check for the CRN mechanism itself: when
    agent_a and agent_b are the literal same policy, every game in a group
    shares the same dice seed AND the same policy on both sides, so the
    entire trajectory (dice, decisions, tie-breaks) is identical regardless
    of which seat is labeled 'a' -- meaning the SAME physical seat wins
    every rotation in the group. Since agent_a occupies each of the
    num_players seats exactly once across the group, it wins in EXACTLY
    ONE of those rotations (whichever one happened to assign it the
    seed's fixed winning seat) -- never 0, never 2+, regardless of
    num_players. If this ever failed, it would mean the seeding isn't
    actually reproducing identical conditions across the seat rotation --
    a real bug, not sampling noise."""
    print("\nTesting duplicate groups award self-play exactly one win to 'a'...")

    same_factory = heuristic.make_heuristic_agent_factory(heuristic.DEFAULT_WEIGHTS)

    for num_players in (2, 3, 4):
        for seed in range(15):
            group = duplicate.play_duplicate_group(
                same_factory, same_factory, num_players=num_players, seed=seed
            )
            assert len(set(group['winners'])) == 1, (
                f"num_players={num_players} seed={seed}: expected the SAME seat "
                f"to win every rotation under self-play, got winners={group['winners']}"
            )
            assert group['a_wins'] == 1, (
                f"num_players={num_players} seed={seed}: expected exactly 1 a_win "
                f"under self-play, got {group['a_wins']} (winners={group['winners']})"
            )
    print("✓ Self-play duplicate groups award 'a' exactly one win, at num_players in {2,3,4}")


def test_duplicate_match_pools_groups_correctly():
    """play_duplicate_match's aggregate wins_a/n_games must equal the exact
    sum over its own groups, and its CI must be wilson_score_interval of
    that same pooled (wins, n) -- no separate/drifted computation."""
    print("\nTesting play_duplicate_match pools its groups correctly...")

    from parchis.evaluation import stats as eval_stats

    tuned_factory = heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS)
    default_factory = heuristic.make_heuristic_agent_factory(heuristic.DEFAULT_WEIGHTS)

    result = duplicate.play_duplicate_match(
        tuned_factory, default_factory, n_pairs=25, num_players=2, seed=11
    )

    assert result['n_games'] == 25 * 2
    assert sum(g['a_wins'] for g in result['groups']) == result['wins_a']
    assert abs(result['win_rate_a'] - result['wins_a'] / result['n_games']) < 1e-12
    expected_ci = eval_stats.wilson_score_interval(result['wins_a'], result['n_games'])
    assert result['win_rate_a_ci'] == expected_ci

    pr = result['pair_record']
    assert pr['a_better'] + pr['split'] + pr['b_better'] == 25
    print(f"✓ play_duplicate_match pools correctly: {result['wins_a']}/{result['n_games']}, "
          f"pair_record={pr}")


def test_variance_reduction_self_play_is_the_zero_variance_edge_case():
    """Documents the degenerate case explicitly: since every self-play
    duplicate GROUP awards 'a' exactly one win regardless of seed (per the
    test above), the pooled win_rate_a is the same constant (1/num_players)
    on every repeat -- zero empirical variance across repeats -- while the
    independent-seed method still shows its usual sampling spread."""
    print("\nTesting measure_variance_reduction's self-play edge case...")

    same_factory = heuristic.make_heuristic_agent_factory(heuristic.DEFAULT_WEIGHTS)
    result = duplicate.measure_variance_reduction(
        same_factory, same_factory, n_pairs=8, num_players=2, repeats=8, seed=3
    )
    assert abs(result['duplicate_mean_win_rate'] - 0.5) < 1e-9
    assert result['duplicate_std'] == 0.0
    assert result['effective_n_multiplier'] == float('inf')
    assert result['independent_std'] > 0.0, (
        "Expected the independent-seed method to show real sampling spread "
        "across repeats, even under self-play"
    )
    print(f"✓ Self-play: duplicate_std=0.0 exactly, independent_std={result['independent_std']:.4f}, "
          f"multiplier=inf")


def test_variance_reduction_between_distinct_agents_is_finite_and_positive():
    """The useful case: two distinct, comparably-matched agents (TUNED_WEIGHTS
    vs DEFAULT_WEIGHTS) must show a REAL, finite, positive variance
    reduction -- the duplicate method's win_rate_a must vary less across
    repeats than the independent-seed method's does, over the same
    per-repeat game budget, giving an effective-n multiplier >= 1."""
    print("\nTesting measure_variance_reduction between two distinct agents...")

    tuned_factory = heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS)
    default_factory = heuristic.make_heuristic_agent_factory(heuristic.DEFAULT_WEIGHTS)

    result = duplicate.measure_variance_reduction(
        tuned_factory, default_factory, n_pairs=25, num_players=2, repeats=16, seed=123
    )
    print(f"  n_games/repeat={result['n_games_per_repeat']} repeats={result['repeats']} "
          f"duplicate_std={result['duplicate_std']:.4f} independent_std={result['independent_std']:.4f} "
          f"effective_n_multiplier={result['effective_n_multiplier']:.2f}x")

    assert 0.0 < result['duplicate_std'] <= result['independent_std'], (
        "Expected the duplicate method's win_rate to vary no more than the "
        "independent-seed method's across repeats"
    )
    assert result['effective_n_multiplier'] >= 1.0
    print(f"✓ Duplicate pairing measured a real {result['effective_n_multiplier']:.2f}x "
          f"effective-n multiplier over independent-seed matches")


if __name__ == '__main__':
    test_duplicate_group_self_play_always_wins_exactly_once()
    test_duplicate_match_pools_groups_correctly()
    test_variance_reduction_self_play_is_the_zero_variance_edge_case()
    test_variance_reduction_between_distinct_agents_is_finite_and_positive()
    print("\nAll duplicate-match tests passed!")
