#!/usr/bin/env python3
"""
Unit tests for parchis/evaluation/stats.py -- confidence-interval helpers
introduced in docs/RL_DESIGN_REVIEW.md Phase 4 (win rate is a Bernoulli
proportion; before this module, every evaluation entry point in the repo
reported only a bare point estimate).
"""
import pytest

from parchis.evaluation import stats


# ── wilson_score_interval ─────────────────────────────────────────────────

def test_wilson_score_interval_known_value_5_of_10():
    # Reference value for wins=5, n=10, 95% confidence (hand-computed /
    # cross-checked against a standard Wilson interval calculator):
    # approximately (0.2366, 0.7634).
    lower, upper = stats.wilson_score_interval(5, 10, confidence=0.95)
    assert lower == pytest.approx(0.2366, abs=1e-3)
    assert upper == pytest.approx(0.7634, abs=1e-3)


def test_wilson_score_interval_contains_point_estimate():
    lower, upper = stats.wilson_score_interval(7, 20)
    assert lower <= 7 / 20 <= upper


def test_wilson_score_interval_bounds_within_0_and_1():
    for wins, n in [(0, 10), (10, 10), (1, 1), (0, 1)]:
        lower, upper = stats.wilson_score_interval(wins, n)
        assert 0.0 <= lower <= upper <= 1.0


def test_wilson_score_interval_widens_with_smaller_n():
    lower_small, upper_small = stats.wilson_score_interval(5, 10)
    lower_large, upper_large = stats.wilson_score_interval(50, 100)
    assert (upper_small - lower_small) > (upper_large - lower_large)


def test_wilson_score_interval_perfect_record():
    # A 10/10 record's Wilson interval is exactly [something > 0, 1.0] --
    # the upper bound mathematically reaches 1.0 at p_hat=1 (unlike the
    # normal approximation, which would wrongly claim a zero-width
    # interval at exactly 100%); the lower bound still reflects real
    # uncertainty from the small sample size.
    lower, upper = stats.wilson_score_interval(10, 10)
    assert lower > 0.0
    assert upper == pytest.approx(1.0)


def test_wilson_score_interval_invalid_n_raises():
    with pytest.raises(ValueError):
        stats.wilson_score_interval(0, 0)


def test_wilson_score_interval_wins_out_of_range_raises():
    with pytest.raises(ValueError):
        stats.wilson_score_interval(11, 10)
    with pytest.raises(ValueError):
        stats.wilson_score_interval(-1, 10)


def test_wilson_score_interval_invalid_confidence_raises():
    with pytest.raises(ValueError):
        stats.wilson_score_interval(5, 10, confidence=1.0)
    with pytest.raises(ValueError):
        stats.wilson_score_interval(5, 10, confidence=0.0)


# ── mean_confidence_interval ──────────────────────────────────────────────

def test_mean_confidence_interval_contains_mean():
    values = [0.5, 0.6, 0.55]
    lower, upper = stats.mean_confidence_interval(values)
    mean = sum(values) / len(values)
    assert lower <= mean <= upper


def test_mean_confidence_interval_zero_variance_is_a_point():
    lower, upper = stats.mean_confidence_interval([0.5, 0.5, 0.5])
    assert lower == pytest.approx(0.5)
    assert upper == pytest.approx(0.5)


def test_mean_confidence_interval_widens_with_more_spread():
    tight = stats.mean_confidence_interval([0.50, 0.51, 0.49])
    wide = stats.mean_confidence_interval([0.10, 0.90, 0.50])
    tight_width = tight[1] - tight[0]
    wide_width = wide[1] - wide[0]
    assert wide_width > tight_width


def test_mean_confidence_interval_too_few_values_raises():
    with pytest.raises(ValueError):
        stats.mean_confidence_interval([0.5])
    with pytest.raises(ValueError):
        stats.mean_confidence_interval([])


def test_mean_confidence_interval_invalid_confidence_raises():
    with pytest.raises(ValueError):
        stats.mean_confidence_interval([0.1, 0.2], confidence=1.5)


# ── intervals_overlap ─────────────────────────────────────────────────────

def test_intervals_overlap_true_when_overlapping():
    assert stats.intervals_overlap((0.2, 0.5), (0.4, 0.7))


def test_intervals_overlap_false_when_disjoint():
    assert not stats.intervals_overlap((0.2, 0.3), (0.4, 0.5))


def test_intervals_overlap_true_when_touching():
    assert stats.intervals_overlap((0.2, 0.4), (0.4, 0.6))


def test_intervals_overlap_true_when_nested():
    assert stats.intervals_overlap((0.1, 0.9), (0.4, 0.5))


def test_intervals_overlap_symmetric():
    a, b = (0.2, 0.5), (0.4, 0.7)
    assert stats.intervals_overlap(a, b) == stats.intervals_overlap(b, a)


# ── breakdown_win_rates ───────────────────────────────────────────────────

def test_breakdown_win_rates_computes_per_key_rate_and_ci():
    wins_by_key = {0: 8, 1: 2}
    games_by_key = {0: 10, 1: 10}
    breakdown = stats.breakdown_win_rates(wins_by_key, games_by_key)
    assert breakdown[0]['win_rate'] == pytest.approx(0.8)
    assert breakdown[1]['win_rate'] == pytest.approx(0.2)
    assert breakdown[0]['n'] == 10
    assert breakdown[0]['ci'][0] <= 0.8 <= breakdown[0]['ci'][1]


def test_breakdown_win_rates_skips_zero_game_keys():
    breakdown = stats.breakdown_win_rates({0: 0, 1: 3}, {0: 0, 1: 5})
    assert 0 not in breakdown
    assert 1 in breakdown


def test_breakdown_win_rates_missing_wins_key_defaults_to_zero():
    breakdown = stats.breakdown_win_rates({}, {0: 4})
    assert breakdown[0]['win_rate'] == 0.0


# ── aggregate_phase4_stats ────────────────────────────────────────────────

def test_aggregate_phase4_stats_returns_expected_keys():
    result = stats.aggregate_phase4_stats(
        wins=6, n_episodes=10,
        wins_by_seat={0: 3, 1: 3}, games_by_seat={0: 5, 1: 5},
        wins_by_color={'RED': 3, 'YELLOW': 3}, games_by_color={'RED': 5, 'YELLOW': 5},
        captures_by_agent=[1, 0, 2], captures_against_agent=[0, 1, 0],
        legal_moves_counts=[1, 2, 3, 2], bonus_chain_lengths=[1, 2],
        three_sixes_penalty_count=2,
    )
    assert result['win_rate_ci'][0] <= 0.6 <= result['win_rate_ci'][1]
    assert set(result['win_rate_by_seat'].keys()) == {0, 1}
    assert set(result['win_rate_by_color'].keys()) == {'RED', 'YELLOW'}
    assert result['capture_rate'] == pytest.approx(3 / 10)
    assert result['capture_rate_against'] == pytest.approx(1 / 10)
    assert result['three_sixes_penalty_rate'] == pytest.approx(2 / 10)
    assert result['mean_legal_moves_count'] == pytest.approx(2.0)
    assert result['mean_bonus_chain_length'] == pytest.approx(1.5)


def test_aggregate_phase4_stats_omits_legal_moves_and_chain_keys_when_empty():
    result = stats.aggregate_phase4_stats(
        wins=1, n_episodes=2,
        wins_by_seat={0: 1}, games_by_seat={0: 2},
        wins_by_color={'RED': 1}, games_by_color={'RED': 2},
        captures_by_agent=[0, 0], captures_against_agent=[0, 0],
        legal_moves_counts=[], bonus_chain_lengths=[],
        three_sixes_penalty_count=0,
    )
    assert 'mean_legal_moves_count' not in result
    assert 'std_legal_moves_count' not in result
    assert 'mean_bonus_chain_length' not in result
    assert 'std_bonus_chain_length' not in result
    assert result['capture_rate'] == 0.0
    assert result['three_sixes_penalty_rate'] == 0.0


# ── rank_by_mean_with_ci ──────────────────────────────────────────────────

def test_rank_by_mean_with_ci_sorts_descending():
    entries = [("a", 0.5, (0.4, 0.6)), ("b", 0.8, (0.7, 0.9)), ("c", 0.2, (0.1, 0.3))]
    ranked, _ = stats.rank_by_mean_with_ci(entries)
    assert [label for label, _, _ in ranked] == ["b", "a", "c"]


def test_rank_by_mean_with_ci_confirmed_when_non_overlapping():
    entries = [("a", 0.8, (0.7, 0.9)), ("b", 0.3, (0.2, 0.4))]
    _, confirmed = stats.rank_by_mean_with_ci(entries)
    assert confirmed is True


def test_rank_by_mean_with_ci_not_confirmed_when_overlapping():
    entries = [("a", 0.55, (0.4, 0.7)), ("b", 0.50, (0.35, 0.65))]
    _, confirmed = stats.rank_by_mean_with_ci(entries)
    assert confirmed is False


def test_rank_by_mean_with_ci_not_confirmed_when_ci_missing():
    entries = [("a", 0.8, None), ("b", 0.3, (0.2, 0.4))]
    _, confirmed = stats.rank_by_mean_with_ci(entries)
    assert confirmed is False


def test_rank_by_mean_with_ci_single_entry_is_trivially_confirmed():
    ranked, confirmed = stats.rank_by_mean_with_ci([("only", 0.5, None)])
    assert ranked == [("only", 0.5, None)]
    assert confirmed is True


def test_rank_by_mean_with_ci_empty_returns_empty_and_unconfirmed():
    ranked, confirmed = stats.rank_by_mean_with_ci([])
    assert ranked == []
    assert confirmed is False


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
