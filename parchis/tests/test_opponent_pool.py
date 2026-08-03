#!/usr/bin/env python3
"""
Unit tests for parchis/rl/opponent_pool.py -- the pure sampling/weighting
functions behind the self-play opponent pool (see
parchis/tests/test_selfplay.py and test_selfplay_pool_callback.py for
end-to-end coverage of how they're wired into ParchisSelfPlayEnv and
SelfPlayCallback).
"""
import random

import pytest

from parchis.rl import opponent_pool


# ── compute_recency_weights ──────────────────────────────────────────────

def test_compute_recency_weights_monotonic_increasing():
    weights = opponent_pool.compute_recency_weights(5)
    assert weights == [1.0, 2.0, 3.0, 4.0, 5.0]
    for i in range(len(weights) - 1):
        assert weights[i] < weights[i + 1]


def test_compute_recency_weights_newest_last_gets_highest_weight():
    weights = opponent_pool.compute_recency_weights(3)
    assert weights[-1] == max(weights)


def test_compute_recency_weights_single_element():
    assert opponent_pool.compute_recency_weights(1) == [1.0]


def test_compute_recency_weights_invalid_n_raises():
    with pytest.raises(ValueError):
        opponent_pool.compute_recency_weights(0)


# ── compute_win_rate_weights ─────────────────────────────────────────────

def test_compute_win_rate_weights_favors_lower_win_rate():
    weights = opponent_pool.compute_win_rate_weights([0.9, 0.1])
    # The opponent with the LOWER win-rate-against-us (harder to beat is
    # win_rate close to 0 from the *training model's* perspective) gets the
    # HIGHER sampling weight -- hard-example mining.
    assert weights[1] > weights[0]


def test_compute_win_rate_weights_formula():
    weights = opponent_pool.compute_win_rate_weights([0.3, 0.7], epsilon=0.01)
    assert weights == [pytest.approx(0.7), pytest.approx(0.3)]


def test_compute_win_rate_weights_epsilon_floor():
    weights = opponent_pool.compute_win_rate_weights([1.0], epsilon=0.05)
    assert weights == [0.05]


def test_compute_win_rate_weights_empty_raises():
    with pytest.raises(ValueError):
        opponent_pool.compute_win_rate_weights([])


def test_compute_win_rate_weights_out_of_range_raises():
    with pytest.raises(ValueError):
        opponent_pool.compute_win_rate_weights([1.5])
    with pytest.raises(ValueError):
        opponent_pool.compute_win_rate_weights([-0.1])


# ── sample_pool_index ─────────────────────────────────────────────────────

def test_sample_pool_index_distribution_matches_weights():
    rng = random.Random(42)
    weights = [1.0, 1.0, 8.0]  # index 2 should dominate
    counts = [0, 0, 0]
    for _ in range(2000):
        counts[opponent_pool.sample_pool_index(weights, rng)] += 1
    assert counts[2] > counts[0] + counts[1]
    # Every index should get sampled at least once out of 2000 draws.
    assert all(c > 0 for c in counts)


def test_sample_pool_index_deterministic_with_seeded_rng():
    weights = [1.0, 2.0, 3.0]
    seq_a = [opponent_pool.sample_pool_index(weights, random.Random(7)) for _ in range(20)]
    seq_b = [opponent_pool.sample_pool_index(weights, random.Random(7)) for _ in range(20)]
    assert seq_a == seq_b


def test_sample_pool_index_single_weight_always_zero():
    rng = random.Random(0)
    for _ in range(10):
        assert opponent_pool.sample_pool_index([1.0], rng) == 0


def test_sample_pool_index_empty_weights_raises():
    with pytest.raises(ValueError):
        opponent_pool.sample_pool_index([], random.Random(0))


def test_sample_pool_index_zero_sum_weights_raises():
    with pytest.raises(ValueError):
        opponent_pool.sample_pool_index([0.0, 0.0], random.Random(0))


# ── pool_diversity_entropy ────────────────────────────────────────────────

def test_pool_diversity_entropy_uniform_is_one():
    entropy = opponent_pool.pool_diversity_entropy([10, 10, 10, 10])
    assert entropy == pytest.approx(1.0)


def test_pool_diversity_entropy_degenerate_all_one_bucket_is_zero():
    entropy = opponent_pool.pool_diversity_entropy([50, 0, 0, 0])
    assert entropy == pytest.approx(0.0)


def test_pool_diversity_entropy_single_member_pool_is_one():
    assert opponent_pool.pool_diversity_entropy([42]) == 1.0


def test_pool_diversity_entropy_empty_is_one():
    assert opponent_pool.pool_diversity_entropy([]) == 1.0


def test_pool_diversity_entropy_partial_selection_is_between_zero_and_one():
    # 2 of 5 pool members selected, evenly between them -- less diverse than
    # full-uniform-across-5 but more diverse than a single bucket.
    entropy = opponent_pool.pool_diversity_entropy([10, 10, 0, 0, 0])
    assert 0.0 < entropy < 1.0


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
