#!/usr/bin/env python3
"""
Unit tests for parchis/evaluation/elo.py -- the pure Elo-rating functions
behind parchis/evaluation/elo_ladder.py (see docs/RL_DESIGN_REVIEW.md
Phase 4).
"""
import random

import pytest

from parchis.evaluation import elo


# ── expected_score ────────────────────────────────────────────────────────

def test_expected_score_equal_ratings_is_half():
    assert elo.expected_score(1200.0, 1200.0) == pytest.approx(0.5)


def test_expected_score_higher_rating_favored():
    assert elo.expected_score(1400.0, 1200.0) > 0.5
    assert elo.expected_score(1200.0, 1400.0) < 0.5


def test_expected_score_symmetric_complement():
    a = elo.expected_score(1300.0, 1250.0)
    b = elo.expected_score(1250.0, 1300.0)
    assert a + b == pytest.approx(1.0)


def test_expected_score_400_point_gap_is_about_ten_to_one():
    # Standard Elo property: a 400-point gap implies a ~10:1 expected
    # score ratio (expected_score ~= 0.909...).
    assert elo.expected_score(1600.0, 1200.0) == pytest.approx(10.0 / 11.0, abs=1e-6)


# ── update_ratings ─────────────────────────────────────────────────────────

def test_update_ratings_equal_start_full_win_moves_by_half_k():
    new_a, new_b = elo.update_ratings(1200.0, 1200.0, score_a=1.0, k_factor=32.0)
    assert new_a == pytest.approx(1200.0 + 16.0)
    assert new_b == pytest.approx(1200.0 - 16.0)


def test_update_ratings_equal_score_leaves_ratings_unchanged():
    new_a, new_b = elo.update_ratings(1200.0, 1200.0, score_a=0.5, k_factor=32.0)
    assert new_a == pytest.approx(1200.0)
    assert new_b == pytest.approx(1200.0)


def test_update_ratings_zero_sum():
    rating_a, rating_b = 1350.0, 1180.0
    new_a, new_b = elo.update_ratings(rating_a, rating_b, score_a=0.7, k_factor=20.0)
    delta_a = new_a - rating_a
    delta_b = new_b - rating_b
    assert delta_a == pytest.approx(-delta_b)


def test_update_ratings_symmetry_swapping_sides():
    # update_ratings(a, b, s) should be the mirror of update_ratings(b, a, 1-s).
    a1, b1 = elo.update_ratings(1250.0, 1300.0, score_a=0.3, k_factor=32.0)
    b2, a2 = elo.update_ratings(1300.0, 1250.0, score_a=0.7, k_factor=32.0)
    assert a1 == pytest.approx(a2)
    assert b1 == pytest.approx(b2)


def test_update_ratings_underdog_win_gains_more_than_favorite_win():
    # A (underdog) beating B (favorite) outright should move A's rating up
    # by more than the reverse (favorite A beating underdog B outright).
    underdog_gain, _ = elo.update_ratings(1100.0, 1400.0, score_a=1.0, k_factor=32.0)
    favorite_gain, _ = elo.update_ratings(1400.0, 1100.0, score_a=1.0, k_factor=32.0)
    assert (underdog_gain - 1100.0) > (favorite_gain - 1400.0)


def test_update_ratings_invalid_score_raises():
    with pytest.raises(ValueError):
        elo.update_ratings(1200.0, 1200.0, score_a=1.5)
    with pytest.raises(ValueError):
        elo.update_ratings(1200.0, 1200.0, score_a=-0.1)


# ── round_robin_pairings ───────────────────────────────────────────────────

def test_round_robin_pairings_count_matches_n_choose_2():
    participants = ["a", "b", "c", "d"]
    pairs = elo.round_robin_pairings(participants, random.Random(0))
    assert len(pairs) == 6  # 4*3/2


def test_round_robin_pairings_covers_every_unordered_pair_exactly_once():
    participants = ["a", "b", "c"]
    pairs = elo.round_robin_pairings(participants, random.Random(1))
    seen = {frozenset(p) for p in pairs}
    expected = {frozenset(("a", "b")), frozenset(("a", "c")), frozenset(("b", "c"))}
    assert seen == expected
    assert len(pairs) == len(seen)  # no duplicate pairs


def test_round_robin_pairings_deterministic_with_seeded_rng():
    participants = ["a", "b", "c", "d", "e"]
    seq_a = elo.round_robin_pairings(participants, random.Random(42))
    seq_b = elo.round_robin_pairings(participants, random.Random(42))
    assert seq_a == seq_b


def test_round_robin_pairings_too_few_participants_raises():
    with pytest.raises(ValueError):
        elo.round_robin_pairings(["a"], random.Random(0))
    with pytest.raises(ValueError):
        elo.round_robin_pairings([], random.Random(0))


def test_round_robin_pairings_duplicate_participants_raises():
    with pytest.raises(ValueError):
        elo.round_robin_pairings(["a", "a", "b"], random.Random(0))


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
