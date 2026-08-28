#!/usr/bin/env python3
"""
Tests for parchis/evaluation/ratings.py: Bradley-Terry maximum-likelihood
ratings fit over parchis.evaluation.ladder's accumulated pairings.jsonl.
"""

import math

import pytest

from parchis.agents import heuristic
from parchis.az.selfplay import random_factory
from parchis.evaluation import ladder, ratings


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _synthetic_pairing(name_a, name_b, true_rating_a, true_rating_b, n_games=100_000):
    """A pairing record whose wins_a exactly matches the Bradley-Terry
    model's own predicted probability for the given true ratings (at large
    n_games, so the MLE fit converges tightly to those true ratings --
    this is testing the MATH, not simulating noisy real games)."""
    p_a_beats_b = _sigmoid(true_rating_a - true_rating_b)
    return {
        "participant_a": name_a, "participant_b": name_b,
        "n_games": n_games, "wins_a": round(n_games * p_a_beats_b),
    }


TRUE_RATINGS = {"random": 0.0, "weak": -1.0, "strong": 2.0}


def _synthetic_pairings():
    names = list(TRUE_RATINGS)
    return [
        _synthetic_pairing(a, b, TRUE_RATINGS[a], TRUE_RATINGS[b])
        for i, a in enumerate(names) for b in names[i + 1:]
    ]


def test_fit_ratings_recovers_known_strength_ordering_and_gaps():
    print("\nTesting fit_ratings recovers a known Bradley-Terry rating structure...")
    fitted = ratings.fit_ratings(_synthetic_pairings())

    assert fitted["random"] == pytest.approx(0.0, abs=1e-6), "anchor must be exactly 0.0"
    assert fitted["weak"] == pytest.approx(TRUE_RATINGS["weak"], abs=0.05)
    assert fitted["strong"] == pytest.approx(TRUE_RATINGS["strong"], abs=0.05)
    assert fitted["strong"] > fitted["random"] > fitted["weak"]
    print(f"✓ fitted={fitted}, matches true ratings {TRUE_RATINGS} within tolerance")


def test_fit_ratings_anchors_on_alphabetically_first_when_no_random_participant():
    print("\nTesting fit_ratings anchors on the alphabetically-first participant "
          "when 'random' isn't among the participants...")
    pairings = [_synthetic_pairing("alpha", "beta", 1.0, -1.0, n_games=1000)]
    fitted = ratings.fit_ratings(pairings)
    assert fitted["alpha"] == pytest.approx(0.0, abs=1e-6)
    print(f"✓ anchored on 'alpha' (alphabetically first): {fitted}")


def test_fit_ratings_explicit_anchor():
    print("\nTesting fit_ratings honors an explicitly passed anchor...")
    pairings = _synthetic_pairings()
    fitted = ratings.fit_ratings(pairings, anchor="strong")
    assert fitted["strong"] == pytest.approx(0.0, abs=1e-6)
    # Relative gaps must be preserved regardless of which participant is
    # the zero point.
    assert (fitted["random"] - fitted["weak"]) == pytest.approx(
        TRUE_RATINGS["random"] - TRUE_RATINGS["weak"], abs=0.05
    )
    print(f"✓ anchored on 'strong': {fitted}")


def test_fit_ratings_rejects_unknown_anchor():
    print("\nTesting fit_ratings rejects an anchor not among the participants...")
    with pytest.raises(ValueError):
        ratings.fit_ratings(_synthetic_pairings(), anchor="not_a_participant")
    print("✓ raises ValueError")


def test_fit_ratings_requires_at_least_two_participants():
    print("\nTesting fit_ratings rejects fewer than 2 participants...")
    with pytest.raises(ValueError):
        ratings.fit_ratings([])
    print("✓ raises ValueError")


def test_bootstrap_rating_cis_contains_point_estimate_and_narrows_with_more_games():
    print("\nTesting bootstrap_rating_cis brackets the point estimate...")
    pairings = _synthetic_pairings()
    fitted = ratings.fit_ratings(pairings)
    cis = ratings.bootstrap_rating_cis(pairings, n_reps=50, seed=0)

    for name, point in fitted.items():
        lower, upper = cis[name]
        assert lower <= point <= upper, (
            f"{name}: point estimate {point} not within bootstrap CI [{lower}, {upper}]"
        )
    # random is the anchor -- every replicate that fits at all pins it at
    # exactly 0.0, so its CI must be a degenerate point, not a spread.
    assert cis["random"] == pytest.approx((0.0, 0.0), abs=1e-9)
    print(f"✓ all point estimates fall within their bootstrap CIs: {cis}")


def test_load_pairings_missing_file_returns_empty_list(tmp_path):
    print("\nTesting load_pairings returns [] for a nonexistent file...")
    assert ratings.load_pairings(str(tmp_path / "nope.jsonl")) == []
    print("✓ returns []")


def test_end_to_end_ladder_then_ratings(tmp_path):
    """Integration: a real (small) ladder run's output must load and fit
    without error, with one rating per rung -- structural check only (real
    small-sample win rates are noisy, so this doesn't assert an ordering,
    unlike the synthetic-data tests above which test the math itself)."""
    print("\nTesting ladder.run_ladder's real output loads and fits cleanly...")
    pairings_path = tmp_path / "pairings.jsonl"
    rungs = {
        "random": random_factory,
        "heuristic_tuned": heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS),
    }
    ladder.run_ladder(rungs, num_players=2, n_pairs=5, max_turns=300, seed=3,
                       pairings_path=str(pairings_path), verbose=0)

    loaded = ratings.load_pairings(str(pairings_path))
    assert len(loaded) == 1
    fitted = ratings.fit_ratings(loaded)
    assert set(fitted) == set(rungs)
    assert fitted["random"] == pytest.approx(0.0, abs=1e-6)
    print(f"✓ end-to-end ladder -> pairings.jsonl -> ratings: {fitted}")


if __name__ == '__main__':
    test_fit_ratings_recovers_known_strength_ordering_and_gaps()
    test_fit_ratings_anchors_on_alphabetically_first_when_no_random_participant()
    test_fit_ratings_explicit_anchor()
    test_fit_ratings_rejects_unknown_anchor()
    test_fit_ratings_requires_at_least_two_participants()
    test_bootstrap_rating_cis_contains_point_estimate_and_narrows_with_more_games()
    print("\n(other tests need tmp_path -- run via pytest for full coverage)")
