#!/usr/bin/env python3
"""
Tests for parchis/az/targets.py (docs/AGENT_REBUILD_PLAN.md Part 3 Phase 3):
value-target blending, policy-target softmax, the self-play temperature
anneal, and Dirichlet root-noise mixing.
"""

import numpy as np
import pytest

from parchis.az import targets


def test_blend_value_target_lambda_extremes():
    print("\nTesting blend_value_target at lambda=0/1/0.5...")
    outcome = np.array([1.0, 0.0])
    root_value = np.array([0.7, 0.3])

    assert np.allclose(targets.blend_value_target(outcome, root_value, lam=0.0), outcome)
    assert np.allclose(targets.blend_value_target(outcome, root_value, lam=1.0), root_value)
    assert np.allclose(targets.blend_value_target(outcome, root_value, lam=0.5), [0.85, 0.15])
    print("✓ lam=0 -> outcome, lam=1 -> root_value, lam=0.5 -> the average")


def test_blend_value_target_rejects_bad_input():
    print("\nTesting blend_value_target rejects shape mismatch / bad lambda...")
    with pytest.raises(ValueError):
        targets.blend_value_target([1.0, 0.0], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        targets.blend_value_target([1.0, 0.0], [0.5, 0.5], lam=1.5)
    with pytest.raises(ValueError):
        targets.blend_value_target([1.0, 0.0], [0.5, 0.5], lam=-0.1)
    print("✓ raises ValueError on shape mismatch and out-of-[0,1] lambda")


def test_policy_target_sums_to_one_and_masks_illegal():
    print("\nTesting policy_target_from_move_values sums to 1 and masks illegal actions...")
    # Only piece_ids 0, 1, 3 are legal here (2 is missing = illegal).
    move_values = {0: np.array([0.5, 0.5]), 1: np.array([0.6, 0.4]), 3: np.array([0.4, 0.6])}
    z = targets.policy_target_from_move_values(move_values, mover_seat=0, tau_target=0.5)

    assert z.shape == (4,)
    assert z[2] == 0.0, "Illegal piece_id 2 must get exactly 0.0 probability"
    assert abs(z.sum() - 1.0) < 1e-6
    assert all(z[pid] > 0.0 for pid in (0, 1, 3))
    print(f"✓ z_policy={np.round(z, 4).tolist()}, sums to 1.0, zero on the illegal slot")


def test_policy_target_sharper_at_lower_temperature():
    print("\nTesting policy_target_from_move_values sharpens as tau_target decreases...")
    # piece 0 clearly better than piece 1 for the mover (seat 0).
    move_values = {0: np.array([0.9, 0.1]), 1: np.array([0.5, 0.5])}

    z_hot = targets.policy_target_from_move_values(move_values, mover_seat=0, tau_target=1.0)
    z_cold = targets.policy_target_from_move_values(move_values, mover_seat=0, tau_target=0.05)

    assert z_hot[0] < z_cold[0], (
        f"Expected lower tau_target to concentrate more mass on the better move: "
        f"z_hot[0]={z_hot[0]:.4f}, z_cold[0]={z_cold[0]:.4f}"
    )
    assert z_cold[0] > 0.99, f"Expected near-argmax behavior at tau_target=0.05, got {z_cold[0]:.4f}"
    print(f"✓ z_hot[0]={z_hot[0]:.4f} < z_cold[0]={z_cold[0]:.4f} (cold ~= argmax)")


def test_policy_target_requires_nonempty_move_values():
    print("\nTesting policy_target_from_move_values rejects an empty move_values...")
    with pytest.raises(ValueError):
        targets.policy_target_from_move_values({}, mover_seat=0)
    print("✓ raises ValueError on empty move_values")


def test_anneal_temperature_schedule():
    print("\nTesting anneal_temperature's default and custom schedules...")
    assert targets.anneal_temperature(0) == pytest.approx(1.0)
    assert targets.anneal_temperature(15) == pytest.approx(0.25)
    assert targets.anneal_temperature(100) == pytest.approx(0.25), "Must clamp past anneal_plies"

    # Custom schedule with an exact linear-interpolation midpoint.
    mid = targets.anneal_temperature(5, tau_start=1.0, tau_end=0.25, anneal_plies=10)
    assert mid == pytest.approx(0.625)
    print(f"✓ ply0=1.0, ply15=0.25, ply100=0.25 (clamped); custom midpoint={mid:.4f}")


def test_dirichlet_mixed_probs_epsilon_zero_matches_plain_softmax():
    print("\nTesting dirichlet_mixed_probs at epsilon=0 matches a plain masked softmax...")
    move_values = {0: np.array([0.8, 0.2]), 2: np.array([0.3, 0.7])}
    rng = np.random.default_rng(0)

    probs, legal = targets.dirichlet_mixed_probs(move_values, mover_seat=0, tau=0.5, rng=rng, epsilon=0.0)
    expected = targets.policy_target_from_move_values(move_values, mover_seat=0, tau_target=0.5)

    assert legal == [0, 2]
    assert np.allclose(probs, expected, atol=1e-6)
    print(f"✓ epsilon=0 gives probs={np.round(probs, 4).tolist()}, matching the plain softmax")


def test_dirichlet_mixed_probs_masks_illegal_and_sums_to_one():
    print("\nTesting dirichlet_mixed_probs masks illegal actions and sums to 1...")
    move_values = {1: np.array([0.5, 0.5]), 3: np.array([0.6, 0.4])}
    rng = np.random.default_rng(1)

    probs, legal = targets.dirichlet_mixed_probs(move_values, mover_seat=0, tau=1.0, rng=rng, epsilon=0.25)
    assert legal == [1, 3]
    assert probs[0] == 0.0 and probs[2] == 0.0
    assert abs(probs.sum() - 1.0) < 1e-6
    print(f"✓ probs={np.round(probs, 4).tolist()}, zero on illegal slots, sums to 1.0")


def test_dirichlet_mixed_probs_epsilon_controls_whether_noise_matters():
    print("\nTesting dirichlet noise actually perturbs the distribution when epsilon > 0...")
    move_values = {0: np.array([0.5, 0.5]), 1: np.array([0.5, 0.5]), 2: np.array([0.5, 0.5])}
    rng = np.random.default_rng(2)

    # epsilon=0: deterministic across repeated calls (same rng stream keeps advancing,
    # but the RESULT never depends on the draw since noise is never sampled/used).
    probs_a, _ = targets.dirichlet_mixed_probs(move_values, mover_seat=0, tau=1.0, rng=rng, epsilon=0.0)
    probs_b, _ = targets.dirichlet_mixed_probs(move_values, mover_seat=0, tau=1.0, rng=rng, epsilon=0.0)
    assert np.allclose(probs_a, probs_b), "epsilon=0 must never depend on the rng draw"

    # epsilon=1 (pure noise): repeated calls on a fresh rng stream must vary.
    rng2 = np.random.default_rng(3)
    probs_c, _ = targets.dirichlet_mixed_probs(move_values, mover_seat=0, tau=1.0, rng=rng2,
                                                epsilon=1.0, alpha=0.1)
    probs_d, _ = targets.dirichlet_mixed_probs(move_values, mover_seat=0, tau=1.0, rng=rng2,
                                                epsilon=1.0, alpha=0.1)
    assert not np.allclose(probs_c, probs_d), "epsilon=1 (pure noise) must vary across draws"
    print("✓ epsilon=0 is draw-independent; epsilon=1 (pure noise) varies across draws")


def test_sample_move_piece_id_respects_a_concentrated_distribution():
    print("\nTesting sample_move_piece_id always returns the concentrated piece_id...")
    probs = np.zeros(4)
    probs[2] = 1.0
    rng = np.random.default_rng(4)
    draws = {targets.sample_move_piece_id(probs, [0, 2, 3], rng) for _ in range(50)}
    assert draws == {2}
    print("✓ 50/50 draws returned the only piece_id with nonzero probability")


def test_sample_move_piece_id_never_returns_an_illegal_id():
    print("\nTesting sample_move_piece_id never draws outside legal_piece_ids...")
    probs = np.array([0.1, 0.0, 0.4, 0.5])  # index 1 has weight but isn't "legal" here
    rng = np.random.default_rng(5)
    draws = {targets.sample_move_piece_id(probs, [0, 2, 3], rng) for _ in range(200)}
    assert draws <= {0, 2, 3}
    assert len(draws) > 1, "Test setup error: expected more than one distinct draw over 200 samples"
    print(f"✓ 200 draws all within legal_piece_ids, saw {sorted(draws)}")


if __name__ == '__main__':
    test_blend_value_target_lambda_extremes()
    test_blend_value_target_rejects_bad_input()
    test_policy_target_sums_to_one_and_masks_illegal()
    test_policy_target_sharper_at_lower_temperature()
    test_policy_target_requires_nonempty_move_values()
    test_anneal_temperature_schedule()
    test_dirichlet_mixed_probs_epsilon_zero_matches_plain_softmax()
    test_dirichlet_mixed_probs_masks_illegal_and_sums_to_one()
    test_dirichlet_mixed_probs_epsilon_controls_whether_noise_matters()
    test_sample_move_piece_id_respects_a_concentrated_distribution()
    test_sample_move_piece_id_never_returns_an_illegal_id()
    print("\nAll targets tests passed!")
