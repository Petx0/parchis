#!/usr/bin/env python3
"""
Tests for parchis/evaluation/calibration.py (docs/AGENT_REBUILD_PLAN.md
Part 3 item 12): value calibration (bucketing + expected calibration
error).
"""

import numpy as np

from parchis.evaluation.calibration import _bucket_calibration, value_calibration


def test_perfectly_calibrated_predictions_give_zero_ece():
    """Predictions that exactly match the actual within-bucket frequency
    must give ECE == 0.0."""
    print("\nTesting perfectly calibrated predictions give ECE == 0...")

    # Bucket ~[0.0,0.1): predicted 0.05, 2/20 actually won (0.10)... use an
    # exact match instead: predicted 0.1 for 10 examples, exactly 1 won.
    predicted = np.array([0.1] * 10 + [0.9] * 10)
    actual = np.array([1.0] + [0.0] * 9 + [1.0] * 9 + [0.0])  # 1/10 and 9/10

    ece, table = _bucket_calibration(predicted, actual, n_buckets=10)
    assert abs(ece) < 1e-9, f"Expected ECE == 0, got {ece}"
    assert len(table) == 2
    print(f"✓ Perfectly calibrated predictions give ECE={ece:.6f}, {len(table)} buckets")


def test_maximally_miscalibrated_predictions_give_large_ece():
    """Predictions that are exactly backwards (predict 0.9 for losers,
    0.1 for winners) must give a large ECE (close to the worst case, 0.8
    for these two buckets)."""
    print("\nTesting maximally miscalibrated predictions give a large ECE...")

    predicted = np.array([0.9] * 10 + [0.1] * 10)
    actual = np.array([0.0] * 10 + [1.0] * 10)  # always loses when predicted 0.9, always wins when predicted 0.1

    ece, table = _bucket_calibration(predicted, actual, n_buckets=10)
    assert ece > 0.75, f"Expected a large ECE (~0.8), got {ece}"
    for row in table:
        assert abs(row['mean_predicted'] - row['actual_frequency']) > 0.75
    print(f"✓ Maximally miscalibrated predictions give ECE={ece:.4f}")


def test_ece_is_weighted_by_bucket_size():
    """A large, well-calibrated bucket plus a tiny, badly-calibrated one
    must give an ECE dominated by the large bucket's (small) error, not
    the tiny one's (large) error -- confirms the (count/n) weighting."""
    print("\nTesting ECE is weighted by bucket size, not a plain bucket average...")

    # Bucket ~0.5: 100 examples, predicted 0.5, exactly 50 won -> perfectly calibrated.
    # Bucket ~0.95: 1 example, predicted 0.95, actually lost -> badly miscalibrated.
    predicted = np.array([0.5] * 100 + [0.95])
    actual = np.array([1.0] * 50 + [0.0] * 50 + [0.0])

    ece, table = _bucket_calibration(predicted, actual, n_buckets=10)
    # Unweighted average of the two buckets' errors would be (0 + 0.95) / 2 = 0.475.
    # Weighted: (100/101)*0 + (1/101)*0.95 ~= 0.0094.
    assert ece < 0.02, f"Expected a small, size-weighted ECE, got {ece}"
    print(f"✓ ECE={ece:.4f} correctly weighted toward the large, well-calibrated bucket")


def test_bucket_calibration_rejects_mismatched_or_empty_input():
    print("\nTesting _bucket_calibration rejects mismatched/empty input...")

    try:
        _bucket_calibration(np.array([0.1, 0.2]), np.array([1.0]))
        assert False, "expected ValueError on shape mismatch"
    except ValueError:
        pass
    try:
        _bucket_calibration(np.array([]), np.array([]))
        assert False, "expected ValueError on empty input"
    except ValueError:
        pass
    print("✓ _bucket_calibration raises ValueError on shape mismatch and empty input")


def test_value_calibration_end_to_end_on_a_small_trained_model():
    """Integration smoke test: value_calibration must run against a real
    (if tiny/quickly-trained) model and held-out example set, returning a
    finite ECE in [0, 1] and a bucket table covering every recorded
    example exactly once."""
    print("\nTesting value_calibration end-to-end on a small trained model...")

    from parchis.az import selfplay, train

    pool = selfplay.default_pool_factories(noisy_seed=31)
    examples, _stats = selfplay.generate_games(pool, n_games=150, num_players=2,
                                                max_turns=500, seed=30)
    train_ex, val_ex, test_ex = train.split_by_game(examples, train_frac=0.7, val_frac=0.15, seed=0)

    model, _history = train.bootstrap_train(
        train_ex, val_ex, num_players=2, hidden_sizes=(32, 32),
        max_epochs=5, patience=5, batch_size=256, seed=0, log_every=0,
    )

    ece, table = value_calibration(model, test_ex, num_players=2, n_buckets=10)
    assert 0.0 <= ece <= 1.0
    assert sum(row['n'] for row in table) == len(test_ex)
    for row in table:
        assert 0.0 <= row['mean_predicted'] <= 1.0
        assert 0.0 <= row['actual_frequency'] <= 1.0
    print(f"✓ value_calibration ran end-to-end: ece={ece:.4f} over {len(test_ex)} "
          f"held-out examples, {len(table)} non-empty buckets")


if __name__ == '__main__':
    test_perfectly_calibrated_predictions_give_zero_ece()
    test_maximally_miscalibrated_predictions_give_large_ece()
    test_ece_is_weighted_by_bucket_size()
    test_bucket_calibration_rejects_mismatched_or_empty_input()
    test_value_calibration_end_to_end_on_a_small_trained_model()
    print("\nAll calibration tests passed!")
