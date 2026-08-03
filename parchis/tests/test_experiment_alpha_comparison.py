#!/usr/bin/env python3
"""
Tests for the multi-seed aggregation added to
parchis/training/experiment_alpha_comparison.py in docs/RL_DESIGN_REVIEW.md
Phase 4 -- exercises _aggregate_seed_results() directly against synthetic
ExperimentResult lists rather than running real training loops (matching
the fast-unit-test style of test_selfplay_pool_callback.py, which drives
SelfPlayCallback._update_opponents() directly for the same reason).
"""
import json
from dataclasses import asdict

import pytest

from parchis.training.experiment_alpha_comparison import (
    ExperimentResult, _aggregate_seed_results,
)


def _make_result(alpha, win_rate, seed):
    return ExperimentResult(
        alpha=alpha,
        opponent_weighting="mean",
        win_rate=win_rate,
        avg_player_progress=0.5,
        avg_opponent_progress=0.4,
        avg_episode_reward=0.1,
        total_episodes=100,
        model_path=f"/fake/path_seed{seed}",
        seed=seed,
    )


def test_aggregate_seed_results_single_seed_has_no_ci():
    """Default --seeds [42] behavior: one seed can't produce a CI."""
    per_seed = [_make_result(0.5, 0.6, seed=42)]
    aggregated = _aggregate_seed_results(0.5, "mean", per_seed)

    assert aggregated.win_rate_mean == pytest.approx(0.6)
    assert aggregated.win_rate_std == pytest.approx(0.0)
    assert aggregated.win_rate_ci is None
    assert aggregated.per_seed_results == per_seed


def test_aggregate_seed_results_multi_seed_computes_mean_std_ci():
    per_seed = [
        _make_result(0.5, 0.4, seed=1),
        _make_result(0.5, 0.6, seed=2),
        _make_result(0.5, 0.5, seed=3),
    ]
    aggregated = _aggregate_seed_results(0.5, "mean", per_seed)

    assert aggregated.win_rate_mean == pytest.approx(0.5)
    assert aggregated.win_rate_std > 0.0
    assert aggregated.win_rate_ci is not None
    lower, upper = aggregated.win_rate_ci
    assert lower <= aggregated.win_rate_mean <= upper
    assert len(aggregated.per_seed_results) == 3


def test_aggregate_seed_results_identical_seeds_zero_std_no_ci_width():
    per_seed = [_make_result(0.0, 0.5, seed=s) for s in (1, 2, 3)]
    aggregated = _aggregate_seed_results(0.0, "mean", per_seed)

    assert aggregated.win_rate_std == pytest.approx(0.0)
    lower, upper = aggregated.win_rate_ci
    assert lower == pytest.approx(0.5)
    assert upper == pytest.approx(0.5)


def test_aggregate_seed_results_is_json_serializable():
    """AggregatedResult (including a None win_rate_ci) must round-trip
    through json.dumps -- this is what _save_results_txt's per-seed
    section and any future JSON export rely on."""
    single_seed = _aggregate_seed_results(0.5, "mean", [_make_result(0.5, 0.6, seed=42)])
    multi_seed = _aggregate_seed_results(
        0.5, "mean", [_make_result(0.5, w, seed=i) for i, w in enumerate([0.4, 0.6, 0.5])]
    )

    for aggregated in (single_seed, multi_seed):
        serialized = json.dumps(asdict(aggregated))
        restored = json.loads(serialized)
        assert restored['alpha'] == aggregated.alpha
        assert len(restored['per_seed_results']) == len(aggregated.per_seed_results)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
