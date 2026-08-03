#!/usr/bin/env python3
"""
Tests for the multi-seed aggregation added to
parchis/training/experiment_grid.py in docs/RL_DESIGN_REVIEW.md Phase 4 --
exercises _aggregate_seed_results() directly against synthetic
ExperimentResult lists rather than running real training loops (matching
the fast-unit-test style of test_selfplay_pool_callback.py, which drives
SelfPlayCallback._update_opponents() directly for the same reason).
"""
import json
from dataclasses import asdict

import pytest

from parchis.training.experiment_grid import (
    ExperimentResult, _aggregate_seed_results, build_experiment_list,
)
from parchis.evaluation import stats as eval_stats


CONFIG = {
    "name": "small_progress_delta",
    "arch_name": "small",
    "net_arch": [64, 64],
    "activation_fn": type("FakeActivation", (), {"__name__": "Tanh"}),
    "reward_type": "progress_delta",
}


def _make_result(win_rate, seed):
    return ExperimentResult(
        name=CONFIG["name"],
        arch_name=CONFIG["arch_name"],
        net_arch=CONFIG["net_arch"],
        activation="Tanh",
        reward_type=CONFIG["reward_type"],
        win_rate=win_rate,
        avg_player_progress=0.5,
        avg_opponent_progress=0.4,
        std_opponent_progress=0.1,
        avg_episode_reward=0.1,
        std_episode_reward=0.05,
        total_eval_episodes=100,
        training_time_seconds=10.0,
        model_path=f"/fake/{CONFIG['name']}_seed{seed}",
        total_timesteps=1000,
        seed=seed,
    )


def test_aggregate_seed_results_single_seed_has_no_ci():
    per_seed = [_make_result(0.6, seed=42)]
    aggregated = _aggregate_seed_results(CONFIG, per_seed)

    assert aggregated.win_rate_mean == pytest.approx(0.6)
    assert aggregated.win_rate_ci is None
    assert aggregated.training_time_seconds_total == pytest.approx(10.0)


def test_aggregate_seed_results_multi_seed_computes_mean_std_ci():
    per_seed = [_make_result(w, seed=i) for i, w in enumerate([0.4, 0.6, 0.5])]
    aggregated = _aggregate_seed_results(CONFIG, per_seed)

    assert aggregated.win_rate_mean == pytest.approx(0.5)
    assert aggregated.win_rate_std > 0.0
    lower, upper = aggregated.win_rate_ci
    assert lower <= aggregated.win_rate_mean <= upper
    # Training time should sum across seeds, not average.
    assert aggregated.training_time_seconds_total == pytest.approx(30.0)


def test_rank_by_mean_with_ci_flags_overlapping_configs_as_unconfirmed():
    """Two configs whose win-rate CIs overlap should not be reported as a
    statistically confirmed 'best' -- the exact scenario
    docs/RL_DESIGN_REVIEW.md Phase 4 flags: a raw point-estimate max()
    can't distinguish this from seed noise."""
    config_a = _aggregate_seed_results(CONFIG, [_make_result(w, seed=i) for i, w in enumerate([0.55, 0.60, 0.50])])
    config_b_config = {**CONFIG, "name": "small_win_loss"}
    config_b = _aggregate_seed_results(config_b_config,
                                        [_make_result(w, seed=i) for i, w in enumerate([0.50, 0.55, 0.45])])

    entries = [(config_a.name, config_a.win_rate_mean, config_a.win_rate_ci),
               (config_b.name, config_b.win_rate_mean, config_b.win_rate_ci)]
    ranked, confirmed = eval_stats.rank_by_mean_with_ci(entries)

    assert ranked[0][0] == config_a.name  # higher mean ranks first
    assert confirmed is False  # but CIs overlap given the small seed count


def test_aggregate_seed_results_is_json_serializable():
    single_seed = _aggregate_seed_results(CONFIG, [_make_result(0.6, seed=42)])
    multi_seed = _aggregate_seed_results(CONFIG, [_make_result(w, seed=i) for i, w in enumerate([0.4, 0.6, 0.5])])

    for aggregated in (single_seed, multi_seed):
        serialized = json.dumps(asdict(aggregated))
        restored = json.loads(serialized)
        assert restored['name'] == aggregated.name
        assert len(restored['per_seed_results']) == len(aggregated.per_seed_results)


def test_build_experiment_list_unaffected_by_seed_changes():
    """Sanity check: the 3x3 grid definition itself is untouched by the
    multi-seed changes."""
    experiments = build_experiment_list()
    assert len(experiments) == 9


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
