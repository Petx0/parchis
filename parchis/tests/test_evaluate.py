#!/usr/bin/env python3
"""
Tests for the Phase 4 KPI/CI additions to evaluate_agent
(parchis/evaluation/evaluate.py) and evaluate_model
(parchis/training/common.py) -- see docs/RL_DESIGN_REVIEW.md Phase 4.

Trains tiny real models (small total_timesteps, no mocks) rather than
mocking, matching the established convention in test_eval_issue.py.
"""
import os
import tempfile

import pytest
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from parchis.rl.env import ParchisEnv
from parchis.evaluation.evaluate import evaluate_agent
from parchis.training.common import make_env, evaluate_model


def mask_fn(env):
    return env.unwrapped._get_info()['action_masks']


def _train_tiny_model(num_players=4):
    env = ParchisEnv(num_players=num_players)
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=None,
        n_steps=64,
        batch_size=32,
    )
    model.learn(total_timesteps=200)
    env.close()
    return model


def _assert_phase4_stats_are_sane(stats, n_episodes):
    lower, upper = stats['win_rate_ci']
    assert 0.0 <= lower <= stats['win_rate'] <= upper <= 1.0

    assert sum(v['n'] for v in stats['win_rate_by_seat'].values()) == n_episodes
    assert sum(v['n'] for v in stats['win_rate_by_color'].values()) == n_episodes
    for breakdown in (stats['win_rate_by_seat'], stats['win_rate_by_color']):
        for entry in breakdown.values():
            b_lower, b_upper = entry['ci']
            assert 0.0 <= b_lower <= entry['win_rate'] <= b_upper <= 1.0

    assert stats['capture_rate'] >= 0.0
    assert stats['capture_rate_against'] >= 0.0
    assert stats['three_sixes_penalty_rate'] >= 0.0

    # Every environment step contributes a legal_moves_count sample, so
    # this should always be present as long as at least one game ran.
    assert 'mean_legal_moves_count' in stats
    assert stats['mean_legal_moves_count'] >= 0.0
    assert stats['std_legal_moves_count'] >= 0.0

    # Bonus chains aren't guaranteed to occur in a handful of short games,
    # so only check sanity if the field is present at all.
    if 'mean_bonus_chain_length' in stats:
        assert stats['mean_bonus_chain_length'] >= 1.0


def test_evaluate_agent_reports_phase4_stats():
    """evaluate_agent() (random-opponent path) returns the new Wilson CI,
    per-seat/color breakdown, and KPI fields introduced in Phase 4."""
    print("\nTesting evaluate_agent() Phase 4 stats (random opponent)...")

    model = _train_tiny_model(num_players=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "tiny_model")
        model.save(model_path)

        n_games = 8
        stats = evaluate_agent(
            agent_model_path=model_path,
            opponent_model_path=None,
            n_games=n_games,
            num_players=4,
            deterministic=True,
            max_steps_per_episode=300,
            verbose=0,
        )

    _assert_phase4_stats_are_sane(stats, n_games)
    print("✓ evaluate_agent() (random opponent) reports sane Phase 4 stats")


def test_evaluate_agent_reports_phase4_stats_selfplay():
    """evaluate_agent() (checkpoint-vs-checkpoint self-play path) also
    returns the Phase 4 fields -- a separate code path from the
    random-opponent branch (see evaluate.py's env construction)."""
    print("\nTesting evaluate_agent() Phase 4 stats (self-play opponent)...")

    agent_model = _train_tiny_model(num_players=2)
    opponent_model = _train_tiny_model(num_players=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_path = os.path.join(tmpdir, "agent_model")
        opponent_path = os.path.join(tmpdir, "opponent_model")
        agent_model.save(agent_path)
        opponent_model.save(opponent_path)

        n_games = 6
        stats = evaluate_agent(
            agent_model_path=agent_path,
            opponent_model_path=opponent_path,
            n_games=n_games,
            num_players=2,
            deterministic=True,
            max_steps_per_episode=300,
            verbose=0,
        )

    _assert_phase4_stats_are_sane(stats, n_games)
    print("✓ evaluate_agent() (self-play opponent) reports sane Phase 4 stats")


def test_evaluate_model_reports_phase4_stats():
    """evaluate_model() (parchis/training/common.py) returns the same new
    Wilson CI, per-seat/color breakdown, and KPI fields."""
    print("\nTesting evaluate_model() Phase 4 stats...")

    model = _train_tiny_model(num_players=4)
    eval_env = make_env(num_players=4, seed=7)

    n_episodes = 8
    stats = evaluate_model(model, eval_env, n_eval_episodes=n_episodes,
                            max_steps_per_episode=300, verbose=0)
    eval_env.close()

    _assert_phase4_stats_are_sane(stats, n_episodes)
    print("✓ evaluate_model() reports sane Phase 4 stats")


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
