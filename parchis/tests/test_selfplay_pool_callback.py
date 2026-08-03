#!/usr/bin/env python3
"""
Orchestration tests for SelfPlayCallback's opponent-pool management
(parchis/training/train_selfplay.py): pool growth/eviction, checkpoint-file
retention on disk, the opponent_pool_diversity KPI's first-update edge case,
and the "win_rate" sampling strategy. See parchis/tests/test_opponent_pool.py
for the underlying pure-function coverage and test_selfplay.py for
ParchisSelfPlayEnv-level pool-sampling coverage.

These call SelfPlayCallback._update_opponents() directly (not via a full
model.learn() loop) -- SB3's BaseCallback only needs .model, .training_env,
.num_timesteps and .logger to be set for that method to run, all of which
are plain settable attributes/properties, so this is a much faster and more
deterministic way to exercise the same orchestration logic than actually
training.
"""
import glob
import os
import tempfile

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure as configure_logger

from parchis.training.train_selfplay import SelfPlayCallback, make_selfplay_env


def _build_model_and_callback(tmpdir, pool_size=2, pool_sampling_strategy="uniform",
                               pool_eval_episodes=2, num_players=4):
    """A tiny (untrained -- randomly initialized policy is enough for
    save/load/predict to all work) MaskablePPO model plus a SelfPlayCallback
    wired up exactly as train_selfplay() wires it, without paying for an
    actual training loop."""
    train_env = make_selfplay_env(num_players=num_players, seed=0)
    model = MaskablePPO("MlpPolicy", train_env, verbose=0, tensorboard_log=None,
                         n_steps=64, batch_size=32)
    model.set_logger(configure_logger(folder=None, format_strings=[]))

    callback = SelfPlayCallback(
        update_freq=1,  # irrelevant here -- we call _update_opponents() directly
        save_path=tmpdir,
        pool_size=pool_size,
        pool_sampling_strategy=pool_sampling_strategy,
        pool_eval_episodes=pool_eval_episodes,
        num_players=num_players,
        verbose=0,
    )
    callback.model = model  # callback.training_env derives from model.get_env()
    return model, train_env, callback


def test_pool_grows_to_cap_and_keeps_all_checkpoint_files_on_disk():
    """The live in-memory pool should stay capped at pool_size, but
    evicted checkpoints must remain on disk (Phase 4's future round-robin
    ladder needs the full training history, not just the live pool)."""
    print("\nTesting opponent pool caps in-memory but retains all checkpoint files...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model, train_env, callback = _build_model_and_callback(tmpdir, pool_size=2)
        selfplay_env = callback._unwrap_to_selfplay_env(callback.training_env.envs[0])
        assert selfplay_env is not None

        for i in range(3):
            callback.num_timesteps = (i + 1) * 100
            callback._update_opponents()

        assert len(callback.pool) == 2, f"pool should be capped at 2, got {len(callback.pool)}"
        assert len(selfplay_env.opponent_pool) == 2, (
            "the env's live pool should reflect the callback's capped pool"
        )

        checkpoint_files = glob.glob(os.path.join(tmpdir, "opponent_checkpoint_*"))
        assert len(checkpoint_files) == 3, (
            f"expected all 3 opponent checkpoints to remain on disk despite "
            f"only 2 staying in the live pool, found {checkpoint_files}"
        )
        print(f"✓ Live pool capped at 2, all 3 checkpoint files retained on disk")
        train_env.close()


def test_diversity_metric_not_logged_on_first_update_but_logged_after():
    """metrics/opponent_pool_diversity has nothing to measure on the very
    first update (pool was empty, opponent was random) -- it must be
    skipped then, and present from the second update onward."""
    print("\nTesting opponent_pool_diversity is skipped on the first update...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model, train_env, callback = _build_model_and_callback(tmpdir, pool_size=3)

        callback.num_timesteps = 100
        callback._update_opponents()
        assert 'metrics/opponent_pool_diversity' not in callback.logger.name_to_value, (
            "diversity should NOT be logged on the first update"
        )
        assert callback.logger.name_to_value.get('metrics/opponent_pool_size') == 1

        # Play a handful of episodes so opponent_selection_counts is non-empty
        # before the next pool update reads it.
        obs, info = train_env.reset(seed=0)
        for _ in range(20):
            mask = info['action_masks']
            legal = np.where(mask)[0]
            action = int(legal[0]) if len(legal) else 0
            obs, reward, terminated, truncated, info = train_env.step(action)
            if terminated or truncated:
                obs, info = train_env.reset(seed=0)

        callback.num_timesteps = 200
        callback._update_opponents()
        assert 'metrics/opponent_pool_diversity' in callback.logger.name_to_value, (
            "diversity SHOULD be logged from the second update onward"
        )
        assert callback.logger.name_to_value.get('metrics/opponent_pool_size') == 2
        print("✓ Diversity metric correctly skipped on update #1, present on update #2")
        train_env.close()


def test_win_rate_strategy_scores_every_pool_member():
    """The "win_rate" sampling strategy must score every pool member
    (weights length == pool length) and build exactly one persistent
    scoring env, reused across updates rather than rebuilt each time."""
    print("\nTesting win_rate pool-sampling strategy scores all pool members...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model, train_env, callback = _build_model_and_callback(
            tmpdir, pool_size=2, pool_sampling_strategy="win_rate", pool_eval_episodes=2,
        )

        callback.num_timesteps = 100
        callback._update_opponents()
        assert callback._pool_eval_env is not None, "scoring env should be built lazily on first use"
        first_eval_env = callback._pool_eval_env

        callback.num_timesteps = 200
        callback._update_opponents()
        assert callback._pool_eval_env is first_eval_env, (
            "the scoring env should be reused across updates, not rebuilt"
        )

        selfplay_env = callback._unwrap_to_selfplay_env(callback.training_env.envs[0])
        assert len(selfplay_env.opponent_pool_weights) == len(callback.pool) == 2, (
            "every pool member must get a win-rate-derived weight"
        )

        callback._pool_eval_env.close()
        train_env.close()
        print(f"✓ win_rate strategy scored all {len(callback.pool)} pool members, "
              f"scoring env reused across updates")


if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main([__file__, '-v']))
