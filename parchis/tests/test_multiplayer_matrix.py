#!/usr/bin/env python3
"""
Tests for parchis/evaluation/multiplayer_matrix.py -- the 3-4 player
pairwise win-rate matrix (Elo has no valid interpretation there, see
elo_ladder.py's own module docstring).

Drives run_matrix() with tiny real trained models (small total_timesteps,
no mocks), matching test_elo_ladder.py's established convention.
"""
import os
import json
import tempfile

import pytest
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from parchis.rl.env import ParchisEnv
from parchis.evaluation.multiplayer_matrix import run_matrix, RANDOM_PARTICIPANT


def mask_fn(env):
    return env.unwrapped._get_info()['action_masks']


def _train_tiny_model():
    env = ParchisEnv(num_players=4)
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


@pytest.fixture(scope="module")
def two_checkpoints():
    """Two tiny trained 4-player checkpoints, saved to a shared tempdir for
    the whole module (training is the expensive part; reused across tests)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {}
        for name in ("ckpt_a", "ckpt_b"):
            model = _train_tiny_model()
            path = os.path.join(tmpdir, name)
            model.save(path)
            paths[name] = path
        yield paths


def test_run_matrix_with_random_baseline_only_plays_checkpoint_vs_random_direction(two_checkpoints):
    """2 checkpoints + random -> 3 participants. Real-checkpoint pair plays
    both directions (2 entries); each (checkpoint, random) pair plays only
    the checkpoint-as-agent direction (1 entry each) -- 4 entries total,
    never "random" as the agent."""
    print("\nTesting run_matrix() with the random baseline included...")

    entries = run_matrix(
        checkpoint_paths=two_checkpoints,
        num_players=4,
        include_random_baseline=True,
        games_per_pairing=4,
        seed=1,
        save_path=None,
        verbose=0,
    )

    assert len(entries) == 4  # (a,b) + (b,a) + (a,random) + (b,random)
    assert all(e.agent != RANDOM_PARTICIPANT for e in entries), (
        "random must never appear as the tracked agent -- there's no "
        "stand-in random-policy model for evaluate_agent's agent slot"
    )
    agent_opponent_pairs = {(e.agent, e.opponent) for e in entries}
    assert agent_opponent_pairs == {
        ("ckpt_a", "ckpt_b"), ("ckpt_b", "ckpt_a"),
        ("ckpt_a", RANDOM_PARTICIPANT), ("ckpt_b", RANDOM_PARTICIPANT),
    }
    for e in entries:
        assert e.num_players == 4
        assert e.games == 4
        assert 0 <= e.wins <= e.games
        lower, upper = e.win_rate_ci
        assert 0.0 <= lower <= e.win_rate <= upper <= 1.0
    print(f"✓ 4 entries ran, directions: {agent_opponent_pairs}")


def test_run_matrix_without_random_baseline_plays_both_directions_only(two_checkpoints):
    """2 real checkpoints only -> exactly the 2 directions of that one pair."""
    print("\nTesting run_matrix() without the random baseline...")

    entries = run_matrix(
        checkpoint_paths=two_checkpoints,
        num_players=4,
        include_random_baseline=False,
        games_per_pairing=4,
        seed=2,
        save_path=None,
        verbose=0,
    )

    assert len(entries) == 2
    agent_opponent_pairs = {(e.agent, e.opponent) for e in entries}
    assert agent_opponent_pairs == {("ckpt_a", "ckpt_b"), ("ckpt_b", "ckpt_a")}
    print(f"✓ 2 directions ran: {agent_opponent_pairs}")


def test_run_matrix_rejects_two_players(two_checkpoints):
    """2-player comparisons have a valid Elo interpretation -- this tool is
    specifically for 3-4 players, and should say so rather than silently
    running a degenerate matrix."""
    with pytest.raises(ValueError):
        run_matrix(checkpoint_paths=two_checkpoints, num_players=2, save_path=None, verbose=0)


def test_run_matrix_saves_and_round_trips_json(two_checkpoints):
    """save_path writes a results.json that round-trips the same entries."""
    print("\nTesting run_matrix() JSON output...")

    with tempfile.TemporaryDirectory() as save_dir:
        entries = run_matrix(
            checkpoint_paths=two_checkpoints,
            num_players=4,
            include_random_baseline=False,
            games_per_pairing=4,
            seed=4,
            save_path=save_dir,
            verbose=0,
        )

        results_file = os.path.join(save_dir, "results.json")
        assert os.path.exists(results_file)
        with open(results_file) as f:
            data = json.load(f)

        assert len(data['entries']) == len(entries) == 2
    print("✓ results.json written and round-trips correctly")


def test_run_matrix_too_few_participants_raises():
    with pytest.raises(ValueError):
        run_matrix(checkpoint_paths={}, num_players=4, include_random_baseline=False,
                   save_path=None, verbose=0)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
