#!/usr/bin/env python3
"""
Tests for parchis/evaluation/elo_ladder.py -- the round-robin checkpoint
Elo ladder introduced in docs/RL_DESIGN_REVIEW.md Phase 4.

Drives run_ladder() with tiny real trained models (small total_timesteps,
no mocks), matching the established convention in test_eval_issue.py /
test_evaluate.py.
"""
import os
import json
import tempfile

import pytest
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from parchis.rl.env import ParchisEnv
from parchis.evaluation import elo
from parchis.evaluation.elo_ladder import run_ladder, RANDOM_PARTICIPANT


def mask_fn(env):
    return env.unwrapped._get_info()['action_masks']


def _train_tiny_model():
    env = ParchisEnv(num_players=2)
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
def three_checkpoints():
    """Three tiny trained 2-player checkpoints, saved to a shared tempdir
    for the whole module (training is the expensive part; reused across
    tests in this file)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {}
        for name in ("ckpt_a", "ckpt_b", "ckpt_c"):
            model = _train_tiny_model()
            path = os.path.join(tmpdir, name)
            model.save(path)
            paths[name] = path
        yield paths


def test_run_ladder_with_random_baseline_covers_all_pairings(three_checkpoints):
    """3 checkpoints + the random baseline -> 4 participants -> 6 pairings
    (4 choose 2), every rating moved off its initial value."""
    print("\nTesting run_ladder() with the random baseline included...")

    ratings, pairing_results = run_ladder(
        checkpoint_paths=three_checkpoints,
        include_random_baseline=True,
        games_per_pairing=6,
        seed=1,
        save_path=None,
        verbose=0,
    )

    assert len(pairing_results) == 6  # 4 participants, 4*3/2 pairings
    assert set(ratings.keys()) == {"ckpt_a", "ckpt_b", "ckpt_c", RANDOM_PARTICIPANT}
    assert any(r != elo.DEFAULT_INITIAL_RATING for r in ratings.values()), (
        "At least one participant's rating should have moved from the default initial rating"
    )
    for pairing in pairing_results:
        assert pairing.games == 6
        assert 0 <= pairing.wins_a <= pairing.games
        lower, upper = pairing.win_rate_a_ci
        assert 0.0 <= lower <= pairing.win_rate_a <= upper <= 1.0
    print(f"✓ 6 pairings ran, ratings: {ratings}")


def test_run_ladder_without_random_baseline(three_checkpoints):
    """3 checkpoints only -> 3 participants -> 3 pairings, no 'random' key."""
    print("\nTesting run_ladder() without the random baseline...")

    ratings, pairing_results = run_ladder(
        checkpoint_paths=three_checkpoints,
        include_random_baseline=False,
        games_per_pairing=4,
        seed=2,
        save_path=None,
        verbose=0,
    )

    assert len(pairing_results) == 3  # 3 participants, 3*2/2 pairings
    assert RANDOM_PARTICIPANT not in ratings
    print(f"✓ 3 pairings ran, ratings: {ratings}")


def test_run_ladder_random_baseline_result_is_correctly_inverted(three_checkpoints):
    """When the random baseline is participant A in a pairing, its win
    rate/wins must be inverted from evaluate_agent's real-checkpoint
    perspective, not silently reused as-is (see _play_pairing's docstring)."""
    print("\nTesting random-baseline result inversion...")

    # A single real checkpoint + baseline gives exactly one pairing, with a
    # deterministic (checkpoint, "random") or ("random", checkpoint) order
    # depending on the seed -- either order must produce a self-consistent
    # (wins_a, games, win_rate_a) triple.
    single = {"ckpt_a": three_checkpoints["ckpt_a"]}
    ratings, pairing_results = run_ladder(
        checkpoint_paths=single,
        include_random_baseline=True,
        games_per_pairing=8,
        seed=3,
        save_path=None,
        verbose=0,
    )

    assert len(pairing_results) == 1
    pairing = pairing_results[0]
    assert {pairing.participant_a, pairing.participant_b} == {"ckpt_a", RANDOM_PARTICIPANT}
    assert pairing.win_rate_a == pytest.approx(pairing.wins_a / pairing.games)
    print(f"✓ Random-baseline pairing self-consistent: {pairing}")


def test_run_ladder_saves_and_round_trips_json(three_checkpoints):
    """save_path writes a results.json that round-trips the same ratings."""
    print("\nTesting run_ladder() JSON output...")

    with tempfile.TemporaryDirectory() as save_dir:
        ratings, _ = run_ladder(
            checkpoint_paths=three_checkpoints,
            include_random_baseline=True,
            games_per_pairing=4,
            seed=4,
            save_path=save_dir,
            verbose=0,
        )

        results_file = os.path.join(save_dir, "results.json")
        assert os.path.exists(results_file)
        with open(results_file) as f:
            data = json.load(f)

        assert data['ratings'] == ratings
        assert len(data['pairings']) == 6
    print("✓ results.json written and round-trips correctly")


def test_run_ladder_too_few_participants_raises():
    with pytest.raises(ValueError):
        run_ladder(checkpoint_paths={}, include_random_baseline=False, save_path=None, verbose=0)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
