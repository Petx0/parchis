#!/usr/bin/env python3
"""
Unit tests for parchis/evaluation/group_comparison.py -- the Phase 5
group-vs-group pooling helper for a completed elo_ladder.py run (see
docs/RL_DESIGN_REVIEW.md Phase 5).
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from parchis.rl.env import ParchisEnv
from parchis.evaluation import group_comparison
from parchis.evaluation.elo_ladder import run_ladder


def _pairing(a, b, games, wins_a):
    return {'participant_a': a, 'participant_b': b, 'games': games, 'wins_a': wins_a}


# ── aggregate_group_win_rate ──────────────────────────────────────────────

def test_aggregate_group_win_rate_pools_same_order_pairings():
    pairings = [
        _pairing('base_1', 'redesigned_1', 10, 3),
        _pairing('base_1', 'redesigned_2', 10, 4),
    ]
    result = group_comparison.aggregate_group_win_rate(
        pairings, group_a_names=['base_1'], group_b_names=['redesigned_1', 'redesigned_2']
    )
    assert result['wins'] == 7
    assert result['games'] == 20
    assert result['win_rate'] == pytest.approx(0.35)
    assert result['n_pairings'] == 2


def test_aggregate_group_win_rate_inverts_swapped_order_pairings():
    # participant_a here is redesigned_1 (group B), so group A's (base_1's)
    # wins are the complement.
    pairings = [_pairing('redesigned_1', 'base_1', 10, 6)]
    result = group_comparison.aggregate_group_win_rate(
        pairings, group_a_names=['base_1'], group_b_names=['redesigned_1']
    )
    assert result['wins'] == 4  # base_1 won 10 - 6 = 4
    assert result['games'] == 10
    assert result['win_rate'] == pytest.approx(0.4)


def test_aggregate_group_win_rate_mixed_order_pairings_pool_correctly():
    pairings = [
        _pairing('base_1', 'redesigned_1', 10, 7),   # A wins 7
        _pairing('redesigned_1', 'base_2', 10, 2),    # A (base_2) wins 10-2=8
    ]
    result = group_comparison.aggregate_group_win_rate(
        pairings, group_a_names=['base_1', 'base_2'], group_b_names=['redesigned_1']
    )
    assert result['wins'] == 15
    assert result['games'] == 20
    assert result['n_pairings'] == 2


def test_aggregate_group_win_rate_ignores_within_group_and_unrelated_pairings():
    pairings = [
        _pairing('base_1', 'base_2', 10, 5),          # within-group A: ignored
        _pairing('redesigned_1', 'redesigned_2', 10, 5),  # within-group B: ignored
        _pairing('base_1', 'random', 10, 6),          # involves neither group fully: ignored
        _pairing('base_1', 'redesigned_1', 10, 4),    # the only real cross-group pairing
    ]
    result = group_comparison.aggregate_group_win_rate(
        pairings, group_a_names=['base_1', 'base_2'], group_b_names=['redesigned_1', 'redesigned_2']
    )
    assert result['n_pairings'] == 1
    assert result['wins'] == 4
    assert result['games'] == 10


def test_aggregate_group_win_rate_ci_contains_point_estimate():
    pairings = [_pairing('base_1', 'redesigned_1', 20, 8)]
    result = group_comparison.aggregate_group_win_rate(
        pairings, group_a_names=['base_1'], group_b_names=['redesigned_1']
    )
    lower, upper = result['ci']
    assert lower <= result['win_rate'] <= upper


def test_aggregate_group_win_rate_overlapping_groups_raises():
    with pytest.raises(ValueError):
        group_comparison.aggregate_group_win_rate(
            [_pairing('a', 'b', 10, 5)], group_a_names=['a', 'shared'], group_b_names=['shared', 'b']
        )


def test_aggregate_group_win_rate_no_cross_group_pairing_raises():
    pairings = [_pairing('base_1', 'base_2', 10, 5)]
    with pytest.raises(ValueError):
        group_comparison.aggregate_group_win_rate(
            pairings, group_a_names=['base_1', 'base_2'], group_b_names=['redesigned_1']
        )


def test_aggregate_group_win_rate_empty_pairings_raises():
    with pytest.raises(ValueError):
        group_comparison.aggregate_group_win_rate([], group_a_names=['a'], group_b_names=['b'])


# ── end-to-end: real (tiny) elo_ladder.py output, not synthetic dicts ─────

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


def test_aggregate_group_win_rate_against_real_elo_ladder_output():
    """Dry-run of the actual Phase 5 Stage 4 path (docs/RL_DESIGN_REVIEW.md
    plan verification item 3): a real (tiny, seconds-long) elo_ladder.py
    run's results.json, not a hand-built synthetic dict -- confirms the
    field names/shapes this module assumes actually match what
    elo_ladder.py produces, before this is ever pointed at the real
    multi-hour Phase 5 output."""
    print("\nTesting group_comparison against a real elo_ladder.py run...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_paths = {}
        for name in ("base_1", "base_2", "redesigned_1", "redesigned_2"):
            model = _train_tiny_model()
            path = os.path.join(tmpdir, name)
            model.save(path)
            checkpoint_paths[name] = path

        save_path = os.path.join(tmpdir, "elo_results")
        ratings, _ = run_ladder(
            checkpoint_paths=checkpoint_paths,
            include_random_baseline=False,
            games_per_pairing=4,
            seed=1,
            save_path=save_path,
            verbose=0,
        )

        results_file = os.path.join(save_path, "results.json")
        with open(results_file) as f:
            data = json.load(f)

        result = group_comparison.aggregate_group_win_rate(
            data['pairings'],
            group_a_names=['base_1', 'base_2'],
            group_b_names=['redesigned_1', 'redesigned_2'],
        )
        assert result['n_pairings'] == 4  # 2x2 cross-group pairings
        assert result['games'] == 4 * 4  # games_per_pairing=4, 4 pairings
        assert 0.0 <= result['ci'][0] <= result['win_rate'] <= result['ci'][1] <= 1.0

        # Also exercise the actual CLI entrypoint end-to-end.
        cli = subprocess.run(
            [sys.executable, "-m", "parchis.evaluation.group_comparison",
             "--results-json", results_file,
             "--group-a", "base_1", "base_2",
             "--group-b", "redesigned_1", "redesigned_2"],
            capture_output=True, text=True,
        )
        assert cli.returncode == 0, cli.stderr
        assert "GROUP COMPARISON" in cli.stdout
        print("✓ group_comparison works end-to-end against a real elo_ladder.py run")


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
