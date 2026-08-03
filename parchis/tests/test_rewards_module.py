#!/usr/bin/env python3
"""
Unit tests for parchis/rl/rewards.py -- one test per extracted term, per
docs/CODE_REVIEW.md's rebuild target ("move the reward function into its
own module with named constants and one unit test per term"). See
parchis/tests/test_new_rewards.py for end-to-end ParchisEnv-level
regression coverage of the same formulas.
"""

import pytest

from parchis.game.player import Player
from parchis.rl import rewards


def _player_with_piece_states(color, states):
    """Build a Player and force its 4 pieces into the given states.
    states is a list of 4 dicts, each with keys among
    {finished, in_base, position}."""
    player = Player(0, color)
    for piece, state in zip(player.pieces, states):
        if "finished" in state:
            piece.finished = state["finished"]
        if "in_base" in state:
            piece.in_base = state["in_base"]
        if "position" in state:
            piece.position = state["position"]
    return player


# ── calculate_normalized_progress ──────────────────────────────────────

def test_calculate_normalized_progress_in_base():
    player = _player_with_piece_states("YELLOW", [
        {"finished": False, "in_base": True, "position": None},
        {"finished": False, "in_base": True, "position": None},
        {"finished": False, "in_base": True, "position": None},
        {"finished": False, "in_base": True, "position": None},
    ])
    assert rewards.calculate_normalized_progress(player) == 0.0


def test_calculate_normalized_progress_on_board():
    player = _player_with_piece_states("YELLOW", [
        {"finished": False, "in_base": False, "position": 38},  # halfway: 38/76 = 0.5
        {"finished": False, "in_base": True, "position": None},
        {"finished": False, "in_base": True, "position": None},
        {"finished": False, "in_base": True, "position": None},
    ])
    # (0.5 + 0 + 0 + 0) / 4
    assert abs(rewards.calculate_normalized_progress(player) - 0.125) < 1e-9


def test_calculate_normalized_progress_finished():
    player = _player_with_piece_states("YELLOW", [
        {"finished": True, "in_base": False, "position": 76},
        {"finished": False, "in_base": True, "position": None},
        {"finished": False, "in_base": True, "position": None},
        {"finished": False, "in_base": True, "position": None},
    ])
    assert abs(rewards.calculate_normalized_progress(player) - 0.25) < 1e-9


def test_calculate_normalized_progress_mixed_average():
    player = _player_with_piece_states("YELLOW", [
        {"finished": True, "in_base": False, "position": 76},   # 1.0
        {"finished": False, "in_base": False, "position": 38},  # 0.5
        {"finished": False, "in_base": True, "position": None},  # 0.0
        {"finished": False, "in_base": True, "position": None},  # 0.0
    ])
    expected = (1.0 + 0.5 + 0.0 + 0.0) / 4.0
    assert abs(rewards.calculate_normalized_progress(player) - expected) < 1e-9


# ── combine_opponent_deltas ─────────────────────────────────────────────

def test_combine_opponent_deltas_mean_matches_manual_average_two_opponents():
    deltas = {1: 0.05, 2: -0.02}
    start_progress = {1: 0.3, 2: 0.1}
    expected = sum(deltas.values()) / len(deltas)
    assert rewards.combine_opponent_deltas(deltas, start_progress, weighting="mean") == expected


def test_combine_opponent_deltas_mean_matches_manual_average_three_opponents():
    deltas = {1: 0.05, 2: -0.02, 3: 0.10}
    start_progress = {1: 0.3, 2: 0.1, 3: 0.6}
    expected = sum(deltas.values()) / len(deltas)
    assert rewards.combine_opponent_deltas(deltas, start_progress, weighting="mean") == expected


def test_combine_opponent_deltas_mean_empty_returns_zero():
    assert rewards.combine_opponent_deltas({}, {}, weighting="mean") == 0.0


def test_combine_opponent_deltas_leader_picks_highest_start_progress():
    # Opponent 2 is the leader (highest start progress) but has a NEGATIVE
    # delta this cycle, while opponent 3 (not the leader) has a large
    # positive delta. The result must be opponent 2's delta, proving this
    # is keyed on start-of-cycle progress, not "whoever gained the most".
    deltas = {1: 0.01, 2: -0.03, 3: 0.20}
    start_progress = {1: 0.2, 2: 0.8, 3: 0.5}
    result = rewards.combine_opponent_deltas(deltas, start_progress, weighting="leader")
    assert result == -0.03


def test_combine_opponent_deltas_invalid_weighting_raises():
    with pytest.raises(ValueError):
        rewards.combine_opponent_deltas({1: 0.1}, {1: 0.5}, weighting="bogus")


# ── compute_reward ───────────────────────────────────────────────────────

def test_compute_reward_progress_delta_formula():
    my_delta = 0.05
    combined_opponent_delta = 0.02
    opponent_weight = 0.5
    reward, progress_delta = rewards.compute_reward(
        "progress_delta", my_delta, combined_opponent_delta, opponent_weight,
        terminated=False, agent_won=False,
    )
    expected = my_delta - opponent_weight * combined_opponent_delta
    assert reward == expected
    assert progress_delta == expected


def test_compute_reward_win_loss_win():
    reward, _ = rewards.compute_reward(
        "win_loss", my_delta=0.1, combined_opponent_delta=0.0, opponent_weight=0.5,
        terminated=True, agent_won=True,
    )
    assert reward == rewards.WIN_REWARD


def test_compute_reward_win_loss_loss():
    reward, _ = rewards.compute_reward(
        "win_loss", my_delta=0.1, combined_opponent_delta=0.0, opponent_weight=0.5,
        terminated=True, agent_won=False,
    )
    assert reward == rewards.LOSS_REWARD


def test_compute_reward_win_loss_midgame_zero():
    reward, _ = rewards.compute_reward(
        "win_loss", my_delta=0.1, combined_opponent_delta=0.05, opponent_weight=0.5,
        terminated=False, agent_won=False,
    )
    assert reward == 0.0


def test_compute_reward_win_loss_shaped_terminal():
    reward, _ = rewards.compute_reward(
        "win_loss_shaped", my_delta=0.1, combined_opponent_delta=0.0, opponent_weight=0.5,
        terminated=True, agent_won=True,
    )
    assert reward == rewards.WIN_REWARD


def test_compute_reward_win_loss_shaped_midgame_scaled():
    my_delta = 0.1
    combined_opponent_delta = 0.02
    opponent_weight = 0.5
    reward, progress_delta = rewards.compute_reward(
        "win_loss_shaped", my_delta, combined_opponent_delta, opponent_weight,
        terminated=False, agent_won=False,
    )
    expected_progress_delta = my_delta - opponent_weight * combined_opponent_delta
    assert progress_delta == expected_progress_delta
    assert reward == rewards.WIN_LOSS_SHAPED_MIDGAME_SCALE * expected_progress_delta


def test_compute_reward_invalid_reward_type_raises():
    with pytest.raises(ValueError):
        rewards.compute_reward(
            "bogus", my_delta=0.1, combined_opponent_delta=0.0, opponent_weight=0.5,
            terminated=False, agent_won=False,
        )


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
