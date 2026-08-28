#!/usr/bin/env python3
"""
Tests for parchis/agents/agent_spec.py: the shared "SPEC string -> agent"
grammar behind parchis.visualization.play_instrumented_game's --agent flag
and parchis.evaluation.ladder's --rung flag.
"""

import pytest

from parchis.agents import agent_spec, heuristic
from parchis.az.selfplay import random_factory
from parchis.evaluation import arena


def test_parse_spec_random():
    print("\nTesting parse_spec('random')...")
    kind, params, label = agent_spec.parse_spec('random')
    assert kind == 'random' and params is None and label == 'random'
    print("✓ 'random' parses to kind='random', params=None")


def test_parse_spec_heuristic_tuned_and_default():
    print("\nTesting parse_spec('heuristic:tuned'/'heuristic:default')...")
    kind, params, label = agent_spec.parse_spec('heuristic:tuned')
    assert kind == 'heuristic'
    assert (params == heuristic.TUNED_WEIGHTS).all()
    assert label == 'heuristic:tuned'

    kind, params, label = agent_spec.parse_spec('heuristic:default')
    assert (params == heuristic.DEFAULT_WEIGHTS).all()

    # Bare 'heuristic' (no ':which') defaults to tuned.
    kind, params, _label = agent_spec.parse_spec('heuristic')
    assert (params == heuristic.TUNED_WEIGHTS).all()
    print("✓ 'heuristic:tuned'/'heuristic:default'/bare 'heuristic' all parse correctly")


def test_parse_spec_unknown_which_raises():
    print("\nTesting parse_spec rejects an unknown heuristic variant...")
    with pytest.raises(ValueError):
        agent_spec.parse_spec('heuristic:bogus')
    print("✓ raises ValueError")


def test_parse_spec_checkpoint_requires_run_dir():
    print("\nTesting parse_spec rejects a bare 'checkpoint' with no run_dir...")
    with pytest.raises(ValueError):
        agent_spec.parse_spec('checkpoint')
    print("✓ raises ValueError")


def test_parse_spec_unknown_kind_raises():
    print("\nTesting parse_spec rejects an unknown kind...")
    with pytest.raises(ValueError):
        agent_spec.parse_spec('not_a_real_kind')
    print("✓ raises ValueError")


def test_build_factory_random_and_heuristic_play_a_full_game():
    """End-to-end smoke test: build_factory's output for 'random' and
    'heuristic' must be genuine arena-compatible factories."""
    print("\nTesting build_factory produces real, playable arena factories...")
    _kind_r, params_r, _label_r = agent_spec.parse_spec('random')
    _kind_h, params_h, _label_h = agent_spec.parse_spec('heuristic:tuned')

    random_f = agent_spec.build_factory('random', params_r)
    heuristic_f = agent_spec.build_factory('heuristic', params_h)
    assert random_f is random_factory

    winner = arena.play_one_game(
        {0: random_f, 1: heuristic_f}, num_players=2, max_turns=400, seed=0,
    )
    assert winner is not None
    print(f"✓ random vs heuristic:tuned completed a full game (winner seat {winner})")


def test_build_factory_unknown_kind_raises():
    print("\nTesting build_factory rejects an unknown kind...")
    with pytest.raises(ValueError):
        agent_spec.build_factory('not_a_real_kind', None)
    print("✓ raises ValueError")


if __name__ == '__main__':
    test_parse_spec_random()
    test_parse_spec_heuristic_tuned_and_default()
    test_parse_spec_unknown_which_raises()
    test_parse_spec_checkpoint_requires_run_dir()
    test_parse_spec_unknown_kind_raises()
    test_build_factory_random_and_heuristic_play_a_full_game()
    test_build_factory_unknown_kind_raises()
    print("\nAll agent_spec tests passed!")
