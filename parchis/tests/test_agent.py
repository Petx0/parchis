#!/usr/bin/env python3
"""
Tests for parchis/az/agent.py (docs/AGENT_REBUILD_PLAN.md Part 3 item 9):
wiring encoding + net + search.py into an arena-compatible agent.
"""

import random

import pytest

from parchis.agents.decision_recorder import DecisionRecorder
from parchis.az import agent, search
from parchis.evaluation import arena


def test_depth_must_be_at_least_1():
    """make_search_agent_factory(depth=0) is undefined here by design --
    see the module docstring's "depth=0 is not this module's concern"."""
    print("\nTesting make_search_agent_factory rejects depth < 1...")
    with pytest.raises(ValueError):
        agent.make_search_agent_factory(agent.heuristic_position_evaluator, depth=0)
    print("✓ make_search_agent_factory(depth=0) raises ValueError")


def test_search_agent_plays_full_games_at_depth1():
    """End-to-end smoke test: a search-agent-vs-itself game (depth=1, the
    cheap end) must complete cleanly via arena.play_one_game, across
    several seeds and player counts, exercising bonus chains/six-again/
    three-sixes along the way (not just a few turns)."""
    print("\nTesting the search agent plays full games to completion at depth=1...")

    factory = agent.make_search_agent_factory(agent.heuristic_position_evaluator, depth=1)
    completed = 0
    for num_players in (2, 3):
        for seed in range(3):
            winner_seat = arena.play_one_game(
                {seat: factory for seat in range(num_players)},
                num_players=num_players, max_turns=500, seed=seed + num_players * 100,
            )
            assert winner_seat is not None, (
                f"num_players={num_players} seed={seed}: game did not finish within max_turns"
            )
            completed += 1
    print(f"✓ {completed} full games completed cleanly with the depth=1 search agent")


def test_bonus_decision_never_confused_with_a_fresh_roll(monkeypatch):
    """Regression test for §1.4's mcts.py bug this agent is designed not to
    repeat: every search.search() call made while resolving a bonus must
    have pending_bonus set and roll=None -- never a stale prior dice
    value. Instruments search.search itself to record every call's
    (roll, pending_bonus) across a real game with a real bonus chain."""
    print("\nTesting bonus decisions are never confused with a fresh roll...")

    calls = []
    original_search = search.search

    def recording_search(game, roll=None, pending_bonus=None, consecutive_sixes=0,
                          depth=search.DEFAULT_DEPTH, evaluator=None):
        calls.append((roll, pending_bonus))
        return original_search(game, roll=roll, pending_bonus=pending_bonus,
                                consecutive_sixes=consecutive_sixes, depth=depth, evaluator=evaluator)

    monkeypatch.setattr(search, "search", recording_search)
    # agent.py calls search.search via the module attribute (search.search(...)),
    # so patching the module-level name is visible to it too.
    monkeypatch.setattr("parchis.az.agent.search.search", recording_search)

    factory = agent.make_search_agent_factory(agent.heuristic_position_evaluator, depth=1)

    random.seed(0)
    bonus_calls_seen = 0
    for seed in range(30):
        calls.clear()
        arena.play_one_game(
            {0: factory, 1: factory}, num_players=2, max_turns=400, seed=seed,
        )
        for roll, pending_bonus in calls:
            if pending_bonus is not None:
                bonus_calls_seen += 1
                assert roll is None, (
                    f"Bonus decision incorrectly carried a stale roll={roll} "
                    f"alongside pending_bonus={pending_bonus}"
                )
            else:
                assert roll is not None, "A non-bonus decision must have a real roll"

    assert bonus_calls_seen > 0, (
        "Expected at least one bonus decision across 30 games -- test isn't "
        "exercising the bonus-chain path at all"
    )
    print(f"✓ {bonus_calls_seen} bonus decisions across 30 games, all correctly "
          f"had roll=None and a real pending_bonus")


def test_recording_factory_matches_plain_factory():
    """make_recording_search_agent_factory must be a pure superset of
    make_search_agent_factory's behavior: given the identical seed, it must
    choose the exact same moves turn-by-turn (never merely 'a similarly
    good' move) -- catches the two implementations drifting apart."""
    print("\nTesting make_recording_search_agent_factory matches make_search_agent_factory's moves...")

    plain_factory = agent.make_search_agent_factory(agent.heuristic_position_evaluator, depth=1)
    recorder = DecisionRecorder()
    recording_factory = agent.make_recording_search_agent_factory(
        agent.heuristic_position_evaluator, depth=1, recorder=recorder,
    )

    for seed in range(5):
        plain_winner = arena.play_one_game(
            {0: plain_factory, 1: plain_factory}, num_players=2, max_turns=400, seed=seed,
        )
        recorder.records.clear()
        recording_winner = arena.play_one_game(
            {0: recording_factory, 1: recording_factory}, num_players=2, max_turns=400, seed=seed,
        )
        assert plain_winner == recording_winner, (
            f"seed={seed}: plain factory winner={plain_winner}, recording factory winner={recording_winner}"
        )
    print("✓ recording factory reproduces the plain factory's exact outcomes across 5 seeds")


def test_recording_factory_captures_consistent_move_values():
    """Every DecisionRecord's move_values must have one entry per legal
    move actually available, and chosen_piece_id must always be a key of
    move_values (the move actually chosen was necessarily a candidate)."""
    print("\nTesting recorded move_values are internally consistent...")

    recorder = DecisionRecorder()
    factory = agent.make_recording_search_agent_factory(
        agent.heuristic_position_evaluator, depth=1, recorder=recorder,
    )
    arena.play_one_game({0: factory, 1: factory}, num_players=2, max_turns=400, seed=3)

    assert recorder.records, "Expected at least one recorded decision"
    for record in recorder.records:
        assert record.kind == "search"
        assert record.move_values, "move_values must be non-empty for every recorded decision"
        assert record.chosen_piece_id in record.move_values, (
            f"chosen_piece_id={record.chosen_piece_id} not among move_values keys {list(record.move_values)}"
        )
        assert record.root_value is not None
    print(f"✓ {len(recorder.records)} recorded decisions, all internally consistent")


if __name__ == '__main__':
    test_depth_must_be_at_least_1()
    test_search_agent_plays_full_games_at_depth1()
    test_bonus_decision_never_confused_with_a_fresh_roll()
    test_recording_factory_matches_plain_factory()
    test_recording_factory_captures_consistent_move_values()
    print("\nAll agent tests passed!")
