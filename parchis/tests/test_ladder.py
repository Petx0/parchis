#!/usr/bin/env python3
"""
Tests for parchis/evaluation/ladder.py: round-robin duplicate-match
comparisons across a fixed set of rungs, appended to runs/pairings.jsonl.
"""

import json

import pytest

from parchis.agents import heuristic
from parchis.az.selfplay import random_factory
from parchis.evaluation import ladder


def _tiny_rungs():
    return {
        "random": random_factory,
        "heuristic_tuned": heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS),
        "heuristic_default": heuristic.make_heuristic_agent_factory(heuristic.DEFAULT_WEIGHTS),
    }


def test_run_ladder_covers_every_pair_and_appends_jsonl(tmp_path):
    print("\nTesting run_ladder covers every rung pair and appends valid JSONL...")
    pairings_path = tmp_path / "pairings.jsonl"
    results = ladder.run_ladder(
        _tiny_rungs(), num_players=2, n_pairs=3, max_turns=300, seed=0,
        pairings_path=str(pairings_path), verbose=0,
    )
    assert len(results) == 3  # 3 rungs -> 3 unordered pairs

    seen_pairs = {frozenset((r["participant_a"], r["participant_b"])) for r in results}
    expected_pairs = {
        frozenset(("random", "heuristic_tuned")),
        frozenset(("random", "heuristic_default")),
        frozenset(("heuristic_tuned", "heuristic_default")),
    }
    assert seen_pairs == expected_pairs

    with open(pairings_path) as f:
        lines = [json.loads(line) for line in f]
    assert lines == results
    for record in lines:
        assert record["n_games"] == 3 * 2  # n_pairs * num_players
        assert 0 <= record["wins_a"] <= record["n_games"]
        lower, upper = record["win_rate_a_ci"]
        assert 0.0 <= lower <= record["win_rate_a"] <= upper <= 1.0
    print(f"✓ {len(results)} pairings written, covering all {len(expected_pairs)} rung pairs")


def test_run_ladder_appends_without_overwriting_existing_content(tmp_path):
    print("\nTesting run_ladder appends to an existing pairings.jsonl rather than overwriting...")
    pairings_path = tmp_path / "pairings.jsonl"
    pairings_path.write_text('{"participant_a": "old", "participant_b": "stuff", "fake": true}\n')

    ladder.run_ladder(
        _tiny_rungs(), num_players=2, n_pairs=2, max_turns=300, seed=1,
        pairings_path=str(pairings_path), verbose=0,
    )
    with open(pairings_path) as f:
        lines = [json.loads(line) for line in f]
    assert lines[0] == {"participant_a": "old", "participant_b": "stuff", "fake": True}
    assert len(lines) == 1 + 3  # the pre-existing line + 3 new pairings
    print(f"✓ pre-existing line preserved, {len(lines) - 1} new pairings appended")


def test_run_ladder_creates_parent_directories(tmp_path):
    print("\nTesting run_ladder creates missing parent directories...")
    pairings_path = tmp_path / "nested" / "dir" / "pairings.jsonl"
    ladder.run_ladder(
        _tiny_rungs(), num_players=2, n_pairs=2, max_turns=300, seed=2,
        pairings_path=str(pairings_path), verbose=0,
    )
    assert pairings_path.exists()
    print("✓ nested parent directories created as needed")


def test_run_ladder_requires_at_least_two_rungs(tmp_path):
    print("\nTesting run_ladder rejects fewer than 2 rungs...")
    with pytest.raises(ValueError):
        ladder.run_ladder({"only_one": random_factory}, pairings_path=str(tmp_path / "p.jsonl"))
    print("✓ raises ValueError")


def test_parse_rung_spec():
    print("\nTesting _parse_rung_spec's NAME=SPEC grammar...")
    name, factory = ladder._parse_rung_spec("my_random=random")
    assert name == "my_random"
    assert factory is random_factory

    with pytest.raises(ValueError):
        ladder._parse_rung_spec("no_equals_sign_here")
    print("✓ NAME=SPEC parses correctly, malformed input raises ValueError")


if __name__ == '__main__':
    print("Most tests in this file need tmp_path -- run via pytest.")
    test_parse_rung_spec()
