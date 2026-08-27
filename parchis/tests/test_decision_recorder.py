#!/usr/bin/env python3
"""
Tests for parchis/agents/decision_recorder.py: the shared recording types
behind make_recording_search_agent_factory (parchis/az/agent.py) and
make_recording_heuristic_agent_factory (parchis/agents/heuristic.py).
"""

from parchis.agents.decision_recorder import DecisionRecorder


def test_next_index_increments_within_a_turn():
    print("\nTesting next_index increments within the same turn...")
    rec = DecisionRecorder()
    indices = [rec.next_index(turn_number=3) for _ in range(4)]
    assert indices == [0, 1, 2, 3], f"Expected a clean 0..3 run, got {indices}"
    print(f"✓ next_index produced {indices} across 4 calls within turn 3")


def test_next_index_resets_on_turn_change():
    print("\nTesting next_index resets when turn_number changes...")
    rec = DecisionRecorder()
    seq = []
    for turn in (1, 1, 2, 2, 2, 5, 5):
        seq.append(rec.next_index(turn))
    assert seq == [0, 1, 0, 1, 2, 0, 1], f"Expected per-turn resets, got {seq}"
    print(f"✓ next_index correctly reset at each turn boundary: {seq}")


def test_next_index_no_gaps_or_dupes_across_many_turns():
    print("\nTesting next_index has no gaps or duplicates across many turns...")
    rec = DecisionRecorder()
    seen = {}
    for turn in [t for t in range(1, 21) for _ in range(t % 4 + 1)]:  # variable decisions/turn
        idx = rec.next_index(turn)
        seen.setdefault(turn, []).append(idx)
    for turn, indices in seen.items():
        assert indices == list(range(len(indices))), (
            f"Turn {turn}: expected a gap-free 0..N-1 run, got {indices}"
        )
    print(f"✓ No gaps/duplicates across {len(seen)} turns")


if __name__ == '__main__':
    test_next_index_increments_within_a_turn()
    test_next_index_resets_on_turn_change()
    test_next_index_no_gaps_or_dupes_across_many_turns()
    print("\nAll decision_recorder tests passed!")
