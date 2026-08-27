#!/usr/bin/env python3
"""
Tests for parchis/visualization/agentinfo_io.py: the sidecar JSON carrying
per-decision agent value data alongside a GameLogger move log.
"""

import numpy as np

from parchis.agents.decision_recorder import DecisionRecord, DecisionRecorder
from parchis.visualization import agentinfo_io


def test_agentinfo_path_for_convention():
    print("\nTesting agentinfo_path_for's naming convention...")
    path = agentinfo_io.agentinfo_path_for("logs/game_20260827_100159_BLUE.json")
    assert str(path) == "logs/game_20260827_100159_BLUE.agentinfo.json"
    print(f"✓ {path}")


def test_save_and_load_round_trips_numpy_arrays(tmp_path):
    print("\nTesting save_agentinfo/load_agentinfo round-trip numpy arrays as lists...")
    recorder = DecisionRecorder()
    recorder.records.append(DecisionRecord(
        seat=0, turn_number=3, decision_index_in_turn=0, kind="search",
        chosen_piece_id=0,
        root_value=np.array([0.52, 0.48]),
        move_values={0: np.array([0.55, 0.45]), 2: np.array([0.5, 0.5])},
    ))
    log_path = tmp_path / "game_x.json"
    agentinfo_path = agentinfo_io.save_agentinfo(
        {0: recorder}, {0: "champion (search depth=1)"}, str(log_path), num_players=2,
    )
    assert agentinfo_path == str(agentinfo_io.agentinfo_path_for(str(log_path)))

    loaded = agentinfo_io.load_agentinfo(str(log_path))
    assert loaded["schema_version"] == agentinfo_io.SCHEMA_VERSION
    assert loaded["num_players"] == 2
    seat0 = loaded["seats"]["0"]
    assert seat0["agent_label"] == "champion (search depth=1)"
    decision = seat0["decisions"][0]
    assert np.allclose(decision["root_value"], [0.52, 0.48])
    assert np.allclose(decision["move_values"]["0"], [0.55, 0.45])
    assert np.allclose(decision["move_values"]["2"], [0.5, 0.5])
    assert decision["chosen_piece_id"] == 0
    print("✓ numpy arrays survive the JSON round-trip as lists reloadable via np.allclose")


def test_load_agentinfo_returns_none_when_missing(tmp_path):
    print("\nTesting load_agentinfo returns None for a nonexistent sidecar...")
    assert agentinfo_io.load_agentinfo(str(tmp_path / "no_such_game.json")) is None
    print("✓ load_agentinfo gracefully returns None")


def _roll(legal_moves_count, **kwargs):
    return {"legal_moves_count": legal_moves_count, **kwargs}


def test_roll_had_decision_table():
    print("\nTesting _roll_had_decision across bonus/plain/penalty cases...")
    cases = [
        (_roll(2), True, "plain roll with legal moves"),
        (_roll(0), False, "plain roll with no legal moves"),
        (_roll(0, bonus_type="capture_bonus", bonus_squares=20), False, "bonus roll with no legal moves"),
        (_roll(1, bonus_type="finish_bonus", bonus_squares=10), True, "bonus roll with a legal move"),
        (_roll(0), False, "three-sixes penalty roll (legal_moves_count=0, no choose_move call)"),
    ]
    for roll_data, expected, label in cases:
        assert agentinfo_io._roll_had_decision(roll_data) == expected, label
    print(f"✓ all {len(cases)} cases matched")


def test_build_seat_by_player_id_follows_turn_order_not_player_id_value():
    """The whole point of build_seat_by_player_id: seat is "order of first
    appearance in turn order," which can be a completely different number
    from player_id (e.g. the dice-determined starting player's player_id
    need not be 0 -- see Game.__init__'s post-rotation comment and this
    module's docstring)."""
    print("\nTesting build_seat_by_player_id derives seats from turn order, not player_id values...")
    turns = [
        {"player_id": 5, "turn_number": 1},  # first to act -> seat 0, despite player_id=5
        {"player_id": 2, "turn_number": 2},  # second to act -> seat 1, despite player_id=2
        {"player_id": 5, "turn_number": 3},  # player_id 5 acts again -> still seat 0
        {"player_id": 2, "turn_number": 4},
    ]
    seat_by_player_id = agentinfo_io.build_seat_by_player_id(turns)
    assert seat_by_player_id == {5: 0, 2: 1}, (
        f"Expected seat to follow first-appearance ORDER, not player_id's numeric value, "
        f"got {seat_by_player_id}"
    )
    print(f"✓ {seat_by_player_id} -- seat assigned by turn order, independent of player_id's value")


def test_decision_for_roll_none_cases():
    print("\nTesting decision_for_roll's None-returning graceful-degradation cases...")
    turn_data = {
        "turn_number": 3, "player_id": 7,  # player_id deliberately != its seat (0)
        "rolls": [_roll(2), _roll(0)],
    }
    seat_by_player_id = {7: 0}

    # No sidecar at all.
    assert agentinfo_io.decision_for_roll(turn_data, 0, None, seat_by_player_id) is None

    agentinfo_data = {"schema_version": 1, "num_players": 2, "seats": {}}
    # Sidecar exists, but this seat was never instrumented.
    assert agentinfo_io.decision_for_roll(turn_data, 0, agentinfo_data, seat_by_player_id) is None

    agentinfo_data = {"schema_version": 1, "num_players": 2, "seats": {
        "0": {"agent_label": "x", "decisions": [
            {"turn_number": 3, "decision_index_in_turn": 0, "kind": "search",
             "chosen_piece_id": 0, "root_value": [0.5, 0.5], "move_values": {"0": [0.5, 0.5]}},
        ]},
    }}
    # This roll had no decision (legal_moves_count == 0).
    assert agentinfo_io.decision_for_roll(turn_data, 1, agentinfo_data, seat_by_player_id) is None
    # This roll DID have a decision -- must be found via player_id=7 -> seat 0, NOT via
    # player_id=7 taken literally (there is no "seats"."7" key at all).
    found = agentinfo_io.decision_for_roll(turn_data, 0, agentinfo_data, seat_by_player_id)
    assert found is not None and found["chosen_piece_id"] == 0

    # An unmapped player_id (missing from seat_by_player_id) must also degrade to None,
    # not raise.
    assert agentinfo_io.decision_for_roll(turn_data, 0, agentinfo_data, {}) is None
    print("✓ decision_for_roll returns None in every degradation case, and finds the real "
          "decision via the player_id->seat mapping (not player_id taken literally)")


def test_decision_for_roll_indexes_multiple_decisions_in_one_turn():
    print("\nTesting decision_for_roll correctly counts decision_index_in_turn across several rolls...")
    turn_data = {
        "turn_number": 5, "player_id": 9,  # again deliberately != its seat (1)
        "rolls": [_roll(2), _roll(0), _roll(3, bonus_type="capture_bonus", bonus_squares=20)],
    }
    seat_by_player_id = {9: 1}
    agentinfo_data = {"schema_version": 1, "num_players": 2, "seats": {
        "1": {"agent_label": "x", "decisions": [
            {"turn_number": 5, "decision_index_in_turn": 0, "kind": "heuristic",
             "chosen_piece_id": 2, "move_scores": {"2": 1.0}},
            {"turn_number": 5, "decision_index_in_turn": 1, "kind": "heuristic",
             "chosen_piece_id": 3, "move_scores": {"3": 0.4}},
        ]},
    }}
    assert agentinfo_io.decision_for_roll(turn_data, 0, agentinfo_data, seat_by_player_id)["chosen_piece_id"] == 2
    assert agentinfo_io.decision_for_roll(turn_data, 1, agentinfo_data, seat_by_player_id) is None  # legal_moves_count=0
    assert agentinfo_io.decision_for_roll(turn_data, 2, agentinfo_data, seat_by_player_id)["chosen_piece_id"] == 3
    print("✓ decision_index_in_turn correctly skips the no-decision roll in between")


if __name__ == '__main__':
    test_agentinfo_path_for_convention()
    test_roll_had_decision_table()
    test_build_seat_by_player_id_follows_turn_order_not_player_id_value()
    test_decision_for_roll_none_cases()
    test_decision_for_roll_indexes_multiple_decisions_in_one_turn()
    print("\nAll agentinfo_io tests passed! (run via pytest for the tmp_path-based tests)")
