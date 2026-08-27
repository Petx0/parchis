#!/usr/bin/env python3
"""
Tests for parchis/visualization/instrumented_play.py::play_and_record --
the missing "play a real game with real agents AND log it" capability.
"""

from parchis.agents import heuristic
from parchis.az.agent import heuristic_position_evaluator
from parchis.utils.logger import GameLogger
from parchis.visualization import agentinfo_io
from parchis.visualization.instrumented_play import play_and_record


def test_play_and_record_produces_both_files_for_one_instrumented_seat(tmp_path):
    print("\nTesting play_and_record with one instrumented seat + one random seat...")
    log_path, agentinfo_path = play_and_record(
        {0: ("search", (heuristic_position_evaluator, 1))},
        num_players=2, max_turns=300, seed=0, log_dir=str(tmp_path),
    )
    assert log_path is not None and agentinfo_path is not None

    # The log still loads via the unchanged GameLogger contract.
    log_data = GameLogger.load_from_file(log_path)
    assert log_data["schema_version"] == 1
    assert log_data["metadata"]["num_players"] == 2
    assert len(log_data["turns"]) > 0

    agentinfo_data = agentinfo_io.load_agentinfo(log_path)
    assert set(agentinfo_data["seats"].keys()) == {"0"}, (
        "Expected only the instrumented seat (0) in the sidecar -- seat 1 was random"
    )
    seat0_turns = {d["turn_number"] for d in agentinfo_data["seats"]["0"]["decisions"]}
    log_turns_for_seat0 = {
        t["turn_number"] for t in log_data["turns"] if t["player_id"] == 0
    }
    assert seat0_turns <= log_turns_for_seat0, (
        f"Recorded turns {seat0_turns} must be a subset of seat 0's actual turns {log_turns_for_seat0}"
    )
    print(f"✓ log + sidecar both produced; seat 0 has {len(seat0_turns)} recorded-decision turns")


def test_play_and_record_with_no_instrumented_seats_produces_no_sidecar(tmp_path):
    print("\nTesting play_and_record with only random seats produces no sidecar...")
    log_path, agentinfo_path = play_and_record(
        {}, num_players=2, max_turns=300, seed=1, log_dir=str(tmp_path),
    )
    assert agentinfo_path is None
    assert agentinfo_io.load_agentinfo(log_path) is None
    print("✓ no agent_specs -> no sidecar file, load_agentinfo returns None")


def test_play_and_record_with_heuristic_seat(tmp_path):
    print("\nTesting play_and_record with a heuristic-instrumented seat...")
    log_path, agentinfo_path = play_and_record(
        {0: ("heuristic", heuristic.DEFAULT_WEIGHTS), 1: ("search", (heuristic_position_evaluator, 1))},
        num_players=2, max_turns=300, seed=2, log_dir=str(tmp_path),
    )
    agentinfo_data = agentinfo_io.load_agentinfo(log_path)
    assert set(agentinfo_data["seats"].keys()) == {"0", "1"}
    assert agentinfo_data["seats"]["0"]["decisions"][0]["kind"] == "heuristic"
    assert agentinfo_data["seats"]["1"]["decisions"][0]["kind"] == "search"
    print("✓ both seats instrumented with their correct kinds")


def test_decision_matching_is_correct_regardless_of_who_wins_the_starting_roll(tmp_path):
    """Regression test for the seat vs. GameLogger player_id conflation
    bug: Player.player_id is assigned BEFORE Game.__init__ rotates
    self.players so the dice-determined starting player ends up at seat 0
    -- so player_id and seat coincide only by chance (roughly 1/num_players
    of the time for a fair coin). Matching a DecisionRecord's seat directly
    against turn_data['player_id'] silently found almost nothing whenever
    that coincidence didn't hold. Across many seeds (so plenty land on the
    non-coincidental case), decision_for_roll must find a real decision for
    nearly all of the instrumented seat's own rolls -- not drop to near-zero
    roughly half the time."""
    print("\nTesting decision matching is correct regardless of who wins the starting roll...")

    total_own_rolls = 0
    total_matched = 0
    for seed in range(10):
        log_path, agentinfo_path = play_and_record(
            {0: ("search", (heuristic_position_evaluator, 1))},
            num_players=2, max_turns=250, seed=seed, log_dir=str(tmp_path),
        )
        log_data = GameLogger.load_from_file(log_path)
        agentinfo_data = agentinfo_io.load_agentinfo(log_path)
        seat_by_player_id = agentinfo_io.build_seat_by_player_id(log_data['turns'])
        instrumented_player_ids = {pid for pid, seat in seat_by_player_id.items() if seat == 0}

        for turn_data in log_data['turns']:
            if turn_data['player_id'] not in instrumented_player_ids:
                continue
            for roll_idx, roll_data in enumerate(turn_data['rolls']):
                if not agentinfo_io._roll_had_decision(roll_data):
                    continue
                total_own_rolls += 1
                decision = agentinfo_io.decision_for_roll(
                    turn_data, roll_idx, agentinfo_data, seat_by_player_id,
                )
                if decision is not None:
                    total_matched += 1

    assert total_own_rolls > 0, "Test setup error: no decisions to check across 10 seeds"
    match_rate = total_matched / total_own_rolls
    assert match_rate > 0.95, (
        f"Expected nearly every one of seat 0's own decisions to be matched "
        f"({total_matched}/{total_own_rolls} = {match_rate:.2%}) -- a low rate here is exactly "
        f"the seat/player_id conflation regression this test guards against"
    )
    print(f"✓ {total_matched}/{total_own_rolls} ({match_rate:.2%}) of seat 0's decisions "
          f"matched across 10 seeds with varying starting players")


if __name__ == '__main__':
    print("All tests in this file need tmp_path -- run via pytest.")
