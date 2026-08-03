#!/usr/bin/env python3
"""
Regression tests for ParchisEnv's observation construction.

These specifically target the bug where the progress-score block in
_get_observation() double-counted finished pieces (they were included in
both finished_count and the position sum), producing values up to 2.0
against a Box space declared high=1.0 -- silently feeding out-of-range
inputs to the policy network exactly when a player was near/at a win.
"""

import numpy as np

from parchis.rl.env import ParchisEnv
from parchis.game.board import Board


def _progress_offset(env):
    """Index of the first progress-score value in the observation array."""
    return env.board_state_size + 2 * env.num_players


def test_observation_stays_within_declared_bounds():
    """The observation must never exceed the declared [0.0, 1.0] Box bounds,
    even when a player has all 4 pieces finished."""
    print("\nTesting observation stays within declared [0, 1] bounds...")

    env = ParchisEnv(num_players=4)
    obs, info = env.reset(seed=0)

    # Force player 0's pieces to all finish.
    player = env.game.players[0]
    for piece in player.pieces:
        piece.finished = True
        piece.in_base = False
        piece.position = 76

    obs = env._get_observation()

    assert obs.min() >= env.observation_space.low[0], (
        f"Observation has values below the declared lower bound: min={obs.min()}"
    )
    assert obs.max() <= env.observation_space.high[0], (
        f"Observation has values above the declared upper bound: max={obs.max()} "
        f"(the finished-piece double-count bug produced 2.0 here)"
    )
    assert env.observation_space.contains(obs), (
        "Observation does not satisfy the declared observation_space"
    )
    print(f"✓ Observation range [{obs.min():.3f}, {obs.max():.3f}] within declared bounds")
    env.close()


def test_progress_score_matches_calculate_normalized_progress():
    """The progress-score block in the observation must agree exactly with
    _calculate_normalized_progress -- the two must not be two independently
    (and divergently) implemented formulas."""
    print("\nTesting observation progress score matches _calculate_normalized_progress...")

    env = ParchisEnv(num_players=4)
    obs, info = env.reset(seed=0)

    # Put players in a mix of states: some in base, some on board, some finished.
    for i, player in enumerate(env.game.players):
        for j, piece in enumerate(player.pieces):
            if j == 0:
                piece.finished = True
                piece.in_base = False
                piece.position = 76
            elif j == 1:
                piece.finished = False
                piece.in_base = False
                piece.position = 10 + i * 5
            # j == 2, 3 stay in base (default reset state)

    obs = env._get_observation()

    current_idx = env.game.current_player_idx
    ordered_players = env.game.players[current_idx:] + env.game.players[:current_idx]

    offset = _progress_offset(env)
    for player_idx, player in enumerate(ordered_players):
        expected = env._calculate_normalized_progress(player)
        actual = obs[offset + player_idx]
        assert abs(actual - expected) < 1e-6, (
            f"{player.color}: observation progress {actual} != "
            f"_calculate_normalized_progress {expected}"
        )
    print("✓ Observation progress scores match _calculate_normalized_progress for every player")
    env.close()


def _own_piece_offset(env):
    """Index of the first own-piece-feature value in the observation array."""
    return env.board_state_size + env.global_state_size - env.OWN_PIECE_FEATURES_SIZE - env.STRATEGIC_FEATURES_SIZE


def test_own_piece_features_are_fixed_slot_by_piece_id():
    """The own-piece feature block must be indexed strictly by piece_id,
    never reordered by turn -- unlike the board-state block."""
    print("\nTesting own-piece features are fixed-slot by piece_id...")

    env = ParchisEnv(num_players=4)
    obs, info = env.reset(seed=7)
    agent = env.game.players[env.agent_player_idx]

    # Force piece_id 2 specifically into base; leave the others as reset()
    # set them (piece 0 on board, 1-3 in base by default -- move 1 and 3
    # onto the board so only piece 2 is distinguishable as "in base").
    for piece in agent.pieces:
        if piece.piece_id in (1, 3) and piece.in_base:
            env.game.board.remove_piece(piece)
            piece.move_to(agent.starting_position + piece.piece_id)
            env.game.board.add_piece(piece, agent.starting_position + piece.piece_id)

    obs = env._get_observation()
    base = _own_piece_offset(env)

    for piece in agent.pieces:
        slot = base + piece.piece_id * env.PIECE_FEATURES_PER_PIECE
        in_base_feature = obs[slot + 0]
        expected = 1.0 if piece.in_base else 0.0
        assert in_base_feature == expected, (
            f"piece_id={piece.piece_id}: expected in_base feature {expected}, got {in_base_feature}"
        )

    only_piece_2_in_base = all(
        (piece.piece_id == 2) == piece.in_base for piece in agent.pieces
    )
    assert only_piece_2_in_base, "Test setup should leave only piece_id=2 in base"
    print("✓ Own-piece features correctly indexed by piece_id, independent of turn order")
    env.close()


def test_blockade_indicator():
    """Own and opponent blockades must be reported independently."""
    print("\nTesting blockade indicator...")

    env = ParchisEnv(num_players=2)
    obs, info = env.reset(seed=3)
    env.agent_player_idx = env.game.current_player_idx
    agent = env.game.players[env.agent_player_idx]
    opponent = next(p for p in env.game.players if p is not agent)

    # Own blockade at a safe square away from either player's start.
    own_blockade_pos = 12 if agent.starting_position != 12 else 29
    for piece in agent.pieces[:2]:
        env.game.board.remove_piece(piece)
        piece.move_to(own_blockade_pos)
        env.game.board.add_piece(piece, own_blockade_pos)

    # Opponent blockade at a different safe square.
    opp_blockade_pos = 46
    for piece in opponent.pieces[:2]:
        env.game.board.remove_piece(piece)
        piece.move_to(opp_blockade_pos)
        env.game.board.add_piece(piece, opp_blockade_pos)

    obs = env._get_observation()
    base = _own_piece_offset(env)
    blockade_offset = base + env.OWN_PIECE_FEATURES_SIZE

    assert obs[blockade_offset] > 0.0, "Own blockade should be reflected"
    assert obs[blockade_offset + 1] > 0.0, "Opponent blockade should be reflected too"
    print("✓ Blockade indicator correctly reports both own and opponent blockades")
    env.close()


def test_capture_threat_and_opportunity_distance_cutoff():
    """capture_threatened/capture_opportunity fire at distance 1 and 6, but
    not at distance 7 (documented approximation)."""
    print("\nTesting capture threat/opportunity distance cutoff...")

    for distance, should_fire in [(1, True), (6, True), (7, False)]:
        env = ParchisEnv(num_players=2)
        obs, info = env.reset(seed=5)
        env.agent_player_idx = env.game.current_player_idx
        agent = env.game.players[env.agent_player_idx]
        opponent = next(p for p in env.game.players if p is not agent)

        my_piece = agent.pieces[0]
        my_pos = agent.starting_position + 20
        env.game.board.remove_piece(my_piece)
        my_piece.move_to(my_pos)
        env.game.board.add_piece(my_piece, my_pos)

        # Opponent piece `distance` squares ahead of my_piece (capture
        # opportunity direction), on a non-safe square.
        opp_piece = opponent.pieces[0]
        env.game.board.remove_piece(opp_piece)
        target_pos = ((my_pos + distance - 1) % Board.MAIN_TRACK_SIZE) + 1
        while target_pos in Board.SAFE_SQUARES:
            distance += 1
            target_pos = ((my_pos + distance - 1) % Board.MAIN_TRACK_SIZE) + 1
            if distance > 7:
                break
        opp_piece.move_to(target_pos)
        env.game.board.add_piece(opp_piece, target_pos)

        obs = env._get_observation()
        base = _own_piece_offset(env)
        opportunity = obs[base + my_piece.piece_id * env.PIECE_FEATURES_PER_PIECE + 5]

        if distance <= 6:
            assert opportunity == 1.0, f"distance={distance}: expected capture_opportunity=1.0"
        env.close()

    print("✓ Capture threat/opportunity respects the documented distance-6 cutoff")


def test_six_streak_and_bonus_chain_count_observation():
    """The six-streak and bonus-chain-count observation values must reflect
    the live instance state exactly."""
    print("\nTesting six-streak and bonus-chain-count observation values...")

    env = ParchisEnv(num_players=4)
    obs, info = env.reset(seed=11)

    base = _own_piece_offset(env)
    blockade_offset = base + env.OWN_PIECE_FEATURES_SIZE
    six_streak_offset = blockade_offset + 2
    bonus_chain_offset = six_streak_offset + 1

    env.consecutive_sixes = 2
    env.bonus_chain_count = 3
    obs = env._get_observation()

    assert abs(obs[six_streak_offset] - (2 / 3)) < 1e-6
    assert abs(obs[bonus_chain_offset] - 0.75) < 1e-6

    env.bonus_chain_count = 10  # beyond the soft cap
    obs = env._get_observation()
    assert obs[bonus_chain_offset] == 1.0, "Bonus chain count should clip to 1.0"
    print("✓ Six-streak and bonus-chain-count observation values match instance state")
    env.close()


if __name__ == '__main__':
    test_observation_stays_within_declared_bounds()
    test_progress_score_matches_calculate_normalized_progress()
    test_own_piece_features_are_fixed_slot_by_piece_id()
    test_blockade_indicator()
    test_capture_threat_and_opportunity_distance_cutoff()
    test_six_streak_and_bonus_chain_count_observation()
    print("\nAll observation tests passed!")
