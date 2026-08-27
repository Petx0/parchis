#!/usr/bin/env python3
"""
Regression tests for ParchisEnv's observation construction.

These specifically target the bug where the progress-score block in
_get_observation() double-counted finished pieces (they were included in
both finished_count and the position sum), producing values up to 2.0
against a Box space declared high=1.0 -- silently feeding out-of-range
inputs to the policy network exactly when a player was near/at a win.

Also covers the Decision 1-5 observation-space changes
(docs/observation_space_changes.md): the two-flag bonus indicator, the
roll-based capture_threat_score/capture_opportunity scores (replacing the
old hand-rolled distance-6 binary checks), and the removal of the
blockade indicator and bonus-chain-count blocks.
"""

import numpy as np

from parchis.rl.env import ParchisEnv
from parchis.game.board import Board


def _advance(pos, steps):
    """Modular main-track advance, matching RuleEngine.compute_path's
    behavior for paths that don't cross a home-entry point."""
    return ((pos - 1 + steps) % Board.MAIN_TRACK_SIZE) + 1


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
    """Index of the first own-piece-feature value in the observation
    array. The own-piece block is no longer a clean 4xN rectangle: per
    piece_id it's a 5-wide stride (indices 0-3 relative to _own_piece_offset
    + piece_id * PIECE_FEATURES_PER_PIECE), plus one extra shared
    capture_opportunity slot at _own_piece_offset(env) + 20, not indexed by
    piece_id at all."""
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


def test_get_observation_perspective_seat_overrides_default():
    """_get_observation(perspective_seat=...) must build the own-piece
    block (and, transitively, capture_opportunity) from the REQUESTED
    seat's pieces, not always from self.agent_player_idx.

    Regression test for docs/AGENT_REBUILD_PLAN.md §1.3: before this
    parameter existed, ParchisSelfPlayEnv._choose_opponent_move called
    _get_observation() with no way to ask for anyone but the learning
    agent's perspective, so every opponent model was fed a hybrid
    observation (an opponent-relative board, but the learning agent's own
    pieces). Scripts two seats into provably different piece layouts (one
    fully in base, the other fully on board at known positions) and checks
    each perspective's own-piece block reflects ONLY that seat."""
    print("\nTesting _get_observation(perspective_seat=...) overrides the default agent perspective...")

    env = ParchisEnv(num_players=3)
    obs, info = env.reset(seed=42)
    env.agent_player_idx = 0
    agent = env.game.players[0]
    other_seat = 1
    other = env.game.players[other_seat]

    # Agent: all 4 pieces in base.
    for piece in agent.pieces:
        if not piece.in_base:
            env.game.board.remove_piece(piece)
            piece.send_to_base()

    # Other seat: all 4 pieces on board at distinct, known positions.
    other_positions = [10, 20, 30, 40]
    for piece, pos in zip(other.pieces, other_positions):
        env.game.board.remove_piece(piece)
        piece.move_to(pos)
        env.game.board.add_piece(piece, pos)

    base = _own_piece_offset(env)

    obs_default = env._get_observation()
    obs_agent = env._get_observation(perspective_seat=0)
    obs_other = env._get_observation(perspective_seat=other_seat)

    assert np.array_equal(obs_default, obs_agent), (
        "perspective_seat=None must behave exactly like "
        "perspective_seat=self.agent_player_idx"
    )

    # Agent perspective: every own-piece slot shows in_base=1.0.
    for piece in agent.pieces:
        slot = base + piece.piece_id * env.PIECE_FEATURES_PER_PIECE
        assert obs_agent[slot + 0] == 1.0, (
            "Agent-perspective observation should show the agent's own pieces in base"
        )

    # Other-seat perspective: every own-piece slot shows in_base=0.0 and the
    # scripted position -- i.e. it reflects `other`, never `agent`.
    for piece, pos in zip(other.pieces, other_positions):
        slot = base + piece.piece_id * env.PIECE_FEATURES_PER_PIECE
        assert obs_other[slot + 0] == 0.0, (
            "perspective_seat=other_seat must show THAT seat's pieces, not the agent's"
        )
        assert abs(obs_other[slot + 2] - pos / Board.FINAL_POSITION) < 1e-6, (
            f"perspective_seat=other_seat own-piece position feature should reflect pos={pos}"
        )

    own_piece_block = slice(base, base + env.OWN_PIECE_FEATURES_SIZE)
    assert not np.array_equal(obs_agent[own_piece_block], obs_other[own_piece_block]), (
        "Own-piece feature block must differ between two provably-different perspectives"
    )
    print("✓ _get_observation(perspective_seat=...) correctly isolates each seat's own pieces")
    env.close()


def test_bonus_indicator_flags():
    """has_finish_bonus/has_capture_bonus are mutually-exclusive binary
    flags reflecting self.pending_bonus (Decision 1: replaces the old
    single continuum-encoded bonus_squares/20.0 scalar)."""
    print("\nTesting bonus indicator flags...")

    env = ParchisEnv(num_players=2)
    obs, info = env.reset(seed=3)

    dice_offset = _progress_offset(env) + env.num_players
    bonus_offset = dice_offset + 7

    # No pending bonus: both flags 0.0.
    env.pending_bonus = None
    obs = env._get_observation()
    assert obs[bonus_offset] == 0.0
    assert obs[bonus_offset + 1] == 0.0

    # Finish bonus pending.
    env.pending_bonus = {'type': 'finish_bonus', 'squares': 10}
    obs = env._get_observation()
    assert obs[bonus_offset] == 1.0, "has_finish_bonus should be set"
    assert obs[bonus_offset + 1] == 0.0, "has_capture_bonus should stay 0"

    # Capture bonus pending.
    env.pending_bonus = {'type': 'capture_bonus', 'squares': 20}
    obs = env._get_observation()
    assert obs[bonus_offset] == 0.0, "has_finish_bonus should stay 0"
    assert obs[bonus_offset + 1] == 1.0, "has_capture_bonus should be set"

    print("✓ Bonus indicator flags correctly reflect pending_bonus['type'], mutually exclusive")
    env.close()


def test_capture_threat_score_direct_hit():
    """capture_threat_score reflects a direct roll-based capture at
    several distances (excluding the mandatory-5-entry roll, which is
    tested separately)."""
    print("\nTesting capture_threat_score: direct hits...")

    for distance in (1, 3, 6):
        env = ParchisEnv(num_players=2)
        obs, info = env.reset(seed=5)
        env.agent_player_idx = env.game.current_player_idx
        agent = env.game.players[env.agent_player_idx]
        opponent = next(p for p in env.game.players if p is not agent)

        my_piece = agent.pieces[0]
        my_pos = 15
        env.game.board.remove_piece(my_piece)
        my_piece.move_to(my_pos)
        env.game.board.add_piece(my_piece, my_pos)

        # Move the opponent's default on-board piece well out of the way
        # so it can't also (coincidentally) threaten my_piece.
        opp_start_piece = opponent.pieces[0]
        env.game.board.remove_piece(opp_start_piece)
        opp_start_piece.move_to(60)
        env.game.board.add_piece(opp_start_piece, 60)

        opp_piece = opponent.pieces[1]
        opp_pos = my_pos - distance
        env.game.board.remove_piece(opp_piece)
        opp_piece.move_to(opp_pos)
        env.game.board.add_piece(opp_piece, opp_pos)

        obs = env._get_observation()
        base = _own_piece_offset(env)
        threat = obs[base + my_piece.piece_id * env.PIECE_FEATURES_PER_PIECE + 4]

        assert abs(threat - 1 / 6) < 1e-6, (
            f"distance={distance}: expected capture_threat_score=1/6 "
            f"(one opponent, one hit face), got {threat}"
        )
        env.close()

    print("✓ capture_threat_score correctly fires on direct hits at several distances")


def test_capture_threat_score_mandatory_five_entry():
    """capture_threat_score respects the mandatory-5-entry rule: a
    distance-5 threat only counts if the opponent's own entry is
    currently blocked (2 of their own pieces stacked on their own
    starting square). If entry is legal it's mandatory, so the
    distance-5 piece cannot move instead and must not count."""
    print("\nTesting capture_threat_score: mandatory-5-entry exception...")

    my_pos = 30

    # --- Entry blocked: distance-5 threat is real ---
    env = ParchisEnv(num_players=2)
    obs, info = env.reset(seed=9)
    env.agent_player_idx = env.game.current_player_idx
    agent = env.game.players[env.agent_player_idx]
    opponent = next(p for p in env.game.players if p is not agent)

    my_piece = agent.pieces[0]
    env.game.board.remove_piece(my_piece)
    my_piece.move_to(my_pos)
    env.game.board.add_piece(my_piece, my_pos)

    # Stack 2 own pieces at the opponent's starting square -- entry blocked.
    opp_second_piece = opponent.pieces[1]
    env.game.board.remove_piece(opp_second_piece)
    opp_second_piece.move_to(opponent.starting_position)
    env.game.board.add_piece(opp_second_piece, opponent.starting_position)

    opp_threat_piece = opponent.pieces[2]
    env.game.board.remove_piece(opp_threat_piece)
    opp_threat_piece.move_to(my_pos - 5)
    env.game.board.add_piece(opp_threat_piece, my_pos - 5)

    legal_moves_5 = env.game.get_legal_moves(opponent, 5)
    assert (opp_threat_piece, my_pos, 'move') in legal_moves_5, (
        "Test setup error: entry should be blocked, so opp_threat_piece "
        "should have a legal roll-of-5 move onto my_pos"
    )

    obs = env._get_observation()
    base = _own_piece_offset(env)
    threat = obs[base + my_piece.piece_id * env.PIECE_FEATURES_PER_PIECE + 4]
    assert threat > 0.0, (
        "distance-5 threat should count when the opponent's own entry is blocked"
    )
    env.close()

    # --- Entry legal (mandatory): the same distance-5 piece must NOT
    # count, since a roll of 5 forces entry instead ---
    env = ParchisEnv(num_players=2)
    obs, info = env.reset(seed=9)
    env.agent_player_idx = env.game.current_player_idx
    agent = env.game.players[env.agent_player_idx]
    opponent = next(p for p in env.game.players if p is not agent)

    my_piece = agent.pieces[0]
    env.game.board.remove_piece(my_piece)
    my_piece.move_to(my_pos)
    env.game.board.add_piece(my_piece, my_pos)

    # Opponent's starting square left in its default single-piece state --
    # entry is legal and, per the mandatory-entry rule, exclusive.
    opp_threat_piece = opponent.pieces[2]
    env.game.board.remove_piece(opp_threat_piece)
    opp_threat_piece.move_to(my_pos - 5)
    env.game.board.add_piece(opp_threat_piece, my_pos - 5)

    legal_moves_5 = env.game.get_legal_moves(opponent, 5)
    assert legal_moves_5 and all(m[2] == 'enter' for m in legal_moves_5), (
        "Test setup error: entry should be mandatory and exclusive when "
        "the opponent's own starting square doesn't have 2 own pieces"
    )

    obs = env._get_observation()
    base = _own_piece_offset(env)
    threat = obs[base + my_piece.piece_id * env.PIECE_FEATURES_PER_PIECE + 4]
    assert threat == 0.0, (
        "distance-5 threat must NOT count when entry is mandatory "
        "(the threatening piece cannot move this turn)"
    )
    env.close()

    print("✓ capture_threat_score correctly respects the mandatory-5-entry exception")


def test_capture_threat_score_bonus_chain():
    """capture_threat_score includes a hit delivered via another
    opponent piece's bonus-chain move (the roll captures a decoy piece
    elsewhere; the resulting 20-square bonus, chosen freely among the
    opponent's on-board pieces, lands on the target piece)."""
    print("\nTesting capture_threat_score: bonus-chain capture...")

    env = ParchisEnv(num_players=2)
    obs, info = env.reset(seed=13)
    env.agent_player_idx = env.game.current_player_idx
    agent = env.game.players[env.agent_player_idx]
    opponent = next(p for p in env.game.players if p is not agent)

    s = opponent.starting_position
    p_pos = _advance(s, 30)
    d_pos = _advance(p_pos, 3)
    z_pos = _advance(s, 5)
    x_pos = _advance(z_pos, 20)

    x_piece = agent.pieces[0]   # target: threatened only via the bonus chain
    d_piece = agent.pieces[1]   # decoy: directly captured on v=3
    p_piece = opponent.pieces[1]
    z_piece = opponent.pieces[2]

    for piece, pos in ((x_piece, x_pos), (d_piece, d_pos), (p_piece, p_pos), (z_piece, z_pos)):
        env.game.board.remove_piece(piece)
        piece.move_to(pos)
        env.game.board.add_piece(piece, pos)

    # Isolate the bonus-chain path: rolling 3 must not directly land on x_pos.
    direct_moves_3 = env.game.get_legal_moves(opponent, 3)
    assert x_pos not in [m[1] for m in direct_moves_3], (
        "Test setup error: roll=3 should not directly threaten x_piece"
    )
    assert (p_piece, d_pos, 'move') in direct_moves_3, (
        "Test setup error: p_piece should capture d_piece on roll=3"
    )

    obs = env._get_observation()
    base = _own_piece_offset(env)
    threat = obs[base + x_piece.piece_id * env.PIECE_FEATURES_PER_PIECE + 4]
    assert abs(threat - 1 / 6) < 1e-6, (
        f"expected capture_threat_score=1/6 (bonus-chain hit via z_piece "
        f"on v=3), got {threat}"
    )
    env.close()
    print("✓ capture_threat_score correctly includes a bonus-chain capture threat")


def test_capture_threat_score_same_piece_bonus_chain_not_modeled():
    """Documents an accepted, deliberate limitation: capture_threat_score
    does NOT simulate the triggering piece continuing its own bonus chain
    from its post-capture position -- only other, already-on-board
    opponent pieces are checked (see _capture_threat_scores' docstring).
    This is a regression anchor against silently "fixing" this into an
    inconsistent state later."""
    print("\nTesting capture_threat_score: same-piece chain is not simulated...")

    env = ParchisEnv(num_players=2)
    obs, info = env.reset(seed=13)
    env.agent_player_idx = env.game.current_player_idx
    agent = env.game.players[env.agent_player_idx]
    opponent = next(p for p in env.game.players if p is not agent)

    s = opponent.starting_position
    p_pos = _advance(s, 30)
    d_pos = _advance(p_pos, 3)
    x_pos = _advance(d_pos, 20)  # where p_piece WOULD land if it continued its own chain

    x_piece = agent.pieces[0]
    d_piece = agent.pieces[1]
    p_piece = opponent.pieces[1]

    for piece, pos in ((x_piece, x_pos), (d_piece, d_pos), (p_piece, p_pos)):
        env.game.board.remove_piece(piece)
        piece.move_to(pos)
        env.game.board.add_piece(piece, pos)

    direct_moves_3 = env.game.get_legal_moves(opponent, 3)
    assert (p_piece, d_pos, 'move') in direct_moves_3, (
        "Test setup error: p_piece should capture d_piece on roll=3"
    )

    obs = env._get_observation()
    base = _own_piece_offset(env)
    threat = obs[base + x_piece.piece_id * env.PIECE_FEATURES_PER_PIECE + 4]
    assert threat == 0.0, (
        "capture_threat_score should NOT flag x_piece: the only path to it "
        "is p_piece continuing its own chain from its post-capture "
        "position, which is a deliberately unmodeled case"
    )
    env.close()
    print("✓ Same-piece bonus-chain continuation is correctly left unmodeled")


def test_capture_threat_score_double_threat_not_deduplicated():
    """Two different opponents each threatening the same own piece with
    the same face value must both count -- "double threat = double
    risk", not deduplicated to a single hit."""
    print("\nTesting capture_threat_score: double threat from two opponents...")

    env = ParchisEnv(num_players=3)
    obs, info = env.reset(seed=17)
    env.agent_player_idx = env.game.current_player_idx
    agent = env.game.players[env.agent_player_idx]
    opponents = [p for p in env.game.players if p is not agent]
    assert len(opponents) == 2

    x_piece = agent.pieces[0]
    x_pos = 15  # far enough from all 3 default starting squares (5, 22, 39)
    env.game.board.remove_piece(x_piece)
    x_piece.move_to(x_pos)
    env.game.board.add_piece(x_piece, x_pos)

    threat_pos = 13  # 2 squares behind x_pos
    for opp in opponents:
        opp_piece = opp.pieces[1]  # leave pieces[0] at their own starting square
        env.game.board.remove_piece(opp_piece)
        opp_piece.move_to(threat_pos)
        env.game.board.add_piece(opp_piece, threat_pos)

    obs = env._get_observation()
    base = _own_piece_offset(env)
    threat = obs[base + x_piece.piece_id * env.PIECE_FEATURES_PER_PIECE + 4]
    assert abs(threat - 2 / 6) < 1e-6, (
        f"expected capture_threat_score=2/6 (two opponents, one hit face "
        f"each, not deduplicated), got {threat}"
    )
    env.close()
    print("✓ capture_threat_score correctly sums (not deduplicates) threats across opponents")


def test_capture_opportunity_score_basic():
    """capture_opportunity is a single shared value: exactly one face
    value producing a legal capturing move should give a 1/6 score."""
    print("\nTesting capture_opportunity_score: basic single-roll capture...")

    env = ParchisEnv(num_players=2)
    obs, info = env.reset(seed=21)
    env.agent_player_idx = env.game.current_player_idx
    agent = env.game.players[env.agent_player_idx]
    opponent = next(p for p in env.game.players if p is not agent)

    my_piece = agent.pieces[0]
    my_pos = 20
    env.game.board.remove_piece(my_piece)
    my_piece.move_to(my_pos)
    env.game.board.add_piece(my_piece, my_pos)

    v = 4
    target_pos = my_pos + v  # 24, non-safe
    opp_piece = opponent.pieces[0]
    env.game.board.remove_piece(opp_piece)
    opp_piece.move_to(target_pos)
    env.game.board.add_piece(opp_piece, target_pos)

    obs = env._get_observation()
    base = _own_piece_offset(env)
    opportunity = obs[base + 20]  # shared slot: own_piece_offset + 20
    assert abs(opportunity - 1 / 6) < 1e-6, (
        f"expected capture_opportunity=1/6 (single roll value captures), got {opportunity}"
    )
    env.close()
    print("✓ capture_opportunity_score correctly reflects a single-roll capture")


def test_capture_opportunity_score_or_across_pieces_not_summed():
    """A single roll value that lets TWO different own pieces each
    capture something must still count once toward capture_opportunity,
    not twice -- contrasts with capture_threat_score's per-opponent
    summing behavior."""
    print("\nTesting capture_opportunity_score: OR across pieces, not summed...")

    env = ParchisEnv(num_players=3)
    obs, info = env.reset(seed=23)
    env.agent_player_idx = env.game.current_player_idx
    agent = env.game.players[env.agent_player_idx]
    opponents = [p for p in env.game.players if p is not agent]
    assert len(opponents) == 2

    v = 3
    piece_a = agent.pieces[0]
    piece_a_pos = 20
    piece_b = agent.pieces[1]
    piece_b_pos = 40

    env.game.board.remove_piece(piece_a)
    piece_a.move_to(piece_a_pos)
    env.game.board.add_piece(piece_a, piece_a_pos)

    env.game.board.remove_piece(piece_b)
    piece_b.move_to(piece_b_pos)
    env.game.board.add_piece(piece_b, piece_b_pos)

    target_a = piece_a_pos + v  # 23, non-safe
    target_b = piece_b_pos + v  # 43, non-safe
    opp0_piece = opponents[0].pieces[0]
    opp1_piece = opponents[1].pieces[0]

    env.game.board.remove_piece(opp0_piece)
    opp0_piece.move_to(target_a)
    env.game.board.add_piece(opp0_piece, target_a)

    env.game.board.remove_piece(opp1_piece)
    opp1_piece.move_to(target_b)
    env.game.board.add_piece(opp1_piece, target_b)

    obs = env._get_observation()
    base = _own_piece_offset(env)
    opportunity = obs[base + 20]
    assert abs(opportunity - 1 / 6) < 1e-6, (
        f"two own pieces capturing with the same roll value should count "
        f"once, expected capture_opportunity=1/6, got {opportunity}"
    )
    env.close()
    print("✓ capture_opportunity_score correctly ORs across pieces instead of summing")


def test_capture_opportunity_score_no_bonus_chain():
    """capture_opportunity_score is explicitly single-roll only (1-6): a
    piece only reachable via a 20-square bonus roll must not affect the
    score, unlike capture_threat_score's opponent-side chain check."""
    print("\nTesting capture_opportunity_score: no bonus-chain extension...")

    env = ParchisEnv(num_players=2)
    obs, info = env.reset(seed=29)
    env.agent_player_idx = env.game.current_player_idx
    agent = env.game.players[env.agent_player_idx]
    opponent = next(p for p in env.game.players if p is not agent)

    my_piece = agent.pieces[0]
    my_pos = 10
    env.game.board.remove_piece(my_piece)
    my_piece.move_to(my_pos)
    env.game.board.add_piece(my_piece, my_pos)

    v = 3
    direct_target_pos = my_pos + v          # 13: captured directly on roll=3
    chain_only_target_pos = my_pos + 20     # 30: only reachable via a 20-square
                                             # bonus roll, which
                                             # capture_opportunity_score never
                                             # evaluates (only checks 1-6)
    opp_piece_1 = opponent.pieces[0]
    opp_piece_2 = opponent.pieces[1]

    env.game.board.remove_piece(opp_piece_1)
    opp_piece_1.move_to(direct_target_pos)
    env.game.board.add_piece(opp_piece_1, direct_target_pos)

    env.game.board.remove_piece(opp_piece_2)
    opp_piece_2.move_to(chain_only_target_pos)
    env.game.board.add_piece(opp_piece_2, chain_only_target_pos)

    obs = env._get_observation()
    base = _own_piece_offset(env)
    opportunity = obs[base + 20]
    assert abs(opportunity - 1 / 6) < 1e-6, (
        f"expected capture_opportunity=1/6 (single-roll only, no chain "
        f"extension to opp_piece_2), got {opportunity}"
    )
    env.close()
    print("✓ capture_opportunity_score correctly excludes bonus-chain extension")


def test_six_streak_observation():
    """The six-streak observation value must reflect the live instance
    state exactly. It's now the final block in the observation, since the
    blockade indicator and bonus-chain-count were both cut."""
    print("\nTesting six-streak observation value...")

    env = ParchisEnv(num_players=4)
    obs, info = env.reset(seed=11)

    base = _own_piece_offset(env)
    six_streak_offset = base + env.OWN_PIECE_FEATURES_SIZE

    env.consecutive_sixes = 2
    obs = env._get_observation()

    assert abs(obs[six_streak_offset] - (2 / 3)) < 1e-6
    assert six_streak_offset == len(obs) - 1, "Six-streak should be the last observation index"
    print("✓ Six-streak observation value matches instance state")
    env.close()


def test_bonus_chain_count_still_in_info_not_observation():
    """bonus_chain_count was cut from the observation array (Decision 4)
    but must still be tracked on the instance and exposed via _get_info()
    for KPI logging (parchis/training/common.py,
    parchis/evaluation/evaluate.py both read info['bonus_chain_count'])."""
    print("\nTesting bonus_chain_count is still in info, not in the observation...")

    env = ParchisEnv(num_players=4)
    obs, info = env.reset(seed=11)

    obs_before = env._get_observation()
    env.bonus_chain_count = 3
    obs_after = env._get_observation()

    assert obs_before.shape == obs_after.shape
    assert np.array_equal(obs_before, obs_after), (
        "bonus_chain_count must have no effect on the observation array"
    )

    info = env._get_info()
    assert info['bonus_chain_count'] == 3, (
        "bonus_chain_count must still be exposed via _get_info() for KPI logging"
    )
    print("✓ bonus_chain_count remains live in info, absent from the observation")
    env.close()


if __name__ == '__main__':
    test_observation_stays_within_declared_bounds()
    test_progress_score_matches_calculate_normalized_progress()
    test_own_piece_features_are_fixed_slot_by_piece_id()
    test_get_observation_perspective_seat_overrides_default()
    test_bonus_indicator_flags()
    test_capture_threat_score_direct_hit()
    test_capture_threat_score_mandatory_five_entry()
    test_capture_threat_score_bonus_chain()
    test_capture_threat_score_same_piece_bonus_chain_not_modeled()
    test_capture_threat_score_double_threat_not_deduplicated()
    test_capture_opportunity_score_basic()
    test_capture_opportunity_score_or_across_pieces_not_summed()
    test_capture_opportunity_score_no_bonus_chain()
    test_six_streak_observation()
    test_bonus_chain_count_still_in_info_not_observation()
    print("\nAll observation tests passed!")
