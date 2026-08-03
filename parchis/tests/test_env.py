"""
Test script for ParchisEnv to verify it works correctly.
"""

import numpy as np
from parchis.rl.env import ParchisEnv
from parchis.game.board import Board


def test_basic_environment():
    """Test basic environment initialization and reset."""
    print("Testing basic environment initialization...")
    env = ParchisEnv(num_players=4)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Observation shape: {env.observation_space.shape}")

    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial action masks: {info['action_masks']}")
    print(f"Current player: {info['current_player']}")
    print(f"Dice roll: {info['dice_roll']}")
    print(f"Is bonus move: {info['is_bonus_move']}")

    print("\n✓ Basic initialization test passed!")
    return env


def test_step_execution():
    """Test executing steps in the environment."""
    print("\nTesting step execution...")
    env = ParchisEnv(num_players=4)
    obs, info = env.reset(seed=42)

    # Execute a few steps
    for step_num in range(10):
        # Get valid actions from action mask
        valid_actions = np.where(info['action_masks'] == 1)[0]

        if len(valid_actions) > 0:
            action = valid_actions[0]  # Choose first valid action
        else:
            action = 0  # Default action

        print(f"\nStep {step_num + 1}:")
        print(f"  Current player: {info['current_player']}")
        print(f"  Dice roll: {info['dice_roll']}")
        print(f"  Effective roll: {info['effective_roll']}")
        print(f"  Is bonus: {info['is_bonus_move']}")
        print(f"  Valid actions: {valid_actions}")
        print(f"  Chosen action: {action}")

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")

        if terminated or truncated:
            print(f"\n{'Game ended!' if terminated else 'Episode truncated!'}")
            break

    print("\n✓ Step execution test passed!")
    env.close()


def test_bonus_move_handling():
    """Test that bonus moves are handled correctly."""
    print("\nTesting bonus move handling...")
    env = ParchisEnv(num_players=4)

    # Run multiple episodes to hopefully trigger a bonus
    bonus_detected = False
    max_episodes = 5

    for episode in range(max_episodes):
        obs, info = env.reset(seed=episode)
        print(f"\nEpisode {episode + 1}")

        for step_num in range(100):
            valid_actions = np.where(info['action_masks'] == 1)[0]

            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)

            if info['is_bonus_move']:
                print(f"  ★ Bonus move detected at step {step_num + 1}!")
                print(f"    Bonus type: {info['bonus_type']}")
                print(f"    Bonus squares: {info['bonus_squares']}")
                print(f"    Bonus chain count: {info['bonus_chain_count']}")
                bonus_detected = True

            if terminated or truncated:
                break

        if bonus_detected:
            break

    if bonus_detected:
        print("\n✓ Bonus move handling test passed!")
    else:
        print("\n⚠ No bonus moves detected in test episodes (this is okay, just rare)")

    env.close()


def test_observation_structure():
    """
    Test that observation structure matches the current dynamic layout:
    total size = 79 * num_players + 36, split into a num_players x 76
    board-state block and a (3 * num_players + 8 + 24 + 4) global-state
    block (piece counts, progress scores, one-hot dice, bonus indicator,
    own-piece features, blockade indicator, six-streak, bonus-chain count).
    See docs/README_ENVIRONMENT.md for the full layout description.
    """
    print("\nTesting observation structure...")

    for num_players in (2, 3, 4):
        env = ParchisEnv(num_players=num_players)
        obs, info = env.reset(seed=42)

        expected_size = 79 * num_players + 36
        assert len(obs) == expected_size, (
            f"num_players={num_players}: expected observation size "
            f"{expected_size}, got {len(obs)}"
        )
        assert obs.shape == env.observation_space.shape

        board_state_size = num_players * Board.FINAL_POSITION
        global_offset = board_state_size

        # Piece counts block: 2 values per player, both in [0, 1]
        piece_counts = obs[global_offset:global_offset + 2 * num_players]
        assert np.all((piece_counts >= 0) & (piece_counts <= 1))

        # Progress scores block: 1 value per player, in [0, 1]
        progress_offset = global_offset + 2 * num_players
        progress_scores = obs[progress_offset:progress_offset + num_players]
        assert np.all((progress_scores >= 0) & (progress_scores <= 1))

        # Dice roll block: 7 values, one-hot (exactly one 1.0 once a dice
        # has been rolled, which reset() always does)
        dice_offset = progress_offset + num_players
        dice_block = obs[dice_offset:dice_offset + 7]
        assert dice_block.sum() == 1.0, f"Dice one-hot block should sum to 1, got {dice_block}"

        # Bonus indicator: 1 value, in [0, 1]
        bonus_offset = dice_offset + 7
        assert dice_offset + 7 == bonus_offset  # sanity on the offset arithmetic
        assert 0.0 <= obs[bonus_offset] <= 1.0

        # Own-piece features: 24 values (4 pieces x 6), all in [0, 1]
        own_piece_offset = bonus_offset + 1
        own_piece_block = obs[own_piece_offset:own_piece_offset + env.OWN_PIECE_FEATURES_SIZE]
        assert len(own_piece_block) == 24
        assert np.all((own_piece_block >= 0) & (own_piece_block <= 1))

        # Blockade indicator: 2 values, in [0, 1]
        blockade_offset = own_piece_offset + env.OWN_PIECE_FEATURES_SIZE
        assert 0.0 <= obs[blockade_offset] <= 1.0
        assert 0.0 <= obs[blockade_offset + 1] <= 1.0

        # Six-streak: 1 value, in [0, 1)
        six_streak_offset = blockade_offset + 2
        assert 0.0 <= obs[six_streak_offset] < 1.0

        # Bonus chain count: 1 value, in [0, 1]
        bonus_chain_offset = six_streak_offset + 1
        assert 0.0 <= obs[bonus_chain_offset] <= 1.0
        assert bonus_chain_offset == expected_size - 1, "Should be the last index"

        assert env.observation_space.contains(obs), (
            f"num_players={num_players}: observation violates the declared observation_space"
        )

        print(f"  num_players={num_players}: size={len(obs)} OK")
        env.close()

    print("\n✓ Observation structure test passed!")


def test_action_masking():
    """Test that action masking works correctly."""
    print("\nTesting action masking...")
    env = ParchisEnv(num_players=4)
    obs, info = env.reset(seed=42)

    # Test that masked actions are respected
    print(f"Action masks: {info['action_masks']}")
    print(f"Legal moves count: {info['legal_moves_count']}")

    # Count valid actions
    valid_count = np.sum(info['action_masks'])
    print(f"Valid actions: {valid_count}")

    # If there are no legal moves, all actions should be valid (to skip turn)
    if info['legal_moves_count'] == 0:
        assert valid_count == 4, "All actions should be valid when no legal moves"
    else:
        assert valid_count >= 1, "Should have at least one valid action"

    print("\n✓ Action masking test passed!")
    env.close()


def _scripted_then_random_rolls(values):
    """A dice.roll() replacement that yields `values` in order, then falls
    back to real random rolls once exhausted (so downstream auto-played
    turns, e.g. an opponent's turn following a scripted sequence, still
    behave like a normal game instead of raising StopIteration)."""
    import random
    it = iter(values)

    def roll():
        try:
            return next(it)
        except StopIteration:
            return random.randint(1, 6)
    return roll


def _prep_env_for_scripted_agent_turn(num_players=2):
    """Reset an env, force it to be the agent's own turn (so scripted dice
    control the agent's decisions, not an opponent's), and return
    (env, agent) with agent.pieces[0]/[1] still on their default squares."""
    env = ParchisEnv(num_players=num_players)
    obs, info = env.reset(seed=1)
    env.agent_player_idx = env.game.current_player_idx
    agent = env.game.players[env.agent_player_idx]
    return env, agent


def test_six_again_rerolls_same_player():
    """Rolling a 6 should grant the same player another decision (a fresh
    roll) instead of ending the turn -- docs/RULES.md 'Rolling a 6'."""
    print("\nTesting six-again reroll keeps the same player...")

    env, agent = _prep_env_for_scripted_agent_turn()
    player_idx_before = env.game.current_player_idx

    piece = agent.pieces[0]
    env.game.board.remove_piece(piece)
    piece.move_to(10)
    env.game.board.add_piece(piece, 10)

    env.current_dice_roll = 6
    env.consecutive_sixes = 1
    env.game.dice.roll = _scripted_then_random_rolls([3])  # reroll comes up 3, not 6

    action_masks = env._get_info()['action_masks']
    action = int(np.where(action_masks == 1)[0][0])
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.game.current_player_idx == player_idx_before, (
        "A six should reroll the same player, not advance the turn"
    )
    assert env.current_dice_roll == 3, "The reroll should reflect the scripted next roll"
    assert reward == 0.0, "No reward should be given while a six-streak is still open"
    print("✓ Six-again correctly rerolls the same player")
    env.close()


def test_three_sixes_penalty_captures_second_six_piece():
    """Three consecutive 6s: the piece moved on the *second* six is
    captured, the piece moved on the first six is untouched, and the third
    six is never offered as a move at all (docs/RULES.md)."""
    print("\nTesting three-sixes penalty captures the second-six piece...")

    env, agent = _prep_env_for_scripted_agent_turn()

    piece_a, piece_b = agent.pieces[0], agent.pieces[1]
    # Positions chosen 1 and 3 squares past the agent's own starting square:
    # a 6-square move from either lands far from that color's home entry
    # point (which sits just behind the starting square), so this doesn't
    # accidentally trigger home-entry protection.
    pos_a = agent.starting_position + 1
    pos_b = agent.starting_position + 3
    env.game.board.remove_piece(piece_a)
    piece_a.move_to(pos_a)
    env.game.board.add_piece(piece_a, pos_a)
    env.game.board.remove_piece(piece_b)
    piece_b.move_to(pos_b)
    env.game.board.add_piece(piece_b, pos_b)

    env.current_dice_roll = 6
    env.consecutive_sixes = 1
    env.game.dice.roll = _scripted_then_random_rolls([6, 6])

    # First six: move piece_a.
    obs, reward, terminated, truncated, info = env.step(piece_a.piece_id)
    assert env.consecutive_sixes == 2, "Should now be on the second six"

    # Second six: move piece_b (this is the piece that must be penalized).
    obs, reward, terminated, truncated, info = env.step(piece_b.piece_id)

    assert piece_b.in_base, "The piece moved on the second six should be captured"
    assert piece_b.position is None
    assert not piece_a.in_base, "The piece moved on the first six should be untouched"
    assert piece_a.position == pos_a + 6
    print("✓ Three-sixes penalty correctly captures only the second-six piece")
    env.close()


def test_three_sixes_home_entry_protection():
    """Exception 2 (docs/RULES.md): if the piece moved on the second six
    transitioned into its home column on that exact move, it is protected
    from the three-sixes penalty and stays put."""
    print("\nTesting three-sixes home-entry protection exception...")

    env, agent = _prep_env_for_scripted_agent_turn()

    piece_a, piece_b = agent.pieces[0], agent.pieces[1]
    pos_a = agent.starting_position + 1
    # 6 squares from home_entry_point - 6 lands exactly on home_entry_point
    # itself (still main track); from there compute_path already enters
    # the home column on the very next square. Instead, place piece_b so a
    # 6-square move crosses from main track into the home column.
    pos_b = agent.home_entry_point - 4
    env.game.board.remove_piece(piece_a)
    piece_a.move_to(pos_a)
    env.game.board.add_piece(piece_a, pos_a)
    env.game.board.remove_piece(piece_b)
    piece_b.move_to(pos_b)
    env.game.board.add_piece(piece_b, pos_b)

    env.current_dice_roll = 6
    env.consecutive_sixes = 1
    env.game.dice.roll = _scripted_then_random_rolls([6, 6])

    obs, reward, terminated, truncated, info = env.step(piece_a.piece_id)
    obs, reward, terminated, truncated, info = env.step(piece_b.piece_id)

    assert not piece_b.in_base, (
        "A piece that entered its home column on the second six must be "
        "protected from the three-sixes penalty"
    )
    assert piece_b.position >= Board.HOME_COLUMN_START
    print("✓ Home-entry protection correctly exempts the second-six piece")
    env.close()


def test_three_sixes_no_legal_move_on_second_six_no_capture():
    """Regression: if the second six has no legal move at all, there is no
    piece to penalize on the third six (docs/RULES.md: 'If no piece was
    moved with the second 6 ... no piece is captured')."""
    print("\nTesting three-sixes with no legal move on the second six...")

    env, agent = _prep_env_for_scripted_agent_turn()

    # All of the agent's pieces start in base except one, placed in the
    # home column at the first square (69): a 6-square move from there is
    # legal (69 -> 75, an ordinary home-column move, not a finish -- no
    # bonus chain to worry about), but the *next* 6-square move from 75
    # would overshoot position 76 and is illegal. With every other piece
    # in base (roll is 6, not 5: mandatory entry doesn't apply), this
    # guarantees zero legal moves on the roll that becomes "the second six".
    for piece in agent.pieces:
        env.game.board.remove_piece(piece)
        piece.send_to_base()
    piece_a = agent.pieces[0]
    piece_a.move_to(69)
    env.game.board.add_piece(piece_a, 69)

    env.current_dice_roll = 6
    env.consecutive_sixes = 1
    env.game.dice.roll = _scripted_then_random_rolls([6, 6])

    obs, reward, terminated, truncated, info = env.step(piece_a.piece_id)
    assert env.consecutive_sixes == 2
    assert piece_a.position == 75, "First six should move 69 -> 75 (no overshoot, no bonus)"
    assert env.pending_bonus is None

    # Second six: no legal moves at all (75 + 6 = 81 overshoots 76; every
    # other piece is in base and roll is 6, not 5).
    assert env._get_info()['legal_moves_count'] == 0
    obs, reward, terminated, truncated, info = env.step(0)

    assert piece_a.position == 75, "Piece should not have moved on the illegal second six"
    assert not piece_a.in_base, "No piece was moved on the second six, so nothing is captured"
    print("✓ No piece is penalized when the second six has no legal move")
    env.close()


def test_six_then_capture_bonus_then_six_chain():
    """A six-streak decision only resumes after any bonus chain triggered
    by that six has fully resolved (docs/RULES.md: bonus moves happen
    immediately, before the next roll)."""
    print("\nTesting bonus chain resolves before the next six-streak decision...")

    env, agent = _prep_env_for_scripted_agent_turn()
    opponent = next(p for p in env.game.players if p is not agent)

    piece = agent.pieces[0]
    capture_target = opponent.pieces[0]
    # Move the opponent's default starting piece somewhere capturable,
    # then place the agent's piece 6 squares behind it (non-safe square).
    env.game.board.remove_piece(capture_target)
    capture_pos = agent.starting_position + 10
    while capture_pos in Board.SAFE_SQUARES:
        capture_pos += 1
    capture_target.move_to(capture_pos)
    env.game.board.add_piece(capture_target, capture_pos)

    env.game.board.remove_piece(piece)
    piece.move_to(capture_pos - 6)
    env.game.board.add_piece(piece, capture_pos - 6)

    env.current_dice_roll = 6
    env.consecutive_sixes = 1
    env.game.dice.roll = _scripted_then_random_rolls([3])  # roll after the bonus chain

    # This move captures capture_target, which must trigger a 20-square
    # capture bonus -- the turn must NOT reroll yet.
    obs, reward, terminated, truncated, info = env.step(piece.piece_id)
    assert capture_target.in_base, "Capture should have happened"
    assert env.pending_bonus is not None, "Capture must trigger a pending bonus"
    assert env.pending_bonus['squares'] == 20
    assert reward == 0.0, "No reward while a bonus chain is open"
    assert env.current_dice_roll == 6, "Six-streak decision must wait for the bonus chain"

    # Resolve the bonus move (any legal piece).
    bonus_masks = env._get_info()['action_masks']
    bonus_action = int(np.where(bonus_masks == 1)[0][0])
    obs, reward, terminated, truncated, info = env.step(bonus_action)

    # Only now should the six-streak decision (reroll) actually happen.
    assert env.pending_bonus is None, "Bonus chain should be fully resolved"
    assert env.current_dice_roll == 3, (
        "The six-streak reroll should only happen after the bonus chain resolves"
    )
    print("✓ Bonus chain fully resolves before the next six-streak decision")
    env.close()


def test_captures_by_agent_surfaced_on_turn_cycle_complete():
    """
    docs/RL_DESIGN_REVIEW.md Phase 4: info['captures_by_agent'] tallies
    pieces the agent captured this turn cycle, and only appears once the
    cycle actually completes (mirrors final_progress/pieces_finished,
    which are also only added on terminated/truncated).
    """
    print("\nTesting captures_by_agent KPI signal...")

    env, agent = _prep_env_for_scripted_agent_turn()
    opponent = next(p for p in env.game.players if p is not agent)

    piece = agent.pieces[0]
    capture_target = opponent.pieces[0]

    env.game.board.remove_piece(capture_target)
    capture_pos = agent.home_entry_point - 2
    while capture_pos in Board.SAFE_SQUARES or capture_pos < 1:
        capture_pos -= 1
    capture_target.move_to(capture_pos)
    env.game.board.add_piece(capture_target, capture_pos)

    env.game.board.remove_piece(piece)
    piece.move_to(capture_pos - 4)
    env.game.board.add_piece(piece, capture_pos - 4)

    env.current_dice_roll = 4
    env.consecutive_sixes = 0
    # A plain non-6 roll: the capture bonus (20 squares from capture_pos,
    # very close to home_entry_point) overshoots Board.FINAL_POSITION and
    # has no legal moves, so the whole cycle -- capture, skipped bonus,
    # opponent auto-play -- resolves within this single step() call.
    env.game.dice.roll = _scripted_then_random_rolls([])

    obs, reward, terminated, truncated, info = env.step(piece.piece_id)

    assert capture_target.in_base, "The captured piece should be sent to base"
    assert info['captures_by_agent'] == 1
    assert info['captures_against_agent'] == 0
    print("✓ captures_by_agent correctly surfaced once the turn cycle completes")
    env.close()


def test_captures_against_agent_surfaced_on_turn_cycle_complete():
    """
    docs/RL_DESIGN_REVIEW.md Phase 4: info['captures_against_agent']
    tallies the agent's own pieces captured by an opponent during their
    auto-played turn this cycle.
    """
    print("\nTesting captures_against_agent KPI signal...")

    env, agent = _prep_env_for_scripted_agent_turn()
    opponent = next(p for p in env.game.players if p is not agent)

    victim, mover = agent.pieces[0], agent.pieces[1]

    # The piece the agent will NOT move this turn -- left on the main
    # track for the opponent to capture afterward.
    victim_pos = agent.starting_position + 10
    while victim_pos in Board.SAFE_SQUARES:
        victim_pos += 1
    env.game.board.remove_piece(victim)
    victim.move_to(victim_pos)
    env.game.board.add_piece(victim, victim_pos)

    # The piece the agent moves this turn (an ordinary, non-capturing move
    # so the agent's turn ends immediately on this non-6 roll).
    env.game.board.remove_piece(mover)
    mover.move_to(agent.starting_position)
    env.game.board.add_piece(mover, agent.starting_position)

    # The opponent's only on-board piece, placed so their forced move
    # (their only legal one, since their other 3 pieces stay in base)
    # lands exactly on victim_pos, capturing it.
    opp_roll = 4
    opp_piece = opponent.pieces[0]
    opp_start = victim_pos - opp_roll
    if opp_start < 1:
        opp_start += Board.MAIN_TRACK_SIZE
    env.game.board.remove_piece(opp_piece)
    opp_piece.move_to(opp_start)
    env.game.board.add_piece(opp_piece, opp_start)

    env.current_dice_roll = 3
    env.consecutive_sixes = 0
    # Consumed by _start_new_turn_for_next_player when control passes to
    # the opponent right after this step (docs/RL_DESIGN_REVIEW.md Phase
    # 1's turn-cycle helpers) -- the opponent's only legal move captures
    # victim, forcing the capture regardless of the (default random)
    # opponent policy.
    env.game.dice.roll = _scripted_then_random_rolls([opp_roll])

    obs, reward, terminated, truncated, info = env.step(mover.piece_id)

    assert victim.in_base, "The agent's piece should have been captured by the opponent"
    assert info['captures_against_agent'] == 1
    assert info['captures_by_agent'] == 0
    print("✓ captures_against_agent correctly surfaced once the turn cycle completes")
    env.close()


def test_three_sixes_penalty_surfaced_in_info():
    """
    docs/RL_DESIGN_REVIEW.md Phase 4: info['three_sixes_penalty'] mirrors
    Game.apply_three_sixes_penalty's return value (previously discarded
    at the ParchisEnv call site), surfaced once the turn cycle completes.
    """
    print("\nTesting three_sixes_penalty KPI signal...")

    env, agent = _prep_env_for_scripted_agent_turn()

    piece_a, piece_b = agent.pieces[0], agent.pieces[1]
    pos_a = agent.starting_position + 1
    pos_b = agent.starting_position + 3
    env.game.board.remove_piece(piece_a)
    piece_a.move_to(pos_a)
    env.game.board.add_piece(piece_a, pos_a)
    env.game.board.remove_piece(piece_b)
    piece_b.move_to(pos_b)
    env.game.board.add_piece(piece_b, pos_b)

    env.current_dice_roll = 6
    env.consecutive_sixes = 1
    env.game.dice.roll = _scripted_then_random_rolls([6, 6])

    obs, reward, terminated, truncated, info = env.step(piece_a.piece_id)
    assert 'three_sixes_penalty' not in info, "Cycle still open after the first six"

    obs, reward, terminated, truncated, info = env.step(piece_b.piece_id)

    assert piece_b.in_base, "Sanity check: the second-six piece should be captured"
    assert info['three_sixes_penalty'] is True
    print("✓ three_sixes_penalty correctly surfaced once the penalty fires")
    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("ParchisEnv Test Suite")
    print("=" * 60)

    try:
        test_basic_environment()
        test_observation_structure()
        test_action_masking()
        test_step_execution()
        test_bonus_move_handling()
        test_six_again_rerolls_same_player()
        test_three_sixes_penalty_captures_second_six_piece()
        test_three_sixes_home_entry_protection()
        test_three_sixes_no_legal_move_on_second_six_no_capture()
        test_six_then_capture_bonus_then_six_chain()
        test_captures_by_agent_surfaced_on_turn_cycle_complete()
        test_captures_against_agent_surfaced_on_turn_cycle_complete()
        test_three_sixes_penalty_surfaced_in_info()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
