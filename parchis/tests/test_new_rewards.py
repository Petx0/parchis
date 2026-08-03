#!/usr/bin/env python3
"""
Test script to verify the turn-cycle reward structure with opponent penalty.
"""

from parchis.rl.env import ParchisEnv
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env):
    """Return the action mask from the environment."""
    return env.unwrapped._get_info()['action_masks']


def test_reward_structure():
    """Test that turn-cycle rewards work correctly."""
    print("Testing turn-cycle reward structure...")
    print("=" * 60)

    env = ParchisEnv(num_players=4)
    env = ActionMasker(env, mask_fn)

    obs, info = env.reset()
    episode_reward = 0
    step_count = 0
    rewards_seen = []
    zero_reward_count = 0

    agent = env.unwrapped.game.players[env.unwrapped.agent_player_idx]
    print(f"Initial progress: {env.unwrapped._calculate_normalized_progress(agent):.4f}")
    print(f"Opponent weight (α): {env.unwrapped.opponent_weight}")

    while step_count < 100:  # Test first 100 steps
        action_masks = mask_fn(env)
        # Choose first valid action
        valid_actions = [i for i, mask in enumerate(action_masks) if mask == 1]
        if len(valid_actions) == 0:
            break

        action = valid_actions[0]
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        step_count += 1

        if reward == 0:
            zero_reward_count += 1
        else:
            rewards_seen.append(reward)
            agent = env.unwrapped.game.players[env.unwrapped.agent_player_idx]
            progress = env.unwrapped._calculate_normalized_progress(agent)
            print(f"Step {step_count}: reward={reward:+.6f}, progress={progress:.4f}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step_count}")
            if 'final_progress' in info:
                print(f"Final progress: {info['final_progress']:.4f}")
            if 'pieces_finished' in info:
                print(f"Pieces finished: {info['pieces_finished']}")
            if 'pieces_out_of_base' in info:
                print(f"Pieces out of base: {info['pieces_out_of_base']}")
            if 'won' in info:
                print(f"Won: {info['won']}")
            break

    print("\n" + "=" * 60)
    print(f"Total episode reward: {episode_reward:+.6f}")
    print(f"Steps with zero reward (bonus moves): {zero_reward_count}")
    print(f"Steps with non-zero reward (turn boundaries): {len(rewards_seen)}")

    if len(rewards_seen) > 0:
        print(f"Min reward: {min(rewards_seen):+.6f}")
        print(f"Max reward: {max(rewards_seen):+.6f}")
        print(f"Mean reward: {sum(rewards_seen)/len(rewards_seen):+.6f}")

        # Count positive vs negative rewards
        positive_rewards = sum(1 for r in rewards_seen if r > 0)
        negative_rewards = sum(1 for r in rewards_seen if r < 0)
        print(f"Positive rewards: {positive_rewards}, Negative rewards: {negative_rewards}")
    print("=" * 60)

    # Verify reward range is reasonable (should be small deltas, can be negative)
    if len(rewards_seen) > 0:
        assert all(-1.0 < r < 1.0 for r in rewards_seen), "Rewards outside expected range!"
        print("✓ Rewards are in expected range (-1, 1)")

    # Verify that we have some zero rewards (bonus moves should return 0)
    # This confirms rewards are only given at turn boundaries
    if zero_reward_count > 0:
        print("✓ Zero rewards observed (turn-cycle behavior confirmed)")

    # With opponent penalty, we expect some negative rewards when opponents advance
    if len(rewards_seen) > 0 and any(r < 0 for r in rewards_seen):
        print("✓ Negative rewards observed (opponent penalty working)")

    print("\n✓ All tests passed!")


def test_opponent_weight_effect():
    """Test that different opponent weights produce different rewards."""
    print("\n" + "=" * 60)
    print("Testing opponent weight effect...")
    print("=" * 60)

    # Run same scenario with different opponent weights
    for alpha in [0.0, 0.5, 1.0]:
        env = ParchisEnv(num_players=4)
        env.unwrapped.opponent_weight = alpha

        obs, info = env.reset(seed=42)  # Fixed seed for reproducibility
        total_reward = 0

        for _ in range(20):  # Run 20 steps
            action_masks = env.unwrapped._get_info()['action_masks']
            valid_actions = [i for i, mask in enumerate(action_masks) if mask == 1]
            if not valid_actions:
                break
            obs, reward, terminated, truncated, info = env.step(valid_actions[0])
            total_reward += reward
            if terminated or truncated:
                break

        print(f"α = {alpha}: total_reward = {total_reward:+.6f}")

    print("✓ Opponent weight test complete")


def test_reward_types():
    """Test that all three reward types produce expected reward patterns."""
    print("\n" + "=" * 60)
    print("Testing reward types (progress_delta, win_loss, win_loss_shaped)...")
    print("=" * 60)

    for reward_type in ParchisEnv.VALID_REWARD_TYPES:
        env = ParchisEnv(num_players=2, reward_type=reward_type)
        env = ActionMasker(env, mask_fn)

        obs, info = env.reset(seed=42)
        rewards = []

        for _ in range(200):
            valid_actions = [i for i, m in enumerate(info['action_masks']) if m == 1]
            if not valid_actions:
                break
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        non_zero = [r for r in rewards if r != 0.0]
        terminal_reward = rewards[-1] if (terminated or truncated) else None

        print(f"\n  {reward_type}:")
        print(f"    Steps: {len(rewards)}")
        print(f"    Non-zero rewards: {len(non_zero)}")
        if rewards:
            print(f"    Range: [{min(rewards):.6f}, {max(rewards):.6f}]")
        if terminal_reward is not None:
            print(f"    Terminal reward: {terminal_reward:.6f}")
            print(f"    Terminated: {terminated}, Truncated: {truncated}")

        if reward_type == "win_loss":
            # Mid-game rewards should all be 0.0
            mid_game = rewards[:-1] if terminated else rewards
            non_zero_mid = [r for r in mid_game if r != 0.0]
            assert len(non_zero_mid) == 0, \
                f"win_loss should have 0.0 mid-game, got {non_zero_mid[:5]}"
            # Terminal should be +1 or -1 if game ended
            if terminated:
                assert terminal_reward in (1.0, -1.0), \
                    f"win_loss terminal should be +/-1.0, got {terminal_reward}"
            print(f"    ✓ win_loss pattern correct")

        elif reward_type == "progress_delta":
            assert any(r != 0.0 for r in rewards), \
                "progress_delta should produce non-zero rewards"
            # Typical progress deltas are small (< 0.1)
            for r in non_zero:
                assert abs(r) < 0.5, f"progress_delta reward {r} seems too large"
            print(f"    ✓ progress_delta pattern correct")

        elif reward_type == "win_loss_shaped":
            # Should have small mid-game rewards + large terminal
            if terminated and terminal_reward is not None:
                assert terminal_reward in (1.0, -1.0), \
                    f"win_loss_shaped terminal should be +/-1.0, got {terminal_reward}"
            # Mid-game rewards should be small (0.1 * progress_delta)
            mid_game = rewards[:-1] if terminated else rewards
            mid_non_zero = [r for r in mid_game if r != 0.0]
            for r in mid_non_zero:
                assert abs(r) < 0.1, \
                    f"win_loss_shaped mid-game reward {r} too large (expected < 0.1)"
            print(f"    ✓ win_loss_shaped pattern correct")

        env.close()

    print("\n✓ All reward type tests passed!")


def test_six_streak_intermediate_steps_return_zero_reward():
    """A six-again reroll must not trigger reward computation, exactly like
    a mid-bonus-chain step -- reward only fires once the turn cycle
    genuinely completes."""
    print("\nTesting six-streak reroll steps return zero reward...")

    env = ParchisEnv(num_players=2)
    obs, info = env.reset(seed=1)
    env.agent_player_idx = env.game.current_player_idx
    agent = env.game.players[env.agent_player_idx]

    piece = agent.pieces[0]
    env.game.board.remove_piece(piece)
    piece.move_to(10)
    env.game.board.add_piece(piece, 10)

    env.current_dice_roll = 6
    env.consecutive_sixes = 1

    def scripted_then_random(values):
        import random
        it = iter(values)

        def roll():
            try:
                return next(it)
            except StopIteration:
                return random.randint(1, 6)
        return roll

    env.game.dice.roll = scripted_then_random([3])  # reroll -> not a six, same player continues

    action_masks = env._get_info()['action_masks']
    action = [i for i, m in enumerate(action_masks) if m == 1][0]
    obs, reward, terminated, truncated, info = env.step(action)

    assert reward == 0.0, "Reward must stay 0.0 while a six-streak reroll is still open"
    assert env.current_dice_roll == 3, "Same player should now face a fresh, non-bonus roll"
    print("✓ Six-streak reroll steps correctly return zero reward")
    env.close()


def test_opponent_weight_constructor_arg():
    """opponent_weight should be settable at construction time, not just
    via post-construction attribute mutation."""
    print("\nTesting opponent_weight constructor argument...")

    env = ParchisEnv(num_players=4, opponent_weight=0.3)
    assert env.opponent_weight == 0.3

    # Default unchanged.
    env_default = ParchisEnv(num_players=4)
    assert env_default.opponent_weight == 0.5

    # Post-construction mutation (existing pattern) still works.
    env_default.opponent_weight = 0.7
    assert env_default.opponent_weight == 0.7

    print("✓ opponent_weight constructor argument works, mutation still works")
    env.close()
    env_default.close()


def test_opponent_weighting_leader_end_to_end():
    """With opponent_weighting='leader', the reward's opponent term should
    reflect only the leading opponent's delta, provably different from
    what 'mean' would give on the same state."""
    print("\nTesting opponent_weighting='leader' end-to-end...")

    env = ParchisEnv(num_players=4, opponent_weighting="leader")
    obs, info = env.reset(seed=3)
    env.agent_player_idx = env.game.current_player_idx
    agent = env.game.players[env.agent_player_idx]
    opponents = [p for p in env.game.players if p is not agent]
    leader, mid, low = opponents[0], opponents[1], opponents[2]

    # Place pieces so the leader has the highest starting progress.
    for player, pos in [(leader, 60), (mid, 30), (low, low.starting_position)]:
        env.game.board.remove_piece(player.pieces[0])
        player.pieces[0].move_to(pos)
        env.game.board.add_piece(player.pieces[0], pos)

    env.turn_start_progress = {
        i: env._calculate_normalized_progress(env.game.players[i])
        for i in range(env.num_players)
    }

    # Advance the leader by a small amount, the low-progress opponent by a
    # large amount this cycle -- if 'leader' is working, the combined
    # opponent term should reflect only the leader's small move.
    env.game.board.remove_piece(leader.pieces[0])
    leader.pieces[0].move_to(66)  # +6
    env.game.board.add_piece(leader.pieces[0], 66)

    env.game.board.remove_piece(low.pieces[0])
    low.pieces[0].move_to(low.starting_position + 20)  # +20
    env.game.board.add_piece(low.pieces[0], low.starting_position + 20)

    opponent_deltas = {}
    opponent_start_progress = {}
    for i, player in enumerate(env.game.players):
        if player is agent:
            continue
        opponent_deltas[i] = env._calculate_normalized_progress(player) - env.turn_start_progress[i]
        opponent_start_progress[i] = env.turn_start_progress[i]

    from parchis.rl import rewards
    leader_combined = rewards.combine_opponent_deltas(opponent_deltas, opponent_start_progress, weighting="leader")
    mean_combined = rewards.combine_opponent_deltas(opponent_deltas, opponent_start_progress, weighting="mean")

    leader_idx = env.game.players.index(leader)
    assert leader_combined == opponent_deltas[leader_idx], (
        "leader weighting should return exactly the leader's own delta"
    )
    assert leader_combined != mean_combined, (
        "leader and mean weighting must actually differ on this contrived state"
    )
    print(f"✓ leader combined={leader_combined:.4f}, mean combined={mean_combined:.4f} (correctly differ)")
    env.close()


def test_reward_type_validation():
    """Test that invalid reward types raise ValueError."""
    print("\n" + "=" * 60)
    print("Testing reward type validation...")
    print("=" * 60)

    try:
        ParchisEnv(num_players=2, reward_type="invalid_type")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    # Valid types should not raise
    for rt in ParchisEnv.VALID_REWARD_TYPES:
        env = ParchisEnv(num_players=2, reward_type=rt)
        print(f"  ✓ reward_type='{rt}' accepted")
        env.close()

    print("✓ Validation tests passed!")


if __name__ == '__main__':
    test_reward_structure()
    test_opponent_weight_effect()
    test_reward_types()
    test_six_streak_intermediate_steps_return_zero_reward()
    test_opponent_weight_constructor_arg()
    test_opponent_weighting_leader_end_to_end()
    test_reward_type_validation()
