"""
Test that episode truncation actually fires when max_episode_length is reached.
"""

from parchis.rl.env import ParchisEnv


def test_forced_truncation():
    """Setting a very low max_episode_length should reliably truncate the
    episode within that bound rather than running forever."""
    print("\nTesting forced truncation with low max_episode_length...")

    env = ParchisEnv(num_players=4)
    env.max_episode_length = 50  # Force truncation quickly

    obs, info = env.reset()

    step_count = 0
    terminated = False
    truncated = False

    # Generous upper bound: an agent step can span many game turns (a full
    # opponent auto-play cycle), so this should truncate well before 500
    # agent-level steps even though max_episode_length counts game turns.
    while not (terminated or truncated) and step_count < 500:
        action_masks = info['action_masks']
        action = next((i for i, mask in enumerate(action_masks) if mask), 0)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    assert terminated or truncated, (
        f"Episode neither terminated nor truncated within {step_count} steps "
        f"despite max_episode_length={env.max_episode_length}"
    )
    assert env.episode_length >= env.max_episode_length or terminated, (
        f"Expected truncation once episode_length ({env.episode_length}) reaches "
        f"max_episode_length ({env.max_episode_length}), unless the game ended "
        f"first (terminated={terminated})"
    )
    print(f"✓ Episode ended after {step_count} agent steps "
          f"(episode_length={env.episode_length}, terminated={terminated}, truncated={truncated})")
    env.close()


if __name__ == "__main__":
    test_forced_truncation()
    print("\nAll tests passed!")
