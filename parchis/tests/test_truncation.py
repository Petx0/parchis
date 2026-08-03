"""
Test that episodes reliably end (terminate or truncate) under normal random
play with the default max_episode_length -- a regression guard against the
env ever looping forever. See test_truncation_force.py for a test that the
truncation mechanism itself fires correctly when forced with a low limit.
"""

from parchis.rl.env import ParchisEnv


def test_episode_ends_under_default_limit():
    """A random-play episode must end (terminated or truncated) within a
    generous step bound under the default max_episode_length."""
    print("\nTesting episode reliably ends under the default max_episode_length...")

    env = ParchisEnv(num_players=4)
    obs, info = env.reset(seed=0)

    step_count = 0
    terminated = False
    truncated = False

    # env.max_episode_length counts game turns (across all players), while
    # this loop counts agent-level step() calls -- each of which can span
    # many game turns via the opponent auto-play loop -- so 2000 agent
    # steps is a generous bound relative to the default max_episode_length
    # of 1000 game turns.
    while not (terminated or truncated) and step_count < 2000:
        action_masks = info['action_masks']
        action = next((i for i, mask in enumerate(action_masks) if mask), 0)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    assert terminated or truncated, (
        f"Episode never ended within {step_count} agent steps "
        f"(episode_length={env.episode_length}, max_episode_length={env.max_episode_length})"
    )
    print(f"✓ Episode ended after {step_count} agent steps "
          f"(episode_length={env.episode_length}, terminated={terminated}, truncated={truncated})")
    env.close()


if __name__ == "__main__":
    test_episode_ends_under_default_limit()
    print("\nAll tests passed!")
