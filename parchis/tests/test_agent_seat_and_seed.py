#!/usr/bin/env python3
"""
Regression tests for two bugs documented in docs/CODE_REVIEW.md (domain B,
"still open"):

1. The learning agent was hardcoded to internal player index 0, and
   Game.__init__ always rotates its player list so the dice-determined
   starting player is index 0 -- so the agent always moved first, a real
   and unaccounted-for advantage in this game. Fixed by randomizing
   ParchisEnv.agent_player_idx each reset().
2. `seed` was cosmetic: it only set self.np_random, which nothing
   downstream read, since real game randomness (Dice.roll, Player.choose_move)
   comes from Python's global `random` module. Fixed by seeding that module
   too whenever an explicit seed is given.
"""

import numpy as np

from parchis.rl.env import ParchisEnv


def test_agent_seat_is_randomized_across_episodes():
    """Over many resets, the agent should not always land on the same seat."""
    print("\nTesting agent_player_idx varies across episodes...")

    env = ParchisEnv(num_players=4)
    seats_seen = set()
    for _ in range(200):
        env.reset()
        assert 0 <= env.agent_player_idx < env.num_players
        seats_seen.add(env.agent_player_idx)

    assert len(seats_seen) > 1, (
        f"Agent only ever occupied seat(s) {seats_seen} across 200 resets -- "
        "agent_player_idx does not appear to be randomized"
    )
    print(f"✓ Agent occupied {len(seats_seen)} distinct seats across 200 resets: {sorted(seats_seen)}")
    env.close()


def test_seed_makes_agent_seat_reproducible():
    """The same seed must assign the agent the same seat."""
    print("\nTesting seed reproducibility for agent_player_idx...")

    env_a = ParchisEnv(num_players=4)
    env_a.reset(seed=123)

    env_b = ParchisEnv(num_players=4)
    env_b.reset(seed=123)

    assert env_a.agent_player_idx == env_b.agent_player_idx, (
        f"Same seed produced different agent seats: "
        f"{env_a.agent_player_idx} vs {env_b.agent_player_idx}"
    )
    print(f"✓ Both envs assigned the agent seat {env_a.agent_player_idx} under seed=123")
    env_a.close()
    env_b.close()


def test_seed_makes_dice_and_moves_reproducible():
    """The same seed must reproduce the same dice rolls and opponent moves,
    since both now draw from a Python `random` state seeded in reset()."""
    print("\nTesting seed reproducibility for dice rolls / opponent moves...")

    def run(seed, n_steps=30):
        env = ParchisEnv(num_players=4)
        obs, info = env.reset(seed=seed)
        dice_sequence = [env.current_dice_roll]
        for _ in range(n_steps):
            mask = info['action_masks']
            legal = np.where(mask)[0]
            action = int(legal[0]) if len(legal) else 0
            obs, reward, terminated, truncated, info = env.step(action)
            dice_sequence.append(env.current_dice_roll)
            if terminated or truncated:
                break
        env.close()
        return dice_sequence

    seq_a = run(seed=99)
    seq_b = run(seed=99)

    assert seq_a == seq_b, (
        f"Same seed produced different dice-roll sequences:\n{seq_a}\nvs\n{seq_b}"
    )
    print(f"✓ Dice-roll sequence reproducible across {len(seq_a)} rolls under seed=99")


if __name__ == '__main__':
    test_agent_seat_is_randomized_across_episodes()
    test_seed_makes_agent_seat_reproducible()
    test_seed_makes_dice_and_moves_reproducible()
    print("\nAll agent-seat/seed tests passed!")
