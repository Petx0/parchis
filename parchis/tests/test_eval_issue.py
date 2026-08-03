"""
Regression test for the evaluation freeze issue described in
docs/EVALUATION_FIX.md: evaluation must never hang, action values from
model.predict() must convert cleanly to int, and max_steps_per_episode must
be honored as a hard safety limit on episode length.

This replaces an earlier version of this file that was a module-level debug
script (no test functions, no assertions, ran a full training loop as a side
effect of being imported) rather than a real, fast pytest test.
"""

import os
import tempfile

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from parchis.rl.env import ParchisEnv
from parchis.evaluation.evaluate import evaluate_agent


def mask_fn(env):
    return env.unwrapped._get_info()['action_masks']


def test_evaluate_agent_completes_without_hanging():
    """evaluate_agent() must run to completion (not hang) and honor
    max_steps_per_episode as a hard cap on every episode's length."""
    print("\nTesting evaluate_agent() completes without hanging...")

    env = ParchisEnv(num_players=4)
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=None,
        n_steps=64,
        batch_size=32,
    )
    model.learn(total_timesteps=200)
    env.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "tiny_model")
        model.save(model_path)

        max_steps_per_episode = 300
        stats = evaluate_agent(
            agent_model_path=model_path,
            opponent_model_path=None,
            n_games=2,
            num_players=4,
            deterministic=True,
            max_steps_per_episode=max_steps_per_episode,
            verbose=0,
        )

    assert stats['n_games'] == 2
    assert stats['max_length'] <= max_steps_per_episode, (
        f"An episode exceeded max_steps_per_episode: max_length="
        f"{stats['max_length']} > {max_steps_per_episode} -- the safety "
        f"timeout from docs/EVALUATION_FIX.md is not being honored"
    )
    print(f"✓ evaluate_agent() completed: {stats['n_games']} games, "
          f"max episode length {stats['max_length']} <= {max_steps_per_episode}")


if __name__ == "__main__":
    test_evaluate_agent_completes_without_hanging()
    print("\nAll tests passed!")
