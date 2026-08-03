"""
Smoke test verifying the MaskablePPO training pipeline runs end-to-end
without TensorBoard: env creation, action masking, a short training run,
and prediction with the trained model.

Kept as a real pytest test (not module-level side-effecting code) with a
small timestep count, so it stays fast and can't silently execute training
as a side effect of test collection.
"""

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from parchis.rl.env import ParchisEnv


def mask_fn(env):
    """Return the action mask from the environment."""
    return env.unwrapped._get_info()['action_masks']


def test_training_pipeline_runs_without_tensorboard():
    """MaskablePPO should train for a few steps and then produce valid,
    int-convertible actions without TensorBoard logging enabled."""
    print("\nTesting training pipeline (no TensorBoard)...")

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
    print("✓ Training completed successfully")

    obs, info = env.reset()
    for i in range(10):
        action_masks = mask_fn(env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        action = int(action)  # Must convert cleanly from numpy scalar/array
        assert action_masks[action], f"Predicted action {action} is not a legal move"

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"  Episode ended at step {i + 1}")
            break

    print("✓ Trained model produces valid actions")
    env.close()


if __name__ == "__main__":
    test_training_pipeline_runs_without_tensorboard()
    print("\nAll tests passed!")
