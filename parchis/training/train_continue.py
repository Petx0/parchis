"""
Continue training a previously saved Parchís PPO model.

This script loads a trained model and continues training for additional timesteps,
preserving all learned weights and training progress.

Usage:
    python -m parchis.training.train_continue --model-path ./models/parchis_quick_test/final_model --timesteps 2000000
"""

import os
import argparse
from datetime import datetime

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

from parchis.training.common import make_env, evaluate_model, ProgressLoggingCallback
from parchis.training import cli


def continue_training(
    model_path,
    additional_timesteps=2_000_000,
    num_players=2,
    opponent_weight=0.5,
    reward_type="progress_delta",
    checkpoint_freq=100_000,
    n_eval_episodes=100,
    save_path=None,
    log_path="./logs",
    model_name=None,
    seed=42,
    verbose=1
):
    """
    Continue training from a saved model checkpoint.

    Args:
        model_path: Path to the saved model to load
        additional_timesteps: Additional timesteps to train for
        num_players: Number of players in the game
        opponent_weight: α value for the reward's opponent-progress term
        reward_type: One of ParchisEnv.VALID_REWARD_TYPES
        checkpoint_freq: Save checkpoint every N steps
        n_eval_episodes: Number of episodes for final evaluation
        save_path: Directory to save continued model (auto-generated if None)
        log_path: Directory for TensorBoard logs
        model_name: Name for the continued model (auto-generated if None)
        seed: Random seed
        verbose: Verbosity level

    Returns:
        Trained model
    """
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"parchis_continued_{timestamp}"

    if save_path is None:
        save_path = f"./models/{model_name}"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print("=" * 70)
    print("Parchís PPO Continue Training")
    print("=" * 70)
    print(f"Loading model from: {model_path}")
    print(f"Additional timesteps: {additional_timesteps:,}")
    print(f"Save path: {save_path}")
    print(f"Model name: {model_name}")
    print(f"Checkpoint frequency: {checkpoint_freq:,}")
    print("=" * 70)

    print("\nLoading model...")
    model = MaskablePPO.load(model_path)

    print("Creating environments...")
    print(f"Number of players: {num_players}")
    print(f"Opponent weight (α): {opponent_weight}")
    print(f"Reward type: {reward_type}")
    train_env = make_env(num_players=num_players, opponent_weight=opponent_weight,
                          reward_type=reward_type, seed=seed)
    eval_env = make_env(num_players=num_players, opponent_weight=opponent_weight,
                         reward_type=reward_type, seed=seed + 1)

    model.set_env(train_env)
    model.tensorboard_log = log_path

    print("Setting up callbacks...")
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    progress_callback = ProgressLoggingCallback(verbose=0)

    print("\nContinuing training...")
    print(f"TensorBoard logs: {log_path}/{model_name}")
    print(f"To monitor training, run: tensorboard --logdir {log_path}")
    print("=" * 70)

    try:
        model.learn(
            total_timesteps=additional_timesteps,
            callback=[checkpoint_callback, progress_callback],
            tb_log_name=model_name,
            reset_num_timesteps=False,  # Important! Continue from previous timesteps
            progress_bar=True
        )

        final_model_path = os.path.join(save_path, "final_model")
        model.save(final_model_path)
        print(f"\n✓ Training completed!")
        print(f"✓ Final model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupted_model_path = os.path.join(save_path, "interrupted_model")
        model.save(interrupted_model_path)
        print(f"✓ Model saved to: {interrupted_model_path}")

    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    stats = evaluate_model(model, eval_env, n_eval_episodes=n_eval_episodes)
    print(f"Mean reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")

    train_env.close()
    eval_env.close()

    return model


def main():
    """Main script with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Continue training a Parchís PPO model from checkpoint"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model to continue training from"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Additional training timesteps (default: 2,000,000)"
    )
    cli.add_env_args(parser, default_players=2)
    cli.add_checkpoint_eval_args(parser, default_checkpoint_freq=100_000, default_n_eval_episodes=100)
    cli.add_io_args(parser, default_save_path=None)
    cli.add_common_args(parser)

    args = parser.parse_args()

    if not os.path.exists(args.model_path + ".zip"):
        print(f"Error: Model not found at {args.model_path}")
        print("Note: Do not include the .zip extension in the path")
        return

    continue_training(
        model_path=args.model_path,
        additional_timesteps=args.timesteps,
        num_players=args.players,
        opponent_weight=args.opponent_weight,
        reward_type=args.reward_type,
        checkpoint_freq=args.checkpoint_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_path=args.save_path,
        log_path=args.log_path,
        model_name=args.model_name,
        seed=args.seed,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
