"""
Main training script for Parchís environment using PPO with action masking.

Trains a PPO agent against random opponents, with checkpointing, TensorBoard
logging, and evaluation. Shared building blocks (environment factory, action
masking, progress logging, evaluation loop) live in parchis.training.common;
shared CLI argument groups live in parchis.training.cli. See
docs/CODE_REVIEW.md for why this split exists.
"""

import os
import argparse
from datetime import datetime

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from parchis.training.common import make_env, evaluate_model, ProgressLoggingCallback
from parchis.training import cli


def train(
    total_timesteps=1_000_000,
    num_players=4,
    opponent_weight=0.5,
    reward_type="progress_delta",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    checkpoint_freq=50_000,
    eval_freq=None,  # Disabled by default due to compatibility issues with MaskablePPO
    n_eval_episodes=10,
    save_path="./models",
    log_path="./logs",
    model_name=None,
    seed=42,
    verbose=1
):
    """
    Train a MaskablePPO agent on the Parchís environment.

    Args:
        total_timesteps: Total training timesteps
        num_players: Number of players in the game
        opponent_weight: α value for the reward's opponent-progress term
        reward_type: One of ParchisEnv.VALID_REWARD_TYPES
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps per rollout
        batch_size: Batch size for training
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clipping parameter
        ent_coef: Entropy coefficient for exploration
        checkpoint_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps (None to disable mid-training evaluation)
        n_eval_episodes: Number of episodes for evaluation
        save_path: Directory to save models
        log_path: Directory for TensorBoard logs
        model_name: Name for the model (auto-generated if None)
        seed: Random seed
        verbose: Verbosity level

    Returns:
        Trained model
    """
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"parchis_ppo_{num_players}p_{timestamp}"

    print("=" * 70)
    print("Parchís PPO Training")
    print("=" * 70)
    print(f"Model name: {model_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of players: {num_players}")
    print(f"Opponent weight (α): {opponent_weight}")
    print(f"Reward type: {reward_type}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"n_epochs: {n_epochs}, gae_lambda: {gae_lambda}, clip_range: {clip_range}")
    print(f"Checkpoint frequency: {checkpoint_freq:,}")
    eval_freq_str = f"{eval_freq:,}" if eval_freq is not None else "Disabled"
    print(f"Evaluation frequency: {eval_freq_str}")
    print("=" * 70)

    print("\nCreating environments...")
    train_env = make_env(num_players=num_players, opponent_weight=opponent_weight,
                          reward_type=reward_type, seed=seed)
    eval_env = make_env(num_players=num_players, opponent_weight=opponent_weight,
                         reward_type=reward_type, seed=seed + 1)

    print("Creating model...")
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=verbose,
        tensorboard_log=log_path,
        seed=seed
    )

    print("Setting up callbacks...")
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(save_path, model_name),
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    progress_callback = ProgressLoggingCallback(verbose=0)
    callbacks = [checkpoint_callback, progress_callback]

    if eval_freq is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(save_path, model_name),
            log_path=os.path.join(log_path, model_name),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )
        callbacks.append(eval_callback)
        print(f"Mid-training evaluation enabled (every {eval_freq:,} steps)")
    else:
        print("Mid-training evaluation disabled (will evaluate at end only)")

    print("\nStarting training...")
    print(f"TensorBoard logs: {log_path}/{model_name}")
    print(f"To monitor training, run: tensorboard --logdir {log_path}")
    print("=" * 70)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=model_name,
            progress_bar=True
        )

        final_model_path = os.path.join(save_path, model_name, "final_model")
        model.save(final_model_path)
        print(f"\n✓ Training completed!")
        print(f"✓ Final model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupted_model_path = os.path.join(save_path, model_name, "interrupted_model")
        model.save(interrupted_model_path)
        print(f"✓ Model saved to: {interrupted_model_path}")

    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    stats = evaluate_model(model, eval_env, n_eval_episodes=100)
    print(f"Mean reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")

    train_env.close()
    eval_env.close()

    return model


def load_and_evaluate(model_path, num_players=4, opponent_weight=0.5,
                       reward_type="progress_delta", n_eval_episodes=10):
    """
    Load a trained model and evaluate it.

    Args:
        model_path: Path to the saved model
        num_players: Number of players
        opponent_weight: α value for the reward's opponent-progress term
        reward_type: One of ParchisEnv.VALID_REWARD_TYPES
        n_eval_episodes: Number of episodes to evaluate
    """
    print(f"Loading model from {model_path}...")
    model = MaskablePPO.load(model_path)

    print("Creating evaluation environment...")
    env = make_env(num_players=num_players, opponent_weight=opponent_weight,
                    reward_type=reward_type, seed=42)

    print(f"\nEvaluating for {n_eval_episodes} episodes...")
    evaluate_model(model, env, n_eval_episodes=n_eval_episodes)

    env.close()


def main():
    """Main training script with command-line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent for Parchís")

    parser.add_argument("--timesteps", type=int, default=1_000_000,
                         help="Total training timesteps (default: 1,000,000)")
    cli.add_env_args(parser, default_players=4)
    cli.add_ppo_hyperparam_args(parser, default_gamma=0.995)
    cli.add_checkpoint_eval_args(parser, default_checkpoint_freq=50_000, default_n_eval_episodes=10)
    parser.add_argument("--eval-freq", type=int, default=None,
                         help="Evaluation frequency in timesteps (default: None - disabled). "
                              "Set to enable mid-training evaluation.")
    cli.add_io_args(parser)
    cli.add_common_args(parser)

    parser.add_argument("--evaluate", type=str, default=None,
                         help="Path to model to evaluate (skips training)")

    args = parser.parse_args()

    if args.evaluate:
        load_and_evaluate(
            args.evaluate,
            num_players=args.players,
            opponent_weight=args.opponent_weight,
            reward_type=args.reward_type,
            n_eval_episodes=args.n_eval_episodes
        )
    else:
        train(
            total_timesteps=args.timesteps,
            num_players=args.players,
            opponent_weight=args.opponent_weight,
            reward_type=args.reward_type,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            checkpoint_freq=args.checkpoint_freq,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            save_path=args.save_path,
            log_path=args.log_path,
            model_name=args.model_name,
            seed=args.seed,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
