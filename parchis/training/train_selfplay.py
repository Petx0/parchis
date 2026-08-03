"""
Self-play training script for Parchís using PPO with rolling opponent updates.

This script implements rolling self-play where:
- The learning agent (being trained) occupies a randomly-assigned seat each
  episode (see ParchisEnv.agent_player_idx)
- Every other seat uses a frozen copy of the agent's policy from earlier in training
- Opponents are periodically updated with the latest agent weights

This creates a curriculum where the agent faces increasingly stronger opponents,
leading to faster learning and stronger final performance compared to random opponents.
"""

import os
import collections
import argparse
from datetime import datetime

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from parchis.rl.env_selfplay import ParchisSelfPlayEnv
from parchis.rl import opponent_pool
from parchis.training.common import (
    mask_fn, evaluate_model, make_env, ProgressLoggingCallback, FixedOpponentEvalCallback,
)
from parchis.training import cli


class SelfPlayCallback(BaseCallback):
    """
    Callback to periodically update the self-play opponent pool during
    training. Maintains a sliding window of the last `pool_size` opponent
    checkpoints (older checkpoint FILES are never deleted -- only what's
    live-sampled during training is capped -- so a future checkpoint-ladder
    evaluation still has the full training history to work with).
    """

    def __init__(
        self,
        update_freq=50_000,
        save_path="./models",
        pool_size=opponent_pool.DEFAULT_POOL_SIZE,
        pool_sampling_strategy=opponent_pool.DEFAULT_POOL_SAMPLING_STRATEGY,
        pool_eval_episodes=10,
        num_players=2,
        opponent_weight=0.5,
        reward_type="progress_delta",
        seed=None,
        verbose=1
    ):
        """
        Initialize the self-play callback.

        Args:
            update_freq: Update the opponent pool every N timesteps
            save_path: Directory to save opponent checkpoints
            pool_size: Max number of past checkpoints kept live-sampled from
                (a deque -- the oldest is evicted from the live pool once
                full, but its checkpoint file on disk is never deleted)
            pool_sampling_strategy: One of
                opponent_pool.VALID_POOL_SAMPLING_STRATEGIES. "win_rate"
                re-evaluates every pool member against the live training
                model each update (pool_size * pool_eval_episodes extra
                episodes per update -- a real, recurring cost)
            pool_eval_episodes: Episodes per pool member when scoring for
                the "win_rate" strategy (unused for "uniform"/"recency")
            num_players, opponent_weight, reward_type, seed: only used to
                lazily build a persistent scoring env for "win_rate"
            verbose: Verbosity level
        """
        super().__init__(verbose)
        if pool_sampling_strategy not in opponent_pool.VALID_POOL_SAMPLING_STRATEGIES:
            raise ValueError(
                f"pool_sampling_strategy must be one of "
                f"{opponent_pool.VALID_POOL_SAMPLING_STRATEGIES}, got {pool_sampling_strategy!r}"
            )
        self.update_freq = update_freq
        self.save_path = save_path
        self.pool_size = pool_size
        self.pool_sampling_strategy = pool_sampling_strategy
        self.pool_eval_episodes = pool_eval_episodes
        self.num_players = num_players
        self.opponent_weight = opponent_weight
        self.reward_type = reward_type
        self.seed = seed
        self.last_update = 0
        self.update_count = 0
        self.pool = collections.deque(maxlen=pool_size)
        self._pool_eval_env = None
        self._pool_eval_selfplay_env = None

    def _on_step(self) -> bool:
        """
        Called at each step. Checks if it's time to update the opponent pool.

        Returns:
            bool: True to continue training
        """
        if self.num_timesteps - self.last_update >= self.update_freq:
            self._update_opponents()
            self.last_update = self.num_timesteps

        return True

    @staticmethod
    def _unwrap_to_selfplay_env(env):
        """Unwrap through Monitor/ActionMasker wrappers to the underlying
        ParchisSelfPlayEnv, or None if it isn't one."""
        while hasattr(env, 'env'):
            env = env.env
        return env if isinstance(env, ParchisSelfPlayEnv) else None

    def _score_pool_members(self):
        """Re-score every pool member's win-rate against the CURRENT
        training model (scores go stale as the model keeps training, so
        recompute all rather than incrementally track). Reuses one
        persistent scoring env across calls rather than rebuilding it."""
        if self._pool_eval_env is None:
            self._pool_eval_env = make_selfplay_env(
                num_players=self.num_players,
                opponent_weight=self.opponent_weight,
                reward_type=self.reward_type,
            )
            self._pool_eval_selfplay_env = self._unwrap_to_selfplay_env(self._pool_eval_env)

        win_rates = []
        for entry in self.pool:
            self._pool_eval_selfplay_env.update_opponent_pool([entry['model']])
            stats = evaluate_model(
                self.model, self._pool_eval_env,
                n_eval_episodes=self.pool_eval_episodes,
                deterministic=True, verbose=0,
            )
            win_rates.append(stats['win_rate'])
        return win_rates

    def _update_opponents(self):
        """Update the opponent pool with the agent's current weights."""
        self.update_count += 1

        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"Self-Play Update #{self.update_count} at {self.num_timesteps:,} timesteps")
            print(f"{'='*70}")

        # Save current model as an opponent checkpoint. Never deleted, even
        # once evicted from the live pool below -- see class docstring.
        opponent_path = os.path.join(
            self.save_path,
            f"opponent_checkpoint_{self.update_count}_{self.num_timesteps}steps"
        )
        self.model.save(opponent_path)

        if self.verbose > 0:
            print(f"✓ Saved opponent checkpoint: {opponent_path}")

        try:
            # Get the environment (handling Monitor and ActionMasker wrappers)
            env = self._unwrap_to_selfplay_env(self.training_env.envs[0])  # DummyVecEnv wraps a single env

            if env is None:
                if self.verbose > 0:
                    print("Warning: Environment is not ParchisSelfPlayEnv, skipping opponent update")
                return

            # Diversity KPI for the OUTGOING pool window, before this update
            # changes it. Skipped on the very first update: the pool was
            # empty and opponents were random, so there's nothing to measure.
            old_pool_size = len(env.opponent_pool)
            if old_pool_size > 0:
                counts = [env.opponent_selection_counts.get(i, 0) for i in range(old_pool_size)]
                diversity = opponent_pool.pool_diversity_entropy(counts)
                self.logger.record('metrics/opponent_pool_diversity', diversity)

            # Load a fresh, independent copy for the pool (frozen at this
            # checkpoint; device="cpu" keeps pool members off whatever
            # device the actively-training model uses).
            opponent_model = MaskablePPO.load(opponent_path, device="cpu")
            self.pool.append({
                'model': opponent_model,
                'path': opponent_path,
                'timesteps': self.num_timesteps,
            })

            if self.pool_sampling_strategy == "uniform":
                weights = [1.0] * len(self.pool)
            elif self.pool_sampling_strategy == "recency":
                weights = opponent_pool.compute_recency_weights(len(self.pool))
            else:  # "win_rate"
                weights = opponent_pool.compute_win_rate_weights(self._score_pool_members())

            env.update_opponent_pool([entry['model'] for entry in self.pool], weights)
            self.logger.record('metrics/opponent_pool_size', len(self.pool))

            if self.verbose > 0:
                print(f"✓ Opponent pool updated: {len(self.pool)} member(s), "
                      f"strategy={self.pool_sampling_strategy}")

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Failed to update opponents: {e}")

        if self.verbose > 0:
            print(f"{'='*70}\n")


def make_selfplay_env(opponent_model=None, num_players=2, opponent_weight=0.5,
                       reward_type="progress_delta", seed=None, pool_seed=None):
    """
    Create and wrap a self-play Parchís environment with action masking.

    Args:
        opponent_model: Model to use for opponents (if None, uses random)
        num_players: Number of players (2-4)
        opponent_weight: α value for the reward's opponent-progress term
        reward_type: One of ParchisEnv.VALID_REWARD_TYPES
        seed: Random seed (drives the game's dice rolls via the global
            `random` module -- see ParchisEnv.reset())
        pool_seed: Seed for the dedicated opponent-pool-sampling RNG.
            Deliberately independent of `seed` -- see ParchisSelfPlayEnv.

    Returns:
        Wrapped environment ready for training
    """
    env = ParchisSelfPlayEnv(
        opponent_model=opponent_model,
        num_players=num_players,
        opponent_weight=opponent_weight,
        reward_type=reward_type,
        pool_seed=pool_seed,
    )
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)

    return env


def train_selfplay(
    initial_model_path=None,
    total_timesteps=2_000_000,
    num_players=2,
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
    opponent_update_freq=50_000,
    pool_size=opponent_pool.DEFAULT_POOL_SIZE,
    pool_sampling_strategy=opponent_pool.DEFAULT_POOL_SAMPLING_STRATEGY,
    pool_eval_episodes=10,
    baseline_eval_freq=None,
    baseline_eval_episodes=20,
    checkpoint_freq=100_000,
    n_eval_episodes=100,
    save_path="./models",
    log_path="./logs",
    model_name=None,
    seed=42,
    verbose=1
):
    """
    Train a PPO agent using self-play.

    Args:
        initial_model_path: Path to initial model (if None, starts from scratch with random opponents)
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
        opponent_update_freq: Update the opponent pool every N timesteps
        pool_size: Max number of past checkpoints kept live-sampled from
            (default 5 -- this is a default-behavior change from the old
            single-rolling-snapshot self-play; pass 1 to reproduce the old
            steady-state behavior exactly)
        pool_sampling_strategy: One of opponent_pool.VALID_POOL_SAMPLING_STRATEGIES
            ("uniform" default, "recency", or "win_rate" -- the latter costs
            pool_size * pool_eval_episodes extra evaluation episodes every
            opponent_update_freq interval)
        pool_eval_episodes: Episodes per pool member for "win_rate" scoring
        baseline_eval_freq: Evaluate against a fixed random-opponent baseline
            every N timesteps, logged as metrics/win_rate_vs_baseline --
            this is what makes self-play progress visible as a genuine
            curve, decoupled from the moving self-play opponent (see
            FixedOpponentEvalCallback). Defaults to opponent_update_freq
            if not given.
        baseline_eval_episodes: Episodes per baseline evaluation (default 20;
            kept small since this runs repeatedly during training)
        checkpoint_freq: Save checkpoint every N steps
        n_eval_episodes: Number of episodes for evaluation
        save_path: Directory to save models
        log_path: Directory for TensorBoard logs
        model_name: Name for the model (auto-generated if None)
        seed: Random seed
        verbose: Verbosity level

    Returns:
        Trained model
    """
    if baseline_eval_freq is None:
        baseline_eval_freq = opponent_update_freq
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"parchis_selfplay_{timestamp}"

    selfplay_save_path = os.path.join(save_path, model_name)
    os.makedirs(selfplay_save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print("=" * 70)
    print("Parchís Self-Play Training")
    print("=" * 70)
    print(f"Model name: {model_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of players: {num_players}")
    print(f"Opponent weight (α): {opponent_weight}")
    print(f"Reward type: {reward_type}")
    print(f"Opponent update frequency: {opponent_update_freq:,}")
    print(f"Opponent pool size: {pool_size} (sampling strategy: {pool_sampling_strategy})")
    print(f"Checkpoint frequency: {checkpoint_freq:,}")
    if initial_model_path:
        print(f"Starting from: {initial_model_path}")
    else:
        print("Starting from: Random initialization")
    print("=" * 70)

    # Load initial model or create new one
    if initial_model_path and os.path.exists(initial_model_path + ".zip"):
        print("\nLoading initial model...")
        initial_model = MaskablePPO.load(initial_model_path)
        opponent_model = MaskablePPO.load(initial_model_path)  # Copy for opponents
        print("✓ Loaded model for both learning agent and initial opponents")
    else:
        if initial_model_path:
            print(f"\nWarning: Model not found at {initial_model_path}, starting from scratch")
        initial_model = None
        opponent_model = None  # Start with random opponents
        print("✓ Starting with random opponents")

    # Create training environment
    print("\nCreating self-play environment...")
    train_env = make_selfplay_env(
        opponent_model=opponent_model,
        num_players=num_players,
        opponent_weight=opponent_weight,
        reward_type=reward_type,
        seed=seed,
        pool_seed=(seed + 3) if seed is not None else None,
    )
    print("✓ Self-play environment created")

    # Create or update model
    if initial_model is not None:
        print("\nSetting up model from checkpoint...")
        model = initial_model
        model.set_env(train_env)
        model.tensorboard_log = log_path
    else:
        print("\nCreating new model...")
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

    print("✓ Model ready")

    # Setup self-play callback
    print("\nSetting up self-play callback...")
    selfplay_callback = SelfPlayCallback(
        update_freq=opponent_update_freq,
        save_path=selfplay_save_path,
        pool_size=pool_size,
        pool_sampling_strategy=pool_sampling_strategy,
        pool_eval_episodes=pool_eval_episodes,
        num_players=num_players,
        opponent_weight=opponent_weight,
        reward_type=reward_type,
        seed=seed,
        verbose=verbose
    )
    print(f"✓ Opponent pool (size {pool_size}, strategy={pool_sampling_strategy}) "
          f"will be updated every {opponent_update_freq:,} timesteps")

    # Progress logging (metrics/final_progress, metrics/win_rate --
    # measured against the self-play opponent, which itself keeps
    # improving) and a fixed-baseline evaluator (metrics/win_rate_vs_baseline
    # -- measured against stationary random opponents, so it isolates
    # whether the agent itself is improving over the course of training).
    # Neither was previously wired into this script at all.
    progress_callback = ProgressLoggingCallback(verbose=0)
    baseline_eval_env = make_env(num_players=num_players, reward_type=reward_type, seed=seed + 2)
    baseline_callback = FixedOpponentEvalCallback(
        eval_env=baseline_eval_env,
        eval_freq=baseline_eval_freq,
        n_eval_episodes=baseline_eval_episodes,
        verbose=verbose,
    )
    print(f"✓ Baseline (vs random) evaluation every {baseline_eval_freq:,} timesteps "
          f"({baseline_eval_episodes} episodes)")

    # Train the model
    print("\nStarting self-play training...")
    print(f"TensorBoard logs: {log_path}/{model_name}")
    print(f"To monitor training, run: tensorboard --logdir {log_path}")
    print("=" * 70)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[selfplay_callback, progress_callback, baseline_callback],
            tb_log_name=model_name,
            reset_num_timesteps=(initial_model is None),  # Reset only if starting fresh
            progress_bar=True
        )

        final_model_path = os.path.join(selfplay_save_path, "final_model")
        model.save(final_model_path)
        print(f"\n✓ Training completed!")
        print(f"✓ Final model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupted_model_path = os.path.join(selfplay_save_path, "interrupted_model")
        model.save(interrupted_model_path)
        print(f"✓ Model saved to: {interrupted_model_path}")

    # Final evaluation against random opponents (to measure absolute strength)
    print("\n" + "=" * 70)
    print("Final Evaluation (vs Random Opponents)")
    print("=" * 70)

    eval_env = make_env(num_players=num_players, reward_type=reward_type, seed=seed + 1)
    stats = evaluate_model(model, eval_env, n_eval_episodes=n_eval_episodes)
    print(f"Mean reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    eval_env.close()

    train_env.close()
    baseline_eval_env.close()
    if selfplay_callback._pool_eval_env is not None:
        selfplay_callback._pool_eval_env.close()

    return model


def main():
    """Main training script with command-line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent with self-play for Parchís")

    parser.add_argument(
        "--initial-model",
        type=str,
        default=None,
        help="Path to initial model to start from (optional)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total training timesteps (default: 2,000,000)"
    )
    cli.add_env_args(parser, default_players=2)
    cli.add_ppo_hyperparam_args(parser, default_gamma=0.995)
    parser.add_argument(
        "--opponent-update-freq",
        type=int,
        default=50_000,
        help="Update the opponent pool every N timesteps (default: 50,000)"
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=opponent_pool.DEFAULT_POOL_SIZE,
        help="Max number of past checkpoints kept live-sampled as opponents "
             "(default: 5). NOTE: this changes self-play's default behavior "
             "from the old single-rolling-snapshot opponent -- pass "
             "--pool-size 1 to reproduce the old steady-state behavior."
    )
    parser.add_argument(
        "--pool-sampling-strategy",
        type=str,
        default=opponent_pool.DEFAULT_POOL_SAMPLING_STRATEGY,
        choices=list(opponent_pool.VALID_POOL_SAMPLING_STRATEGIES),
        help="How to sample an opponent from the pool each episode: "
             "'uniform' (default, equal weight), 'recency' (bias toward "
             "newer checkpoints), or 'win_rate' (bias toward checkpoints "
             "the current model is weakest against -- COSTS pool-size * "
             "pool-eval-episodes extra evaluation episodes every "
             "opponent-update-freq interval)"
    )
    parser.add_argument(
        "--pool-eval-episodes",
        type=int,
        default=10,
        help="Episodes per pool member when scoring for "
             "--pool-sampling-strategy win_rate (default: 10; unused for "
             "other strategies)"
    )
    parser.add_argument(
        "--baseline-eval-freq",
        type=int,
        default=None,
        help="Evaluate against a fixed random-opponent baseline every N timesteps, "
             "logged as metrics/win_rate_vs_baseline (default: same as "
             "--opponent-update-freq)"
    )
    parser.add_argument(
        "--baseline-eval-episodes",
        type=int,
        default=20,
        help="Episodes per baseline evaluation (default: 20)"
    )
    cli.add_checkpoint_eval_args(parser, default_checkpoint_freq=100_000, default_n_eval_episodes=100)
    cli.add_io_args(parser)
    cli.add_common_args(parser)

    args = parser.parse_args()

    train_selfplay(
        initial_model_path=args.initial_model,
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
        opponent_update_freq=args.opponent_update_freq,
        pool_size=args.pool_size,
        pool_sampling_strategy=args.pool_sampling_strategy,
        pool_eval_episodes=args.pool_eval_episodes,
        baseline_eval_freq=args.baseline_eval_freq,
        baseline_eval_episodes=args.baseline_eval_episodes,
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
