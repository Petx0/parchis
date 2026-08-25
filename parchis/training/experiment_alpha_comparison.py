#!/usr/bin/env python3
"""
Experiment: Compare different opponent_weight (α) values in the turn-cycle reward structure.

This script trains 3 agents from scratch (2-player games) with different α values:
- α = 0.0: Only care about own progress
- α = 0.5: Balanced (default)
- α = 0.9: Strongly penalize opponent progress

After training each, evaluates with test games and reports:
- Average win rate
- Average final progress (player)
- Average final progress (opponent)

--opponent-weighting lets the whole sweep be run against an alternative
opponent-weighting scheme instead of the default "mean" (see
parchis/rl/rewards.py) -- applied uniformly to every α value in a given
run; compare schemes by running this script twice, once per scheme.
"leader" is only meaningfully different from "mean" with 2+ opponents, so
pass --players 3 or --players 4 when sweeping it (the default --players 2
has exactly one opponent, where the two schemes are mathematically
identical).
"""

import os
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sb3_contrib import MaskablePPO

from parchis.training.common import make_env, evaluate_model, ProgressLoggingCallback
from parchis.rl.rewards import VALID_OPPONENT_WEIGHTING_SCHEMES
from parchis.evaluation import stats as eval_stats


@dataclass
class ExperimentResult:
    """Results from training and evaluation with a specific α value and seed."""
    alpha: float
    opponent_weighting: str
    win_rate: float
    avg_player_progress: float
    avg_opponent_progress: float
    avg_episode_reward: float
    total_episodes: int
    model_path: str
    seed: int


@dataclass
class AggregatedResult:
    """
    A single α value's results aggregated across all --seeds.

    win_rate_ci is None when only one seed was run (mean_confidence_interval
    needs >= 2 samples) -- the default --seeds [42] preserves this codebase's
    original single-seed behavior exactly, so this is the common case, not
    an error state.
    """
    alpha: float
    opponent_weighting: str
    win_rate_mean: float
    win_rate_std: float
    win_rate_ci: Optional[Tuple[float, float]]
    avg_player_progress_mean: float
    avg_opponent_progress_mean: float
    avg_episode_reward_mean: float
    per_seed_results: List[ExperimentResult]


def _aggregate_seed_results(alpha, opponent_weighting, per_seed_results):
    """Aggregate one α value's per-seed ExperimentResults into an AggregatedResult."""
    win_rates = [r.win_rate for r in per_seed_results]
    win_rate_ci = (
        eval_stats.mean_confidence_interval(win_rates) if len(win_rates) >= 2 else None
    )
    return AggregatedResult(
        alpha=alpha,
        opponent_weighting=opponent_weighting,
        win_rate_mean=float(np.mean(win_rates)),
        win_rate_std=float(np.std(win_rates)),
        win_rate_ci=win_rate_ci,
        avg_player_progress_mean=float(np.mean([r.avg_player_progress for r in per_seed_results])),
        avg_opponent_progress_mean=float(np.mean([r.avg_opponent_progress for r in per_seed_results])),
        avg_episode_reward_mean=float(np.mean([r.avg_episode_reward for r in per_seed_results])),
        per_seed_results=per_seed_results,
    )


def train_agent(
    opponent_weight: float,
    opponent_weighting: str = "mean",
    total_timesteps: int = 1_000_000,
    num_players: int = 2,
    save_path: str = "./models",
    log_path: str = "./logs",
    seed: int = 42,
    verbose: int = 1
) -> str:
    """
    Train an agent with a specific opponent_weight (α) value.

    Args:
        opponent_weight: α value for turn-cycle reward
        opponent_weighting: How multiple opponents' deltas combine -- one
            of ParchisEnv.VALID_OPPONENT_WEIGHTING_SCHEMES (default "mean")
        total_timesteps: Training steps
        num_players: Number of players
        save_path: Directory to save model
        log_path: Directory for logs
        seed: Random seed
        verbose: Verbosity level

    Returns:
        Path to saved model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # No raw '.' in the name -- SB3's save() (save_util.open_path_pathlib)
    # skips appending .zip whenever pathlib.Path(name).suffix is already
    # non-empty, and a decimal point (e.g. "alpha_0.9_mean_...") makes it
    # think the name already has an extension, silently saving the
    # checkpoint without .zip. Use 'p' for the decimal point instead.
    alpha_str = f"{opponent_weight:.1f}".replace(".", "p")
    model_name = f"alpha_{alpha_str}_{opponent_weighting}_{timestamp}"

    print(f"\n{'='*70}")
    print(f"Training with α = {opponent_weight}, opponent_weighting = {opponent_weighting}")
    print(f"{'='*70}")

    env = make_env(num_players=num_players, opponent_weight=opponent_weight,
                    opponent_weighting=opponent_weighting, seed=seed)
    print(f"Environment opponent_weight: {env.unwrapped.opponent_weight}")
    print(f"Environment opponent_weighting: {env.unwrapped.opponent_weighting}")

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=verbose,
        tensorboard_log=log_path,
        seed=seed
    )

    callback = ProgressLoggingCallback(verbose=verbose)

    print(f"Starting training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=model_name,
        progress_bar=True
    )

    model_path = os.path.join(save_path, model_name)
    os.makedirs(save_path, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    env.close()

    return model_path


def run_experiment(
    alpha_values: List[float] = [0.0, 0.5, 0.9],
    opponent_weighting: str = "mean",
    total_timesteps: int = 1_000_000,
    n_eval_episodes: int = 100,
    num_players: int = 2,
    save_path: str = "./models/alpha_experiment",
    log_path: str = "./logs/alpha_experiment",
    seeds: List[int] = [42],
    verbose: int = 1
) -> List[AggregatedResult]:
    """
    Run the full experiment comparing different α values, each trained and
    evaluated once per seed in `seeds` and aggregated across seeds.

    Args:
        alpha_values: List of α values to test
        opponent_weighting: How multiple opponents' deltas combine -- one
            of ParchisEnv.VALID_OPPONENT_WEIGHTING_SCHEMES (default "mean"),
            applied uniformly to every α value in this run
        total_timesteps: Training steps per agent
        n_eval_episodes: Evaluation episodes per agent
        num_players: Number of players
        save_path: Directory to save models
        log_path: Directory for logs
        seeds: Random seeds -- one full train+eval run per (alpha, seed)
            pair. Defaults to a single seed, reproducing this script's
            original single-seed behavior exactly; pass multiple seeds
            (e.g. [42, 43, 44]) for a statistically trustworthy comparison
            (docs/RL_DESIGN_REVIEW.md Phase 4) at the cost of running the
            whole sweep once per seed.
        verbose: Verbosity level

    Returns:
        List of AggregatedResult objects, one per α value
    """
    aggregated_results = []

    print("="*70)
    print("EXPERIMENT: Comparing opponent_weight (α) values")
    print("="*70)
    print(f"α values to test: {alpha_values}")
    print(f"Opponent weighting: {opponent_weighting}")
    print(f"Training timesteps: {total_timesteps:,}")
    print(f"Evaluation episodes: {n_eval_episodes}")
    print(f"Seeds: {seeds}")
    print("="*70)

    for alpha in alpha_values:
        per_seed_results = []
        for seed in seeds:
            model_path = train_agent(
                opponent_weight=alpha,
                opponent_weighting=opponent_weighting,
                total_timesteps=total_timesteps,
                num_players=num_players,
                save_path=save_path,
                log_path=log_path,
                seed=seed,
                verbose=verbose
            )

            print(f"\nEvaluating α = {alpha}, seed = {seed} agent...")
            model = MaskablePPO.load(model_path)
            eval_env = make_env(num_players=num_players, seed=seed + 1000)
            stats = evaluate_model(model, eval_env, n_eval_episodes=n_eval_episodes, verbose=0)
            eval_env.close()

            result = ExperimentResult(
                alpha=alpha,
                opponent_weighting=opponent_weighting,
                win_rate=stats['win_rate'],
                avg_player_progress=stats.get('mean_final_progress', 0.0),
                avg_opponent_progress=stats.get('mean_opponent_progress', 0.0),
                avg_episode_reward=stats['mean_reward'],
                total_episodes=stats['n_episodes'],
                model_path=model_path,
                seed=seed,
            )
            per_seed_results.append(result)

            print(f"\nResults for α = {alpha}, seed = {seed}:")
            print(f"  Win rate: {result.win_rate:.1%}")
            print(f"  Avg player progress: {result.avg_player_progress:.4f}")
            print(f"  Avg opponent progress: {result.avg_opponent_progress:.4f}")
            print(f"  Avg episode reward: {result.avg_episode_reward:.4f}")

        aggregated = _aggregate_seed_results(alpha, opponent_weighting, per_seed_results)
        aggregated_results.append(aggregated)

        if len(seeds) > 1:
            ci_str = (f"[{aggregated.win_rate_ci[0]:.1%}, {aggregated.win_rate_ci[1]:.1%}]"
                      if aggregated.win_rate_ci else "n/a")
            print(f"\nAggregated across {len(seeds)} seeds for α = {alpha}: "
                  f"win_rate = {aggregated.win_rate_mean:.1%} ± {aggregated.win_rate_std:.1%} "
                  f"(95% CI {ci_str})")

    _print_summary(aggregated_results, len(seeds))
    _save_results_txt(aggregated_results, opponent_weighting, total_timesteps,
                       n_eval_episodes, seeds, save_path, len(seeds))

    return aggregated_results


def _print_summary(aggregated_results, n_seeds):
    """Print the final comparison table and, when multiple seeds were run,
    a CI-gated 'best config' verdict (docs/RL_DESIGN_REVIEW.md Phase 4) --
    a raw point-estimate max() is indistinguishable from seed noise."""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'α':<8} {'Weighting':<12} {'Win Rate':<16} {'Player Prog':<14} {'Opp Prog':<14} {'Avg Reward':<12}")
    print("-"*70)

    for r in aggregated_results:
        win_rate_str = f"{r.win_rate_mean:.1%} ± {r.win_rate_std:.1%}" if n_seeds > 1 else f"{r.win_rate_mean:.1%}"
        print(f"{r.alpha:<8.1f} {r.opponent_weighting:<12} {win_rate_str:<16} "
              f"{r.avg_player_progress_mean:<14.4f} {r.avg_opponent_progress_mean:<14.4f} "
              f"{r.avg_episode_reward_mean:<12.4f}")

    print("="*70)

    entries = [(r.alpha, r.win_rate_mean, r.win_rate_ci) for r in aggregated_results]
    ranked, confirmed = eval_stats.rank_by_mean_with_ci(entries)
    if ranked:
        best_alpha = ranked[0][0]
        if n_seeds > 1:
            verdict = "statistically confirmed" if confirmed else "NOT statistically confirmed (CIs overlap)"
            print(f"\nBest by win rate: α = {best_alpha} ({verdict} across {n_seeds} seeds)")
        else:
            print(f"\nBest by win rate: α = {best_alpha} (single seed -- run with --seeds for a confirmed result)")


def _save_results_txt(aggregated_results, opponent_weighting, total_timesteps,
                       n_eval_episodes, seeds, save_path, n_seeds):
    """Save results to a plain-text file (matches this script's existing format)."""
    results_file = os.path.join(save_path, "experiment_results.txt")
    with open(results_file, 'w') as f:
        f.write("EXPERIMENT: Comparing opponent_weight (α) values\n")
        f.write(f"Opponent weighting: {opponent_weighting}\n")
        f.write(f"Training timesteps: {total_timesteps:,}\n")
        f.write(f"Evaluation episodes: {n_eval_episodes}\n")
        f.write(f"Seeds: {seeds}\n\n")
        f.write(f"{'α':<8} {'Weighting':<12} {'Win Rate':<16} {'Player Prog':<14} {'Opp Prog':<14} {'Avg Reward':<12}\n")
        f.write("-"*70 + "\n")
        for r in aggregated_results:
            win_rate_str = f"{r.win_rate_mean:.1%} ± {r.win_rate_std:.1%}" if n_seeds > 1 else f"{r.win_rate_mean:.1%}"
            f.write(f"{r.alpha:<8.1f} {r.opponent_weighting:<12} {win_rate_str:<16} "
                    f"{r.avg_player_progress_mean:<14.4f} {r.avg_opponent_progress_mean:<14.4f} "
                    f"{r.avg_episode_reward_mean:<12.4f}\n")
            for seed_result in r.per_seed_results:
                f.write(f"    seed={seed_result.seed}: win_rate={seed_result.win_rate:.1%}, "
                        f"model={seed_result.model_path}\n")

    print(f"\nResults saved to: {results_file}")


def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Experiment comparing different α (opponent_weight) values"
    )

    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 0.9],
        help="α values to test (default: 0.0 0.5 0.9)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Training timesteps per agent (default: 1,000,000)"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Evaluation episodes per agent (default: 100)"
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        help="Number of players (default: 2)"
    )
    parser.add_argument(
        "--opponent-weighting",
        type=str,
        default="mean",
        choices=list(VALID_OPPONENT_WEIGHTING_SCHEMES),
        help="How multiple opponents' progress deltas combine into the "
             "reward's opponent-progress term (default: mean). 'leader' "
             "only differs from 'mean' with --players 3 or 4."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./models/alpha_experiment",
        help="Path to save models"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="./logs/alpha_experiment",
        help="Path for TensorBoard logs"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds -- one full train+eval run per (alpha, seed) "
             "pair (default: 42, reproducing the original single-seed "
             "behavior). Pass multiple, e.g. --seeds 42 43 44, for a "
             "statistically trustworthy comparison (docs/RL_DESIGN_REVIEW.md "
             "Phase 4) at the cost of running the sweep once per seed."
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1],
        help="Verbosity level (default: 1)"
    )

    args = parser.parse_args()

    run_experiment(
        alpha_values=args.alphas,
        opponent_weighting=args.opponent_weighting,
        total_timesteps=args.timesteps,
        n_eval_episodes=args.eval_episodes,
        num_players=args.players,
        save_path=args.save_path,
        log_path=args.log_path,
        seeds=args.seeds,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
