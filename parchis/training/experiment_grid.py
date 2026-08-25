#!/usr/bin/env python3
"""
Experiment Grid: Compare reward structures x network architectures for Parchis RL.

Runs a 3x3 grid of experiments:
  Reward types:  progress_delta, win_loss, win_loss_shaped
  Architectures: small [64,64] Tanh, medium [256,256] ReLU, large [512,256,128] ReLU

Each experiment trains a MaskablePPO agent for a configurable number of timesteps,
evaluates against random opponents, and logs results to TensorBoard + JSON.

Usage:
    python -m parchis.training.experiment_grid
    python -m parchis.training.experiment_grid --timesteps 500000 --players 2
    python -m parchis.training.experiment_grid --filter-reward win_loss --filter-arch medium
"""

import os
import json
import time
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
from sb3_contrib import MaskablePPO

from parchis.training.common import make_env, evaluate_model, ProgressLoggingCallback
from parchis.training.cli import ARCHITECTURES
from parchis.evaluation import stats as eval_stats


# ──────────────────────────────────────────────────────────────────────
# Experiment grid definition
# ──────────────────────────────────────────────────────────────────────

REWARD_TYPES = ["progress_delta", "win_loss", "win_loss_shaped"]

# Fixed hyperparameters (identical across all 9 experiments)
FIXED_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}


def build_experiment_list(
    filter_reward: Optional[str] = None,
    filter_arch: Optional[str] = None,
) -> List[Dict]:
    """Build the list of experiment configurations, optionally filtered."""
    experiments = []
    for arch_name in ["small", "medium", "large"]:
        arch_cfg = ARCHITECTURES[arch_name]
        for reward_type in REWARD_TYPES:
            if filter_reward and reward_type != filter_reward:
                continue
            if filter_arch and arch_name != filter_arch:
                continue
            experiments.append({
                "name": f"{arch_name}_{reward_type}",
                "arch_name": arch_name,
                "net_arch": arch_cfg["net_arch"],
                "activation_fn": arch_cfg["activation_fn"],
                "reward_type": reward_type,
            })
    return experiments


# ──────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Results from a single experiment (one config, one seed) in the grid."""
    name: str
    arch_name: str
    net_arch: List[int]
    activation: str
    reward_type: str
    win_rate: float
    avg_player_progress: float
    avg_opponent_progress: float
    std_opponent_progress: float
    avg_episode_reward: float
    std_episode_reward: float
    total_eval_episodes: int
    training_time_seconds: float
    model_path: str
    total_timesteps: int
    seed: int


@dataclass
class AggregatedResult:
    """
    A single config's results aggregated across all --seeds.

    win_rate_ci is None when only one seed was run (mean_confidence_interval
    needs >= 2 samples) -- the default --seeds [42] preserves this script's
    original single-seed behavior exactly, so this is the common case, not
    an error state.
    """
    name: str
    arch_name: str
    net_arch: List[int]
    activation: str
    reward_type: str
    win_rate_mean: float
    win_rate_std: float
    win_rate_ci: Optional[Tuple[float, float]]
    avg_player_progress_mean: float
    avg_opponent_progress_mean: float
    avg_episode_reward_mean: float
    training_time_seconds_total: float
    per_seed_results: List[ExperimentResult]


def _aggregate_seed_results(config, per_seed_results):
    """Aggregate one config's per-seed ExperimentResults into an AggregatedResult."""
    win_rates = [r.win_rate for r in per_seed_results]
    win_rate_ci = (
        eval_stats.mean_confidence_interval(win_rates) if len(win_rates) >= 2 else None
    )
    return AggregatedResult(
        name=config["name"],
        arch_name=config["arch_name"],
        net_arch=config["net_arch"],
        activation=config["activation_fn"].__name__,
        reward_type=config["reward_type"],
        win_rate_mean=float(np.mean(win_rates)),
        win_rate_std=float(np.std(win_rates)),
        win_rate_ci=win_rate_ci,
        avg_player_progress_mean=float(np.mean([r.avg_player_progress for r in per_seed_results])),
        avg_opponent_progress_mean=float(np.mean([r.avg_opponent_progress for r in per_seed_results])),
        avg_episode_reward_mean=float(np.mean([r.avg_episode_reward for r in per_seed_results])),
        training_time_seconds_total=float(sum(r.training_time_seconds for r in per_seed_results)),
        per_seed_results=per_seed_results,
    )


# ──────────────────────────────────────────────────────────────────────
# Single experiment training
# ──────────────────────────────────────────────────────────────────────

def train_single_experiment(
    config: Dict,
    total_timesteps: int,
    num_players: int,
    base_save_path: str,
    base_log_path: str,
    seed: int,
    verbose: int,
) -> str:
    """Train a single (config, seed) experiment. Returns path to saved model."""
    # Always seed-suffixed: multiple seeds of the same config would
    # otherwise silently overwrite each other's checkpoint (docs/
    # RL_DESIGN_REVIEW.md Phase 4 multi-seed support).
    name = f"{config['name']}_seed{seed}"

    print(f"\n{'='*70}")
    print(f"  Experiment: {name}")
    print(f"  Architecture: {config['arch_name']} -> {config['net_arch']}")
    print(f"  Activation: {config['activation_fn'].__name__}")
    print(f"  Reward type: {config['reward_type']}")
    print(f"{'='*70}")

    env = make_env(num_players=num_players, reward_type=config["reward_type"], seed=seed)

    policy_kwargs = dict(
        net_arch=config["net_arch"],
        activation_fn=config["activation_fn"],
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=base_log_path,
        seed=seed,
        verbose=verbose,
        **FIXED_HYPERPARAMS,
    )

    callback = ProgressLoggingCallback(verbose=verbose)

    print(f"  Training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=name,
        progress_bar=True,
    )

    model_path = os.path.join(base_save_path, name)
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    env.close()
    return model_path


# ──────────────────────────────────────────────────────────────────────
# Results I/O and display
# ──────────────────────────────────────────────────────────────────────

def _save_results_json(results: List[AggregatedResult], save_path: str):
    """Save results to a JSON file (called after each config for crash recovery)."""
    results_file = os.path.join(save_path, "results.json")
    serializable = [asdict(r) for r in results]
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_experiments": len(results),
            "results": serializable,
        }, f, indent=2)


def _print_comparison_table(results: List[AggregatedResult], n_seeds: int):
    """Print a formatted comparison table of all experiment results, plus a
    CI-gated 'best config' verdict when n_seeds > 1 (docs/RL_DESIGN_REVIEW.md
    Phase 4) -- a raw point-estimate max() is indistinguishable from seed
    noise."""
    print("\n" + "=" * 90)
    print("EXPERIMENT GRID RESULTS")
    print("=" * 90)

    header = (
        f"{'Experiment':<28} "
        f"{'Win Rate':>16} "
        f"{'Progress':>9} "
        f"{'Opp Prog':>9} "
        f"{'Avg Reward':>11} "
        f"{'Time(s)':>8}"
    )
    print(header)
    print("-" * 90)

    for arch_name in ["small", "medium", "large"]:
        arch_results = [r for r in results if r.arch_name == arch_name]
        for r in arch_results:
            win_rate_str = (f"{r.win_rate_mean:.1%} ± {r.win_rate_std:.1%}"
                             if n_seeds > 1 else f"{r.win_rate_mean:.1%}")
            row = (
                f"{r.name:<28} "
                f"{win_rate_str:>16} "
                f"{r.avg_player_progress_mean:>9.4f} "
                f"{r.avg_opponent_progress_mean:>9.4f} "
                f"{r.avg_episode_reward_mean:>+11.4f} "
                f"{r.training_time_seconds_total:>8.0f}"
            )
            print(row)
        if arch_results:
            print("-" * 90)

    entries = [(r.name, r.win_rate_mean, r.win_rate_ci) for r in results]
    ranked, confirmed = eval_stats.rank_by_mean_with_ci(entries)
    if ranked:
        best_name = ranked[0][0]
        best = next(r for r in results if r.name == best_name)
        if n_seeds > 1:
            verdict = "statistically confirmed" if confirmed else "NOT statistically confirmed (CIs overlap)"
            print(f"\nBest by win rate: {best.name} ({verdict} across {n_seeds} seeds)")
        else:
            print(f"\nBest by win rate: {best.name} ({best.win_rate_mean:.1%}, "
                  f"single seed -- run with --seeds for a confirmed result)")
        print(f"  Model: {best.per_seed_results[0].model_path}")
    print("=" * 90)


# ──────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────

def run_experiment_grid(
    total_timesteps: int = 1_000_000,
    n_eval_episodes: int = 100,
    num_players: int = 2,
    base_save_path: str = "./models/experiment_grid",
    base_log_path: str = "./logs/experiment_grid",
    seeds: List[int] = [42],
    verbose: int = 1,
    filter_reward: Optional[str] = None,
    filter_arch: Optional[str] = None,
) -> List[AggregatedResult]:
    """Run the full 3x3 experiment grid (or a filtered subset), each config
    trained and evaluated once per seed in `seeds` and aggregated across
    seeds."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_save_path = os.path.join(base_save_path, timestamp)
    run_log_path = os.path.join(base_log_path, timestamp)
    os.makedirs(run_save_path, exist_ok=True)
    os.makedirs(run_log_path, exist_ok=True)

    experiments = build_experiment_list(filter_reward, filter_arch)

    print("=" * 70)
    print("EXPERIMENT GRID: Reward Structure x Network Architecture")
    print("=" * 70)
    print(f"  Experiments to run: {len(experiments)}")
    print(f"  Timesteps per experiment: {total_timesteps:,}")
    print(f"  Eval episodes: {n_eval_episodes}")
    print(f"  Players: {num_players}")
    print(f"  Seeds: {seeds}")
    print(f"  Save path: {run_save_path}")
    print(f"  Log path: {run_log_path}")
    print("=" * 70)
    for i, exp in enumerate(experiments):
        print(f"  [{i+1}] {exp['name']}")
    print("=" * 70)

    aggregated_results = []

    for exp_idx, config in enumerate(experiments):
        print(f"\n>>> Experiment {exp_idx + 1}/{len(experiments)}: {config['name']}")

        per_seed_results = []
        for seed in seeds:
            # Train
            start_time = time.time()
            model_path = train_single_experiment(
                config=config,
                total_timesteps=total_timesteps,
                num_players=num_players,
                base_save_path=run_save_path,
                base_log_path=run_log_path,
                seed=seed,
                verbose=verbose,
            )
            training_time = time.time() - start_time

            # Evaluate
            print(f"\n  Evaluating {config['name']} (seed={seed})...")
            model = MaskablePPO.load(model_path)
            eval_env = make_env(num_players=num_players, reward_type=config["reward_type"], seed=seed + 1000)
            stats = evaluate_model(model, eval_env, n_eval_episodes=n_eval_episodes,
                                    max_steps_per_episode=10000, verbose=0)
            eval_env.close()

            result = ExperimentResult(
                name=config["name"],
                arch_name=config["arch_name"],
                net_arch=config["net_arch"],
                activation=config["activation_fn"].__name__,
                reward_type=config["reward_type"],
                win_rate=stats["win_rate"],
                avg_player_progress=stats.get("mean_final_progress", 0.0),
                avg_opponent_progress=stats.get("mean_opponent_progress", 0.0),
                std_opponent_progress=stats.get("std_opponent_progress", 0.0),
                avg_episode_reward=stats["mean_reward"],
                std_episode_reward=stats["std_reward"],
                total_eval_episodes=stats["n_episodes"],
                training_time_seconds=training_time,
                model_path=model_path,
                total_timesteps=total_timesteps,
                seed=seed,
            )
            per_seed_results.append(result)

            print(f"\n  Results for {config['name']} (seed={seed}):")
            print(f"    Win rate: {result.win_rate:.1%}")
            print(f"    Avg progress: {result.avg_player_progress:.4f}")
            print(f"    Training time: {training_time:.0f}s")

        aggregated = _aggregate_seed_results(config, per_seed_results)
        aggregated_results.append(aggregated)

        if len(seeds) > 1:
            ci_str = (f"[{aggregated.win_rate_ci[0]:.1%}, {aggregated.win_rate_ci[1]:.1%}]"
                      if aggregated.win_rate_ci else "n/a")
            print(f"\n  Aggregated across {len(seeds)} seeds for {config['name']}: "
                  f"win_rate = {aggregated.win_rate_mean:.1%} ± {aggregated.win_rate_std:.1%} "
                  f"(95% CI {ci_str})")

        # Save intermediate results (crash recovery)
        _save_results_json(aggregated_results, run_save_path)

    # Final summary
    _print_comparison_table(aggregated_results, len(seeds))
    _save_results_json(aggregated_results, run_save_path)

    print(f"\nResults saved to: {os.path.join(run_save_path, 'results.json')}")
    print(f"TensorBoard: tensorboard --logdir {run_log_path}")

    return aggregated_results


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment grid: reward structures x network architectures"
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Training timesteps per experiment (default: 1,000,000)")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Evaluation episodes per experiment (default: 100)")
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4],
                        help="Number of players (default: 2)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds -- one full train+eval run per (config, seed) "
                             "pair (default: 42, reproducing the original single-seed "
                             "behavior). Pass multiple, e.g. --seeds 42 43 44, for a "
                             "statistically trustworthy comparison (docs/RL_DESIGN_REVIEW.md "
                             "Phase 4) at the cost of running the grid once per seed.")
    parser.add_argument("--save-path", type=str, default="./models/experiment_grid",
                        help="Base path to save models")
    parser.add_argument("--log-path", type=str, default="./logs/experiment_grid",
                        help="Base path for TensorBoard logs")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1],
                        help="Verbosity level (default: 1)")
    parser.add_argument("--filter-reward", type=str, default=None,
                        choices=REWARD_TYPES,
                        help="Only run experiments with this reward type")
    parser.add_argument("--filter-arch", type=str, default=None,
                        choices=list(ARCHITECTURES.keys()),
                        help="Only run experiments with this architecture")

    args = parser.parse_args()

    run_experiment_grid(
        total_timesteps=args.timesteps,
        n_eval_episodes=args.eval_episodes,
        num_players=args.players,
        base_save_path=args.save_path,
        base_log_path=args.log_path,
        seeds=args.seeds,
        verbose=args.verbose,
        filter_reward=args.filter_reward,
        filter_arch=args.filter_arch,
    )


if __name__ == "__main__":
    main()
