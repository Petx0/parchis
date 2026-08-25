#!/usr/bin/env python3
"""
Hyperparameter Random Search: sample random PPO hyperparameter combinations
and evaluate them against a fixed set of (architecture, reward_type) configs.

Stage 2 of the experiment_grid.py -> experiment_hyperparam_search.py pipeline:
experiment_grid.py found the best architecture x reward_type combos with
fixed hyperparameters; this script takes those winning configs and randomly
samples PPO hyperparameter combinations to see if tuning improves on the
fixed-hyperparameter baseline.

Usage:
    python -m parchis.training.experiment_hyperparam_search
    python -m parchis.training.experiment_hyperparam_search \\
        --configs small:win_loss large:progress_delta --n-samples 20

Configs may optionally pin opponent_weight/opponent_weighting (otherwise
defaulting to make_env's own 0.5/"mean", matching every config run before
this option existed): "arch:reward_type:opponent_weight:opponent_weighting",
e.g. --configs small:progress_delta:0.9:mean small:progress_delta:0.25:leader
"""

import os
import json
import time
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict

from sb3_contrib import MaskablePPO

from parchis.training.common import make_env, evaluate_model, ProgressLoggingCallback
from parchis.training.cli import ARCHITECTURES
from parchis.rl.env import ParchisEnv
from parchis.rl.rewards import VALID_OPPONENT_WEIGHTING_SCHEMES


# ──────────────────────────────────────────────────────────────────────
# Hyperparameter search space
# ──────────────────────────────────────────────────────────────────────

# Default, then two sensible alternatives per hyperparameter (see the
# experiment-grid plan for rationale). n_steps/batch_size candidates are
# powers of 2 so every pairing divides evenly -- SB3 minibatching never
# errors regardless of which two get sampled together.
HYPERPARAM_SPACE = {
    "learning_rate": [3e-4, 1e-4, 1e-3],
    "n_steps":       [2048, 512, 4096],
    "batch_size":    [64, 32, 256],
    "n_epochs":      [10, 4, 20],
    "gamma":         [0.995, 0.98, 0.999],
    "gae_lambda":    [0.95, 0.90, 0.99],
    "clip_range":    [0.2, 0.1, 0.3],
    "ent_coef":      [0.01, 0.0, 0.05],
}

DEFAULT_CONFIGS = ["small:win_loss", "large:progress_delta"]
DEFAULT_HP_SEARCH_SEED = 123
DEFAULT_N_SAMPLES = 20
# Matches make_env's own defaults -- a config that doesn't pin opponent_weight/
# opponent_weighting behaves exactly as every config did before this option existed.
DEFAULT_OPPONENT_WEIGHT = 0.5
DEFAULT_OPPONENT_WEIGHTING = "mean"


def sample_hyperparam_combos(n_samples: int, seed: int) -> List[Dict]:
    """Randomly sample n_samples unique hyperparameter combinations from
    HYPERPARAM_SPACE, independently choosing one value per hyperparameter
    per combination, deduplicated against exact repeats. Deterministic
    given the same seed."""
    rng = random.Random(seed)
    combos = []
    seen = set()
    while len(combos) < n_samples:
        combo = {k: rng.choice(v) for k, v in HYPERPARAM_SPACE.items()}
        key = tuple(combo[k] for k in HYPERPARAM_SPACE)
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)
    return combos


def parse_configs(config_strs: List[str]) -> List[Dict]:
    """Parse 'arch:reward_type' or
    'arch:reward_type:opponent_weight:opponent_weighting' strings into
    config dicts. The 2-token form defaults opponent_weight/opponent_weighting
    to DEFAULT_OPPONENT_WEIGHT/DEFAULT_OPPONENT_WEIGHTING (make_env's own
    defaults), reproducing every pre-existing config's exact behavior."""
    configs = []
    for s in config_strs:
        parts = s.split(":")
        if len(parts) == 2:
            arch_name, reward_type = parts
            opponent_weight = DEFAULT_OPPONENT_WEIGHT
            opponent_weighting = DEFAULT_OPPONENT_WEIGHTING
        elif len(parts) == 4:
            arch_name, reward_type, opponent_weight_str, opponent_weighting = parts
            opponent_weight = float(opponent_weight_str)
        else:
            raise ValueError(
                f"Config '{s}' must be 'arch:reward_type' or "
                f"'arch:reward_type:opponent_weight:opponent_weighting'"
            )
        if arch_name not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture '{arch_name}', must be one of {list(ARCHITECTURES.keys())}")
        if reward_type not in ParchisEnv.VALID_REWARD_TYPES:
            raise ValueError(f"Unknown reward_type '{reward_type}', must be one of {ParchisEnv.VALID_REWARD_TYPES}")
        if opponent_weighting not in VALID_OPPONENT_WEIGHTING_SCHEMES:
            raise ValueError(
                f"Unknown opponent_weighting '{opponent_weighting}', "
                f"must be one of {VALID_OPPONENT_WEIGHTING_SCHEMES}"
            )
        configs.append({
            "arch_name": arch_name,
            "reward_type": reward_type,
            "opponent_weight": opponent_weight,
            "opponent_weighting": opponent_weighting,
        })
    return configs


def config_label(config: Dict) -> str:
    """Human-readable, filename-safe config identifier. Only appends the
    opponent_weight/opponent_weighting suffix when it's non-default, so
    pre-existing 2-token configs (e.g. small:win_loss) keep their exact
    original names -- disambiguation only kicks in for configs that share
    (arch_name, reward_type) but pin different opponent_weight/weighting
    (e.g. two progress_delta configs at different alphas)."""
    label = f"{config['arch_name']}_{config['reward_type']}"
    if (config["opponent_weight"], config["opponent_weighting"]) != (
        DEFAULT_OPPONENT_WEIGHT, DEFAULT_OPPONENT_WEIGHTING
    ):
        # No raw '.' in the label -- SB3's save() (save_util.open_path_pathlib)
        # skips appending .zip whenever pathlib.Path(name).suffix is already
        # non-empty, and a decimal point (e.g. "mean0.9") makes it think the
        # name already has an extension, silently saving the checkpoint
        # without .zip. Use 'p' for the decimal point instead: "mean0.9" -> "mean0p9".
        weight_str = f"{config['opponent_weight']:g}".replace(".", "p")
        label += f"_{config['opponent_weighting']}{weight_str}"
    return label


# ──────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Results from a single (config, hyperparameter combo) experiment."""
    name: str
    arch_name: str
    reward_type: str
    opponent_weight: float
    opponent_weighting: str
    combo_index: int
    hyperparams: Dict
    win_rate: float
    avg_player_progress: float
    avg_opponent_progress: float
    avg_episode_reward: float
    std_episode_reward: float
    total_eval_episodes: int
    training_time_seconds: float
    model_path: str
    total_timesteps: int
    seed: int


def _save_results_json(results: List[ExperimentResult], save_path: str):
    """Save results to a JSON file (called after each experiment for crash recovery)."""
    results_file = os.path.join(save_path, "results.json")
    serializable = [asdict(r) for r in results]
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_experiments": len(results),
            "results": serializable,
        }, f, indent=2)


def _print_comparison_table(results: List[ExperimentResult]):
    print("\n" + "=" * 100)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 100)

    header = (
        f"{'Experiment':<38} "
        f"{'Win Rate':>9} "
        f"{'Progress':>9} "
        f"{'Avg Reward':>11} "
        f"{'Time(s)':>8}"
    )
    print(header)
    print("-" * 100)

    configs_seen = sorted(set(
        (r.arch_name, r.reward_type, r.opponent_weight, r.opponent_weighting) for r in results
    ))
    for arch_name, reward_type, opponent_weight, opponent_weighting in configs_seen:
        config_results = sorted(
            [r for r in results if r.arch_name == arch_name and r.reward_type == reward_type
             and r.opponent_weight == opponent_weight and r.opponent_weighting == opponent_weighting],
            key=lambda r: r.win_rate, reverse=True,
        )
        label = config_label({
            "arch_name": arch_name, "reward_type": reward_type,
            "opponent_weight": opponent_weight, "opponent_weighting": opponent_weighting,
        })
        for r in config_results:
            row = (
                f"{r.name:<38} "
                f"{r.win_rate:>8.1%} "
                f"{r.avg_player_progress:>9.4f} "
                f"{r.avg_episode_reward:>+11.4f} "
                f"{r.training_time_seconds:>8.0f}"
            )
            print(row)
        if config_results:
            best = config_results[0]
            print(f"  -> Best for {label}: combo{best.combo_index} "
                  f"({best.win_rate:.1%}), hyperparams={best.hyperparams}")
        print("-" * 100)

    overall_best = max(results, key=lambda r: r.win_rate)
    print(f"\nBest overall: {overall_best.name} ({overall_best.win_rate:.1%})")
    print(f"  Model: {overall_best.model_path}")
    print(f"  Hyperparameters: {overall_best.hyperparams}")
    print("=" * 100)


# ──────────────────────────────────────────────────────────────────────
# Single experiment training
# ──────────────────────────────────────────────────────────────────────

def train_single_experiment(
    config: Dict,
    combo_index: int,
    hyperparams: Dict,
    total_timesteps: int,
    num_players: int,
    base_save_path: str,
    base_log_path: str,
    seed: int,
    verbose: int,
) -> str:
    """Train a single (config, hyperparameter combo) experiment. Returns
    path to saved model."""
    name = f"{config_label(config)}_combo{combo_index}_seed{seed}"

    print(f"\n{'='*70}")
    print(f"  Experiment: {name}")
    print(f"  Architecture: {config['arch_name']} -> {ARCHITECTURES[config['arch_name']]['net_arch']}")
    print(f"  Reward type: {config['reward_type']}")
    print(f"  Opponent weight (alpha): {config['opponent_weight']}")
    print(f"  Opponent weighting: {config['opponent_weighting']}")
    print(f"  Hyperparameters: {hyperparams}")
    print(f"{'='*70}")

    env = make_env(num_players=num_players, reward_type=config["reward_type"],
                    opponent_weight=config["opponent_weight"],
                    opponent_weighting=config["opponent_weighting"], seed=seed)

    policy_kwargs = dict(
        net_arch=ARCHITECTURES[config["arch_name"]]["net_arch"],
        activation_fn=ARCHITECTURES[config["arch_name"]]["activation_fn"],
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=base_log_path,
        seed=seed,
        verbose=verbose,
        **hyperparams,
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
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────

def run_hyperparam_search(
    configs: List[Dict],
    n_samples: int = DEFAULT_N_SAMPLES,
    hp_search_seed: int = DEFAULT_HP_SEARCH_SEED,
    total_timesteps: int = 500_000,
    n_eval_episodes: int = 100,
    num_players: int = 2,
    base_save_path: str = "./models/experiment_hyperparam_search",
    base_log_path: str = "./logs/experiment_hyperparam_search",
    seed: int = 42,
    verbose: int = 1,
) -> List[ExperimentResult]:
    """Run a random hyperparameter search over `configs`, `n_samples`
    combinations per config, and evaluate each."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_save_path = os.path.join(base_save_path, timestamp)
    run_log_path = os.path.join(base_log_path, timestamp)
    os.makedirs(run_save_path, exist_ok=True)
    os.makedirs(run_log_path, exist_ok=True)

    combos = sample_hyperparam_combos(n_samples, hp_search_seed)
    total_experiments = len(configs) * len(combos)

    print("=" * 70)
    print("HYPERPARAMETER RANDOM SEARCH")
    print("=" * 70)
    print(f"  Configs: {[config_label(c) for c in configs]}")
    print(f"  Hyperparameter combos: {len(combos)} (seed={hp_search_seed})")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Timesteps per experiment: {total_timesteps:,}")
    print(f"  Eval episodes: {n_eval_episodes}")
    print(f"  Players: {num_players}")
    print(f"  Save path: {run_save_path}")
    print(f"  Log path: {run_log_path}")
    print("=" * 70)
    i = 0
    for config in configs:
        for combo_index in range(1, len(combos) + 1):
            i += 1
            print(f"  [{i}] {config_label(config)}_combo{combo_index}")
    print("=" * 70)

    results = []
    exp_num = 0
    for config in configs:
        for combo_index, hyperparams in enumerate(combos, start=1):
            exp_num += 1
            print(f"\n>>> Experiment {exp_num}/{total_experiments}: "
                  f"{config_label(config)}_combo{combo_index}")

            start_time = time.time()
            model_path = train_single_experiment(
                config=config,
                combo_index=combo_index,
                hyperparams=hyperparams,
                total_timesteps=total_timesteps,
                num_players=num_players,
                base_save_path=run_save_path,
                base_log_path=run_log_path,
                seed=seed,
                verbose=verbose,
            )
            training_time = time.time() - start_time

            print(f"\n  Evaluating {config_label(config)}_combo{combo_index}...")
            model = MaskablePPO.load(model_path)
            eval_env = make_env(num_players=num_players, reward_type=config["reward_type"],
                                 opponent_weight=config["opponent_weight"],
                                 opponent_weighting=config["opponent_weighting"], seed=seed + 1000)
            stats = evaluate_model(model, eval_env, n_eval_episodes=n_eval_episodes,
                                    max_steps_per_episode=10000, verbose=0)
            eval_env.close()

            result = ExperimentResult(
                name=f"{config_label(config)}_combo{combo_index}",
                arch_name=config["arch_name"],
                reward_type=config["reward_type"],
                opponent_weight=config["opponent_weight"],
                opponent_weighting=config["opponent_weighting"],
                combo_index=combo_index,
                hyperparams=hyperparams,
                win_rate=stats["win_rate"],
                avg_player_progress=stats.get("mean_final_progress", 0.0),
                avg_opponent_progress=stats.get("mean_opponent_progress", 0.0),
                avg_episode_reward=stats["mean_reward"],
                std_episode_reward=stats["std_reward"],
                total_eval_episodes=stats["n_episodes"],
                training_time_seconds=training_time,
                model_path=model_path,
                total_timesteps=total_timesteps,
                seed=seed,
            )
            results.append(result)

            print(f"\n  Results for {result.name}:")
            print(f"    Win rate: {result.win_rate:.1%}")
            print(f"    Avg progress: {result.avg_player_progress:.4f}")
            print(f"    Training time: {training_time:.0f}s")

            # Save intermediate results (crash recovery)
            _save_results_json(results, run_save_path)

    _print_comparison_table(results)
    _save_results_json(results, run_save_path)

    print(f"\nResults saved to: {os.path.join(run_save_path, 'results.json')}")
    print(f"TensorBoard: tensorboard --logdir {run_log_path}")

    return results


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Random search over PPO hyperparameters for fixed (architecture, reward_type) configs"
    )
    parser.add_argument("--configs", type=str, nargs="+", default=DEFAULT_CONFIGS,
                        help=f"'arch:reward_type' pairs to search over (default: {DEFAULT_CONFIGS})")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES,
                        help=f"Number of random hyperparameter combinations per config "
                             f"(default: {DEFAULT_N_SAMPLES})")
    parser.add_argument("--hp-search-seed", type=int, default=DEFAULT_HP_SEARCH_SEED,
                        help=f"Seed for sampling hyperparameter combinations "
                             f"(default: {DEFAULT_HP_SEARCH_SEED})")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Training timesteps per experiment (default: 500,000)")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Evaluation episodes per experiment (default: 100)")
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4],
                        help="Number of players (default: 2)")
    parser.add_argument("--save-path", type=str, default="./models/experiment_hyperparam_search",
                        help="Base path to save models")
    parser.add_argument("--log-path", type=str, default="./logs/experiment_hyperparam_search",
                        help="Base path for TensorBoard logs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed for every experiment (default: 42)")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1],
                        help="Verbosity level (default: 1)")

    args = parser.parse_args()
    configs = parse_configs(args.configs)

    run_hyperparam_search(
        configs=configs,
        n_samples=args.n_samples,
        hp_search_seed=args.hp_search_seed,
        total_timesteps=args.timesteps,
        n_eval_episodes=args.eval_episodes,
        num_players=args.players,
        base_save_path=args.save_path,
        base_log_path=args.log_path,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
