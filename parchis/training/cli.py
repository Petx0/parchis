"""
Shared argparse argument groups for training scripts.

Common flags (--players, --opponent-weight, --reward-type, PPO
hyperparameters, --checkpoint-freq, --seed, --verbose, save/log paths...)
were previously redeclared per script, which let defaults silently drift
(e.g. train_ppo.py's train() had gamma=0.995 as a function default but
0.99 as its own CLI default). Declaring each group once here means there
is exactly one default to change.

Each training script calls only the groups it needs.
"""

import torch.nn as nn

from parchis.rl.env import ParchisEnv

ARCHITECTURES = {
    "small":  {"net_arch": [64, 64],        "activation_fn": nn.Tanh},
    "medium": {"net_arch": [256, 256],      "activation_fn": nn.ReLU},
    "large":  {"net_arch": [512, 256, 128], "activation_fn": nn.ReLU},
}


def add_env_args(parser, default_players=4):
    """--players, --opponent-weight, --reward-type."""
    parser.add_argument("--players", type=int, default=default_players, choices=[2, 3, 4],
                         help=f"Number of players (default: {default_players})")
    parser.add_argument("--opponent-weight", type=float, default=0.5,
                         help="Opponent weight alpha for the progress_delta/win_loss_shaped "
                              "reward's opponent-progress term (default: 0.5)")
    parser.add_argument("--reward-type", type=str, default="progress_delta",
                         choices=list(ParchisEnv.VALID_REWARD_TYPES),
                         help="Reward structure (default: progress_delta)")


def add_ppo_hyperparam_args(parser, default_gamma=0.995):
    """PPO hyperparameters shared by every script that constructs a fresh model.

    n_epochs/gae_lambda/clip_range were previously hardcoded identically in
    every script (10, 0.95, 0.2) with no way to tune or sweep them -- every
    experiment run through this codebase varied only reward shaping and
    network size, never PPO's own optimization hyperparameters. vf_coef/
    max_grad_norm/target_kl are deliberately left at MaskablePPO's own
    defaults rather than exposed here: they're less likely to be the
    binding constraint for this problem size, and there's no evidence yet
    that they need tuning (see docs/RL_DESIGN_REVIEW.md).
    """
    parser.add_argument("--lr", type=float, default=3e-4,
                         help="Learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=64,
                         help="Batch size (default: 64)")
    parser.add_argument("--n-steps", type=int, default=2048,
                         help="Steps per rollout (default: 2048)")
    parser.add_argument("--gamma", type=float, default=default_gamma,
                         help=f"Discount factor (default: {default_gamma})")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                         help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--n-epochs", type=int, default=10,
                         help="Number of epochs per PPO update (default: 10)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                         help="GAE lambda parameter (default: 0.95)")
    parser.add_argument("--clip-range", type=float, default=0.2,
                         help="PPO clipping parameter (default: 0.2)")


def add_network_args(parser, default_arch="small"):
    """--arch: preset (net_arch, activation_fn) for a freshly constructed
    policy network. 'small' ([64,64] Tanh) matches SB3's own default --
    every training run so far has used it implicitly. Ignored by
    train_selfplay.py when --initial-model is supplied: SB3 restores the
    loaded checkpoint's actual saved architecture instead (see that
    script's own warning for this case).
    """
    parser.add_argument("--arch", type=str, default=default_arch,
                         choices=list(ARCHITECTURES.keys()),
                         help=f"Network architecture preset: small ([64,64] Tanh), "
                              f"medium ([256,256] ReLU), large ([512,256,128] ReLU) "
                              f"(default: {default_arch})")


def add_checkpoint_eval_args(parser, default_checkpoint_freq=50_000, default_n_eval_episodes=10):
    """--checkpoint-freq, --n-eval-episodes."""
    parser.add_argument("--checkpoint-freq", type=int, default=default_checkpoint_freq,
                         help=f"Checkpoint frequency (default: {default_checkpoint_freq:,})")
    parser.add_argument("--n-eval-episodes", type=int, default=default_n_eval_episodes,
                         help=f"Number of evaluation episodes (default: {default_n_eval_episodes})")


def add_io_args(parser, default_save_path="./models", default_log_path="./logs"):
    """
    --save-path, --log-path, --model-name. Pass default_save_path=None for
    scripts that auto-generate a per-model-name save directory instead of
    using a fixed default.
    """
    save_path_help = (f"Path to save models (default: {default_save_path})" if default_save_path
                       else "Path to save the continued model (default: auto-generated from the model name)")
    parser.add_argument("--save-path", type=str, default=default_save_path, help=save_path_help)
    parser.add_argument("--log-path", type=str, default=default_log_path,
                         help=f"Path for TensorBoard logs (default: {default_log_path})")
    parser.add_argument("--model-name", type=str, default=None,
                         help="Model name (auto-generated if not provided)")


def add_common_args(parser):
    """--seed, --verbose -- present in every training script."""
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed (default: 42)")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                         help="Verbosity level (default: 1)")
