"""
Parchís Training Scripts Module.

Shared building blocks: common.py (environment factory, action masking,
progress logging, evaluation loop) and cli.py (shared argparse argument
groups). Every script below imports from these instead of reimplementing
them.

Entry-point scripts:
- train_ppo: Main training script with full configuration (pass
  --initial-model to resume a previous checkpoint instead of training
  from scratch)
- train_selfplay: Self-play training with rolling opponent updates
- experiment_alpha_comparison: Sweep opponent_weight (α) values
- experiment_grid: Sweep reward_type x network architecture
- experiment_hyperparam_search: Broader hyperparameter grid search
"""

__all__ = []
