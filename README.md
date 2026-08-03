# Parchís Reinforcement Learning Project

A complete implementation of the Parchís board game with reinforcement learning agents trained using PPO (Proximal Policy Optimization).

## Project Structure

```
Reinforcement/
├── parchis/
│   ├── game/                    # Game simulation (core game logic)
│   │   ├── game.py             # Main game controller
│   │   ├── board.py            # Board representation
│   │   ├── rules.py            # Move-legality rule engine
│   │   ├── player.py           # Player logic
│   │   ├── piece.py            # Piece state and movement
│   │   ├── dice.py             # Dice rolling
│   │   ├── records.py          # MoveInfo/RollEntry/TurnInfo data records
│   │   ├── constants.py        # Dice/turn rule constants
│   │   └── formatting.py       # Human-readable rendering
│   │
│   ├── rl/                      # Reinforcement Learning environment
│   │   ├── env.py              # Base Gymnasium environment (ParchisEnv)
│   │   ├── env_selfplay.py     # Self-play environment wrapper
│   │   ├── rewards.py          # Reward formulas (progress_delta/win_loss/win_loss_shaped)
│   │   └── opponent_pool.py    # Self-play opponent-pool sampling
│   │
│   ├── training/                # Training scripts
│   │   ├── common.py           # Shared env factory, callbacks, evaluation loop
│   │   ├── cli.py               # Shared CLI argument groups
│   │   ├── train_ppo.py        # Main PPO training
│   │   ├── train_quick.py      # Quick testing (10K timesteps)
│   │   ├── train_continue.py   # Continue from checkpoint
│   │   ├── train_selfplay.py   # Self-play training (opponent pool)
│   │   ├── experiment_alpha_comparison.py  # Sweep opponent_weight (α), multi-seed
│   │   └── experiment_grid.py  # Sweep reward_type x network architecture, multi-seed
│   │
│   ├── evaluation/              # Evaluation, statistics, and cross-checkpoint comparison
│   │   ├── evaluate.py         # Agent evaluation script (vs. random or another model)
│   │   ├── stats.py            # Wilson/t-distribution confidence intervals
│   │   ├── elo.py              # Elo rating math
│   │   ├── elo_ladder.py       # Round-robin checkpoint Elo ladder
│   │   └── group_comparison.py # Pool a group of checkpoints vs. another group
│   │
│   ├── visualization/           # GUI and rendering
│   │   ├── visualizer.py       # Game visualizer
│   │   ├── demo_visualization.py
│   │   └── visualize_game.py
│   │
│   ├── tests/                   # pytest test suite (one file per module, roughly)
│   │
│   └── utils/                   # Utility modules
│       └── logger.py           # Game logging utility (non-RL simulator only)
│
├── scripts/
│   └── run_phase5.sh            # Baseline-vs-redesigned validation runbook (see docs/RL_DESIGN_REVIEW.md)
│
├── models/                      # Trained models
├── logs/                        # TensorBoard logs
├── docs/                        # Documentation
│   ├── RULES.md                # Game rules
│   ├── TRAINING_GUIDE.md       # Training guide
│   ├── README_ENVIRONMENT.md   # Environment/observation-space details
│   ├── REWARD_STRUCTURE.md     # Reward formulas, narrative description
│   ├── RL_DESIGN_REVIEW.md     # RL design initiative: phases, decisions, rationale
│   ├── CODE_REVIEW.md          # Prior code-correctness review pass
│   ├── EVALUATION_FIX.md       # Why mid-training eval is disabled by default
│   ├── VISUALIZATION.md        # Visualization guide
│   ├── archive/                # Historical changelogs
│   └── images/                 # Images and screenshots
└── requirements.txt            # Python dependencies
```

## Quick Start

### Training

```bash
# Quick test (10K timesteps, ~1-2 minutes)
python -m parchis.training.train_quick

# Full training (1M timesteps, ~1-2 hours)
python -m parchis.training.train_ppo --timesteps 1000000

# Self-play training (recommended for best results; opponent pool of past
# checkpoints, not just the single most recent one)
python -m parchis.training.train_selfplay --timesteps 2000000
```

See `docs/TRAINING_GUIDE.md` for the full parameter reference (reward types, opponent-weight α, self-play opponent-pool settings, multi-seed sweeps).

### Evaluation

```bash
# Evaluate against random opponents (reports win rate + 95% CI, per-seat/
# color fairness breakdown, and capture/bonus-chain/three-sixes KPIs)
python -m parchis.evaluation.evaluate \
    --model ./models/my_model/final_model \
    --n-games 100

# Round-robin a set of saved checkpoints (+ a random baseline) into an Elo
# ranking, so "is checkpoint N+1 actually stronger" is answerable directly
python -m parchis.evaluation.elo_ladder \
    --checkpoints ./models/my_run/checkpoint_100000_steps ./models/my_run/checkpoint_500000_steps
```

## Documentation

- `docs/RULES.md` — Parchís rules this implementation follows
- `docs/TRAINING_GUIDE.md` — full training/evaluation parameter reference
- `docs/README_ENVIRONMENT.md` — observation/action space details
- `docs/REWARD_STRUCTURE.md` — reward formulas
- `docs/RL_DESIGN_REVIEW.md` — the RL design initiative: what was wrong, what changed and why, phase by phase

## License

MIT License
