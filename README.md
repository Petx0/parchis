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
│   │   ├── train_ppo.py        # Main PPO training (--initial-model to resume a checkpoint)
│   │   ├── train_selfplay.py   # Self-play training (opponent pool)
│   │   ├── experiment_alpha_comparison.py  # Sweep opponent_weight (α), multi-seed
│   │   ├── experiment_grid.py  # Sweep reward_type x network architecture, multi-seed
│   │   └── experiment_hyperparam_search.py # Broader hyperparameter grid search
│   │
│   ├── search/                  # AlphaZero-style MCTS search on top of a trained checkpoint
│   │   ├── mcts.py             # PUCT search engine (depth-limited, chance-node aware)
│   │   ├── agents.py           # Player.choose_move-compatible search/plain agent factories
│   │   ├── network_eval.py     # MaskablePPO-backed evaluate_fn (priors + value)
│   │   ├── heuristic_eval.py   # Uniform-prior/progress-heuristic evaluate_fn (no trained model needed)
│   │   ├── state_view.py       # Reuses ParchisEnv._get_observation() for simulated states
│   │   ├── isolated_random.py  # Save/restore global RNG state around chance-node sampling
│   │   └── benchmark_mcts.py   # Deepcopy/search cost benchmark
│   │
│   ├── evaluation/              # Evaluation, statistics, and cross-checkpoint comparison
│   │   ├── evaluate.py         # Agent evaluation script (vs. random or another model)
│   │   ├── stats.py            # Wilson/t-distribution confidence intervals
│   │   ├── elo.py              # Elo rating math
│   │   ├── elo_ladder.py       # Round-robin checkpoint Elo ladder (2-player)
│   │   ├── multiplayer_matrix.py # Win-rate matrix for 3-4 player checkpoints (Elo doesn't apply)
│   │   ├── arena.py            # Game-level match harness for search-capable agents
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
├── models/                      # Trained models
├── logs/                        # TensorBoard logs
├── docs/                        # Documentation
│   ├── RULES.md                # Game rules
│   ├── TRAINING_GUIDE.md       # Training guide
│   ├── README_ENVIRONMENT.md   # Environment/observation-space details
│   ├── REWARD_STRUCTURE.md     # Reward formulas, narrative description
│   ├── RL_DESIGN_REVIEW.md     # RL design initiative: phases, decisions, rationale
│   ├── SEARCH_MCTS.md          # MCTS search: what's live, what was tried and archived
│   ├── CODE_REVIEW.md          # Prior code-correctness review pass
│   ├── EVALUATION_FIX.md       # Why mid-training eval is disabled by default
│   ├── VISUALIZATION.md        # Visualization guide
│   └── images/                 # Images and screenshots
└── requirements.txt            # Python dependencies
```

## Quick Start

### Training

```bash
# Quick test (10K timesteps, ~1-2 minutes)
python -m parchis.training.train_ppo --timesteps 10000 --players 4

# Full training (1M timesteps, ~1-2 hours)
python -m parchis.training.train_ppo --timesteps 1000000

# Resume a previous checkpoint instead of starting from scratch
python -m parchis.training.train_ppo --initial-model ./models/my_model/final_model --timesteps 500000

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
# (2-player only -- Elo has no valid interpretation at 3-4 players)
python -m parchis.evaluation.elo_ladder \
    --checkpoints ./models/my_run/checkpoint_100000_steps ./models/my_run/checkpoint_500000_steps

# 3-4 player checkpoints: use the win-rate matrix instead of Elo
python -m parchis.evaluation.multiplayer_matrix \
    --checkpoints ./models/my_run_4p/checkpoint_a ./models/my_run_4p/checkpoint_b
```

### Search-augmented play (MCTS)

Any trained MaskablePPO checkpoint can be played with Monte Carlo Tree
Search on top for stronger inference-time decisions, no retraining needed
-- see `docs/SEARCH_MCTS.md` for what this is and the confirmed win-rate
result. `parchis/search/agents.py::make_mcts_ppo_agent_factory` plugs a
checkpoint + search into `parchis/evaluation/arena.py` for a head-to-head
comparison against the same checkpoint's plain (unsearched) inference.

## Documentation

- `docs/RULES.md` — Parchís rules this implementation follows
- `docs/TRAINING_GUIDE.md` — full training/evaluation parameter reference
- `docs/README_ENVIRONMENT.md` — observation/action space details
- `docs/REWARD_STRUCTURE.md` — reward formulas
- `docs/RL_DESIGN_REVIEW.md` — the RL design initiative: what was wrong, what changed and why, phase by phase
- `docs/SEARCH_MCTS.md` — MCTS search: the confirmed-useful engine that's kept live, and what was tried and archived

## License

MIT License
