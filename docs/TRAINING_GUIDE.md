# Parchís RL Training Guide

Complete guide for training reinforcement learning agents to play Parchís.

All training scripts live in `parchis/training/` and share two modules so
their flags and behavior stay consistent: `common.py` (environment factory,
action masking, progress logging, evaluation loop) and `cli.py` (shared
argparse argument groups). See `docs/CODE_REVIEW.md` for why this split
exists.

## The Training Entry Points

| Script | Purpose |
|--------|---------|
| `train_ppo` | Main training script against random opponents (`--initial-model` to resume a saved checkpoint instead of starting fresh) |
| `train_selfplay` | Self-play with a periodically-updated opponent model |
| `experiment_alpha_comparison` | Sweep `opponent_weight` (α) values |
| `experiment_grid` | Sweep `reward_type` × network architecture (3×3 grid) |
| `experiment_hyperparam_search` | Broader hyperparameter grid search (multiple architectures/reward types/seeds in one sweep) |

## Quick Start

### 1. Quick Test Training (~1-2 minutes)

Perfect for testing and development:

```bash
python -m parchis.training.train_ppo --timesteps 10000 --players 4 --model-name parchis_quick_test
```

This trains for 10,000 timesteps, checkpoints every 10,000 steps, skips
mid-training evaluation, and saves the final model to
`./models/parchis_quick_test/`.

### 2. Full Training (1-2 hours)

For production training:

```bash
python -m parchis.training.train_ppo --timesteps 1000000
```

### 3. Custom Training

```bash
python -m parchis.training.train_ppo \
    --timesteps 2000000 \
    --players 4 \
    --opponent-weight 0.5 \
    --reward-type progress_delta \
    --lr 0.0003 \
    --batch-size 128 \
    --checkpoint-freq 100000 \
    --eval-freq 50000 \
    --model-name my_agent
```

### 4. Continue Training from a Checkpoint

```bash
python -m parchis.training.train_ppo \
    --initial-model ./models/parchis_quick_test/final_model \
    --timesteps 1000000
```

The loaded checkpoint's own architecture is used (`--arch` is ignored, with
a warning, when `--initial-model` is given), and the timestep counter picks
up where the checkpoint left off (so TensorBoard/checkpoint-frequency
counting stays continuous rather than restarting from 0).

### 5. Self-Play Training

```bash
python -m parchis.training.train_selfplay \
    --initial-model ./models/parchis_quick_test/final_model \
    --timesteps 2000000
```

The learning agent occupies a randomly-assigned seat each episode
(`ParchisEnv.agent_player_idx`); every other seat is controlled by an
opponent sampled from a **pool** of past checkpoints, refreshed every
`--opponent-update-freq` timesteps (default 50,000).

**Opponent pool** (`--pool-size`, default **5** — this is a default-behavior
change from earlier versions of this script, which always used a single
rolling snapshot; pass `--pool-size 1` to reproduce that exact steady-state
behavior). Every `--opponent-update-freq` timesteps, the current agent
weights are saved as a new checkpoint and added to the pool; once the pool
is full, the oldest live member is evicted (the checkpoint *file* itself is
never deleted — see "Automatic Checkpoints" below). One pool member is
sampled at the start of each episode (`--pool-sampling-strategy`) and used
for every opponent seat that whole episode:

| Strategy | Behavior | Cost |
|----------|----------|------|
| `uniform` (default) | Equal weight to every pool member | none |
| `recency` | Linearly biased toward newer checkpoints | none |
| `win_rate` | Biased toward whichever checkpoints the current model is currently weakest against (hard-example mining) | re-evaluates every pool member each update: `--pool-size × --pool-eval-episodes` extra episodes every `--opponent-update-freq` interval — a real, recurring wall-clock cost, only paid when explicitly selected |

Sampling a fresh pool of past opponents (instead of always the single latest
snapshot) is meant to reduce the risk of the agent overfitting to counter
only its own immediate past self and forgetting how to beat older strategic
styles. `metrics/opponent_pool_diversity` (normalized entropy of how evenly
episodes were spread across the pool since the last update; `1.0` = fully
even, `0.0` = every episode used the same member) and
`metrics/opponent_pool_size` make this visible during training — see
"Monitoring Training" below.

Because the self-play opponent pool keeps improving, `metrics/win_rate`
(measured against that moving target) conflates "is the agent improving"
with "is its opponent also improving." `train_selfplay` also logs
`metrics/win_rate_vs_baseline`, measured against a **fixed**, stationary
random-opponent environment held constant for the whole run (`--baseline-eval-freq`,
default same as `--opponent-update-freq`; `--baseline-eval-episodes`, default
20) — this is the metric to watch for whether the agent is genuinely getting
stronger over time.

## Training Parameters

### Environment (all scripts, via `cli.add_env_args`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--players` | 4 (2 for self-play/continue) | Number of players (2-4) |
| `--opponent-weight` | 0.5 | α for the `progress_delta`/`win_loss_shaped` reward's opponent-progress term |
| `--reward-type` | progress_delta | One of `progress_delta`, `win_loss`, `win_loss_shaped` — see `docs/REWARD_STRUCTURE.md` |

Both reward knobs are always available together — every script accepts
both `--opponent-weight` and `--reward-type`.

### PPO Hyperparameters (`train_ppo`, `train_selfplay`, via `cli.add_ppo_hyperparam_args`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timesteps` | 1,000,000 (2,000,000 for self-play) | Total training steps |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 64 | Batch size for training |
| `--n-steps` | 2048 | Steps per rollout |
| `--gamma` | 0.995 | Discount factor |
| `--ent-coef` | 0.01 | Entropy coefficient (exploration) |
| `--n-epochs` | 10 | Number of epochs per PPO update |
| `--gae-lambda` | 0.95 | GAE lambda parameter |
| `--clip-range` | 0.2 | PPO clipping parameter |

`--n-epochs`/`--gae-lambda`/`--clip-range` were previously hardcoded with no
CLI flag at all -- every experiment run through this codebase varied only
reward shaping (`--opponent-weight`/`--reward-type`) and network size
(`experiment_grid.py`), never PPO's own optimization hyperparameters.
`--vf-coef`/`--max-grad-norm`/`--target-kl` remain unexposed, left at
MaskablePPO's own defaults (see `docs/RL_DESIGN_REVIEW.md` for why).

### Opponent Pool (`train_selfplay` only)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pool-size` | 5 | Max past checkpoints kept live-sampled as opponents (default-behavior change from the old single-snapshot self-play — pass `1` for the old behavior) |
| `--pool-sampling-strategy` | uniform | One of `uniform`, `recency`, `win_rate` — see "Self-Play Training" above |
| `--pool-eval-episodes` | 10 | Episodes per pool member when scoring for `--pool-sampling-strategy win_rate` (unused otherwise) |

### Checkpointing and Evaluation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint-freq` | 50,000 (100,000 for continue/self-play) | Save checkpoint every N steps |
| `--eval-freq` | None (disabled) | Mid-training evaluation frequency, `train_ppo` only — see "Why mid-training eval is disabled by default" below |
| `--n-eval-episodes` | 10 (100 for continue/self-play) | Episodes for final evaluation |

### Paths and Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--save-path` | ./models | Where to save models |
| `--log-path` | ./logs | TensorBoard log directory |
| `--model-name` | auto | Model name (timestamp-based if not provided) |
| `--seed` | 42 | Random seed |

## Monitoring Training

### TensorBoard

Monitor training progress in real-time:

```bash
tensorboard --logdir ./logs
```

Then open your browser to `http://localhost:6006`

**Key Metrics to Watch:**

1. **rollout/ep_rew_mean**: Average episode reward
   - Should increase over time

2. **metrics/win_rate**, **metrics/final_progress**, **metrics/pieces_finished**, **metrics/pieces_out_of_base**
   - Logged every episode by `ProgressLoggingCallback` (`parchis/training/common.py`), as a rolling mean over the last 100 episodes
   - `final_progress` is the primary metric to watch — it moves smoothly even when win/loss is noisy
   - Every training script attaches this callback (previously `train_selfplay` and the now-folded-in checkpoint-resume path didn't, so self-play/continued runs had no progress curve at all)

3. **metrics/win_rate_vs_baseline**, **metrics/final_progress_vs_baseline** (`train_selfplay` only)
   - Logged periodically by `FixedOpponentEvalCallback` against a fixed random-opponent baseline held constant for the whole run — see "Self-Play Training" above for why this is a different (and more trustworthy) signal than `metrics/win_rate` during self-play

4. **metrics/opponent_pool_diversity**, **metrics/opponent_pool_size** (`train_selfplay` only)
   - Logged by `SelfPlayCallback` on every opponent-pool update (except the very first, before any sampling has happened yet)
   - `opponent_pool_diversity` is the normalized entropy of episode counts per pool member since the last update — `1.0` means episodes were spread evenly across the pool, `0.0` means every episode used the same member (a low value here is the "cyclic strategy" risk this pool exists to mitigate, made visible)
   - `opponent_pool_size` tracks the live pool growing up to `--pool-size` and staying there

5. **train/entropy_loss**: Exploration vs exploitation
   - Should decrease gradually; if it drops too fast, increase `--ent-coef`

6. **train/policy_loss** / **train/value_loss**: Should decrease and stabilize; large spikes suggest instability

## Checkpoints and Saving

### Automatic Checkpoints

Models are automatically saved every `--checkpoint-freq` steps:

```
models/
  └── parchis_ppo_4p_20260115_143022/
      ├── checkpoint_50000_steps.zip
      ├── checkpoint_100000_steps.zip
      ├── best_model.zip              # Only produced if --eval-freq was set
      └── final_model.zip              # Final model after training
```

`best_model.zip` is only written when `--eval-freq` is explicitly set — by
default mid-training evaluation is disabled (see below), so most runs only
produce checkpoints + `final_model.zip`.

`train_selfplay` doesn't use this `CheckpointCallback`-based scheme at all.
Instead, every `--opponent-update-freq` timesteps it writes an opponent-pool
checkpoint, `opponent_checkpoint_{update_number}_{timesteps}steps.zip`,
alongside `final_model.zip`/`interrupted_model.zip`. **These files are never
deleted**, even once evicted from the live `--pool-size`-capped pool — only
what's sampled *during training* is capped; the full checkpoint history
stays on disk (useful for later cross-checkpoint comparisons). Over a long
run this means many small model files accumulate under
`models/<model_name>/` — clean them up manually if disk space is a concern.

### Loading a Checkpoint

```python
from sb3_contrib import MaskablePPO
from parchis.training.common import make_env, evaluate_model

model = MaskablePPO.load("./models/parchis_ppo_4p_20260115_143022/final_model")

# Continue training
model.learn(total_timesteps=1000000)

# Or just evaluate
env = make_env(num_players=4)
evaluate_model(model, env, n_eval_episodes=20)
```

(Prefer `python -m parchis.training.train_ppo --initial-model ...` over
calling `.learn()` directly — it also handles checkpointing and TensorBoard
logging for you.)

## Evaluation

### Evaluate a Trained Model

```bash
python -m parchis.training.train_ppo \
    --evaluate ./models/parchis_ppo_4p_20260115_143022/final_model \
    --n-eval-episodes 20
```

Or use the standalone evaluation script, which also supports evaluating
against another trained model instead of random opponents:

```bash
python -m parchis.evaluation.evaluate \
    --model ./models/my_model/final_model \
    --opponent ./models/opponent_model/final_model \
    --n-games 100
```

Output:
```
Episode 1/20: Reward = 1.07, Length = 88, Status = WIN
Episode 2/20: Reward = -0.97, Length = 73, Status = LOSS
...
Evaluation Results:
  Episodes: 20
  Mean reward: 0.12 +/- 1.01
  Mean length: 83.1
  Wins: 9/20 (45.0%)
  Win rate 95% CI: [24.9%, 66.8%]
  Losses: 11/20 (55.0%)
  Mean final progress: 0.88 (0.0 to 1.0)
  Mean pieces finished: 2.9 (0 to 4)
  Mean pieces out of base: 3.9 (0 to 4)
  Mean opponent progress: 0.73 +/- 0.12
  Win rate by seat: seat 0: 40.0% (n=5), seat 1: 50.0% (n=6), ...
  Win rate by color: BLUE: 42.9% (n=7), GREEN: 46.2% (n=13), ...
  Capture rate: 3.80/game (against agent: 3.55/game)
  Legal moves per decision: 2.31 +/- 1.18
  Bonus chain length: 1.04 +/- 0.19
  Three-sixes penalty rate: 1.10/game
```

### Interpreting Results

**Win Rate** (against `num_players - 1` random opponents):
- `1 / num_players` = random baseline (25% for 4 players, 50% for 2 players)
- Meaningfully above baseline = agent is learning
- 70%+ (2-player) = strong agent
- **Win rate 95% CI**: a Wilson score interval, not the point estimate alone — with only 20 games, a 45% win rate could plausibly be anywhere from ~25% to ~67%; don't read small differences between two runs as meaningful unless their CIs don't overlap.
- **Win rate by seat / by color**: a fairness check on the per-episode random seat assignment (`ParchisEnv.agent_player_idx`) — if one seat or color is consistently over/under-represented in wins relative to the others (beyond what small-sample noise would explain), that's worth investigating, not expected behavior.

**Mean Reward:** depends on `reward_type` — see `docs/REWARD_STRUCTURE.md` for what each one's scale means; they are not comparable to each other.

**Game KPIs** (capture rate, legal moves per decision, bonus chain length, three-sixes penalty rate): descriptive statistics about how the games themselves played out, not directly about agent skill — useful for sanity-checking that games look like real Parchís (e.g. a near-zero three-sixes penalty rate across many games would be suspicious) and for comparing how differently-tuned agents' games differ structurally.

### Comparing Checkpoints Directly (Elo Ladder)

Self-play's win rate is measured against a moving target (the opponent pool itself keeps improving), which conflates "is the agent improving" with "is its opponent also improving." To answer "is checkpoint N+1 actually stronger" directly, round-robin a set of saved checkpoints against each other and a random baseline:

```bash
python -m parchis.evaluation.elo_ladder \
    --checkpoints ./models/my_run/checkpoint_100000_steps ./models/my_run/checkpoint_500000_steps \
    --games-per-pairing 40
```

2-player matches only (`ParchisSelfPlayEnv` puts the same opponent model in every seat for 3-4 players, so a multiplayer match isn't a clean pairwise comparison). Prints a ratings table and saves per-pairing results (win rate + Wilson CI) to `results.json`.

If you're comparing two *groups* of checkpoints (e.g. 3 seeds trained one way vs. 3 seeds trained another) rather than individual models, pool the cross-group pairings from that `results.json` into one win rate:

```bash
python -m parchis.evaluation.group_comparison \
    --results-json ./logs/elo_ladder/<timestamp>/results.json \
    --group-a baseline_42 baseline_43 baseline_44 \
    --group-b redesigned_42 redesigned_43 redesigned_44
```

See `docs/RL_DESIGN_REVIEW.md`'s Phase 5 section for the design context this
was built for (baseline-vs-redesigned reward/α validation) — real training
runs are involved (~15-25 hours total), so run each stage (`experiment_grid`/
`experiment_alpha_comparison` screening, then `train_ppo`/`train_selfplay`,
then `elo_ladder`/`group_comparison`) as a deliberate, staged process, not a
single command. (The one-shot `scripts/run_phase5.sh` runbook that
originally encoded this sequence has since been retired — it was never run
end-to-end as written; run each stage's script directly instead.)

## Training Strategies

### 1. Self-Play Instead of Random Opponents

Random opponents are a weak, non-adaptive baseline. `train_selfplay`
(above) trains against a pool of periodically-refreshed past copies of the
agent's own policy, which tends to produce stronger final agents:

```bash
python -m parchis.training.train_selfplay \
    --initial-model ./models/parchis_quick_test/final_model \
    --timesteps 2000000 \
    --opponent-update-freq 50000
```

### 2. Hyperparameter Tuning

**For faster learning:**
```bash
python -m parchis.training.train_ppo \
    --lr 0.001 \
    --ent-coef 0.02 \
    --n-steps 4096
```

**For more stable learning:**
```bash
python -m parchis.training.train_ppo \
    --lr 0.0001 \
    --ent-coef 0.005 \
    --batch-size 128
```

**Broader grid search** (multiple architectures × reward types × seeds in
one sweep, rather than hand-picking one combination at a time):
```bash
python -m parchis.training.experiment_hyperparam_search --players 2
```
Writes a `results.json` per run (win rate, mean reward per combination) so
the winning configuration can be identified without re-reading every
checkpoint's own training log.

### 3. Comparing Reward Structures

Rather than hand-editing the reward function, use `--reward-type` (see
`docs/REWARD_STRUCTURE.md` for what each does) or run the grid sweep:

```bash
python -m parchis.training.experiment_grid --filter-reward win_loss
```

### 4. Comparing Opponent-Weight (α) Values

```bash
python -m parchis.training.experiment_alpha_comparison --alphas 0.0 0.5 0.9
```

### Multi-Seed Sweeps

Both `experiment_grid` and `experiment_alpha_comparison` default to a
single seed (`--seeds 42`) — a raw win-rate comparison between two configs
at one seed each is indistinguishable from seed noise. Pass multiple seeds
for a statistically grounded comparison:

```bash
python -m parchis.training.experiment_grid --filter-reward win_loss --seeds 42 43 44
```

Each config is trained and evaluated once per seed; results report
mean ± std and a 95% CI (`docs/RL_DESIGN_REVIEW.md` Phase 4), and the
"best config" verdict only claims a winner outright when its CI doesn't
overlap the runner-up's — otherwise it says so explicitly rather than
picking one by a possibly-noisy point estimate. This multiplies training
time by the number of seeds, so it's opt-in, not the default.

## Troubleshooting

### Agent Not Learning

**Symptoms:** `metrics/final_progress` stays flat near 0

**Solutions:**
1. Increase entropy coefficient: `--ent-coef 0.05`
2. Lower learning rate: `--lr 0.0001`
3. Train longer: `--timesteps 2000000`
4. Check action masking is working correctly (`info['action_masks']`)

### Training Too Slow

**Symptoms:** Low FPS (< 500)

**Solutions:**
1. Reduce `n_steps`: `--n-steps 1024`
2. Reduce `batch_size`: `--batch-size 32`
3. Use fewer evaluation episodes: `--n-eval-episodes 5`
4. Increase checkpoint frequency: `--checkpoint-freq 100000`

### Unstable Training

**Symptoms:** Reward oscillates wildly, crashes

**Solutions:**
1. Lower learning rate: `--lr 0.0001`
2. Increase batch size: `--batch-size 128`
3. Try a smaller `--clip-range` (e.g. `0.1`) for more conservative policy updates

### Out of Memory

**Solutions:**
1. Reduce batch size: `--batch-size 32`
2. Reduce n_steps: `--n-steps 1024`
3. Use CPU instead of GPU (PPO is often faster on CPU anyway for this environment's small network)

### Why mid-training evaluation is disabled by default

`--eval-freq` defaults to `None` (disabled) because stable-baselines3's
`EvalCallback` had compatibility issues with `MaskablePPO` that could hang
training indefinitely. See `docs/EVALUATION_FIX.md` for the history and
the safety-timeout fix (`max_steps_per_episode`) that makes evaluation safe
to run — final evaluation at the end of training always uses it.

## Network Architecture

`train_ppo.py` and `train_selfplay.py` both expose `--arch {small,medium,large}`
(default `small`, matching SB3's own unconfigured default — every run before this
flag existed was implicitly `small`), sourced from the single shared
`parchis.training.cli.ARCHITECTURES` dict:

| Preset | `net_arch` | Activation |
|---|---|---|
| `small` (default) | `[64, 64]` | Tanh |
| `medium` | `[256, 256]` | ReLU |
| `large` | `[512, 256, 128]` | ReLU |

```bash
python -m parchis.training.train_ppo --arch medium --timesteps 1000000
```

`--arch` only affects a **freshly constructed** model. On `train_selfplay.py`, it's
silently ignored whenever `--initial-model` is also given — the loaded checkpoint's
own saved architecture is used instead (SB3 restores it automatically from the
`.zip`), and the script prints a warning if you pass both together with a non-default
`--arch`.

`experiment_grid.py` sweeps all three architectures × all three reward types (9 runs);
use `--filter-arch {small,medium,large}` to run just one architecture across the
reward-type sweep (mirrors `--filter-reward`, e.g.
`experiment_grid.py --filter-reward win_loss --filter-arch medium`).

To try an architecture outside the three presets (not exposed on any CLI):

```python
from sb3_contrib import MaskablePPO
import torch.nn as nn
from parchis.training.common import make_env

env = make_env(num_players=4)
policy_kwargs = dict(
    net_arch=[256, 256, 128],
    activation_fn=nn.ReLU,
)
model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1_000_000)
```

## Next Steps

1. **Start small**: Run `python -m parchis.training.train_ppo --timesteps 10000 --players 4`
2. **Monitor**: Watch TensorBoard to understand learning
3. **Evaluate**: Test your model with `--evaluate` or `parchis.evaluation.evaluate`
4. **Iterate**: Adjust hyperparameters or `--reward-type`/`--opponent-weight` based on results
5. **Scale up**: Run `train_selfplay` for a stronger curriculum than random opponents
6. **Compare systematically**: Use `experiment_alpha_comparison`/`experiment_grid` instead of one-off manual runs
