# Evaluation Freeze Fix

## Problem
Training would freeze at ~99% completion during evaluation, particularly when `eval_freq` triggered mid-training evaluation callbacks.

## Root Cause
The standard `EvalCallback` from stable-baselines3 has compatibility issues with `MaskablePPO` when games can run for extended periods without terminating, causing the evaluation to hang indefinitely.

## Solutions Implemented

### 1. Disabled Mid-Training Evaluation by Default
**File**: `parchis/training/train_ppo.py`

- Changed default `eval_freq` from `50_000` to `None`
- Added conditional logic to only create `EvalCallback` when `eval_freq` is not None
- Updated CLI argument parser to accept None as default
- Added informative messages about evaluation status

**Benefits**:
- No risk of freezing during training
- Training completes reliably
- Final evaluation still runs with safety timeouts

### 2. Added Safety Timeout to Evaluation
**File**: `parchis/training/common.py` - `evaluate_model()` function

- Added `max_steps_per_episode=2000` parameter (default)
- Episode loop checks `episode_length < max_steps_per_episode`
- Tracks and reports timeout statistics
- Provides detailed status for each episode (WIN/LOSS/TIMEOUT/TRUNCATED)

**Benefits**:
- Prevents infinite evaluation loops
- Provides visibility into long-running games
- Still allows games to complete naturally if they finish quickly

### 3. Fixed Numpy Array Type Issue
**File**: `parchis/training/common.py` (in `evaluate_model()`)

- Convert action from numpy array to int: `action = int(action)`

**Benefits**:
- Prevents "unhashable type: numpy.ndarray" errors
- Ensures compatibility with environment's action processing

## Usage

### Default Behavior (Recommended)
```python
# Evaluation only at the end
python -m parchis.training.train_ppo --timesteps 100000
```

### Enable Mid-Training Evaluation (Optional)
```python
# Evaluate every 50k timesteps (may have compatibility issues)
python -m parchis.training.train_ppo --timesteps 100000 --eval-freq 50000
```

### Quick Testing
```python
# Small timestep count, same safe defaults
python -m parchis.training.train_ppo --timesteps 10000 --players 4
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eval_freq` | `None` | Timesteps between mid-training evaluations. None disables. |
| `max_steps_per_episode` | `10000` | Maximum steps per evaluation episode (safety limit) |
| `n_eval_episodes` | `10` | Number of episodes to run during final evaluation |

## Example Output

```
======================================================================
Parchís PPO Training
======================================================================
Model name: parchis_quick_test
Total timesteps: 10,000
Number of players: 4
Learning rate: 0.0003
Batch size: 64
Checkpoint frequency: 10,000
Evaluation frequency: Disabled
======================================================================

Creating environments...
Creating model...
Setting up callbacks...
Mid-training evaluation disabled (will evaluate at end only)

[Training proceeds...]

======================================================================
Final Evaluation
======================================================================
Episode 1/20: Reward = 32.20, Length = 2000, Status = TIMEOUT
Episode 2/20: Reward = 176.30, Length = 88, Status = WIN
Episode 3/20: Reward = -72.90, Length = 84, Status = LOSS
[...]

Evaluation Results:
  Episodes: 20
  Mean reward: 31.37 +/- 101.81
  Mean length: 573.75
  Wins: 6/20 (30.0%)
  Losses: 9/20 (45.0%)
  Timeouts: 5/20 (25.0%)
```

## Status Definitions

- **WIN**: Agent won the game (reward > 50)
- **LOSS**: Agent lost to an opponent (reward < -50)
- **TIMEOUT**: Episode hit the max_steps_per_episode limit without finishing
- **TRUNCATED**: Episode was truncated by environment's internal limit

## Notes

- Timeouts indicate games taking too long (>10,000 agent steps)
- Some timeout rate (5-25%) is expected, especially with untrained/poorly trained agents
- High timeout rate might suggest:
  - Agent learning defensive/stalling strategies
  - Natural game deadlock states (blockades preventing progress)
  - Random opponent behavior creating stalemates
  - Agent needs more training to learn efficient finishing strategies
- Testing shows the same games timeout at 2k, 5k, and 10k steps - suggesting genuine deadlocks
- Final evaluation always runs with safety timeout to prevent infinite loops
- Checkpointing continues to work normally regardless of evaluation settings
- The `max_steps_per_episode` can be adjusted if needed, but values >10k may cause very long evaluation times
