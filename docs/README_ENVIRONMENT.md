# Parchís RL Environment - Getting Started

## Quick Start

### 1. Basic Environment Usage

```python
from parchis.rl.env import ParchisEnv

# Create environment
env = ParchisEnv(num_players=4)

# Reset environment
obs, info = env.reset(seed=42)

# Take a step
action = 0  # Move piece 0
obs, reward, terminated, truncated, info = env.step(action)

# Check action masks for valid moves
valid_actions = info['action_masks']  # [0, 1, 0, 1] means pieces 1 and 3 can move
```

### 2. Training with PPO (Basic)

```python
from stable_baselines3 import PPO
from parchis.rl.env import ParchisEnv

env = ParchisEnv(num_players=4)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("parchis_ppo")

model = PPO.load("parchis_ppo")
obs, info = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### 3. Training with Action Masking (Recommended)

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from parchis.rl.env import ParchisEnv

def mask_fn(env):
    """Return the action mask from the environment."""
    return env.unwrapped._get_info()['action_masks']

env = ParchisEnv(num_players=4)
env = ActionMasker(env, mask_fn)

model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

obs, info = env.reset()
for _ in range(100):
    action_masks = mask_fn(env)
    action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

In practice, use `parchis.training.train_ppo.make_env()` rather than constructing/wrapping the environment by hand — see `docs/TRAINING_GUIDE.md`.

## Environment Details

### Observation Space (dynamic size: `79 * num_players + 31`)

For 4 players this is 347 values; for 3 players, 268; for 2 players, 189. The observation is a flat `float32` array in `[0.0, 1.0]`:

1. **Board state (`num_players × 76` values)**: one 76-value channel per player, ordered by turn with the *current* player first. Each channel covers positions 1-76 (1-68 main track, 69-76 home column); a position's value is `0.0` (no piece there), `0.5` (one piece), or `1.0` (two pieces, a stack/blockade).

2. **Piece counts (`2 × num_players` values)**: per player, `pieces_in_base / 4.0` and `pieces_finished / 4.0`.

3. **Progress scores (`num_players` values)**: per player, average completion across their 4 pieces — `0.0` in base, `position / 76.0` on board, `1.0` finished (see `docs/REWARD_STRUCTURE.md` for how this feeds the reward too).

4. **Dice roll, one-hot (7 values)**: `is_dice_1` .. `is_dice_5`, `is_dice_6_normal` (rolled 6, still have pieces in base), `is_dice_6_no_base` (rolled 6, all pieces already out — moves 7 squares).

5. **Bonus indicator (2 values)**: `has_finish_bonus` and `has_capture_bonus`, mutually-exclusive binary flags reflecting whether a finish (10-square) or capture (20-square) bonus move is pending for the current player. Both `0.0` when no bonus is pending. (Previously a single continuum-encoded `bonus_squares / 20.0` scalar — replaced because finish-bonus and capture-bonus are qualitatively different situations, not degrees of the same quantity; see `docs/observation_space_changes.md` Decision 1.)

6. **Own-piece features (21 values = 4 pieces × 5 per-piece features, + 1 shared value)**: unlike the board-state block above (which is reordered by turn), the per-piece features are indexed *strictly by `piece.piece_id`* so they line up with the `Discrete(4)` action space — the network can otherwise never distinguish "choose action 0" from "choose action 3" beyond legality.
   - Per piece (stride 5): `in_base`, `finished`, `normalized_position` (`0.0` in base, `1.0` finished, else `position/76.0`), `on_safe_square` (main-track safe square or home column), `capture_threat_score` — a roll-based `[0.0, 1.0]` score: the fraction of the 6 dice faces (summed across *all* opponents, not deduplicated — two opponents each threatening with the same roll value counts twice) that would let some opponent capture this piece this turn, either directly or via the capture/finish bonus move that roll unlocks. Computed by querying `RuleEngine.get_legal_moves`/the new `RuleEngine.would_capture` helper for each candidate roll, never by hand-rolled distance math — this makes it exact with respect to every rule edge case (mandatory-5-entry, the 6-with-0-base 7-move, the 6-with-blockade must-open restriction, entry captures on an opponent's "safe" starting square), unlike the old binary `capture_threatened` bit it replaces (which was capped at distance 6, ignored mandatory-5-entry, and ignored bonus chains). One accepted, deliberate approximation remains: the bonus-chain check doesn't simulate the *same* piece that triggered a capture/finish continuing its own chain from its new position — only other already-on-board opponent pieces are considered.
   - Shared (1 value, `own_piece_offset + 20`, **not** indexed by piece_id): `capture_opportunity` — a roll-based `[0.0, 1.0]` score covering all 4 of the agent's own pieces at once: the fraction of the 6 dice faces for which the agent has at least one legal capturing move via *any* of its 4 pieces (OR'd across pieces, so a roll value that lets two different own pieces each capture only counts once). Explicitly single-roll only — unlike `capture_threat_score`, it does not extend through bonus chains. Replaces the old per-piece binary `capture_opportunity` bit.

   See `docs/observation_space_changes.md` (Decisions 2 and 5) for the full design rationale and the pure, non-mutating `RuleEngine.would_capture` helper these scores are built on.

7. **Six-streak (1 value)**: `consecutive_sixes / 3` for whoever's turn is currently being resolved (agent or, mid-auto-play, an opponent). This is now the final block in the observation.

The blockade indicator (own/opponent blockade counts) and the bonus-chain-count block have both been **removed entirely** from the observation (not zeroed — the observation is simply shorter), per `docs/observation_space_changes.md` Decisions 3 and 4: the blockade count wasn't tied to any specific actionable decision (illegal moves from blockades are already fully captured by `action_masks`), and the bonus-chain-length counter didn't add information beyond what the board-state/piece-count blocks already encode. `bonus_chain_count` is still tracked internally and exposed via `info['bonus_chain_count']` (used for KPI logging in `parchis/training/common.py` and `parchis/evaluation/evaluate.py`) — it's just no longer part of the observation array.

### Action Space

- `Discrete(4)`: choose which piece (0-3) to move.
- Use `info['action_masks']` to find valid actions; the environment does not penalize invalid actions itself (masking is meant to prevent them from being chosen at all — see `docs/CODE_REVIEW.md` for the open finding about actions with no legal moves producing an all-ones mask).

### Rewards

See `docs/REWARD_STRUCTURE.md` for the full description. In short: `ParchisEnv(reward_type=...)` selects one of `"progress_delta"` (default, dense, turn-cycle based with an opponent-progress term), `"win_loss"` (sparse ±1.0), or `"win_loss_shaped"` (±1.0 terminal plus a small dense term). There is no fixed per-event reward table (no flat "+10 for finishing a piece" style bonus) — reward is always derived from the progress calculation above.

### Bonus Moves

The learning agent controls its own bonus moves:
- **Capture bonus**: 20 squares after capturing an opponent piece.
- **Finish bonus**: 10 squares after a piece reaches the final position (76).
- **Chaining**: bonuses can chain (capture → 20 → capture → 20 → finish → 10 → ...); each bonus move is a separate `step()` call for the agent (`info['is_bonus_move']` / `info['bonus_type']` / `info['bonus_squares']` indicate this).
- Opponent bonus moves are resolved automatically and internally (see `opponent_policy_fn` in `parchis/rl/env.py`), never surfaced as a `step()` call to the agent.

### Six-Again / Three-Sixes

`ParchisEnv` implements the full rule (`docs/RULES.md`), not a simplified variant:
- Rolling a 6 grants a reroll: the same player gets another `step()` call for a fresh roll instead of the turn ending.
- Three consecutive 6s in one turn: the third six is never used to move a piece — the piece moved on the *second* six is captured (sent back to base) and the turn ends immediately, unless that piece transitioned into its home column on that exact move (home-entry protection) or no piece was moved on the second six.
- A bonus chain triggered by a six (e.g. a capture) always resolves fully before the six-streak decision (reroll vs. penalty vs. normal end) is made.
- Reward stays `0.0` during six-streak rerolls, exactly like during a bonus chain — it's only computed once the whole turn cycle genuinely ends.
- This applies symmetrically to the agent and to every auto-played opponent, via one shared implementation (`Game.apply_three_sixes_penalty`).

## Files

- `parchis/game/`: core game engine (`game.py`, `board.py`, `rules.py`, `player.py`, `piece.py`, `dice.py`, `records.py`, `formatting.py`, `constants.py`)
- `parchis/rl/env.py`: main Gymnasium environment (`ParchisEnv`)
- `parchis/rl/env_selfplay.py`: self-play wrapper (`ParchisSelfPlayEnv`)
- `parchis/tests/test_new_rewards.py`: reward-structure tests
- `parchis/tests/test_selfplay.py`: self-play opponent-model wiring tests
- `parchis/training/`: training scripts (see `docs/TRAINING_GUIDE.md`)

## Running Examples

```bash
# Run the reward-structure tests
python -m pytest parchis/tests/test_new_rewards.py -v

# Run the self-play wiring tests
python -m pytest parchis/tests/test_selfplay.py -v

# Quick training smoke test (10K timesteps)
python -m parchis.training.train_ppo --timesteps 10000 --players 4
```

## Game Rules

The game engine (`parchis/game/`) implements the full rule set in `docs/RULES.md`, including:
- Blockades block ALL players (including whoever created them).
- Home-entry and exact-landing-on-76 calculations (fixed as of the domain-A rebuild described in `docs/CODE_REVIEW.md` — see that document if you're working with a checkout from before that fix).
- Bonus moves are fully controllable by the agent, with chaining.
- Six-again rerolls and the three-consecutive-sixes penalty (see "Six-Again / Three-Sixes" above).

## Troubleshooting

### Agent not learning
- Check action masking is working (`info['action_masks']`).
- Try a different `reward_type` (see `docs/REWARD_STRUCTURE.md`).
- Increase training time or adjust hyperparameters.
- Monitor with TensorBoard (see `docs/TRAINING_GUIDE.md` for the exact flags each training script logs).

### Old saved models won't load / observation shape mismatch
If you have a model checkpoint from before the domain-A/observation fixes in `docs/CODE_REVIEW.md`, from before the piece-indexed observation redesign in `docs/RL_DESIGN_REVIEW.md` (observation size `79*num_players+8` → `79*num_players+36`), or from before the Decision 1-5 observation-space changes in `docs/observation_space_changes.md` (`79*num_players+36` → `79*num_players+31`), it was trained against a different observation space and will fail to load against the current `ParchisEnv` with a shape-mismatch error. It needs to be retrained.
