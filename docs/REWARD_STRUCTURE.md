# Reward Structure

This document describes the reward structure actually implemented in `parchis/rl/rewards.py` (the formulas and constants) and `parchis/rl/env.py` (`ParchisEnv`, which wires them into `step()`). It replaces an earlier version of this document that described a simpler, per-step, no-opponent-term reward design that was never what the code computed — see "History" at the bottom.

## Overview

`ParchisEnv` supports three reward structures, selected via the `reward_type` constructor argument (`ParchisEnv.VALID_REWARD_TYPES`):

- `"progress_delta"` (default): dense, turn-cycle-based progress delta with an opponent-progress penalty term.
- `"win_loss"`: sparse +1.0/-1.0 on game end, 0.0 otherwise.
- `"win_loss_shaped"`: sparse +1.0/-1.0 on game end, plus a small (0.1×) `progress_delta` signal mid-game.

All three share the same underlying progress calculation and the same "when is reward computed" timing described below.

## Progress Per Piece

Each piece's progress is calculated as (`rewards.calculate_normalized_progress`, exposed on the env as `ParchisEnv._calculate_normalized_progress`):

```python
if piece.finished:
    progress = 1.0
elif piece.in_base:
    progress = 0.0
else:
    progress = piece.position / Board.FINAL_POSITION  # position/76.0
```

A player's total progress is the average across their 4 pieces — range `[0.0, 1.0]`.

## When Reward Is Computed: Turn Cycles, Not Steps

Reward is **not** computed on every `step()` call. `step()` returns `reward = 0.0` while the learning agent is still mid-turn — either resolving its own capture/finish bonus chain (`self.pending_bonus is not None`) or mid-way through a six-again reroll streak (`self.consecutive_sixes` between 1 and 2, another roll pending for the same player).

Reward is computed once per **turn cycle**: the learning agent's full turn (including any six-again rerolls and bonus chains it plays) plus every opponent's full turn (including their own rerolls and bonus chains), which the environment auto-plays internally via `opponent_policy_fn` before returning control to the agent. This means a single `step()` call from the agent's perspective can span many dice rolls and multiple players' turns internally, and the reward reflects the *net effect of the whole cycle* — including any captures opponents inflicted on the agent's pieces during their turns, and any three-sixes penalty applied to either side.

## Turn-Cycle Reward (`progress_delta`, the default)

```python
my_delta                = my_progress_now - my_progress_at_cycle_start
combined_opponent_delta = combine_opponent_deltas(opponent_deltas, opponent_start_progress, weighting=opponent_weighting)
reward                  = my_delta - opponent_weight * combined_opponent_delta
```

- `opponent_weight` (α) defaults to `0.5`. It's a `ParchisEnv` constructor argument (`ParchisEnv(opponent_weight=0.3)`) **and** remains a plain mutable attribute afterward (`env.opponent_weight = 0.3` still works, matching the pattern most training scripts and `experiment_alpha_comparison.py` already use).
- `my_delta` reflects the agent's own progress change across the *entire* cycle, so getting captured by an opponent during their turn shows up as a negative contribution here, not as a separate penalty term.
- `combined_opponent_delta` combines every opponent's progress change during their turns this cycle into a single scalar, per `opponent_weighting` (see below). Setting α > 0 makes the reward account for opponents advancing (encourages blocking/defensive play); α = 0 recovers a pure self-progress reward with no opponent term at all.
- At episode end (`terminated`), this same delta is still computed (used to build `win_loss_shaped`'s mid-game term when applicable) but no extra terminal bonus is added for `progress_delta` — progress already reflects finishing pieces.

Typical magnitudes: a single piece moving 5 squares forward contributes `5/76/4 ≈ 0.016` to `my_delta`; a full finished piece contributes `0.25`.

## Opponent Weighting Schemes

`opponent_weighting` (`ParchisEnv` constructor argument, `ParchisEnv.VALID_OPPONENT_WEIGHTING_SCHEMES`) controls how multiple opponents' deltas combine into the single `combined_opponent_delta` term above:

- `"mean"` (default): equal weight to every opponent — `sum(opponent_deltas) / len(opponent_deltas)`. In a 4-player game this dilutes any one opponent's influence to `1/(N-1)` of what it'd be in a 2-player game.
- `"leader"`: weight only the delta of whichever opponent had the **highest progress at the start of the cycle** — the biggest rival entering the cycle, regardless of `num_players`. Deliberately keyed on start-of-cycle progress (not "who gained the most this cycle," which would reward/punish based on that cycle's luck rather than the standing threat).

For `num_players=2` (exactly one opponent), the two schemes are mathematically identical. `"leader"` is **experiment-only**: it is not exposed on the shared training-script CLI (`cli.py`), only via `parchis/training/experiment_alpha_comparison.py --opponent-weighting leader` — no training script defaults to anything but `"mean"` without deliberately passing this flag.

## `win_loss`

```python
reward = 1.0 if terminated and agent_won else (-1.0 if terminated else 0.0)
```

Sparse: zero everywhere except episode end.

## `win_loss_shaped`

```python
if terminated:
    reward = 1.0 if agent_won else -1.0
else:
    reward = 0.1 * progress_delta  # same progress_delta as above
```

Combines a strong terminal win/loss signal with a small dense shaping term mid-game.

## Episode Metrics

`info` includes, on episode end (`terminated` or `truncated`): `final_progress`, `pieces_finished`, `pieces_out_of_base`, `won`. Training scripts' `ProgressLoggingCallback` classes log these (and win rate) to TensorBoard as `metrics/final_progress`, `metrics/win_rate`, `metrics/pieces_finished`, `metrics/pieces_out_of_base`.

## Implementation Reference

- `parchis/rl/rewards.py` — the source of truth for the reward formulas and named constants:
  - `calculate_normalized_progress(player)` — per-player progress (pure function)
  - `combine_opponent_deltas(opponent_deltas, opponent_start_progress, weighting=...)` — the `"mean"`/`"leader"` opponent-weighting schemes
  - `compute_reward(reward_type, my_delta, combined_opponent_delta, opponent_weight, terminated, agent_won)` — the three `reward_type` branches
  - `VALID_REWARD_TYPES`, `VALID_OPPONENT_WEIGHTING_SCHEMES`, `DEFAULT_OPPONENT_WEIGHT`, `DEFAULT_OPPONENT_WEIGHTING`, `WIN_REWARD`, `LOSS_REWARD`, `WIN_LOSS_SHAPED_MIDGAME_SCALE`
- `parchis/rl/env.py`:
  - `step()` — calls into `rewards.compute_reward`/`combine_opponent_deltas`, gated on `turn_cycle_complete or terminated` (a turn cycle stays open across six-again rerolls and bonus chains alike)
  - `ParchisEnv.__init__(reward_type=..., opponent_weight=..., opponent_weighting=...)` — selects the reward structure and its parameters; `env.opponent_weight`/`env.opponent_weighting` remain mutable attributes afterward too
  - `_calculate_normalized_progress()` — thin wrapper around `rewards.calculate_normalized_progress`, kept for backward compatibility (used by `common.py`'s `evaluate_model`, the observation's progress-scores block, etc.)
  - `Game.apply_three_sixes_penalty()` (`parchis/game/game.py`) — the shared three-sixes rule implementation, called from both `step()` and `Game.play_turn()`; unrelated to the reward module but the other cross-cutting shared-implementation pattern in this file

## History

An earlier version of this document described a much simpler design: a single per-*step* `reward = new_progress - old_progress` with explicitly no opponent term ("removing opponent noise" was listed as a design goal). That description never matched the shipped code — the default reward has always included the `opponent_weight` term above, and reward has always been computed per turn-cycle, not per step. This revision replaces that description with what `env.py` actually does.

A later research pass compared this design against AlphaZero/AlphaGo Zero (pure sparse `±1`, credit assignment offloaded to MCTS + the value network — this project uses plain PPO with no search, so a sparse-only signal is a harder assignment problem here), TD-Gammon (also sparse, relying on TD(λ) rather than hand-encoded progress), OpenAI Five's "team spirit" zero-sum adjustment (structurally the same as `mean` opponent-weighting, but annealed over training rather than fixed — noted as a future idea, not built), and potential-based reward shaping theory (Ng, Harada & Russell 1999 — `win_loss_shaped` structurally resembles but doesn't literally satisfy the theorem, since it uses undiscounted `Φ(end) - Φ(start)`; plain `progress_delta` is a proxy objective, not shaping on top of a base one). No formula changed as a result — recorded here since it's real design context, not restated in the sections above.

That pass also flagged that `leader` vs. `mean` opponent-weighting were mathematically identical in every comparison run so far, because they only diverge at `num_players > 2` (see "Opponent Weighting Schemes" above) and every run at the time was 2-player. This gap has since been closed: the 4-player win-rate matrix (`parchis/evaluation/multiplayer_matrix.py`) directly compares `leader`- and `mean`-weighted checkpoints at 4 players.
