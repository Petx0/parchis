# RL Training Approach Review — Parchís PPO Agent

## Context

The project (Parchís/Ludo self-play PPO agent) already went through one full code-review pass (`docs/CODE_REVIEW.md`) that fixed genuine correctness bugs: self-play was silently training against random play, home-column movement was broken, and the observation double-counted finished pieces. That pass is closed out and `docs/REWARD_STRUCTURE.md`/`docs/README_ENVIRONMENT.md` now accurately describe the current code.

This review goes one level up: assuming the code does what it's documented to do, **is the RL design itself — game representation, reward shaping, self-play curriculum, hyperparameter exposure, and KPI tracking — actually capable of producing a strong Parchís agent?** Two research passes over `parchis/rl/`, `parchis/game/`, `parchis/training/`, and `parchis/evaluation/` surfaced two structural gaps more consequential than anything in the prior bug-fix pass, plus a set of secondary gaps in training rigor and observability:

1. **The environment plays a simplified variant of the real game.** `ParchisEnv.step()` hand-rolls a single-roll-per-turn loop that never grants a reroll on 6 and never triggers the three-consecutive-sixes penalty — even though `Game.play_turn()` (used by the non-RL simulator) already implements both correctly. The agent has never faced the push-your-luck tension (reroll vs. risk losing a piece) that's core to real Parchís strategy. **Decision made: fix this properly**, not just document it as a known limitation.
2. **Action space / observation space mismatch.** The action space picks a `piece_id` (0-3) to move, but the observation only encodes aggregate per-square occupancy — nothing tells the network which square any *specific* piece_id occupies, whether that piece is about to be captured, or whether it's one move from finishing. When 2+ pieces have simultaneous legal moves, the network has no signal distinguishing the consequence of choosing one action index over another, only legality (via the mask). This is likely the single highest-leverage fix available.

Beyond the environment itself, the training pipeline has real rigor gaps: PPO's most impactful hyperparameters (`n_epochs`, `gae_lambda`, `clip_range`) are hardcoded and have never been tuned or swept; self-play uses a single rolling opponent snapshot with no progress curve logged during training; every experiment sweep (`experiment_alpha_comparison.py`, `experiment_grid.py`) runs a single seed per configuration and picks a "best" config by raw point estimate, indistinguishable from seed noise; and there's no cross-checkpoint skill tracking (Elo-style or otherwise) to confirm training is actually making the agent stronger over time.

**Goal for this work: a stronger final agent.** KPI/instrumentation improvements are included because they're needed to trust that later changes (reward tuning, self-play curriculum changes) are real improvements and not noise — not as an end in themselves.

This is a large, multi-phase initiative, executed incrementally. **Phases 1-4 were implemented starting 2026-08-02, each once the prior phase's effect was validated.** Phase 5 was detailed and its tooling/runbook built starting 2026-08-03; the real multi-hour training runs it depends on have not been executed yet (see Phase 5 below).

---

## Phase 1 — Foundational fidelity + representation fixes

Four independent-but-sequenced changes. **Order**: items 1→2 are hard-sequenced (item 2's six-streak observation feature reads state item 1 creates); items 3 and 4 have no dependency on 1/2 or each other.

### 1. Six-again / three-sixes rule fidelity — `parchis/rl/env.py`, `parchis/game/game.py`

**Key constraint that makes this tractable:** a turn's roll sequence is always exactly one of `[non-6]`, `[6, non-6]`, `[6, 6, non-6]`, `[6, 6, 6→penalty]` — at most 3 dice-roll decisions per turn, never unbounded. Bonus chains (capture/finish, already unbounded-in-principle via the existing `pending_bonus` mechanism) are a separate, unrelated mechanism.

**Shared rule logic** — new `Game` staticmethod so the penalty has exactly one implementation, called by both `Game.play_turn()` (existing simulator) and the new `ParchisEnv` opponent-turn helper:

```python
@staticmethod
def apply_three_sixes_penalty(board, second_six_piece, second_six_entered_home):
    """Capture the piece moved on the *second* of three consecutive 6s,
    unless it entered the home column on that move (docs/RULES.md Exception 2)
    or no piece was moved on the second six specifically."""
    if second_six_piece is None or second_six_entered_home or second_six_piece.finished:
        return False, bool(second_six_entered_home)
    board.remove_piece(second_six_piece)
    second_six_piece.send_to_base()
    return True, False
```

While refactoring `Game.play_turn()` to call this, fix two latent bugs uncovered in this review (both distinct from the home-column-exception ambiguity a prior session already closed):
- `play_turn()`'s `last_moved_piece` tracking isn't scoped to specifically "the roll where the streak count is exactly 2" — a second six with no legal move currently leaves stale state that a third six could wrongly punish. Fixed by tracking `second_six_piece`/`second_six_entered_home` scoped precisely to that roll.
- The sixes check previously ran even after `player.has_won()` already fired in the same iteration. Added an early return once a win is detected, before the sixes check, in both `Game.play_turn()` and the new `ParchisEnv` opponent helper.

**New `ParchisEnv` instance state** (init'd in `__init__`, reset in `reset()`): `self.consecutive_sixes = 0`, `self.second_six_piece = None`, `self.second_six_entered_home = False`. Imports `BONUS_TURN_ROLL`, `THREE_SIXES_LIMIT` from `parchis/game/constants.py`.

**Two new private helpers** replace every place `env.py` previously did `self.current_dice_roll = self.game.dice.roll()`:
- `_roll_dice_for_current_player()` — rolls, updates `consecutive_sixes` (increment on 6, reset to 0 otherwise), clears `second_six_piece`/`second_six_entered_home` whenever the streak count leaves 2.
- `_advance_to_next_player()` — resets all three streak-state fields, calls `self.game.next_player()`, then `_roll_dice_for_current_player()`. `reset()` calls `_roll_dice_for_current_player()` directly (no prior player to advance from).

**Move-execution block**: when a move executes outside a bonus chain and `consecutive_sixes == 2`, records `second_six_piece = piece` and computes `second_six_entered_home` by comparing old/new position against `Board.HOME_COLUMN_START`.

**Resolution order in `step()`** — bonus chains always resolve fully *before* any six-streak decision:
```
if current_player.has_won(): terminated = True
elif pending_bonus is not None: turn_over = False   # unchanged bonus-chain handling
else:
    if consecutive_sixes == THREE_SIXES_LIMIT:
        Game.apply_three_sixes_penalty(...); turn_over = True
    elif current_dice_roll == BONUS_TURN_ROLL:
        turn_over = False; _roll_dice_for_current_player()   # same player, fresh roll
    else:
        turn_over = True
```
`turn_over` remains the single gate for reward computation (`if turn_over or terminated:` in `step()`, unchanged) — reward still fires exactly once per full turn-cycle, which can now correctly span 1-3 agent dice-roll decisions instead of always exactly 1. No reward-formula change; `docs/REWARD_STRUCTURE.md` gets a one-line addendum noting this.

**New opponent-turn helper** `_auto_play_full_turn(player)` replaces the old single-roll auto-play loop body — uses **local variables** (not instance state, since it resolves synchronously within one agent `step()` call), mirrors `Game.play_turn()`'s loop but drives moves through `self.opponent_policy_fn` instead of `player.choose_move`, and calls the shared `Game.apply_three_sixes_penalty` helper.

**Tests** (`parchis/tests/test_env.py`, `test_game.py`, `test_new_rewards.py`, `test_selfplay.py`):
- `test_six_again_rerolls_same_player`, `test_three_sixes_penalty_captures_second_six_piece`, `test_three_sixes_home_entry_protection`, `test_three_sixes_no_legal_move_on_second_six_no_capture` (regression for the bug fixed above), `test_six_then_capture_bonus_then_six_chain` (tests bonus-before-six-decision resolution order) — all in `test_env.py`, using scripted dice sequences.
- `test_apply_three_sixes_penalty_helper` (direct unit test of all four cases: normal, home-protected, no-piece, already-finished) and a `game.py`-side no-legal-move regression, both in `test_game.py`.
- `test_six_streak_intermediate_steps_return_zero_reward` in `test_new_rewards.py`.
- `test_selfplay.py`: re-run as regression only.

### 2. Piece-indexed + strategic-feature observation redesign — `parchis/rl/env.py`

**New total size: `79 * num_players + 36`** (was `79N + 8`; 4p: 324→352, 2p: 166→194). `models/` was confirmed empty, so no existing checkpoint is invalidated beyond what the prior review already cleared.

New layout, appended after the existing 5 unchanged blocks (board state, piece counts, progress scores, dice one-hot, bonus indicator):

| # | Field | Size | Offset | Normalization | Meaning |
|---|---|---|---|---|---|
| 6 | Own-piece features | 24 | `[79N+8, 79N+32)` | mixed, see below | 4 pieces × 6 features, **fixed slot by `piece_id`** (not turn/position order) |
| 7 | Blockade indicator | 2 | `[79N+32, 79N+34)` | `/12.0` clipped | own-blockade count, opponent-blockade count |
| 8 | Six-streak | 1 | `79N+34` | `/THREE_SIXES_LIMIT` | depends on item 1's `self.consecutive_sixes` |
| 9 | Bonus chain count | 1 | `79N+35` | `/4.0` clipped | exposes existing `self.bonus_chain_count` (previously info-only) |

**Own-piece block**, offset `base = 79N + 8 + piece_id * 6` — indexed strictly by `piece.piece_id`, never reordered by turn like the board-state block is:

| Slot | Feature | Value |
|---|---|---|
| `base+0` | `in_base` | 1.0/0.0 |
| `base+1` | `finished` | 1.0/0.0 |
| `base+2` | `normalized_position` | 0.0 in base; 1.0 finished; else `position/Board.FINAL_POSITION` |
| `base+3` | `on_safe_square` | 1.0 if on-board and (`position in Board.SAFE_SQUARES` or `position >= Board.HOME_COLUMN_START`) |
| `base+4` | `capture_threatened` | 1.0 if any opponent piece is 1-6 squares *behind* this piece on a capturable (non-safe, on-board) square |
| `base+5` | `capture_opportunity` | 1.0 if any opponent piece is 1-6 squares *ahead* of this piece on a non-safe square |

Threat/opportunity computed via wraparound-safe modular distance. Two deliberate, documented simplifications: capped at distance 6 (misses the rare 7-square move when an opponent has no base pieces), and ignores intervening blockades (coarse heuristic, not a legality guarantee).

**Blockade indicator**: `own_blockades = count of Game.get_blockades() positions owned by the agent's color`; `opponent_blockades = total - own`; both `/12.0` (12 safe squares total) and clipped to 1.0.

**Six-streak / bonus-chain-count**: `obs[79N+34] = consecutive_sixes / THREE_SIXES_LIMIT`, `obs[79N+35] = min(bonus_chain_count / 4.0, 1.0)`.

`_get_info()`/`action_masks` required **no change** — confirmed independent of observation layout.

**Docs**: `docs/README_ENVIRONMENT.md`'s observation-space section updated (size formula, new blocks, worked examples for 2p/4p).

**Tests** (`parchis/tests/test_observation.py`, `test_env.py::test_observation_structure`):
- Total-size assertion updated to `79N+36` for 2/3/4 players; `observation_space.contains(obs)` re-verified.
- Fixed-slot-by-`piece_id` test: force piece_id 2 specifically into base, assert only that piece's 6-slot block reflects it, independent of turn order.
- Blockade indicator test: construct an own blockade and a separate opponent blockade, assert both indices respond independently.
- `capture_threatened`/`capture_opportunity` tests at distances 1, 6 (should fire), and 7 (should not — documents the accepted approximation).
- Six-streak/bonus-chain-count tests: set `env.consecutive_sixes`/`env.bonus_chain_count` directly, assert the corresponding observation indices.

### 3. Expose missing PPO hyperparameters — `parchis/training/cli.py`, `train_ppo.py`, `train_selfplay.py`

`n_epochs=10`, `gae_lambda=0.95`, `clip_range=0.2` were hardcoded identically in every training script and never exposed to CLI or to `experiment_grid.py`'s sweep — meaning no experiment run through this codebase had ever varied PPO's own optimization hyperparameters, only reward shaping and network size.

- Added `--n-epochs` (default 10), `--gae-lambda` (default 0.95), `--clip-range` (default 0.2) to `add_ppo_hyperparam_args` in `cli.py`, threaded through to `train()`/`train_selfplay()`'s `MaskablePPO` construction, replacing the hardcoded values.
- **Decision: did not expose `vf_coef`/`max_grad_norm`/`target_kl`.** They're left at SB3 defaults everywhere and are less likely to be the binding constraint for this problem size than `n_epochs`/`gae_lambda`/`clip_range`; revisit only if training shows instability that these specifically would address.

### 4. Self-play progress logging + fixed-baseline win-rate tracking — `parchis/training/common.py`, `train_selfplay.py`, `train_continue.py`

Two gaps: (a) `ProgressLoggingCallback` was wired into `train_ppo.py` and both experiment scripts but **not** `train_selfplay.py`/`train_continue.py` — self-play runs produced no progress curve at all. (b) Self-play's only win-rate measurement happened once, at the very end, against random opponents — no way to see whether the agent's win rate against a *fixed, stationary* reference improved monotonically during training, since the self-play opponent itself keeps changing.

- **Mechanical fix (a)**: added `ProgressLoggingCallback` to the callback list in `train_selfplay.py` and `train_continue.py`, matching the existing pattern in `train_ppo.py`.
- **New fix (b)**: new `FixedOpponentEvalCallback` (a `BaseCallback` subclass in `common.py`, following the existing `SelfPlayCallback` pattern for periodic-trigger structure — **not** SB3's built-in `EvalCallback`, which is documented in `docs/EVALUATION_FIX.md` as hanging with `MaskablePPO`). Every `eval_freq` timesteps, evaluates against a fixed baseline (random-opponent env by default) using the existing `evaluate_model()` (already has the safety-timeout pattern), logging to TensorBoard as `metrics/win_rate_vs_baseline` — decoupled from the moving self-play opponent.

---

## Phase 1 verification

1. `pytest parchis/tests/` green after each item lands.
2. `python -m parchis.training.train_quick` end-to-end smoke test — no crashes, observation shape matches `79N+36`, TensorBoard metrics populate.
3. A short `train_selfplay` smoke run — confirm `metrics/win_rate_vs_baseline` and `metrics/final_progress` actually appear in TensorBoard.
4. A multi-hundred-episode stress test confirming the observation never leaves `[0,1]` bounds, now covering six-streak scenarios explicitly.
5. Hand-traced spot checks against `docs/RULES.md`'s worked examples for a forced three-sixes sequence and a six-then-capture chain.

---

## Phase 2 — Reward structure refinement

Reward computation extracted into its own `parchis/rl/rewards.py` module (`calculate_normalized_progress`, `combine_opponent_deltas`, `compute_reward`, named constants `WIN_REWARD`/`LOSS_REWARD`/`WIN_LOSS_SHAPED_MIDGAME_SCALE`/etc.), closing the rebuild target `docs/CODE_REVIEW.md` had flagged as open. `ParchisEnv.step()` now calls into this module instead of computing the formula inline; `_calculate_normalized_progress()` became a thin backward-compatible wrapper. Verified behavior-preserving via a bit-for-bit regression check: 6,022 reward computations across all 3 `reward_type`s and 30 seeds each, compared against an independently-written reference implementation of the old inline formula — zero mismatches.

`opponent_weight` became a real `ParchisEnv`/`make_env()` constructor argument (previously post-construction-mutation only, `env.opponent_weight = 0.3`) — the old mutation pattern still works unchanged, the constructor arg is additive.

Added a new `opponent_weighting` scheme, `"leader"`, alongside the existing `"mean"` (default, unchanged behavior): `"mean"` dilutes any one opponent's influence to `1/(N-1)` of its 2-player strength as `num_players` grows; `"leader"` instead weights only the delta of whichever opponent had the highest progress at the *start* of the cycle, so the term's strength doesn't dilute with more players. Verified mathematically identical to `"mean"` for `num_players=2` (exactly 1 opponent) and genuinely different for `num_players=4`, via matched-action sequential episode replay (avoids the pitfall of running two `ParchisEnv` instances interleaved, which would fight over Python's shared global `random` stream `Dice.roll()` draws from). Deliberately **experiment-only**: not exposed on the shared training-script CLI (`cli.py`), only via `parchis/training/experiment_alpha_comparison.py --opponent-weighting {mean,leader}` — no training script's default behavior changed.

18 new tests: `parchis/tests/test_rewards_module.py` (one test per extracted term — progress calculation, both weighting schemes including an invalid-scheme-raises case, all three reward-type branches including invalid-type-raises) plus 2 new end-to-end tests in `parchis/tests/test_new_rewards.py` (constructor-arg equivalence, `"leader"` vs `"mean"` provably differing on a contrived 4-player state).

---

## Phase 3 — Self-play curriculum upgrade

The single rolling-latest-snapshot opponent was replaced with a small pool of past checkpoints (`parchis/rl/opponent_pool.py`: `compute_recency_weights`, `compute_win_rate_weights`, `sample_pool_index`, `pool_diversity_entropy`, plus `VALID_POOL_SAMPLING_STRATEGIES`/named constants — mirrors the `rewards.py` extraction pattern from Phase 2). `ParchisSelfPlayEnv` gained `update_opponent_pool()` (the new primitive; `update_opponent_model()` is now a thin one-member wrapper over it) and samples one pool member per **episode** at `reset()` — a deliberate, documented change from the old mid-episode-swap timing — via a dedicated `random.Random(pool_seed)` instance, never the shared global `random` module the dice rolls draw from.

`SelfPlayCallback` (`parchis/training/train_selfplay.py`) now maintains a `deque(maxlen=pool_size)` of checkpoints, refreshed every `--opponent-update-freq` timesteps: `--pool-size` (default **5** — a genuine default-behavior change from the old implicit pool-of-1; `--pool-size 1` reproduces the old steady-state behavior exactly) and `--pool-sampling-strategy` (`uniform` default, `recency`, or `win_rate` — the latter re-scores every pool member against the live training model each update via the existing `evaluate_model()`, at a real recurring cost of `pool_size × pool_eval_episodes` extra episodes per update, so it's opt-in only). Evicted pool members' checkpoint *files* are never deleted from disk — only what's live-sampled during training is capped — so Phase 4's planned round-robin checkpoint ladder still has the full training history available. Pool members load with `device="cpu"` (a new precedent, avoiding GPU contention with the actively-training model).

Pool diversity — how evenly episodes were spread across the pool, not just whether it has >1 member — is logged as a new KPI: `metrics/opponent_pool_diversity` (normalized Shannon entropy over per-member selection counts, dense over the whole pool so under-sampled members aren't silently hidden; skipped on the very first update, when the pool was empty) and `metrics/opponent_pool_size`.

25 new tests: `parchis/tests/test_opponent_pool.py` (pure-function coverage — weight formulas, seeded-RNG sampling distribution, entropy edge cases), 3 new tests in `parchis/tests/test_selfplay.py` (uniform/skewed pool sampling at the `ParchisSelfPlayEnv` level, `update_opponent_model()` backward-compatibility), and `parchis/tests/test_selfplay_pool_callback.py` (pool growth/eviction with checkpoint-file retention, the diversity metric's first-update edge case, `"win_rate"` strategy scoring — all driven by calling `SelfPlayCallback._update_opponents()` directly rather than a full training loop, for speed).

---

## Phase 4 — KPI & evaluation framework

Four additions, all analysis-layer only — no reward, environment, or training-loop semantics changed. Two new pure-function modules mirror the `rewards.py`/`opponent_pool.py` style (named constants, `ValueError` on invalid input, no I/O):

- `parchis/evaluation/stats.py`: `wilson_score_interval(wins, n, confidence=0.95)` (Wilson, not the normal approximation — stays valid at small `n` and at `p` near 0/1, which is exactly the regime evaluation runs live in), `mean_confidence_interval(values, confidence=0.95)` (Student's t, not normal — correct even at 2-3 seeds), `intervals_overlap(a, b)`, `breakdown_win_rates(wins_by_key, games_by_key)`, `rank_by_mean_with_ci(entries)` (the shared "best config" gate below), and `aggregate_phase4_stats(...)` (the shared tail-stage aggregator both evaluation entry points call — see below). Uses `scipy.stats` for correct critical values (now a declared dependency, `requirements.txt`); was already present transitively but hadn't been an explicit requirement.
- `parchis/evaluation/elo.py`: `expected_score(rating_a, rating_b)`, `update_ratings(rating_a, rating_b, score_a, k_factor=32.0)` (one update per *pairing*, not per game — a pairing's whole `games_per_pairing` block is treated as one Elo observation, since it's built directly on `evaluate_agent()`'s existing per-pairing win rate rather than requiring per-game-level integration into the RL episode loop; documented simplification, acceptable for a *lightweight* ladder), `round_robin_pairings(participant_ids, rng)` (shuffled via a dedicated `random.Random`, never the bare module, matching `opponent_pool.py`'s convention).

**Elo ladder** (`parchis/evaluation/elo_ladder.py`, new CLI entry point): round-robins a set of saved checkpoints (+ an optional random-baseline pseudo-participant) against each other, reusing `evaluate_agent()` per pairing (so it inherits the hang-safety timeout from `docs/EVALUATION_FIX.md` for free). **2-player matches only** — `ParchisSelfPlayEnv` puts the *same* opponent model in every non-agent seat for `num_players > 2`, so a 3-4p match between two checkpoints is actually "A vs 3×B," not a clean pairwise comparison; no valid Elo interpretation without a much larger multiplayer-Elo redesign the "lightweight" framing doesn't call for. Checkpoints are passed explicitly via `--checkpoints` rather than auto-discovered — checkpoint naming has no single convention across this codebase's training scripts (`checkpoint_<N>_steps`, `opponent_checkpoint_<n>_<steps>steps`, bare `final_model`, `alpha_<v>_<weighting>_<timestamp>`, `<arch>_<reward_type>` with no shared prefix at all), so auto-detection would be real, error-prone complexity for no clear benefit. Because `agent_player_idx` is already randomized per-episode, a pairing's games already split roughly evenly across both seats regardless of which checkpoint is passed as `evaluate_agent`'s nominal "agent" vs. "opponent" — no manual seat-alternation needed. Results print as a ratings table + per-pairing win-rate/CI lines, and save incrementally to `results.json` (crash-recovery pattern, mirrors `experiment_grid.py`).

**Wilson CI + per-seat/color breakdown**, wired into both `evaluate_model` (`parchis/training/common.py`) and `evaluate_agent` (`parchis/evaluation/evaluate.py`) — these are two separate, pre-existing near-duplicate evaluation loops (one used by training scripts, one by the standalone CLI against an arbitrary opponent checkpoint or random); rather than duplicating the new CI/KPI formulas in both (the exact class of drift `docs/CODE_REVIEW.md` already flagged once for these two files), both call the shared `stats.aggregate_phase4_stats(...)` at their existing stats-dict-construction tail. New stats-dict keys: `win_rate_ci`, `win_rate_by_seat`, `win_rate_by_color` (each `{key: {win_rate, n, ci}}`, bucketed by the already-randomized `agent_player_idx`/its color — confirms whether the seat-randomization fix from `docs/CODE_REVIEW.md` actually produced fair outcomes, which nothing had checked before).

**Richer KPIs** — capture rate, legal-move-count distribution, bonus-chain-length distribution, three-sixes-penalty rate — via a **lightweight** route rather than literal `GameLogger`/`TurnInfo` wiring (a real, deliberate scope decision, confirmed with the user): `GameLogger.log_turn()` is only ever called from `Game.play_turn()`, which `ParchisEnv` never calls — the RL env reimplements the turn loop itself, fragmented across `step()` calls for the agent's turn and a separate synchronous loop for opponents. Full `GameLogger` wiring would mean replicating `play_turn()`'s `TurnInfo`/`RollEntry` bookkeeping inside that fragmented flow for no KPI benefit the lighter route doesn't already get. Instead, `ParchisEnv` (`parchis/rl/env.py`) gained three small additions, each at a point where the underlying data was already computed locally and discarded: `captures_by_agent`/`captures_against_agent` counters (reset per turn cycle, incremented at the three existing move-execution capture checks — the agent's own move in `step()`, and both the initial move and chained bonus moves in the opponent-turn path, the latter filtered to only count captures of the agent's own color, since a 3-4p opponent move can capture a *different* opponent), and `three_sixes_penalty` (capturing `Game.apply_three_sixes_penalty`'s previously-discarded return value in `_roll_and_check_sixes()`). All three surface in `info`, gated on the same `turn_cycle_complete or terminated` condition reward computation already uses. `legal_moves_count`/`bonus_chain_count` needed no new env state — both were already exposed on every `step()` call; the evaluation loops just needed to sample them at the right granularity (every step; and the value just before a chain resets to 0, respectively).

**Multi-seed support**, `experiment_alpha_comparison.py` and `experiment_grid.py`: new `--seeds` flag (`nargs="+"`, **default `[42]`** — preserves each script's original single-seed behavior exactly; unlike Phase 3's `--pool-size` default bump, running N seeds literally multiplies training wall-clock cost by N, which shouldn't silently change for existing callers). Each config's per-seed `ExperimentResult`s aggregate into a new `AggregatedResult` (mean/std/CI + the raw per-seed results nested for auditability). "Best config" selection (`experiment_grid.py` previously did a raw `max(results, key=lambda r: r.win_rate)`; `experiment_alpha_comparison.py` had no selection at all) now goes through `stats.rank_by_mean_with_ci`: the top config is only reported as confirmed-best when its CI doesn't overlap the runner-up's; otherwise both scripts say so explicitly rather than silently picking one. `experiment_grid.py`'s per-experiment model filenames gained a `_seed{N}` suffix (a mechanical necessity — multiple seeds of the same config would otherwise overwrite each other's checkpoint file).

51 new tests: `parchis/tests/test_stats.py` (29, one function per formula plus edge cases — 0/n and n/n Wilson bounds, CI overlap, aggregation), `parchis/tests/test_elo.py` (15), `parchis/tests/test_elo_ladder.py` (5, driven with tiny real trained checkpoints), 3 new scripted-dice tests in `parchis/tests/test_env.py` for the new `info` fields, `parchis/tests/test_evaluate.py` (3, real tiny-model runs through both evaluation entry points and both the random-opponent and self-play-opponent code paths), and `parchis/tests/test_experiment_alpha_comparison.py` / `test_experiment_grid.py` (4 + 5, aggregation/CI-gating logic driven directly against synthetic per-seed results, not full training loops — matching `test_selfplay_pool_callback.py`'s established "call the function directly, skip the full loop" convention for speed).

---

## Phase 5 — Empirical validation (tooling prepared; real runs not yet executed)

**Scoping decision, made before detailing this phase**: the literal pre-Phase-1 environment (old six-again-less dice loop, old 79N+8 observation) cannot be reconstructed — this project has no git history, Phase 1's rewrite was an unconditional in-place change with no legacy toggle, and every checkpoint on disk already uses the current 352-dim observation (confirmed via `docs/README_ENVIRONMENT.md`'s own compatibility warning and by loading every checkpoint's `observation_space` directly). Building a `legacy_mode` toggle to chase that literal comparison was considered and rejected — real new engineering scope, and academic, since Phase 1 fixed genuine bugs rather than offering an alternative valid design. **"Baseline vs redesigned" instead means**: a simple default recipe (`train_ppo.py`, random opponents, out-of-the-box settings) vs. the Phase 2-4 recipe (self-play opponent pool + reward-shaping config chosen by a real multi-seed sweep), both trained on today's (already-fixed) environment — exactly what the Elo ladder and multi-seed infrastructure were built to answer, with zero legacy-reconstruction code.

Also confirmed before detailing: nobody had run `experiment_grid.py`/`experiment_alpha_comparison.py` at real scale before this (only smoke-tested at a few hundred to a few thousand timesteps) — real runs cost roughly 1-2 hours per 1M timesteps on this hardware. Full rigor was chosen over a quick single-seed pass: a real 3-seed screening sweep to pick the redesigned recipe's config, then a real 3-seed-each baseline-vs-redesigned training run, then an Elo-ladder comparison. Total estimated cost of that pipeline is **~15-25 hours of training** — genuinely not a one-sitting run, so **only the tooling and runbook were built in this pass; no training was started.**

**What's ready now**:
- **New module `parchis/evaluation/group_comparison.py`**: `aggregate_group_win_rate(pairings, group_a_names, group_b_names)` (pure function, mirrors `stats.py`/`elo.py`'s style) pools every cross-group pairing from a completed `elo_ladder.py` run into one group-vs-group win rate + Wilson CI — the Elo ladder's per-pairing output doesn't itself answer "is redesigned better than baseline" as one number; this does, correctly handling either pairing order (`round_robin_pairings` shuffles order, so a cross-group pairing might list either group first) and raising `ValueError` on overlapping group names or zero matching cross-group pairings (e.g. a name typo). Plus a thin CLI (`--results-json`/`--group-a`/`--group-b`) that loads `elo_ladder.py`'s `results.json` directly.
- **`scripts/run_phase5.sh`**: the full runbook as three shell functions (`stage1`/`stage3`/`stage4` — no `stage2` command, since picking the winning config from Stage 1's results is a manual read-and-edit-this-file step in between), with every exact command, flag, and path from this plan. Not run as part of building it — `bash scripts/run_phase5.sh` with no argument just prints usage and exits.
- **9 new tests** (`parchis/tests/test_group_comparison.py`): pairing-pooling logic (both orderings, within-group/unrelated pairings correctly excluded, overlap/empty-input `ValueError`s), plus one end-to-end test that trains tiny real throwaway models, runs a real (seconds-long) `elo_ladder.py` session, and confirms `group_comparison`'s CLI works against its actual `results.json` output — catching any field-name mismatch now, before this is ever pointed at the real multi-hour Phase 5 output.

**Still to do, deliberately deferred** (requires explicit go-ahead given the wall-clock cost): run `scripts/run_phase5.sh stage1`, read its results and fill in `REDESIGNED_REWARD_TYPE`/`REDESIGNED_ALPHA`, run `stage3`, then `stage4` — the actual "did this work" answer.
