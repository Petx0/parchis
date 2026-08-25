# Observation Space Changes — Design Notes for Implementation

Source: `parchis/rl/env.py` (`ParchisEnv._get_observation()` or equivalent), current
layout documented in `docs/README_ENVIRONMENT.md`. Size *before* the decisions below:
`79*num_players + 36`. These decisions have since shipped -- the current size, matching
`docs/README_ENVIRONMENT.md` and the code today, is `79*num_players + 31` (see the
"Running total" note at the end of this file).

This document records **decisions only** (reviewed by a Parchís domain expert against
the design at the time). Each entry references the block number/name from
`docs/README_ENVIRONMENT.md` for traceability.

---

## Decision 1 — Block 5: Bonus indicator

**Current:** 1 value, `pending_bonus['squares'] / 20.0` — continuum-encoded, but only
ever takes 3 actual values: `0.0` (no bonus pending), `0.5` (10-square finish bonus),
`1.0` (20-square capture bonus). `pending_bonus['squares']` is never summed across a
chain — a new bonus fully replaces the pending one, so it's always exactly 0/10/20,
never higher.

**Problem:** finish-bonus and capture-bonus are qualitatively different situations
(different consequences, different follow-on chain likelihood), not degrees of the
same continuous quantity. Continuum-encoding them is inconsistent with how the dice
block (block 4) handles an analogous case — a 6-with-base-pieces vs 6-without is
given as two distinct one-hot slots, not `dice_value/6.0`.

**Decision:** Replace the single scalar with **two binary flags**:
- `has_finish_bonus` (1.0 if a 10-square finish bonus is pending, else 0.0)
- `has_capture_bonus` (1.0 if a 20-square capture bonus is pending, else 0.0)

Mutually exclusive when a bonus is pending; both `0.0` when no bonus is pending (no
separate "none" slot needed — unambiguous by construction since exactly one or
neither flag is ever set).

**Net size change:** Block 5 goes from 1 value → 2 values (**+1** to total observation
size).

---

---

## Decision 2 — Block 6: `capture_threatened` / `capture_opportunity`

**Current:** for each own piece, a single binary `capture_threatened` bit — true if
any opponent piece sits 1-6 squares behind on a capturable square. Known,
documented limitations: caps at distance 6 (misses the 7-square move), ignores the
mandatory-5-entry rule (treats every distance-5 opponent piece as a threat even when
the opponent is rules-forced to enter from base instead), doesn't count multiple
simultaneous threats, and doesn't account for bonus-chain captures (10/20 squares).

**Ground truth confirmed against `parchis/game/rules.py::RuleEngine.get_legal_moves`:**
- Rolling a 6 moves 7 squares instead of 6 iff the player has 0 pieces in base
  (`effective_roll = ALL_OUT_BONUS_ROLL`).
- Rolling a 5: entry from base is legal (and, per the mandatory-entry rule, is the
  *only* legal move that turn — no other piece may move 5) **unless** the player's
  starting square is occupied by exactly 2 of their own pieces. Every other
  occupancy state (empty / 1 own / 1 opponent / 1 own+1 opponent / 2 opponents)
  still permits — and forces — entry. So a distance-5 move-threat exists only when:
  the opponent has 0 pieces in base, **or** the opponent's starting square has
  exactly 2 of the opponent's own pieces on it.

**Decision — replace both binary flags with a combined, roll-based threat score:**

For each own piece and each opponent, check each face value `v` in `1..6` for
whether rolling `v` this turn would result in that opponent capturing this piece,
either:
- **Directly** — `v` (or the effective 7-move if `v == 6` and the opponent has 0
  base pieces) lands exactly on this piece's square, subject to the mandatory-5-entry
  exception above (a distance-5 opponent piece only counts if entry is currently
  blocked/unavailable for that opponent).
- **Via a bonus chain** — `v` produces a legal capture (of some *other* piece) or a
  finish, and the resulting bonus move (20 squares for capture, 10 for finish) would
  land on this piece. This subsumes the old "distance 10/20" idea — no separate
  conditional check needed, since a bonus is only ever reachable through a specific
  face value in the first place.

**Score:** sum the count of "hit" face values across **all** opponents (not
deduplicated — two different opponents each threatening with a 4 contributes 2, per
the "double threat = double risk" requirement). Normalize: **divide by 6 and clip at
1.0** (not divided by `num_opponents` — a single very dangerous opponent should read
as maximally dangerous regardless of player count, rather than being diluted as the
table gets more crowded).

This replaces the single `capture_threatened` bit with one continuous
`capture_threat_score` value per piece (same slot count as before — 1 value; only
the *content* changes from binary to a `[0,1]` score). `capture_opportunity` (the
mirror-image "can I capture something" feature) was not part of this decision yet —
still open (see below).

**Net size change:** Block 6 stays at 6 values × 4 pieces = 24 total; no change in
count, only in how `capture_threatened`'s single slot is computed.

**Still open on this item:** whether/how `capture_opportunity` gets the same
roll-based-score treatment (currently untouched by this decision).

---

---

## Decision 3 — Block 7: Blockade indicator

**Current:** 2 values, `own_blockades / 12.0` and `opponent_blockades / 12.0` —
raw counts of blockades on the board, own vs. opponent.

**Problem:** a blockade's only mechanical effect (`docs/RULES.md`) is blocking
movement across that square for every color — it adds no extra safety beyond what
any safe square already gives (safe squares are capture-immune regardless of
blockade status). So the raw count isn't a proxy for safety, and:
- Whether a specific legal move right now is blocked by a blockade is already fully
  captured by `action_masks` (illegal moves are masked out) — the count adds nothing
  there.
- The count isn't tied to any specific piece or decision — it's a vague
  board-wide aggregate, not something the network can act on directly.
- It also doesn't capture the one thing that *would* be tactically relevant (e.g.
  "is the agent sitting in a blockade it will be forced to break on a 6") — but that
  wasn't judged worth building as a replacement.

**Decision:** cut entirely. No replacement feature.

**Net size change:** Block 7 removed — **-2** to total observation size.

---

---

## Decision 4 — Block 9: Bonus chain count

**Current:** 1 value, `min(bonus_chain_count / 4.0, 1.0)` — count of consecutive
bonus-triggering moves (captures/finishes) within the current chain; resets to 0
whenever a move doesn't trigger a new bonus.

**Problem:** every capture/finish earlier in the chain is already reflected in the
board state (blocks 1-3 — pieces sent back to base, pieces marked finished, progress
scores). The chain-length counter doesn't add decision-relevant information beyond
what the current board state already encodes — which piece to move on the current
bonus step depends on where pieces are *now*, not on how many steps it took to get
here. It was originally added because it happened to already be computed internally
for KPI logging, not from a decision-relevant argument.

**Decision:** cut entirely. No replacement feature.

**Net size change:** Block 9 removed — **-1** to total observation size.

---

---

## Decision 5 — Block 6: `capture_opportunity`

**Current:** per own piece, a single binary bit — true if any opponent piece sits
1-6 squares ahead on a non-safe square.

**Decision — replace the 4 per-piece bits with one combined, roll-based
probability score covering all 4 of the agent's own pieces:**

For each face value `v` in `1..6`, check whether the agent has **at least one**
legal move this turn (any of its 4 pieces) that results in a capture. Count how many
of the 6 face values produce a hit (OR'd across pieces — a roll value counts once
even if multiple own pieces could capture with it), divide by 6. So: `1.0` if every
possible roll captures something regardless of which piece is chosen, `1/6` if only
one specific roll value produces a capturing move, `0.0` if none do.

**Explicitly scoped to single-roll captures only** — does not extend through bonus
chains (asymmetric with Decision 2's threat score, which does include chained
10/20-square hits; intentional per discussion, not an oversight).

**Net size change:** was 4 values (1 per piece, part of the 6-per-piece block) →
becomes 1 shared value. Block 6 goes from 24 total (6 features × 4 pieces) to 21
(5 remaining per-piece features × 4 pieces, + 1 shared opportunity score) — **-3**
to total observation size.

---

## Implementation note (applies to Decisions 2 and 5)

Both the threat score and the opportunity score should be computed by querying
`RuleEngine.get_legal_moves(player, roll_value)` for each candidate roll (1-6, and
for Decision 2's chained case, the resulting bonus's `get_legal_moves(player, 10)` /
`get_legal_moves(player, 20)`) — **not** by hand-rolling distance/modular-arithmetic
checks the way the current `capture_threatened`/`capture_opportunity` code does. The
rule engine already correctly implements every edge case that matters here
(mandatory-5-entry, entry-specific capture rules that differ from mid-board capture
rules, blockade-crossing legality, home-column boundaries) — reimplementing that
logic by hand in the observation builder is exactly the kind of duplication that's
already caused documented bugs elsewhere in this codebase (`docs/CODE_REVIEW.md`).

"Is this specific legal move a capture" needs one small **pure** (non-mutating)
helper, since `board.py`'s actual capture logic (`enter_piece`/`move_piece`) mutates
state as a side effect:
- `'enter'` moves: capture iff the starting square currently holds exactly 1 own +
  1 opponent piece, or 2 opponent pieces (mirrors `Board.enter_piece`).
- `'move'`/`'finish'` moves: capture iff the destination is not a safe square, is
  before `HOME_COLUMN_START`, and currently holds an opponent piece (mirrors
  `Board.move_piece`).

---

## Running total: observation size delta

| Decision | Block | Change |
|---|---|---|
| 1 | Bonus indicator (5) | +1 |
| 2 | `capture_threatened` (6) | 0 (content change only) |
| 3 | Blockade indicator (7) | -2 |
| 4 | Bonus chain count (9) | -1 |
| 5 | `capture_opportunity` (6) | -3 |

**Net: -5.** New total observation size: `79*num_players + 31` (was `79*num_players + 36`).

---

## Discussion complete
All flagged blocks (5, 6, 7, 9) have been reviewed. Blocks 1-4 and 8 were not raised
for discussion and are assumed unchanged.

---

## Appendix — External precedent (research only, no action items)

Reviewed for context against other RL game agents; informed the discussion above but
does not itself require any implementation.

- **TD-Gammon (backgammon)** — the closest precedent, as the other major dice-race
  game. Raw board occupancy alone reached intermediate play; adding hand-crafted
  expert features — notably **probability of a blot being hit** — pushed it to
  master level. This independently validates the roll-based `capture_threat_score`
  approach in Decision 2: real backgammon computes it over 36 two-dice combinations,
  Parchís's single die makes our "count of 6 face values / 6" the exact single-die
  analog, not an approximation.
- **AlphaZero (chess/shogi/Go)** — piece-identity-indexed planes (not just
  positional ones) validate the existing `piece_id`-indexed own-piece block. Its
  8-step history planes are used mainly for repetition/draw detection and move
  counting — Parchís has no repetition-draw rule and is fully Markov from the
  current state, so there's no equivalent need; noted here as a considered-and-
  rejected idea rather than a silent gap.
- **Ludo-specific RL literature** — a small existing body of work. One project
  independently uses booleans for "token vulnerable" / "token under attack" — same
  concept as our threat/opportunity features, just binary rather than roll-
  probability-scored. A 2026 comparative paper (DQN/PPO on Ludo) found a structured
  4-channel CNN spatial encoding (treating token positions as a spatial layout
  rather than a flat vector) outperformed a flat-vector baseline, with the CNN
  encoding alone accounting for roughly half the improvement in their ablation.

**CNN/spatial board encoding — noted as a future research direction, not scoped for
implementation:**
- Parchís's board is a 68-square ring with 4 branch-off home columns and 4 off-board
  base areas — not a uniform grid like chess/Go. A workable version would need a 1D
  circular convolution over the ring (our existing per-player 76-slot, entry-point-
  normalized channel in block 1 is already close to this shape), non-circular
  padding for the home-column tail, and a hybrid architecture (conv trunk + flat MLP
  branch for base/finished counts, dice, threat/opportunity scores, etc. — those
  have no board position and can't live in a conv layer).
- Likely smaller marginal benefit here than the cited paper's ~50% figure: that
  result was measured against a raw flat-vector baseline with no hand-engineered
  threat/blockade features, whereas this environment already hand-engineers much of
  what a CNN would otherwise learn implicitly. A CNN might still catch longer-range
  or multi-piece spatial patterns the fixed 6-square-window features don't.
- This would be a network-architecture change (custom SB3 feature extractor), not
  an observation-content change, and would likely invalidate the existing Phase 2-4
  hyperparameter tuning done around the current flat `MlpPolicy`. Out of scope for
  this round.
