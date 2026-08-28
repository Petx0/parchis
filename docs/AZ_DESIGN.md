# AlphaZero-style rebuild — measured numbers

Running record of what was actually measured during `docs/AGENT_REBUILD_PLAN.md`, as distinct
from what was planned there. Append to this file at the end of each phase/gate; don't edit past
entries except to correct an error (note the correction inline).

Machine: Apple Silicon Mac (this environment), Python 3.12.4, macOS (Darwin 24.6.0).
`docs/AGENT_REBUILD_PLAN.md` targets an M4 Mac specifically — numbers below are from whatever
machine ran the work and are not claimed to be M4 numbers unless stated.

---

## Phase 0

### Item 1 — `Game.snapshot()`/`restore()` (2026-08-25)

**Correctness.** `parchis/tests/test_snapshot.py`, 6 tests, all passing:

- `test_snapshot_restore_matches_deepcopy_over_10000_random_positions` — 10,000 sampled
  decision points across many games, `num_players` randomized over {2, 3, 4} per game, each game
  run to completion. At every sample: `snapshot()` → real `play_turn()` mutation → `restore()`,
  compared against an independent `copy.deepcopy()` taken at the same pre-mutation instant.
  Comparison scoped to exactly the fields the plan specifies (`board.positions`, every piece's
  `position/in_base/finished/move_order`, `board.move_counter`, `current_player_idx`,
  `turn_number`, `game_over`, `winner`), keyed by `(color, piece_id)` rather than object identity
  so a live `Game` and a `deepcopy` of it compare on content. **0 mismatches over 10,000 samples.**
- `test_restore_mutates_pieces_in_place_preserves_identity` — confirms `restore()` writes onto
  the same `Piece`/`Player` objects rather than replacing them (the property the design leans on
  to avoid ever copying a `Piece`).
- Three deterministic anchors for cases a random walk could rarely hit: a capture (piece sent to
  base and back), a piece finishing (the `Board.move_piece` special case where a finished piece is
  *not* added to `board.positions`), and a game-ending move (`game_over`/`winner` round-trip).
- `test_restored_game_can_continue_playing` — structural smoke test that `restore()` leaves
  `board.positions` as genuinely mutable lists, not e.g. the tuples `snapshot()` builds
  internally.

**Performance** (`copy.deepcopy` vs. `snapshot()`/`restore()`, mid-game positions, 2000/20,000
calls respectively after warm-up — see the one-off benchmark script used for this run, not
committed to the repo):

| Quantity | 2p (turn 40) | 4p (turn 70) |
|---|---|---|
| `copy.deepcopy(game)` | 69.2 µs | 119.4 µs |
| `game.snapshot()` | 2.0 µs | 3.3 µs |
| `game.restore(snap)` | 1.7 µs | 3.0 µs |
| snapshot+restore round trip | 3.8 µs | 6.4 µs |
| **deepcopy / round trip** | **18x** | **19x** |

Both absolute numbers and the speedup ratio differ from §1.1's placeholder figures (`deepcopy`
104 µs; tuple snapshot+restore 1.0 µs + 1.2 µs ⇒ ~100x) — those were evidently a rougher/earlier
measurement or a different machine, not a reproducible target. The *decisive-win* conclusion the
plan draws from them holds either way: an 18-19x reduction in per-node state-management cost is
already large enough that it, not `deepcopy`, stops being the bottleneck once real search logic
sits on top of it in Phase 1. No micro-optimization (the plan's `__slots__` / flat-array
fallbacks) was attempted here — out of scope for items 1-2, and not obviously needed yet.

### Item 2 — Fix §1.3 opponent-observation-perspective bug (2026-08-25)

**Fix.** `ParchisEnv._get_observation(perspective_seat=None)` — the own-piece feature block (and
the `capture_opportunity` score derived from it) is now built from `perspective_seat` when given,
defaulting to `self.agent_player_idx` (unchanged behavior for `reset()`/`step()`'s own calls).
`ParchisSelfPlayEnv._choose_opponent_move` now passes `perspective_seat=seat` (the seat actually
deciding), which was the missing wiring described in §1.3.

**Confirmed the bug was real and pervasive before fixing it**: reverted the fix, re-ran the new
tests, both failed — the self-play wiring-level test alone found the opponent's fed observation
disagreeing with its actual own-piece state on **1135 of ~1145 opponent decisions** in a single
400-step episode. This was not a rare edge case; it fired almost every time the acting opponent's
piece layout differed at all from the learning agent's. Re-applied the fix afterward; both tests
pass.

**Tests added:**
- `parchis/tests/test_observation.py::test_get_observation_perspective_seat_overrides_default` —
  unit-level: scripts two seats into provably different piece layouts, asserts each
  `perspective_seat` sees only its own.
- `parchis/tests/test_selfplay.py::test_opponent_model_observation_reflects_acting_players_own_pieces`
  — wiring-level: instruments a fake opponent model to check, on *every* `predict()` call across a
  full episode, that the observation matches the actually-acting player's live piece state.

**Full suite**: 210 pre-existing tests still green, +8 new (6 snapshot/restore, 1 observation, 1
self-play) = **218 passed**, 31.5s.

### Item 3 — `parchis/evaluation/duplicate.py` (CRN/seat-swap paired matches) (2026-08-25)

Built as a prerequisite for Phase 1's item 10 gate (which needs Wilson-CI'd duplicate-pair
matches). Generalizes the classic 2-player duplicate pair (same dice seed, seats swapped) to
`num_players` seats: one group = `num_players` games all on the same seed, rotating which seat
`agent_a` occupies. `parchis/tests/test_duplicate.py`, 4 tests:

- Self-play sanity check: under `agent_a is agent_b`, every duplicate group awards `a` **exactly
  one** win, deterministically, for any `num_players` — because the whole group shares one seed
  and one policy on both sides, the same physical seat wins every rotation, and `a` occupies each
  seat exactly once. A genuine zero-variance property of CRN pairing, not a bug.
- `measure_variance_reduction`: initially designed around comparing two single-run Wilson-interval
  *widths* — wrong, since `wilson_score_interval` only ever sees one aggregate `(wins, n)` count
  and can't distinguish correlated (paired) trials from independent ones. Corrected to the
  statistically right quantity: repeat the whole evaluation procedure many times (fresh top-level
  seed each time) under both protocols, over the same per-repeat game budget, and compare the
  **empirical standard deviation of the resulting win rate across repeats**. Confirmed empirically:
  self-play gives `duplicate_std == 0.0` exactly (the property above) vs. real sampling spread for
  the independent-seed method (multiplier: infinite); two distinct agents (`TUNED_WEIGHTS` vs.
  `DEFAULT_WEIGHTS`) gave a genuine, finite **effective-n multiplier** confirming duplicate pairing
  is a real, measurable win on non-degenerate comparisons, not just the self-play edge case.
- Pooling correctness: `play_duplicate_match`'s aggregate wins/CI is exactly
  `wilson_score_interval` over the pooled `(wins, n)` from its own groups (no separate/drifted
  computation).

`ratings.py` (Bradley-Terry) is deferred — item 10's gate only needs a single A-vs-B Wilson CI, not
cross-agent ratings.

### Item 4 — `parchis/agents/heuristic.py` (handcrafted linear-score agent) (2026-08-25)

10 features per candidate move (capture value weighted by the captured piece's own progress,
enters-from-base, progress gained, lands-safe, lands-in-threat, forms-a-blocking-blockade,
exact-finish, home-column-advance, leading-opponent-suppression, develops-most-behind-own-piece),
each computed with the real rule engine via `Game.snapshot()/restore()` around a hypothetical
`execute_move` rather than hand-rolled distance math. `parchis/tests/test_heuristic.py`, 6 tests:
move-scoring never mutates the game, always returns a legal move, and two isolated-feature checks
(capture_value prefers the more-advanced victim; lands_in_threat avoids a directly-capturable
square) confirm each feature's direction independently of the others.

**CEM tuning** (`cem_tune_weights`, opponents = {`DEFAULT_WEIGHTS` heuristic, random},
`num_players=2`, population=20, games/candidate=60, generations=10, seed=20260825): mean population
score rose 0.534 → 0.777 over 10 generations (~128s, ~94 games/sec average). Compared the final
generation's population **mean** against the single best-ever-sampled **individual** on 300 held-out
games each (fresh seeds): both performed comparably vs. `DEFAULT_WEIGHTS` (has 95%-CI overlap), the
mean edged ahead vs. random (88.3% vs. 83.3%) — picked the mean as `TUNED_WEIGHTS` for its lower
per-candidate evaluation noise. Final held-out numbers: **88.3% [84.2%, 91.5%] vs. random**,
**62–66% vs. `DEFAULT_WEIGHTS`** across two independent seeds — "clearly above an untuned one and
far above random," as the plan expects.

### Item 5 (the Phase 0 GATE) — not measured; superseded by Phase 1's own gate

The literal throughput gate (depth 1/2/3 search benchmark, ≥200 games/sec at depth 1, "randomly
initialized net") was never run as originally specified. By the time `search.py` existed
(Phase 1), a *real* hand-built evaluator (item 4) was also available, so Phase 1's item 10 gate —
which measures search quality, a strictly more informative signal than raw throughput on a
meaningless untrained net — was run directly instead. Throughput numbers were still measured
along the way (see Phase 1 below) and are well below the ≥200 games/sec target as literally
stated, but item 10's actual result shows this doesn't matter in practice: see below.

---

## Phase 1

### Item 6 — `parchis/az/encoding.py` (canonical path-relative encoding) (2026-08-25)

`parchis/tests/test_encoding.py`, 7 tests. Actual sizes: **216 floats (2p), 298 (3p), 380 (4p)**
— §2.1's own estimate ("2p ≈ 220, 4p ≈ 430") was explicitly approximate; the 4p gap is because none
of the 40 "own piece" features scale with `num_players`, only the track/home/per-seat blocks do.

**Colour-invariance test caught a real bug during development**: encoding a hand-scripted position
and its 17-square rotation (colours permuted YELLOW→BLUE→RED→GREEN→YELLOW) did *not* produce
byte-identical arrays. Root cause: the per-seat `mean_path_progress` scalar initially reused
`parchis.rl.rewards.calculate_normalized_progress`, which divides each piece's **raw absolute**
board position by 76 — correct enough for its original reward-delta use, but not colour-invariant,
since Blue/Red/Green's own paths wrap the 1-68 boundary partway through (Blue's own path runs
22→68→1→17, not monotonically increasing in absolute terms) while Yellow's doesn't (5→68 directly).
Fixed by computing progress via the same relative (per-owner-start) path-step transform the
own-piece block already used. Both the hand-scripted rotation test and a second version rotating an
organically-reached random position (all 4 colour rotations) now pass byte-for-byte, and the 100k
random-state bounds property test (all finite, within [0,1]) passed in 13s.

### Item 7 — `parchis/az/net.py` (dual-head net, torch + numpy paths) (2026-08-25)

`parchis/tests/test_net.py`, 4 tests. NumPy and torch forward paths agree to **< 1e-5** (both heads)
across `num_players` ∈ {2,4} and batch sizes {1, 8, 64}, on randomly-initialized weights.

### Item 8 — `parchis/az/search.py` (expectimax over decision/chance nodes) (2026-08-25)

`parchis/tests/test_search.py`, 6 tests — all 5 correctness properties Part 3 item 8 specifies,
plus one more found along the way:

1. depth=1 with a progress-only value function exactly reproduces greedy-progress move choices.
2. `_chance_node`'s 6-way average matches an independently-computed brute-force mean.
3. Reversing `Game.get_legal_moves`' return order changes neither the value vector nor the chosen
   move (after excluding genuine value-ties, e.g. two base pieces entering via the same roll land
   on the identical square).
4. A hand-built double-capture chain (capture → 20-square bonus → second capture) is found at
   depth=2 and missed at depth=1, using an evaluator engineered to tie on its dominant term at
   depth=1 and get decided by a misleading tiebreaker — exactly the failure mode depth=1's
   "evaluate immediately, don't resolve the bonus" design implies.
5. `search()` never mutates the real `Game` (full-state fingerprint before/after, depths 1-3).

**Real bug found and fixed while sizing item 10's gate**: a "no legal move" decision (e.g. a player
with pieces stuck in base needing a 5) and a three-sixes-penalty transition both advanced to the
next decision/chance node *without consuming a depth unit* — reasonable-looking, since chance nodes
themselves are deliberately free in this design (see search.py's module docstring), but wrong for
these two specific transitions: exact expectimax must also explore the branch where a player
*never* rolls the value they need, for arbitrarily long, which a single real dice sequence never
has to confront but exhaustive enumeration does. Result: `RecursionError` (confirmed via a live
stack trace showing the exact same 4-function cycle repeating with `depth` never decreasing).
Fixed by making an empty decision cost a depth unit exactly like a real one; added
`test_no_legal_move_chains_do_not_recurse_without_bound` as a regression test, and confirmed by
reverting the fix that it fails (`RecursionError`) without it.

**Measured throughput** (2p, mid-game position, a real trained-shape but randomly-initialized net
evaluator via `NumpyAZNet`, unbatched — one leaf-at-a-time forward pass, no batching layer built
this session — see below):

| Depth | ms/decision | decisions/sec |
|---|---|---|
| 1 | 0.72 | 1388 |
| 2 | 8.27 | 121 |
| 3 | 185.9 | 5.4 |

Well below the Phase 0 gate's literal "≥200 games/sec at depth 1" target once converted to
games/sec (~9/sec at depth 1, 2p, ~150 decisions/game) — `encoding.encode()`'s own cost
(~127µs/call, dominated by `_capture_threat_scores`' `get_legal_moves` calls) dominates over the
net forward pass itself. The doc's own "leaves are collected and evaluated in one batched forward
pass per search, not one at a time" optimization was **not built this session** — item 8's
checklist only requires the 5 correctness properties above, which don't depend on batching, and
item 10's gate (below) used the much cheaper hand-built evaluator, not a net, sidestepping the
encoding cost entirely. Revisit batching before Phase 3 (self-play generation) needs real
throughput.

### Item 9 — `parchis/az/agent.py` (search agent, arena wiring) (2026-08-25)

`parchis/tests/test_agent.py`, 3 tests, including a regression test that instruments
`search.search` across 30 real games (with real bonus chains) and confirms every call resolving a
bonus has `pending_bonus` set and `roll=None` — never a stale prior dice value carried over, the
exact bug §1.4 documents in `mcts.py`. Arena wiring required zero changes to `arena.py` itself
(same `factory(game, seat, roll_box)` convention as `parchis/search/agents.py`); confirmed via
`arena.play_one_game` driving the new agent through complete games. `depth=0` is deliberately out
of scope for this module (see its docstring) — `parchis.agents.heuristic`'s own `choose_move` *is*
the "no search" baseline. Full `ladder.py` (fixed rungs, `leaderboard.json`) deferred — not needed
for item 10.

### Item 10 (the Phase 1 GATE) — PASSED (2026-08-25)

`heuristic (TUNED_WEIGHTS) + depth 2` vs. `heuristic (TUNED_WEIGHTS) + depth 0`, using
`agent.heuristic_position_evaluator` (a hand-built, non-learned evaluator: tanh of each seat's
own-vs.-mean-opponent relative progress, generalizing `parchis/search/heuristic_eval.py`'s existing
Phase-A shape to a full per-seat vector) as the search's leaf evaluator — satisfying the gate's own
stated intent ("if search doesn't help a hand-built evaluator, it will not help a learned one").

400 duplicate pairs (800 games total, `parchis.evaluation.duplicate.play_duplicate_match`,
`num_players=2`, seed=20260825), run in 41.1s (~19.5 games/sec):

| | |
|---|---|
| wins_a / n_games | 523 / 800 |
| win_rate_a | **65.4%** |
| Wilson 95% CI | **[62.0%, 68.6%]** |
| pair record | 172 pairs search did strictly better · 179 split · 49 baseline did better |

**Gate: PASS** — Wilson lower bound (62.0%) is clearly clear of 50%, on the full ≥400-pair budget
the gate calls for, not a reduced one. Search measurably and decisively improves the hand-built
heuristic's own play.

---

## Phase 2 (2026-08-26)

### Scale: 20,000 games, not ~200k

Item 11 asks for ~200k generated games. This session used **20,000** instead: at the measured
~50 games/sec (2p, heuristic-pool games, encoding recorded per decision), 200k games would take
~65 minutes of generation alone, plus the resulting ~31M decisions (~27 GB as dense float32 arrays)
exceed what's sensible to hold in memory or push through in one session on one machine -- the plan
itself anticipates this at full scale ("the loop is embarrassingly parallel at the game-generation
level... N preemptible CPU workers writing shards to GCS", Part 4). 20,000 games (3,142,248
decisions, 2.71 GB, 400s to generate) was chosen as a size that fits comfortably in memory
end-to-end on this machine while still being large enough to bootstrap a real net. Scaling up is a
config change (`n_games`), not new code -- `parchis/az/train.py`'s array-native
`bootstrap_train_arrays`/`split_indices_by_game` path was written specifically so a
disk-scale dataset never has to round-trip through a per-decision Python dict list.

### Item 11 — generation + training infrastructure

Built:
- **`parchis/az/turn_context.py`** (`TurnContextTracker`) -- the bonus-vs-fresh-roll-and-own-six-
  streak detection `parchis/az/agent.py` already needed (§1.4's bug) was needed *again*, verbatim,
  by `selfplay.py`'s recording layer. Refactored out of `agent.py` into its own module rather than
  duplicated a second time (both live under `parchis/az/`, so the import is the natural
  direction) -- `agent.py`'s own tests still pass unchanged after the refactor.
- **`parchis/az/selfplay.py`** (`generate_games`, `examples_to_arrays`) -- plays games from a pool
  of `{TUNED_WEIGHTS heuristic, ε-noisy heuristic (new: `heuristic.DEFAULT_EPSILON=0.15`,
  `make_epsilon_noisy_heuristic_agent_factory`), random}`, one member sampled per seat per game,
  recording every decision's encoding + chosen move + (once the game concludes) outcome. Reuses
  `arena.play_one_game` rather than a custom loop.
- **`parchis/az/config.py`** (`BootstrapConfig`) -- one dataclass, written to `runs/<name>/config.json`.
- **`parchis/az/train.py`** (`split_by_game` / `split_indices_by_game`, `bootstrap_train` /
  `bootstrap_train_arrays`, `save_checkpoint`) -- AdamW, cosine LR, weight decay (Part 4's table),
  early stopping on validation loss. Split at the **game level**, never per-decision: decisions
  from the same game are highly correlated (they share, or nearly share, the same outcome), so a
  naive per-row split would leak a "held-out" game's likely result into training. 80/10/10
  train/val/test by game.

10 new test files / additions (`test_turn_context.py`, `test_selfplay_generation.py`,
`test_train.py`, `test_calibration.py`, plus a `heuristic.py` addition) -- 26 new tests, all green.

### The bug: value targets stored in absolute seat order, not mover-relative

`parchis.az.encoding.encode(game, seat, ...)` builds every per-seat block (track/home/per-seat
scalars) in an order **relative to `seat`** -- channel 0 is always "the observer's own", channel k
is the seat k turns after the observer (`_ordered_seats`). The first training run stored
`outcome[i]` as a plain **absolute**-seat one-hot (`outcome[winner_seat] = 1.0`) and used the exact
same vector for every decision in a game regardless of who was actually deciding. Net effect: the
net was trained to map "an encoding centered on whoever is moving" to "a label centered on seat 0"
-- consistent-*looking* (both are length-`num_players` vectors) but genuinely meaningless, since
the same input pattern needed a different "correct" channel-0 answer depending on which absolute
seat happened to be moving, which the (deliberately relative) encoding cannot tell the net.

**How it surfaced**: the first trained net's value loss barely moved off `ln(2)` across 19 epochs
(0.694 -> 0.701, drifting the *wrong* way) while policy loss improved normally -- and item 13's
gate came back at a decisive **11.4%** win rate (Wilson upper bound 13.8%) for net@depth1 against
the plain tuned heuristic, not a mere "not-yet-better" result. That magnitude of loss (not ~45%,
literally 8x worse than break-even) was the signal that something was structurally wrong, not just
undertrained -- confirmed by inspecting `agent.py`'s `NetEvaluator` (which correctly remaps
relative-\>absolute via `np.roll(probs, observer_seat)` on the assumption the net's channel 0
means "observer's own") against `selfplay.py`'s target construction (which did not apply the
matching remap when *building* that same channel 0's training label).

**Fix**: `generate_games` now remaps the game's absolute outcome into **each decision's own**
mover-relative order before storing it: `example['outcome'] = np.roll(absolute_outcome,
-example['mover_seat'])` -- the exact inverse of `NetEvaluator`'s own roll, and consistent with
`_ordered_seats(observer_seat, N) = [(observer_seat+k) % N ...]`. Applied post-hoc to the
already-generated dataset (remapping `value_targets` using the already-recorded `mover_seats`
array -- `policy_targets`/`X`/`game_indices` were never affected, so no need to regenerate the
400s of games) rather than regenerating. `parchis/evaluation/calibration.py`'s extraction
simplified to match (`value_targets[:, 0]` directly, no more indexing by absolute `mover_seats`).

Three tests added to `test_selfplay_generation.py` specifically for this (`test_outcome_backfill_
matches_actual_winner_per_game` rewritten, `test_game_index_covers_every_game_with_a_consistent_
implied_winner` rewritten, `test_outcome_is_mover_relative_not_absolute` new) -- confirmed by
reverting the fix that all three fail without it.

**Before/after (same architecture, same data, only the target convention fixed)**:

| | value_loss @ epoch 0 | value_loss trend | item 13 win rate |
|---|---|---|---|
| Buggy (absolute targets) | 0.694 (~ln 2, uninformative) | rose to 0.701 over 19 epochs | 11.4% [9.4%, 13.8%] |
| Fixed (relative targets) | 0.522 | fell to ~0.52, stable | 48.0% [44.6%, 51.5%] |

### Item 12 — calibration gate: PASSED

Final model (`runs/bootstrap_2p_v3`, `value_loss_weight=2.0` -- see below): **ECE = 0.0145** on
314,487 held-out decisions from 2,000 entirely-held-out games (< 0.05 threshold). Predictions span
the full range with roughly even bucket populations (23k-44k per decile), not a collapse onto the
base rate:

| bucket (mean predicted) | n | actual frequency |
|---|---|---|
| 0.040 | 37,865 | 0.049 |
| 0.150 | 23,057 | 0.168 |
| 0.251 | 23,698 | 0.275 |
| 0.352 | 28,099 | 0.374 |
| 0.456 | 42,428 | 0.461 |
| 0.546 | 41,165 | 0.537 |
| 0.648 | 30,398 | 0.633 |
| 0.749 | 25,736 | 0.724 |
| 0.850 | 24,707 | 0.828 |
| 0.960 | 37,334 | 0.949 |

### Item 13 — net@depth1 vs. tuned heuristic: MARGINAL, not a confident pass

After the bug fix, 400 duplicate pairs at equal loss weighting (`value_loss_weight=1.0`) gave
48.0% [44.6%, 51.5%] -- a near-tie, but the point estimate sat just under 50%. Since the policy
head's loss runs larger and improves faster on the shared trunk than the value head's, added a
`value_loss_weight` parameter to `bootstrap_train_arrays` (default 1.0, preserves old behavior)
and retrained at `value_loss_weight=2.0`. This did not change the value head's *achieved* loss
(~0.52 either way -- two separate runs plateaued at the same value, pointing at dataset
size/capacity as the binding constraint, not epochs or loss weighting), but did shift the
resulting policy slightly: 400 pairs gave 53.1% [49.7%, 56.6%], and a second, independent 800-pair
run gave 51.4% [49.0%, 53.9%]. **Combined (1,200 pairs, 2,400 games): 52.0% win rate, Wilson 95%
CI [49.9996%, 54.0%]** -- the lower bound is a hair under 50%, essentially indistinguishable from
break-even at this sample size, leaning positive.

**Honest read**: this is not the confident "beats tuned heuristic" result item 13 asks for, but it
is a dramatic, real improvement over the pre-fix 11.4% -- net@depth1 is now genuinely competitive
with, and probably marginally better than, the heuristic used directly. Not chased further to a
clean pass in this session: the plateaued value_loss across both training runs is the more
informative signal here than the win-rate noise, and it points at **more generated games** (the
20k-vs-200k gap above) as the most likely lever, not more epochs or another loss-weight sweep.
Concrete next step for whoever continues this: regenerate at a larger `n_games` (the code already
supports it unchanged) before re-attempting item 13, and only then move to Phase 3.

### Follow-up (2026-08-26): full-scale run (200,000 games) — item 13 now PASSES

The reduced-scale write-up above ended with a specific, testable hypothesis: value_loss had
plateaued at ~0.52 across two training runs (equal weighting and `value_loss_weight=2.0` alike),
which pointed at dataset size/capacity as the binding constraint rather than epochs or loss
weighting. This follow-up tests that hypothesis directly by generating the full ~200k games item 11
originally asked for.

**New infrastructure needed.** 200k games produces ~31M decisions, ~26 GB as dense float32 arrays --
too large to hold as one in-memory array (this machine has 24 GB RAM) or to generate in one
unsharded pass. Built:
- A resumable sharded generation script: games generated in shards of 10,000, each shard's
  `generate_games` + `examples_to_arrays` output written to its own `.npz`, with a `manifest.json`
  (atomic write) tracking which shards are done so an interrupted run resumes without regenerating
  completed shards.
- `parchis/az/train.py`: **`split_shards(shard_paths, train_frac, val_frac, seed)`** -- splits at
  the *shard* level rather than the per-game level (`split_indices_by_game`'s existing scope); still
  game-independent (each shard is many whole games) since games only get coarser-grained, not
  reshuffled across shards. Requires >= 3 shards and guarantees train/val/test are **all**
  non-empty regardless of how skewed the requested fractions are -- a 3-way split with an empty
  group is useless, and unlike the per-game split's existing rounding, shard counts here are small
  (e.g. 20) so naive rounding can genuinely zero out a group.
- **`bootstrap_train_sharded(train_shard_paths, val_shard_paths, ...)`** -- loads validation shards
  once into memory up front, then streams training shards one at a time (freshly, in a reshuffled
  order) per epoch, so peak memory is O(one training shard) + O(validation set), never O(whole
  corpus).
- `test_train_sharded.py` (6 tests): split correctness/no-leakage, the non-empty-groups guarantee
  across shard counts 3-20, rejection of < 3 shards or invalid fractions, and a regression guard
  (via a `_load_shard`-call-counting monkeypatch) that a training shard is loaded exactly once per
  epoch -- never the whole training corpus concatenated up front, which would defeat the point of
  sharding.
- **Bug found via dry run** (4 synthetic shards before committing to the real 70-minute
  generation): `split_shards`'s first rounding formula could produce zero validation shards at
  small `n`, crashing `_load_and_concat_shards`'s `np.concatenate` on an empty list. Fixed by the
  non-empty-groups guarantee above; re-verified by re-running the same dry run to completion.

**Generation**: 200,000 games / 20 shards of 10,000 games each / 31,416,380 total decisions / 70.0
minutes (~47.9 games/sec, consistent with the 20k run's measured rate) / 26 GB on disk / 0 truncated
games (every game reached a real winner within `max_turns=500`).

**Training**: `split_shards` (seed=0) → 16 train / 2 val / 2 test shards. Same architecture as
before (hidden=(256,256), `value_loss_weight=2.0` carried over from the 20k-game experiment), 40
epochs, patience=6 -- early stopping never triggered; val_loss kept improving, if slowly, through
epoch 40. 2,553s total (63.8s/epoch) on this machine's MPS backend.

**Gate 12 (calibration)**, on 3,139,397 held-out decisions from the 2 entirely-held-out test
shards: **ECE = 0.0021** — roughly 7x better than the 20k-game run's 0.0145 — with large,
roughly-even bucket populations (228k-449k per decile), again real discrimination across the full
range, not collapse to the base rate:

| bucket (mean predicted) | n | actual frequency |
|---|---|---|
| 0.038 | 354,569 | 0.042 |
| 0.150 | 227,942 | 0.152 |
| 0.251 | 245,171 | 0.252 |
| 0.352 | 277,710 | 0.354 |
| 0.457 | 435,730 | 0.457 |
| 0.543 | 448,874 | 0.543 |
| 0.648 | 296,814 | 0.650 |
| 0.749 | 251,133 | 0.747 |
| 0.850 | 243,291 | 0.846 |
| 0.960 | 358,163 | 0.956 |

**GATE 12: PASS** (ECE = 0.0021 < 0.05 threshold).

**Gate 13** (`net@depth1` vs. `heuristic(TUNED_WEIGHTS)@depth0`), 800 duplicate pairs (1,600 games,
seed 20260828): **win_rate_a = 0.6138 (982/1,600)**, Wilson 95% CI **[0.5896, 0.6373]** -- lower
bound decisively clear of 50%. Pair record: `{a_better: 314, split: 354, b_better: 132}`.

**GATE 13: PASS** (win_rate_a = 0.6138, Wilson lower bound = 0.5896).

**Confirms the reduced-scale run's hypothesis, with a wrinkle**: the value head's raw cross-entropy
loss at convergence looks similar either way (~0.50-0.51 in both the 20k and 200k runs -- not the
"loss finally drops a lot" story one might naively expect from 10x the data). What moved sharply
with more data was **calibration** (ECE 0.0145 → 0.0021, ~7x better) and, downstream of that, the
actual gate that matters: the duplicate-match win rate went from a marginal 52.0% (CI lower bound a
hair under 50%) to a decisive 61.4% (CI lower bound 59%). Lesson for next time: raw value
cross-entropy magnitude was a misleading proxy for "is this net good enough to search on top of" --
calibration error and the downstream duplicate-match gate are the numbers that actually tracked
search strength, and both moved together in the direction the more-data hypothesis predicted.

**Checkpoint**: `runs/bootstrap_2p_v4_large/` (config.json, metrics.jsonl, model.pt,
test_shards.json) — supersedes `runs/bootstrap_2p_v3/` as the project's current-best checkpoint.
Phase 2 is complete; Phase 3 (the continuous self-play loop) is next.

## Phase 3 (2026-08-26)

The continuous self-play loop (docs/AGENT_REBUILD_PLAN.md Part 3 Phase 3): each round generates
games with the current champion (root exploration), warm-start retrains on a recency-windowed
replay buffer, and promotes only on a confirmed win over the champion. Seeded from Phase 2's
`runs/bootstrap_2p_v4_large/` checkpoint rather than starting over.

### New infrastructure

- **`parchis/az/targets.py`** -- pure functions turning one `search.search()` call's output into
  Phase 3's two SOFT training targets: `blend_value_target` (`z_value = (1-lambda)*outcome +
  lambda*root_value`, §1.6's variance-reduction fix for "value targets from a single rollout are
  hopeless in a dice game") and `policy_target_from_move_values` (`z_policy` = masked softmax over
  root move values, §2.3). Also the self-play ACTING policy: `anneal_temperature` (temperature
  1.0 → 0.25 over the first 15 plies) and `dirichlet_mixed_probs` (AlphaZero-style root noise) --
  deliberately a SEPARATE, noisier distribution from `z_policy`'s own un-perturbed one (training the
  net to imitate its own exploration noise, rather than its own judgment, would be a real bug).
- **`parchis/az/champion_pool.py`** -- the `{champion, last 4 promoted, tuned heuristic, random}`
  pool. `build_pool` returns raw nets (for generation's own exploration-aware wrapping) separately
  from the two hand-built anchor factories (never search-capable, used as-is). A small on-disk FIFO
  (`append_promoted`, capped at 4) tracks promoted-checkpoint history across rounds/process restarts.
- **`parchis/az/selfplay.py`**: `generate_round_games` -- the champion occupies a randomly-chosen
  seat EVERY game (guarantees at least one recorded seat per game); every other seat samples the
  pool independently, and ANY search-capable seat that gets sampled (champion again, or a promoted
  net) is also recorded -- true self-play, both sides can contribute. `round_examples_to_arrays`
  packs a SOFT `(n, 4)` policy target (unlike Phase 2's hard `(n,)` class-index one) --
  `train.py`'s `F.cross_entropy` call already dispatches on the target's own dtype/shape, so no
  training-code branch was needed, only this different packing.
- **`parchis/az/train.py`**: `init_state_dict` parameter added to `bootstrap_train_arrays` /
  `bootstrap_train_sharded` (warm-start from existing weights instead of AZNet's random init --
  Part 3's "cap warm-start epochs"; `None` preserves Phase 2's exact prior behavior). `split_shards_
  train_val` -- a simpler 2-way (no held-out test group) shard split for a round's own replay
  buffer, needing only 2 shards minimum (a round's buffer is far smaller than Phase 2's full corpus).
- **`parchis/az/round_loop.py`** (new) -- `run_round` (one round: generate → replay buffer → warm-
  start train → promote-or-not → escalate-or-not, checkpointed under `runs/<name>/rounds/round_NNNN/`
  ending in a `done.json` sentinel) and `run_continuous` (loops `run_round`, resumable via
  `find_resume_round` scanning for the first round missing its sentinel).
- **`parchis/az/config.py`**: `SelfPlayRoundConfig` dataclass, same "always saved to
  runs/<name>/config.json" pattern as `BootstrapConfig`.

43 new tests across 6 files (`test_targets.py`, `test_champion_pool.py`, `test_selfplay_round.py`,
one addition to `test_turn_context.py`, 6 additions to `test_train_sharded.py`,
`test_round_loop.py`) -- full suite: **323 passed**.

### Bug found & fixed: `TurnContextTracker`'s stale pending bonus

While testing `generate_round_games` against a real (untrained, high-exploration) net, hit a
`ValueError` from `dirichlet_mixed_probs` on an empty `move_values` after ~80 games -- `search.
search()` correctly returning "no legal move" for a state that, by every other signal, should have
had one. Root cause, in the SHARED `parchis/az/turn_context.py` (used by `agent.py`'s evaluation-time
search agent too, not just this new code): a bonus (capture/finish) can have **zero legal moves**
(e.g. a +10 finish bonus when every other piece is still in base), in which case
`Game._execute_bonus_move` (game.py) returns without EVER calling `player.choose_move` --
`TurnContextTracker.record_move` never runs to clear `self._pending_bonus` for that attempt, so the
NEXT real decision (a genuinely fresh roll, possibly the same player on six-again, possibly the next
player) was incorrectly still reported as "continuing that already-resolved bonus".

This is exactly the class of bug the project's rebuild set out to catch -- **silent, not a crash**,
in every OTHER caller: `agent.py`'s search agent would get a stale `pending_bonus`, `search.search()`
would (correctly, given that bad input) return `(None, {}, draw_vector)`, and the wrapping
`choose_move` would return `None` despite `Game.play_turn()`'s own `legal_moves` (computed
correctly, for the real fresh roll) being non-empty -- `Game.play_turn()` treats a `None` return
exactly like "no legal move" (game.py:454's `if chosen_move:`), silently forfeiting a turn the
player should have been able to play. Phase 3's generation is simply the first caller to exercise
this path hard enough (real exploration + an untrained net wandering into unusual states) to
surface it loudly instead of silently. It's rare enough that it evidently never visibly corrupted
Phase 1/2's own gate numbers (both used `agent.py`'s search agent extensively), but "rare and
silent" is precisely the failure mode this whole rebuild exists to eliminate, so it was fixed at the
source rather than worked around locally.

**Fix, attempt 1 (wrong, shipped briefly, caught by the round loop itself)**: `context_for` took
`(game, seat, roll_box)` and self-healed by re-checking `game.get_legal_moves` -- the exact same
check `Game._execute_bonus_move` itself uses -- before trusting a remembered pending bonus. This
passed the full test suite and a real-checkpoint smoke run, so the round loop was launched on it.
It crashed for real about 21 minutes in, partway through round 2's generation (rounds 0 and 1 had
completed cleanly in ~546s and ~559s respectively), on a `StopIteration`: a sampled move's piece_id
didn't match any entry in the real legal-moves list.
Root-caused via a deterministic replay of the exact failing seed: the re-check is **unsound**,
because board state keeps changing turn to turn -- "is a move of this exact SQUARE COUNT
coincidentally legal RIGHT NOW" can come back true for a totally unrelated reason long after the
real bonus was already silently resolved with nothing played. That's exactly what happened: a
genuinely fresh entry-roll decision (roll=5) got mis-served a stale `finish_bonus=10` context
because some on-board piece happened to have an unrelated, coincidentally-legal +10 move by then.

**Fix, attempt 2 (correct)**: predict the right answer AT THE MOMENT it matters, instead of
re-guessing later. `record_move` now simulates the move it's about to record (snapshot → execute →
`game.get_legal_moves` → restore -- the same "peek at a hypothetical future state, then undo it"
technique `search.py`/`heuristic.py` already use elsewhere) and only sets `_pending_bonus` if that
simulation shows the resulting bonus will genuinely have a legal move -- exactly mirroring what
`Game._execute_bonus_move` will do for real a moment later, with no ambiguity about *when* the
check applies. `context_for`'s signature reverted to plain `(roll_box)` (the fix no longer needs
`game`/`seat` there at all). Verified: the exact seed that crashed round 2 now runs clean through
2,000 games; an additional 8,000 games across 4 fresh seeds also ran clean; the regression test
(`test_finish_bonus_with_no_legal_move_is_never_recorded_as_pending`) confirmed failing against
attempt 1's code and passing against attempt 2's; full suite re-verified at 323 passed both times.

### Design choices the plan left unspecified

The plan gives exact numbers for `lambda` (0.5) and the acting-policy temperature schedule
(1.0 → 0.25 over ~15 plies), but leaves several mechanics to the implementation:

- **`tau_target`** (the policy TRAINING target's own temperature, separate from the acting
  policy's): no number given. Chose **0.25** -- matching the acting schedule's own "settled" end
  value, on the theory the target should reflect the same sharpness self-play converges to once
  exploration cools down, rather than an independently-guessed number.
- **Dirichlet noise**: no numbers given. `epsilon=0.25` (AlphaZero's own published mixing weight);
  `alpha=1.0`, scaled from AlphaZero's own chess/go values by this game's much smaller ~2.76
  legal-moves-per-decision branching factor (§1.1) rather than reusing either constant directly.
- **Pool sampling / which seats get recorded**: the plan's "opponents sampled from {...}" phrasing
  (singular "opponents" against an implied "self") was read as: the champion always occupies one
  randomly-chosen seat (recorded), every OTHER seat samples the full pool independently (also
  recorded when it lands on a search-capable member) -- guarantees zero wasted games, while still
  allowing true both-sides self-play when opponent sampling lands on a net.
- **Replay buffer mechanism**: "last ~3 rounds" implemented as -- each round's games saved as
  multiple shards (matching Phase 2's on-disk shape); a round's training step reads its own shards
  plus every shard from the previous `replay_window_rounds - 1` rounds (never deleted, just never
  read again once outside the window), split train/val via the new `split_shards_train_val`.
- **Escalation reset**: after `escalate_after_failures` (3) consecutive non-promotions, exactly the
  NEXT round runs at `escalation_depth` (2); `consecutive_failures` resets to 0 after that round
  REGARDLESS of whether it promoted, giving base-depth a full fresh run of attempts before
  escalating again (not specified numerically by the plan).
- **Warm-start epoch cap**: `warm_start_max_epochs=5`, `patience=2` -- much smaller than Phase 2's
  40/6, since each round warm-starts from already-reasonable weights on a small recent-rounds
  buffer (nudging, not re-learning from scratch).
- **Initial `max_rounds` target**: 40 -- inside the plan's own stated "interesting region around
  20-50 rounds", not unbounded; `run_continuous` is resumable, so this is a starting checkpoint to
  reassess at, not a hard ceiling on the project.

### Generation throughput: measured, and n_games_per_round scaled down accordingly

Mirroring Phase 2's own "measure before committing to scale" precedent: Phase 3's generation calls
a real `search.search()` (and therefore a net forward pass, ~3 leaves at depth=1) for every
recorded decision, unlike Phase 2's pool generation (hand-scored heuristics/random, no net calls at
all). Measured **10.71 games/sec** at depth=1 with the actual trained `bootstrap_2p_v4_large`
checkpoint (300 games, 32,040 recorded decisions, 28.0s) -- about **4.5x slower** than Phase 2's
~48 games/sec. At the plan's literal ~50k games/round this would be ~78 minutes of generation alone
per round, making even 20 rounds a multi-day commitment. Scaled `n_games_per_round` down to
**6,000** (3 shards of 2,000 games, ~10 minutes of generation per round) so a practical number of
rounds (into the plan's own "interesting region") completes in a practical amount of wall-clock
time. Batched leaf evaluation (one net forward pass per search, not per leaf -- flagged as deferred
"until Phase 3's self-play generation actually needs the throughput" back in Phase 0/1) is the
obvious next lever if 6,000/round proves too slow in practice; not implemented yet since generation
at this size is workable without it.

### Status and results

Launched 2026-08-26 against `runs_dir` in the scratchpad (not the project's own `runs/` -- the
per-round shards are large enough over many rounds that writing them into the project directory,
which iCloud Drive syncs, would be disruptive; only the small current-champion checkpoint gets
copied into the project's `runs/` once Phase 3 has a new best worth keeping, mirroring how
`bootstrap_2p_v4_large` itself was promoted). Config: `run_name=selfplay_2p_v1`, 2p,
`n_games_per_round=6000`, `hidden_sizes=(256,256)`, `value_loss_weight=2.0`,
`warm_start_max_epochs=5`, `promotion_n_pairs=600`, `max_rounds=40`.

Round-by-round results (updated as rounds complete):

| round | depth | win_rate_a | 95% CI | promoted? | consecutive_failures after | time |
|---|---|---|---|---|---|---|
| 0 | 1 | 0.5008 | [0.4726, 0.5291] | no | 1 | 546.5s |
| 1 | 1 | 0.5042 | [0.4759, 0.5324] | no | 2 | 559.2s |
| 2 | 1 | 0.5217 | [0.4934, 0.5498] | no | 3 | 581.9s |
| 3 | **2 (escalated)** | 0.5175 | [0.4892, 0.5457] | no | 0 (reset) | **7348.1s (~2.04h)** |
| 4 | 1 | 0.5408 | [0.5126, 0.5689] | **YES** | 0 | 588.8s |
| 5 | 1 | 0.4892 | [0.4610, 0.5174] | no | 1 | 646.9s |
| 6 | 1 | 0.5350 | [0.5067, 0.5631] | **YES** | 0 | 647.7s |
| 7 | 1 | 0.5058 | [0.4776, 0.5341] | no | 1 | 677.5s |
| 8 | 1 | 0.5217 | [0.4934, 0.5498] | no | 2 | 674.6s |
| 9 | 1 | 0.4908 | [0.4626, 0.5191] | no | 3 | 670.9s |
| 10 | **2 (escalated)** | 0.5150 | [0.4867, 0.5432] | no | 0 (reset) | **8739.9s (~2.43h)** |
| 11 | 1 | 0.4850 | [0.4568, 0.5133] | no | 1 | 673.7s |
| 12 | 1 | 0.5092 | [0.4809, 0.5374] | no | 2 | 683.0s |
| 13 | 1 | 0.5233 | [0.4950, 0.5515] | no | 3 | 682.0s |
| 14 | **2 (escalated)** | 0.4917 | [0.4635, 0.5199] | no | 0 (reset) | **8788.0s (~2.44h)** |
| 15 | 1 | 0.4842 | [0.4560, 0.5124] | no | 1 | 674.6s |
| 16 | 1 | 0.4958 | [0.4676, 0.5241] | no | 2 | 665.6s |
| 17 | 1 | 0.5208 | [0.4925, 0.5490] | no | 3 | 650.7s |
| 18 | **2 (escalated)** | 0.4933 | [0.4651, 0.5216] | no | 0 (reset) | **8862.4s (~2.46h)** |
| 19 | 1 | 0.5108 | [0.4826, 0.5390] | no | 1 | 683.2s |
| 20 | 1 | 0.4917 | [0.4635, 0.5199] | no | 2 | 687.9s |
| 21 | 1 | 0.5242 | [0.4959, 0.5523] | no | 3 | 683.2s |
| 22 | **2 (escalated)** | 0.5200 | [0.4917, 0.5482] | no | 0 (reset) | **8694.7s (~2.42h)** |
| 23 | 1 | 0.5308 | [0.5025, 0.5589] | **YES** | 0 | 644.1s |
| 24 | 1 | 0.4825 | [0.4543, 0.5108] | no | 1 | 658.9s |
| 25 | 1 | 0.4983 | [0.4701, 0.5266] | no | 2 | 656.9s |
| 26 | 1 | 0.4875 | [0.4593, 0.5158] | no | 3 | 659.3s |
| 27 | **2 (escalated)** | 0.4867 | [0.4585, 0.5149] | no | 0 (reset) | **8833.8s (~2.45h)** |
| 28 | 1 | 0.4842 | [0.4560, 0.5124] | no | 1 | 687.5s |
| 29 | 1 | 0.5083 | [0.4801, 0.5365] | no | 2 | 687.8s |
| 30 | 1 | 0.4808 | [0.4527, 0.5091] | no | 3 | 685.0s |
| 31 | **2 (escalated)** | 0.5183 | [0.4900, 0.5465] | no | 0 (reset) | **9020.8s (~2.51h)** |
| 32 | 1 | 0.4950 | [0.4668, 0.5233] | no | 1 | 669.4s |
| 33 | 1 | 0.4783 | [0.4502, 0.5066] | no | 2 | 665.4s |
| 34 | 1 | 0.4850 | [0.4568, 0.5133] | no | 3 | 675.4s |
| 35 | **2 (escalated)** | 0.4950 | [0.4668, 0.5233] | no | 0 (reset) | **9168.5s (~2.55h)** |

| 36 | 1 | 0.4808 | [0.4527, 0.5091] | no | 1 | 695.7s |
| 37 | 1 | 0.4575 | [0.4295, 0.4858] | no | 2 | 677.2s |
| 38 | 1 | 0.4867 | [0.4585, 0.5149] | no | 3 | 669.5s |
| 39 | **2 (escalated)** | 0.4883 | [0.4601, 0.5166] | no | 0 | 9250.4s (~2.57h) |

### Run complete: 40/40 rounds (initial target reached)

`run_continuous` finished cleanly (exit code 0). Final state: `{'round': 39, 'promotions': 3,
'consecutive_failures': 0}`. Promoted history (3 entries -- every promotion ever made, since 3 < the
4-slot cap): rounds 4, 6, 23's candidates.

**Final tally**: 40 rounds, ~99,117s (**~27.5 hours**) total wall-clock, **3 promotions** (rounds 4,
6, 23 -- current champion is round 23's candidate). Split by round type:

| | rounds | promotions | promotion rate | total time |
|---|---|---|---|---|
| base-depth (1) | 31 | 3 | ~9.7% | ~20,410s (~5.67h) |
| escalated (2) | 9 | **0** | **0%** | ~78,707s (~21.86h) |

**Escalation never once promoted, across all 9 attempts (rounds 3, 10, 14, 18, 22, 27, 31, 35, 39),
while consuming ~79% of total wall-clock time.** This is now a complete run's worth of evidence, not
an early/small-sample pattern -- base-depth rounds promoted roughly 1 in 10 tries; escalated rounds
promoted 0 in 9, at ~13x the per-round cost. Combined with the structural read from round 18 (an
escalated round's promotion match pits the candidate against a champion that ALSO gets to search
deeper, raising the bar exactly when the mechanism is trying to clear it), the honest conclusion is
that `escalate_after_failures=3` / `escalation_depth=2` as configured is not paying for itself in
this run, and shouldn't be re-enabled unchanged for a continuation without addressing that -- e.g.
evaluating an escalated round's candidate against the champion at base_depth instead, so the test
isolates "did the depth-2 data help" from "is the champion also playing better right now".

Net progress over the whole run: the champion went from the Phase 2 bootstrap checkpoint to a net
that beat it (round 4), beat that (round 6), and beat that (round 23) -- each step individually
Wilson-CI-confirmed at >=600 duplicate pairs. Not yet re-measured against the Phase 2 gate's own
fixed benchmark (tuned-heuristic @ depth 0) to quantify total improvement since Phase 2 in one
number, comparable to the 61.4% Phase 2 finished at.

**Checkpoint**: round 23's candidate (the current champion) copied to
`runs/selfplay_2p_v1_champion/` (model.pt, config.json, champion_meta.json,
promoted_history.json) in the project's own tracked `runs/` -- supersedes `runs/bootstrap_2p_v4_large/`
as the project's current-best checkpoint. The bulky per-round shard data stays in the scratchpad
only, per the same iCloud-sync reasoning as the rest of Phase 3.

### Escalation fix (2026-08-27): decoupling generation depth from evaluation depth

Root cause (see round_loop.py's module docstring for the full account): the promotion match used
the SAME `depth` variable for both generation and evaluation, so an escalated round's candidate-vs-
champion comparison ran BOTH sides at `escalation_depth` -- meaning the champion also got to search
deeper during that specific evaluation, not just the candidate's training data. That confounds "did
depth-2 training data help" with "is the champion also playing better right now", and the full
40-round run's own numbers made the cost concrete: 9/9 escalated rounds failed to promote (~79% of
total wall-clock time), while 3/31 base-depth rounds (never confounded, since depth == base_depth
there already) succeeded.

**Fix**: `run_round`'s promotion step now always evaluates at `eval_depth = cfg.base_depth`,
independent of `depth` (still `escalation_depth` on an escalated round, still used for GENERATION
only). `promotion_result.json` now records both `generation_depth` and `eval_depth` separately
(previously a single `depth` field) so this stays visible/auditable per round going forward.

**Verified**: `test_escalation_triggers_after_n_failures_and_resets` (test_round_loop.py) rewritten
to assert `eval_depths == [1, 1, 1]` across a 3-round forced-failure sequence where
`generation_depths == [1, 1, 2]` -- confirmed failing (`eval_depths` came back `[1, 1, 2]`) against
the reverted pre-fix code, passing against the fix. Full suite re-verified green after restoring.

Not yet re-run against real data -- the next escalation (whenever training resumes and 3 more
consecutive base-depth failures accumulate) will be the first real test of whether escalation
actually helps once the confound is removed.

### Benchmark: Phase 3 champion vs. the pre-Phase-3 (Phase 2) checkpoint

Direct net-vs-net duplicate match, both @ depth=1 (matching Phase 2's own Gate 13 depth and the
vast majority of Phase 3's own working promotion gates), 800 pairs / 1,600 games, seed 20260827:

**win_rate (Phase 3 champion) = 0.5581 (893/1,600), Wilson 95% CI [0.5337, 0.5823]** -- lower bound
decisively clear of 50%. Pair record: 226 pairs Phase 3 did strictly better, 441 splits, 133 Phase 2
did strictly better (of the non-split pairs, ~63% favor Phase 3). 120.7s for 1,600 games (13.3
games/sec).

**Verdict: the 40-round Phase 3 run produced a genuinely, decisively stronger net than Phase 2's
bootstrap** -- the self-play loop is working, net of the escalation mechanism's own separate
problems (which contributed none of this gain -- all 3 promotions behind it came from base-depth
rounds). +5.8pp over the seed checkpoint in ~5.67 hours of base-depth-round compute (excluding the
~21.86 hours spent on escalation, which produced none of it) is a reasonable per-hour return for a
first 40-round pass, though this is a single-checkpoint-pair comparison, not yet a trend across
multiple champions.

**36 rounds in (5 short of the 40-round initial target), ~24.4 hours elapsed, 3 promotions**
(rounds 4, 6, 23; champion is round 23's candidate). **All 8 escalated rounds so far have failed to
promote**, now totaling ~19.3 hours -- **~79% of all elapsed time** -- for zero promotions, while
the 28 base-depth rounds (~5.1 hours combined) produced all 3.

**2 promotions in 19 rounds** (rounds 4 and 6); current champion is still round 6's candidate.
Elapsed for rounds 0-18: ~43,362s (~12.0 hours).

**All 4 escalated rounds so far (3, 10, 14, 18) have failed to promote.** These 4 rounds alone
total ~33,738s (~9.37 hours) -- **~78% of all elapsed wall-clock time** -- for zero promotions,
while the 15 base-depth rounds (~5.6 hours combined) produced both of the run's 2 promotions.

**A likely structural reason, not just bad luck**: an escalated round's promotion match pits the
candidate against the champion **at the SAME escalated depth** (`round_loop.py` passes the same
`depth` to both `candidate_factory` and `champion_factory`). So escalation doesn't just give the
candidate better training data -- it ALSO hands the champion a stronger search-time opponent to
beat, in the same round. The two effects pull against each other: any value-function improvement
from depth-2 data has to be large enough to overcome an old champion that's ALSO now searching
deeper, which is a harder bar than depth=1's "improved value function vs. an unchanged shallow
search" comparison. That's a real hypothesis worth testing, not a confirmed diagnosis -- I haven't
verified it beyond re-reading the code path.

Not changing anything without your input, but this now warrants a decision rather than continued
deferral -- options, roughly in order of how much they change: (a) leave it running as-is and see
if a 5th escalation eventually breaks through; (b) evaluate an escalated round's promotion at
base_depth instead of escalation_depth, isolating "did the training data help" from "is the
champion also playing better"; (c) raise `escalate_after_failures` so this fires less often; (d)
disable escalation entirely for now given the cost/benefit so far. Say the word on any of these.

**Escalation cost, measured**: round 3's depth=2 round took **7,348s (~2.04 hours)** end-to-end vs.
~550-680s (~10-11 min) for every depth=1 round -- roughly **12x**, not the naive ~18x leaf-count
ratio (training + promotion overhead dilutes it slightly, but it's still overwhelmingly generation-
bound). This was NOT accounted for in the original "6-10 hours for 40 rounds" estimate, which only
measured depth=1 throughput. With 2 escalations already in 10 rounds (and a 3rd about to start),
total wall-clock for 40 rounds is realistically **15-25+ hours**, not 6-10 -- revised honestly here
rather than left standing. Elapsed so far (rounds 0-9): ~12,942s (~3.6 hours).

Rounds 0-1 took 546.5s / 559.2s (~9.1 / ~9.3 min) end-to-end -- both essentially a dead-even tie
against the seed champion, unsurprising for a single 5-epoch warm-start nudge. Round 2 then hit the
`TurnContextTracker` bug above (crashed ~21 min after launch); after the fix, verification (2,000
games replaying the exact crashing seed + 8,000 more across 4 fresh seeds, all clean; full suite
323 passed) and re-launch, the loop **resumed cleanly at round 2** (round_loop.py's own
checkpointing meant rounds 0-1's results and the unchanged champion were never at risk -- only
round 2's partial, uncommitted shards were regenerated). See the escalation-cost note above for the
revised (larger) wall-clock estimate for the full 40-round target.

### Deferred out of Phase 0/1 (explicitly, not silently)

- ~~`parchis/evaluation/ratings.py` (Bradley-Terry Elo MLE)~~ — **built 2026-08-28**, see "Ladder +
  ratings tooling" below.
- ~~`parchis/evaluation/ladder.py` (fixed rungs + `leaderboard.json`)~~ — **built 2026-08-28**, see
  below (as `runs/pairings.jsonl`, not `leaderboard.json` — see that section for why).
- Batched leaf evaluation in `search.py` (one forward pass per search, not per leaf) — a
  performance optimization the doc calls for, not a correctness requirement item 8's tests check;
  still deferred as of 2026-08-28 (sized as its own follow-up session, not undertaken yet — see
  "Pre-Phase-4 review" below).
- The literal Phase 0 item 5 throughput gate (≥200 games/sec at depth 1 on a randomly-initialized
  net) — superseded by running Phase 1's item 10 gate directly once it existed (see item 5 above).

### Pre-Phase-4 review (2026-08-28): what's actually pending before "2p clears the ladder"

Before starting Phase 4 (4-player extension), reviewed the whole plan (`AGENT_REBUILD_PLAN.md`,
this file, `SEARCH_MCTS.md`, `CODE_REVIEW.md`, `RULES.md`) for pending items. Findings:

- **Two `CODE_REVIEW.md` items checked and found already resolved**, not by this pass but by
  earlier refactors: `env.py`'s "magic number 76" complaint — every literal `76` left in the file
  is in a docstring/comment, not code; and `evaluate.py`'s bare `except:` swallowing mask-read
  errors — `evaluate.py` no longer reads masks itself at all, it calls the shared `mask_fn`
  (`parchis/training/common.py:25`), a plain dict lookup with nothing to swallow.
- **Phase 4's own "4-seat duplicate-match protocol" concrete-work item is already done**, ahead of
  schedule: `duplicate.py`'s `play_duplicate_group`/`play_duplicate_match` were built generally
  from the start (their own docstring says so) and already rotate the tested agent through every
  seat at `num_players > 2`, not just 2.
- **Tactical puzzle suite** (§5.4, `parchis/evaluation/puzzles/`) — still not built; the user is
  building this one by hand (curating/judging 40-60 positions needs real Parchís expertise, not
  something to delegate).
- **Ladder + ratings tooling and batched leaf evaluation** — see below and the deferred-list update
  above.

### Ladder + ratings tooling (2026-08-28)

Built `parchis/evaluation/ladder.py` + `parchis/evaluation/ratings.py` — the actual mechanism
"no 4p work until 2p clears the ladder" (`AGENT_REBUILD_PLAN.md` Part 6) refers to, which didn't
exist until now. **Not** built on the existing `elo_ladder.py`/`multiplayer_matrix.py`: those wrap
`evaluate_agent()`, which is MaskablePPO-`.load()`-only — can't host a search-driven AZ agent,
which needs live search at inference time (the same reason `arena.py` exists as its own parallel
tool, per that module's docstring). Built on `duplicate.py`'s `play_duplicate_match` instead
(already CRN-variance-reduced, already generalizes past 2 players).

`ladder.py`: round-robins named "rungs" (any arena-style factory — random, heuristic, or a
`checkpoint:<run_dir>[:depth=N]` spec loaded via `checkpoint_loading.load_agent_numpy_net`) via
`play_duplicate_match`, appending one JSON line per pairing to `runs/pairings.jsonl` —
project-wide and append-only, per §5.2's spec, not per-run. (The spec also mentions a
`leaderboard.json`; skipped for now since `pairings.jsonl` plus `ratings.py`'s own on-demand fit
already answers "who's ahead" without a second, easily-stale cached artifact — can add one later
if actually wanted.)

`ratings.py`: fits Bradley-Terry ratings by maximum likelihood over the whole accumulated
`pairings.jsonl` (`scipy.optimize.minimize`, L-BFGS-B, on the pairwise negative log-likelihood),
anchored at `rating_random = 0`. Replaces `elo.py`'s order-dependent sequential K-factor updates
with one number per agent comparable across the whole project's history, exactly as §5.3 asks.
Percentile bootstrap for CIs, resampling pairing records (not individual games) with the anchor
held fixed across every replicate — letting each replicate pick its own anchor would silently
compare ratings against different zero points and contaminate the CIs.

**A shared `parse_spec` grammar** (`parchis/agents/agent_spec.py`, extracted from
`play_instrumented_game.py`'s `--agent` flag, which had grown this exact grammar first) is now used
by both the visualization CLI and the ladder's `--rung` flag — one spec syntax
(`checkpoint:<run_dir>[:depth=N] | heuristic:tuned|default | random`) instead of two independently-
drifting ones.

**Verified end to end** with real agents, not just synthetic data: `random` / `heuristic:default` /
`heuristic:tuned` / the Phase 3 champion checkpoint, run as two separate `ladder.py` invocations
into the same `pairings.jsonl` (confirming accumulation works across runs, not just within one) —
`ratings.py` recovered the exact known strength ordering: `az_champion (2.63) > heuristic_tuned
(1.78) > heuristic_default (1.25) > random (0.0)`. Synthetic-data unit tests
(`test_ratings.py`) separately confirm the MLE fit recovers known planted ratings within 0.05 at
large sample size, independent of real self-play noise. Full suite: 365 passed.

### Benchmark: Phase 3 champion vs. tuned heuristic @ depth 0 (Gate 13's own protocol) (2026-08-28)

Re-ran Phase 2's exact Gate 13 protocol (`net@depth1` vs. `heuristic(TUNED_WEIGHTS)@depth0`, 800
duplicate pairs, seed 20260828) with the Phase 3 champion (`runs/selfplay_2p_v1_champion/`, round
23's candidate) instead of the Phase 2 bootstrap checkpoint — the number the "not yet re-measured"
note two sections up was waiting on.

**win_rate_a = 0.6756 (1,081/1,600), Wilson 95% CI [0.6523, 0.6981]** — decisively clear of both
50% and Phase 2's own finishing number. Pair record: `{a_better: 372, split: 337, b_better: 91}`.

**vs. Phase 2's Gate 13 result on the identical protocol: 0.6138 (CI [0.5896, 0.6373])** — the two
CIs don't overlap. First like-for-like number confirming Phase 3's self-play loop added real
strength beyond Phase 2's bootstrap net, not just against Phase 2's own checkpoint directly (the
0.5581 net-vs-net-at-depth-1 result above) but against the fixed external yardstick both phases are
actually scored against.

### Rounds 40-57 continuation: final results — stopped by decision, 0 new promotions

The rounds-40-79 continuation (same run, same config, resumed with the escalation-depth fix now in
the code) ran rounds 40-57 (18 rounds: 14 base-depth, 4 escalated at rounds 43/47/51/55) before
being stopped deliberately — no new champion had been found across all 18, a real diminishing-
returns signal, not an interruption. Full per-round results:

| round | win_rate_a | CI | gen depth | promoted |
|---|---|---|---|---|
| 40 | 0.5025 | [0.4742, 0.5307] | 1 | no |
| 41 | 0.4692 | [0.4411, 0.4975] | 1 | no |
| 42 | 0.4792 | [0.4510, 0.5075] | 1 | no |
| 43 | 0.4825 | [0.4543, 0.5108] | **2** | no |
| 44 | 0.4883 | [0.4601, 0.5166] | 1 | no |
| 45 | 0.4850 | [0.4568, 0.5133] | 1 | no |
| 46 | 0.5267 | [0.4984, 0.5548] | 1 | no (closest miss) |
| 47 | 0.5150 | [0.4867, 0.5432] | **2** | no |
| 48 | 0.5025 | [0.4742, 0.5307] | 1 | no |
| 49 | 0.4717 | [0.4436, 0.5000] | 1 | no |
| 50 | 0.5017 | [0.4734, 0.5299] | 1 | no |
| 51 | 0.4667 | [0.4386, 0.4950] | **2** | no |
| 52 | 0.4792 | [0.4510, 0.5075] | 1 | no |
| 53 | 0.4908 | [0.4626, 0.5191] | 1 | no |
| 54 | 0.4925 | [0.4643, 0.5208] | 1 | no |
| 55 | 0.5142 | [0.4859, 0.5424] | **2** | no |
| 56 | 0.4758 | [0.4477, 0.5041] | 1 | no |
| 57 | 0.4733 | [0.4452, 0.5016] | 1 | no |

**Escalation, with the confound fix applied, still didn't pay off**: all 4 escalated rounds (43,
47, 51, 55) failed to promote, `eval_depth` correctly staying at `1` (base depth) on every one —
confirming the confound is genuinely gone (previously `eval_depth` silently matched
`generation_depth`, i.e. `2`, every time) — but that alone wasn't enough to make escalation
productive. Combined with the original run's 0/9, **escalation is now 0/13 across this run's entire
history**, both before and after the confound fix. This is a real, decision-worthy result: the fix
was correct and necessary (it removed a genuine measurement bias), but it wasn't sufficient to make
`escalate_after_failures=3` / `escalation_depth=2` pay for itself at these settings. Recommend
disabling escalation (or trying a materially different configuration — a shallower/cheaper
escalation depth, or a much higher failure threshold so it fires rarely) for any future continuation
of this lineage, rather than re-enabling it unchanged.

**Final state**: 58 rounds total (0-57), **3 promotions** (rounds 4, 6, 23) — unchanged from before
this continuation. The champion is still round 23's candidate
(`runs/selfplay_2p_v1_champion/`, already the project's tracked checkpoint — no update needed).
Stopped cleanly: round 57 was allowed to finish before killing the process, so no partial/corrupted
round data; round 58 has an incomplete `shards/` directory with no `done.json`, which
`find_resume_round` will correctly treat as not-yet-done and regenerate from scratch if this run is
ever resumed.

### Ladder run (2026-08-28): "2p clears the ladder" — decisive full round-robin

With training stopped and the champion final (round 23's candidate, unchanged since the original
40-round run), ran the first real, decisive ladder pass -- the actual evidence
`AGENT_REBUILD_PLAN.md`'s Phase 4 gate ("no 4p work until 2p clears the ladder") asks for, not an
ad hoc one-off comparison. 5 rungs, full round-robin (10 pairings), 600 duplicate-pairs each (1,200
games/pairing, 12,000 games total): `random`, `heuristic_default`, `heuristic_tuned`,
`phase2_bootstrap` (`runs/bootstrap_2p_v4_large`, depth=1), `phase3_champion`
(`runs/selfplay_2p_v1_champion`, depth=1).

**Where each participant's "intelligence" actually comes from**, plain English, weakest to
strongest:

- **`random`** — no intel at all. Picks a uniformly random legal move every time
  (`Player.choose_move`'s own default).
- **`heuristic_default`** — trying to maximize a hand-picked linear score over 10 features
  (`parchis/agents/heuristic.py`: capture value, escaping base, own progress, landing safety,
  avoiding capture threats, forming a blockade, finishing exactly, home-column advance, suppressing
  whoever's currently leading, developing its most-behind piece). The *signs* are reasoned about by
  hand ("capture is good, walking into a threat is bad"); the *magnitudes* are untuned guesses,
  never checked against real games.
- **`heuristic_tuned`** — the exact same 10-feature formula and the same hand-reasoned signs, but
  with the magnitudes fit by CEM (the cross-entropy method: sample many candidate weight vectors,
  play them against a mix of the untuned heuristic and random, keep refitting toward whatever
  scored best) — same "understanding" of what matters, better calibrated on how much each thing
  should matter.
- **`phase2_bootstrap`** — a small neural network's value estimate plus 1 ply of lookahead search
  (`parchis/az/search.py`). The network itself was trained once, by ordinary supervised learning,
  on ~200,000 games generated by `heuristic_tuned` (with some random exploration mixed in) — so its
  "intel" is a distillation of heuristic-level play into a neural net, not anything it worked out
  for itself.
- **`phase3_champion`** — the same network-plus-1-ply-search architecture, but the network was
  produced by ~58 rounds of genuine AlphaZero-style self-play starting from `phase2_bootstrap`'s
  own net: repeatedly generate games (against itself, past versions of itself, and the
  heuristic/random pool), train a candidate on that self-generated data, and only keep the update
  if it decisively beats the incumbent in a rigorous statistical test (`parchis/evaluation/duplicate.py`).
  Its "intel" comes from iterative self-play improvement, not from imitating another player.

**Bradley-Terry ratings** (anchored at `random = 0`, 500-replicate bootstrap CIs):

| participant | rating | 95% CI |
|---|---|---|
| phase3_champion | 2.955 | [2.723, 3.420] |
| phase2_bootstrap | 2.721 | [2.515, 3.123] |
| heuristic_tuned | 2.154 | [1.984, 2.586] |
| heuristic_default | 1.540 | [1.355, 2.006] |
| random | 0.000 | [0.000, 0.000] |

**A complete, statistically-supported strength chain**, every adjacent gap confirmed by that
pairing's own Wilson CI (not just the aggregate rating): `heuristic_default` clearly beats `random`
(79.5%), `heuristic_tuned` clearly beats `heuristic_default` (64.4%), `phase2_bootstrap` clearly
beats `heuristic_tuned` (62.9%), and `phase3_champion` clearly beats `phase2_bootstrap` (55.8%, CI
[52.9%, 58.5%] on the win-rate scale -- entirely above 50%). This is the first time this project has
had a full, one-shot, internally-consistent ranking of its entire capability history in a single
run, rather than a chain of separate pairwise benchmarks run at different times with different
protocols.

**Cross-validation against independent, earlier measurements** (same checkpoints, different
benchmark runs, different seeds/pair counts) landed within noise of each other every time:
`phase2_bootstrap` vs. `phase3_champion` gave 44.2% here → phase3_champion at 55.8%, matching the
dedicated Phase-3-vs-Phase-2 benchmark's 55.81% almost exactly; `heuristic_tuned` vs.
`phase3_champion` gave 31.2% here → phase3_champion at 68.8%, matching the Gate-13-protocol rerun's
67.56% closely. Three independently-run measurements agreeing this tightly is strong evidence the
ladder/ratings implementation itself is correct, not just that the numbers look plausible in
isolation.

**Full 5×5 cross matrix, empirical** (row's win probability against column, from the actual 12,000
games played; each cell is exactly complementary to its mirror across the diagonal — e.g.
55.8% + 44.2% = 100% — since a duplicate match plays the SAME games from both sides, not two
separate samples):

| beats ↓ / loses to → | phase3_champion | phase2_bootstrap | heuristic_tuned | heuristic_default | random |
|---|---|---|---|---|---|
| **phase3_champion** | — | 55.8% | 68.8% | 78.9% | 96.8% |
| **phase2_bootstrap** | 44.2% | — | 62.9% | 75.7% | 95.5% |
| **heuristic_tuned** | 31.2% | 37.1% | — | 64.4% | 89.0% |
| **heuristic_default** | 21.1% | 24.3% | 35.6% | — | 79.5% |
| **random** | 3.2% | 4.5% | 11.0% | 20.5% | — |

**Full 5×5 cross matrix, Bradley-Terry model's implied probabilities** (`sigmoid(rating_i -
rating_j)` from the fitted ratings above — the fully self-consistent, transitive version the model
predicts, as opposed to each pairing measured independently above):

| beats ↓ / loses to → | phase3_champion | phase2_bootstrap | heuristic_tuned | heuristic_default | random |
|---|---|---|---|---|---|
| **phase3_champion** | — | 55.8% | 69.0% | 80.5% | 95.1% |
| **phase2_bootstrap** | 44.2% | — | 63.8% | 76.5% | 93.8% |
| **heuristic_tuned** | 31.0% | 36.2% | — | 64.9% | 89.6% |
| **heuristic_default** | 19.5% | 23.5% | 35.1% | — | 82.3% |
| **random** | 4.9% | 6.2% | 10.4% | 17.7% | — |

The two matrices agree closely everywhere (within 1-2pp on most cells, up to ~4pp on the
widest-margin ones like `phase3_champion` vs. `random`) — expected, since the model fit is a joint
least-surprise summary across all 10 pairings at once, while the empirical table is each pairing
measured independently. The gaps involving `random` are the largest because those probabilities sit
near the extremes of the sigmoid curve, where a small rating shift moves the predicted probability
more.

**Caveat, stated honestly**: the top two ratings' bootstrap CIs (`phase3_champion` [2.723, 3.420]
vs. `phase2_bootstrap` [2.515, 3.123]) overlap somewhat, even though their direct pairwise win-rate
CI does not — a known Bradley-Terry property (the joint MLE fit pools information across every
rung's connections, which can widen a top rung's marginal uncertainty even when its single most
relevant pairwise comparison is unambiguous). The direct pairwise result is the more decisive
evidence for "is the champion better than the previous checkpoint specifically"; the rating is the
right number for "how does everything compare to everything else at once."

**Verdict: 2p clears the ladder.** `runs/pairings.jsonl` now holds this run's 10 pairings as a
permanent, project-wide record (fixed `.gitignore`'s `runs/` rule to `runs/*/` so this file — small,
meant to accumulate forever — doesn't get swept into the same ignore rule as the bulky per-run
checkpoint directories).
