# Building the strongest Parchís agent — research findings and build plan

**Status**: Phase 0, Phase 1, and Phase 2 all complete (2026-08-25 / 2026-08-26), all gates passed.
Phase 2 was first completed at reduced data scale (20k games), where item 13 came back marginal;
re-running item 11 at the full ~200k-game target on a new sharded generation/training pipeline
turned that into a decisive pass (61.4% vs. tuned heuristic, Wilson 95% CI [58.96%, 63.73%]) --
confirming the reduced-scale run's own hypothesis that dataset size, not epochs or loss weighting,
was the binding constraint. See Part 3 and `docs/AZ_DESIGN.md` for the full numbers.

Phase 3 (2026-08-26 onward): the continuous self-play round loop (`parchis/az/round_loop.py` and
supporting modules -- `targets.py`, `champion_pool.py`, `selfplay.generate_round_games`) is built,
tested, and launched, seeded from the Phase 2 checkpoint above. Found and fixed a real latent bug
in the shared `TurnContextTracker` along the way (a bonus with zero legal moves left stale state
for the next decision -- see `docs/AZ_DESIGN.md`). Round size scaled down from the plan's ~50k
games to 6,000/round based on measured throughput (searching against a real net during generation
is ~4.5x slower than Phase 2's heuristic-only generation). The initial 40-round run completed
2026-08-27 (3 promotions; round 23's candidate is the champion) and decisively cleared Phase 2's
own Gate 13 benchmark on re-measurement (67.56% vs. Phase 2's 61.38%, non-overlapping CIs -- see
`docs/AZ_DESIGN.md`). Also found and fixed a depth-confound bug in the escalation mechanism
(an escalated round's promotion match was accidentally handing the OLD champion a search-time
boost too); a rounds-40-79 continuation with the fix applied ran rounds 40-57 (2026-08-28) and was
then stopped by decision -- 18 more rounds (4 more escalations, all confound-free) produced zero
new promotions, confirming the fix was correct but not sufficient to make escalation pay for itself.
A further 10-round continuation (rounds 58-67, 2026-08-29) after fixing the color-blind
home-column bug and adding batched leaf evaluation also produced zero new promotions (escalation
now 0/16 lifetime; the champion has gone 44 rounds without a promotion, win rates a tight
0.47-0.52 band both before and after those two engine changes -- neither bug explains the plateau).
Final state: 68 rounds, 3 promotions total, champion still round 23's candidate
(`runs/selfplay_2p_v1_champion/`, unchanged). A decision on this lineage (change something
structural, or treat round 23 as its ceiling and move to Phase 4) is flagged but not yet made. See
`docs/AZ_DESIGN.md` for the full round-by-round log.

Ahead of Phase 4: `parchis/evaluation/ladder.py` + `ratings.py` (2026-08-28) now implement the
"2p clears the ladder" gate Part 6 requires before any 4p work, and a decisive full round-robin
(random / heuristic-default / heuristic-tuned / Phase 2 bootstrap / Phase 3 champion, 600 pairs
each) confirms it: a complete, CI-backed strength chain from random up through the current
champion, cross-validated against three independent earlier benchmarks -- **2p clears the ladder**.
See `docs/AZ_DESIGN.md`'s "Ladder run" entry for the full numbers. The tactical puzzle suite
(Part 5.4)'s loader/runner/CLI (`parchis/evaluation/puzzles/`) are built and tested against a CSV
schema agreed with the user, plus a visualizer (`parchis/visualization/visualize_puzzles.py`,
2026-08-29) that renders a puzzle's position, an agent's per-move evaluation, and the ground-truth
answer on the real board -- see `docs/AZ_DESIGN.md`'s "Tactical puzzle suite: loader + runner" and
"Puzzle suite visualization" entries. The user has since started filling in the real 40-60
positions (`parchis/evaluation/puzzles/my_puzzles.csv`); fixed a `;`-delimiter/BOM mismatch from the
user's spreadsheet export (now auto-detected per file) and extended the schema to accept multiple
correct answers (`correct_piece_id` may be `'/'`-separated, e.g. `2/3`) after one real puzzle turned
up genuinely needing it -- all 10 of the user's puzzles so far load and validate cleanly. A first
depth-1-5 accuracy sweep of the champion against those 10 puzzles found accuracy is **not**
monotonic in depth (30%/50%/60%/40%/60%) -- a search-pathology signature, not noise -- plus two
puzzles wrong at every depth; small-sample (n=10), flagged to revisit once the real 40-60 exist --
see `docs/AZ_DESIGN.md`'s "Puzzle accuracy vs. search depth" entry. Batched
leaf evaluation in `search.py` (2026-08-28) is now implemented: every leaf across a whole
`search()` call is collected and evaluated in one `NumpyAZNet.forward()` call instead of one per
leaf, measured 1.3x/2.0x/2.3x faster at depth=1/2/3 with no change to any move/value search() ever
returns (`test_batched_and_eager_search_agree`, `test_net_evaluator_batched_matches_eager_call_path`)
-- see `docs/AZ_DESIGN.md`'s "Batched leaf evaluation" entry.

The round-23-champion-plateau decision above is now resolved by evidence rather than left open:
a full analysis (training results + puzzle findings + a TD-Gammon/post-AlphaZero literature review)
found three independent signals -- flat validation loss for all 44 plateau rounds, never-promoted
candidates statistically tied in strength with the champion back to round 10, and a confirmed,
significant self-bias in `root_value` relative to independent rollouts (p<0.0001) -- pointing at a
training-data/target-quality plateau rather than a search-depth problem or a dead end. Plan and
full diagnostic numbers: `.claude/plans/twinkly-marinating-hinton.md` and
`docs/AZ_DESIGN.md`'s "Strength-improvement plan" entry (2026-08-29/30). Implemented, tested, AND
run for 50 rounds combined (68-117): escalation retired by default (0/16 lifetime promotions, ~79%
of wall-clock, not worth its cost); rollout-refined value targets (`parchis/az/rollouts.py`,
opt-in, cost-controlled subsampling -- caught and fixed a real 100x-oversized cost mistake on the
first attempt, see `docs/AZ_DESIGN.md`); the self-play pool broadened with a "recent" (every round,
not just promoted) checkpoint history; and the Part 4 depth/parallelism table corrected to match
actual practice. **Result: none of the three moved the needle** -- 0 promotions across all 50
rounds, and three successive verification ladders show the same candidates bouncing around a fixed
strength band with no trend, not a lineage building on itself (full numbers in
`docs/AZ_DESIGN.md`'s "Result: 50 rounds... no detectable improvement" entry). Moved to the next
lever: an auxiliary prediction head (Phase 4.1, `parchis/az/net.py`'s `aux_head`, predicting
whether each own piece finishes by game end -- free supervision, no new generation cost), with
backward-compatible loading for every existing pre-aux-head checkpoint
(`AZNet.load_state_dict_compat`) and a shard-schema migration so old shards contribute zero aux
gradient rather than a fabricated one. 433 tests passing (up from 422). Next: run a fresh round
with `aux_loss_weight` turned on (isolated from the rollout-targets variable, which showed no clear
signal) and judge the same way -- promotion rate + ladder verification, not a single round's CI.
Nothing else is pending before 4p.

## How to use this document

Open the repo in VS Code with Claude Code and work Part 3 top to bottom. Each phase ends in a
**gate** — a measured, falsifiable condition. Do not start the next phase until the current gate
passes; every gate exists because a specific past failure in this project would have been caught
by it (see Part 1 for which).

Suggested prompt to start a session:

> Read `docs/AGENT_REBUILD_PLAN.md`. Implement Phase 0 items 1–2 only, with the tests described
> in §5.6. Stop at the Phase 0 gate and report the measured numbers.

Progress is tracked by ticking the checkboxes in Part 3 and by appending measured numbers to
`docs/AZ_DESIGN.md` (create it in Phase 0 — it is the running record of what was actually
measured, as distinct from what was planned here).

---

## Context

The project has a correct, well-tested game engine (210 tests), a Gymnasium env, a MaskablePPO
self-play stack, an evaluation/Elo layer, and an MCTS bolt-on. Progress has stalled: the only
confirmed strength result in the repo is **56.2% [51.4%, 61.0%] for MCTS-on-checkpoint vs. the
same checkpoint's plain inference** (`docs/SEARCH_MCTS.md`), and the AlphaZero-style iterative
loop that followed produced a *clean, reproducible regression* (49% → 41%, plateaued) rather
than compounding gains.

This plan explains why that happened, and replaces the PPO+MCTS architecture with a
**value-network + full expectimax** design (the TD-Gammon shape) that fits this game's actual
structure. Decisions already made:

- **Trainer**: new value-first loop in a new `parchis/az/` package. SB3 checkpoints are retained
  as frozen evaluation baselines only.
- **Observation**: switch to a path-relative, perspective-canonical encoding, accepting that
  existing checkpoints stop loading. The encoding must model the per-colour private home stretch,
  not just the 68 shared squares.
- **Compute**: Apple M4 Mac, long runs acceptable; GCP as a later phase.
- **Ruleset**: single-die variant per `docs/RULES.md` is the final target. No engine rule changes.

Related reading: `docs/RL_DESIGN_REVIEW.md` (what was built and why), `docs/SEARCH_MCTS.md` (the
MCTS result and the failed iterative loop), `docs/CODE_REVIEW.md` (the earlier correctness pass).

---

## Part 1 — Diagnosis: why it is stuck

### 1.1 Measured properties of this game

Measured against the current engine, not assumed. Re-runnable; see `docs/AZ_DESIGN.md` for the
scripts once Phase 0 lands.

| Quantity | Value | How |
|---|---|---|
| Random 2p games/sec, 1 core | 576 | 5s loop over `Game.play_game()` |
| Mean turns per 2p game | 142 | same |
| Random 4p games/sec, 1 core | 320 | same |
| Mean turns per 4p game | 263 | same |
| Decisions (`choose_move` calls) per 2p game | 176 | instrumented `Player.choose_move` |
| Mean legal moves per decision | **2.76** (0:5%, 1:15%, 2:17%, 3:23%, 4:39% of 35,143) | same |
| `get_legal_moves` | 6.7 µs | 20k calls |
| `copy.deepcopy(Game)` | **104 µs** | 2k copies |
| Tuple snapshot + restore of `Game` | **1.0 µs + 1.2 µs** | 50k each |
| Leaves in a *full-turn* expansion (all 6 faces, bonus chains, six-rerolls) | ~370 mean / 1654 max | 60 real mid-game positions |
| Leaves in a *2-decision-ply* expansion (2.9 × 6 × 2.9) | **~54** | derived |

Three consequences follow directly, and they drive the whole design:

1. **Branching factor is 2.76 and the chance node has exactly 6 outcomes.** This is the regime
   where full-width expectimax dominates sampled MCTS. MCTS earns its keep when the branching
   factor makes exhaustive expansion impossible; here it does not.
2. **Search granularity must be the individual decision, not the turn.** Expanding a whole turn
   (including six-rerolls and bonus chains) costs ~370 leaves; expanding two decision plies
   costs ~54. `parchis/search/mcts.py` deliberately chose turn granularity — that is why it
   needed 400 simulations to get a 6-point edge.
3. **`deepcopy` is 100× a snapshot and 15× a legality query.** Any search built on `deepcopy`
   (as `mcts.py::_expand` is) pays ~5.6 ms per 2-ply-equivalent decision in copying alone. A
   tuple snapshot/restore makes the same work cost 0.12 ms.

### 1.2 The board is rotationally symmetric with period 17 — and the encoding throws that away

Starting squares are `{YELLOW:5, BLUE:22, RED:39, GREEN:56}` — **exactly 17 apart**. Home entry
points are `{68, 17, 34, 51}`. Verified for all four colours:

- steps from own start square to own home-entry square: **63**, identically
- safe squares relative to own start: `{0, 7, 12, 17, 24, 29, 34, 41, 46, 51, 58, 63}`, identically
- the other three starts relative to own start: `{17, 34, 51}`, identically

So every colour plays a **structurally identical** game. Each colour's path is a uniform
**72-position line** indexed `s ∈ [0, 71]`: `s=0` is the own start square, `s=0..63` runs along the
shared track, `s=63` is the own home-entry square, `s=64..71` is the private home column
(absolute 69–76), and `s=71` is the final square. The private home stretch is the one part this
relative frame cannot fold into the shared track, so it is carried as its own per-seat block (§2.1).

`ParchisEnv._get_observation()` (`parchis/rl/env.py:841`) encodes **absolute** squares 1–68 in
`num_players × 76` channels. The network therefore has to learn the same strategy four times,
once per colour geometry, and can transfer nothing between seats. This is pure wasted capacity
and a plausible large chunk of the plateau.

### 1.3 Critical bug: opponents in self-play see the *agent's* piece features

`ParchisSelfPlayEnv._choose_opponent_move` (`parchis/rl/env_selfplay.py:202`) calls
`self.base_env._get_observation()` to build the observation the **opponent model** acts on. But
`_get_observation()` splits its perspective:

- board-state / piece-count / progress / dice blocks are ordered by `current_player_idx` (correct
  for the opponent), while
- the own-piece block and `capture_opportunity` are built from
  `self.game.players[self.agent_player_idx]` (`env.py:924`, `env.py:962`) — the **learning agent's**
  pieces.

The opponent model is therefore fed a hybrid observation: opponent-relative board, agent-relative
piece features, and an action mask that is correct for the opponent. Every self-play run to date
trained against a partly-lobotomised opponent — the pool "curriculum" was weaker than it looked.
`parchis/search/network_eval.py`'s own docstring notes this quirk and avoids it, which confirms it
is real and known but never fixed.

### 1.4 The MCTS is unsound for this game in four separate ways

In `parchis/search/mcts.py`:

- **Open-loop chance.** A node's dice roll is sampled once at creation and frozen (`MCTSNode`
  docstring, line 151). All 400 simulations through a node re-live *one* sampled future, so `Q`
  converges to the value of a lucky/unlucky line, not an expectation. With only 6 outcomes,
  enumerating them exactly is cheaper than sampling them badly.
- **Opponents play randomly inside the tree.** `default_random_opponent_policy` is the fixed
  policy for every non-agent seat during simulation (`network_eval.py` documents this as a
  deliberate scope decision). The search evaluates "what if I do X and my opponent then blunders",
  which systematically overvalues risky lines against a strong opponent.
- **Value-scale mixing.** Terminal leaves back up ±1.0; non-terminal leaves back up
  `model.policy.predict_values(...)`, a PPO critic predicting the discounted return of
  `progress_delta` (or `win_loss` at γ=0.995) — a different, uncalibrated scale. Averaging them
  in `n.W += value` is not a coherent estimator of anything.
- **Bonus moves are searched in real play but random in simulation.** In `arena.py` the MCTS
  agent's `choose_move` is installed on the `Player`, so `Game._execute_bonus_move` calls it for
  20/10-square bonus moves too — with `roll_box["last_roll"]` still holding the *dice* roll, so
  the observation's dice one-hot is wrong and `mcts.search` treats a bonus decision as a
  turn-start root. Inside `_expand`, the override is deleted (line 230) so simulated bonus moves
  fall back to random. Real play and simulated play disagree.

### 1.5 The value signal is a proxy, and the discount destroys the sparse one

Default `reward_type="progress_delta"` optimises average piece advancement, not winning. The
alternative `win_loss` is paired with `gamma=0.995` (`parchis/training/cli.py:53`); at ~88 agent
steps per 2p game that discounts the terminal signal to 0.64, and PPO's critic then predicts a
discounted proxy return rather than P(win). **Search needs a calibrated win probability at its
leaves.** Neither configuration provides one. This — more than the promotion-gate/overfitting
issues correctly identified in `docs/SEARCH_MCTS.md` — is why the Phase C bootstrap could not
compound.

### 1.6 Value targets from a single rollout are hopeless in a dice game

Phase C regressed the value head onto the final game outcome of one self-play game. In Parchís the
outcome of a single game is dominated by dice, not by the move being labelled. The label noise
swamps the signal. Modern practice (KataGo, Stochastic MuZero) blends the *search's own* root
value into the target, which is exactly the variance reduction that was missing.

### 1.7 Evaluation cannot resolve the effect sizes being chased

`arena.play_match` uses independent random seeds per game. To separate a genuine 53% from 50% at
95% confidence needs ~4,300 games. The repo's Phase C rounds used 200. Dice games have a standard
answer for this — **common random numbers / duplicate play** (as used by GNU Backgammon's
rollouts): play every match twice on the same dice seed with seats swapped, and score the pair.
Nothing in the repo does this.

Two secondary gaps: `parchis/evaluation/elo.py::update_ratings` does one sequential K-factor update
per *pairing* (order-dependent, not a maximum-likelihood rating), and there is **no absolute
anchor** — no handcrafted strong baseline, so "56% vs. our own checkpoint" cannot be converted
into "is this agent actually good at Parchís".

### 1.8 Smaller items worth carrying into the rebuild

- `max_episode_length = 1000` truncation (`env.py:240`) yields reward 0.0 under `win_loss` —
  a stalling agent is not penalised. The new loop must score truncation explicitly as a draw.
- `_capture_threat_scores` costs ~18 `get_legal_moves` calls per observation, and self-play calls
  `_get_observation()` on *every* opponent move including bonus-chain moves (documented perf note
  at `env.py:750`).
- `Game.get_legal_moves` returns one entry per (piece, target); `env.py:366` dedups by `piece_id`,
  silently dropping alternatives — harmless today (a piece has at most one target per roll) but
  worth an assertion.

---

## Part 2 — Target architecture

**One network, evaluated by full-width expectimax over the 6 dice faces.**

### 2.1 Canonical, path-relative encoding (`parchis/az/encoding.py`)

Everything is expressed from the perspective of one *observer seat*, along that seat's own
72-position path.

- **Own pieces (4 × ~10)**, indexed strictly by `piece_id` so slots line up with the 4-way action:
  `in_base`, `finished`, path step `s ∈ [0,71]` normalised, `steps_to_home_entry`,
  `steps_to_finish`, `in_home_column`, `on_safe_square`, `is_stacked_with_own` (blockade member),
  `capture_threat_score` (reuse the existing exact roll-based computation in
  `env.py::_capture_threat_scores`, which is correct and rule-exact), and
  `blockade_member_and_forced_to_open_on_6`.
- **Track occupancy, relative (N × 68)**: for each shared-track square
  `j = (abs_pos − my_start) mod 68`, one channel for own occupancy (0/0.5/1.0) and one per
  opponent, ordered by seat *relative to me* (next-to-move first). Because starts are 17 apart,
  opponent home entries land at fixed relative offsets — the network learns one geometry.
- **Home columns (N × 8)**: each seat's private stretch as its own 8-slot block. This is the part
  the relative track encoding cannot express and must be carried separately.
- **Per-seat scalars (N × ~6)**: pieces in base / on board / finished, mean path progress,
  max piece progress, count of own blockades on track.
- **Turn context (~12)**: dice one-hot including the 6-with-empty-base (7-square) case, pending
  bonus flags (finish/capture), `consecutive_sixes / 3`, whose turn it is relative to the observer,
  turn number normalised.

Size for 2p ≈ 220 floats; for 4p ≈ 430. Provide

```python
encode(game, observer_seat, roll=None, pending_bonus=None, consecutive_sixes=0) -> np.ndarray
```

as a pure function of a `Game` — no env instance, no puppeting (`parchis/search/state_view.py`'s
`ObservationAdapter` trick goes away).

**Colour-invariance is a testable property**: encoding a position and its 17-square rotation with
colours permuted must produce byte-identical arrays. That test is the guarantee that the geometry
duplication is really gone.

### 2.2 Network (`parchis/az/net.py`)

MLP trunk `[256, 256]` + ReLU + LayerNorm (scale to three layers only if it demonstrably helps).
Two heads:

- **Value head → softmax over `num_players`**: predicted probability that each seat wins this
  game. For 2p this is a calibrated `P(I win)`; for 4p it generalises with no redesign, which is
  what makes the 4-player extension cheap. Trained with cross-entropy against a seat-win
  distribution.
- **Policy head → 4 logits, masked**: used for move ordering, for cheap 0-ply play during data
  generation, and as the opponent model inside search when depth runs out.

Value is *defined on pre-decision states* — a state with the roll already known (or a bonus
pending). That is the natural unit here, and it removes the need for afterstate machinery.

Ship a NumPy-only forward path alongside the torch path. Inference batches in search are 50–1000
rows of a small MLP; NumPy matmuls avoid torch's per-call overhead. Use torch (MPS on the M4) for
the training step.

### 2.3 Expectimax search (`parchis/az/search.py`)

```
value(state, depth) -> np.ndarray[num_players]      # win-probability vector
  terminal            -> one-hot on the winner
  truncated (cap)     -> draw vector (1/N each)
  depth == 0          -> net.value(encode(state, player_to_move))
  DECISION node       -> for each legal move: apply -> value(child, depth-1)
                         return the child vector maximising v[player_to_move]   # max^n
  CHANCE node         -> (1/6) * sum over the 6 faces of value(child, depth-1)
```

- A **decision node** is one `choose_move` — a turn-start roll, a six-again reroll, *or* a
  bonus-chain move. Uniform treatment fixes §1.4's real-vs-simulated inconsistency by construction.
- A **chance node** is one `Dice.roll()`, enumerated exactly over 6 faces (not sampled).
- `max^n` on the win-probability vector reduces to minimax at 2 players and is the correct
  generalisation at 3–4 — no separate 4-player search.
- State transitions use `Game.snapshot()/restore()` (new, ~2 µs round trip), never `deepcopy`.
- Depths: `depth=1` (≈3 leaves) is the cheap self-play policy; `depth=2` (≈54 leaves) and
  `depth=3` (≈1,000 leaves) are the strong play settings. All three come from one code path.
- Leaves are collected and evaluated in **one batched forward pass** per search, not one at a time.
- Root output: the argmax move, plus the full per-move value vector — the latter becomes the
  policy training target (a softmax over root move values at temperature τ), which is a far
  better target than MCTS visit counts at this branching factor.

### 2.4 Handcrafted heuristic agent (`parchis/agents/heuristic.py`)

Not optional — it is the absolute anchor the project currently lacks, the bootstrap opponent, and
a pool member that prevents single-lineage collapse. Linear score over ~10 features per candidate
move, evaluated with the existing rule engine:

capture value weighted by *the captured piece's own path progress* · entering from base ·
progress gained · landing safe · landing in threat range (reuse `_capture_threat_scores`) ·
forming a blockade on a square that actually blocks a trailing opponent · exact finish ·
home-column advance · leading-opponent suppression.

Tune the ~10 weights by CEM against the ladder (a few thousand fast games — minutes, not hours).
Expect a tuned heuristic to sit clearly above an untuned one and far above random.

---

## Part 3 — Implementation plan

New package layout. Nothing existing is deleted; `parchis/search/` stays as the legacy MCTS path
for the frozen SB3 baselines.

```
parchis/game/game.py            + snapshot() / restore()          (only change to existing engine)
parchis/rl/env.py               + _get_observation(perspective_seat=None)   (bug §1.3)
parchis/agents/heuristic.py     handcrafted agent + CEM weight tuner
parchis/az/encoding.py          canonical relative encoding
parchis/az/net.py               torch dual-head net + numpy inference path
parchis/az/search.py            expectimax over decision/chance nodes
parchis/az/agent.py             Player.choose_move-compatible agent, depth 0..3
parchis/az/selfplay.py          parallel game generation + target construction
parchis/az/train.py             replay buffer, training step, promotion gate, run loop
parchis/az/config.py            one dataclass; every run writes its config to runs/<name>/
parchis/evaluation/duplicate.py CRN / seat-swap paired matches
parchis/evaluation/ratings.py   Bradley-Terry Elo MLE over a pairing log
parchis/evaluation/ladder.py    benchmark ladder runner + leaderboard.json
parchis/evaluation/puzzles/     tactical suite (positions JSON + runner)
```

### Phase 0 — Foundations and feasibility gate (~2 days)

- [x] **1. `Game.snapshot()/restore()`.** Property test: 10,000 random positions,
  `snapshot → mutate → restore` produces a state byte-identical to `deepcopy` on every field
  (`board.positions`, every piece's `position/in_base/finished/move_order`, `board.move_counter`,
  `current_player_idx`, `turn_number`, `game_over`, `winner`). This is the single highest-risk
  piece of new code — prove equivalence, do not assume it.
  *Done 2026-08-25: `parchis/tests/test_snapshot.py`, 6 tests (10k-sample property test +
  identity/capture/finish/game-over anchors). See `docs/AZ_DESIGN.md`.*
- [x] **2. Fix §1.3**: `ParchisEnv._get_observation(perspective_seat=None)` defaulting to
  `agent_player_idx`; `env_selfplay._choose_opponent_move` passes the opponent's seat. Regression
  test: a scripted position where the two perspectives provably differ, asserting the opponent
  sees its own pieces. Keep this even though the SB3 stack is being retired — it makes the frozen
  PPO baselines honest.
  *Done 2026-08-25: `parchis/tests/test_observation.py` (unit-level) +
  `parchis/tests/test_selfplay.py` (wiring-level, 1135 mismatches confirmed on the pre-fix code).
  See `docs/AZ_DESIGN.md`.*
- [x] **3. `parchis/evaluation/duplicate.py` and `ratings.py`** (see Part 5) — built *before* any
  training, because every later gate depends on them.
  *`duplicate.py` done 2026-08-25 (built as a Phase 1 item-10 prerequisite, retroactively checked
  off here) — see `docs/AZ_DESIGN.md`. `ratings.py` (Bradley-Terry) built 2026-08-28, ahead of
  Phase 4 (its "2p clears the ladder" gate needs it) — see `docs/AZ_DESIGN.md`'s "Ladder + ratings
  tooling" entry.*
- [x] **4. `parchis/agents/heuristic.py`** untuned v1 + ladder entry.
  *Done 2026-08-25 (also built as a Phase 1 prerequisite, retroactively checked off here): shipped
  with CEM tuning too (`TUNED_WEIGHTS`), not just an untuned v1 — see `docs/AZ_DESIGN.md`.
  `ladder.py` built 2026-08-28 (see item 3's note above); heuristic-tuned/default are both wired in
  as ladder rungs.*
- [ ] **5. GATE**: measured throughput of `search.py` at depth 1/2/3 with a randomly-initialised
  net, and games/sec for 1-ply self-play across all M4 performance cores. Target **≥ 200 games/sec
  at depth 1**. If it lands below ~50, stop and revisit — fallbacks, in order: NumPy-only
  inference, `__slots__` on `Piece`/`Player`, a flat-array shadow state. Record real numbers in
  `docs/AZ_DESIGN.md`.

### Phase 1 — Encoding, net, search (~4 days)

- [x] **6.** `encoding.py` + colour-invariance test (§2.1) + bounds test over 100k states.
  *Done 2026-08-25: `parchis/tests/test_encoding.py`, 7 tests. Actual sizes: 216 (2p), 298 (3p),
  380 (4p) — smaller than §2.1's ~220/~430 estimate at 4p since none of the 10 own-piece features
  scale with N; see docs/AZ_DESIGN.md. Colour-invariance caught a real bug during development:
  `_per_seat_scalars` initially reused `parchis.rl.rewards.calculate_normalized_progress`, which
  divides raw ABSOLUTE position by 76 — not colour-invariant, since Blue/Red/Green's own paths
  wrap the 1-68 boundary before reaching home. Fixed with a relative (per-owner-start) formula.*
- [x] **7.** `net.py` with both forward paths; assert numpy and torch outputs agree to 1e-5.
  *Done 2026-08-25: `parchis/tests/test_net.py`, 4 tests, agreement confirmed to 1e-5 across
  num_players 2/4 and batch sizes 1/8/64.*
- [x] **8.** `search.py`. Correctness tests:
  - `depth=1` with a value function equal to normalised progress reproduces a greedy-progress
    agent exactly;
  - chance node values equal a brute-force `mean` over the 6 faces computed independently;
  - `depth=d` result is invariant to move ordering;
  - a hand-built position where a 2-ply-only win exists is found at depth 2 and missed at depth 1;
  - search never mutates the real `Game` (snapshot hash before/after).
  *Done 2026-08-25: `parchis/tests/test_search.py`, all 5 required properties + 1 more. Found and
  fixed a real bug while sizing item 10's gate: a "no legal move" decision (and a three-sixes
  penalty transition) consumed zero depth, so exact expectimax exploring "what if a player never
  rolls the 5 they need" could recurse without bound — confirmed via RecursionError, fixed by
  making an empty decision cost a depth unit like any other, regression-tested. See
  docs/AZ_DESIGN.md.*
- [x] **9.** `agent.py` + wire into `parchis/evaluation/arena.py` (its factory interface already
  fits) and into the ladder.
  *Done 2026-08-25: `parchis/tests/test_agent.py`, 3 tests, including a regression test that
  bonus decisions are never confused with a fresh roll (§1.4's mcts.py bug) across real games.
  Arena wiring needed no arena.py changes (same factory interface, confirmed via
  `arena.play_one_game` in tests). The full `ladder.py` (fixed rungs, leaderboard.json) is
  deferred — not needed for item 10's gate, which only needs duplicate.py's Wilson CI.*
- [x] **10. GATE**: `heuristic + depth 2` must beat `heuristic + depth 0` on ≥ 400 duplicate pairs
  with a Wilson lower bound clear of 50%. This validates the search independent of any learned
  value — if search doesn't help a hand-built evaluator, it will not help a learned one.
  ***PASSED 2026-08-25**: 400/400 duplicate pairs (800 games), win_rate 65.4% (523/800), Wilson
  95% CI [62.0%, 68.6%] — lower bound clearly clear of 50%. Pair record: 172 pairs where search
  did strictly better, 179 splits, only 49 where the no-search baseline did better. See
  docs/AZ_DESIGN.md for the full run.*

### Phase 2 — Bootstrap the value net (~3 days)

- [x] **11.** Generate ~200k games from a mixed pool (tuned heuristic, ε-noisy heuristic, random)
  and train the value head to predict the seat-win distribution; train the policy head on the
  moves played.
  *Done 2026-08-26, at reduced scale: 20,000 games / 3.14M decisions (not 200k -- a single-session,
  single-machine time/memory budget; see docs/AZ_DESIGN.md for the exact reasoning and measured
  numbers) via `parchis/az/selfplay.py`, trained via `parchis/az/train.py`
  (AdamW/cosine/weight-decay, early stopping, game-level train/val/test split so correlated
  same-game decisions never leak across the split). `parchis/az/config.py` and
  `parchis/az/turn_context.py` (shared bonus-vs-fresh-roll tracker, refactored out of
  `agent.py`) also built. Found and fixed a real, serious bug along the way: the value target was
  stored in ABSOLUTE seat order while the encoding it trains against is mover-relative order --
  training the net on a consistent-looking but meaningless mapping. See docs/AZ_DESIGN.md.
  **Follow-up, same session**: full ~200k-game target subsequently reached -- 200,000 games /
  31.4M decisions across 20 resumable shards (70 min, 26GB, 0 truncated) via a new sharded
  generation script, trained via `train.py`'s new `split_shards`/`bootstrap_train_sharded`
  shard-streaming path (holds one training shard in memory at a time; needed once the corpus
  exceeds RAM). See docs/AZ_DESIGN.md follow-up section.*
- [x] **12. GATE**: value calibration — bucket predictions into deciles, compare predicted vs.
  actual win frequency on held-out games; **expected calibration error < 0.05**. A miscalibrated
  value makes expectimax actively harmful, so this is checked before any self-play.
  *PASSED 2026-08-26: ECE = 0.0145 (< 0.05), on 314,487 held-out decisions from 2,000
  entirely-held-out games, predictions spanning the full [0,1] range with roughly even bucket
  counts (real discrimination, not collapse to the base rate). See docs/AZ_DESIGN.md.*
- [x] **13. GATE**: `net@depth1` ≥ tuned heuristic on duplicate pairs.
  *PASSED 2026-08-26 at full scale: 800 duplicate pairs (1,600 games), win rate 61.4% (982/1,600),
  Wilson 95% CI [58.96%, 63.73%] -- lower bound decisively clear of 50%. Pair record: 314 pairs
  net@depth1 did strictly better, 354 splits, 132 heuristic did strictly better. Supersedes the
  reduced-scale (20k games) run's MARGINAL result (52.0%, CI lower bound a hair under 50%),
  confirming that hypothesis's own prediction that dataset size -- not epochs or value-loss
  weighting, both already tried at 20k -- was the binding constraint: at 10x the data, calibration
  (ECE) also improved ~7x (0.0145 -> 0.0021) alongside the win-rate jump. Checkpoint:
  `runs/bootstrap_2p_v4_large/`, now the project's current-best checkpoint. See docs/AZ_DESIGN.md
  for the full numbers.*

### Phase 3 — Self-play loop (the main event, continuous)

*Status 2026-08-27: built (`parchis/az/round_loop.py`, `targets.py`, `champion_pool.py`,
`selfplay.generate_round_games`), tested (43 new tests across 6 new/extended test files), and the
initial 40-round target **completed** (~27.5 hours, seeded from the Phase 2 checkpoint). 3
promotions (rounds 4, 6, 23) -- current champion copied to `runs/selfplay_2p_v1_champion/`,
superseding the Phase 2 bootstrap checkpoint. All 9 depth-2 escalations failed to promote while
consuming ~79% of total wall-clock time -- a real finding, not noise, that needs a decision (fix,
reconfigure, or drop the escalation mechanism) before any continuation. See docs/AZ_DESIGN.md for
the concrete design choices the plan left unspecified, the full round-by-round log, and the
escalation analysis.*

Each round:

- **Generate** ~50k games with the champion at `depth=1`, exploration via softmax over root move
  values with temperature τ annealed 1.0 → 0.25 over the first ~15 plies, plus Dirichlet noise on
  the root's move-value softmax. Log every decision node: encoding, root value vector, root move
  values, seat, and the final outcome.
- **Target construction** (the fix for §1.6):
  `z_value = (1−λ)·outcome_onehot + λ·root_value_vector` with λ ≈ 0.5, tuned once;
  `z_policy = softmax(root move values / τ_target)`.
- **Train** on a replay buffer holding the last ~3 rounds — a recency window, not unbounded
  accumulation, which is the specific mistake documented in `docs/SEARCH_MCTS.md` — with weight
  decay, a held-out validation split, and early stopping on validation loss. Cap warm-start epochs.
- **Promote** only on a CI-confirmed win over the champion on ≥ 600 duplicate pairs. If three
  consecutive rounds fail to promote, escalate: raise generation depth to 2 for one round (expert
  iteration — stronger data than the current net can produce on its own). This is the mechanism
  the failed Phase C had no equivalent of.
- **Pool**: opponents sampled from {champion, last 4 promoted, tuned heuristic, random}. The
  heuristic anchor is what stops a single-lineage collapse.

Run continuously; each round is checkpointed and resumable. Expect first meaningful gains within
~5 rounds and the interesting region around 20–50 rounds.

### Phase 4 — 4-player extension (after 2p is strong)

The design already carries it: `max^n` search over the per-seat win-probability vector, and an
encoding whose seat ordering is relative to the observer. Concrete work: `num_players=4` configs;
a 4-seat duplicate-match protocol (rotate the tested agent through all 4 seats on the same dice
seed — the natural CRN generalisation); replace `multiplayer_matrix.py`'s pairwise-only view with
a seat-rotated round-robin. Optionally warm-start the 4p net from the 2p trunk, since the encoding
is deliberately shaped to make that possible.

---

## Part 4 — Training plan

| Setting | Value | Why |
|---|---|---|
| Value target | seat-win distribution, cross-entropy | calibrated P(win), directly usable by search |
| Discount | none (γ=1) | the objective is the outcome, not a discounted proxy |
| Value target blend λ | 0.5 (search root value vs. outcome) | §1.6 variance reduction |
| Generation depth | 1 (2 on escalation, retired by default 2026-08-29 -- see docs/AZ_DESIGN.md) | ~3 leaves/decision keeps generation cheap |
| Play/eval depth | 1, always -- matches generation depth (revised 2026-08-29; this row originally said "2 default, 3 for strongest," an aspiration never actually used: every promotion gate and benchmark in this project's history ran eval at base_depth=1. See `round_loop.py`'s eval-depth-confound fix and the puzzle suite's search-pathology finding, both in docs/AZ_DESIGN.md, for why a real depth increase needs much more puzzle-suite evidence before adopting) | ~3 leaves at depth 1; ~54 / ~1000 at depth 2/3 if ever revisited |
| Truncation | 1,000 turns → scored as a draw (1/N per seat) | §1.8; never silently 0 |
| Optimiser | AdamW, lr 1e-3 cosine, weight decay 1e-4 | early stopping on validation loss |
| Buffer | last ~3 rounds | recency window, not unbounded accumulation |
| Promotion | ≥600 duplicate pairs, Wilson lower bound > 50% | the gate Phase C lacked |
| Parallelism | batched leaf eval per search (revised 2026-08-29: `multiprocessing` over M4 performance cores was never actually implemented -- no code under `parchis/az/` uses it; only the batched-leaf-eval half of this row shipped) | dominant cost is net evals |

**GCP phase (later, only if the M4 saturates):** the loop is embarrassingly parallel at the
game-generation level. Move generation to N preemptible CPU workers writing shards to GCS, keep a
single trainer. No architectural change needed — this is why generation and training are separate
modules from the start.

---

## Part 5 — Tracking and evaluation plan

### 5.1 Duplicate (paired) matches — `parchis/evaluation/duplicate.py`

Every A-vs-B match is a set of **pairs**: the same dice seed played twice with seats swapped.
Score each pair as {A wins both, split, B wins both}. This cancels most of the dice luck that
currently forces 4,000-game runs. Phase 0 measures the actual variance reduction on a
known-equal pairing (a net vs. itself) and records the effective-n multiplier in
`docs/AZ_DESIGN.md` — that number is then used to size every later run.

### 5.2 Fixed benchmark ladder — `parchis/evaluation/ladder.py`

Fixed rungs so results stay comparable across months: `random` · `heuristic-v1` ·
`heuristic-tuned` · frozen PPO flagship (`small_win_loss_combo15_seed42`, plain and MCTS) ·
frozen `az-r{N}` snapshots at depths 1/2/3. Every ladder run appends per-pairing results to a
single append-only `runs/pairings.jsonl`.

### 5.3 Ratings — `parchis/evaluation/ratings.py`

Fit **Bradley-Terry ratings by maximum likelihood** over the whole `pairings.jsonl`, anchored at
`random = 0`, with bootstrap confidence intervals. This replaces `elo.py`'s order-dependent
sequential K-factor updates and gives one number per agent comparable across the whole project
history.

### 5.4 Tactical puzzle suite — `parchis/evaluation/puzzles/`

40–60 hand-built positions with a defensible best move, each with a one-line rationale:
capture a far-advanced piece over a near-base one · mandatory entry vs. moving out · take the
exact finish · avoid the square where 4 of 6 faces capture you · break the right blockade piece
on a 6 · spend the 20-square bonus on the piece that gains most, not the one that captured.
Reported as `puzzle_accuracy`. Doubles as a fast deterministic regression test and gives an
interpretable read on *what kind* of mistakes the agent makes.

*Loader/runner/CLI built 2026-08-28 (`docs/AZ_DESIGN.md`'s "Tactical puzzle suite: loader +
runner"); CSV schema below, `python -m parchis.evaluation.puzzles --agent <spec>` replacing this
section's original `--agent az-latest` placeholder (predates `parchis.agents.agent_spec`'s actual
spec grammar). A visualizer (`parchis/visualization/visualize_puzzles.py`, 2026-08-29) renders any
puzzle's position, an agent's per-move evaluation, and the ground-truth answer on the real board --
see AZ_DESIGN.md's "Puzzle suite visualization". The user has started filling in the real 40-60
positions in `parchis/evaluation/puzzles/my_puzzles.csv`.*

CSV schema (one row = one decision; colors fixed A=RED/B=YELLOW): `puzzle_id`, `category`,
`a_piece_0`..`a_piece_3`, `b_piece_0`..`b_piece_3` (`0`=base, `1`-`68`=main track, `69`-`75`=that
color's home column, `76`=finished), `turn` (`A`/`B`), `roll` (`1`-`6`, or `capture_bonus`/
`finish_bonus`), `consecutive_sixes` (`0`-`2`, must be `0` unless `roll`==6), `correct_piece_id`
(`0`-`3` — not a destination, which is fully determined and loader-computed; or several `0`-`3`
values separated by `/`, e.g. `2/3`, when more than one move is genuinely correct), `rationale`.
The file itself may be `,`- or `;`-delimited (auto-detected per file — spreadsheet software in a
`,`-as-decimal-separator locale exports the latter by default) and may start with a UTF-8 BOM.

### 5.5 Per-run metrics

`runs/<name>/metrics.jsonl` + TensorBoard, per round: value/policy loss (train and validation),
**value calibration ECE**, win rate vs. each ladder rung with Wilson CI, Bradley-Terry rating,
`puzzle_accuracy`, promotion decision, mean game length, capture rate, three-sixes rate,
mean/max search leaves per decision, games/sec, and win-rate-by-seat (the existing fairness check).

### 5.6 Testing plan

- **Keep all 210 existing tests green**, including for the SB3 stack being retired — the frozen
  baselines depend on it.
- **New unit tests**: snapshot/restore equivalence vs. `deepcopy`; encoding colour-invariance under
  17-square rotation; encoding bounds over 100k states; numpy/torch forward agreement; expectimax
  chance-node value vs. brute force; depth-invariance to move ordering; search never mutates the
  real `Game`; truncation scored as a draw; the `_get_observation(perspective_seat)` regression.
- **Property tests**: 100k random positions with no exception, no NaN, no out-of-bounds encoding.
- **Integration smoke**: a 3-round self-play run at tiny scale (200 games/round) that must
  complete, promote or decline cleanly, and write a well-formed `metrics.jsonl`.
- **Determinism**: a fixed seed reproduces an identical game, identical search output, and an
  identical training batch.

---

## Part 6 — Risks and how each is handled

| Risk | Handling |
|---|---|
| Python search too slow on M4 | Phase 0 gate with a hard number; fallbacks pre-identified (NumPy inference, `__slots__`, flat-array shadow state) |
| Value net miscalibrated → search hurts | Explicit ECE gate in Phase 2 before self-play begins |
| Self-play collapses again (Phase C repeat) | Recency-window buffer, early stopping, promotion gate, heuristic anchor in the pool, depth escalation after 3 failed rounds |
| Effect sizes too small to measure | Duplicate matches with a measured effective-n multiplier; every gate stated as a Wilson lower bound, not a point estimate |
| `snapshot/restore` subtly wrong → silent corruption | Byte-equivalence property test vs. `deepcopy` on 10k positions, plus a search-does-not-mutate assertion |
| Scope creep into 4p too early | 4p is Phase 4; the design carries it (max^n + per-seat value vector) but no 4p work happens until 2p clears the ladder |

## Part 7 — Rough sequencing

Phase 0 ≈ 2 days · Phase 1 ≈ 4 days · Phase 2 ≈ 3 days (plus bootstrap generation time) ·
Phase 3 continuous, gated round by round · Phase 4 after 2p is strong.

The first externally meaningful milestone is the **Phase 1 gate** — search demonstrably improving a
hand-built evaluator — because it isolates the one thing the current MCTS could never prove.

## Verification

- `pytest parchis/tests/` green throughout (210 existing + ~35 new).
- `python -m parchis.az.search --benchmark` prints depth-1/2/3 leaf counts and ms/decision.
- `python -m parchis.evaluation.ladder --agents random heuristic-tuned az-latest --pairs 600`
  prints the ladder table with Wilson CIs and refits Bradley-Terry ratings.
- `python -m parchis.evaluation.puzzles --agent az-latest` prints puzzle accuracy per category.
- `python -m parchis.az.train --config configs/2p_smoke.yaml --rounds 3` completes end to end and
  writes a well-formed `runs/<name>/metrics.jsonl`.

## References

- TD-Gammon (Tesauro): TD-learned value + full expectimax at play time — the architectural
  template for this plan.
- [\*-Minimax / Star2 in backgammon](https://www.researchgate.net/publication/220962545_-MINIMAX_performance_in_backgammon) —
  expectimax with pruning for stochastic games.
- [Stochastic MuZero / planning in stochastic environments](https://www.julian.ac/blog/2022/05/15/planning-in-stochastic-environments-with-a-learned-model/) —
  afterstates and the decision/chance node alternation; search-improved value targets.
- [GNU Backgammon rollouts and variance reduction](http://www.gnubg.org/documentation/doku.php?id=rollouts) —
  common random numbers / quasi-random dice, the basis for §5.1.
- [Variance reduction in backgammon rollouts](https://bkgm.com/articles/GOL/Feb00/var.htm)
