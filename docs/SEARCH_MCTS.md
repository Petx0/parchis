# Search-augmented play: what was built, what was tried, what's kept

`parchis/search/` adds AlphaZero-style Monte Carlo Tree Search (MCTS) on
top of the existing game engine and trained checkpoints. This is a genuinely
different system from the one-shot `model.predict(obs)` inference everywhere
else in the project (no lookahead there at all) -- see the design rationale
in `docs/RL_DESIGN_REVIEW.md` for why it doesn't slot into the existing PPO
training loop as a simple variant. This doc records what was actually built
and validated, since the exploration spanned three phases and only some of
it is still live code.

## What's kept: the search engine + arena (Phase A/B)

**Live, reusable, confirmed useful.** `parchis/search/mcts.py` is a
depth-limited PUCT search over the real `Game` object (`parchis/game/`),
handling the two things vanilla MCTS doesn't support out of the box:

- **Stochastic transitions** (dice rolls) via `isolated_random.py` --
  chance nodes are sampled, not expanded, using a save/restore of Python's
  global `random` state (the game engine's `Dice.roll()` draws from the
  global module directly, not an injectable RNG, so this is the only way
  to keep search-internal randomness from perturbing real gameplay).
- **Multi-agent turns**, by only building tree nodes at the *searching
  agent's own* turn-starting decisions -- opponent turns and the agent's
  own bonus-chain continuations are resolved by a fixed policy between
  nodes, reusing `Game.play_turn()` unmodified via a `choose_move`
  instance-attribute override and an exception-based "pause" mechanism.

`parchis/search/heuristic_eval.py` is a placeholder evaluate_fn (uniform
priors + a progress-based value heuristic) used while validating the
engine itself, independent of any trained network.
`parchis/search/network_eval.py::make_network_evaluate_fn` and
`parchis/search/agents.py` wire a real trained **MaskablePPO checkpoint**'s
own policy/value outputs into the search as priors/leaf-value -- no
retraining involved, just smarter use of an existing checkpoint at
inference time. `parchis/evaluation/arena.py` is the comparison harness
(`Game`-level, not `MaskablePPO`-specific like `evaluate_agent()`/`elo_ladder.py`,
since a search-backed agent needs live search at inference time, not just
saved weights) -- reports win rate + Wilson CI, same statistical standard
as the rest of the project's evaluation tooling.

**Confirmed result** (search vs. plain inference, same flagship checkpoint
`small_win_loss_combo15_seed42`, three independent runs at increasing
rigor):

| Games | Simulations | Win rate | 95% CI |
|---|---|---|---|
| 60 | 100 | 55.0% | [42.5%, 66.9%] |
| 200 | 400 | 56.5% | [49.6%, 63.2%] |
| 400 | 400 | 56.2% | **[51.4%, 61.0%]** |

**CONFIRMED**: search measurably beats the same checkpoint's un-searched
policy (lower CI bound clears 50% at n=400). This is a real, reusable
capability -- any existing MaskablePPO checkpoint can be played with search
on top via `parchis/search/agents.py::make_mcts_ppo_agent_factory`, no
retraining required. Default search hyperparameters, evidence-backed by
this result: `n_simulations=400`, `c_puct=1.4`, `max_depth=3` (the depth
cap matters -- an earlier unbounded-depth version let a clearly-worse move
accumulate more visits than a clearly-better one, since "who's ahead"
stops reflecting the root decision many turns deep).

## What was tried and archived: the iterative self-play loop (Phase C)

**Not kept as live code** -- the code and large data artifacts (self-play
data pools, per-round candidate checkpoints) were deleted after this
concluded; this section is the permanent record of what happened, since it
only existed in an ephemeral planning document before.

The natural next step after confirming search helps: use search to
generate better self-play training data, train a new network on it, and
iterate (classic AlphaZero). Two things were tried:

**Round 1 (single bootstrap)**: generate self-play games with the flagship
checkpoint + search (both seats), train a new small dual-head network
(policy + value) on the resulting visit-count/outcome targets via
supervised regression, arena-test it against the flagship (both with
search). Result: **98/200 (49.0%), CI [42.2%, 55.9%]** -- statistically a
coin flip, not a confirmed improvement. Plausible explanation: the new
network was trained to imitate what "flagship + search" already does, so
landing at the same strength (not beyond it) is a reasonable outcome for
one bootstrap pass -- real AlphaZero-style gains are expected to compound
over multiple rounds, which this deliberately deferred.

**5 further iterative rounds** (self-play with the current-best network →
append to a growing data pool → warm-start-train the candidate → arena-test
→ promote only if CI-confirmed better, the standard AlphaZero promotion
gate): produced a **clean, reproducible regression**, not noise --
declining for 3 rounds then plateauing at a stable ~40-41%, confirmed worse
than the flagship from round 3 onward.

| Round | vs. flagship | 95% CI | Promoted? |
|---|---|---|---|
| 1 (seeded) | 49.0% | [42.2%, 55.9%] | No |
| 2 | 43.5% | [36.8%, 50.4%] | No |
| 3 | 41.0% | [34.4%, 47.9%] | No -- confirmed worse |
| 4 | 40.5% | [33.9%, 47.4%] | No -- confirmed worse |
| 5 | 40.0% | [33.5%, 46.9%] | No -- confirmed worse |
| 6 | 41.5% | [34.9%, 48.4%] | No -- confirmed worse |

**Root cause**: the flagship was never promoted (nothing ever beat it), so
self-play data came from the *same* source every round -- the "better
network → better self-play data → better network" compounding loop that
makes AlphaZero-style training actually work never engaged. Meanwhile the
candidate kept accumulating training (120 epochs total by round 6,
warm-started round over round, no regularization or early stopping) on
that same-quality data -- a textbook overfitting setup, compounding an
already-weak value-head learning signal noted back in round 1.

**Real fixes for a future attempt, not built**: add regularization (weight
decay/dropout) and/or validation-loss-based early stopping per round
instead of a fixed epoch count; reconsider open-ended warm-starting (a
periodic reset, or capping total accumulated epochs); most fundamentally,
the single-lineage design has no mechanism to inject useful variety into
self-play once the generator is stuck -- a real structural gap, not just a
hyperparameter to retune.

Phase B's result stands untouched regardless of this outcome: search alone
(no retraining) gives a confirmed ~56% edge over plain inference on the
flagship, and that capability is live in `parchis/search/`/
`parchis/evaluation/arena.py` today.
