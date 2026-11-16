# Research backlog

Ranked list of experiments most likely to make the agent stronger. Maintained by the Researcher
(PILA-4). The list is re-ranked after every verdict, and the ledger at the bottom records every idea
already tried. **First version: 2026-11-15. Re-ranked 2026-11-16 after the PILA-6 depth study, and again the
same day after the PILA-8 verdict (weight soups).**

How to read it:
- **Champion:** soup S1, `runs/soups_2026-11-16/s1/model.pt`, the uniform weight average of V2.1c,
  win3 and win10. It passed protocol v1 against V2.1c in PILA-8 (below); the pointer moves when the
  owner merges PR #7. It is ranked at depth 1 (`docs/EVAL_PROTOCOL.md` §1). The recommended play
  depth is 3, but that was measured on V2.1c (PILA-6). S1 has not been measured above depth 1.
- **"pp"** means percentage points of head-to-head win rate against the champion in duplicate
  pairs, **at depth 1** unless stated. Every expected effect below is a **hypothesis** unless a
  recorded run is cited.
- **Training arms are judged by their soup, not their endpoint.** The 20-round endpoints of the
  V2.1c recipe lose to their start (48.2% and 48.4%, §10.11), yet the average of the start and
  those two endpoints beats the start by +3.1pp (PILA-8). An endpoint-only comparison therefore
  understates a recipe by several pp. From now on every training item starts from S1, keeps its
  per-round candidates, and is compared soup against soup with the PILA-5 harvest (#2) as the
  matched control. That makes #2 a prerequisite for #3, #4, #5 and #7.
- **Search shrinks net-to-net gaps.** V2.1c's lead over V2.0c is 52.91% [52.22, 53.60] at depth 1
  and 51.65% [50.96, 52.34] at depth 2, over 20,000 games on the same deals each (PILA-6, seeds
  5002100+c). So a net gain measured at depth 1 is worth roughly half to two thirds as much to the
  deployed agent. This is one pair of nets, so read it as a rule of thumb.
- **Ranking** uses expected gain × probability ÷ compute. Ideas that open a new axis (search at
  play time, 4 players) get a bonus. Ideas that resemble a closed negative result get a penalty.
- **Evaluation** follows `docs/EVAL_PROTOCOL.md` (protocol v1, merged in PILA-2):
  - 10,000 duplicate pairs (20,000 games).
  - Wilson 95% lower bound above 50% to promote.
  - Anything chosen as best of N is re-confirmed on fresh seeds.
  - The Evaluator allocates every new seed block.
  - Studies at a different depth are a different time budget, so they are not promotions (§1).
- **Training seeds** use a base of `10_000_000 × n`, with n unique per run (lab rules).
- **Compute** is wall-clock on the Mac mini (M4, 10 cores), one heavy job at a time.

### Search cost and strength by depth (PILA-6, recorded)

V2.1c with one process and one-thread BLAS. Times are per decision over 500 self-play positions
(seeds 5002900+i). Strength comes from duplicate-pair matches in seed block 2 (kind `study`),
code `c621f7c`. Raw results: `runs/depth_study_v2_1c/eval/depth_study_b2.json`.

| Depth | Mean ms | p90 ms | Max ms | Match | Win rate [Wilson 95%] | Games |
|---|---|---|---|---|---|---|
| 1 | 0.27 | 0.36 | 0.49 | — | — | — |
| 2 | 2.9 | 5.9 | 6.4 | d2 vs d1 | 60.39% [59.72, 61.07] | 20,000 |
| 3 | 51 | 132 | 161 | d3 vs d2 | 58.48% [56.94, 59.99] | 4,000 |
| 4 | 993 | 3,077 | 3,397 | not played | — | — |

Each ply costs about 10–20× more than the one before, and returns are not yet diminishing
(+10.4pp, then +8.5pp). Cost calibration for planning: the 2,000-pair d3-vs-d2 match took 56 min
on 8 workers, about 6.7 s of worker time per game, or about 250 decisions per game.

---

## Ranked list

| # | Idea | Status | Expected effect | Compute (Mac) | Issue |
|---|---|---|---|---|---|
| 1 | Pruned and time-budgeted deeper search (Star2, depth 4 under a time cap) | not tried; unlocked by PILA-6 | **+3 to +7pp at play time**, depth 4 vs depth 3 | ~1 day code + ~1.5 h | — |
| 2 | Harvest a 20-round continuation **from S1**, then soup its trajectory | not tried; **prior raised by PILA-8**; re-designed around S1 | +0 to +1.5pp vs S1 | ~1 h code + ~1.9 h | PILA-5 (re-scope pending) |
| 3 | Depth-3 targets on a slice of decisions | dropped by the owner 2026-11-02 before depth 3 was measured; **re-proposed, see what's different** | +0 to +2pp | ~1 h code + ~4.5 h (~3 h if #1 lands first) | after #2 |
| 4 | Weight averaging inside the loop (averaged generator and expert) | **new** after PILA-8 | +0.5 to +2pp vs #2's soup | ~2 h code + ~1.9 h | only if #2 is positive |
| 5 | TD(λ) value targets for cheap (outcome-only) rows | not tried | +0.5 to +2.5pp | ~1 day code + ~1.9 h | after #2 |
| 6 | 4-player bootstrap and 4-player evaluation harness (V2.3 stage 1) | not tried; new axis | new axis (target: beat the tuned heuristic at a 4-player table) | 1–2 days code + ~4 h | — |
| 7 | Per-head early stopping / lower value weight after epoch 1 | not tried (§10.10 reading 3) | 0 to +1pp | ~2 h code + ~1.9 h | after #2 |
| 8 | Exploiter probe against frozen S1, then a league pool | not tried | 0 to +2pp if S1 is exploitable | ~1.5 h probe | — |
| 9 | Chance-resolved ("depth 1.5") targets on cheap rows | partly tried: C3 and deep_fraction 0.5 | 0 to +1pp | ~1 day code + ~2 h | only if #5 is flat |
| 10 | Extra input features, zero-initialised for warm start | not tried | 0 to +1.5pp | ~1 day code + ~1.9 h | — |
| 11 | Exact endgame race database (bear-off analogue) | not tried | +0.5 to +1pp at play time | 2–3 days code + hours of DP | — |
| 12 | Population of recipe replicates, souped rather than selected | resembles §10.9/§10.11; re-read after PILA-8 | +0.5 to +1.5pp | ~5 h | — |
| 13 | Revive PPO, or PUCT-MCTS on PPO checkpoints | **not recommended** | — | — | — |

**What changed after PILA-8 (2026-11-16, second re-rank).** Weight soups (old #3) passed: S1 beats
V2.1c 52.79% [52.09, 53.48] on the 20,000-game confirmation block, 53.08% [52.60, 53.57] pooled
over 40,000 (closed below). That is the largest single step in the lineage and twice the top of my
predicted range. Four consequences:
- S1 is the new start point and the new bar for every item.
- The PILA-5 harvest moves from #4 to #2. It is now cheap insurance for every later training item,
  because it doubles as their matched control.
- A new #4 tests averaging inside the loop, but only if #2 shows that averaging compounds.
- #12 (replicates) is re-read: replicates are soup ingredients, not lottery tickets.

**What changed on 2026-11-16 (first re-rank).** PILA-6 measured play-time depth on V2.1c (closed below). Search is
now the largest measured lever, so the two search ideas move to the top. Old #6 (pruned search) is
now #1, because its kill criterion (depth 2 below 52% vs depth 1) did not fire. Depth-3 targets
are #2. Net-only ideas keep their order, but their value to the deployed agent is
discounted (see "Search shrinks net-to-net gaps" above).

**Not an experiment, for the owner:** deploying V2.1c at depth 3 is a configuration choice. It
costs 51 ms per decision on average and 161 ms at most. It does not change the champion pointer,
which ranks nets at depth 1. Whether to add a separate depth class to the leaderboard is also
the owner's call (EVAL_PROTOCOL §1).

**Also for the owner, after PILA-8:** whether the deployed agent should switch from V2.1c to S1
depends on a measurement nobody has made: S1 vs V2.1c at depth 2 and 3. Net gaps shrink with
search (V2.1c's lead over V2.0c fell from +2.9 to +1.7pp at depth 2), so my expectation is that S1
keeps roughly +1.5 to +2pp at depth 2 and somewhat less at depth 3. That is a hypothesis. It is an
Evaluator `study`, not an experiment. At the PILA-6 cost calibration, 2,000 pairs with both sides at
depth 2 take about 6 min on 8 workers, and 1,000 pairs with both sides at depth 3 about 1 h. Adding S1 to the round robin is the
other open item from the verdict.

### 1. Pruned and time-budgeted deeper search (depth 3–4)

- **Hypothesis.** Depth 3 beats depth 2 by +8.5pp and the returns are not yet diminishing
  (PILA-6). Depth 4 is the next step, but full width costs 993 ms per decision on average and up to
  3.4 s. Two changes can bring depth 4 to interactive speed:
  - Star1/Star2 pruning at chance nodes, on the mover's value channel, which is bounded in [0, 1].
  - Ordering moves by the policy head, so the cutoffs come early.

  Star2 is exact: it returns the same value as full-width expectimax, only cheaper.
  - Expected node savings: 2–5× at depth 3–4, judging by backgammon (Hauk et al. 2004). That is a
    hypothesis for Parchís.
  - Expected strength: depth 4 vs depth 3 at +3 to +7pp, extrapolating the +10.4 then +8.5pp
    sequence. Also a hypothesis.
- **Change.**
  - `parchis/az/search.py`: add `prune="star2"` with probing at chance nodes and policy-ordered
    children. Keep full width as the default, so all recorded results stay reproducible.
  - Add a time-budgeted mode: iterative deepening (2 → 3 → 4) that returns the deepest completed
    search within `max_ms`. This gives the deployed agent a latency guarantee.
  - Risks:
    - Parchís gives an extra turn on a 6, so the same player can move on consecutive plies. The
      bound bookkeeping must follow the mover, not alternate.
    - Leaf values are a vector. Pruning must use only the root mover's channel and stay exact.
- **Compute.** About 1 day of code (Engineer), then about 1.5 h of Mac time, one heavy job.

  | Step | Size | Estimated time |
  |---|---|---|
  | Exactness: pruned and full-width pick the same move and value (to 1e-6) at depth 3 | 1,000 positions | ~5 min, light |
  | Timing at depths 3–4 on PILA-6's 500 positions | 500 positions | ~10 min |
  | Pruned depth 4 vs depth 3 | 500 pairs (≈ ±2.5pp), hard cap 90 min | ~60–90 min |

- **Net.** Use S1 once PR #7 merges. The exactness and timing steps do not depend on the net.
- **Evaluation.**
  - A study, not a promotion (EVAL_PROTOCOL §1). The Evaluator allocates a `study` block.
  - Report a strength-vs-ms row for pruned depth 3, pruned depth 4, and the time-budgeted agent at
    100 ms and 500 ms caps.
- **Kill criteria.**
  - Exactness: any move disagreement at depth 3 is a bug. Fix it before any match.
  - If pruned depth 4 still averages above 500 ms per decision, skip the strength match. Ship
    pruned depth 3 (cheaper evaluation and targets) and record depth 4 as too slow.
  - If depth 4 vs depth 3 reads below 52% at 500 pairs, the deployed agent stays at depth 3.
- **Prior evidence.**
  - PILA-6 (above); `docs/V2_PLAN.md` C7 named Star1 for a depth-3 "strongest" setting. It was
    never built.
  - V2.1c solves 4/4/7/6/6 of 10 puzzles at depths 1–5 (`runs/puzzles_2026-11-03/report.json`).
    That is too few puzzles to show diminishing returns.
- **Sources.**
  - Ballard 1983, *The \*-minimax search procedure for trees containing chance nodes*.
  - Hauk, Buro, Schaeffer 2004, *\*-Minimax performance in backgammon*.
  - Superhuman Stratego with test-time search, [arXiv:2511.07312](https://arxiv.org/abs/2511.07312).

### 2. Harvest a continuation from S1, then soup its trajectory (PILA-5, re-scoped again)

- **Hypothesis.** PILA-8 averaged a start (V2.1c) with two 20-round continuations of it and beat the
  start by +3.1pp, although each continuation on its own loses to the start (48.2%, 48.4%).
  Running the same recipe from S1 and keeping every per-round candidate tests whether that step
  **compounds**: a soup of S1 and its own trajectory should beat S1.
- **Expected effect.** +0 to +1.5pp vs S1, probability ~0.35 (hypothesis). Lower than PILA-8's gain:
  S1 already sits nearer the centre of the basin, so a second averaging step should return less.
- **What is different from the 2026-11-15 re-scope** (my comment on PILA-5, not yet approved):
  - Start from S1 and compare against S1. A continuation from V2.1c could only add one more
    ingredient to S1's own recipe.
  - Candidates are soups of the trajectory, including S1 itself: the shape that won in PILA-8.
    Single per-round candidates are no longer screened. On average they lose to their start
    (§10.11), and screening ten of them inflates the best-of-N bias. PILA-8's best-of-4 screen read
    54.17%, against 53.08% under the protocol: about 1.1pp of winner's curse with N = 4.
  - The run's endpoint and soups are the matched control for #3, #4, #5 and #7 (same start, same
    recipe, no change), so it runs before them.
- **Change.**
  - `scripts/v2/run_harvest.py`, based on `run_window.py`:
    - start from `runs/soups_2026-11-16/s1/model.pt`;
    - the exact V2.1c recipe: 6,000 games/round, `deep_fraction` 0.25 at depth 2, replay window 3,
      8 workers, 20 ungated rounds;
    - a new `10_000_000 × n` seed base;
    - copy every round's `candidate.pt` to `runs/<run_id>/candidates/` (~0.5 MB each).
  - Soups with `scripts/v2/soup.py` (PILA-8), five candidates in all:
    - H1 = avg(S1, c10, c20), the S1 shape (start plus two points of the trajectory);
    - H2 = avg(c11 … c20), SWA-10;
    - H3 = 0.5·S1 + 0.5·H2;
    - H4 = avg(c1 … c20), SWA-20;
    - c20, the endpoint, which also replicates §10.11 from a new start.
  - The PILA-8 health check (value log-loss on fresh shards) before any match.
- **Compute.** About 1.9 h, cap 2.8 h (1.5×).

  | Step | Size | Estimated time |
  |---|---|---|
  | 20-round run (win3 took 97 min) | — | ~100 min |
  | Soups and health check | 1,500 games | ~2 min |
  | Screen 5 nets vs S1 (Engineer, run's own seed band) | 2,000 pairs each | ~4 min |
  | Protocol run and confirmation of the winner (Evaluator) | 10,000 pairs × 2 | ~8 min |

- **Evaluation.** Protocol v1 vs S1 with best-of-5 selection, so a PROVISIONAL first block and then a
  confirmation block. Also report c20 vs S1 at 2,000 pairs, as the replicate of the §10.11 endpoint
  result.
- **Kill criterion.**
  - Drop any soup whose value log-loss is more than 0.01 worse than its worst ingredient.
  - If every soup screens below 50% vs S1, skip the protocol and record the null.
- **What a null would mean.** S1's gain was a one-off harvest of noise that had already
  accumulated, not a step that repeats. #4 would then be dropped. Comparing arms soup against soup
  stays, because it is still the fairer comparison.
- **Sources.** Izmailov et al. 2018, SWA, [arXiv:1803.05407](https://arxiv.org/abs/1803.05407);
  Wortsman et al. 2022, model soups, [arXiv:2203.05482](https://arxiv.org/abs/2203.05482).

### 3. Depth-3 targets on a slice of decisions

- **Hypothesis.** C2 (V2.1c) is the only training change that moved strength in this lineage. Its
  mechanism: the depth-2 expert that writes the targets is stronger than the depth-1 apprentice.
  Doubling the share of depth-2 rows bought nothing (V2.1d, deep_fraction 0.5: 49.3% vs V2.1c), so
  more rows from the same expert are saturated. A **stronger expert** has never been tried: depth 3
  beats depth 2 by 58.48% [56.94, 59.99] on V2.1c (4,000 games, PILA-6, seeds 5002200+c).
  Hypothesis: depth-3 targets on a small slice of decisions raise the net's fixed point, at
  +0 to +2pp vs S1 at depth 1, with probability ~0.25. The gain to the deployed agent at depth
  3 would be smaller.
- **What is different from the 2026-11-02 drop.** The owner dropped depth-3 targets in §10.9 as the
  cheaper alternative to C10. At that point:
  - no match had measured depth 3 vs depth 2 on this lineage (only 10 puzzles);
  - its cost was unknown.

  Now both are measured: +8.5pp of headroom, and 51 ms per decision. This idea **needs the
  owner's explicit go**, not just a promotion, because it reverses an owner decision.
- **Change.** Small, in `parchis/az/selfplay.py` and `config.py`:
  - Add a second gate, `deep3_fraction`, checked before the existing `deep_fraction` gate. When it
    fires, search at depth 3; those rows get the same blended targets as depth-2 deep rows.
  - Arm: `deep3_fraction=0.05`, `deep_fraction=0.20`, so the deep share stays 0.25 as in V2.1c.
    Everything else is the #2 harvest recipe, from S1, for 20 rounds, keeping every candidate.
  - With both fractions at 0, the code is byte-identical to today (add a test).
  - The no-code alternative, `deep_depth=3, deep_fraction=0.05`, is confounded: it also removes
    80% of the depth-2 rows. Do not use it.
- **Compute.** Estimate: ~13 min per round. That is V2.1c's ~5 min plus about 8 min for the depth-3
  slice (6,000 games × ~250 decisions × 0.05 × 51 ms ÷ 8 workers), plus 20–30% parallel slowdown.
  - Run: 20 rounds, ~4.5 h. If #1's pruning halves the cost of depth 3 first, ~3 h.
  - Matches: the arm's best soup vs S1, and its SWA-10 vs #2's SWA-10 (the matched control),
    10,000 pairs each, ~6 min.
- **Evaluation.** EVAL_PROTOCOL v1 promotion match vs S1 at depth 1. If it promotes, add a
  depth-2 confirmation study, since that is closer to how the agent is deployed.
- **Kill criterion.**
  - At round 10: 2,000 pairs vs #2's round-10 candidate below 47%.
  - Or the realised round time above 20 min, which is 1.5× the estimate.
- **Prior evidence.** `docs/V2_PLAN.md` §10.7 (C2), §10.9 (deep_fraction 0.5 flat, depth-3 targets
  dropped), §10.11 (20-round endpoints of the V2.1c recipe read 48.2–48.4% vs V2.1c), PILA-8 (why
  the comparison is soup against soup, with #2 as the matched control).
- **Sources.**
  - Veness, Silver, Uther, Blair 2009, *Bootstrapping from Game Tree Search*
    ([NeurIPS](https://papers.nips.cc/paper/3722-bootstrapping-from-game-tree-search)).
  - Wu 2019, KataGo playout-cap randomisation, [arXiv:1902.10565](https://arxiv.org/abs/1902.10565).

### 4. Weight averaging inside the loop (averaged generator and expert)

- **Hypothesis.** PILA-8 suggests that each round's trained net is a noisy sample around a better
  mean. S2 = avg(win3, win10) screened at 51.78% [50.23, 53.32] vs V2.1c (4,000 games, seeds
  11000000+c, a ranking screen only), while each of its ingredients loses to V2.1c (48.2%, 48.4%,
  20,000 games each). Today the loop generates self-play games and depth-2 targets with that noisy
  sample. Using an average of recent candidates as the generator **and** the depth-2 expert makes
  the expert stronger. That is the same mechanism as C2, the only training lever that has moved.
- **Expected effect.** +0.5 to +2pp vs #2's best soup, probability ~0.3 (hypothesis).
- **Change.**
  - `parchis/az/round_loop.py`: a `generator_avg_window=k` option. The self-play generator and
    deep-target searcher for round r+1 is the uniform average of the last k candidates; training
    still warm-starts from the latest candidate, as in standard SWA.
  - Arm: k = 5, from S1, otherwise the #2 recipe, keeping every candidate.
  - `generator_avg_window=0` keeps today's behaviour byte for byte (add a test).
- **Compute.** About 2 h of code plus ~1.9 h: a 20-round run (~100 min), soups, and matches.
- **Evaluation.** The arm's best soup vs S1 under protocol v1, and its SWA-10 vs #2's SWA-10 (the
  matched control) at 10,000 pairs.
- **Kill criterion.** At round 10, the averaged generator vs #2's avg(c6 … c10) at 2,000 pairs below
  48%.
- **Run only if** #2 is positive. If averaging after the fact does not compound, building it into
  the loop has no evidence behind it.
- **Sources.** Izmailov et al. 2018 (SWA); Tarvainen & Valpola 2017, *Mean teachers*
  ([arXiv:1703.01780](https://arxiv.org/abs/1703.01780)): an averaged model as the source of
  training targets.

### 5. TD(λ) value targets for cheap rows

- **Hypothesis.** Under the V2.1c recipe, 75% of decisions are "cheap": their value target is the
  game outcome alone, with λ = 0. Two findings point at those targets as the problem:
  - The value head reaches its best held-out loss at epoch 0–2. Only 18,000 independent outcomes
    sit behind 2.6M correlated rows (§10.10 reading 1).
  - Luck-adjusted scoring removes only 24% of variance (×1.3), so the head is under-resolved with
    respect to the dice (§10.8 reading 1).

  A TD(λ) return over the game's own sequence of decisions replaces most of the dice noise in the
  target with the net's later value estimates. This is TD-Gammon's target, it lowers target variance
  per row, and it adds no search cost.
- **Expected effect.** About +0.5 to +2.5pp, with probability ~0.3. Evidence that the mechanism
  works would be a rising C8 luck multiplier.
- **What is different from the ledger.**
  - "λ not implicated" (§10.1 item 7) concerned the blend at deep decisions.
  - deep_fraction 0.5 added more depth-2 targets.
  - Neither changes the 75% of rows trained on raw outcomes.
  - PILA-8 adds indirect support: averaging weights gained +3.1pp, which says per-round training
    is noise-dominated. TD(λ) attacks the same noise from the target side, so the two can stack.
- **Code.**
  - `selfplay.py`: record each decision's depth-1 root value vector. It is already computed during
    search; cheap rows discard it today.
  - `targets.py`: compute a vector-valued λ-return backward over each game, for cheap rows only.
    Deep rows keep the current blend.
  - Config: `cheap_value_target="td_lambda"`, `td_lambda=0.7`.
  - Tests: λ = 0 is byte-identical to today; λ = 1 reproduces the plain bootstrap chain.
- **Compute.** About 1.8 h.
  - Run: 20 rounds from S1, otherwise the exact #2 recipe, keeping every candidate, ~100 min.
  - Matches: SWA-10 vs #2's SWA-10 (the matched control), and the best soup vs S1, 10,000 pairs
    each, ~6 min.
- **Kill criterion.** If the round-10 candidate reads below 47% against #2's round-10 candidate at
  2,000 pairs, stop.
- **Sources.**
  - Tesauro 1995, TD-Gammon.
  - Sutton & Barto, ch. 12, λ-returns.
  - Veness, Silver, Uther, Blair 2009, *Bootstrapping from Game Tree Search*
    ([NeurIPS](https://papers.nips.cc/paper/3722-bootstrapping-from-game-tree-search)): training
    toward search values beats training toward outcomes.

### 6. 4-player bootstrap and evaluation harness (V2.3, stage 1)

- **Hypothesis.** No 4-player net exists (`docs/LEADERBOARD.md` gap 1). The 2-player loop's lessons
  carry over directly: generator temperatures, no gate, deep targets. Its fixed point does not:
  max^n search with three opponents is a different problem. Stage 1 builds the evaluation path and
  a supervised bootstrap net. Stage 2 (a separate issue) runs the V2.1c recipe at 4 players.
- **Code.**
  - An AZ-agent path in `parchis/evaluation/multiplayer_matrix.py` and `elo_ladder.py`. Today they
    load only PPO `.zip` files.
  - A 4-player duplicate format: rotate seats over the same dice.
  - The bootstrap: `BootstrapConfig(num_players=4)` on mixed heuristic / ε-heuristic games. The
    2-player recipe used 200k games (`docs/AZ_DESIGN.md`, Phase 2).
- **Expected effect.** A new axis, not measured in pp vs the 2-player champion. Gate: at depth 1
  the bootstrap net beats three tuned heuristics at one table with a win share above 25%, Wilson
  lower bound above 25%, on ≥4,000 games.
- **Compute.** About 4 h (estimate). A 4-player game costs ~3.5× a 2-player game (§4 C11).
  - Generation: 200k heuristic games, ~1 h.
  - Training: ~30 min.
  - Evaluation: ~1 h.
- **Kill criterion.** Stop if the bootstrap fails calibration (ECE > 0.02 on held-out data), or
  scores below the heuristic share.
- **Sources.**
  - Sturtevant 2002; Schadd & Winands 2011 (multiplayer search).
  - `docs/V2_PLAN.md` §4 C11.
- **Needs the owner.** Is 4 players a goal now? If yes, it moves up to #3, after pruned search and
  the PILA-5 harvest.

### 7. Per-head early stopping / lower value weight after epoch 1

- **Idea.** Today each round's training stops early on the *total* validation loss. That lets the
  value head degrade while the policy head keeps training: for 2×256 the chosen epoch's value loss
  is 0.5936 against a minimum of 0.5923 (§10.10 reading 3).
- **Change.** Keep the best value head and the best policy head separately, or drop
  `value_loss_weight` to 0.5 after the first epoch.
- **Expected effect.** Small, about 0 to +1pp.
- **Plan.** Same as item 5: a run from S1, judged soup against soup with #2 as the control.

### 8. Exploiter probe, then a league pool

- **Probe.** Train a copy of S1 for 10 rounds against frozen S1 only. Self-play from S1 is
  excluded, and the pool holds S1 alone.
- **Reading the result.**
  - If the exploiter beats S1 by more than 3pp, S1 has a hole the self-play pool never
    showed. The exploiter then joins the pool (AlphaStar-style league / prioritized fictitious
    self-play).
  - If not, single-lineage self-play is not the problem, and that is worth recording.
- **Supporting signal.** Bradley-Terry overdispersion (χ²/df ≈ 2.1) shows strength here is not
  one-dimensional, so exploitability is plausible.
- **Compute.** ~1.5 h.
- **Sources.** Vinyals et al. 2019, AlphaStar, *Nature* 575; Jaderberg et al. 2017, PBT
  ([arXiv:1711.09846](https://arxiv.org/abs/1711.09846)).

### 9. Chance-resolved ("depth 1.5") targets on cheap rows

- **Idea.** For cheap rows, use the mean of the six post-roll values as the bootstrap. That enforces
  pre-roll/post-roll consistency directly. The gate (< 0.5pp) is still unmet: 1.12pp for V2.1c
  (§10.7 item 4).
- **Ledger penalty.** It overlaps C3 (null) and deep_fraction 0.5 (null).
- **When to run.** Only as the follow-up arm if item 5 is flat.

### 10. Extra input features with zero-initialised weights

- **Idea.** Add features such as pip-count race, per-piece distance to home, and a count of
  safe-square moves. Their first-layer weights start at zero, so warm start from V2.1c is exact.
- **Target.** The roll under-resolution that C8 found (the raw net reads the roll only through
  threat features).
- **Cost and risk.** ~1 day of code and an encoding change, which also touches 4-player.

### 11. Exact endgame race database

- **Idea.** Once no contact is possible, the game is a pure race that dynamic programming can solve
  exactly, as GNU Backgammon's bear-off databases do. This would replace the net in those positions
  at play time and in targets.
- **State space.** Needs measuring first (a 5-minute diagnostic).
- **Expected effect.** Small, because races are a minority of decisions.

### 12. Population of recipe replicates, souped rather than selected

- **Idea.** Three seed replicates of the recipe from S1, averaged rather than picked.
- **After PILA-8.** Selection harvests about +1.3pp from the best of 3 draws. Averaging beat that in
  PILA-8: three nets gave +3.1pp. Replicates are now soup ingredients, and more independent
  ingredients may average out more noise than one trajectory's candidates.
- **Ranking.** Five hours. #2 tests the cheaper single-trajectory version first; run this only if
  #2 is positive and its soups gain more from more ingredients (H4 over H2).

### 13. Not recommended: revive PPO, or PUCT-MCTS on PPO checkpoints

- **PPO checkpoints.** They are not on this machine.
- **PUCT-MCTS.** The PPO line's MCTS gave 56.2% [51.4, 61.0] over 400 games over plain PPO
  inference (`docs/SEARCH_MCTS.md`). Its iterative loop regressed to ~40%.
- **Why not.** The AZ line already has exact expectimax with a better net. PILA-6 and item 1
  cover search at play time.

---

## Closed since the first version

### Weight soups of existing checkpoints (PILA-8, was #3): positive, new champion

- **Verdict (Evaluator, 2026-11-16): PASS** under protocol v1, best-of-4 selection, code `eb1c8bb`.
  Duplicate pairs at depth 1, Wilson 95% CIs:
  - S1 = avg(V2.1c, win3, win10) vs V2.1c, confirmation block 4: **52.79% [52.09, 53.48]**, 20,000
    games, seeds 5004000+c (c < 8).
  - First run, block 3 (PROVISIONAL because of the selection): 53.38% [52.69, 54.07], 20,000 games,
    seeds 5003000+c.
  - Both blocks were drawn after the selection, so they pool: 53.08% [52.60, 53.57] over 40,000
    games, about +21 Elo.
  - Generalisation anchor, block 4: S1 74.00% [72.03, 75.88] vs the tuned heuristic, against
    V2.1c's 71.85% [69.84, 73.78] on the same 2,000 games (seeds 5004100+c). S1 vs random: 97.10%.
  - Engineer's screen (2,000 pairs each, seeds 11000000+c in the run's training band; a ranking
    only): S1 54.17%, S4 53.70%, S2 51.78%, S3 51.78%. Every soup's value log-loss was below that
    of all its ingredients.
- **What it changes.**
  - S1 is the champion once PR #7 merges. Every item now starts from S1 and is measured against it.
  - The PILA-5 harvest moves to #2, re-designed around S1. Training arms are judged soup against
    soup, with #2 as their control.
  - New #4: averaging inside the loop.
- **What my predictions got wrong.** I expected 0 to +1.5pp with probability ~0.3. The model behind
  that was "the soup lands at the centre of the ±1.5pp checkpoint scatter". The measured gain is
  +3.1pp, twice the top of that range, and the soup beats V2.1c, the strongest of its ingredients. S2
  makes the point most clearly: avg(win3, win10) screened above V2.1c, although both of its
  ingredients lose to V2.1c. So the error that averaging removes is per-round training noise, and
  it is larger than the scatter between selected checkpoints suggested.
- **Calibration for later screens.**
  - The best-of-4 screen overstated the protocol result by about 1.1pp (54.17% vs 53.08%).
  - Value log-loss on self-play data moved by only ~0.001 for a +3pp gain, and it ranked S4 above
    S1. Use it as a barrier check, not to pick a winner.
- **Record.** `runs/soups_2026-11-16/` (PR #6); protocol records
  `runs/soups_2026-11-16/s1/eval/protocol_b3.json` and `protocol_b4.json`; leaderboard and champion
  pointer in PR #7.

### Play-time search depth on V2.1c (PILA-6, was #1): positive

- **Verdict (Evaluator, 2026-11-16).** Search depth is the largest measured lever. Results, all
  duplicate pairs with Wilson 95% CIs, seed block 2 (kind `study`), code `c621f7c`:
  - Depth 2 vs depth 1: 60.39% [59.72, 61.07], 20,000 games, seeds 5002000+c. The prediction was
    +5 to +8pp; the measurement is +10.4pp.
  - Depth 3 vs depth 2: 58.48% [56.94, 59.99], 4,000 games, seeds 5002200+c. The prediction was
    +1 to +4pp; the measurement is +8.5pp.
  - V2.1c vs V2.0c: 52.91% at depth 1 and 51.65% at depth 2, on the same deals, seeds 5002100+c.
- **What it changes.**
  - Recommended play depth: 3, at 51 ms per decision on average and 161 ms at most.
  - Pruned deeper search moves from #6 to #1. Depth-3 targets are re-proposed (now #3).
  - Net-only ideas are worth less to the deployed agent than their depth-1 pp suggest.
- **What my predictions got wrong.** I expected a smaller depth-2 gain on V2.1c than on round 23,
  because V2.1c's depth-1/depth-2 inconsistency is smaller (§10.7 item 4). The gain was larger
  (+10.4 vs +7.1pp). So a better net does not make shallow search redundant here; it makes deeper
  search pay more.
- **Record.** `runs/depth_study_v2_1c/` (`meta.json` validates). Leaderboard section and caveat 5
  in PR #5.

---

## Tried-ideas ledger

Kept current after every verdict. Sources: `docs/V2_PLAN.md` §§1–10, `docs/LEADERBOARD.md`,
`docs/SEARCH_MCTS.md`.

| Idea | Result | Why it did or didn't work | Where |
|---|---|---|---|
| PPO + reward shaping, single-snapshot self-play | Superseded | Simplified rules, an observation that couldn't tell pieces apart, single-seed sweeps | RL_DESIGN_REVIEW |
| PUCT-MCTS on a PPO checkpoint (play time) | +: 56.2% [51.4, 61.0], 400 games | Search adds information the raw policy lacks | SEARCH_MCTS |
| PUCT-MCTS iterative self-play (PPO) | −: regressed to ~40% | Gate never promoted, so the data never improved, and the candidate overfit | SEARCH_MCTS |
| Phase 3 gated self-play, 132 rounds | Plateau | Temperature/Dirichlet scale bug made generation near-random; the gate froze the lineage | V2_PLAN §1–2 |
| V2.0: temperature scale fix, no gate | Weak +: 54.4% vs round 23 | The generator started playing real moves; flat after round 5 | §10 |
| V2.0b: τ_end 0.005, lr 3e-4, pool changes | +: 54.4% vs V2.0; 72% vs heuristic | Less exploration loss; warm starts nudge instead of re-fitting | §10.4 |
| C3 afterstate rows (V2.1a) | 0: 51.1% [49.6, 52.6] | Its bootstrap target equals the net's own value, so the row's only signal is a half-weight outcome | §10.4 |
| V2.0c: 20 more rounds of the V2.0b recipe | 0: 49.7% | The recipe saturates; more data at depth 1 adds nothing | §10.6 |
| C6 throughput | Infra: 80 games/s on 8 workers | — | §10.5 |
| **C2 deep targets, 0.25 (V2.1c)** | **+: 52.5% [51.8, 53.2], 20k games** | **The expert is finally stronger than the apprentice** | §10.7 |
| deep_fraction 0.5 (V2.1d) | 0: 49.3% vs V2.1c | A quarter of the decisions already carries the signal | §10.9 |
| C8 luck-adjusted scoring | Built; ×1.3 only | The value head is under-resolved with respect to the roll | §10.8 |
| C10 capacity 3×512 | 0 (pre-check) | Value overfits at once; capacity is not binding at 6k games/round | §10.10 |
| Depth-3 targets | Dropped by the owner, never run | Dropped before depth 3 was measured; re-proposed after PILA-6 (now #3) | §10.9 note |
| Replay window 10 vs 3 | 0: 49.2% | Stale games cancel the extra outcomes | §10.11 |
| Continue V2.1c 20 rounds (endpoint) | −: 48.2% / 48.4% vs V2.1c | Most likely per-round training noise: averaging these two endpoints with V2.1c gives +3.1pp (PILA-8) | §10.11 |
| Depth 2 vs 1 at play time, same net | +: 57.1% [53.7, 60.5], 800 games (round 23) | Superseded on V2.1c by PILA-6 (next row) | M9 |
| **Play-time depth on V2.1c (PILA-6)** | **+: d2 vs d1 60.39% [59.72, 61.07], 20k games; d3 vs d2 58.48% [56.94, 59.99], 4k games** | **Returns not diminishing yet; deeper search shrinks net gaps (V2.1c vs V2.0c 52.9% → 51.7%)** | LEADERBOARD (PR #5), `runs/depth_study_v2_1c/` |
| **Weight soups of existing checkpoints (PILA-8)** | **+: S1 = avg(V2.1c, win3, win10) vs V2.1c 52.79% [52.09, 53.48], 20k games, confirmation block 4; pooled 53.08% [52.60, 53.57], 40k games** | **Shared lineage and LayerNorm, so no loss barrier; the average most likely cancels per-round training noise; it beats V2.1c, its strongest ingredient** | LEADERBOARD (PR #7), `runs/soups_2026-11-16/` |

## Literature consulted

- Tesauro 1995, TD-Gammon; GNU Backgammon rollouts and bear-off databases.
- Silver et al. 2017/2018, AlphaGo Zero / AlphaZero.
- Wu 2019, KataGo, [arXiv:1902.10565](https://arxiv.org/abs/1902.10565).
- Antonoglou et al. 2022, Stochastic MuZero.
- Danihelka et al. 2022, *Policy improvement by planning with Gumbel*
  ([ICLR](https://iclr.cc/virtual/2022/poster/6418)). It is relevant only if the project moves to
  sampled search with few simulations. Exact expectimax at depth 1–2 already sees every root move,
  so it is not ranked.
- Veness et al. 2009, *Bootstrapping from Game Tree Search*
  ([NeurIPS](https://papers.nips.cc/paper/3722-bootstrapping-from-game-tree-search)).
- Izmailov et al. 2018, SWA, [arXiv:1803.05407](https://arxiv.org/abs/1803.05407).
- Tarvainen & Valpola 2017, *Mean teachers are better role models*,
  [arXiv:1703.01780](https://arxiv.org/abs/1703.01780).
- Wortsman et al. 2022, model soups, [arXiv:2203.05482](https://arxiv.org/abs/2203.05482).
- Jaderberg et al. 2017, PBT, [arXiv:1711.09846](https://arxiv.org/abs/1711.09846).
- Vinyals et al. 2019, AlphaStar (league training).
- Ballard 1983, *The \*-minimax search procedure for trees containing chance nodes*.
- Hauk, Buro, Schaeffer 2004, *\*-Minimax performance in backgammon*.
- Sturtevant 2002; Schadd & Winands 2011 (multiplayer search).
- Superhuman Stratego with self-play RL and test-time search,
  [arXiv:2511.07312](https://arxiv.org/abs/2511.07312).
- Ludo-specific work: Alvi & Ahmed 2011; Alhajry, Alvi & Ahmed 2012 (TD(λ)/Q-learning against
  heuristics). Nothing peer-reviewed beyond that, and nothing on Parchís.
