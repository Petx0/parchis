# Research backlog

Ranked list of experiments most likely to make the agent stronger. Maintained by the Researcher
(PILA-4). The list is re-ranked after every verdict, and the ledger at the bottom records every idea
already tried. **First version: 2026-11-15. Re-ranked 2026-11-16 after the PILA-6 depth study, again the
same day after the PILA-8 verdict (weight soups), a third time after the PILA-10 design
challenge, and a fourth time after the PILA-5 verdict (harvest from S1).**

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
  matched control. **That control now exists** (`runs/pila5_harvest_s1/`, PR #10): candidates
  c01–c20, soups h1–h4, and their screens vs S1 on seeds 21000000+c (c20 45.77%, SWA-10 50.52%,
  SWA-20 52.70%; 4,000 games each, a ranking only).
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

Items keep their `#` as a stable ID; the **Rank** column orders them. Items 14–17 come from the
PILA-10 design challenge (`docs/DESIGN_CHALLENGE.md`).

| Rank | # | Idea | Status | Expected effect | Compute (Mac) | Issue |
|---|---|---|---|---|---|---|
| 1 | 1 | Pruned and time-budgeted deeper search (Star2, depth 4 under a time cap) | not tried; unlocked by PILA-6 | **+3 to +7pp at play time**, depth 4 vs depth 3 | ~1 day code + ~1.5 h | — |
| 2 | 14 | **Search follows the six-again rule after a bonus** (rules mismatch found in PILA-10), flagged; depth-2 study | **new**; correctness; measured: depth-2 move changes 0.28×/game, capture-on-6 undervalued ~2pp | +0 to +0.7pp at depth 2–3 | ~2 h code + ~35 min | PILA-11 |
| 3 | 12 | **Soup of S1 with independent replicates** (re-scoped after PILA-5: one more 20-round run, window 10, reuses PILA-5's c20; one pre-registered soup) | **re-scoped**; raised from rank 14 by PILA-5 | +0 to +2pp vs S1 | ~30 min code + ~1.8 h | — |
| 4 | 3 | Depth-3 targets on a slice of decisions | dropped by the owner 2026-11-02 before depth 3 was measured; **re-proposed, see what's different**; control now exists | +0 to +2pp | ~1 h code + ~4.5 h (~3 h if #1 lands first) | after #14 |
| 5 | 15 | **Offline representation probe**: Markov flags, side-to-move shots / opponent / blockade features, piece-permutation augmentation, conv and token trunks, symmetric aux targets | **new**; subsumes #10 | go/no-go per arm | ~1 day code + ~1.8 h | PILA-12 |
| 6 | 16 | **KPI validation** against PILA-1 ratings, plus d3-vs-d1 headroom (Evaluator) | **new** | enables early kills | ~2 h code + ~40 min | PILA-13 |
| 7 | 17 | **Low-variance value targets for cheap rows**: luck-corrected return (unbiased) vs TD(λ) | **new**; replaces #5; control now exists | +0.5 to +2pp vs #2's control | ~1 day code + ~4.1 h | PILA-14 |
| 8 | 6 | 4-player bootstrap and 4-player evaluation harness (V2.3 stage 1) | not tried; new axis | new axis (target: beat the tuned heuristic at a 4-player table) | 1–2 days code + ~4 h | — |
| 9 | 8 | Exploiter probe against frozen S1, then a league pool | not tried; **prior raised a little by PILA-5** (h4 gains in-lineage, loses vs the heuristic) | 0 to +2pp if S1 is exploitable | ~1.5 h probe | — |
| 10 | 7 | Per-head early stopping / lower value weight after epoch 1 | not tried (§10.10 reading 3) | 0 to +1pp | ~2 h code + ~1.9 h | — |
| 11 | 4 | Weight averaging inside the loop (averaged generator and expert); EMA-of-weights arm | **lowered by PILA-5**: post-hoc averaging of one trajectory gave only +0.8pp | +0 to +1pp vs S1 | ~2 h code + ~1.9 h | only if #12 is positive |
| 12 | 9 | Chance-resolved ("depth 1.5") targets on cheap rows | partly tried: C3 and deep_fraction 0.5 | 0 to +1pp | ~1 day code + ~2 h | only if #17 is flat |
| 13 | 11 | Exact endgame race database (bear-off analogue) | not tried | +0.5 to +1pp at play time | 2–3 days code + hours of DP | — |
| — | 2 | Harvest a 20-round continuation from S1, then soup its trajectory | **closed (PILA-5): FAIL**, h4 vs S1 50.83% [50.14, 51.53], heuristic guard −3.5pp | — | — | PILA-5 |
| — | 5 | TD(λ) value targets for cheap rows | **superseded by #17** (now one of its two arms) | — | — | — |
| — | 10 | Extra input features, zero-initialised | **superseded by #15** (feature arm, with an offline gate) | — | — | — |
| — | 13 | Revive PPO, or PUCT-MCTS on PPO checkpoints | **not recommended** | — | — | — |

**What changed after PILA-5 (2026-11-16, fourth re-rank: harvest from S1).** The harvest (#2) is
closed: SWA-20 of the trajectory (h4) beat S1 head to head by +0.83pp, 50.83% [50.14, 51.53] over
20,000 games (block 5), but failed the heuristic guard by 0.5pp (closed below). Four consequences:
- **Averaging compounds, but weakly.** The second averaging step returned about a quarter of the
  first (+0.8pp vs PILA-8's +3.1pp). That sits at the low end of my +0 to +1.5pp prediction.
- **Independent ingredients look like the active part (hypothesis).** S1 averaged its start with
  two *independent* runs. h1 = avg(S1, c10, c20) has the same shape but two *correlated* points of
  one trajectory, and it screened at 50.35% [48.80, 51.90]. #12 tests independence directly, and it
  is cheap now that PILA-5's c20 is one ingredient. It moves from rank 14 to rank 3.
- **#4 (averaging inside the loop) drops to rank 11.** Its premise was that averaging compounds
  strongly; PILA-5 says +0.8pp. It now waits for #12.
- **Screens and guards.** The best-of-5 screen read 1.9pp high (52.70% vs 50.83%). From now on each
  arm pre-registers **one** soup (`--selection fixed`, no confirmation block), and the Engineer's
  screen includes the heuristic guard on the same deals, so a guard failure is found before the
  protocol run.

**What changed after PILA-10 (2026-11-16, third re-rank: design challenge).** A first-principles
review of the representation, targets and metrics (`docs/DESIGN_CHALLENGE.md`) adds four items:
- **#14 (rank 2): a rules mismatch in search.** A capture or finish made with a 6 is followed in
  the search by the opponent's turn, while the rules give the mover a reroll. Depth ≥ 2 play and the
  depth-2 deep targets undervalue capturing with a 6. It is cheap, it is correctness, and it should
  land before #1's pruning (same file) and before #3 (whose targets it changes).
- **#15 (rank 5): an offline representation probe.** It tests the encoding's Markov gaps, the
  missing side-to-move "shots" features (the likely cause of C8's under-resolved value head),
  piece-permutation augmentation, structured trunks and symmetric aux targets. All arms are fitted
  on one frozen dataset, and only the winners go to self-play. It subsumes #10.
- **#16 (rank 6): KPI validation.** Checks five cheap between-promotion KPIs against the PILA-1
  ratings, and measures d3 vs d1 directly.
- **#17 (rank 7): low-variance value targets.** Replaces #5 with two arms: the unbiased C8
  luck-corrected return and TD(λ).

Two framing facts from PILA-10:
- **The skill headroom above S1 at depth 1 is at least ~130 Elo** (search alone; chained, to be
  measured by #16).
- Per-round training is noise-dominated (PILA-8). Effects below ~1pp are not worth designing for
  unless they are free.

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

### 2. Harvest a continuation from S1, then soup its trajectory (PILA-5): closed

Closed by the PILA-5 verdict (FAIL, 2026-11-16). The result, what it changes and the full original
design are under "Closed since the first version" below and in PR #8 (`scripts/v2/run_harvest.py`).
Its run, `runs/pila5_harvest_s1/`, is the matched control for #3, #7 and #17.

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
- **Expected effect.** +0 to +1pp vs S1, probability ~0.2 (hypothesis; lowered after PILA-5, was
  +0.5 to +2pp at ~0.3). Post-hoc SWA-20 of one trajectory from S1 gained only +0.83pp
  (50.83% [50.14, 51.53], 20,000 games, seeds 5005000+c).
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
- **Run only if** #12 is positive. #2 was positive head to head but small (+0.83pp) and failed the
  heuristic guard. If independent replicates do not lift the soup either, averaging has run out at
  this level and building it into the loop has little evidence behind it.
- **Sources.** Izmailov et al. 2018 (SWA); Tarvainen & Valpola 2017, *Mean teachers*
  ([arXiv:1703.01780](https://arxiv.org/abs/1703.01780)): an averaged model as the source of
  training targets.

### 5. TD(λ) value targets for cheap rows

**Superseded by #17 (PILA-14, 2026-11-16):** TD(λ) is now one of two arms, next to the unbiased
luck-corrected return. The text below is kept for the record.

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
- **Supporting signals.**
  - Bradley-Terry overdispersion (χ²/df ≈ 2.1) shows strength here is not one-dimensional, so
    exploitability is plausible.
  - PILA-5: h4 beat S1 (50.83%, 20,000 games) and V2.1c (54.40% [52.21, 56.57], 2,000 games), yet
    lost 3.5pp to S1 against the tuned heuristic on the same 2,000 deals (73.75% vs 77.25%, block
    5). A net that gains inside its own lineage and loses outside it is what non-transitive
    self-play looks like. One run, so a hypothesis.
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

**Superseded by #15 (PILA-12, 2026-11-16):** the feature arm of the offline representation probe,
with a named mechanism (side-to-move shots) and an offline gate before any self-play.

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

### 12. Soup of S1 with independent replicates (re-scoped after PILA-5)

- **Hypothesis.** S1's +3.1pp came from averaging V2.1c with two **independent** 20-round
  continuations (win3, win10: different seeds and replay windows). PILA-5 repeated the averaging
  from S1 with points of **one** trajectory, and the exact S1 shape built from correlated points,
  h1 = avg(S1, c10, c20), screened at 50.35% [48.80, 51.90] (4,000 games, seeds 21000000+c). The
  best soup, SWA-20, gained only +0.83pp under the protocol. Two readings remain:
  - (a) S1 is already near the centre of the basin, so any second averaging step returns little;
  - (b) correlated ingredients cancel little noise. Consecutive warm-started rounds share most of
    their training data (replay window 3) and their whole history.

  Repeating S1's construction one level up, with an independent replicate, separates the two. If
  (b) is right, R = avg(S1, A, B) beats S1 by roughly the PILA-8 margin, with diminishing returns.
- **Expected effect.** +0 to +2pp vs S1, probability ~0.3 (hypothesis).
- **What is different from what was tried.**
  - PILA-8 souped V2.1c with two independent runs; this souping starts from S1, the new champion.
  - PILA-5 souped one trajectory's correlated candidates; this uses an independent second run.
  - The earlier version of this item asked for three new replicates (~5 h). PILA-5's c20 is
    already one independent ingredient, so one new run is enough.
- **Change.**
  - `scripts/v2/run_harvest.py`: add a `--window` option (replay window, default 3, so PILA-5
    stays reproducible) and a soup spec for R. About 30 min of code.
  - Run B: from S1, the V2.1c recipe with **replay window 10**, 20 ungated rounds, a new
    `10_000_000 × n` seed base, keeping every candidate. Window 10 mirrors win10, S1's second
    ingredient.
  - **Pre-registered candidate:** R = avg(S1, A, B), where A = PILA-5's c20
    (`runs/pila5_harvest_s1/candidates/c20.pt`) and B = run B's c20. This is S1's exact shape.
  - Diagnostics only, never promoted from this issue: R' = avg(S1, h2, SWA-10 of B), and
    B's c20 vs S1.
- **Compute.** About 1.8 h, cap 2.7 h (1.5×), one heavy job.

  | Step | Size | Estimated time |
  |---|---|---|
  | Run B, 20 rounds (PILA-5 took 96.7 min at window 3; window 10 trains on more rows) | — | ~100–110 min |
  | Soup R, health check | 1,500 games | ~2 min |
  | Screen R vs S1 on PILA-5's screening deals (21000000+c), plus R and S1 vs the tuned heuristic on the same 1,000 pairs | 2,000 + 1,000 + 1,000 pairs | ~3 min |
  | Protocol v1 on R vs S1, `--selection fixed` (Evaluator) | 10,000 pairs + guards | ~4 min |

- **Evaluation.** Protocol v1 vs S1 at depth 1, `--selection fixed`, because R is named here before
  any result exists. No confirmation block is needed (EVAL_PROTOCOL §4).
- **Kill criteria.**
  - R's value log-loss more than 0.01 worse than its worst ingredient's: stop.
  - R screens below 50.5% vs S1 at 2,000 pairs, or more than 2pp below S1 against the heuristic on
    the same deals: skip the protocol and record the result.
- **What each outcome means.**
  - R promotes: independence is the active part. #4 is then re-read, and more replicates (C, D)
    become the cheapest strength left.
  - R is flat: reading (a). Weight averaging has run out at this level, #4 is dropped, and the
    soup-against-soup comparison stays only as the fairer way to judge training arms.
- **Sources.** Wortsman et al. 2022, model soups, [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)
  (souping independently fine-tuned models from one start); Izmailov et al. 2018, SWA,
  [arXiv:1803.05407](https://arxiv.org/abs/1803.05407); PILA-8 and PILA-5 records.

### 13. Not recommended: revive PPO, or PUCT-MCTS on PPO checkpoints

- **PPO checkpoints.** They are not on this machine.
- **PUCT-MCTS.** The PPO line's MCTS gave 56.2% [51.4, 61.0] over 400 games over plain PPO
  inference (`docs/SEARCH_MCTS.md`). Its iterative loop regressed to ~40%.
- **Why not.** The AZ line already has exact expectimax with a better net. PILA-6 and item 1
  cover search at play time.

### 14. Search follows the six-again rule after a bonus (PILA-11)

- **Finding (PILA-10, code reading).** `Game.play_turn` gives a reroll after a 6 even when the move
  triggers a capture or finish bonus. `search.py` decides six-again with
  `pending_bonus is None and roll == 6`, and enters the bonus decision with `roll=None`. So inside
  the search the opponent moves next.
- **Consequence.** At depth ≥ 2, and in the depth-2 deep targets, capturing or finishing with a 6
  is valued without its reroll.
- **Plan.** A flagged fix (legacy stays reproducible), parity tests against the engine, then a
  10,000-pair study of S1 fixed vs S1 legacy, both at depth 2 (~35 min).
- **Ordering.** Arms compared against the PILA-5 control keep the legacy flag.
- **Details.** `docs/DESIGN_CHALLENGE.md` §1.1.

### 15. Offline representation probe (PILA-12)

- **Arms.** Seven arms fitted from scratch on one frozen S1 dataset, two seeds each: Markov flags,
  side-to-move shots plus opponent and blockade features, piece-permutation augmentation, a
  circular-conv stem, a token trunk, and symmetric aux targets.
- **Go bar.**
  - held-out value loss −0.002 and more than twice the seed spread, or decision error vs a depth-3
    reference −10%;
  - **and** inference ≤ 2× the MLP's.
- **After the probe.** Each winner gets its own self-play issue, judged against the PILA-5 control.
- **Details.** `docs/DESIGN_CHALLENGE.md` §1–2, §3.4.

### 16. KPI validation and headroom (PILA-13, Evaluator)

- **Method.** Five candidate between-promotion KPIs are computed for the 15 rated nets and
  rank-correlated with the PILA-1 Elo:
  - K1 luck multiplier;
  - K2 decision error vs a depth-3 reference;
  - K3 frozen-set calibration;
  - K4 search consistency;
  - K5 high-stakes accuracy.

  The study also plays S1 d3 vs S1 d1 at 1,000 pairs.
- **Cost.** ~40 min.
- **Details.** `docs/DESIGN_CHALLENGE.md` §4.

### 17. Low-variance value targets for cheap rows (PILA-14)

- **Arms.** Two 20-round arms from S1 on the PILA-5 recipe:
  - the luck-corrected return z̃ = outcome − Σ future luck, which is unbiased (the C8 control
    variate used as a training target);
  - TD(λ = 0.7).
- **Judging.** Soup against soup, with PILA-5 as the control.
- **Cost.** ~4.1 h.
- **Details.** `docs/DESIGN_CHALLENGE.md` §3.3.

---

## Closed since the first version

### Harvest from S1 and soup its trajectory (PILA-5, was #2): FAIL, small head-to-head gain

- **Verdict (Evaluator, 2026-11-16): FAIL** under protocol v1, best-of-5 selection, code `13cd24f`.
  Candidate h4 = SWA-20 = avg(c01 … c20), `runs/pila5_harvest_s1/soups/h4/model.pt` (sha256
  `d7b2b08552b5`). Duplicate pairs at depth 1, seed block 5, Wilson 95% CIs:
  - h4 vs S1 (decisive): **50.83% [50.14, 51.53]**, 20,000 games, seeds 5005000+c (c < 8). The
    lower bound clears 50%; about +6 Elo.
  - Heuristic guard: h4 73.75% [71.78, 75.63] against S1's 77.25% [75.36, 79.03] on the same 2,000
    games (seeds 5005100+c). Δ −3.50pp against a −3pp limit: **fails**. On an unpaired SE (1.36pp)
    the drop is 2.6σ, and the paired SE on shared deals is smaller, so I read it as probably real
    rather than bad luck. S1 scored 74.0% and 74.2% on blocks 3 and 4, so these deals are easier
    than average; that does not change the same-deal comparison.
  - h4 vs random 98.25% [97.58, 98.74]; h4 vs V2.1c 54.40% [52.21, 56.57], 2,000 games, seeds
    5005300+c.
  - Under §4.5 a FAIL is final: h4 is not retried on new seeds.
- **The run.** 20 ungated rounds of the V2.1c recipe (window 3) from S1, 120,000 games, 96.7 min;
  every candidate kept. Engineer's screen vs S1 (2,000 pairs each, seeds 21000000+c in the run's
  band; a ranking only): h4 52.70%, h3 51.08%, h2 50.52%, h1 50.35%, c20 45.77% [44.24, 47.32].
  Every soup passed the value log-loss health check.
- **What it changes.**
  - S1 stays champion. The run is the matched control for #3, #7 and #17.
  - #12 (independent replicates) moves up to rank 3; #4 (averaging inside the loop) moves down to
    rank 11 and waits for #12.
  - Each arm pre-registers one soup, and the screen includes the heuristic guard.
- **What my prediction got right and wrong.** I predicted +0 to +1.5pp vs S1 with probability
  ~0.35. The head-to-head result (+0.83pp) is inside that range. I did not anticipate a guard
  failure: I assumed averaging only removes noise, and noise removal should not cost strength
  against an outside opponent. h4 gains inside its lineage (vs S1, vs V2.1c) and loses outside it,
  which points to some drift in the trajectory's style, not only noise.
- **Other findings.**
  - The endpoint loses to its start again, and by more than in §10.11: c20 45.77% vs S1, against
    48.2–48.4% for the V2.1c endpoints vs V2.1c. Twenty rounds of the recipe from a soup start drift
    further downhill than from a trained start (hypothesis: the soup sits off the trajectory's own
    path, so the first rounds pull it away).
  - Winner's curse: the best-of-5 screen read 1.9pp high (52.70% vs 50.83%); best-of-4 in PILA-8
    read 1.1pp high.
  - More ingredients screened better (SWA-20 52.70% vs SWA-10 50.52%, same deals), which fits
    noise cancelling with N but is inside the curse.
  - The round-11 "PROMOTED" log label was cosmetic: the gate is off in the V2.1c recipe (PR #10
    renames it).
- **Record.** `runs/pila5_harvest_s1/` (PR #10); protocol record
  `runs/pila5_harvest_s1/soups/h4/eval/protocol_b5.json` and leaderboard in PR #11 (stacked on #10).

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
| Harvest from S1, SWA-20 of one trajectory (PILA-5) | FAIL: h4 vs S1 50.83% [50.14, 51.53], 20k games, block 5; heuristic guard Δ −3.5pp (73.75% vs 77.25%, 2k games) | A second averaging step from S1 returns ~¼ of the first; correlated ingredients from one trajectory (hypothesis); in-lineage gain with an out-of-lineage loss | LEADERBOARD (PR #11), `runs/pila5_harvest_s1/` |

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
- Veness, Lanctot & Bowling 2011, *Variance Reduction in Monte-Carlo Tree Search* (control variates
  at chance nodes), NeurIPS.
- Zaheer et al. 2017, *Deep Sets*; Lee et al. 2019, *Set Transformer* (permutation-symmetric trunks).
- Ludo-specific work: Alvi & Ahmed 2011; Alhajry, Alvi & Ahmed 2012 (TD(λ)/Q-learning against
  heuristics). Nothing peer-reviewed beyond that, and nothing on Parchís.
