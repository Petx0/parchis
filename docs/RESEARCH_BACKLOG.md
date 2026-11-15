# Research backlog

Ranked list of experiments most likely to make the agent stronger. Maintained by the Researcher
(PILA-4). The list is re-ranked after every verdict, and the ledger at the bottom records every idea
already tried. **First version: 2026-11-15.**

How to read it:
- **Champion:** V2.1c, `runs/v2_1c_champion/model.pt`, played at depth 1 (`docs/LEADERBOARD.md`).
- **"pp"** means percentage points of head-to-head win rate against the champion in duplicate
  pairs. Every expected effect below is a **hypothesis** unless a recorded run is cited.
- **Ranking** uses expected gain × probability ÷ compute. Ideas that open a new axis (search at
  play time, 4 players) get a bonus. Ideas that resemble a closed negative result get a penalty.
- **Evaluation** follows the interim protocol (`parchis-eval-protocol` skill) until PILA-2 merges
  `docs/EVAL_PROTOCOL.md`:
  - 10,000 duplicate pairs (20,000 games).
  - Wilson 95% lower bound above 50% to promote.
  - Anything chosen as best of N is re-confirmed on fresh seeds.
  - The Evaluator allocates every new seed range.
- **Training seeds** use a base of `10_000_000 × n`, with n unique per run (lab rules).
- **Compute** is wall-clock on the Mac mini (M4, 10 cores), one heavy job at a time.

### Timing baseline (diagnostic, 2026-11-15)

This is a sizing measurement, not a strength claim. V2.1c played itself in one process with
one-thread BLAS, both seats searching, seed 990000, script `diag/depth_timing.py` (not committed).

| Depth | Games | Seconds per game | Approx. ms per move |
|---|---|---|---|
| 1 | 20 | 0.04 | 0.15 |
| 2 | 8 | 0.49 | 1.6 |
| 3 | 2 | 9.9 | 33 |

Each extra ply costs about 12–20× more. Depth 3 is cheap enough for interactive play but expensive
to evaluate at 10,000 pairs. The depth-3 figure comes from only 2 games, so read it as an order of
magnitude.

---

## Ranked list

| # | Idea | Status | Expected effect | Compute (Mac) | Issue |
|---|---|---|---|---|---|
| 1 | Play-time search depth on V2.1c | not tried on V2.1c (+7pp measured on round 23) | **+5 to +8pp** at depth 2 vs depth 1 | ~1.5 h, measurement only | PILA-6 (re-scoped) |
| 2 | Weight soups of existing checkpoints | not tried (named open in §10.11) | 0 to +1.5pp | ~1 h code + ~20 min | new (backlog) |
| 3 | Continuation that keeps every candidate, then SWA and selection | not tried; differs from §10.11 | +0.5 to +1.5pp | ~2.5 h | PILA-5 (re-scoped) |
| 4 | TD(λ) value targets for cheap (outcome-only) rows | not tried | +0.5 to +2.5pp | ~1 day code + ~1.8 h | — |
| 5 | 4-player bootstrap and 4-player evaluation harness (V2.3 stage 1) | not tried; new axis | new axis (target: beat the tuned heuristic at a 4-player table) | 1–2 days code + ~4 h | — |
| 6 | Pruned deeper search (Star2 + policy-ordered moves) for depth 3–4 | not tried (C7 named it) | depth-3 strength at a fraction of the cost | ~1 day code + ~1 h | depends on #1 |
| 7 | Per-head early stopping / lower value weight after epoch 1 | not tried (§10.10 reading 3) | 0 to +1pp | ~2 h code + ~1.8 h | — |
| 8 | Exploiter probe against frozen V2.1c, then a league pool | not tried | 0 to +2pp if V2.1c is exploitable | ~1.5 h probe | — |
| 9 | Chance-resolved ("depth 1.5") targets on cheap rows | partly tried: C3 and deep_fraction 0.5 | 0 to +1pp | ~1 day code + ~2 h | only if #4 is flat |
| 10 | Extra input features, zero-initialised for warm start | not tried | 0 to +1.5pp | ~1 day code + ~1.8 h | — |
| 11 | Exact endgame race database (bear-off analogue) | not tried | +0.5 to +1pp at play time | 2–3 days code + hours of DP | — |
| 12 | Population of recipe replicates, keep the best | resembles §10.9/§10.11 | +0.5 to +1pp, mostly selection | ~5 h | — |
| 13 | Revive PPO, or PUCT-MCTS on PPO checkpoints | **not recommended** | — | — | — |

### 1. Play-time search depth on V2.1c (re-scope of PILA-6)

- **Hypothesis.** The cheapest strength on the table is search at play time. The same net at depth
  2 beat itself at depth 1 by 57.1% [53.7, 60.5] over 800 games (round-23 net, `docs/V2_PLAN.md` M9).
  That has never been measured on V2.1c or at depth 3. Expected effects:
  - V2.1c at depth 2 vs depth 1: +5 to +8pp. V2.1c's depth-1/depth-2 inconsistency is smaller than
    round 23's (§10.7 item 4), so the gain may sit at the low end.
  - Depth 3 vs depth 2: +1 to +4pp.
  - The puzzle set is too small to settle this: V2.1c solves 4/4/7/6/6 of 10 puzzles at depths 1–5
    (`runs/puzzles_2026-11-03/report.json`).
- **Second question.** Does checkpoint choice still matter at depth 2? V2.1c leads V2.0c by 2.5pp
  at depth 1. If the lead disappears at depth 2, net improvements of 1–2pp matter less than search
  for the deployed agent.
- **Code.** None in `parchis/az`. Add a `depth_a=/depth_b=` option to `scripts/v2/h2h.py`; today
  `DEPTH` is fixed at 1 and applies to both sides. Also add a ms/move script over a fixed set of
  positions, run at depths 1–4.
- **Not in scope.** `parchis/search/` (PUCT-MCTS) only loads MaskablePPO checkpoints, and those are
  not on this machine. The AZ line's search is the exact expectimax in `parchis/az/search.py`, so
  "MCTS on top of the champion" becomes "deeper expectimax on top of the champion".
- **Compute.** About 1.5 h, one heavy job.

  | Step | Pairs | Estimated time |
  |---|---|---|
  | V2.1c depth 2 vs V2.1c depth 1 | 10,000 | ~11 min |
  | V2.1c depth 2 vs V2.0c depth 2 | 10,000 | ~20 min |
  | V2.1c depth 3 vs V2.1c depth 2 | 2,000 | ~45 min |
  | ms/move curve, depths 1–4, 500 positions | — | ~10 min |

- **Evaluation.** Interim protocol, with seeds from the Evaluator. Report Wilson CIs. Report the
  depth-3 result as a 2,000-pair estimate (±1.5pp), not as a promotion.
- **Kill criterion.** If depth 2 vs depth 1 on V2.1c reads below 52% at 10,000 pairs, skip depth 3
  and drop item 6.
- **Sources.**
  - `docs/V2_PLAN.md` M9 and C7; `docs/LEADERBOARD.md` caveat 5.
  - Hauk, Buro, Schaeffer 2004, *-Minimax in backgammon.
  - Superhuman Stratego with test-time search (arXiv 2511.07312): search at play time adds
    strength on top of a self-play net.

### 2. Weight soups of existing checkpoints

- **Hypothesis.** Checkpoints of one recipe scatter about ±1.5pp around the recipe's level
  (§10.11 reading 2). Averaging their weights should land nearer the centre of the basin and lower
  the variance of the value head. Three of the candidate ingredients share a lineage: win3 and win10
  are 20-round continuations from V2.1c. V2.1d is a sibling, trained from the same V2.0b
  start. AZNet uses LayerNorm, not BatchNorm, so an average needs no statistics recomputed.
- **Expected effect.** About 0 to +1.5pp vs V2.1c, with probability ~0.3 of a CI-confirmed gain.
- **Code.**
  - `scripts/v2/soup.py <out.pt> <a.pt> <b.pt> ... [weights=...]`: a uniform average of state
    dicts, with a check that architectures match.
  - The soups:
    - S1 = avg(V2.1c, win3, win10)
    - S2 = avg(win3, win10)
    - S3 = avg(V2.1c, V2.1d)
    - S4 = avg(V2.1c, V2.1d, win3, win10)
  - A free health check: each soup's value log-loss on 2 freshly generated V2.1c shards, against
    its ingredients. A soup much worse than its ingredients sits on a loss barrier.
- **Compute.** About 20 min of heavy compute plus ~1 h of code.
  - Screening: each soup vs V2.1c at 2,000 pairs, ~40 s each.
  - Confirmation: the best soup vs V2.1c at 10,000 pairs on fresh seeds, ~3 min.
  - Sanity: the best soup vs the tuned heuristic at 1,000 pairs.
- **Kill criterion.** If every soup screens below 50% against V2.1c, stop and record it.
- **Prior evidence.** `docs/V2_PLAN.md` §10.11 names weight averaging as an open, untried direction.
- **Sources.**
  - Izmailov et al. 2018, SWA, [arXiv:1803.05407](https://arxiv.org/abs/1803.05407).
  - Wortsman et al. 2022, model soups, [arXiv:2203.05482](https://arxiv.org/abs/2203.05482).

### 3. Continuation that keeps every candidate, then SWA and selection (re-scope of PILA-5)

- **Hypothesis.** §10.11 showed that the *endpoint* of a 20-round V2.1c continuation loses to V2.1c
  (48.2–48.4%). The cause is checkpoint-to-checkpoint scatter plus selection, not a declining
  recipe. Only the endpoint was kept: the per-round candidates of every V2 run have since been
  deleted, and `~/parchis_training_archive` holds only `selfplay_2p_v1`. Keeping all 20 candidates
  makes two things possible:
  - an SWA average of the last 5 and the last 10;
  - picking the best of the last 10 by match, then confirming it on fresh seeds.
- **Expected effect.** About +0.5 to +1.5pp vs V2.1c. The SWA average is the likelier winner.
- **What is different from §10.11.**
  - The deliverable is an averaged or selected net, not the endpoint.
  - New seed base. The run doubles as the missing seed replicate of the continuation.
- **Code.** A `run_harvest.py` based on `run_window.py`:
  - Start from V2.1c with the V2.1c recipe and replay window 3, for 20 rounds.
  - Copy each round's `candidate.pt` into `runs/<run_id>/candidates/` (~0.5 MB each).
  - Reuse `soup.py` from item 2.
- **Compute.** About 2.5 h, one heavy job.

  | Step | Pairs | Estimated time |
  |---|---|---|
  | Run (win3 took 97 min) | — | ~100 min |
  | Screen 12 nets vs V2.1c: rounds 11–20, SWA-5, SWA-10 | 2,000 each | ~10 min |
  | Top 3 vs V2.1c | 10,000 each | ~10 min |
  | Confirm the winner on fresh seeds | 10,000 | ~3 min |

- **Evaluation.** The matched control is `runs/window_test/win3/model.pt`, the same recipe's
  endpoint. Promotion requires beating V2.1c under the interim protocol, after the
  fresh-seed re-confirmation.
- **Kill criterion.** If SWA-10 screens below 49% against V2.1c, skip the 10,000-pair stage.
- **Sources.** Same as item 2.

### 4. TD(λ) value targets for cheap rows

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
- **Code.**
  - `selfplay.py`: record each decision's depth-1 root value vector. It is already computed during
    search; cheap rows discard it today.
  - `targets.py`: compute a vector-valued λ-return backward over each game, for cheap rows only.
    Deep rows keep the current blend.
  - Config: `cheap_value_target="td_lambda"`, `td_lambda=0.7`.
  - Tests: λ = 0 is byte-identical to today; λ = 1 reproduces the plain bootstrap chain.
- **Compute.** About 1.8 h.
  - Run: 20 rounds from V2.1c, otherwise the exact win3 recipe, ~100 min.
  - Matches: vs win3 (the matched control) and vs V2.1c, 10,000 pairs each, ~6 min.
- **Kill criterion.** If the ladder at round 10 reads below 47% against win3 at 2,000 pairs, stop.
- **Sources.**
  - Tesauro 1995, TD-Gammon.
  - Sutton & Barto, ch. 12, λ-returns.
  - Veness, Silver, Uther, Blair 2009, *Bootstrapping from Game Tree Search*
    ([NeurIPS](https://papers.nips.cc/paper/3722-bootstrapping-from-game-tree-search)): training
    toward search values beats training toward outcomes.

### 5. 4-player bootstrap and evaluation harness (V2.3, stage 1)

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
- **Needs the owner.** Is 4 players a goal now? If yes, this moves to #2.

### 6. Pruned deeper search for depth 3–4

- **Idea.** Add Star1/Star2 pruning on the mover's channel (values bounded in [0, 1]) and order
  moves by the policy head. The aim is depth 3 at near depth-2 cost, and depth 4 interactively.
- **Value.** Only worth building if item 1 shows depth 3 beats depth 2.
- **Correctness test.** Same chosen move as full-width search on 1,000 positions.
- **Compute.** ~1 day of code, ~1 h of evaluation.
- **Source.** Hauk, Buro, Schaeffer 2004.

### 7. Per-head early stopping / lower value weight after epoch 1

- **Idea.** Today each round's training stops early on the *total* validation loss. That lets the
  value head degrade while the policy head keeps training: for 2×256 the chosen epoch's value loss
  is 0.5936 against a minimum of 0.5923 (§10.10 reading 3).
- **Change.** Keep the best value head and the best policy head separately, or drop
  `value_loss_weight` to 0.5 after the first epoch.
- **Expected effect.** Small, about 0 to +1pp.
- **Plan.** Same as item 4: a run from V2.1c with win3 as the control.

### 8. Exploiter probe, then a league pool

- **Probe.** Train a copy of V2.1c for 10 rounds against frozen V2.1c only. Self-play from V2.1c
  is excluded, and the pool holds V2.1c alone.
- **Reading the result.**
  - If the exploiter beats V2.1c by more than 3pp, V2.1c has a hole the self-play pool never
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
- **When to run.** Only as the follow-up arm if item 4 is flat.

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

### 12. Population of recipe replicates, keep the best

- **Idea.** Three seed replicates of the V2.1c recipe, keeping the best.
- **Expected effect.** Harvests the ±1.5pp scatter by selection: an expected max of 3 draws is
  about +1.3pp over the mean. V2.1c is already a high draw, so the gain over it is smaller.
- **Ranking.** Five hours for mostly selection; items 2–3 harvest the same scatter more cheaply.

### 13. Not recommended: revive PPO, or PUCT-MCTS on PPO checkpoints

- **PPO checkpoints.** They are not on this machine.
- **PUCT-MCTS.** The PPO line's MCTS gave 56.2% [51.4, 61.0] over 400 games over plain PPO
  inference (`docs/SEARCH_MCTS.md`). Its iterative loop regressed to ~40%.
- **Why not.** The AZ line already has exact expectimax with a better net. Item 1 covers search at
  play time.

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
| Depth-3 targets | Dropped by the owner, never run | — | §10.9 note |
| Replay window 10 vs 3 | 0: 49.2% | Stale games cancel the extra outcomes | §10.11 |
| Continue V2.1c 20 rounds (endpoint) | −: 48.2% / 48.4% vs V2.1c | Checkpoint scatter + selection; only the endpoint was kept | §10.11 |
| Depth 2 vs 1 at play time, same net | +: 57.1% [53.7, 60.5], 800 games (round 23) | Not yet measured on V2.1c → item 1 | M9 |

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
- Wortsman et al. 2022, model soups, [arXiv:2203.05482](https://arxiv.org/abs/2203.05482).
- Jaderberg et al. 2017, PBT, [arXiv:1711.09846](https://arxiv.org/abs/1711.09846).
- Vinyals et al. 2019, AlphaStar (league training).
- Hauk, Buro, Schaeffer 2004, *-Minimax in backgammon.
- Sturtevant 2002; Schadd & Winands 2011 (multiplayer search).
- Superhuman Stratego with self-play RL and test-time search,
  [arXiv:2511.07312](https://arxiv.org/abs/2511.07312).
- Ludo-specific work: Alvi & Ahmed 2011; Alhajry, Alvi & Ahmed 2012 (TD(λ)/Q-learning against
  heuristics). Nothing peer-reviewed beyond that, and nothing on Parchís.
