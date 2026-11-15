# Evaluation protocol: when a new model becomes champion

Version 1 (PILA-2, 2026-11-15). This file sets the rule for every "A beats B" claim and every
champion change in the "strongest Parchís agent" project. It is owned by the Evaluator. Changes to
the promotion rule (§4) need the owner's approval on the issue that proposes them (§8).

**One command runs it:**

```bash
python scripts/eval/protocol.py <candidate.pt> --run-id <run_id> --issue PILA-N [--selection fixed|best-of-N]
```

It prints the match table and a verdict (PASS / FAIL / PROVISIONAL / INVALID). It also writes
`runs/<run_id>/eval/protocol_b<block>.json`. Exit code: 0 PASS, 1 FAIL or INVALID, 2 PROVISIONAL,
3 compute lock held, 4 bad input. `--dry-run` checks the inputs and prints the plan. It plays no
games and uses up no seeds.

## 1. Scope

- **Format:** 2 players. The 4-player rule is provisional (§7), because no 4-player net exists yet.
- **Agent:** a checkpoint is played as the net plus expectimax search
  (`parchis.az.agent.make_search_agent_factory`) at the **champion's depth**, which is currently 1.
  Every checkpoint so far was gated and ranked this way. A stronger agent built by deeper search on
  the same net is a different time budget. It is not a candidate under v1. The owner decides whether
  to open a separate depth class.
- **Candidates:** any checkpoint that loads with `champion_pool.load_numpy_net(path, 216, 2, …)`.
  The trunk width is read from the weights, so 2×256 and 3×512 nets both work.

## 2. Match format

- **Duplicate pairs** (`parchis.evaluation.duplicate.play_duplicate_match`). Each deal (dice seed)
  is played twice, with the seats swapped. A pair is 2 games.
- 500-turn limit. A game that hits the limit is a draw and scores ½ for each side. If more than 0.1%
  of the run's games are draws, the verdict is **INVALID** and the run is repeated on a new block.
  So far no run has had a draw.
- 8 spawn workers with one-thread BLAS (`scripts/v2/ladder.single_thread_blas`). Every match is
  split into 8 chunks, and chunk *c* uses one seed (§5).
- **Win rate and CI:** score / games, with a Wilson 95% interval that treats each game as
  independent. Seat-swapped pairs reduce variance only about 1.0–1.1× here (V2_PLAN §2.6), so the
  interval is accurate to slightly conservative. C8 luck-adjusted scoring is not used for verdicts
  (V2_PLAN §10.8).

## 3. Opponent set and games per comparison

Every protocol run plays these five matches. The roles come from `runs/champions.json` (§6).

| # | Match | Duplicate pairs | Games | Role |
|---|---|---|---|---|
| 1 | candidate vs **champion** | 10,000 | 20,000 | **decisive** |
| 2 | candidate vs tuned heuristic | 1,000 | 2,000 | guard (paired with 3) |
| 3 | champion vs tuned heuristic, **same deals as 2** | 1,000 | 2,000 | guard reference |
| 4 | candidate vs random | 1,000 | 2,000 | sanity guard |
| 5 | candidate vs previous champion | 1,000 | 2,000 | context only, not gating |

Why these sizes:

- **Match 1:** 20,000 games give a CI half-width of ±0.69pp. A candidate needs an observed rate of
  about 50.7% to pass. Its chance of passing is 80% if its true edge is +1.0pp (≈7 Elo), and about
  50% at +0.7pp. Real gains in this lineage have been +1 to +2.5pp (V2_PLAN §10.7, §10.11). The
  2,000-game ladders could not detect them.
- **Matches 2–3:** these catch a candidate that exploits the champion but is weaker in general,
  since strength here is not one-dimensional (LEADERBOARD caveat 4). Matching the champion's own
  score on the same deals removes the drift from comparing against an old reference number.

## 4. Promotion rule

A candidate replaces the champion only if all of the following hold:

1. **Decisive:** in match 1, the Wilson 95% **lower bound** of the candidate's score against the
   champion is **strictly above 50%**. A tie keeps the incumbent.
2. **Heuristic guard:** the candidate's rate against the tuned heuristic is no more than **3pp**
   below the champion's rate on the same deals (match 2 − match 3 ≥ −3pp). For two equal agents
   this check fails by chance about 2% of the time.
3. **Random guard:** the candidate scores ≥ 95% against random. The champion scores 97–98%.
4. **No selection effect:**
   - If the checkpoint was **fixed before any evaluation** (for example, the endpoint the
     experiment's plan names), one PASS is enough.
   - If it was **picked from N > 1 checkpoints or runs** by any comparison, on any seeds, use
     `--selection best-of-N`. A passing run then gives **PROVISIONAL**. Run the protocol again on
     a new block with `--confirmation-of <first protocol json>`. Only that confirmation run can
     give PASS. The verdict states how the checkpoint was chosen.
5. **No retries:** a FAIL is final for that checkpoint. Re-running the same checkpoint on new seeds
   until it passes is not allowed. The exception is a run voided as INVALID, or a protocol fault
   that the Evaluator records in the seed registry.
6. **Complete run record:** before the verdict, the run must have its config, git SHA, every seed,
   and the checkpoint path and sha256 recorded (parchis-lab-rules "Run records"). If anything is
   missing, the Evaluator sends the run back to the Engineer before playing any games.

On PASS, the Evaluator opens a PR that updates `runs/champions.json` and `docs/LEADERBOARD.md`,
and the owner merges it. The previous champion becomes `previous`. On FAIL, the result is recorded
in the run's directory and on the issue.

## 5. Seeds: registry and allocation

- Each protocol run takes the next **block** *k* of 1,000 seeds: `5_000_000 + 1000·k … +999`,
  k < 5,000. Match *m* chunk *c* uses `block base + offset_m + c`, with offsets decisive 0,
  heuristic 100 (shared with the champion's heuristic match), random 200, previous champion 300.
- **Blocks are never reused, by any pair.** A block counts as spent once it is allocated, even if
  the run aborts. So no verdict can reuse deals from an earlier evaluation or from training.
- The registry is append-only JSONL. The authoritative copy is `~/parchis_eval_seeds.jsonl` on the
  Mac, and every allocation is mirrored to `runs/eval_seed_registry.jsonl` in the repo. The next
  block is computed from the highest block in either file. That file also lists the legacy ranges:

| Range | Used by | Status |
|---|---|---|
| `9000+i`, i<8 | V2 ladders (`scripts/v2/ladder.py`) | legacy; reused on purpose for comparable ladders; closed for verdicts |
| `70000+i`, i<8 | every 10k-pair `scripts/v2/h2h.py` match | legacy; every pair used the **same** 8 seeds; closed for verdicts |
| `120000+1000k+c` | PILA-1 round robin | legacy; reserved for leaderboard re-runs |
| `0 + round·1000 + shard`, `0 + round` | training (generation/promotion) of every run to date | legacy; overlaps the three ranges above after round 9 |
| `5_000_000 + 1000k`, k<5000 | **this protocol** | one block per run |
| `10_000_000·n + …`, n≥1 | new training runs (generation, train, promotion seeds) | n recorded in the run record |

- Training must never use the 5M–10M band. Engineers who need an evaluation-like screening match
  during training (for example, picking among late checkpoints) take seeds from their run's own
  10M·n band. Seeds from the 5M–10M band would make a later verdict reuse those deals.

## 6. Champion pointer

`runs/champions.json` names, for each format, the `current` champion (path, sha256, depth, date,
evidence), the `previous` one, and the history. The protocol reads it and refuses to run if the
champion file's sha256 does not match. Only a PASS PR merged by the owner changes it. Current
state: 2p champion V2.1c (`runs/v2_1c_champion/model.pt`, depth 1), previous V2.0b, and no 4p
champion.

## 7. Four players (provisional: needs owner approval once a 4p net exists)

There is no 4p net yet, so the command only handles 2p. The proposed rule:

- **Format:** the candidate rotates through all 4 seats on each deal
  (`play_duplicate_group(num_players=4)`), with three copies of the opponent in the other seats.
  The null rate is 25%.
- **Decisive:** 5,000 groups (20,000 games) against 3× the champion. The Wilson 95% lower bound
  must be above 25%.
- **First 4p champion:** with no champion yet, the first 4p net must clear the lower bound against
  3× tuned heuristic (decisive, 5,000 groups) and score ≥ 50% against 3× random.
- The guards and the selection rule are the same as in 2p. The cost will be measured when the
  first 4p net exists. Games are longer, and the net is wider (input and value head sized by
  `num_players`).

## 8. Cost on the owner's Mac

Measured on 2026-11-15 on the Mac mini (M4, 10 cores), 8 workers, depth 1. One full protocol run
(28,000 games: matches 1–5) took **219 s wall-clock** (1,727 s CPU). It is a heavy job (8 workers),
so the command takes `~/parchis_compute.lock` itself and exits with code 3 if the lock is held.
A best-of-N confirmation is a second run, ≈ 4 min more. The planned 4p version is probably
3–4× longer per game. It will be measured before it is adopted.

Calibration run (the first real run of this command, block 1): V2.1d, fixed endpoint, against
champion V2.1c.

| Match | Win rate | Wilson 95% CI | Games | Seeds | Depth |
|---|---|---|---|---|---|
| V2.1d vs V2.1c (decisive) | 48.93% | [48.24, 49.62] | 20,000 | 5001000+c, c<8 | 1 |
| V2.1d vs tuned heuristic | 70.80% | [68.77, 72.75] | 2,000 | 5001100+c | 1 |
| V2.1c vs tuned heuristic (same deals) | 73.05% | [71.06, 74.95] | 2,000 | 5001100+c | 1 |
| V2.1d vs random | 97.55% | [96.78, 98.14] | 2,000 | 5001200+c | 1 |
| V2.1d vs V2.0b | 53.05% | [50.86, 55.23] | 2,000 | 5001300+c | 1 |

Verdict: FAIL (decisive lower bound 48.24%). It reproduces V2.1c's earlier 50.74% [50.05, 51.43]
against V2.1d on fresh deals. Record: `runs/v2_1d_champion/eval/protocol_b1.json`. Block 0 was
spent by the same run, which crashed while writing its record before the fix. Its result was
never read and is not reported.

## 9. Changing this protocol

The promotion rule (§4), match sizes (§3) and seed scheme (§5) change only with the owner's
approval on an issue, and then a PR that bumps the version at the top of this file. Constants
live in `parchis/evaluation/protocol.py`, and the rule is tested in
`parchis/tests/test_protocol.py`. Results under different protocol versions are labelled with
their version and are not mixed in one comparison.

## 10. Verdict comment template

```
## Verdict: PASS | FAIL | PROVISIONAL | INVALID
Candidate: <run_id> (`<path>`, sha256 <12 chars>, code SHA <git sha>)
Champion: <name> (`<path>`), depth <d>, protocol v1, block <k>
| Match | Win rate | Wilson 95% CI | Games | Seeds | Depth |
...
Checks: decisive LB <x>% (> 50%); heuristic Δ <y>pp (≥ −3pp); random <z>% (≥ 95%)
Selection: fixed | best-of-N (<how chosen>; confirmation <block or pending>)
Record: runs/<run_id>/eval/protocol_b<k>.json   Leaderboard PR: <link or n/a>
```
