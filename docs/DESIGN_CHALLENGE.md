# Design challenge: state view, board view, targets and progress metrics

PILA-10, Researcher, 2026-11-16. Desk research plus one sub-minute diagnostic
(`scripts/research/pila10_markov_diag.py`, §1.1; its results are pending the compute lock). No
training, and no change to candidate code or to the promotion rule.

Every experiment so far tuned the recipe (targets, soups, replay window, search depth). This
document steps back and asks whether the fixed parts of the design fit a high-variance dice game:
- what the net sees (§1);
- how the board geometry reaches it (§2);
- what it is trained to predict (§3);
- how progress is measured between promotions (§4);
- which knobs should matter (§5).

Each area ends with a verdict: **keep**, **change** or **test**. Every "change" or "test" is a
backlog issue (§6).

Conventions: "pp" is head-to-head win rate in duplicate pairs at depth 1 unless stated.
Expected effects are **hypotheses** unless a recorded run is cited with games, seeds and CI.

## Summary

| # | Area | Finding | Verdict | Issue |
|---|---|---|---|---|
| 1a | State | **Rules mismatch in search:** after a capture or finish made with a 6, the search ends the turn after the bonus move. The rules give the mover a reroll (`Game.play_turn`). Deep targets and play at depth ≥ 2 undervalue capturing or finishing with a 6. | **change** (fix search, measure) | PILA-11 |
| 1b | State | The encoding is not Markov in three places: (1) a bonus decision cannot see that a reroll is owed; (2) after the second 6 it cannot see which piece the three-sixes penalty would hit; (3) at 4 players, "the most recently moved piece" on a start square is not tracked. | **change** (flags), via the probe | PILA-12 |
| 1c | State | The threat feature describes the side that is **not** about to roll. The side to move has no "shots" feature: which of its rolls capture which opponent piece. This is the most likely cause of the C8 finding that the value head is under-resolved with respect to the roll. | **test** | PILA-12 |
| 1d | State | `capture_threat_score` is an exact fact about the position, not a heuristic opinion. It does not leak the heuristic. | keep | — |
| 1e | State | `turn_number / 300` and the always-zero home slot 76 | keep (harmless); drop at the next encoding change | — |
| 2 | Board | A flat MLP over one-hot slots has no weight sharing along the track and no piece-permutation symmetry. Conv or token trunks are plausible, but every extra µs per leaf is paid ~1,000× at depth 3. | **test offline first** (fixed data, fixed budget, strength per ms) | PILA-12 |
| 2b | Board | Pieces are interchangeable under the rules, but the policy head is indexed by piece_id. Permuting piece ids gives up to 24 free relabelings of every row. | **test** (augmentation arm in the probe) | PILA-12 |
| 3a | Targets | In a 2-player win/loss game, P(win) is the whole outcome distribution. Distributional or equity value adds nothing. | keep | — |
| 3b | Targets | 75% of rows regress on raw outcomes. A luck-corrected return (C8 control variate, unbiased) or TD(λ) (biased, lower variance) should train a better value head, not only evaluate more precisely. | **test** (replaces backlog #5) | PILA-14 |
| 3c | Targets | The aux head predicts only the mover's own piece fates. The opponent's fates and game length are free targets. | **test** (aux arm in the probe) | PILA-12 |
| 4 | KPIs | Calibration and value log-loss are blind to a +3pp gain (PILA-8). Ten puzzles cannot rank nets. The headroom above the champion is at least ~130 Elo (search alone). | **test** five candidate KPIs against the PILA-1 ratings | PILA-13 |
| 5 | Params | Games per round, replay window, deep fraction, capacity, Dirichlet and τ are measured as flat or settled. Per-round training noise (learning rate, epochs) is the knob most likely to matter now. | keep; noise knobs folded into backlog #4 and #7 | — |

## 0. Two facts that frame everything below

**The game has a lot of skill headroom.** This contradicts the intuition that "dice decide".
- S1 beats random 97.10% [96.27, 97.75] (2,000 games, seeds 5004200+c).
- The PILA-1 board spans ~590 Elo from random to the champion.
- The same net gains +10.4pp at depth 2 over depth 1, and a further +8.5pp at depth 3 (PILA-6:
  60.39% [59.72, 61.07], 20,000 games; 58.48% [56.94, 59.99], 4,000 games). Chained, that is
  about +132 Elo, or roughly **68% for depth 3 against depth 1**. This is an estimate; nobody has
  played d3 vs d1 directly (PILA-13 does).
- A perfect player would do better still. So the champion at depth 1 sits at least ~18pp below
  the best achievable player.

Single training steps move 1–3pp (V2.1c +2.5pp, S1 +2.9pp). The game is not near its ceiling. The
per-step effects are small because each training step is noise-dominated (next point), not
because there is nothing left to learn.

**Per-round training is noise-dominated.**
- The 20-round continuations of V2.1c lose to their start (48.2%, 48.4%, §10.11).
- Their uniform average with the start beats the start by +3.1pp (PILA-8).
- The value head reaches its best held-out loss at epoch 0–2 (§10.10).

Each round's net is a noisy sample around a better mean. Most proposals below either reduce the
noise in the targets (§3) or give the net a representation that needs fewer samples to learn the
same thing (§1–2).

## 1. State view: what the net sees

**Current choice** (`parchis/az/encoding.py`, 216 floats at 2 players):
- own pieces, 4 × 10 hand-made features;
- shared-track occupancy, N × 68 (0.5 per piece);
- home columns, N × 8;
- per-seat scalars, N × 6;
- turn context, 12: roll one-hot with a 6/7 split, bonus type, six streak, whose turn, and turn
  number / 300.

The encoding is relative to the observer's path, so it is colour-invariant. That part is sound and
stays (V2_PLAN §2.7).

### 1.1 Is the state Markov for `docs/RULES.md`?

I listed every rule that depends on history rather than on the board, and checked how
`encoding.py`, `turn_context.py` and `search.py` carry it.

| Rule-relevant fact | Carried by the board? | Encoded? | Search models it? |
|---|---|---|---|
| Roll value, six becomes 7 when no piece is in base | — | yes (slots 0–6) | yes |
| Pending bonus (10 or 20) | — | yes (slots 7–8) | yes |
| Six streak (penalty risk on the next 6) | — | yes at roll decisions; **0 at bonus decisions** (`TurnContextTracker.context_for`) | yes |
| **Reroll owed after a bonus that came from a 6** | — | **no** | **no: the turn passes after the bonus** (below) |
| **Which piece the three-sixes penalty would hit** (the piece moved on the second 6; not if it entered the home column) | — | **no**: depth-1 leaves and training rows at streak 2 cannot see it | yes, at chance nodes (depth ≥ 2) |
| Mandatory entry on a 5; forced blockade opening on a 6 | yes | yes (legal moves; `blockade_forced_on_6`) | yes |
| 4 players: which of two opponents on my start square moved there last (entry capture target) | **no** | no | uses the engine, so yes |

**Finding 1a: a rules mismatch in search (not only in the encoding).**
- `Game.play_turn` sets `keep_rolling = (dice_roll == 6)` after the bonus chain. So a 6 that
  captures or finishes still earns the reroll (RULES.md: "the player still gets their bonus turn").
- `search.py` decides six-again with `pending_bonus is None and roll == BONUS_TURN_ROLL`, in both
  `afterstate_context` and `_resolve_turn_continuation`.
- The bonus decision is entered with `roll=None` (`_expand_decision`, capture and finish branches).

So, inside the search, the move after a bonus always hands the turn to the opponent. Three
consequences:
1. At depth ≥ 2, every capture or finish made with a 6 is valued as if it forfeits the reroll. A
   plain 6 move keeps it. The search is biased **against capturing with a 6**, by the value of a
   free roll.
2. The depth-2 deep targets (25% of decisions since V2.1c) inherit the same bias.
3. In real play, `TurnContextTracker` hands bonus decisions `(roll=None, consecutive_sixes=0)`. The
   root bonus search also passes the turn after the bonus, and the encoding of the bonus decision
   cannot tell "reroll owed" from "turn ends".

The fix is small: carry the original roll through the bonus chain, and report it in the tracker.
Because it changes search results, it should be a flagged change measured as a study (PILA-11).

**Finding 1b: two encoding gaps that only depth-1 leaves and training rows see.**
- After a move on the second 6 (streak 2), a 1-in-6 chance sends **that specific piece** back to
  base. The encoding has the streak but not the piece. Two leaves that differ only in which piece
  moved on the second 6 encode identically and have different values. Search at depth ≥ 2 resolves
  this exactly at the chance node; depth-1 leaves and the value head's training rows do not.
- The 4-player "most recently moved" rule is irrelevant at 2 players, since both pieces on the
  square are the single opponent's. Note it for V2.3.

Proposed encoding flags, zero-initialised so that a warm start is exact:
- `reroll_owed` in the turn context;
- `streak_at_bonus` (the real six streak at a bonus decision);
- `at_risk_on_next_six`, one per own piece.

**Diagnostic: how often does it bite?** `scripts/research/pila10_markov_diag.py` plays S1 vs S1 at
depth 1 and counts the affected decisions. At every 6-roll decision with a capturing or finishing
move, it compares depth-2 search as shipped against a patched copy that carries the reroll. The
patch lives only in the script.

<!-- DIAG_RESULTS -->
**Pending.** The 200-game run waits for the compute lock: PILA-5 held it when this was written,
and the issue allows measurements only while the lock is free. The run takes about 20 s on one
process. Its results will be added here on this PR. Until then, the frequency and size of the
effect are unmeasured. The mismatch itself is established from the code, as cited above.

### 1.2 Is anything missing that decides games?

**Finding 1c: the side to move has no "shots" feature.**
- `capture_threat_score` counts, for each of the **observer's** pieces, how many opponent faces
  hit it, including bonus chains.
- Search queries the value head almost only on pre-roll leaves. The observer there is the player
  **about to roll** (`afterstate_context` calls `next_player()`). So the feature describes threats
  from the side that is *not* rolling next. The threats that the coming roll actually decides (the
  mover's own shots at opponent pieces) are encoded only implicitly, as pairs of one-hot slots
  across two 68-wide channels.
- An MLP has to learn each (square, distance) pair separately: about 68 × 6 conjunctions per
  relation, plus the bonus-chain cases.

C8 measured the consequence from the other side (§10.8 reading 1):
- the net's value moves by only about half of what a roll is worth;
- the variance multiplier is ×1.3 with the net and ×2.3 with depth-1 search, which "sees what the
  roll lets the mover *do*".

Hypothesis: symmetric per-piece features close part of that gap. These are, for each opponent
piece, the mover's hit count over the 6 faces plus bonuses, and a per-face summary for the mover
(can capture / can finish / can enter / is forced). If the hypothesis holds, the C8 multiplier rises
toward the depth-1 search value's ×2.3, without search.

**Opponent piece-level features.** Only own pieces get the 10-feature block. Opponent pieces appear
only in the track one-hots and per-seat scalars. At 2 players the observer alternates, so each side
gets the rich block on its own turn. But a leaf's value must judge both sides at once. Adding a
mirrored opponent block (same 10 features, from the opponent's side) costs 40 inputs at 2 players.

**Blockades ahead, distance-to-danger.** Blockades are partly covered:
- `own_blockades_on_track` per seat;
- `blockade_forced_on_6` per own piece;
- the threat count, which respects blockades because it uses legal moves.

"Steps until the next blockade in front of this piece" is missing, for own and opponent blockades.
It is cheap to add, and it decides races near home. It is included in the probe's feature arm.

**Finding 1d: is `capture_threat_score` a leak toward the old heuristic?** No:
- It is an exact count derived from the rules (legal moves over the 6 faces and the bonuses). The
  heuristic's opinion lives in its weights, not in this count.
- Backgammon is the precedent. TD-Gammon 0.0, with a raw encoding only, reached intermediate
  strength. Versions 1.0 and later added expert features such as blot exposure and blockade
  strength, and approached world-class play (Tesauro 1995; Sutton & Barto 2018, §16.1).
- GNU Backgammon's nets also combine raw point encodings with hand-made features.

Exact tactical counts help a small net. Keep it, and make it symmetric (1c).

**Finding 1e: minor items.**
- `turn_number / 300` is only rule-relevant through the 500-turn draw cap. No protocol game has hit
  that cap, so the feature mostly lets the net condition on game length. It is harmless at worst.
- Home-column slot 7 (square 76) is always 0, because finished pieces are excluded.

Neither is worth an experiment on its own. Drop both at the next encoding change.

**Verdict (state view):**
- **change** 1a (search fix, PILA-11);
- **change** the 1b flags, and **test** 1c shots, the opponent block and the blockade-distance
  features, all in the offline representation probe (PILA-12) before any self-play run;
- **keep** 1d and 1e.

## 2. Board view: how geometry is represented

**Current choice:**
- A flat 2 × 256 MLP with LayerNorm (`parchis/az/net.py`, ~125k parameters) over the 216-float
  vector.
- Policy: 4 logits indexed by piece_id.
- Inference uses NumPy, at 0.27 ms per depth-1 decision (PILA-6).

**What the structure of the board suggests:**
- **Translation along the track.** Captures, shots, blockades and safe squares are all relations
  at fixed distances (1–6, 7, 10, 20) along a circular track. A 1-D circular convolution (kernel
  ≥ 7, dilations for 10 and 20) over the per-seat track channels shares those weights across all 68
  squares. The MLP must learn each square separately.
  - The relative geometry is not fully translation-invariant: safe squares and starts sit at fixed
    offsets. So add a fixed "square type" channel (safe, own start, opponent start,
    home-entry).
- **Piece permutation.** Pieces of one colour are interchangeable under the rules. The policy
  ("which piece to move") is permutation-equivariant, and the value is permutation-invariant.
  - A set or token trunk gets this for free: per-piece tokens with relative distances, and a shared
    per-piece policy logit (Deep Sets, Zaheer et al. 2017; Set Transformer, Lee et al. 2019).
  - The MLP has to learn it four times over, from data in which piece ids are not exchangeable
    (piece 0 always starts on the board).
  - The cheapest test is data augmentation: permute own piece ids in the rows and targets. That
    gives up to 24 relabelings per row, with no architecture change (2b).
- **Literature:**
  - Backgammon nets that reached world-class play are flat MLPs over a unary point encoding plus
    expert features: TD-Gammon, GNUbg, and the Stochastic MuZero backgammon agent (Antonoglou et
    al. 2022) builds on the same representation family.
  - Those nets succeeded with features, not with architectural priors. GNUbg adds phase-specific
    nets (contact, crashed, race) and exact bear-off databases (backlog #11).
  - I found no peer-reviewed deep-RL work on Ludo or Parchís representations. What exists is
    tabular TD(λ) or Q-learning against heuristics (Alvi & Ahmed 2011; Alhajry et al. 2012) and
    practitioner write-ups using egocentric, per-token encodings. That matches what this project
    already does.
  - Evidence that convolutions or tokens beat MLP+features in race games is therefore absent.
    That is a reason to test cheaply, not to rebuild.

**The constraint that decides this area: search multiplies inference cost.**
- Depth 3 evaluates ~940 leaves per decision, and each ply is worth +8–10pp (PILA-6).
- A trunk that costs 3× more per leaf turns depth 3 into the time budget of roughly depth 2.5. It
  must be worth more than the lost fraction of a ply.
- The NumPy MLP is a few small matmuls. A transformer over 8–16 tokens, or a conv over 2 × 68, is
  5–20× more FLOPs and harder to batch in NumPy.

So the right metric is **strength per ms of search budget**, not strength per net. The probe
(PILA-12) reports inference µs per row for every arm.

**Verdict (board view): test offline first.** PILA-12 fits fresh nets on one frozen dataset
generated by S1, with the same data and seeds:
- (a) MLP baseline;
- (b) MLP with piece-permutation augmentation;
- (c) MLP + circular-conv track stem;
- (d) per-piece token trunk.

It compares held-out value and policy loss, decision error against a depth-3 reference (§4 K2), and
µs per row. A self-play run follows only for an arm that clears the go bar *and* stays within 2× the
MLP's inference cost.

## 3. Learning targets and what counts as signal

**Current choice:**
- Value = per-seat win distribution (softmax over seats).
- Cheap rows (75%) regress on the raw outcome.
- Deep rows (25%) use the depth-2 root value blended with the outcome at λ = 0.5.
- Policy = softmax of depth-*k* move values at τ = 0.015.
- Aux = the mover's own 4 piece fates (BCE, weight 0.2).

### 3.1 How much of the outcome is luck at a given point?

Two measured anchors and one new measurement:
- The value head's held-out log-loss is 0.584 for V2.1c on its own data, against ln 2 = 0.693 for
  a coin. Averaged over all positions, the net resolves ~16% of the outcome entropy (§10.10).
- C8 found the net captures roughly half of what each roll is worth (c ≈ 0.5, §10.8).
- The same diagnostic (§1.1) also reports, for S1 at depth 1, the share of outcome variance
  already resolved (1 − Brier / base Brier), by fraction of the game played. That measurement is
  pending the compute lock.

Expectation (a hypothesis until that table is in): early on, nearly everything left is dice, and
most of the outcome variance is resolved in the second half. Either way, every row is a noisy
label: 18,000 independent outcomes per training step sit behind 2.6M rows (§10.10).

### 3.2 Distributional or equity targets

At 2 players the outcome is binary. A Bernoulli distribution is fully described by its mean, so the
softmax over seats already is the full distribution and the full equity. Backgammon needs equity
targets because of gammons and the cube. Parchís has neither. **Keep.**

A distributional head over a *margin* (opponent pieces still out at game end, turns to finish)
would be a lower-variance auxiliary signal, like KataGo's score head. That belongs with the aux
targets (3.4).

### 3.3 Luck-corrected and bootstrapped value targets

Two targets reduce the noise in the 75% of rows that use raw outcomes. They differ in bias:
- **Luck-corrected return (new).** z̃_t = outcome − Σ_{rolls r after t} luck_r, with
  luck_r = V(S, r) − mean_f V(S, f), exactly as `parchis/evaluation/luck.py` computes it. Because
  the die is fair, E[luck_r | S] = 0 for any V. So z̃_t is an **unbiased** estimate of the same
  quantity the raw outcome estimates, with lower variance.
  - This is the control-variate idea for chance nodes (Veness, Lanctot & Bowling 2011), applied to
    training targets instead of evaluation.
  - With today's net it removes ~24% of the variance (×1.3). That is equivalent to ~1.3× more games
    per round, without the staleness that sank window 10 (§10.11).
  - The tooling exists. The cost is one 6-row forward per roll during self-play, about +30% per
    game (C8 measured +75% in eval games, which have no search).
- **TD(λ) return (backlog #5).** Bootstraps on the net's own later values. The variance is much
  lower, but the target is biased toward the current net.

The two can be compared on the same control. If the unbiased version gains, the noise was the
problem. If only TD(λ) gains, the bias is acceptable at this noise level. **Test** (PILA-14, which
replaces backlog #5).

Does a better target train a better net, or only evaluate more precisely? Evaluation precision does
not improve with the target. The claim is that the value head's *fit* improves, because it learns
from 2.6M rows with less noise behind each label. The kill criterion and the C8 multiplier check
exactly that.

### 3.4 Auxiliary targets

The aux head learns (val BCE ~0.47 against 0.69), but it covers only the mover's own pieces.
Zero-cost additions, all computed from games already generated:
- the opponent's 4 piece fates;
- turns until the game ends;
- the opponent's pieces still out at the end (a margin).

KataGo's experience is that auxiliary targets mainly buy sample efficiency, which is the binding
constraint here (§0). **Test** offline, as one arm of PILA-12.

**Verdict (targets):** keep the value form; **test** 3.3 (PILA-14) and 3.4 (PILA-12).

## 4. KPIs and progress marks for a random game

Protocol win rate stays the ranking metric. The aim here is cheap numbers to read between
promotions, to screen arms early, and to set kill criteria.

### 4.1 What we have, and how blind it is

| Metric | Evidence about its predictive value |
|---|---|
| Value log-loss and ECE on self-play shards | Moved ~0.001 for a +3pp gain and ranked S4 above S1 (PILA-8). Calibration was already 0.004–0.009 at V2.1c. **A health check, not a predictor.** |
| 2,000-game ladders | ±2.2pp. They move ±2pp between consecutive checkpoints (§10.9 reading 4). Too coarse for 1–3pp steps. |
| Puzzle accuracy, 10 puzzles | One puzzle = 10pp. The binomial SE at n = 10 is ±16pp. Accuracy is not monotonic in depth (30/50/60/40/60%, AZ_DESIGN). Puzzles 2 and 4 fail at every depth 1–5. That is either a value blind spot or a disputed label; nobody has checked with a deep reference. **Not a progress mark.** Useful as a qualitative tactic check (PILA-9). |
| C8 luck multiplier | Moves with sharpness by construction (×1.3 net vs ×2.3 depth-1 value). Never correlated with strength across checkpoints. |

### 4.2 Skill ceiling

- **Lower bound on headroom above S1 at depth 1:** search alone gives ~+132 Elo (≈ 68%) at depth
  3. That is chained, not measured; PILA-13 measures d3 vs d1 directly.
- **Upper bound on the luck share:** the champion beats random 97%. So even the weakest possible
  opponent keeps ~3% through dice alone.
- **Realistic effect sizes.**
  - Net-only steps have been 1–3pp per step (~7–21 Elo).
  - Search plies have been 8–10pp each.
  - 10,000 pairs resolve ±0.7pp. A 0.5pp effect needs ~40,000 pairs, so do not design experiments
    whose expected effect is below ~1pp unless they are free.

### 4.3 Proposed between-promotion KPIs

Each is cheap: under ~5 minutes per checkpoint on 8 workers, except K2's one-time reference.

| KPI | How it is computed | Why it should predict protocol win rate |
|---|---|---|
| **K1. Luck multiplier** (value sharpness) | `h2h.py … luck=default`, fixed 200 pairs vs the tuned heuristic, fixed seeds, S1 as the luck evaluator for every checkpoint | A value head that resolves more of each roll picks better among moves whose outcomes differ by the next roll. It is the target metric for 1c and 3.3. |
| **K2. Decision error vs a deep reference** (backgammon's error rate / PR) | Freeze ~2,000 decision positions (multi-move only) from S1 self-play on fixed seeds. For each, compute depth-3 move values once with a reference net (S1 now; later a soup of nets, to dilute any one net's blind spots). A candidate at depth 1 scores: mean loss = mean(V_ref(best) − V_ref(chosen)) in pp, and blunder rate = share of decisions with loss > 2pp. | In a two-player zero-sum game, the expected-score gap between two players is the accumulated value each gives up per decision, measured by a stronger judge. GNUbg's error rate predicts match results for this reason. It reads decision quality directly and is not diluted by dice. Risk: reference bias toward S1's own views, which is why the reference should be a soup. |
| **K3. Value log-loss and ECE on a frozen held-out set** | One fixed shard set (750 games, S1 generator, fixed seeds), never trained on | Barrier and health check (souping, new encodings). Not a ranker (PILA-8). |
| **K4. Search consistency** | Mean \|V(pre-roll) − mean_f V(post-roll f)\| and the depth-1 minus depth-2 root bias, on the K2 positions | Moved with C2 (1.56 → 1.12pp, §10.7). Measures how much a deeper search would still correct the net. |
| **K5. Puzzle accuracy**, re-founded | Positions from K2 where the reference's best-vs-second gap > 5pp ("high-stakes decisions"), auto-labelled, n ≥ 300 | Same idea as the hand-made suite, but with labels verified by search and a sample size that can rank. The 10 hand-made puzzles stay as a qualitative check. |

**Validation before adoption (PILA-13, Evaluator).** Compute K1–K5 for the 14 net checkpoints of
the PILA-1 round robin plus S1. Rank-correlate each against the Bradley-Terry Elo (range 365–592).
Adopt only those with Spearman ρ ≥ 0.8 overall *and* the right sign among the top 7, which are
within 20 Elo of each other. Adoption into the protocol or registry is the owner's call and the
Evaluator's to implement.

**Verdict (KPIs): test** K1–K5 against existing ratings; keep win rate as the ranking metric; stop
reading 10 puzzles as a progress mark.

## 5. Parameters

| Knob | Evidence | Verdict |
|---|---|---|
| Games per round (6,000) and replay window (3) | V2.0c: more depth-1 rounds flat. Window 10: 49.21% [48.52, 49.90], 20,000 games (§10.11). Staleness cancels extra outcomes. | keep |
| Deep fraction (0.25) | 0.5 flat: 49.26% [48.57, 49.95] vs V2.1c, 20,000 games (§10.9) | keep |
| Search depth of the targets | Depth 2 is the only training lever that moved (C2). Depth 3 is untested (backlog #3). | test (existing #3) |
| Net size (2×256) | 3×512 not better at any data size (§10.10) | keep for now; revisit only with a new encoding |
| τ (acting 0.03 → 0.005, target 0.015) and Dirichlet (ε = 0) | Units bug fixed in V2.0 (§2.1). Dice already explore (TD-Gammon used no injected noise). | keep |
| Dirichlet in chance-heavy trees | Not applicable: the search is exhaustive expectimax with no priors at the root. | keep off |
| Learning rate (3e-4), warm-start epochs (5), early stopping on total loss | PILA-8: averaging three nets gains +3.1pp, so the iterates bounce around a basin, which is the sign of too much step noise per round. Value overfits by epoch 0–2 (§10.10). | **most likely to matter now.** Already queued as backlog #4 (averaging inside the loop; add an EMA-of-weights arm) and #7 (per-head stopping). No new issue. |
| λ (0.5) on deep rows | Not implicated (§10.1) | keep; PILA-14 changes the cheap rows instead |

## 6. Resulting experiments (ranked into `docs/RESEARCH_BACKLOG.md`)

| Backlog rank | Issue | Idea | Expected effect (hypothesis) | Mac compute |
|---|---|---|---|---|
| 2 | PILA-11 | Search fix: reroll after a bonus earned with a 6 (flagged), study at depth 2 | +0 to +0.7pp at depth 2–3; correctness first | ~2 h code + ~40 min |
| 5 | PILA-12 | Offline representation probe: Markov flags, shots / opponent / blockade features, permutation augmentation, conv and token trunks, symmetric aux targets | go/no-go per arm; the winners become self-play issues | ~1 day code + ~2 h |
| 6 | PILA-13 | KPI validation and headroom study (Evaluator) | enables early kills; measures d3 vs d1 | ~1 h |
| 7 | PILA-14 | Low-variance value targets for cheap rows: luck-corrected return vs TD(λ) (replaces backlog #5) | +0.5 to +2pp vs the PILA-5 control | ~1 day code + ~3.6 h |

## Sources

- Tesauro, G. 1995. *Temporal Difference Learning and TD-Gammon.* Communications of the ACM 38(3).
  Raw-only TD-Gammon 0.0 vs feature-augmented 1.0–2.1; see also
  [TD-Gammon (Wikipedia)](https://en.wikipedia.org/wiki/TD-Gammon) and Kaelbling, Littman & Moore
  1996, [*Reinforcement Learning: A Survey*](https://arxiv.org/pdf/cs/9605103).
- Sutton, R. & Barto, A. 2018. *Reinforcement Learning: An Introduction*, 2nd ed., §16.1
  (TD-Gammon) and ch. 12 (λ-returns).
- GNU Backgammon documentation: neural-net inputs, contact/crashed/race nets, bear-off databases,
  error rate and Performance Rating.
- Antonoglou, I. et al. 2022. *Planning in Stochastic Environments with a Learned Model*
  (Stochastic MuZero). ICLR.
- Veness, J., Lanctot, M. & Bowling, M. 2011. *Variance Reduction in Monte-Carlo Tree Search.*
  NeurIPS. Control variates for chance nodes.
- Wu, D. 2019. *Accelerating Self-Play Learning in Go* (KataGo).
  [arXiv:1902.10565](https://arxiv.org/abs/1902.10565). Auxiliary targets for sample efficiency.
- Zaheer, M. et al. 2017. *Deep Sets.* NeurIPS. Lee, J. et al. 2019. *Set Transformer.* ICML.
- Alvi, F. & Ahmed, M. 2011; Alhajry, M., Alvi, F. & Ahmed, M. 2012. TD(λ) and Q-learning Ludo
  players. Practitioner deep-RL Ludo work uses egocentric per-token encodings, e.g.
  [this write-up](https://pub.towardsai.net/how-to-train-ai-to-play-multi-agent-ludo-a-guide-to-t-rex-and-reward-shaping-with-deep-q-network-4eab00c63c0f).
  Not peer-reviewed.
- Project records: `docs/V2_PLAN.md` §2, §10.7–10.11; `docs/LEADERBOARD.md` (PILA-1, PILA-6,
  PILA-8); `docs/AZ_DESIGN.md` (encoding item 6, aux head Phase 4.1, puzzle accuracy).
