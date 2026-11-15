# Leaderboard

Baseline for the "strongest Parchís agent" project (PILA-1, 2026-11-15). Every later experiment is
measured against the champion named here. To update the board, re-run the round robin with the new
checkpoint added to `scripts/leaderboard/round_robin.py::PARTICIPANTS`. Already-played pairings are
skipped and keep their seeds.

## Champions

| Format | Champion | Evidence |
|---|---|---|
| **2-player** | **V2.1c** — `runs/v2_1c_champion/model.pt` (search depth 1) | Top Bradley-Terry rating in 494 of 500 bootstrap replicates (V2.1d in the other 6). Beats every other checkpoint head to head: see the table below |
| **4-player** | **none — no 4-player checkpoint exists** | Every surviving checkpoint is 2-player (216-dim input, 2-seat value head). See Gaps |

V2.1c's margins: +7 Elo [+1, +12] over V2.1d, +16 [+10, +22] over the V2.0c/V2.0b/window group,
+60 over the round-23 champion, +175 over the tuned heuristic, +592 over random.

V2.1c head to head (duplicate pairs, depth 1, Wilson 95% CIs):

| V2.1c vs | Win rate | 95% CI | Games | Seeds | Source |
|---|---|---|---|---|---|
| V2.1d | 50.74% | [50.05, 51.43] | 20,000 | 70000+i | `runs/v2_1d_champion/h2h_vs_v2_1c.json` (inverted) |
| window-3 continuation | 51.56% | [50.86, 52.25] | 20,000 | 70000+i | `runs/window_test/h2h_win3_vs_v2_1c.json` (inverted) |
| window-10 continuation | 51.83% | [51.14, 52.52] | 20,000 | 70000+i | `runs/window_test/h2h_win10_vs_v2_1c.json` (inverted) |
| V2.0c | 52.49% | [51.80, 53.18] | 20,000 | 70000+i | `runs/v2_1c_champion/h2h_vs_v2_0c.json` |
| V2.0b | 52.25% | [50.06, 54.43] | 2,000 | round robin | this board |
| V2.1a | 53.25% | [51.06, 55.43] | 2,000 | round robin | this board |
| round-23 champion | 60.05% | [57.89, 62.18] | 2,000 | round robin | this board |
| tuned heuristic | 72.75% | [70.76, 74.66] | 2,000 | round robin | this board |
| random | 97.15% | [96.33, 97.79] | 2,000 | round robin | this board |

Since PILA-2, every promotion is decided by `docs/EVAL_PROTOCOL.md` (`scripts/eval/protocol.py`)
on fresh seed blocks; the champion pointer is `runs/champions.json`. First protocol run (calibration,
block 1): V2.1d scores 48.93% [48.24, 49.62] against V2.1c over 20,000 games, seeds 5001000+c,
verdict FAIL (`runs/v2_1d_champion/eval/protocol_b1.json`), which replicates the row above on new deals.

The 20k-game matches were run before this board, under the same protocol on a different seed range.
In this board the same pairs read 50.6 / 50.6 / 51.4 / 51.9, consistent within ±2.2pp.

**Bottom line:** the 2-player champion is real but its lead over the next tier is small, about
0.7–2.5pp per game (≈5–17 Elo). Ranks 3–7 are statistically tied with each other.

## Protocol

- **Format:** 2 players, max 500 turns. Every agent is the net plus depth-1 expectimax search
  (`parchis.az.agent.make_search_agent_factory`), which is how every checkpoint was gated and laddered.
- **Pairing:** a full round robin of 16 participants, so 120 pairings. Each pairing is
  1,000 duplicate pairs (`parchis.evaluation.duplicate.play_duplicate_match`): 2,000 games, both
  seat orders on shared dice. That is 240,000 games in total, 30,000 per participant, run in 30 min
  on 8 spawn workers with one-thread BLAS.
- **Seeds:** pairing k (the index into `itertools.combinations(PARTICIPANTS, 2)`) splits into 8
  chunks. Chunk c uses seed `120000 + 1000·k + c`. This range does not overlap the V2 ladders
  (9000+i) or the 10k-pair head-to-heads (70000+i).
- **Ratings:** Bradley-Terry MLE (`parchis.evaluation.ratings.fit_ratings`) with random anchored at
  0, converted to the Elo scale (× 400/ln 10).
- **Rating CIs:** game-level parametric bootstrap with 500 replicates, seed 0. Each replicate redraws
  every pairing's wins as Binomial(n, p̂) and refits. The "Δ vs champion" CIs come from the same
  replicates. Pairwise win rates use Wilson 95% intervals.
- **Artefacts:** `runs/leaderboard_2026-11-15/` contains `pairings.jsonl` (one line per pairing,
  with specs and seeds), `leaderboard.json` (ratings, CIs and the full matrix) and `round_robin.log`.

Reproduce:

```bash
python scripts/leaderboard/round_robin.py runs/leaderboard_2026-11-15 pairs=1000 workers=8
python scripts/leaderboard/analyze.py runs/leaderboard_2026-11-15/pairings.jsonl \
    runs/leaderboard_2026-11-15/leaderboard.json reps=500 seed=0
```

## 2-player ratings

Elo is relative to random = 0. Δ is the Elo gap to V2.1c. Score is the participant's win share
over its 30,000 games. P(top) is the fraction of bootstrap replicates in which the participant is
rated first.

| # | Participant | Checkpoint | Elo [95% CI] | Δ vs champion [95% CI] | Score | P(top) |
|---|---|---|---|---|---|---|
| 1 | **v2_1c** | `runs/v2_1c_champion/model.pt` | 592 [584, 602] | — | 64.9% | 0.99 |
| 2 | **v2_1d** | `runs/v2_1d_champion/model.pt` | 586 [577, 595] | -7 [-12, -1] | 64.0% | 0.01 |
| 3 | **v2_0c** | `runs/v2_0c_champion/model.pt` | 577 [568, 586] | -16 [-22, -10] | 62.9% | 0.00 |
| 4 | **win10** | `runs/window_test/win10/model.pt` | 577 [567, 586] | -16 [-22, -10] | 62.9% | 0.00 |
| 5 | **win3** | `runs/window_test/win3/model.pt` | 576 [568, 585] | -16 [-22, -10] | 62.8% | 0.00 |
| 6 | **v2_0b** | `runs/v2_0b_champion/model.pt` | 576 [567, 585] | -16 [-22, -10] | 62.8% | 0.00 |
| 7 | **v2_1a** | `runs/v2_1a_champion/model.pt` | 574 [564, 583] | -19 [-25, -13] | 62.4% | 0.00 |
| 8 | **v2_0** | `runs/v2_0_champion/model.pt` | 553 [544, 562] | -40 [-45, -34] | 59.7% | 0.00 |
| 9 | **champ23** | `runs/selfplay_2p_v1_champion/model.pt` | 533 [524, 542] | -60 [-66, -54] | 57.0% | 0.00 |
| 10 | **bootstrap_v4_large** | `runs/bootstrap_2p_v4_large/model.pt` | 494 [485, 502] | -99 [-105, -93] | 51.6% | 0.00 |
| 11 | **bootstrap_v3** | `runs/bootstrap_2p_v3/model.pt` | 419 [410, 427] | -174 [-180, -168] | 41.6% | 0.00 |
| 12 | **heuristic_tuned** | `heuristic:tuned` | 417 [409, 427] | -175 [-181, -169] | 41.4% | 0.00 |
| 13 | **c10_start_2x256** | `runs/c10_precheck/start_2x256.pt` | 384 [376, 393] | -208 [-214, -203] | 37.2% | 0.00 |
| 14 | **c10_start_3x512** | `runs/c10_precheck/start_3x512.pt` | 365 [356, 374] | -228 [-234, -222] | 34.8% | 0.00 |
| 15 | **heuristic_default** | `heuristic:default` | 307 [298, 316] | -286 [-292, -280] | 28.1% | 0.00 |
| 16 | **random** | — | 0 [0, 0] | -592 [-602, -584] | 6.0% | 0.00 |

## 2-player win-rate matrix

Each cell is the row agent's win % against the column agent, from 2,000 games. Wilson 95%
half-width is ±2.2pp near 50% and ±1.2pp near 90%. Exact counts and CIs are in
`runs/leaderboard_2026-11-15/pairings.jsonl`.
Abbreviations: ch23 = champ23, bs_v4/bs_v3 = bootstrap_v4_large/v3, h_tun/h_def = heuristic
tuned/default, c10_2x/c10_3x = C10 start nets.

| row vs col | v2_1c | v2_1d | v2_0c | win10 | win3 | v2_0b | v2_1a | v2_0 | ch23 | bs_v4 | bs_v3 | h_tun | c10_2x | c10_3x | h_def | rand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **v2_1c** | — | 50.5 | 51.9 | 51.4 | 50.6 | 52.2 | 53.2 | 54.9 | 60.1 | 66.3 | 74.2 | 72.8 | 77.1 | 79.0 | 82.0 | 97.2 |
| **v2_1d** | 49.5 | — | 52.7 | 50.1 | 52.3 | 50.9 | 53.0 | 56.9 | 57.5 | 63.3 | 72.0 | 69.9 | 75.4 | 78.6 | 81.3 | 97.1 |
| **v2_0c** | 48.1 | 47.3 | — | 50.9 | 49.3 | 51.7 | 49.4 | 54.6 | 56.5 | 62.1 | 71.5 | 70.0 | 76.0 | 76.8 | 82.2 | 96.8 |
| **win10** | 48.6 | 49.9 | 49.1 | — | 50.4 | 50.5 | 51.2 | 53.6 | 55.3 | 62.9 | 71.7 | 68.9 | 74.9 | 77.6 | 81.2 | 97.0 |
| **win3** | 49.4 | 47.7 | 50.7 | 49.6 | — | 51.7 | 49.7 | 55.2 | 56.8 | 61.5 | 71.7 | 68.6 | 74.4 | 77.2 | 80.1 | 97.6 |
| **v2_0b** | 47.8 | 49.1 | 48.3 | 49.5 | 48.3 | — | 50.0 | 54.2 | 56.9 | 63.5 | 71.9 | 71.0 | 74.7 | 77.6 | 81.8 | 96.8 |
| **v2_1a** | 46.8 | 46.9 | 50.6 | 48.8 | 50.3 | 50.0 | — | 53.0 | 57.0 | 60.5 | 70.7 | 69.6 | 76.0 | 77.3 | 81.3 | 97.5 |
| **v2_0** | 45.1 | 43.1 | 45.4 | 46.4 | 44.8 | 45.8 | 47.0 | — | 55.2 | 59.9 | 69.9 | 68.6 | 72.0 | 74.2 | 80.7 | 96.8 |
| **champ23** | 40.0 | 42.5 | 43.5 | 44.7 | 43.2 | 43.1 | 43.0 | 44.8 | — | 55.5 | 67.3 | 67.1 | 70.7 | 73.3 | 79.0 | 96.8 |
| **bootstrap_v4_large** | 33.7 | 36.8 | 38.0 | 37.1 | 38.6 | 36.5 | 39.5 | 40.1 | 44.5 | — | 61.1 | 63.5 | 65.0 | 67.0 | 77.8 | 95.2 |
| **bootstrap_v3** | 25.8 | 28.0 | 28.5 | 28.3 | 28.3 | 28.1 | 29.3 | 30.1 | 32.7 | 38.9 | — | 52.8 | 55.1 | 58.1 | 66.9 | 92.5 |
| **heuristic_tuned** | 27.3 | 30.1 | 30.0 | 31.1 | 31.4 | 29.0 | 30.4 | 31.4 | 32.9 | 36.5 | 47.2 | — | 54.6 | 55.1 | 64.2 | 90.1 |
| **c10_start_2x256** | 22.9 | 24.6 | 24.0 | 25.1 | 25.6 | 25.3 | 24.0 | 27.9 | 29.3 | 35.0 | 44.9 | 45.4 | — | 53.5 | 60.7 | 89.9 |
| **c10_start_3x512** | 21.0 | 21.4 | 23.2 | 22.4 | 22.8 | 22.4 | 22.7 | 25.7 | 26.7 | 33.0 | 41.9 | 44.9 | 46.5 | — | 59.2 | 88.3 |
| **heuristic_default** | 17.9 | 18.7 | 17.8 | 18.8 | 19.9 | 18.1 | 18.7 | 19.3 | 21.1 | 22.2 | 33.1 | 35.8 | 39.2 | 40.8 | — | 80.5 |
| **random** | 2.9 | 2.9 | 3.2 | 2.9 | 2.4 | 3.2 | 2.5 | 3.2 | 3.2 | 4.8 | 7.5 | 9.9 | 10.1 | 11.7 | 19.6 | — |

## Checkpoint inventory

Every checkpoint on disk (repo `runs/` plus `~/parchis_training_archive/`). All are AlphaZero-style
nets (`parchis.az.net.AZNet`) with policy, value and aux heads, 2 players, 216-dim input, trained
with `parchis.az`, and played at search depth 1. None comes from the MaskablePPO pipeline
(`parchis/training`); see Gaps.

| Name | Path | Trunk | How it was trained | Start point | Key config |
|---|---|---|---|---|---|
| bootstrap_v3 | `runs/bootstrap_2p_v3/` | 2×256 | Phase 2 supervised bootstrap on 20k mixed-pool games (heuristic, ε-heuristic, random) | random init | lr 1e-3, value weight 2.0, ε 0.15 |
| bootstrap_v4_large | `runs/bootstrap_2p_v4_large/` | 2×256 | Same as v3 on 200k games | random init | same as v3; ECE 0.0021 |
| champ23 | `runs/selfplay_2p_v1_champion/` (copy: `~/parchis_training_archive/selfplay_2p_v1/champion.pt`) | 2×256 | Phase 3 gated self-play, 132 rounds, 3 promotions (last at round 23) | bootstrap_v4_large | τ_target 0.25, Dirichlet ε 0.25, 600-pair gate, escalation to depth 2 |
| v2_0 | `runs/v2_0_champion/` | 2×256 | V2.0: 20 ungated rounds | champ23 | τ 0.03→0.015, ε 0, lr 1e-3, root-value targets, 5% rollout targets, aux 0.2 |
| v2_0b | `runs/v2_0b_champion/` | 2×256 | V2.0b: 10 ungated rounds | v2_0 | V2.0 + τ_end 0.005, lr 3e-4, no random anchor, heuristic in pool |
| v2_1a | `runs/v2_1a_champion/` | 2×256 | V2.1a: 10 rounds | v2_0 | V2.0b + afterstate rows |
| v2_0c | `runs/v2_0c_champion/` | 2×256 | V2.0c: 20 rounds (control for V2.1c) | v2_0b | V2.0b config |
| **v2_1c** | `runs/v2_1c_champion/` | 2×256 | V2.1c: 20 rounds, C2 deep targets | v2_0b | V2.0b + afterstate rows + deep_fraction 0.25 at depth 2, 8 workers |
| v2_1d | `runs/v2_1d_champion/` | 2×256 | V2.1d: 20 rounds | v2_0b | V2.1c with deep_fraction 0.5 |
| win3 | `runs/window_test/win3/` | 2×256 | 20-round continuation of V2.1c | v2_1c | V2.1c recipe, replay window 3 |
| win10 | `runs/window_test/win10/` | 2×256 | 20-round continuation of V2.1c | v2_1c | V2.1c recipe, replay window 10 |
| c10_start_2x256 | `runs/c10_precheck/start_2x256.pt` | 2×256 | C10 pre-check: fresh net fit on 24 shards (18k games) of V2.1c self-play | random init | lr 1e-3, patience 4 |
| c10_start_3x512 | `runs/c10_precheck/start_3x512.pt` | 3×512 | Same as above | random init | same |

Not ranked: `runs/v2_0_champion/round09_candidate.pt`, `runs/v2_0b_champion/round04_candidate.pt`
and `runs/v2_1a_champion/round04_candidate.pt` are mid-run candidates, not endpoints. Each run's
`rounds/` directory holds only per-round metrics, not weights.

## Gaps and caveats

1. **No 4-player agent exists.** Every checkpoint is 2-player, and the AZ encoding and value head
   are sized by `num_players`, so a 2p net cannot play a 4p seat. The 4-player matrix asked for in
   PILA-1 cannot be built until a 4p net is trained (V2_PLAN's next phase, V2.3).
   `parchis.evaluation.multiplayer_matrix` and `elo_ladder` also only load MaskablePPO `.zip`
   checkpoints, so they would need an AZ-agent path (as `parchis.evaluation.ladder` has) before any
   4p ranking.
2. **The PPO-era checkpoints are missing on this machine.** `models/` is gitignored and absent from
   this checkout. `~/parchis_training_archive/` has only `selfplay_2p_v1`. The original working
   copy appears to live under iCloud Drive (`~/Library/Mobile Documents/.../Academia/Reinforcement`,
   per paths recorded in `promoted_history.json`), but reading it timed out from the agent runtime,
   so any MaskablePPO `.zip` models there (2p or 4p) are uninventoried. The PPO line was never
   measured against the AZ line: the planned "frozen PPO flagship (`small_win_loss_combo15_seed42`)"
   rung (AGENT_REBUILD_PLAN §5.2) never appears in `runs/pairings.jsonl`. Placing it needs that
   folder made readable to the runtime, or the `.zip` copied into `~/parchis_training_archive/`.
   A 4p PPO model, if one exists, would be the only 4p entry.
3. **Window-test weights.** `runs/window_test/{win3,win10}/model.pt` are the repo mirrors of
   `~/parchis_training_archive/window_runs/*/champion.pt`, which the h2h JSONs cite but which is not
   on disk any more. The mirrors are what this board ranked.
4. **Bradley-Terry fit is overdispersed** (Pearson χ²/df ≈ 2.1). Most of the misfit comes from the
   baselines: random scores 19.6% against the default heuristic where the model predicts 14.6%, and
   the heuristics do a little better than predicted against the strong nets, so strength here is
   not one-dimensional. The bootstrap Elo CIs assume the model is correct and are therefore up to
   ~1.4× too narrow. Pairwise Wilson CIs and the 20k-game head-to-heads do not depend on the model,
   which is why the champion claim rests on them.
5. **One search depth only.** Everything is ranked at depth 1. M9 in `docs/V2_PLAN.md` showed depth
   2 is worth about +7pp for the same net, so "strongest agent" at a larger time budget means
   V2.1c at depth ≥2. That is a different axis from checkpoint choice and is not ranked here.
6. **Selection effect.** V2.1c was picked as the best of several runs (V2_PLAN §10.11), so some of
   its 0.7–2.5pp lead over its siblings is selection. Checkpoint-to-checkpoint spread around the
   recipe's level is about ±1.5pp. Treat V2.1c as "the recipe's best draw", not as a better recipe
   than V2.1d.
7. **C10 start nets are not lineage nets.** They are one-shot supervised fits used only for the
   capacity pre-check, and they rank below the tuned heuristic. This fits §10.10's finding that the
   lineage's warm-started accumulation matters more than net size.
