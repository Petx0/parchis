## Puzzle run 2026-11-16: PILA-5 harvest nets vs earlier puzzle results (PILA-9)

Raw data: `report.json` (`my_puzzles.csv`, 10 puzzles), `report_tactical.json` (`tactical_puzzles.csv`, 2 puzzles). Run record: `meta.json`. Same schema as `runs/puzzles_2026-11-03/report.json`: every legal move's search value, the chosen piece and the time, per net and depth.

### Nets

| Name | Checkpoint | sha256 (prefix) | Win-rate context |
|---|---|---|---|
| S1 | `runs/soups_2026-11-16/s1/model.pt` | `3a86d3cf99b1` | Current champion (PILA-8) |
| h4 | `runs/pila5_harvest_s1/soups/h4/model.pt` | `d7b2b08552b5` | 52.70% [51.15, 54.24] vs S1, 4,000 games, depth 1 (PILA-5 screen, best-of-5, not protocol-confirmed) |
| c20 | `runs/pila5_harvest_s1/candidates/c20.pt` | `279401384dd5` | 45.77% [44.24, 47.32] vs S1, 4,000 games, depth 1 (PILA-5 screen) |
| V2.1c | `runs/v2_1c_champion/model.pt` | `30c1234fb42d` | Previous champion |
| round-23 | `runs/selfplay_2p_v1_champion/model.pt` | `fc33d73fe0ed` | 2026-08-29 puzzle baseline |

### Puzzles solved (of 10), by search depth

| Net | d1 | d2 | d3 | d4 | d5 | Sum over d1–d5 (of 50) |
|---|---|---|---|---|---|---|
| round-23 | 3 | 5 | 6 | 4 | 6 | 24 |
| V2.1c | 4 | 4 | 7 | 6 | 6 | 27 |
| **S1** | 4 | **7** | **7** | **8** | 6 | **32** |
| h4 | 5 | 5 | **7** | 7 | 6 | 30 |
| c20 | **6** | 4 | 6 | 6 | **7** | 29 |
| Seconds per decision (all nets) | 0.001 | 0.003 | 0.036 | 0.61 | 12.2 | |

The time per decision depends only on depth, because all nets share one shape (216 → 256 → 256).

### Per-puzzle grid (`+` = solved at d1..d5)

| # | Category | Rationale (from CSV) | round-23 | V2.1c | S1 | h4 | c20 |
|---|---|---|---|---|---|---|---|
| 1 | endgame | Keep the one at 10-distance | `++...` | `++++.` | `++++.` | `+++++` | `+++++` |
| 2 | midgame | Use the second six to enter color-final-strech to avoid 3rd 6 penalty | `.....` | `.....` | `.....` | `.....` | `.....` |
| 3 | endgame | 6-distance is not doable with 1 roll | `+++++` | `.++++` | `+++++` | `+++++` | `+++++` |
| 4 | midgame | Piece at #27 is threatened by a 10 bonus | `.....` | `..+++` | `...++` | `.....` | `.....` |
| 5 | midgame | Piece at #15 is threatened by a 20 bonus | `...++` | `..+++` | `.++++` | `..+++` | `+++++` |
| 14 | endgame | The other 3 pieces are already at 1-roll distance | `.++++` | `..+..` | `.+++.` | `.++++` | `....+` |
| 16 | endgame | Hail Mary, last change to capture | `.++++` | `+...+` | `+.+++` | `+.+++` | `+.+++` |
| 17 | midgame | Don't missuse the 10-bonuses | `..+.+` | `+++++` | `.++++` | `+++++` | `+++++` |
| 18 | midgame | Don't missuse the 10-bonuses | `..+..` | `+++++` | `+++++` | `+.++.` | `+.+++` |
| 19 | endgame | Keep the option of winning in one turn | `+++.+` | `.....` | `.+...` | `.+...` | `.....` |

### Changes (solved → failed, failed → solved)

**Harvest vs its own start point, S1:**

- **h4:**
  - d1: gains 17.
  - d2: loses 5 and 18.
  - d4: loses 4.
  - d5: gains 1 and 14, loses 4 and 18.
- **c20:**
  - d1: gains 5 and 17.
  - d2: loses 14, 18 and 19.
  - d3: loses 14.
  - d4: loses 4 and 14.
  - d5: gains 1 and 14, loses 4.
- **Consistent loss: puzzle 4** ("piece at #27 is threatened by a 10 bonus"). S1 solves it at d4–d5 and V2.1c at d3–d5, but neither harvest net solves it at any depth.
- **Consistent gain: puzzle 1** at d5 ("keep the one at 10-distance"). Both harvest nets solve it at every depth.

**S1 vs V2.1c:**

- d1: gains 3, loses 17.
- d2: gains 5, 14 and 19.
- d3: gains 16, loses 4.
- d4: gains 14 and 16.
- d5: no change.

**Never solved:**

- Puzzle 2 (third-six penalty) is failed by every net at every depth. For V2.1c, S1, h4 and c20, the correct move trails by 0.7 to 5.6 pp at every depth.
- Puzzle 19 ("keep the option of winning in one turn") is solved only by round-23, plus S1 and h4 at d2.

### Variance and how to read this

- **Search is deterministic.** It is exhaustive expectimax over all dice, so there is no seed variance to rerun. Re-running V2.1c and round-23 on today's code reproduced `runs/puzzles_2026-11-03/report.json` exactly: identical choices, and a maximum difference of 0 in any move value.
- **The uncertainty comes from the puzzle set itself, and from near-ties.** For each net and depth, 0 to 4 of the 10 decisions have the correct move within 1 pp of win probability of the best wrong move. A one- or two-puzzle difference between nets is noise. With 10 puzzles, this is a signal, not a strength ranking. Win rate under `docs/EVAL_PROTOCOL.md` stays the ranking metric.
- **Puzzles do not track win rate here.** c20 is clearly weaker than S1 in games (45.77% vs S1), yet it solves about as many puzzles (29 vs 32 over all depths, and more than S1 at d1 and d5). h4 is the stronger harvest net in games and is likewise about level with S1 (30 vs 32). The harvest's win-rate gain (h4) does not show up as a tactical gain on this set. The one visible tactical change is a regression on puzzle 4.
- **More depth does not mean more solved.** Every net peaks at d3–d4 or is flat after it, and d5 costs about 20× the time of d4.
- **The tactical set** (`tactical_puzzles.csv`, 2 puzzles) is 2/2 for every net at every depth except round-23 (1/2). It does not discriminate between these nets.
