# Experiment registry: run records and the comparison report

Every run in this project leaves one machine-readable record, `runs/<run_id>/meta.json`, and one
command shows how all of them compare:

```
python -m parchis.registry compare            # markdown table (also --format csv|json, --sort elo|created|run)
python -m parchis.registry show <run_id>      # one run: config, seeds, checkpoints, every eval
python -m parchis.registry validate [<run_id>]  # exit 1 if a record is incomplete (handoff gate)
```

This replaces the interim rule in the lab rules (`config.json` + `README.md` per run). Owner: the
Evaluator (PILA-7). Seed policy and the seed registry are in [EVAL_PROTOCOL.md §5](EVAL_PROTOCOL.md).

## 1. The run record (`meta.json`, schema `parchis.run/1`)

| Field | Content |
|---|---|
| `run_id`, `title`, `kind`, `issue` | Directory name under `runs/`; one-line title; `training`, `evaluation` or `analysis`; the PILA issue |
| `status`, `created`, `finished`, `error` | `running` / `stopped` / `failed` / `finished`; UTC timestamps; the exception if it failed |
| `git` | `sha`, `dirty`, `branch`, `source`. `source` is `recorded`, or `backfill: …` for runs that predate the registry |
| `sessions` | One entry per start or resume: time, git SHA, dirty flag, command, PID. A run resumed on different code shows it here |
| `script`, `command` | Driver script and full command line |
| `config`, `config_history` | The full config (every dataclass field). A resume with a changed config keeps the old one in `config_history` |
| `init` | Warm-start checkpoint: `path`, `sha256`, owning `run_id` (or `note: random init`) |
| `seeds` | Seed values, `base_n` (training base 10,000,000·n), `policy` (`ok` or `legacy…`), how per-round seeds derive, and which registry entry owns n |
| `checkpoints` | `role` (`final`, `candidate`, …), `label`, `path`, `sha256`, `primary` (ranked by `compare`), `round` |
| `training` | `rounds_completed`, `promotions`, `selfplay_games` (or `dataset_games`), `wall_clock_s`, and per round: seconds, gate result, first generation seed, promotion seed |
| `selection` | How the reported checkpoint was chosen (endpoint fixed in advance, best-of-N, …). Needed for the best-of-N rule |
| `evals` | Every result played by this run's checkpoints (below) |
| `notes`, `doc` | Free text; the design doc section with the history |

Each **eval** entry: `kind` (`protocol`, `h2h`, `ladder`, `round_robin`, `gate`), `checkpoint` +
`checkpoint_sha256`, `opponent` (+ `opponent_sha256`, `opponent_name`), `win_rate` (score, draws
count ½), `ci` (Wilson 95%), `n_games`, `seeds`, `depth`, `source` (the raw result file), `tag`,
`recorded`. Protocol entries add `verdict`, `checks`, `block`, `selection`. Checkpoints are
matched by **sha256**, so a result stays attached to the right weights when files are copied or
renamed.

## 2. Who writes it — automatically

| Writer | What it records |
|---|---|
| `parchis.az.round_loop.run_continuous` (every V2 driver) | Creates the record on first start, adds a session on each resume, updates `training` and the `final` checkpoint after every round, marks `stopped` or `failed`. Optional `record=dict(issue=…, title=…, init_path=…, notes=…)` |
| `scripts/v2/ladder.py` (`run_ladder`) | One `ladder` eval per anchor, on the run owning the checkpoint |
| `scripts/v2/h2h.py` | One `h2h` eval on the run owning checkpoint *a* |
| `scripts/eval/protocol.py` | The protocol's matches as `protocol` evals with the verdict, on `runs/<run_id>`. `--dry-run` also reports whether the candidate's record is complete |
| `python -m parchis.registry curate <scratch_run_dir> <run_id>` | Copies a finished scratch run into `runs/<run_id>/` (`model.pt`, configs, per-round metrics; no shards or intermediate candidates) together with its record |

The run that owns a checkpoint is the nearest directory above it with a `meta.json`. A registry
write never stops training or an evaluation: a failure is logged as `[registry] WARNING` and the
job continues.

Not hooked: the legacy PPO scripts (`parchis/training/experiment_*.py`). Their checkpoints are not
on this machine and the line is retired. A new PPO run would need `begin_run` added to its driver.

## 3. Engineer workflow for a new training run

```python
from parchis.registry import record as registry
seed = registry.allocate_training_base("v2_2a", "PILA-N")   # or: python -m parchis.registry alloc-seed --run-id v2_2a --issue PILA-N
cfg = C.SelfPlayRoundConfig(run_name="v2_2a", ...)
registry.apply_seed_base(cfg, seed["base"])                  # generation/train/promotion seeds = 10,000,000·n
round_loop.run_continuous(cfg, runs_dir=SCRATCH, max_rounds=20, initial_champion_state_dict=init,
                          record=dict(issue="PILA-N", title="…", init_path=INIT))
```

Then `python -m parchis.registry curate <SCRATCH>/v2_2a v2_2a_champion`, run the ladders and h2h
on the curated checkpoint (they record themselves), write `selection` if the checkpoint was not
the fixed endpoint, and check `python -m parchis.registry validate v2_2a_champion` before handing
over. `validate` fails on legacy seeds (< 10,000,000) for any run that is not backfilled.

## 4. Evaluator handoff check

1. `python -m parchis.registry validate <run_id>` must print `OK`. If not, reassign to the Engineer
   with the printed gaps.
2. `python scripts/eval/protocol.py <ckpt> --run-id <run_id> --issue PILA-N --dry-run` repeats the
   check and shows the seed block it will use.
3. After the verdict, `python -m parchis.registry compare` gives the updated cross-run view.

## 5. The report

`compare` shows one row per primary checkpoint. For each row it shows: issue, date, init, rounds,
self-play games, training hours, seed base, git SHA, the result against the current champion
(`runs/champions.json`), the result against the tuned heuristic, the Bradley-Terry Elo from the
newest `runs/leaderboard_*/` board, and the latest protocol verdict.

When several results exist for a cell, it shows a protocol run first, then the result with the
most games, then by kind (h2h, round robin, ladder), preferring a result recorded by the run itself
over one mirrored from the opponent's record (marked ᵢ: score 1−x, CI mirrored). Every cell shows
its game count and source. A cell is empty when no measured result exists.

Snapshot on 2026-11-15, after the backfill (regenerate it with the command above, as this copy goes stale):

| Run | Issue | Date | Init | Rounds | Self-play games | Train h | Seeds | Git | vs champion v2_1c | vs tuned heuristic | Elo (leaderboard_2026-11-15) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **v2_1c_champion** (champion) |  | 2026-11-02 | v2_0b_champion | 20 | 120,000 | 1.6 | legacy 0 | c53cbe0† | — | 72.75% [70.76, 74.66] n=2k RR | 592 [584, 602] |  |
| v2_1d_champion |  | 2026-11-02 | v2_0b_champion | 20 | 120,000 | 2.4 | legacy 0 | 2b70f38† | 48.93% [48.24, 49.62] n=20k P | 70.80% [68.77, 72.75] n=2k P | 586 [577, 595] | FAIL |
| v2_0c_champion |  | 2026-11-01 | v2_0b_champion | 20 | 120,000 | 3.3 | legacy 0 | ee12fb9† | 47.51% [46.82, 48.20] n=20k Hᵢ | 69.95% [67.90, 71.92] n=2k RR | 577 [568, 586] |  |
| window_test/win10 |  | 2026-11-02 | v2_1c_champion | 20 | 120,000 | 2.4 | legacy 0 | 26d2c4e† | 48.17% [47.48, 48.86] n=20k H | 68.90% [66.84, 70.89] n=2k RR | 577 [567, 586] |  |
| window_test/win3 |  | 2026-11-02 | v2_1c_champion | 20 | 120,000 | 1.6 | legacy 0 | 26d2c4e† | 48.45% [47.75, 49.14] n=20k H | 68.60% [66.53, 70.60] n=2k RR | 576 [568, 585] |  |
| v2_0b_champion |  | 2026-10-31 | v2_0_champion | 10 | 60,000 | 1.7 | legacy 0 | ee12fb9† | 47.75% [45.57, 49.94] n=2k RR | 71.00% [68.97, 72.95] n=2k RR | 576 [567, 585] |  |
| v2_1a_champion |  | 2026-10-31 | v2_0_champion | 10 | 60,000 | 2.0 | legacy 0 | ee12fb9† | 46.75% [44.57, 48.94] n=2k RR | 69.60% [67.55, 71.58] n=2k RR | 574 [564, 583] |  |
| v2_0_champion |  | 2026-10-31 | selfplay_2p_v1_champion | 20 | 120,000 | 3.0 | legacy 0 | ee12fb9† | 45.15% [42.98, 47.34] n=2k RR | 68.60% [66.53, 70.60] n=2k RR | 553 [544, 562] |  |
| selfplay_2p_v1_champion |  | 2026-08-28 | bootstrap_2p_v4_large | 133 | 798,000 |  | legacy 0 | b706225† | 39.95% [37.82, 42.11] n=2k RR | 67.10% [65.01, 69.12] n=2k RR | 533 [524, 542] |  |
| bootstrap_2p_v4_large |  | 2026-08-28 | random |  | 200,000 |  | legacy 0 | b706225† | 33.70% [31.66, 35.80] n=2k RR | 63.50% [61.37, 65.58] n=2k RR | 494 [485, 502] |  |
| bootstrap_2p_v3 |  | 2026-08-28 | random |  | 20,000 |  | legacy 0 | b706225† | 25.75% [23.88, 27.71] n=2k RR | 52.75% [50.56, 54.93] n=2k RR | 419 [410, 427] |  |
| c10_precheck:2x256 |  | 2026-11-02 | random |  | 20,250 | 0.3 | legacy 0 | 26d2c4e† | 22.85% [21.06, 24.74] n=2k RR | 45.40% [43.23, 47.59] n=2k RR | 384 [376, 393] |  |
| c10_precheck:3x512 |  | 2026-11-02 | random |  | 20,250 | 0.3 | legacy 0 | 26d2c4e† | 21.00% [19.27, 22.84] n=2k RR | 44.85% [42.68, 47.04] n=2k RR | 365 [356, 374] |  |

Win rates: score of the run's checkpoint, Wilson 95% CI, games. Source: P protocol, RR round robin, H 10k-pair h2h, L ladder, G promotion gate; ᵢ = recorded by the opponent and mirrored. Git: † = backfilled (commit that added the run, not the training commit), * = dirty tree. Seeds: n = training base 10,000,000·n; legacy 0 overlaps evaluation ranges. Full records: `python -m parchis.registry show <run>`.

Records without a ranked checkpoint: leaderboard_2026-11-15 (evaluation), puzzles_2026-11-03 (evaluation).

## 6. Backfill of runs before the registry (2026-11-15)

`scripts/registry/backfill.py` built the 14 records above from files already in `runs/`. Nothing
was re-run, and the script can be re-run with the same result. What the old files could not give:

- **Git SHA:** the commit that *added* the run directory (`git.source = "backfill: …"`, † in the
  report), not the commit that trained it. The training SHA was never recorded.
- **Intermediate ladders** (round 5/10/15) were played on a `champion.pt` that later rounds
  overwrote. They are kept with `checkpoint_sha256: null`. Only the last ladder is tied to the kept
  `model.pt`.
- **Seeds:** every run so far used base 0 (`policy: legacy …`). C10's data used `500000+i`. That
  range is now also listed in the seed registry.
- **champ23** (`selfplay_2p_v1_champion`): `champion_meta.json` gives round index 132, so the record
  says 133 rounds (0–132). The docs say "132 rounds". Its per-round files are only in
  `~/parchis_training_archive/selfplay_2p_v1`.
- **Wall-clock** is the sum of the "round N finished in Xs" lines in `run.log` / `driver.log`. It
  excludes ladders. There is no wall-clock for the bootstraps or champ23.
- **Round robin:** every PILA-1 pairing is recorded in both participants' records, with the b side
  mirrored.
