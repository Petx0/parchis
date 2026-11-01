# Diagnostic scripts (2026-10-31 review)

Measurements behind `docs/V2_PLAN.md`. Run from the repo root with
`/opt/anaconda3/bin/python scripts/diagnostics/<script>.py ...`.

No-argument scripts load the round-23 champion (`runs/selfplay_2p_v1_champion/model.pt`); to point
one at another net, override `_common.CHAMPION` before exec (see §10 of the plan for the one-liner).

| script | measures |
|---|---|
| `profile_generation.py` | cProfile of `generate_round_games` at depth 1: where generation time goes |
| `leaf_and_target_check.py` | leaf-state mismatch (pre-roll vs post-roll value), depth-1 vs depth-2 root bias, policy-target sharpness |
| `exploration_sweep.py` | self-play acting policy vs greedy: agreement rate and duplicate-match win rate per (tau, dirichlet eps) |
| `mp_scaling.py` | aggregate games/sec with 1/4/8/10 worker processes |
| `depth_matchup.py` | champion @depth 2 vs @depth 1, duplicate pairs pooled over 8 workers |

Run-analysis scripts (added after V2.0, `docs/V2_PLAN.md` §10.1) take arguments:

| script | usage | measures |
|---|---|---|
| `round_summary.py` | `<run_dir> [tau_target]` | per round: losses, decoded root bias/ECE, move gaps, acting P(best), value given away, telemetry match; C3 afterstate rows (policy weight 0) are excluded and counted separately |
| `net_agreement.py` | `<driver> <n_games> name=model.pt ...` | depth-1 move agreement between nets on greedy games; game length; policy-head vs search agreement |
| `value_quality.py` | `<shard.npz> name=model.pt ...` | log-loss, Brier, bias, ECE, resolution of each net's raw value head on fixed rows |
| `explore_cost.py` | `<model.pt> <tau_end> <n_pairs>` | exploring net vs its own greedy self, duplicate pairs |
