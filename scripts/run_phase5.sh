#!/usr/bin/env bash
# Phase 5 runbook (docs/RL_DESIGN_REVIEW.md) -- NOT meant to be run
# start-to-finish in one sitting. Stages 1 and 3 are real, multi-hour
# training runs; Stage 2 is a manual step in between where you read
# Stage 1's results and fill in REDESIGNED_REWARD_TYPE / REDESIGNED_ALPHA
# below before Stage 3 can run.
#
# Usage: run one stage at a time, e.g.
#   bash scripts/run_phase5.sh stage1
#   ...(read results, edit this file's REDESIGNED_* variables)...
#   bash scripts/run_phase5.sh stage3
#   bash scripts/run_phase5.sh stage4
#
# Total estimated cost: Stage 1 ~6-10h, Stage 3 ~6-12h, Stage 4 <1h.

set -euo pipefail

# ── Stage 2 output: fill these in after reading Stage 1's results ─────────
# (models/phase5_sweep_grid/<timestamp>/results.json,
#  models/phase5_sweep_alpha/experiment_results.txt)
REDESIGNED_REWARD_TYPE="${REDESIGNED_REWARD_TYPE:-progress_delta}"
REDESIGNED_ALPHA="${REDESIGNED_ALPHA:-0.5}"

stage1() {
    echo "=== Phase 5 Stage 1: real screening sweep (reward_type x alpha) ==="
    python -m parchis.training.experiment_grid \
        --seeds 42 43 44 --timesteps 300000 --eval-episodes 100 --players 2 \
        --filter-arch medium \
        --save-path ./models/phase5_sweep_grid --log-path ./logs/phase5_sweep_grid

    python -m parchis.training.experiment_alpha_comparison \
        --alphas 0.0 0.5 0.9 --seeds 42 43 44 --timesteps 300000 --eval-episodes 100 --players 2 \
        --save-path ./models/phase5_sweep_alpha --log-path ./logs/phase5_sweep_alpha

    echo ""
    echo "Stage 1 done. Read the results above (or"
    echo "  models/phase5_sweep_grid/*/results.json"
    echo "  models/phase5_sweep_alpha/experiment_results.txt"
    echo "), then set REDESIGNED_REWARD_TYPE / REDESIGNED_ALPHA in this"
    echo "script (or as env vars) before running stage3."
}

stage3() {
    echo "=== Phase 5 Stage 3: real baseline vs. redesigned training ==="
    echo "Using REDESIGNED_REWARD_TYPE=${REDESIGNED_REWARD_TYPE} REDESIGNED_ALPHA=${REDESIGNED_ALPHA}"

    for seed in 42 43 44; do
        python -m parchis.training.train_ppo \
            --players 2 --timesteps 1000000 --seed "$seed" \
            --model-name "phase5_baseline_seed${seed}" \
            --save-path ./models/phase5 --log-path ./logs/phase5
    done

    for seed in 42 43 44; do
        python -m parchis.training.train_selfplay \
            --players 2 --timesteps 1000000 --seed "$seed" \
            --reward-type "$REDESIGNED_REWARD_TYPE" --opponent-weight "$REDESIGNED_ALPHA" \
            --pool-size 5 --pool-sampling-strategy uniform \
            --model-name "phase5_redesigned_seed${seed}" \
            --save-path ./models/phase5 --log-path ./logs/phase5
    done
}

stage4() {
    echo "=== Phase 5 Stage 4: Elo ladder + group comparison ==="
    python -m parchis.evaluation.elo_ladder \
        --checkpoints \
            ./models/phase5/phase5_baseline_seed42/final_model \
            ./models/phase5/phase5_baseline_seed43/final_model \
            ./models/phase5/phase5_baseline_seed44/final_model \
            ./models/phase5/phase5_redesigned_seed42/final_model \
            ./models/phase5/phase5_redesigned_seed43/final_model \
            ./models/phase5/phase5_redesigned_seed44/final_model \
        --checkpoint-names \
            baseline_42 baseline_43 baseline_44 \
            redesigned_42 redesigned_43 redesigned_44 \
        --games-per-pairing 40 \
        --save-path ./logs/phase5_elo_ladder

    python -m parchis.evaluation.group_comparison \
        --results-json ./logs/phase5_elo_ladder/results.json \
        --group-a baseline_42 baseline_43 baseline_44 \
        --group-b redesigned_42 redesigned_43 redesigned_44
}

case "${1:-}" in
    stage1) stage1 ;;
    stage3) stage3 ;;
    stage4) stage4 ;;
    *)
        echo "Usage: $0 {stage1|stage3|stage4}"
        echo "(there is no stage2 command -- it's the manual read-results-and-edit-this-file step)"
        exit 1
        ;;
esac
