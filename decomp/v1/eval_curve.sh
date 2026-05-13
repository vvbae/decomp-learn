#!/usr/bin/env bash
# Evaluate every step-checkpoint in a training run on the held-out test set.
# Prints a steps → success-rate table to identify the training plateau.
#
# Usage:
#   bash decomp/v1/eval_curve.sh [ckpt_parent_dir] [num_test_episodes]
#
# Defaults:
#   ckpt_parent_dir  = outputs/train/decomp_v1_delta_curve796/checkpoints
#   num_test_episodes = 100   (enough to see the trend; use 200 for final numbers)

set -e

CKPT_PARENT="${1:-outputs/train/decomp_v1_delta_curve796/checkpoints}"
TEST_N="${2:-100}"
TEST_START=796

RESULTS_FILE="outputs/curve_$(basename "$(dirname "$CKPT_PARENT")")_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p outputs

printf "%-8s  %-12s  %-9s\n" "steps" "success_rate" "successes" | tee "$RESULTS_FILE"
printf "%-8s  %-12s  %-9s\n" "--------" "------------" "---------" | tee -a "$RESULTS_FILE"

for step_dir in $(ls -d "$CKPT_PARENT"/[0-9]* 2>/dev/null | sort -V); do
    step=$(basename "$step_dir" | sed 's/^0*//')
    ckpt="$step_dir/pretrained_model"
    [ -d "$ckpt" ] || continue

    result=$(uv run python decomp/v1/eval.py \
        --ckpt "$ckpt" \
        --start-episode "$TEST_START" \
        --num-episodes "$TEST_N" 2>&1)

    rate=$(echo "$result"   | grep -oP '\d+\.\d+(?=%)' | tail -1)
    counts=$(echo "$result" | grep -oP '\d+/\d+' | tail -1)

    printf "%-8s  %-12s  %-9s\n" "$step" "${rate}%" "$counts" | tee -a "$RESULTS_FILE"
done

echo ""
echo "Results saved to $RESULTS_FILE"
