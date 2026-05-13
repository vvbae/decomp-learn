#!/usr/bin/env bash
# Data efficiency sweep for decomp v1 contact policy.
# Trains at 50/100/200/500/796 demos with early stopping (val loss, patience=5).
# All evals run on held-out demos 796-995 (never seen during training).
#
# Usage:
#   bash decomp/v1/sweep.sh           # run all demo counts sequentially
#   bash decomp/v1/sweep.sh 50 100    # run specific counts only

set -e

TRAJ_PATH="$HOME/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5"

# Max gradient steps (early stopping will trigger before this for most runs)
MAX_STEPS=100000
BATCH_SIZE=256
SAVE_FREQ=100000   # only save via early-stopping logic; this is a safety fallback
LOG_FREQ=500

TEST_START=796
TEST_N=200

DEMO_COUNTS=(50 100 200 500 796)
if [ $# -gt 0 ]; then
    DEMO_COUNTS=("$@")
fi

RESULTS_FILE="outputs/sweep_contact_delta_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p outputs
printf "%-10s  %-8s  %-12s  %-9s  %s\n" \
    "demo_count" "stopped" "success_rate" "successes" "avg_steps" | tee "$RESULTS_FILE"
printf "%-10s  %-8s  %-12s  %-9s  %s\n" \
    "----------" "-------" "------------" "---------" "---------" | tee -a "$RESULTS_FILE"

for N in "${DEMO_COUNTS[@]}"; do
    DATASET_DIR="datasets/peg_insertion_train${N}_contact_delta"
    REPO_ID="local/peg-insertion-contact-delta-train${N}"
    OUTPUT_DIR="outputs/train/decomp_v1_delta_train${N}"
    CKPT="$OUTPUT_DIR/checkpoints/last/pretrained_model"

    echo ""
    echo "=============================="
    echo " N=$N demos"
    echo "=============================="

    # Convert
    echo "[1/3] Converting $N demos (train split 0..$((N-1)))..."
    uv run python baseline/diffusion/v2/convert.py \
        --traj-path "$TRAJ_PATH" \
        --output-dir "$DATASET_DIR" \
        --repo-id "$REPO_ID" \
        --num-demos "$N" \
        --start-demo 0 \
        --contact-split

    # Train (with early stopping — will stop before MAX_STEPS)
    echo "[2/3] Training (max $MAX_STEPS steps, early stopping enabled)..."
    TRAIN_LOG="/tmp/train_${N}.txt"
    uv run python decomp/v1/train.py \
        --dataset.repo_id="$REPO_ID" \
        --dataset.root="$DATASET_DIR" \
        --policy.type=diffusion \
        --policy.push_to_hub=false \
        --policy.down_dims="[256,512,1024]" \
        --policy.noise_scheduler_type=DDIM \
        --policy.num_inference_steps=10 \
        --batch_size="$BATCH_SIZE" \
        --steps="$MAX_STEPS" \
        --save_freq="$SAVE_FREQ" \
        --log_freq="$LOG_FREQ" \
        --output_dir="$OUTPUT_DIR" 2>&1 | tee "$TRAIN_LOG"

    # Extract stopped step from train log
    STOPPED_STEP=$(grep -oP "Early stopping at step \K[0-9]+" "$TRAIN_LOG" | tail -1)
    if [ -z "$STOPPED_STEP" ]; then
        STOPPED_STEP="${MAX_STEPS}(cap)"
    fi

    # Eval on held-out test set
    echo "[3/3] Evaluating on held-out demos $TEST_START-$((TEST_START+TEST_N-1))..."
    EVAL_LOG="/tmp/eval_${N}.txt"
    uv run python decomp/v1/eval.py \
        --ckpt "$CKPT" \
        --start-episode "$TEST_START" \
        --num-episodes "$TEST_N" 2>&1 | tee "$EVAL_LOG"

    # Parse results
    SUMMARY=$(grep "Success rate:" "$EVAL_LOG" | tail -1)
    AVG_STEPS=$(grep "Avg policy steps:" "$EVAL_LOG" | tail -1 | awk '{print $NF}')
    RATE=$(echo "$SUMMARY" | grep -oP '\d+\.\d+(?=%)' | head -1)
    COUNTS=$(echo "$SUMMARY" | grep -oP '\d+/\d+' | head -1)
    printf "%-10s  %-8s  %-12s  %-9s  %s\n" \
        "$N" "$STOPPED_STEP" "${RATE}%" "$COUNTS" "$AVG_STEPS" | tee -a "$RESULTS_FILE"
done

echo ""
echo "=============================="
echo " Sweep complete. Results:"
echo "=============================="
cat "$RESULTS_FILE"
