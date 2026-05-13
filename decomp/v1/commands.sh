#!/usr/bin/env bash
# decomp v1: contact-only DiffusionPolicy.
# Slices each demo at peg-box contact onset; policy learns insertion only.
# Train/test split: demos 0-795 (train), 796-995 (test).
# Edit variables then: bash decomp/v1/commands.sh <convert|train|eval|train_then_eval>

TRAJ_PATH="$HOME/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5"

# Train split (demos 0-795, 796 demos)
TRAIN_DATASET_DIR="datasets/peg_insertion_train796_contact_delta"
TRAIN_REPO_ID="local/peg-insertion-contact-delta-train796"
TRAIN_NUM_DEMOS=796
TRAIN_START_DEMO=0

# Test split (demos 796-995, 200 demos)
TEST_START_EPISODE=796
TEST_NUM_EPISODES=200

OUTPUT_DIR="outputs/train/decomp_v1_delta_train796"
STEPS=100000       # hard cap; early stopping will trigger before this
BATCH_SIZE=256
SAVE_FREQ=100000   # safety fallback; best checkpoint saved by early stopping
LOG_FREQ=500

# For train_curve: no early stopping, checkpoint every N steps
CURVE_OUTPUT_DIR="outputs/train/decomp_v1_delta_curve796"
CURVE_STEPS=100000
CURVE_SAVE_FREQ=5000

CKPT="$OUTPUT_DIR/checkpoints/last/pretrained_model"

convert() {
    uv run python baseline/diffusion/v2/convert.py \
        --traj-path "$TRAJ_PATH" \
        --output-dir "$TRAIN_DATASET_DIR" \
        --repo-id "$TRAIN_REPO_ID" \
        --num-demos "$TRAIN_NUM_DEMOS" \
        --start-demo "$TRAIN_START_DEMO" \
        --contact-split
}

train() {
    uv run python decomp/v1/train.py \
        --dataset.repo_id="$TRAIN_REPO_ID" \
        --dataset.root="$TRAIN_DATASET_DIR" \
        --policy.type=diffusion \
        --policy.push_to_hub=false \
        --policy.down_dims="[256,512,1024]" \
        --policy.noise_scheduler_type=DDIM \
        --policy.num_inference_steps=10 \
        --batch_size="$BATCH_SIZE" \
        --steps="$STEPS" \
        --save_freq="$SAVE_FREQ" \
        --log_freq="$LOG_FREQ" \
        --output_dir="$OUTPUT_DIR"
}

eval() {
    uv run python decomp/v1/eval.py \
        --ckpt "$CKPT" \
        --start-episode "$TEST_START_EPISODE" \
        --num-episodes "$TEST_NUM_EPISODES"
}

train_curve() {
    # Train N=796 with checkpoints every 5k steps, early stopping disabled.
    # After this, run eval_curve to find where success rate plateaus.
    ES_PATIENCE=9999 uv run python decomp/v1/train.py \
        --dataset.repo_id="$TRAIN_REPO_ID" \
        --dataset.root="$TRAIN_DATASET_DIR" \
        --policy.type=diffusion \
        --policy.push_to_hub=false \
        --policy.down_dims="[256,512,1024]" \
        --policy.noise_scheduler_type=DDIM \
        --policy.num_inference_steps=10 \
        --batch_size="$BATCH_SIZE" \
        --steps="$CURVE_STEPS" \
        --save_freq="$CURVE_SAVE_FREQ" \
        --log_freq="$LOG_FREQ" \
        --output_dir="$CURVE_OUTPUT_DIR"
}

eval_curve() {
    # Evaluate every checkpoint from train_curve on held-out demos 796-995.
    bash decomp/v1/eval_curve.sh \
        "$CURVE_OUTPUT_DIR/checkpoints" \
        100   # 100 episodes per checkpoint (fast; use 200 for final result)
}

train_then_eval() {
    tmux new-session -d -s decomp_v1 "
        cd $(pwd) &&
        bash decomp/v1/commands.sh train &&
        bash decomp/v1/commands.sh eval;
        read
    "
    echo "Started tmux session 'decomp_v1'. Attach with: tmux attach -t decomp_v1"
}

case "${1:-}" in
    convert)         convert ;;
    train)           train ;;
    eval)            eval ;;
    train_then_eval) train_then_eval ;;
    train_curve)     train_curve ;;
    eval_curve)      eval_curve ;;
    *)
        echo "Usage: bash decomp/v1/commands.sh <convert|train|eval|train_then_eval|train_curve|eval_curve>"
        exit 1
        ;;
esac
