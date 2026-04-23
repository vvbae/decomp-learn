#!/usr/bin/env bash
# decomp v1: contact-only DiffusionPolicy.
# Slices each demo at peg-box contact onset; policy learns insertion only.
# Edit variables then: bash decomp/v1/commands.sh <convert|train|eval|train_then_eval>

TRAJ_PATH="$HOME/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5"
DATASET_DIR="datasets/peg_insertion_996_contact"
REPO_ID="local/peg-insertion-side-contact-996"

OUTPUT_DIR="outputs/train/decomp_v1_996"
STEPS=200000
BATCH_SIZE=256
SAVE_FREQ=50000
LOG_FREQ=500

CKPT="$OUTPUT_DIR/checkpoints/last/pretrained_model"
NUM_EPISODES=100

convert() {
    uv run python baseline/diffusion/v2/convert.py \
        --traj-path "$TRAJ_PATH" \
        --output-dir "$DATASET_DIR" \
        --repo-id "$REPO_ID" \
        --contact-split
}

train() {
    uv run python decomp/v1/train.py \
        --dataset.repo_id="$REPO_ID" \
        --dataset.root="$DATASET_DIR" \
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
    uv run python baseline/diffusion/v2/eval.py \
        --ckpt "$CKPT" \
        --num-episodes "$NUM_EPISODES"
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
    *)
        echo "Usage: bash decomp/v1/commands.sh <convert|train|eval|train_then_eval>"
        exit 1
        ;;
esac
