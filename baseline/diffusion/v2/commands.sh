#!/usr/bin/env bash
# Edit the variables at the top, then run individual sections with bash commands.sh <command>
# Usage: bash commands.sh convert | train | eval

TRAJ_PATH="$HOME/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5"
DATASET_DIR="datasets/peg_insertion_996"
REPO_ID="local/peg-insertion-side-996"
TASK_NAME="PegInsertionSide-v1"

OUTPUT_DIR="outputs/train/v2_996"
STEPS=200000
BATCH_SIZE=256
SAVE_FREQ=50000
LOG_FREQ=500

CKPT="$OUTPUT_DIR/checkpoints/last/pretrained_model"
NUM_EPISODES=100

# ── commands ──────────────────────────────────────────────────────────────────

convert() {
    uv run python -m mani_skill.trajectory.convert_to_lerobot \
        --traj-path "$TRAJ_PATH" \
        --output-dir "$DATASET_DIR" \
        --task-name "$TASK_NAME"
}

train() {
    uv run baseline/diffusion/v2/train.py \
        --dataset.repo_id="$REPO_ID" \
        --dataset.root="$DATASET_DIR" \
        --policy.type=diffusion \
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
    uv run baseline/diffusion/v2/eval.py \
        --ckpt "$CKPT" \
        --num-episodes "$NUM_EPISODES"
}

train_then_eval() {
    tmux new-session -d -s run "
        cd $(pwd) &&
        bash baseline/diffusion/v2/commands.sh train &&
        bash baseline/diffusion/v2/commands.sh eval;
        read
    "
    echo "Started tmux session 'run'. Attach with: tmux attach -t run"
}

# ── entrypoint ────────────────────────────────────────────────────────────────

case "${1:-}" in
    convert)        convert ;;
    train)          train ;;
    eval)           eval ;;
    train_then_eval) train_then_eval ;;
    *)
        echo "Usage: bash commands.sh <convert|train|eval|train_then_eval>"
        exit 1
        ;;
esac
