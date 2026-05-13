#!/usr/bin/env bash
set -e

DELTA_H5=~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5
DATASET_DIR=datasets/peg_insertion_delta_996
OUTPUT_DIR=outputs/train/baseline_delta_996

tmux new-session -d -s baseline_delta

tmux send-keys -t baseline_delta "cd $(pwd) && \
uv run python -m lerobot.scripts.train \
    --dataset.repo_id=local/peg-insertion-side-delta-996 \
    --dataset.root=$DATASET_DIR \
    --policy.type=diffusion \
    --policy.down_dims='[256,512,1024]' \
    --policy.noise_scheduler_type=DDIM \
    --policy.num_inference_steps=10 \
    --batch_size=256 \
    --steps=200000 \
    --save_freq=50000 \
    --log_freq=500 \
    --policy.push_to_hub=false \
    --output_dir=$OUTPUT_DIR && \
uv run python eval_lerobot_v2.py \
    --ckpt $OUTPUT_DIR/checkpoints/last/pretrained_model \
    --num-episodes 100" Enter

echo "Started in tmux session 'baseline_delta'. Attach with: tmux attach -t baseline_delta"
