#!/usr/bin/env bash
set -e

tmux new-session -d -s baseline_996

tmux send-keys -t baseline_996 "cd $(pwd) && uv run python -m lerobot.scripts.train \
    --dataset.repo_id=local/peg-insertion-side-996 \
    --dataset.root=datasets/peg_insertion_996 \
    --policy.type=diffusion \
    --policy.down_dims='[256,512,1024]' \
    --policy.noise_scheduler_type=DDIM \
    --policy.num_inference_steps=10 \
    --batch_size=256 \
    --steps=200000 \
    --save_freq=50000 \
    --log_freq=500 \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/baseline_996 && \
uv run python eval_lerobot_v2.py \
    --ckpt outputs/train/baseline_996/checkpoints/last/pretrained_model \
    --num-episodes 100" Enter

echo "Started in tmux session 'baseline_996'. Attach with: tmux attach -t baseline_996"
