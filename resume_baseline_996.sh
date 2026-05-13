#!/usr/bin/env bash
set -e

tmux new-session -d -s resume_996

tmux send-keys -t resume_996 "cd $(pwd) && uv run python -m lerobot.scripts.train \
    --config_path=outputs/train/baseline_996/checkpoints/last/pretrained_model/train_config.json \
    --resume=true && \
uv run python eval_lerobot_v2.py \
    --ckpt outputs/train/baseline_996/checkpoints/last/pretrained_model \
    --num-episodes 100" Enter

echo "Started in tmux session 'resume_996'. Attach with: tmux attach -t resume_996"
