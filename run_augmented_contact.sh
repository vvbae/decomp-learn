#!/usr/bin/env bash
# Train DiffusionPolicy on augmented contact-phase data.
#
# Steps (run inside a detached tmux session):
#   1. Convert augmented H5 -> LeRobot v2.1 dataset (skipped if already done)
#   2. Train DiffusionPolicy
#   3. Evaluate the final checkpoint
#
# Fix note: --dataset.root must point to the local dataset directory so LeRobot
# reads from disk rather than fetching from HuggingFace.  The repo_id
# ("local/augmented-contact-996") is just a label; the actual path is root.
set -e

H5_PATH=demos/augmented_contact/augmented_contact.h5
DATASET_DIR=datasets/augmented_contact_996
OUTPUT_DIR=outputs/train/augmented_contact

tmux new-session -d -s train-contact

tmux send-keys -t train-contact "cd $(pwd) && \
if [ ! -d $DATASET_DIR/meta ]; then \
  uv run python convert_augmented_to_lerobot.py \
      --h5-path $H5_PATH \
      --output-dir $DATASET_DIR; \
fi && \
uv run python -m lerobot.scripts.train \
    --dataset.repo_id=local/augmented-contact-996 \
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
    --num-episodes 200" Enter

echo "Started in tmux session 'train-contact'. Attach with: tmux attach -t train-contact"
