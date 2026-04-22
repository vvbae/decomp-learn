# Agent Instructions

## Environment
- **Package manager**: always use `uv`. Never use `pip` or `python` directly.
  - Run scripts: `uv run python script.py`
  - Add packages: `uv add package_name`
  - Never: `pip install`, `python script.py`
- **Terminal**: we use `tmux`. Chain long-running commands in tmux sessions.
- **GPU**: RTX 5070, CUDA 12.8, Ubuntu 24

## Running Things

**Never start training or long eval runs automatically.** Always:
1. Show the exact command
2. Explain what it does and how long it will take
3. Wait for the user to run it

Example of what to do:
```
# Ready to run. This will take ~2 hours. Run in tmux:
tmux new -s train
uv run python train.py --config configs/contact_only.yaml
```

For chaining long runs in tmux:
```
uv run python train.py && uv run python eval.py --ckpt checkpoints/best.pth
```

## Training Code Requirements

Every training script must have:
1. **Checkpointing**: save checkpoint every N steps, always save best model
2. **Resume**: `--resume path/to/checkpoint.pth` flag to continue from checkpoint
3. **Logging**: print step, loss, lr at regular intervals
4. **Early stopping**: monitor validation loss, stop if no improvement

Example checkpoint pattern:
```python
# Save checkpoint
torch.save({
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_loss': best_loss,
}, f'checkpoints/step_{step}.pth')

# Resume
if args.resume:
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_step = ckpt['step']
```

## Research Practices

- **Never delete old checkpoints** without asking
- **Log everything**: training loss, val loss, success rate, step count
- **Reproducibility**: always set random seeds, log the seed used
- **Ablations**: design code so hyperparameters are in config files, not hardcoded
- **Baselines**: keep baseline code separate from method code, don't modify baselines after results are recorded
- When results look surprising (too good or too bad), validate before moving on

## Project Structure

```
decomp-learn/
├── PROJECT.md          # this project's overview
├── AGENTS.md           # this file
├── data/               # demo data and processed splits
├── configs/            # yaml config files for experiments
├── checkpoints/        # saved model weights
├── experiments/        # one folder per experiment run
├── scripts/            # data processing, eval scripts
└── results/            # logged metrics, plots
```

## Smoke Tests

Before claiming any script works, run the corresponding smoke test. All smoke tests should complete in under 60 seconds.

### baseline/diffusion/v1

```bash
# 1 step of training, tiny dataset
uv run baseline/diffusion/v1/train.py \
    --num-demos 2 --num-steps 2 --batch-size 2 \
    --exp-name smoke_v1 --ckpt-dir /tmp/smoke_v1

# eval: 2 episodes
uv run baseline/diffusion/v1/eval.py \
    --ckpt /tmp/smoke_v1/smoke_v1.pth --num-episodes 2
```

### baseline/diffusion/v2

```bash
# convert 2 demos only
uv run python -m mani_skill.trajectory.convert_to_lerobot \
    --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5 \
    --output-dir /tmp/smoke_ds_v2 \
    --task-name PegInsertionSide-v1

# 2 steps of training
uv run baseline/diffusion/v2/train.py \
    --dataset.repo_id=local/smoke \
    --dataset.root=/tmp/smoke_ds_v2 \
    --policy.type=diffusion \
    --batch_size=2 --steps=2 --save_freq=2 \
    --output_dir=/tmp/smoke_v2

# eval against the saved checkpoint
uv run baseline/diffusion/v2/eval.py \
    --ckpt /tmp/smoke_v2/checkpoints/000002/pretrained_model \
    --num-episodes 2
```

### baseline/diffusion/v3

```bash
# convert same as v2 above (reuse /tmp/smoke_ds_v2), then:
uv run python -m lerobot.scripts.train \
    --dataset.repo_id=local/smoke \
    --dataset.root=/tmp/smoke_ds_v2 \
    --policy.type=diffusion \
    --batch_size=2 --steps=2 --save_freq=2 \
    --output_dir=/tmp/smoke_v3

uv run baseline/diffusion/v2/eval.py \
    --ckpt /tmp/smoke_v3/checkpoints/000002/pretrained_model \
    --num-episodes 2
```

**What to check after each smoke test:**
- No import errors or missing keys
- Checkpoint directory exists and contains `pretrained_model/model.safetensors`
- Eval runs without crashing (success rate doesn't matter with 2 steps)

## Key Files

- Demo data: `~/.maniskill/demos/PegInsertionSide-v1/`
- Environment: `PegInsertionSide-v1` in ManiSkill3
- Contact threshold: force > 0.1N = contact onset
- Policy input: TCP pose relative to hole pose (not joint angles)
