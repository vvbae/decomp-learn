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

## Key Files

- Demo data: `~/.maniskill/demos/PegInsertionSide-v1/`
- Environment: `PegInsertionSide-v1` in ManiSkill3
- Contact threshold: force > 0.1N = contact onset
- Policy input: TCP pose relative to hole pose (not joint angles)
