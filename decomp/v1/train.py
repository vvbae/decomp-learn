"""
Train contact-only DiffusionPolicy (decomp v1).

Splits each demo at contact onset (--contact-split) so the policy only learns
the insertion phase (~29 steps, 20% of each trajectory).
Obs/action space is identical to the e2e baseline for now; embodiment-agnostic
features come in v2.

Early stopping: last 10% of training episodes held out as validation set.
Val loss checked every VAL_FREQ steps; training stops after ES_PATIENCE
consecutive checks without improvement. Best checkpoint is saved to `last/`.

Usage:
    uv run decomp/v1/train.py \
        --dataset.repo_id=local/peg-insertion-side-contact-delta-train796 \
        --dataset.root=datasets/peg_insertion_train796_contact_delta \
        --policy.type=diffusion \
        --policy.push_to_hub=false \
        --policy.down_dims="[256,512,1024]" \
        --policy.noise_scheduler_type=DDIM \
        --policy.num_inference_steps=10 \
        --batch_size=256 --steps=100000 \
        --save_freq=100000 \
        --log_freq=500 \
        --output_dir=outputs/train/decomp_v1_delta_train796
"""

import logging
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from termcolor import colored
from torch.amp import GradScaler

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger

# --------------------------------------------------------------------------- #
# Early-stopping hyper-parameters (all overridable via env vars)
# --------------------------------------------------------------------------- #
VAL_RATIO   = 0.10   # fraction of training episodes held out for validation
VAL_FREQ    = int(os.environ.get("VAL_FREQ",    "1000"))
ES_PATIENCE = int(os.environ.get("ES_PATIENCE", "5"))    # set to 9999 to disable
VAL_BATCHES = 50     # mini-batches averaged for each val-loss estimate

TRAJ_PATH = os.environ.get(
    "TRAJ_PATH",
    os.path.expanduser(
        "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/"
        "trajectory.state.pd_joint_delta_pos.physx_cpu.h5"
    ),
)

CONVERT_SCRIPT = Path(__file__).parent.parent.parent / "baseline" / "diffusion" / "v2" / "convert.py"


def _maybe_convert_dataset(cfg: TrainPipelineConfig) -> None:
    if cfg.dataset.root is None:
        return
    root = Path(cfg.dataset.root)
    if root.exists():
        return
    logging.info(f"Dataset not found at {root} — converting (contact-split) from HDF5...")
    subprocess.run(
        [
            sys.executable,
            str(CONVERT_SCRIPT),
            "--traj-path", TRAJ_PATH,
            "--output-dir", str(root),
            "--repo-id", cfg.dataset.repo_id,
            "--contact-split",
        ],
        check=True,
    )


def update_policy(train_metrics, policy, batch, optimizer, grad_clip_norm, grad_scaler, lr_scheduler=None, use_amp=False):
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=False)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(policy, "update"):
        policy.update()
    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@torch.no_grad()
def _compute_val_loss(policy, val_iter, device: torch.device, n_batches: int) -> float:
    policy.eval()
    losses = []
    for _ in range(n_batches):
        batch = next(val_iter)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
        loss, _ = policy.forward(batch)
        losses.append(loss.item())
    policy.train()
    return float(np.mean(losses))


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    wandb_logger = WandBLogger(cfg) if (cfg.wandb.enable and cfg.wandb.project) else None
    if wandb_logger is None:
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    _maybe_convert_dataset(cfg)

    # --------------------------------------------------------------------- #
    # Dataset: load full set, split 90/10 by episode index.
    #
    # We keep a single LeRobotDataset (full_dataset) and never filter it via
    # the `episodes` constructor param — that path remaps episode_data_index
    # to 0-based while item["episode_index"] stays original, causing OOB errors.
    # Instead we restrict training to the first n_train_ep episodes via a
    # custom sampler, and restrict val to the remaining frames via Subset.
    # --------------------------------------------------------------------- #
    logging.info("Creating dataset")
    full_dataset = make_dataset(cfg)

    all_ep     = full_dataset.num_episodes
    n_val_ep   = max(1, int(all_ep * VAL_RATIO))
    n_train_ep = all_ep - n_val_ep
    logging.info(f"Train episodes: {n_train_ep}  |  Val episodes: {n_val_ep}")

    ep_idx = full_dataset.episode_data_index
    # Filtered episode_data_index for the training portion only
    train_ep_data_index = {
        "from": ep_idx["from"][:n_train_ep],
        "to":   ep_idx["to"][:n_train_ep],
    }
    # Collect global frame indices for the val portion
    val_frame_indices = [
        i
        for ep in range(n_train_ep, all_ep)
        for i in range(int(ep_idx["from"][ep]), int(ep_idx["to"][ep]))
    ]
    val_subset = torch.utils.data.Subset(full_dataset, val_frame_indices)

    n_train_frames = sum(
        int(ep_idx["to"][ep]) - int(ep_idx["from"][ep]) for ep in range(n_train_ep)
    )

    # --------------------------------------------------------------------- #
    # Policy, optimiser
    # --------------------------------------------------------------------- #
    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=full_dataset.meta)

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"max steps={cfg.steps}  (early stopping: VAL_FREQ={VAL_FREQ}, patience={ES_PATIENCE})")
    logging.info(f"train frames={n_train_frames}  train episodes={n_train_ep}")
    logging.info(f"val frames={len(val_frame_indices)}  val episodes={n_val_ep}")
    logging.info(f"learnable params: {format_big_number(num_learnable)}")

    # --------------------------------------------------------------------- #
    # Data loaders
    # --------------------------------------------------------------------- #
    if hasattr(cfg.policy, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(
            train_ep_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        shuffle = False
    else:
        sampler = torch.utils.data.SubsetRandomSampler(
            [i for ep in range(n_train_ep)
             for i in range(int(ep_idx["from"][ep]), int(ep_idx["to"][ep]))]
        )
        shuffle = False

    train_loader = torch.utils.data.DataLoader(
        full_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        num_workers=0,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dl_iter  = cycle(train_loader)
    val_iter = cycle(val_loader)

    # --------------------------------------------------------------------- #
    # Metrics
    # --------------------------------------------------------------------- #
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size, n_train_frames, n_train_ep,
        train_metrics, initial_step=step,
    )

    # --------------------------------------------------------------------- #
    # Training loop with early stopping
    # --------------------------------------------------------------------- #
    best_val_loss = float("inf")
    es_counter    = 0

    logging.info("Start offline training")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker, policy, batch, optimizer,
            cfg.optimizer.grad_clip_norm, grad_scaler,
            lr_scheduler=lr_scheduler, use_amp=cfg.policy.use_amp,
        )

        step += 1
        train_tracker.step()

        if cfg.log_freq > 0 and step % cfg.log_freq == 0:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_logger.log_dict({**train_tracker.to_dict(), **(output_dict or {})}, step)
            train_tracker.reset_averages()

        # Regular checkpoint (for recovery / intermediate inspection)
        if cfg.save_checkpoint and step % cfg.save_freq == 0:
            logging.info(f"Checkpoint at step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        # Validation + early stopping
        if step % VAL_FREQ == 0:
            val_loss = _compute_val_loss(policy, val_iter, device, VAL_BATCHES)
            improved = val_loss < best_val_loss - 1e-5
            if improved:
                best_val_loss = val_loss
                es_counter = 0
                # Save best checkpoint (overwrites `last`)
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)
            else:
                es_counter += 1

            logging.info(
                f"[val] step={step}  val_loss={val_loss:.4f}  best={best_val_loss:.4f}"
                f"  patience={es_counter}/{ES_PATIENCE}"
                + ("  ✓ saved" if improved else "")
            )

            if es_counter >= ES_PATIENCE:
                logging.info(
                    f"Early stopping at step {step}  "
                    f"(best val_loss={best_val_loss:.4f}, stopped improving for {ES_PATIENCE} checks)"
                )
                break

    logging.info("End of training")


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
