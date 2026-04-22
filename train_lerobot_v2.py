"""
Train DiffusionPolicy using LeRobot's official training loop and TrainPipelineConfig.
Converts ManiSkill HDF5 to LeRobot format automatically if the dataset root doesn't exist.

Usage:
    uv run train_lerobot_v2.py \
        --dataset.repo_id=local/peg-insertion-side-996 \
        --dataset.root=datasets/peg_insertion_996 \
        --policy.type=diffusion \
        --policy.down_dims="[256,512,1024]" \
        --policy.noise_scheduler_type=DDIM \
        --policy.num_inference_steps=10 \
        --batch_size=256 \
        --steps=200000 \
        --save_freq=50000 \
        --log_freq=500 \
        --output_dir=outputs/train/v2_996

Resume:
    uv run train_lerobot_v2.py --resume --config_path=outputs/train/v2_996/checkpoints/last/pretrained_model/train_config.json
"""

import logging
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat

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

TRAJ_PATH = os.environ.get(
    "TRAJ_PATH",
    os.path.expanduser(
        "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/"
        "trajectory.state.pd_joint_pos.physx_cpu.h5"
    ),
)

def _maybe_convert_dataset(cfg: TrainPipelineConfig) -> None:
    if cfg.dataset.root is None:
        return
    root = Path(cfg.dataset.root)
    if root.exists():
        return
    logging.info(f"Dataset not found at {root} — converting from HDF5 via ManiSkill converter...")
    subprocess.run(
        [
            sys.executable, "-m", "mani_skill.trajectory.convert_to_lerobot",
            "--traj-path", TRAJ_PATH,
            "--output-dir", str(root),
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

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"learnable params: {format_big_number(num_learnable)}")

    if hasattr(cfg.policy, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(dataset.episode_data_index, drop_n_last_frames=cfg.policy.drop_n_last_frames, shuffle=True)
        shuffle = False
    else:
        sampler, shuffle = None, True

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step)

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

        if cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps):
            logging.info(f"Checkpoint at step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

    logging.info("End of training")


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
