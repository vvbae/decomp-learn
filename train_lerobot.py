"""
Train lerobot DiffusionPolicy on PegInsertionSide-v1 demos.

Usage:
    python train_lerobot.py --num-demos 996 --exp-name e2e_lerobot_996
"""

import argparse
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


OBS_DIM = 43
ACTION_DIM = 8
N_OBS_STEPS = 2
HORIZON = 16
N_ACTION_STEPS = 8

TRAJ_PATH = os.path.expanduser(
    "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/"
    "trajectory.state.pd_joint_pos.physx_cpu.h5"
)


class PegDataset(Dataset):
    def __init__(self, traj_path: str, num_demos: int, stats: dict | None = None):
        all_obs, all_actions = [], []
        with h5py.File(traj_path, "r") as f:
            keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[:num_demos]
            for k in keys:
                all_obs.append(f[k]["obs"][:].astype(np.float32))       # (T+1, 43)
                all_actions.append(f[k]["actions"][:].astype(np.float32))  # (T, 8)

        if stats is None:
            obs_cat = np.concatenate(all_obs)
            act_cat = np.concatenate(all_actions)
            self.stats = {
                "observation.state": {
                    "min": obs_cat.min(0),
                    "max": obs_cat.max(0),
                    "mean": obs_cat.mean(0),
                    "std": obs_cat.std(0) + 1e-8,
                },
                "action": {
                    "min": act_cat.min(0),
                    "max": act_cat.max(0),
                    "mean": act_cat.mean(0),
                    "std": act_cat.std(0) + 1e-8,
                },
            }
        else:
            self.stats = stats

        self.samples = []
        for obs, actions in zip(all_obs, all_actions):
            T = len(actions)
            for i in range(T - HORIZON + 1):
                obs_start = max(0, i - N_OBS_STEPS + 1)
                obs_window = obs[obs_start : i + 1]          # (<=N_OBS_STEPS, 43)
                pad = N_OBS_STEPS - len(obs_window)
                if pad > 0:
                    obs_window = np.concatenate(
                        [np.repeat(obs_window[:1], pad, axis=0), obs_window]
                    )
                act_window = actions[i : i + HORIZON]         # (HORIZON, 8)
                self.samples.append((obs_window, act_window))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_window, act_window = self.samples[idx]
        return {
            "observation.state": torch.from_numpy(obs_window),           # (N_OBS_STEPS, 43)
            "action": torch.from_numpy(act_window),                       # (HORIZON, 8)
            "action_is_pad": torch.zeros(HORIZON, dtype=torch.bool),     # no padding
        }


def make_policy(stats: dict, device: str) -> DiffusionPolicy:
    cfg = DiffusionConfig(
        n_obs_steps=N_OBS_STEPS,
        horizon=HORIZON,
        n_action_steps=N_ACTION_STEPS,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(OBS_DIM,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(OBS_DIM,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        },
        normalization_mapping={
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        push_to_hub=False,
        down_dims=(256, 512, 1024),
        noise_scheduler_type="DDIM",
        num_inference_steps=10,
        device=device,
    )
    # Build stats with both obs keys (same data: env state = robot state for PegInsertion)
    full_stats = {
        "observation.state": stats["observation.state"],
        "observation.environment_state": stats["observation.state"],
        "action": stats["action"],
    }
    policy = DiffusionPolicy(cfg, dataset_stats=full_stats)
    return policy


def collate_fn(batch):
    obs = torch.stack([b["observation.state"] for b in batch])         # (B, N_OBS_STEPS, 43)
    act = torch.stack([b["action"] for b in batch])                    # (B, HORIZON, 8)
    pad = torch.stack([b["action_is_pad"] for b in batch])             # (B, HORIZON)
    return {
        "observation.state": obs,
        "observation.environment_state": obs,  # same data
        "action": act,
        "action_is_pad": pad,
    }


def train(num_demos: int, exp_name: str, ckpt_dir: str, num_steps: int, batch_size: int, lr: float):
    os.makedirs(ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training '{exp_name}' | demos={num_demos} | device={device}")

    dataset = PegDataset(TRAJ_PATH, num_demos)
    print(f"Dataset: {len(dataset)} samples from {num_demos} demos")

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    policy = make_policy(dataset.stats, device)
    policy = policy.to(device)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.diffusion.parameters(),
        lr=lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-6,
    )
    # cosine LR warmup then decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    best_loss = float("inf")
    loss_history = []
    step = 0
    log_every = max(1, num_steps // 50)
    loader_iter = iter(loader)

    while step < num_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        loss, _ = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.diffusion.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        step += 1

        loss_history.append(loss.item())

        if step % log_every == 0:
            avg = np.mean(loss_history[-log_every:])
            print(f"  step {step}/{num_steps}  loss={avg:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            ckpt = {
                "policy_state": policy.state_dict(),
                "stats": dataset.stats,
                "cfg": policy.config,
                "loss_history": loss_history,
                "step": step,
            }
            torch.save(ckpt, os.path.join(ckpt_dir, f"{exp_name}.pth"))

    print(f"Done. Best loss={best_loss:.5f} | saved to {ckpt_dir}/{exp_name}.pth")
    _plot_loss(loss_history, exp_name, ckpt_dir)


def _plot_loss(loss_history, exp_name, ckpt_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = list(range(1, len(loss_history) + 1))
    w = max(1, len(loss_history) // 200)
    smooth = np.convolve(loss_history, np.ones(w) / w, mode="same")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, loss_history, linewidth=0.6, color="#E91E63", alpha=0.4, label="raw")
    ax.plot(steps, smooth, linewidth=2, color="#880E4F", label="smoothed")
    ax.set_xlabel("Gradient Step")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"Training Loss — {exp_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(ckpt_dir, f"{exp_name}_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Loss curve → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-demos", type=int, default=996)
    parser.add_argument("--exp-name", type=str, default="e2e_lerobot_996")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--num-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(
        num_demos=args.num_demos,
        exp_name=args.exp_name,
        ckpt_dir=args.ckpt_dir,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
