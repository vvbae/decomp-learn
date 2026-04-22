"""
Train DiffusionPolicy on 996 demos using lerobot's exact default config.

Key differences from train_lerobot.py:
  - down_dims=(512, 1024, 2048)  [lerobot default, 252M params vs our 65M]
  - DDPM 100 steps               [lerobot default, vs our DDIM 10]
  - diffusers cosine+warmup LR   [tied to total steps, decays to 0]
  - no early stopping            [run full schedule]

Usage:
    python train_lerobot_default.py --num-steps 200000 --exp-name lerobot_default_996
"""

import argparse
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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
    def __init__(self, traj_path, num_demos):
        all_obs, all_actions = [], []
        with h5py.File(traj_path, "r") as f:
            keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[:num_demos]
            for k in keys:
                all_obs.append(f[k]["obs"][:].astype(np.float32))
                all_actions.append(f[k]["actions"][:].astype(np.float32))

        obs_cat = np.concatenate(all_obs)
        act_cat = np.concatenate(all_actions)
        self.stats = {
            "observation.state": {
                "min": obs_cat.min(0), "max": obs_cat.max(0),
                "mean": obs_cat.mean(0), "std": obs_cat.std(0) + 1e-8,
            },
            "action": {
                "min": act_cat.min(0), "max": act_cat.max(0),
                "mean": act_cat.mean(0), "std": act_cat.std(0) + 1e-8,
            },
        }

        self.samples = []
        for obs, actions in zip(all_obs, all_actions):
            T = len(actions)
            for i in range(T - HORIZON + 1):
                obs_start = max(0, i - N_OBS_STEPS + 1)
                obs_window = obs[obs_start: i + 1]
                pad = N_OBS_STEPS - len(obs_window)
                if pad > 0:
                    obs_window = np.concatenate([np.repeat(obs_window[:1], pad, axis=0), obs_window])
                self.samples.append((obs_window, actions[i: i + HORIZON]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_window, act_window = self.samples[idx]
        return {
            "observation.state": torch.from_numpy(obs_window),
            "action": torch.from_numpy(act_window),
            "action_is_pad": torch.zeros(HORIZON, dtype=torch.bool),
        }


def collate_fn(batch):
    obs = torch.stack([b["observation.state"] for b in batch])
    return {
        "observation.state": obs,
        "observation.environment_state": obs,
        "action": torch.stack([b["action"] for b in batch]),
        "action_is_pad": torch.stack([b["action_is_pad"] for b in batch]),
    }


def train(num_demos, exp_name, ckpt_dir, num_steps, batch_size, lr):
    os.makedirs(ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training '{exp_name}' | demos={num_demos} | steps={num_steps} | device={device}")

    dataset = PegDataset(TRAJ_PATH, num_demos)
    print(f"Dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_fn,
    )

    # lerobot default config — no overrides except required fields
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
        device=device,
        # Everything below is lerobot default — listed explicitly for clarity
        down_dims=(512, 1024, 2048),       # 252M params
        noise_scheduler_type="DDPM",       # 100-step DDPM
        num_train_timesteps=100,
        num_inference_steps=None,          # uses full 100 steps at eval
    )

    stats = dataset.stats
    full_stats = {
        "observation.state": stats["observation.state"],
        "observation.environment_state": stats["observation.state"],
        "action": stats["action"],
    }
    policy = DiffusionPolicy(cfg, dataset_stats=full_stats).to(device)
    policy.train()
    print(f"Model params: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        policy.diffusion.parameters(),
        lr=lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6,
    )

    # lerobot's exact LR schedule: diffusers cosine with 500 warmup steps
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_steps,
    )

    best_loss = float("inf")
    loss_history = []
    log_every = max(500, num_steps // 200)
    loader_iter = iter(loader)

    for step in range(1, num_steps + 1):
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
        lr_scheduler.step()

        loss_history.append(loss.item())

        if step % log_every == 0:
            avg = np.mean(loss_history[-log_every:])
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  step {step}/{num_steps}  loss={avg:.5f}  lr={current_lr:.2e}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                "policy_state": policy.state_dict(),
                "stats": dataset.stats,
                "cfg": policy.config,
                "loss_history": loss_history,
                "step": step,
            }, os.path.join(ckpt_dir, f"{exp_name}.pth"))

    print(f"Done. Best loss={best_loss:.5f}")
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
    parser.add_argument("--exp-name", type=str, default="lerobot_default_996")
    parser.add_argument("--ckpt-dir", type=str, default="experiments/lerobot_default")
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
