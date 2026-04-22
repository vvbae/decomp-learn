"""
Evaluate a lerobot DiffusionPolicy checkpoint in ManiSkill PegInsertionSide-v1.

Usage:
    python eval_lerobot.py --ckpt checkpoints/e2e_lerobot_996.pth --num-episodes 100
"""

import argparse
import json
import os

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

DEMO_JSON = os.path.expanduser(
    "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/"
    "trajectory.state.pd_joint_pos.physx_cpu.json"
)
OBS_DIM = 43
ACTION_DIM = 8


def load_policy(
    ckpt_path: str,
    device: torch.device,
    noise_scheduler: str = None,
    num_inference_steps: int = None,
) -> DiffusionPolicy:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    stats = ckpt["stats"]

    # Load cfg from checkpoint so architecture always matches
    cfg = ckpt["cfg"]
    cfg.device = device.type if hasattr(device, "type") else str(device)
    # Allow CLI overrides for scheduler/steps (optional)
    if noise_scheduler is not None:
        cfg.noise_scheduler_type = noise_scheduler
    if num_inference_steps is not None:
        cfg.num_inference_steps = num_inference_steps

    full_stats = {
        "observation.state": stats["observation.state"],
        "observation.environment_state": stats["observation.state"],
        "action": stats["action"],
    }
    policy = DiffusionPolicy(cfg, dataset_stats=full_stats)
    policy.load_state_dict(ckpt["policy_state"])
    policy = policy.to(device)
    policy.eval()
    return policy


def load_seeds(num_episodes: int) -> list:
    with open(DEMO_JSON) as f:
        eps = json.load(f)["episodes"]
    return [ep["episode_seed"] for ep in eps[:num_episodes]]


def evaluate(
    ckpt_path: str,
    num_episodes: int = 100,
    seeds: list = None,
    noise_scheduler: str = "DDIM",
    num_inference_steps: int = 10,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(ckpt_path, device, noise_scheduler=noise_scheduler, num_inference_steps=num_inference_steps)
    print(f"Scheduler: {noise_scheduler}  inference_steps: {num_inference_steps}")

    env = gym.make(
        "PegInsertionSide-v1",
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode=None,
        max_episode_steps=500,
    )

    if seeds is None:
        seeds = list(range(num_episodes))

    successes = []
    for ep, seed in enumerate(seeds):
        obs_raw, _ = env.reset(seed=int(seed))
        policy.reset()
        done = False

        while not done:
            obs_frame = np.array(obs_raw).reshape(-1).astype(np.float32)  # (43,)
            obs_tensor = torch.from_numpy(obs_frame).unsqueeze(0).to(device)  # (1, 43)

            batch = {
                "observation.state": obs_tensor,
                "observation.environment_state": obs_tensor,
            }
            with torch.no_grad():
                action = policy.select_action(batch)  # (1, 8) or (8,)

            action_np = action.cpu().numpy().reshape(1, -1)
            obs_raw, _, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

        s = info.get("success", False)
        successes.append(bool(np.array(s).reshape(-1)[0]))

        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1}/{num_episodes}  running_success={np.mean(successes):.2%}")

    env.close()
    rate = np.mean(successes)
    print(f"\n{ckpt_path}  |  success rate: {rate:.2%}  ({int(rate*num_episodes)}/{num_episodes})")
    return rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--scheduler", type=str, default=None, choices=["DDIM", "DDPM"])
    parser.add_argument("--num-inference-steps", type=int, default=None)
    args = parser.parse_args()

    seeds = load_seeds(args.num_episodes)
    evaluate(
        args.ckpt,
        args.num_episodes,
        seeds=seeds,
        noise_scheduler=args.scheduler,
        num_inference_steps=args.num_inference_steps,
    )
