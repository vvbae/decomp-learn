"""
Evaluate a LeRobot v2 checkpoint (saved by train_lerobot_v2.py or lerobot.scripts.train).

Usage:
    uv run eval_lerobot_v2.py \
        --ckpt outputs/train/v2_996/checkpoints/last/pretrained_model \
        --num-episodes 100
"""

import argparse
import json
import os

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

DEMO_JSON = os.path.expanduser(
    "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/"
    "trajectory.state.pd_joint_pos.physx_cpu.json"
)


def load_policy(ckpt_path: str, device: torch.device) -> DiffusionPolicy:
    policy = DiffusionPolicy.from_pretrained(ckpt_path)
    policy = policy.to(device)
    policy.eval()
    return policy


def load_seeds(num_episodes: int) -> list:
    with open(DEMO_JSON) as f:
        eps = json.load(f)["episodes"]
    return [ep["episode_seed"] for ep in eps[:num_episodes]]


def evaluate(ckpt_path: str, num_episodes: int = 100, seeds: list = None) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(ckpt_path, device)
    print(f"Loaded policy from {ckpt_path}")

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
            obs_frame = np.array(obs_raw).reshape(-1).astype(np.float32)
            obs_tensor = torch.from_numpy(obs_frame).unsqueeze(0).to(device)
            batch = {
                "observation.state": obs_tensor,
                "observation.environment_state": torch.zeros(1, 2, device=device),
            }
            with torch.no_grad():
                action = policy.select_action(batch)

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
    parser.add_argument("--ckpt", required=True, help="Path to pretrained_model/ directory")
    parser.add_argument("--num-episodes", type=int, default=100)
    args = parser.parse_args()

    seeds = load_seeds(args.num_episodes)
    evaluate(args.ckpt, args.num_episodes, seeds=seeds)
