"""
Eval for decomp v1 contact policy.

Geometric phase: restored from the stored demo env_state at contact onset.
Contact phase: run the trained DiffusionPolicy until done.

Usage:
    uv run decomp/v1/eval.py \
        --ckpt outputs/train/decomp_v1_996/checkpoints/last/pretrained_model \
        --num-episodes 100
"""

import argparse
import os
import sys

import gymnasium as gym
import h5py
import mani_skill.envs
import numpy as np
import torch

# reuse v2 policy loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baseline", "diffusion", "v2"))
from eval import load_policy  # noqa: E402

H5 = os.path.expanduser(
    "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/"
    "trajectory.state.pd_joint_delta_pos.physx_cpu.h5"
)
TRAJ_JSON = os.path.expanduser(
    "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/"
    "trajectory.state.pd_joint_delta_pos.physx_cpu.json"
)

# --------------------------------------------------------------------------- #
# Contact onset detection (same logic as analyze_contact.py)
# --------------------------------------------------------------------------- #

def _quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z,   2*x*y-2*w*z,   2*x*z+2*w*y],
        [  2*x*y+2*w*z, 1-2*x*x-2*z*z,   2*y*z-2*w*x],
        [  2*x*z-2*w*y,   2*y*z+2*w*x, 1-2*x*x-2*y*y],
    ])


def _find_contact_step(obs: np.ndarray) -> int:
    peg_len = float(obs[0, 32])
    threshold = -peg_len * 1.05
    for t in range(len(obs)):
        R_peg  = _quat_to_rotmat(obs[t, 28:32])
        R_hole = _quat_to_rotmat(obs[t, 38:42])
        peg_head = obs[t, 25:28] + R_peg @ np.array([peg_len, 0., 0.])
        x_in_hole = float((R_hole.T @ (peg_head - obs[t, 35:38]))[0])
        if x_in_hole >= threshold:
            return t
    return 0


# --------------------------------------------------------------------------- #
# State restoration
# --------------------------------------------------------------------------- #

def restore_pre_contact_state(env, h5_file: h5py.File, traj_key: str) -> np.ndarray:
    """
    Find contact step in this demo, load env_state at that step into env,
    return the 43-dim obs.
    """
    obs_h5 = h5_file[traj_key]["obs"][:].astype(np.float32)  # (T+1, 43)
    contact_step = _find_contact_step(obs_h5)

    es = h5_file[traj_key]["env_states"]
    state_dict = {
        "actors": {
            "table-workspace": torch.tensor(es["actors"]["table-workspace"][contact_step:contact_step+1], dtype=torch.float32),
            "peg":             torch.tensor(es["actors"]["peg"][contact_step:contact_step+1],             dtype=torch.float32),
            "box_with_hole":   torch.tensor(es["actors"]["box_with_hole"][contact_step:contact_step+1],   dtype=torch.float32),
        },
        "articulations": {
            "panda_wristcam":  torch.tensor(es["articulations"]["panda_wristcam"][contact_step:contact_step+1], dtype=torch.float32),
        },
    }
    env.unwrapped.set_state_dict(state_dict)
    obs = env.unwrapped.get_obs().cpu().numpy().reshape(-1).astype(np.float32)
    return obs, contact_step


# --------------------------------------------------------------------------- #
# Episode runner
# --------------------------------------------------------------------------- #

def run_episode(env, policy, obs: np.ndarray, device, max_steps: int = 200) -> tuple[bool, int]:
    policy.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        batch = {"observation.state": obs_t, "observation.environment_state": torch.zeros(1, 2, device=device)}
        with torch.no_grad():
            action = policy.select_action(batch)
        obs_raw, _, terminated, truncated, info = env.step(action.cpu().numpy().reshape(1, -1))
        obs = np.array(obs_raw).reshape(-1).astype(np.float32)
        done = bool(terminated) or bool(truncated)
        steps += 1
    success = bool(np.array(info.get("success", False)).reshape(-1)[0])
    return success, steps


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to pretrained_model dir")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--start-episode", type=int, default=0,
                        help="Index of first demo to evaluate (0-based, for held-out test split)")
    parser.add_argument("--max-contact-steps", type=int, default=200,
                        help="max steps for contact policy per episode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.ckpt, device)

    env = gym.make(
        "PegInsertionSide-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode=None,
        max_episode_steps=args.max_contact_steps,
    )
    import json
    with open(TRAJ_JSON) as jf:
        all_seeds = [ep["episode_seed"] for ep in json.load(jf)["episodes"]]

    successes = []
    contact_steps_list = []

    with h5py.File(H5, "r") as f:
        all_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        keys = all_keys[args.start_episode: args.start_episode + args.num_episodes]
        episode_seeds = all_seeds[args.start_episode: args.start_episode + args.num_episodes]
        for i, k in enumerate(keys):
            # reset with original demo seed so peg/box geometry matches the stored state
            env.reset(seed=int(episode_seeds[i]))
            obs, contact_step = restore_pre_contact_state(env, f, k)
            success, steps = run_episode(env, policy, obs, device, args.max_contact_steps)
            successes.append(success)
            contact_steps_list.append(steps)
            label = "OK" if success else "X "
            print(f"[{label}] ep{i+1:3d} ({k})  contact_step={contact_step}  policy_steps={steps}")

    print(f"\n{'='*50}")
    print(f"Success rate: {sum(successes)}/{len(successes)} = {np.mean(successes)*100:.1f}%")
    print(f"Avg policy steps: {np.mean(contact_steps_list):.1f}")
    env.close()


if __name__ == "__main__":
    main()
