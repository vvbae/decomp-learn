"""
Visualize failed contact-policy episodes to diagnose failure mode.

For each episode records peg-head position in hole frame at every step:
  x  = insertion depth  (negative = outside hole, 0 = flush, positive = inserted)
  y,z = lateral offset from hole centre

Two plots per run:
  1. x(t) for successes vs failures — do failures approach the hole at all?
  2. y-z scatter of final peg positions — are failures clustered or scattered?

Usage:
    uv run decomp/v1/viz_failures.py \
        --ckpt outputs/train/decomp_v1_delta_curve796/checkpoints/020000/pretrained_model \
        --num-episodes 50 \
        --start-episode 796 \
        --out-dir visualizations/failures
"""

import argparse
import json
import os
import sys

import gymnasium as gym
import h5py
import matplotlib.pyplot as plt
import mani_skill.envs
import numpy as np
import torch

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
# Geometry helpers (same as eval.py)
# --------------------------------------------------------------------------- #

def _quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z,   2*x*y-2*w*z,   2*x*z+2*w*y],
        [  2*x*y+2*w*z, 1-2*x*x-2*z*z,   2*y*z-2*w*x],
        [  2*x*z-2*w*y,   2*y*z+2*w*x, 1-2*x*x-2*y*y],
    ])


def _peg_head_in_hole_frame(obs: np.ndarray):
    """Return (x, y, z) of peg tip in hole frame from a single 43-dim obs."""
    peg_len  = float(obs[32])
    R_peg    = _quat_to_rotmat(obs[28:32])
    R_hole   = _quat_to_rotmat(obs[38:42])
    peg_head = obs[25:28] + R_peg @ np.array([peg_len, 0., 0.])
    return R_hole.T @ (peg_head - obs[35:38])   # (3,)


def _find_contact_step(obs_seq: np.ndarray) -> int:
    peg_len = float(obs_seq[0, 32])
    threshold = -peg_len * 1.05
    for t in range(len(obs_seq)):
        if _peg_head_in_hole_frame(obs_seq[t])[0] >= threshold:
            return t
    return 0


def _restore_contact_state(env, h5_file, traj_key):
    obs_h5 = h5_file[traj_key]["obs"][:].astype(np.float32)
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
    return obs


# --------------------------------------------------------------------------- #
# Episode runner — records peg trajectory in hole frame
# --------------------------------------------------------------------------- #

def run_and_record(env, policy, obs, device, max_steps=200):
    policy.reset()
    trajectory = [_peg_head_in_hole_frame(obs)]   # list of (3,) arrays
    done = False
    steps = 0
    info = {}
    while not done and steps < max_steps:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        batch = {
            "observation.state": obs_t,
            "observation.environment_state": torch.zeros(1, 2, device=device),
        }
        with torch.no_grad():
            action = policy.select_action(batch)
        obs_raw, _, terminated, truncated, info = env.step(action.cpu().numpy().reshape(1, -1))
        obs = np.array(obs_raw).reshape(-1).astype(np.float32)
        trajectory.append(_peg_head_in_hole_frame(obs))
        done = bool(terminated) or bool(truncated)
        steps += 1
    success = bool(np.array(info.get("success", False)).reshape(-1)[0])
    return success, np.array(trajectory)   # (T+1, 3)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_results(success_trajs, fail_trajs, peg_len, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    hole_radius = 0.015   # approx; not critical for the relative comparison

    # ------------------------------------------------------------------ #
    # Plot 1: x(t) — insertion depth over time
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    for traj in fail_trajs[:20]:
        ax.plot(traj[:, 0], color="tab:red", alpha=0.35, linewidth=1)
    for traj in success_trajs[:10]:
        ax.plot(traj[:, 0], color="tab:green", alpha=0.5, linewidth=1)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="hole entrance (x=0)")
    ax.axhline(-peg_len * 1.05, color="gray", linewidth=0.8, linestyle=":", label="contact threshold")
    ax.set_xlabel("policy step")
    ax.set_ylabel("peg tip x in hole frame (m)")
    ax.set_title(f"Insertion depth over time  (red=fail n={len(fail_trajs)}, green=success n={len(success_trajs)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "insertion_depth.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Plot 2: final peg position — hole-frame y vs z
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 6))
    if fail_trajs:
        yz_fail = np.array([t[-1, 1:] for t in fail_trajs])
        ax.scatter(yz_fail[:, 0], yz_fail[:, 1], c="tab:red", alpha=0.6, label=f"fail (n={len(fail_trajs)})", s=30)
    if success_trajs:
        yz_ok   = np.array([t[-1, 1:] for t in success_trajs])
        ax.scatter(yz_ok[:, 0], yz_ok[:, 1], c="tab:green", alpha=0.6, label=f"success (n={len(success_trajs)})", s=30)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(hole_radius * np.cos(theta), hole_radius * np.sin(theta),
            "k--", linewidth=1, label="~hole radius")
    ax.set_aspect("equal")
    ax.set_xlabel("y in hole frame (m)")
    ax.set_ylabel("z in hole frame (m)")
    ax.set_title("Final peg-tip lateral position (hole frame)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "final_lateral.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Plot 3: lateral error over time
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    for traj in fail_trajs[:20]:
        lateral = np.sqrt(traj[:, 1]**2 + traj[:, 2]**2)
        ax.plot(lateral, color="tab:red", alpha=0.35, linewidth=1)
    for traj in success_trajs[:10]:
        lateral = np.sqrt(traj[:, 1]**2 + traj[:, 2]**2)
        ax.plot(lateral, color="tab:green", alpha=0.5, linewidth=1)
    ax.axhline(hole_radius, color="black", linewidth=0.8, linestyle="--", label="~hole radius")
    ax.set_xlabel("policy step")
    ax.set_ylabel("lateral error (m)")
    ax.set_title("Lateral distance from hole centre over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lateral_error.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out_dir}/")
    print("  insertion_depth.png  — x(t) per episode")
    print("  final_lateral.png    — final y/z scatter")
    print("  lateral_error.png    — lateral error(t) per episode")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--start-episode", type=int, default=796)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--out-dir", default="visualizations/failures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.ckpt, device)

    env = gym.make(
        "PegInsertionSide-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode=None,
        max_episode_steps=args.max_steps,
    )
    with open(TRAJ_JSON) as jf:
        all_seeds = [ep["episode_seed"] for ep in json.load(jf)["episodes"]]
    seeds = all_seeds[args.start_episode: args.start_episode + args.num_episodes]

    success_trajs, fail_trajs = [], []
    peg_len = None

    with h5py.File(H5, "r") as f:
        all_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        keys = all_keys[args.start_episode: args.start_episode + args.num_episodes]

        for i, k in enumerate(keys):
            env.reset(seed=int(seeds[i]))
            obs = _restore_contact_state(env, f, k)
            if peg_len is None:
                peg_len = float(obs[32])
            success, traj = run_and_record(env, policy, obs, device, args.max_steps)
            label = "OK" if success else "X "
            x_final = traj[-1, 0]
            lat_final = float(np.sqrt(traj[-1, 1]**2 + traj[-1, 2]**2))
            print(f"[{label}] ep{i+1:3d} ({k})  x_final={x_final:.3f}m  lateral={lat_final*1000:.1f}mm")
            if success:
                success_trajs.append(traj)
            else:
                fail_trajs.append(traj)

    env.close()
    print(f"\nSuccess: {len(success_trajs)}/{len(success_trajs)+len(fail_trajs)}")

    # Summary stats for failures
    if fail_trajs:
        x_finals  = [t[-1, 0] for t in fail_trajs]
        x_maxes   = [t[:, 0].max() for t in fail_trajs]
        lat_finals = [float(np.sqrt(t[-1, 1]**2 + t[-1, 2]**2)) for t in fail_trajs]
        print(f"\nFailure stats:")
        print(f"  x_final    mean={np.mean(x_finals):.3f}m  std={np.std(x_finals):.3f}m  (0=flush, +=inside)")
        print(f"  x_max      mean={np.mean(x_maxes):.3f}m   (deepest point reached during episode)")
        print(f"  lateral    mean={np.mean(lat_finals)*1000:.1f}mm  std={np.std(lat_finals)*1000:.1f}mm")
        print(f"  peg_len    {peg_len:.3f}m  hole_x_threshold={-peg_len:.3f}m")

    plot_results(success_trajs, fail_trajs, peg_len, args.out_dir)


if __name__ == "__main__":
    main()
