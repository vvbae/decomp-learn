"""
Visualize 996-demo policy: record videos + print success/failure trajectory analysis.

Usage:
    python visualize_policy.py --ckpt experiments/demos_996/policy.pth
"""

import argparse
import json
import os

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs

from eval_lerobot import load_policy

DEMO_JSON = os.path.expanduser(
    "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/"
    "trajectory.state.pd_joint_pos.physx_cpu.json"
)


def load_seeds(n):
    with open(DEMO_JSON) as f:
        eps = json.load(f)["episodes"]
    return [ep["episode_seed"] for ep in eps[:n]]


def run_episode(env, policy, seed, device, record_frames=False):
    obs_raw, _ = env.reset(seed=int(seed))
    policy.reset()
    done = False

    frames = []
    peg_positions = []  # peg_head_pos_at_hole each step
    steps = 0

    while not done:
        if record_frames:
            frame = env.render()
            if frame is not None:
                import torch as _torch
                if isinstance(frame, _torch.Tensor):
                    frame = frame.cpu().numpy()
                frames.append(np.array(frame).squeeze())

        obs_frame = np.array(obs_raw).reshape(-1).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_frame).unsqueeze(0).to(device)
        batch = {"observation.state": obs_tensor, "observation.environment_state": obs_tensor}

        with torch.no_grad():
            action = policy.select_action(batch)

        obs_raw, _, terminated, truncated, info = env.step(action.cpu().numpy().reshape(1, -1))
        done = terminated or truncated
        steps += 1

        pos = np.array(info["peg_head_pos_at_hole"]).reshape(-1)
        peg_positions.append(pos)

    success = bool(np.array(info.get("success", False)).reshape(-1)[0])
    return success, steps, np.array(peg_positions), frames


def save_video(frames, path, fps=20):
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps)
        print(f"  Video saved → {path}")
    except ImportError:
        print("  imageio not available, skipping video save")


def print_trajectory_analysis(results):
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    print(f"\n{'='*60}")
    print(f"TRAJECTORY ANALYSIS  ({len(successes)} success / {len(failures)} failure)")
    print(f"{'='*60}")

    def summarize(traj, label):
        # peg_head_pos_at_hole: (x, y, z) displacement from hole
        # distance from hole over time
        dists = np.linalg.norm(traj, axis=1)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        final_pos = traj[-1]
        final_dist = dists[-1]

        # Phase detection: how close did it ever get in xy (alignment) vs z (insertion)
        xy_dist = np.linalg.norm(traj[:, :2], axis=1)
        z_dist = np.abs(traj[:, 2])
        min_xy = np.min(xy_dist)
        final_z = traj[-1, 2]

        print(f"\n  [{label}]  steps={len(traj)}")
        print(f"    start pos:   ({traj[0,0]:+.3f}, {traj[0,1]:+.3f}, {traj[0,2]:+.3f})  dist={dists[0]:.3f}")
        print(f"    best pos:    ({traj[min_dist_idx,0]:+.3f}, {traj[min_dist_idx,1]:+.3f}, {traj[min_dist_idx,2]:+.3f})  dist={min_dist:.3f}  @step={min_dist_idx}")
        print(f"    final pos:   ({final_pos[0]:+.3f}, {final_pos[1]:+.3f}, {final_pos[2]:+.3f})  dist={final_dist:.3f}")
        print(f"    min xy align: {min_xy:.3f}   final z: {final_z:+.3f}")

    if successes:
        print("\n--- SUCCESS episodes ---")
        for r in successes[:3]:
            summarize(r["peg_pos"], f"seed={r['seed']}")

    if failures:
        print("\n--- FAILURE episodes ---")
        for r in failures[:5]:
            summarize(r["peg_pos"], f"seed={r['seed']}")

    # Aggregate comparison
    if successes and failures:
        print(f"\n--- AGGREGATE ---")
        s_min_dists = [np.min(np.linalg.norm(r["peg_pos"], axis=1)) for r in successes]
        f_min_dists = [np.min(np.linalg.norm(r["peg_pos"], axis=1)) for r in failures]
        s_min_xy = [np.min(np.linalg.norm(r["peg_pos"][:, :2], axis=1)) for r in successes]
        f_min_xy = [np.min(np.linalg.norm(r["peg_pos"][:, :2], axis=1)) for r in failures]
        print(f"  Success avg min dist to hole: {np.mean(s_min_dists):.3f}")
        print(f"  Failure avg min dist to hole: {np.mean(f_min_dists):.3f}")
        print(f"  Success avg min xy alignment: {np.mean(s_min_xy):.3f}")
        print(f"  Failure avg min xy alignment: {np.mean(f_min_xy):.3f}")
        print()
        # Bucket failures by where they got stuck
        never_aligned = sum(1 for r in failures if np.min(np.linalg.norm(r["peg_pos"][:, :2], axis=1)) > 0.05)
        aligned_not_inserted = len(failures) - never_aligned
        print(f"  Failures stuck at alignment (xy > 0.05):  {never_aligned}/{len(failures)}")
        print(f"  Failures aligned but not inserted:        {aligned_not_inserted}/{len(failures)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="experiments/demos_996/policy.pth")
    parser.add_argument("--video-episodes", type=int, default=5)
    parser.add_argument("--analysis-episodes", type=int, default=30)
    parser.add_argument("--out-dir", default="visualizations")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.ckpt, device)

    seeds = load_seeds(max(args.video_episodes, args.analysis_episodes))

    # --- Part 1: Record videos ---
    print(f"\n=== Recording {args.video_episodes} episodes ===")
    video_env = gym.make(
        "PegInsertionSide-v1",
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        max_episode_steps=500,
    )
    for i, seed in enumerate(seeds[:args.video_episodes]):
        success, steps, peg_pos, frames = run_episode(video_env, policy, seed, device, record_frames=True)
        label = "SUCCESS" if success else "FAILURE"
        print(f"  ep{i+1} seed={seed}  {label}  steps={steps}  final_dist={np.linalg.norm(peg_pos[-1]):.3f}")
        if frames:
            save_video(frames, os.path.join(args.out_dir, f"ep{i+1}_seed{seed}_{label.lower()}.mp4"))
    video_env.close()

    # --- Part 2: Trajectory analysis ---
    print(f"\n=== Running {args.analysis_episodes} episodes for trajectory analysis ===")
    analysis_env = gym.make(
        "PegInsertionSide-v1",
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode=None,
        max_episode_steps=500,
    )
    results = []
    for i, seed in enumerate(seeds[:args.analysis_episodes]):
        success, steps, peg_pos, _ = run_episode(analysis_env, policy, seed, device)
        results.append({"seed": seed, "success": success, "steps": steps, "peg_pos": peg_pos})
        label = "OK" if success else "X "
        print(f"  [{label}] ep{i+1:2d} seed={seed}  steps={steps:3d}  min_dist={np.min(np.linalg.norm(peg_pos, axis=1)):.3f}")
    analysis_env.close()

    print_trajectory_analysis(results)


if __name__ == "__main__":
    main()
