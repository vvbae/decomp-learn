"""
Contact-onset data augmentation for PegInsertionSide-v1.

Algorithm:
  For each demo episode:
    1. Find contact onset frame t_c (first frame where peg head enters hole region)
    2. For each perturbation:
       a. Reset env with episode seed, step through actions[0:t_c] to reach t_c
       b. Overwrite peg actor state with a small y/z world-frame displacement
       c. Resume stepping with actions[t_c:] and collect (obs, action) pairs
       d. Keep trajectory if episode ends with success

Output H5 structure mirrors source H5:
  traj_N/
    obs:     (T+1, 43) float32
    actions: (T,   8)  float32

H5 file structure (verified):
  actors/peg shape (T+1, 13): [0:3]=pos(xyz), [3:7]=quat(wxyz), [7:13]=vel
  actors/box_with_hole shape (T+1, 13): same layout
  actors/table-workspace shape (T+1, 13): same layout
  articulations/panda_wristcam shape (T+1, 31)

Obs layout (43-dim):
  [0:9]   joint_qpos
  [9:18]  joint_qvel
  [18:25] tcp_pose  (x,y,z, qw,qx,qy,qz)
  [25:32] peg_pose  (x,y,z, qw,qx,qy,qz)
  [32:35] peg_half_size (half_length, half_radius, half_radius)
  [35:42] box_hole_pose (x,y,z, qw,qx,qy,qz)
  [42]    box_hole_radius

Contact onset formula (verified correct):
  peg_length   = obs[t, 32] * 2          # full length = 2 * half_length
  R_peg        = quat_to_rotmat(obs[t, 28:32])
  peg_head_pos = obs[t, 25:28] + R_peg @ [peg_length, 0, 0]
  R_hole       = quat_to_rotmat(obs[t, 38:42])
  head_in_hole = R_hole.T @ (peg_head_pos - obs[t, 35:38])
  contact      = head_in_hole[0] >= -peg_length * 1.05
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Geometry helpers (no external imports beyond numpy)
# ---------------------------------------------------------------------------

def quat_to_rotmat(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def find_contact_onset(obs):
    """
    Return first timestep t where the peg head has entered the hole face plane.

    Uses:
        peg_length   = obs[t, 32] * 2
        peg_head_pos = obs[t, 25:28] + R_peg @ [peg_length, 0, 0]
        head_in_hole = R_hole.T @ (peg_head_pos - obs[t, 35:38])
        contact      = head_in_hole[0] >= -peg_length * 1.05

    Returns None if no contact is detected in the full trajectory.
    """
    for t in range(len(obs)):
        peg_pos   = obs[t, 25:28]
        peg_quat  = obs[t, 28:32]          # [w, x, y, z]
        peg_len   = float(obs[t, 32]) * 2  # full length
        hole_pos  = obs[t, 35:38]
        hole_quat = obs[t, 38:42]          # [w, x, y, z]

        R_peg  = quat_to_rotmat(peg_quat)
        R_hole = quat_to_rotmat(hole_quat)

        peg_head    = peg_pos + R_peg @ np.array([peg_len, 0.0, 0.0])
        head_in_hole = R_hole.T @ (peg_head - hole_pos)

        if head_in_hole[0] >= -peg_len * 1.05:
            return t

    return None


# ---------------------------------------------------------------------------
# Main augmentation logic
# ---------------------------------------------------------------------------

PERTURBATION_MAGNITUDES_MM = [3, 7, 15]   # millimetres
PERTURBATION_MAGNITUDES_M  = [m / 1000.0 for m in PERTURBATION_MAGNITUDES_MM]

SOURCE_H5   = os.path.expanduser(
    "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/"
    "trajectory.state.pd_joint_delta_pos.physx_cpu.h5"
)
SOURCE_JSON = os.path.expanduser(
    "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/"
    "trajectory.state.pd_joint_delta_pos.physx_cpu.json"
)
OUTPUT_H5 = os.path.join(
    os.path.dirname(__file__), "demos", "augmented_contact", "augmented_contact.h5"
)


def parse_args():
    p = argparse.ArgumentParser(description="Contact-onset data augmentation")
    p.add_argument("--dry-run", action="store_true",
                   help="Process only the first 5 episodes to verify correctness")
    p.add_argument("--n-perturb", type=int, default=10,
                   help="Number of perturbations per episode (default 10)")
    p.add_argument("--max-episodes", type=int, default=996,
                   help="Maximum number of source episodes to process (default 996)")
    p.add_argument("--output", type=str, default=OUTPUT_H5,
                   help="Path to output H5 file")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for perturbation sampling (default 42)")
    return p.parse_args()


def make_env():
    import gymnasium as gym
    import mani_skill.envs  # noqa: registers PegInsertionSide-v1
    env = gym.make(
        "PegInsertionSide-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode=None,
        max_episode_steps=500,
    )
    return env


def inspect_h5_structure(h5_path):
    """Print env_states structure for the first episode so we can verify shapes."""
    print("\n" + "="*60)
    print("H5 FILE STRUCTURE INSPECTION")
    print("="*60)
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        print(f"Total episodes in source H5: {len(keys)}")
        ep = f[keys[0]]
        print(f"\nEpisode '{keys[0]}' top-level keys: {list(ep.keys())}")
        print(f"  obs shape:     {ep['obs'].shape}   dtype={ep['obs'].dtype}")
        print(f"  actions shape: {ep['actions'].shape}   dtype={ep['actions'].dtype}")

        es = ep["env_states"]
        print(f"\n  env_states/actors:")
        for k in es["actors"]:
            d = es["actors"][k]
            print(f"    {k}: shape={d.shape}, dtype={d.dtype}")
            print(f"      t=0 sample: {d[0]}")

        print(f"\n  env_states/articulations:")
        for k in es["articulations"]:
            d = es["articulations"][k]
            print(f"    {k}: shape={d.shape}, dtype={d.dtype}")
            print(f"      t=0 first 10 values: {d[0, :10]}")

        print(f"\nActor state layout (13 floats):")
        print(f"  [0:3]  = position (x, y, z)  <- perturbation target")
        print(f"  [3:7]  = quaternion (qw, qx, qy, qz)")
        print(f"  [7:10] = linear velocity (vx, vy, vz)")
        print(f"  [10:13]= angular velocity (wx, wy, wz)")
    print("="*60 + "\n")


def run_augmentation(args):
    from tqdm import tqdm

    rng = np.random.default_rng(args.seed)

    # Load episode metadata (seeds)
    with open(SOURCE_JSON) as f:
        episodes_meta = json.load(f)["episodes"]

    # Create output directory
    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)

    max_ep = 5 if args.dry_run else args.max_episodes

    # Summary counters
    # success_counts[mag_mm] = [n_success, n_total]
    success_counts = {m: [0, 0] for m in PERTURBATION_MAGNITUDES_MM}
    ep_lengths_success = []   # lengths of successful augmented trajectories
    total_augmented = 0
    skipped_no_contact = 0

    env = make_env()

    with h5py.File(SOURCE_H5, "r") as src_h5, \
         h5py.File(args.output, "w") as dst_h5:

        # Sort keys numerically: traj_0, traj_1, ..., traj_995
        all_keys = sorted(src_h5.keys(), key=lambda k: int(k.split("_")[1]))
        keys_to_process = all_keys[:max_ep]

        pbar = tqdm(keys_to_process, desc="Episodes", unit="ep")

        for ep_idx, key in enumerate(pbar):
            ep = src_h5[key]
            obs_h5    = ep["obs"][:].astype(np.float32)     # (T+1, 43)
            actions   = ep["actions"][:].astype(np.float32) # (T, 8)
            es        = ep["env_states"]

            # Find episode seed from JSON (episodes_meta is in order of episode_id)
            ep_id = int(key.split("_")[1])
            # Build a map the first time or look up by episode_id
            ep_meta = episodes_meta[ep_id]
            seed    = int(ep_meta["episode_seed"])

            # Detect contact onset
            t_c = find_contact_onset(obs_h5)
            if t_c is None:
                skipped_no_contact += 1
                pbar.set_postfix({"aug_total": total_augmented, "no_contact": skipped_no_contact})
                continue

            n_remaining = len(actions) - t_c
            if n_remaining <= 0:
                # Contact at or after last action - nothing to replay
                skipped_no_contact += 1
                continue

            # Sample perturbations for this episode
            # Each perturbation: pick magnitude from {3,7,15}mm, random direction in y-z plane
            for _ in range(args.n_perturb):
                mag_idx = rng.integers(0, len(PERTURBATION_MAGNITUDES_M))
                mag_m   = PERTURBATION_MAGNITUDES_M[mag_idx]
                mag_mm  = PERTURBATION_MAGNITUDES_MM[mag_idx]
                angle   = rng.uniform(0.0, 2 * np.pi)
                delta_y = mag_m * np.cos(angle)
                delta_z = mag_m * np.sin(angle)

                success_counts[mag_mm][1] += 1

                # --- Phase 1: reset with seed and step to t_c ---
                obs_env, _ = env.reset(seed=seed)

                aug_obs_seq    = []   # will hold obs at [0, 1, ..., T_aug]
                aug_action_seq = []   # will hold actions [0, ..., T_aug-1]

                # Step from frame 0 to t_c (not collecting — these come from original demo)
                # We only collect obs AFTER the perturbation is applied (starting at t_c)
                for t in range(t_c):
                    obs_env, _, _, _, _ = env.step(actions[t])

                # --- Phase 2: apply perturbation ---
                peg_state = np.array(es["actors"]["peg"][t_c], dtype=np.float32)
                peg_state[1] += delta_y   # world-frame y
                peg_state[2] += delta_z   # world-frame z

                state_dict = {
                    "actors": {
                        "table-workspace": torch.tensor(
                            es["actors"]["table-workspace"][t_c:t_c+1], dtype=torch.float32),
                        "peg": torch.tensor(
                            peg_state.reshape(1, -1), dtype=torch.float32),
                        "box_with_hole": torch.tensor(
                            es["actors"]["box_with_hole"][t_c:t_c+1], dtype=torch.float32),
                    },
                    "articulations": {
                        "panda_wristcam": torch.tensor(
                            es["articulations"]["panda_wristcam"][t_c:t_c+1], dtype=torch.float32),
                    },
                }
                env.unwrapped.set_state_dict(state_dict)

                # First obs of augmented trajectory (after perturbation at t_c)
                first_obs = env.unwrapped.get_obs().cpu().numpy().reshape(-1).astype(np.float32)
                aug_obs_seq.append(first_obs)

                # --- Phase 3: replay remaining actions ---
                ep_success = False
                for t in range(t_c, len(actions)):
                    action = actions[t]
                    obs_env, _, terminated, truncated, info = env.step(action)
                    obs_np = obs_env.cpu().numpy().reshape(-1).astype(np.float32)
                    aug_obs_seq.append(obs_np)
                    aug_action_seq.append(action)

                    succ = info["success"]
                    if hasattr(succ, "item"):
                        succ = succ.item()
                    else:
                        succ = bool(np.array(succ).reshape(-1)[0])

                    if succ:
                        ep_success = True
                        break

                # --- Phase 4: save if success ---
                if ep_success:
                    success_counts[mag_mm][0] += 1
                    ep_lengths_success.append(len(aug_action_seq))
                    total_augmented += 1

                    grp = dst_h5.create_group(f"traj_{total_augmented - 1}")
                    obs_arr    = np.stack(aug_obs_seq,    axis=0).astype(np.float32)  # (T+1, 43)
                    action_arr = np.stack(aug_action_seq, axis=0).astype(np.float32)  # (T, 8)
                    grp.create_dataset("obs",     data=obs_arr,    compression="gzip", compression_opts=4)
                    grp.create_dataset("actions", data=action_arr, compression="gzip", compression_opts=4)
                    # Store metadata as attributes
                    grp.attrs["source_episode"]     = key
                    grp.attrs["episode_seed"]       = seed
                    grp.attrs["contact_onset_frame"] = t_c
                    grp.attrs["mag_mm"]             = mag_mm
                    grp.attrs["delta_y_m"]          = delta_y
                    grp.attrs["delta_z_m"]          = delta_z

            pbar.set_postfix({"aug_total": total_augmented, "no_contact": skipped_no_contact})

        # Write top-level statistics attribute
        stats = {
            "total_augmented_episodes": total_augmented,
            "skipped_no_contact": skipped_no_contact,
            "source_episodes_processed": len(keys_to_process),
        }
        for mag_mm in PERTURBATION_MAGNITUDES_MM:
            n_s, n_t = success_counts[mag_mm]
            stats[f"success_rate_{mag_mm}mm"] = float(n_s) / n_t if n_t > 0 else 0.0
            stats[f"n_success_{mag_mm}mm"]    = n_s
            stats[f"n_total_{mag_mm}mm"]      = n_t

        for k, v in stats.items():
            dst_h5.attrs[k] = v

    env.close()
    return stats, ep_lengths_success, success_counts


def print_summary(stats, ep_lengths_success, success_counts):
    print("\n" + "="*60)
    print("AUGMENTATION SUMMARY")
    print("="*60)
    print(f"Source episodes processed : {stats['source_episodes_processed']}")
    print(f"Skipped (no contact)      : {stats['skipped_no_contact']}")
    print(f"Total augmented episodes  : {stats['total_augmented_episodes']}")
    print()
    print("Per perturbation level:")
    for mag_mm in PERTURBATION_MAGNITUDES_MM:
        n_s = stats[f"n_success_{mag_mm}mm"]
        n_t = stats[f"n_total_{mag_mm}mm"]
        rate = stats[f"success_rate_{mag_mm}mm"]
        print(f"  {mag_mm:2d} mm : {n_s:5d} / {n_t:5d} succeeded  ({rate:.1%})")
    print()
    if ep_lengths_success:
        lengths = np.array(ep_lengths_success)
        print("Successful episode length statistics:")
        print(f"  mean : {lengths.mean():.1f}")
        print(f"  std  : {lengths.std():.1f}")
        print(f"  min  : {lengths.min()}")
        print(f"  max  : {lengths.max()}")
    print("="*60)


def main():
    args = parse_args()

    if args.dry_run:
        print("[DRY RUN] Processing first 5 episodes only")

    # Always inspect H5 structure first
    inspect_h5_structure(SOURCE_H5)

    print(f"Source H5      : {SOURCE_H5}")
    print(f"Source JSON    : {SOURCE_JSON}")
    print(f"Output H5      : {args.output}")
    print(f"Episodes       : {'5 (dry run)' if args.dry_run else args.max_episodes}")
    print(f"Perturb/episode: {args.n_perturb}")
    print(f"Magnitudes (mm): {PERTURBATION_MAGNITUDES_MM}")
    print(f"RNG seed       : {args.seed}")
    print()

    t0 = time.time()
    stats, ep_lengths_success, success_counts = run_augmentation(args)
    elapsed = time.time() - t0

    print_summary(stats, ep_lengths_success, success_counts)
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
