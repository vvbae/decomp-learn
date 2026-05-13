"""
Convert ManiSkill state trajectory HDF5 to LeRobot dataset format.

Our HDF5 has traj_N/obs (T+1, 43) and traj_N/actions (T, 8) as flat arrays,
which is incompatible with the ManiSkill official converter (which expects nested obs groups).

Usage (full trajectories):
    uv run baseline/diffusion/v2/convert.py \
        --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5 \
        --output-dir datasets/peg_insertion_996 \
        --num-demos 996

Usage (contact portion only):
    uv run baseline/diffusion/v2/convert.py \
        --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5 \
        --output-dir datasets/peg_insertion_996_contact \
        --num-demos 996 \
        --contact-split
"""

import argparse
import os
import shutil

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

OBS_DIM = 43
ACTION_DIM = 8
TASK_NAME = "PegInsertionSide-v1"

TRAJ_PATH = os.path.expanduser(
    "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/"
    "trajectory.state.pd_joint_delta_pos.physx_cpu.h5"
)

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (OBS_DIM,),
        "names": [f"obs_{i}" for i in range(OBS_DIM)],
    },
    # DiffusionPolicy.validate_features() requires env_state_feature (FeatureType.ENV).
    # observation.state alone maps to FeatureType.STATE which doesn't satisfy the check.
    # Dummy required by DiffusionPolicy.validate_features() (needs FeatureType.ENV).
    # Shape (1,) triggers a lerobot scalar-HF-feature path that breaks add_frame, so use (2,).
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["env_state_0", "env_state_1"],
    },
    "action": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": [f"action_{i}" for i in range(ACTION_DIM)],
    },
}


# --- obs layout (43-dim) ---
# [0:9]   joint qpos
# [9:18]  joint qvel
# [18:25] tcp_pose   (x,y,z, qw,qx,qy,qz)
# [25:32] peg_pose   (x,y,z, qw,qx,qy,qz)
# [32:35] peg_half_size (length, radius, radius)
# [35:42] box_hole_pose (x,y,z, qw,qx,qy,qz)
# [42:43] box_hole_radius

def _quat_to_rotmat(q):
    """[w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z,   2*x*y-2*w*z,   2*x*z+2*w*y],
        [  2*x*y+2*w*z, 1-2*x*x-2*z*z,   2*y*z-2*w*x],
        [  2*x*z-2*w*y,   2*y*z+2*w*x, 1-2*x*x-2*y*y],
    ])


def find_contact_step(obs: np.ndarray) -> int:
    """
    Return the first step where the peg tip crosses the box outer face.

    Contact onset = first t where peg_head x-coord in box_hole frame >= -peg_length * 1.05.
    Falls back to 0 if no such step found (peg already at box at episode start).
    """
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


def convert(traj_path: str, output_dir: str, num_demos: int, fps: int, repo_id: str,
            contact_split: bool = False, start_demo: int = 0):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=FEATURES,
        root=output_dir,
        use_videos=False,
    )

    skipped = 0
    with h5py.File(traj_path, "r") as f:
        all_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        keys = all_keys[start_demo:start_demo + num_demos]
        for ep_idx, k in enumerate(keys):
            obs     = f[k]["obs"][:].astype(np.float32)      # (T+1, 43)
            actions = f[k]["actions"][:].astype(np.float32)  # (T, 8)

            start = find_contact_step(obs) if contact_split else 0

            # need at least 2 frames to form a valid episode
            if len(actions) - start < 2:
                skipped += 1
                continue

            for t in range(start, len(actions)):
                dataset.add_frame(
                    {
                        "observation.state": obs[t],
                        "observation.environment_state": np.zeros(2, dtype=np.float32),
                        "action": actions[t],
                    },
                    task=TASK_NAME,
                )
            dataset.save_episode()

            if (ep_idx + 1) % 100 == 0:
                print(f"  converted {ep_idx + 1}/{len(keys)} episodes")

    n_written = len(keys) - skipped
    mode = "contact-only" if contact_split else "full"
    print(f"Done ({mode}) — {n_written} episodes written to {output_dir}"
          + (f"  ({skipped} skipped)" if skipped else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-path", default=TRAJ_PATH)
    parser.add_argument("--output-dir", default="datasets/peg_insertion_996")
    parser.add_argument("--num-demos", type=int, default=996)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--repo-id", default="local/peg-insertion-side")
    parser.add_argument("--contact-split", action="store_true",
                        help="Only keep frames from contact onset onward.")
    parser.add_argument("--start-demo", type=int, default=0,
                        help="Index of first demo to include (0-based).")
    args = parser.parse_args()
    convert(args.traj_path, args.output_dir, args.num_demos, args.fps, args.repo_id,
            contact_split=args.contact_split, start_demo=args.start_demo)
