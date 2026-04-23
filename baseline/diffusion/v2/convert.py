"""
Convert ManiSkill state trajectory HDF5 to LeRobot dataset format.

Our HDF5 has traj_N/obs (T+1, 43) and traj_N/actions (T, 8) as flat arrays,
which is incompatible with the ManiSkill official converter (which expects nested obs groups).

Usage:
    uv run baseline/diffusion/v2/convert.py \
        --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5 \
        --output-dir datasets/peg_insertion_996 \
        --num-demos 996
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
    "trajectory.state.pd_joint_pos.physx_cpu.h5"
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


def convert(traj_path: str, output_dir: str, num_demos: int, fps: int, repo_id: str):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=FEATURES,
        root=output_dir,
        use_videos=False,
    )

    with h5py.File(traj_path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[:num_demos]
        for ep_idx, k in enumerate(keys):
            obs = f[k]["obs"][:].astype(np.float32)       # (T+1, 43)
            actions = f[k]["actions"][:].astype(np.float32)  # (T, 8)
            T = len(actions)
            for t in range(T):
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

    print(f"Done — {len(keys)} episodes written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-path", default=TRAJ_PATH)
    parser.add_argument("--output-dir", default="datasets/peg_insertion_996")
    parser.add_argument("--num-demos", type=int, default=996)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--repo-id", default="local/peg-insertion-side")
    args = parser.parse_args()
    convert(args.traj_path, args.output_dir, args.num_demos, args.fps, args.repo_id)
