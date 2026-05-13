"""
Convert ManiSkill HDF5 trajectory (flat state obs) to LeRobot v2.1 format.
Works for both pd_joint_pos and pd_joint_delta_pos control modes.

Usage:
    uv run python convert_h5_to_lerobot.py \
        --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 \
        --output-dir datasets/peg_insertion_delta_996 \
        [--num-demos 996] [--fps 20]
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


FPS = 20


def convert(traj_path: str, output_dir: str, num_demos: int = None, fps: int = FPS):
    traj_path = Path(traj_path).expanduser()
    output_dir = Path(output_dir)

    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(traj_path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        if num_demos is not None:
            keys = keys[:num_demos]

        global_idx = 0
        episode_records = []
        stats_records = []
        obs_dim = action_dim = None
        all_rows = []

        for ep_idx, key in enumerate(tqdm(keys, desc="Converting")):
            obs = f[key]["obs"][:].astype(np.float32)       # (T+1, obs_dim)
            actions = f[key]["actions"][:].astype(np.float32)  # (T, action_dim)

            T = len(actions)
            obs = obs[:T]  # align: drop final obs

            if obs_dim is None:
                obs_dim = obs.shape[1]
                action_dim = actions.shape[1]

            for t in range(T):
                all_rows.append({
                    "observation.state": obs[t],
                    "observation.environment_state": obs[t],
                    "action": actions[t],
                    "timestamp": float(t) / fps,
                    "frame_index": t,
                    "episode_index": ep_idx,
                    "index": global_idx + t,
                    "task_index": 0,
                })

            episode_records.append({
                "episode_index": ep_idx,
                "tasks": ["PegInsertionSide-v1"],
                "length": T,
            })

            def col_stats(arr):
                return {
                    "min": arr.min(0).tolist(),
                    "max": arr.max(0).tolist(),
                    "mean": arr.mean(0).tolist(),
                    "std": arr.std(0).tolist(),
                    "count": [T],
                }

            ts = np.arange(T, dtype=np.float32) / fps
            fi = np.arange(T, dtype=np.int64)

            stats_records.append({"episode_index": ep_idx, "stats": {
                "observation.state": col_stats(obs),
                "observation.environment_state": col_stats(obs),
                "action": col_stats(actions),
                "timestamp": col_stats(ts.reshape(-1, 1)),
                "frame_index": col_stats(fi.reshape(-1, 1)),
                "episode_index": {"min": [ep_idx], "max": [ep_idx], "mean": [float(ep_idx)], "std": [0.0], "count": [T]},
                "index": col_stats((fi + global_idx).reshape(-1, 1)),
                "task_index": {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [T]},
            }})

            global_idx += T

        # Write all episodes as a single parquet file for fast random access
        print("Writing single parquet file...")
        df = pd.DataFrame(all_rows)
        df.to_parquet(data_dir / "data.parquet", index=False)

    n_episodes = len(keys)
    obs_names = [f"obs_{i}" for i in range(obs_dim)]
    act_names = [f"action_{i}" for i in range(action_dim)]

    info = {
        "codebase_version": "v2.1",
        "robot_type": None,
        "total_episodes": n_episodes,
        "total_frames": global_idx,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/data.parquet",
        "video_path": None,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [obs_dim], "names": obs_names},
            "observation.environment_state": {"dtype": "float32", "shape": [obs_dim], "names": obs_names},
            "action": {"dtype": "float32", "shape": [action_dim], "names": act_names},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4))

    (meta_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in episode_records) + "\n"
    )
    (meta_dir / "episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(r) for r in stats_records) + "\n"
    )
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "PegInsertionSide-v1"}) + "\n"
    )

    print(f"Done. {n_episodes} episodes, {global_idx} frames → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-demos", type=int, default=None)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    convert(args.traj_path, args.output_dir, args.num_demos, args.fps)
