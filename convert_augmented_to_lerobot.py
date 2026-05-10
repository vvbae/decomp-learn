"""
Convert augmented contact-phase H5 data to LeRobot v2.1 format.

Source: demos/augmented_contact/augmented_contact.h5
Output: datasets/augmented_contact_996/

Only the contact phase is kept for each episode (obs[t_c:], actions[t_c:]).
Episodes with no detected contact onset or fewer than 5 contact-phase steps
are skipped.

Usage:
    uv run python convert_augmented_to_lerobot.py \
        [--h5-path demos/augmented_contact/augmented_contact.h5] \
        [--output-dir datasets/augmented_contact_996] \
        [--fps 20] [--min-contact-steps 5]
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


FPS = 20
MIN_CONTACT_STEPS = 5


# ---------------------------------------------------------------------------
# Geometry helpers
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
    Return the first timestep t where the peg head has entered the hole
    (contact phase), or None if not found.

    obs: (T+1, 43) float32 array (full trajectory including final obs)
    """
    for t in range(len(obs)):
        peg_pos   = obs[t, 25:28]
        peg_quat  = obs[t, 28:32]   # [w, x, y, z]
        peg_half_len = obs[t, 32]
        peg_length   = peg_half_len * 2
        R_peg    = quat_to_rotmat(peg_quat)
        peg_head = peg_pos + R_peg @ np.array([peg_length, 0, 0])
        hole_pos  = obs[t, 35:38]
        hole_quat = obs[t, 38:42]
        R_hole    = quat_to_rotmat(hole_quat)
        head_in_hole = R_hole.T @ (peg_head - hole_pos)
        if head_in_hole[0] >= -peg_length * 1.05:
            return t
    return None


# ---------------------------------------------------------------------------
# Stats helper (matches existing dataset convention)
# ---------------------------------------------------------------------------

def col_stats(arr):
    """Compute per-column stats dict for a 2-D array (T, D)."""
    T = arr.shape[0]
    return {
        "min":   arr.min(0).tolist(),
        "max":   arr.max(0).tolist(),
        "mean":  arr.mean(0).tolist(),
        "std":   arr.std(0).tolist(),
        "count": [T],
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(h5_path: str, output_dir: str, fps: int = FPS,
            min_contact_steps: int = MIN_CONTACT_STEPS):
    h5_path    = Path(h5_path).expanduser()
    output_dir = Path(output_dir)

    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    episode_records = []
    stats_records   = []
    global_idx      = 0
    skipped         = 0

    with h5py.File(h5_path, "r") as f:
        # Sort numerically by trajectory index
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))

        ep_idx = 0  # LeRobot episode index (only counts kept episodes)
        for key in tqdm(keys, desc="Converting"):
            obs_full = f[key]["obs"][:].astype(np.float32)     # (T+1, 43)
            actions  = f[key]["actions"][:].astype(np.float32) # (T, 8)

            T = len(actions)
            obs_full_aligned = obs_full  # keep (T+1) for contact detection

            # Detect contact onset on the full obs sequence
            t_c = find_contact_onset(obs_full_aligned)
            if t_c is None:
                skipped += 1
                continue

            # Slice to contact phase; align obs to actions length
            obs_contact     = obs_full_aligned[t_c : t_c + T]   # up to T rows
            actions_contact = actions[t_c:]

            # After slicing, obs and actions may differ by 1 if t_c==0 gives
            # T+1 obs rows; re-align to action length
            L = len(actions_contact)
            obs_contact = obs_contact[:L]

            if L < min_contact_steps:
                skipped += 1
                continue

            # Build per-frame rows
            rows = []
            for t in range(L):
                rows.append({
                    "observation.state":             obs_contact[t],
                    "observation.environment_state": obs_contact[t],
                    "action":                        actions_contact[t],
                    "timestamp":                     np.float32(t / fps),
                    "frame_index":                   np.int64(t),
                    "episode_index":                 np.int64(ep_idx),
                    "index":                         np.int64(global_idx + t),
                    "task_index":                    np.int64(0),
                })

            # Write per-episode parquet (matches datasets/peg_insertion_996 format)
            ep_parquet = data_dir / f"episode_{ep_idx:06d}.parquet"
            df = pd.DataFrame(rows)
            df.to_parquet(ep_parquet, index=False)

            # Episode record
            episode_records.append({
                "episode_index": ep_idx,
                "tasks":         ["PegInsertionSide-v1"],
                "length":        L,
            })

            # Stats record
            ts = np.arange(L, dtype=np.float32) / fps
            fi = np.arange(L, dtype=np.int64)

            stats_records.append({"episode_index": ep_idx, "stats": {
                "observation.state":             col_stats(obs_contact),
                "observation.environment_state": col_stats(obs_contact),
                "action":                        col_stats(actions_contact),
                "timestamp":      col_stats(ts.reshape(-1, 1)),
                "frame_index":    col_stats(fi.reshape(-1, 1)),
                "episode_index":  {
                    "min": [ep_idx], "max": [ep_idx],
                    "mean": [float(ep_idx)], "std": [0.0], "count": [L],
                },
                "index": col_stats((fi + global_idx).reshape(-1, 1)),
                "task_index": {
                    "min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [L],
                },
            }})

            global_idx += L
            ep_idx += 1

    n_episodes   = ep_idx
    total_frames = global_idx

    print(f"\nKept {n_episodes} episodes, skipped {skipped} "
          f"(no contact or < {min_contact_steps} steps), "
          f"{total_frames} total frames.")

    # -----------------------------------------------------------------------
    # Write meta files
    # -----------------------------------------------------------------------
    obs_dim    = 43
    action_dim = 8
    obs_names  = [f"obs_{i}" for i in range(obs_dim)]
    act_names  = [f"action_{i}" for i in range(action_dim)]

    info = {
        "codebase_version": "v2.1",
        "robot_type": None,
        "total_episodes": n_episodes,
        "total_frames":   total_frames,
        "total_tasks":    1,
        "total_videos":   0,
        "total_chunks":   1,
        "chunks_size":    1000,
        "fps":            fps,
        "splits":         {"train": f"0:{n_episodes}"},
        "data_path":      "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path":     None,
        "features": {
            "observation.state": {
                "dtype": "float32", "shape": [obs_dim], "names": obs_names,
            },
            "observation.environment_state": {
                "dtype": "float32", "shape": [obs_dim], "names": obs_names,
            },
            "action": {
                "dtype": "float32", "shape": [action_dim], "names": act_names,
            },
            "timestamp":      {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":    {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index":  {"dtype": "int64",   "shape": [1], "names": None},
            "index":          {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":     {"dtype": "int64",   "shape": [1], "names": None},
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

    print(f"Done. Dataset written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert augmented contact H5 to LeRobot v2.1 format."
    )
    parser.add_argument(
        "--h5-path",
        default="demos/augmented_contact/augmented_contact.h5",
        help="Path to augmented_contact.h5",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/augmented_contact_996",
        help="Output directory for the LeRobot dataset",
    )
    parser.add_argument("--fps",               type=int, default=FPS)
    parser.add_argument("--min-contact-steps", type=int, default=MIN_CONTACT_STEPS)
    args = parser.parse_args()

    convert(args.h5_path, args.output_dir, args.fps, args.min_contact_steps)
