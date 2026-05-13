"""
Fix the ManiSkill→LeRobot converter bug where observation.environment_state
is missing or has wrong shape. Copies observation.state → observation.environment_state
in all parquet files and updates the metadata.

Usage:
    uv run python fix_lerobot_dataset.py --dataset-dir datasets/peg_insertion_delta_996
"""

import argparse
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def fix_parquet_files(dataset_dir: Path) -> dict:
    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    obs_dim = None
    for pf in tqdm(parquet_files, desc="Fixing parquets"):
        df = pd.read_parquet(pf)
        if "observation.state" not in df.columns:
            raise ValueError(f"Missing observation.state in {pf}")

        obs_dim = len(df["observation.state"].iloc[0])
        df["observation.environment_state"] = df["observation.state"]
        df.to_parquet(pf, index=False)

    return {"obs_dim": obs_dim}


def fix_info_json(dataset_dir: Path, obs_dim: int):
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    obs_names = [f"obs_{i}" for i in range(obs_dim)]
    info["features"]["observation.environment_state"] = {
        "dtype": "float32",
        "shape": [obs_dim],
        "names": obs_names,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"Updated {info_path}")


def fix_episodes_stats(dataset_dir: Path):
    stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    lines = stats_path.read_text().splitlines()
    fixed_lines = []
    for line in lines:
        ep = json.loads(line)
        ep["stats"]["observation.environment_state"] = copy.deepcopy(
            ep["stats"]["observation.state"]
        )
        fixed_lines.append(json.dumps(ep))
    stats_path.write_text("\n".join(fixed_lines) + "\n")
    print(f"Updated {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    result = fix_parquet_files(dataset_dir)
    fix_info_json(dataset_dir, result["obs_dim"])
    fix_episodes_stats(dataset_dir)
    print("Done.")
