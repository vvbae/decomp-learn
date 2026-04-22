"""
Convert ManiSkill HDF5 trajectory file to LeRobot format using ManiSkill's official converter.

Usage:
    uv run convert_maniskill_to_lerobot.py \
        --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5 \
        --output-dir datasets/peg_insertion_996 \
        --task-name "PegInsertionSide-v1"
"""

import sys
from mani_skill.trajectory.convert_to_lerobot import Args, main
import tyro

if __name__ == "__main__":
    args = tyro.cli(Args)
    sys.exit(main(args))
