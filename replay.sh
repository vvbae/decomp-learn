#!/bin/bash
# Convert raw motionplanning demos to state+pd_joint_pos format for training.
GIT_PYTHON_REFRESH=quiet uv run python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
  --save-traj \
  --obs-mode state \
  --target-control-mode pd_joint_pos \
  --num-procs 4
