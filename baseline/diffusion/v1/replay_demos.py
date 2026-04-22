"""
Replay demo actions directly in env and measure success rate.
Should be ~100% if eval logic is correct.
"""
import json, os
import h5py
import numpy as np
import gymnasium as gym
import mani_skill.envs

DEMO_JSON = os.path.expanduser(
    "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.json"
)
DEMO_H5 = os.path.expanduser(
    "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5"
)

N = 50

with open(DEMO_JSON) as f:
    episodes = json.load(f)["episodes"][:N]

env = gym.make(
    "PegInsertionSide-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode=None,
    max_episode_steps=500,
)

successes = []
with h5py.File(DEMO_H5, "r") as f:
    keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[:N]
    for i, (ep, key) in enumerate(zip(episodes, keys)):
        actions = f[key]["actions"][:].astype(np.float32)  # (T, 8)
        seed = ep["episode_seed"]

        obs, _ = env.reset(seed=int(seed))
        done = False
        step = 0
        while not done and step < len(actions):
            obs, _, terminated, truncated, info = env.step(actions[step].reshape(1, -1))
            done = terminated or truncated
            step += 1

        success = bool(np.array(info.get("success", False)).reshape(-1)[0])
        successes.append(success)
        print(f"  ep{i+1:2d} seed={seed}  steps={step}  {'OK' if success else 'FAIL'}")

env.close()
print(f"\nDemo replay success rate: {np.mean(successes):.2%}  ({sum(successes)}/{N})")
