"""
Wrapper around mani_skill's motion planning demo collector that adds a
per-seed timeout so a hanging plan_screw call doesn't stall the whole run.
"""
import faulthandler
faulthandler.enable()  # print C-level stack trace on segfault/abort

import signal
import sys
import multiprocessing as mp
from copy import deepcopy
import time
import os
import os.path as osp

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.run import parse_args, MP_SOLUTIONS


SOLVE_TIMEOUT = 30  # seconds per seed before giving up


class _Timeout(Exception):
    pass


def _timed_solve(solve, env, seed, vis):
    def _handler(signum, frame):
        raise _Timeout()

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(SOLVE_TIMEOUT)
    try:
        return solve(env, seed=seed, debug=False, vis=vis)
    except _Timeout:
        tqdm.write(f"seed {seed}: plan_screw timed out after {SOLVE_TIMEOUT}s, skipping")
        return -1
    finally:
        signal.alarm(0)


def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = args.env_id
    # render_backend="none" avoids Vulkan/GPU renderer init, which crashes on
    # newer GPUs (e.g. RTX 5070) with older sapien builds. Safe since we don't
    # save videos during collection.
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=None,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
        render_backend="none",
    )
    if env_id not in MP_SOLUTIONS:
        raise RuntimeError(
            f"No motion planning solution for {env_id}. "
            f"Available: {list(MP_SOLUTIONS.keys())}"
        )

    new_traj_name = args.traj_name if args.traj_name else time.strftime("%Y%m%d_%H%M%S")
    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)

    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, env_id, "motionplanning"),
        trajectory_name=new_traj_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )
    output_h5_path = env._h5_file.filename
    solve = MP_SOLUTIONS[env_id]

    print(f"Motion Planning Running on {env_id}")
    pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0

    while True:
        tqdm.write(f"[{time.strftime('%H:%M:%S')}] seed {seed}: solving...")
        try:
            res = _timed_solve(solve, env, seed, vis=bool(args.vis))
        except Exception as e:
            tqdm.write(f"seed {seed}: error in motion planning: {e}")
            res = -1

        tqdm.write(f"[{time.strftime('%H:%M:%S')}] seed {seed}: done (res={'timeout/fail' if res == -1 else 'ok'})")
        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)

        successes.append(success)

        if args.only_count_success and not success:
            seed += 1
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue

        env.flush_trajectory()
        if args.save_video:
            env.flush_video()
        pbar.update(1)
        if solution_episode_lengths:
            pbar.set_postfix(dict(
                success_rate=np.mean(successes),
                failed_motion_plan_rate=failed_motion_plans / (seed - start_seed + 1),
                avg_episode_length=np.mean(solution_episode_lengths),
                max_episode_length=np.max(solution_episode_lengths),
            ))
        seed += 1
        passed += 1
        if passed == args.num_traj:
            break

    env.close()
    return output_h5_path


def main(args):
    if args.num_procs > 1 and args.num_procs < args.num_traj:
        args.num_traj = args.num_traj // args.num_procs
        seeds = [*range(0, args.num_procs * args.num_traj, args.num_traj)]
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, seeds[i]) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        output_path = res[0][: -len("0.h5")] + "h5"
        merge_trajectories(output_path, res)
        for h5_path in res:
            tqdm.write(f"Remove {h5_path}")
            os.remove(h5_path)
            json_path = h5_path.replace(".h5", ".json")
            tqdm.write(f"Remove {json_path}")
            os.remove(json_path)
    else:
        _main(args)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(parse_args())
