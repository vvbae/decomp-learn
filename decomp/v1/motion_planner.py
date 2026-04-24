"""
Geometric phase of PegInsertionSide-v1: Reach → Grasp → Align.

Extracted from ManiSkill's official motion-planning solution.
Stops at pre_insert_pose (peg tip just outside the hole face) and returns
the obs so the contact policy can take over.

Usage:
    from decomp.v1.motion_planner import run_geometric_phase

    obs, info = env.reset(seed=seed)
    obs, success = run_geometric_phase(env)
    if success:
        # hand off to contact policy
        ...
"""

import numpy as np
import sapien

from mani_skill.envs.tasks import PegInsertionSideEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

FINGER_LENGTH = 0.025


def run_geometric_phase(
    env: PegInsertionSideEnv,
    debug: bool = False,
    vis: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Run Reach → Grasp → Align on a freshly-reset env.

    The env must already be reset before calling this.
    Returns (obs, success): obs is the 43-dim state at pre_insert_pose,
    success=False means the planner failed to reach a waypoint.
    """
    raw = env.unwrapped
    assert raw.control_mode in ("pd_joint_pos", "pd_joint_pos_vel"), raw.control_mode

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=raw.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )

    obb = get_actor_obb(raw.peg)
    approaching = np.array([0, 0, -1])
    target_closing = raw.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    peg_init_pose = raw.peg.pose

    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = raw.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([-max(0.05, raw.peg_half_sizes[0, 0].item() / 2 + 0.01), 0, 0])
    grasp_pose = grasp_pose * offset

    # ---------------------------------------------------------------------- #
    # Reach
    # ---------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    if planner.move_to_pose_with_screw(reach_pose) == -1:
        planner.close()
        return _get_obs(env), False

    # ---------------------------------------------------------------------- #
    # Grasp
    # ---------------------------------------------------------------------- #
    if planner.move_to_pose_with_screw(grasp_pose) == -1:
        planner.close()
        return _get_obs(env), False
    planner.close_gripper()

    # ---------------------------------------------------------------------- #
    # Align: move peg to just outside hole face, refine 3×
    # ---------------------------------------------------------------------- #
    pre_offset = sapien.Pose([-0.01 - raw.peg_half_sizes[0, 0].item(), 0, 0])
    insert_pose = raw.goal_pose * peg_init_pose.inv() * grasp_pose
    pre_insert_pose = insert_pose * pre_offset

    if planner.move_to_pose_with_screw(pre_insert_pose) == -1:
        planner.close()
        return _get_obs(env), False

    for _ in range(3):
        delta_pose = raw.goal_pose * pre_offset * raw.peg.pose.inv()
        pre_insert_pose = delta_pose * pre_insert_pose
        if planner.move_to_pose_with_screw(pre_insert_pose) == -1:
            planner.close()
            return _get_obs(env), False

    planner.close()
    return _get_obs(env), True


def _get_obs(env) -> np.ndarray:
    """Return the current 43-dim state observation."""
    obs = env.unwrapped.get_obs()   # (1, 43) tensor
    return obs.cpu().numpy().reshape(-1).astype(np.float32)
