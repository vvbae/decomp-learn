"""
Analyze 996 demo trajectories: find contact onset and compute split statistics.

Obs layout (43-dim):
  [0:9]   joint qpos
  [9:18]  joint qvel
  [18:25] tcp_pose (x,y,z, qw,qx,qy,qz)
  [25:32] peg_pose (x,y,z, qw,qx,qy,qz)
  [32:35] peg_half_size (length, radius, radius)
  [35:42] box_hole_pose (x,y,z, qw,qx,qy,qz)
  [42:43] box_hole_radius

Contact onset: first step where peg head enters the box face plane.
  peg_head_pos = peg_pos + R_peg @ [length, 0, 0]
  peg_head_in_hole = R_box^T @ (peg_head_pos - box_hole_pos)
  contact = first step where peg_head_in_hole[x] >= -length  (peg tip past box outer face)
"""

import h5py
import numpy as np

H5 = "/home/viviwei/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5"


def quat_to_rotmat(q):
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def peg_head_in_hole_frame(obs):
    """
    Compute peg head position in box_hole frame for all timesteps.
    obs: (T, 43)
    returns: (T, 3)
    """
    peg_pos  = obs[:, 25:28]           # (T, 3)
    peg_quat = obs[:, 28:32]           # (T, 4) [w,x,y,z]
    peg_len  = obs[:, 32]              # (T,)
    hole_pos  = obs[:, 35:38]          # (T, 3)
    hole_quat = obs[:, 38:42]          # (T, 4) [w,x,y,z]
    T = len(obs)

    result = np.zeros((T, 3))
    for t in range(T):
        R_peg  = quat_to_rotmat(peg_quat[t])
        R_hole = quat_to_rotmat(hole_quat[t])
        peg_head = peg_pos[t] + R_peg @ np.array([peg_len[t], 0., 0.])
        result[t] = R_hole.T @ (peg_head - hole_pos[t])
    return result


def find_contact_step(obs):
    """
    First step where peg head x-coordinate in hole frame >= -length.
    The box outer face is at x = -length in box frame, so crossing 0 means fully entering,
    but touching starts at x = -length.  We use x >= -length * 1.05 as onset with a small margin.
    Returns contact_step (int) or None.
    """
    peg_len   = obs[0, 32]   # length is constant per episode
    head_in_hole = peg_head_in_hole_frame(obs)  # (T, 3)
    x = head_in_hole[:, 0]

    # onset = first step where tip has reached the box outer face plane
    threshold = -peg_len * 1.05   # small margin so numerical noise doesn't trigger early
    candidates = np.where(x >= threshold)[0]
    if len(candidates) == 0:
        return None, head_in_hole
    return int(candidates[0]), head_in_hole


def dist_at_contact(head_in_hole, contact_step):
    """Euclidean distance of peg head from hole center at contact onset."""
    if contact_step is None:
        return float('nan')
    return float(np.linalg.norm(head_in_hole[contact_step]))


def main():
    traj_lengths = []
    contact_steps = []
    geo_steps_list = []
    contact_steps_list = []
    dists_at_contact = []
    no_contact_count = 0

    with h5py.File(H5, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        n = len(keys)
        print(f"Processing {n} trajectories...")

        for i, k in enumerate(keys):
            obs = f[k]["obs"][:].astype(np.float32)  # (T+1, 43)
            T = obs.shape[0] - 1  # number of action steps

            contact_step, head_in_hole = find_contact_step(obs)

            traj_lengths.append(T)
            if contact_step is None:
                no_contact_count += 1
                geo_steps_list.append(T)
                contact_steps_list.append(0)
                dists_at_contact.append(float('nan'))
            else:
                contact_steps.append(contact_step)
                geo_steps_list.append(contact_step)
                contact_steps_list.append(T - contact_step)
                dists_at_contact.append(dist_at_contact(head_in_hole, contact_step))

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n} done")

    geo_steps = np.array(geo_steps_list)
    cont_steps = np.array(contact_steps_list)
    dists = np.array([d for d in dists_at_contact if not np.isnan(d)])

    print(f"\n{'='*60}")
    print(f"TRAJECTORY SPLIT STATISTICS  (N={n}, no_contact={no_contact_count})")
    print(f"{'='*60}")
    print(f"\nTotal trajectory length:")
    print(f"  mean={np.mean(traj_lengths):.1f}  std={np.std(traj_lengths):.1f}  min={np.min(traj_lengths)}  max={np.max(traj_lengths)}")
    print(f"\nGeometric part (steps before contact):")
    print(f"  mean={np.mean(geo_steps):.1f}  std={np.std(geo_steps):.1f}  min={np.min(geo_steps)}  max={np.max(geo_steps)}")
    print(f"  fraction of total: {np.mean(geo_steps)/np.mean(traj_lengths):.1%}")
    print(f"\nContact part (steps from contact to end):")
    print(f"  mean={np.mean(cont_steps):.1f}  std={np.std(cont_steps):.1f}  min={np.min(cont_steps)}  max={np.max(cont_steps)}")
    print(f"  fraction of total: {np.mean(cont_steps)/np.mean(traj_lengths):.1%}")
    print(f"\nPeg-head distance from hole center at contact onset:")
    print(f"  mean={np.mean(dists)*100:.1f} cm  std={np.std(dists)*100:.1f} cm  min={np.min(dists)*100:.1f} cm  max={np.max(dists)*100:.1f} cm")

    # histogram of contact steps
    print(f"\nContact onset step distribution (percentiles):")
    cs = np.array([s for s in contact_steps])
    for pct in [10, 25, 50, 75, 90]:
        print(f"  p{pct:2d}: step {np.percentile(cs, pct):.0f}")


if __name__ == "__main__":
    main()
