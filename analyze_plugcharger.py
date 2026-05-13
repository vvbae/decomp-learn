"""
Analyze PlugCharger-v1 demo trajectories: find contact onset and compute split statistics.

Geometry:
  charger peg tips: charger_pos + R_charger @ [0.016, ±0.007, 0]  (16mm in front of base center)
  receptacle face:  receptacle_pos + R_receptacle @ [-0.01, 0, 0]  (socket opening, 10mm from center)

Contact onset: first step where either peg tip crosses the receptacle face plane.
  peg_tip_in_receptacle_frame = R_receptacle^T @ (peg_tip - receptacle_pos)
  contact when peg_tip_in_receptacle_frame[x] >= -0.01  (tip past the face)

env_states layout per actor (13 values):
  [0:3]  pos (x, y, z)
  [3:7]  quat (w, x, y, z)  — SAPIEN convention
  [7:10] linear velocity
  [10:13] angular velocity
"""

import h5py
import numpy as np

H5 = "demos/PlugCharger-v1/motionplanning/trajectory.h5"

_PEG_TIP_X    = 0.016   # peg tip x offset from charger center (m)
_PEG_GAP      = 0.007   # peg y offset from charger centerline (m)
_FACE_X       = -0.01   # receptacle face x in receptacle local frame (m)


def quat_to_rotmat(q):
    """[w, x, y, z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def find_contact_step(charger_states, recept_states):
    """
    Returns (strict_contact_step, broad_contact_step).
    strict: first step where either peg tip crosses the socket face plane.
    broad:  first step where charger center is within 5cm of receptacle center
            (captures full alignment + insertion phase).
    Both are None if not found.
    """
    T1 = charger_states.shape[0]
    charger_pos  = charger_states[:, :3]
    charger_quat = charger_states[:, 3:7]   # [w,x,y,z]
    recept_pos   = recept_states[0, :3]
    recept_quat  = recept_states[0, 3:7]
    R_recept     = quat_to_rotmat(recept_quat)

    tip_offsets = np.array([
        [_PEG_TIP_X,  _PEG_GAP, 0.],
        [_PEG_TIP_X, -_PEG_GAP, 0.],
    ])

    dists = np.linalg.norm(charger_pos - recept_pos, axis=1)
    broad_candidates = np.where(dists < 0.05)[0]
    broad = int(broad_candidates[0]) if len(broad_candidates) else None

    strict = None
    for t in range(T1):
        R_c = quat_to_rotmat(charger_quat[t])
        for offset in tip_offsets:
            tip_world = charger_pos[t] + R_c @ offset
            tip_local = R_recept.T @ (tip_world - recept_pos)
            if tip_local[0] <= _FACE_X:
                strict = t
                break
        if strict is not None:
            break

    return strict, broad


def main():
    traj_lengths        = []
    strict_geo_list     = []
    strict_cont_list    = []
    broad_geo_list      = []
    broad_cont_list     = []
    no_strict           = 0
    no_broad            = 0

    with h5py.File(H5, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        n = len(keys)
        print(f"Processing {n} trajectories...")

        for i, k in enumerate(keys):
            ep = f[k]
            charger = ep["env_states/actors/charger"][:]
            recept  = ep["env_states/actors/receptacle"][:]
            T = charger.shape[0] - 1

            strict, broad = find_contact_step(charger, recept)

            traj_lengths.append(T)

            if strict is None:
                no_strict += 1
                strict_geo_list.append(T)
                strict_cont_list.append(0)
            else:
                strict_geo_list.append(strict)
                strict_cont_list.append(T - strict)

            if broad is None:
                no_broad += 1
                broad_geo_list.append(T)
                broad_cont_list.append(0)
            else:
                broad_geo_list.append(broad)
                broad_cont_list.append(T - broad)

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{n} done")

    traj_lengths  = np.array(traj_lengths)
    strict_geo    = np.array(strict_geo_list)
    strict_cont   = np.array(strict_cont_list)
    broad_geo     = np.array(broad_geo_list)
    broad_cont    = np.array(broad_cont_list)

    mean_T = np.mean(traj_lengths)

    print(f"\n{'='*60}")
    print(f"PLUGCHARGER-V1 TRAJECTORY SPLIT  (N={n})")
    print(f"{'='*60}")
    print(f"\nTotal trajectory length:")
    print(f"  mean={mean_T:.1f}  std={np.std(traj_lengths):.1f}"
          f"  min={np.min(traj_lengths)}  max={np.max(traj_lengths)}")

    print(f"\n--- STRICT definition: peg tip crosses socket face plane ---")
    print(f"    (equivalent to PegInsertionSide contact onset, no_contact={no_strict})")
    print(f"  Geometric part:  mean={np.mean(strict_geo):.1f}  ({np.mean(strict_geo)/mean_T:.1%} of total)")
    print(f"  Contact part:    mean={np.mean(strict_cont):.1f}  ({np.mean(strict_cont)/mean_T:.1%} of total)")

    print(f"\n--- BROAD definition: charger center within 5cm of receptacle ---")
    print(f"    (alignment + insertion, no_contact={no_broad})")
    print(f"  Geometric part:  mean={np.mean(broad_geo):.1f}  ({np.mean(broad_geo)/mean_T:.1%} of total)")
    print(f"  Contact part:    mean={np.mean(broad_cont):.1f}  ({np.mean(broad_cont)/mean_T:.1%} of total)")

    print(f"\nContact onset step distribution (strict, percentiles):")
    cs = strict_geo[np.array(strict_cont_list) > 0]
    for pct in [10, 25, 50, 75, 90]:
        print(f"  p{pct:2d}: step {np.percentile(cs, pct):.0f}")


if __name__ == "__main__":
    main()
