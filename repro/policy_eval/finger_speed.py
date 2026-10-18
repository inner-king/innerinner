# SPDX-License-Identifier: Apache-2.0
"""How fast the simulated Robotiq fingers close and open, from rollouts.hdf5 (repro/README.md, 26.4).

The eval stage's header lists "Lower robot finger max velocity" among the authors' sim tuning steps. This
reads the finger joints out of the serialized simulator state the stage records after every action, so the
check needs no Isaac Sim. The state vector holds the robot's joint positions, then its velocities, then the
arm controller's goal flag and goal; the goal block is the one that matches the recorded actions (as in the
tracking analysis of chapter 26), and the joint block sits 2 * n_dof + 1 entries before it.

For every switch of the gripper command it reports how long the first Robotiq joint takes to finish 95% of
the travel it makes in the next two seconds; switches after which the fingers barely move (already closed
on an object, or already open) are skipped.

    python repro/policy_eval/finger_speed.py out/marker_in_cup/s15_eval_pi05_main/*/*/results_*/rollouts.hdf5
"""
import argparse

import h5py
import numpy as np


def find_goal(S, A):
    """Offset of the arm controller's goal in the state vector: where state[t+1] holds action[t][:7]."""
    match = [i for i in range(S.shape[1] - 6)
             if np.mean(np.abs(S[1:, i:i + 7] - A[:, :7]).max(1) < 1e-6) > 0.9]
    if not match:
        raise SystemExit("no block of the state follows the recorded actions")
    return match[0]


def switch_times(q, on, dt, window):
    """Seconds to 95% of the travel after each rising edge of `on`, for moves of more than 0.05 rad."""
    out = []
    for t in np.where(on[1:] & ~on[:-1])[0] + 1:
        travel = q[t:t + window] - q[t]
        if len(travel) < 5 or np.abs(travel).max() < 0.05:
            continue
        out.append(np.argmax(np.abs(travel) >= 0.95 * np.abs(travel).max()) * dt)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollouts", nargs="+", help="rollouts.hdf5 files")
    ap.add_argument("--n-dof", type=int, default=15, help="robot DOFs (7 arm + 8 Robotiq joints for nv_franka_droid)")
    ap.add_argument("--dt", type=float, default=1 / 15, help="seconds per action (one recorded state each)")
    args = ap.parse_args()

    window = int(round(2.0 / args.dt))
    closes, opens, episodes = [], [], 0
    for path in args.rollouts:
        with h5py.File(path, "r") as f:
            for name in f["data"]:
                g = f["data"][name]
                if "action" not in g or len(g["action"]) < 10:
                    continue  # the stage's extra env.reset() leaves one-state demos between episodes
                S, A = g["state"][:].astype(float), g["action"][:].astype(float)
                qi = find_goal(S, A) - (2 * args.n_dof + 1)
                finger = S[:, qi + 7]  # the first Robotiq joint; 0 open, 0.785 rad closed
                # The recorded action is the one the controller received: inverted from DROID's convention and
                # binarized by the eval stage, so 1 opens. Checked on the data: after a rise the joint goes to 0.
                closed = A[:, 7] < 0.5
                closes += switch_times(finger, closed, args.dt, window)
                opens += switch_times(finger, ~closed, args.dt, window)
                episodes += 1
    print(f"{episodes} episodes")
    for what, ts in (("close", closes), ("open", opens)):
        if ts:
            print(f"{what}: {len(ts)} switches, 95% of the travel in median {np.median(ts):.2f} s "
                  f"(p10 {np.percentile(ts, 10):.2f}, p90 {np.percentile(ts, 90):.2f})")
        else:
            print(f"{what}: none")


if __name__ == "__main__":
    main()
