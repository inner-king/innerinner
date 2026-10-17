# SPDX-License-Identifier: Apache-2.0
"""How closely the simulated arm follows the policy's joint targets, from rollouts.hdf5 (repro/README.md, 18.4).

1_eval_policy_og_scene.py drives the arm with a JointController (absolute joint positions, a moving-average
filter of 5) and records the serialized simulator state after every action. That vector holds, in order, the
arm's joint positions and velocities, the controller's goal (the action just applied) and the filter's
buffer, so tracking can be read without Isaac Sim. The goal block is found by matching the recorded actions;
the joint block sits 2 * n_dof + 1 entries before it (positions, velocities, the goal's valid flag).

Reported, over every action of every episode:
  - whether the filter's 5 samples all equal the current goal when the action ends, i.e. whether the filter's
    delay stays inside one action (it is fed every physics step, 8 per action at 120 Hz / 15 Hz);
  - the fraction of the remaining gap to each target the arm closes within one action;
  - the lag tau = -(q - target) / qdot per joint, which for a first-order follower is its time constant;
  - the residual at a point 15 cm beyond the flange (about the fingertips) by Franka forward kinematics,
    for all actions and for slow ones (commanded under 5 cm/s).

    python repro/policy_eval/arm_tracking.py out/marker_in_cup/s15_eval_pi05_main/*/*/results_*/rollouts.hdf5
"""
import argparse

import h5py
import numpy as np

# Franka Panda, modified DH (a_{i-1}, d_i, alpha_{i-1}); the flange is 0.107 m beyond joint 7.
DH = [(0, 0.333, 0), (0, 0, -np.pi / 2), (0, 0.316, np.pi / 2), (0.0825, 0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0, 0, np.pi / 2), (0.088, 0, np.pi / 2)]
FLANGE = 0.107


def fk(q, tool=0.15):
    """Position of a point `tool` metres beyond the flange along its z axis, in the robot base frame."""
    T = np.eye(4)
    for (a, d, al), th in zip(DH, q):
        ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])
    return (T @ np.array([0, 0, FLANGE + tool, 1]))[:3]


def find_goal(S, A):
    """Offset of the arm controller's goal in the state vector: where state[t+1] holds action[t][:7].

    The goal differs from the action where the controller clipped it to the joint limits, so a block matching
    most actions is taken. The filter's five samples follow it and match too once an action ends.
    """
    match = [i for i in range(S.shape[1] - 6)
             if np.mean(np.abs(S[1:, i:i + 7] - A[:, :7]).max(1) < 1e-6) > 0.9]
    if not match or match[1:] != [match[0] + 7 * k for k in range(1, len(match))]:
        raise SystemExit(f"no goal block followed by the filter's samples; blocks matching the actions: {match}")
    return match[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollouts", nargs="+", help="rollouts.hdf5 files")
    ap.add_argument("--n-dof", type=int, default=15, help="robot DOFs (7 arm + 8 Robotiq joints for nv_franka_droid)")
    ap.add_argument("--dt", type=float, default=1 / 15, help="seconds per action")
    args = ap.parse_args()

    E, C, V, tool_res, tool_cmd, filt_ok, clipped, n_act = [], [], [], [], [], [], [], 0
    for path in args.rollouts:
        with h5py.File(path, "r") as f:
            for name in f["data"]:
                g = f["data"][name]
                if "action" not in g or len(g["action"]) < 10:
                    continue  # the stage's extra env.reset() leaves one-state demos between episodes
                S, A = g["state"][:].astype(float), g["action"][:].astype(float)
                goal = find_goal(S, A)
                qi = goal - (2 * args.n_dof + 1)
                q, qd = S[:, qi:qi + 7], S[:, qi + args.n_dof:qi + args.n_dof + 7]
                a = S[1:, goal:goal + 7]  # the target the drives received: the action, clipped to the limits
                clipped.append(np.abs(a - A[:, :7]).max(1) > 1e-6)
                if np.median(np.abs(q[1:] - a)) > 0.1:
                    raise SystemExit(f"{path}:{name}: joint block at {qi} does not follow the goal; check --n-dof")
                buf = S[1:, goal + 7:goal + 42].reshape(-1, 5, 7)
                filt_ok.append(np.abs(buf - a[:, None, :]).max(axis=(1, 2)) < 1e-6)
                E.append(q[1:] - a)
                C.append(a - q[:-1])
                V.append(qd[1:])
                for t in range(len(a)):
                    x_cmd, x_end, x_start = fk(a[t]), fk(q[t + 1]), fk(q[t])
                    tool_res.append(np.linalg.norm(x_end - x_cmd))
                    tool_cmd.append(np.linalg.norm(x_cmd - x_start))
                n_act += len(a)
    E, C, V = np.concatenate(E), np.concatenate(C), np.concatenate(V)
    tool_res, tool_cmd = np.array(tool_res), np.array(tool_cmd)
    filt_ok, clipped = np.concatenate(filt_ok), np.concatenate(clipped)

    print(f"{n_act} actions from {len(args.rollouts)} file(s)")
    print(f"filter buffer equals the goal at the end of the action: {filt_ok.mean():.1%} of actions")
    print(f"actions the controller clipped to the joint limits: {clipped.mean():.2%}")
    closed = [1 + np.median(E[np.abs(C[:, j]) > 0.005, j] / C[np.abs(C[:, j]) > 0.005, j]) for j in range(7)]
    tau = [np.median(-E[np.abs(V[:, j]) > 0.05, j] / V[np.abs(V[:, j]) > 0.05, j]) for j in range(7)]
    print("fraction of the gap closed within one action, per joint: " + " ".join(f"{c:.2f}" for c in closed))
    print("lag tau = -(q - target) / qdot per joint (s): " + " ".join(f"{t:.3f}" for t in tau))
    slow = tool_cmd / args.dt < 0.05
    print(f"tool-point residual (mm): median {1e3 * np.median(tool_res):.1f}, p95 {1e3 * np.percentile(tool_res, 95):.1f}; "
          f"commanded step median {1e3 * np.median(tool_cmd):.1f}")
    print(f"slow actions (commanded < 5 cm/s, {slow.mean():.0%} of all): residual median "
          f"{1e3 * np.median(tool_res[slow]):.1f} mm, p95 {1e3 * np.percentile(tool_res[slow], 95):.1f} mm")


if __name__ == "__main__":
    main()
