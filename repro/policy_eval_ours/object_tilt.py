# SPDX-License-Identifier: Apache-2.0
"""Count the episodes in which an object tipped over, from the states the eval stage recorded (README 30.15).

rollouts.hdf5 keeps OmniGibson's flattened state at every step, without names. An object's orientation is found
in it as the four values that equal the object's start orientation (from the starts file) in each episode's first
frame; its tilt is then the angle between the object's own z axis and the world's, at every step.

    python repro/policy_eval_ours/object_tilt.py --object white_paper_cup_owccwy_1 \\
        --init results/policy_eval_ours/openpi_local/init_states/marker_in_cup_ours_real.json \\
        marker_in_cup_ours_real:pi0:assisted,assisted_rep2 marker_in_cup_ours_real:gr00t17:assisted,assisted_rep2

Each argument is SCENE:POLICY:TAG[,TAG...] as in milestones.py. Episodes are matched to the recorded rollouts in
order; the file also holds a one-frame entry per reset, which is skipped.
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOTS = {"gr00t": "gr00t", "gr00t17": "gr00t", "pi05": "openpi_local", "pi0": "openpi_local"}


def tilt_deg(q):
    """Angle (deg) between the body z axis of xyzw quaternions q (..., 4) and the world z axis."""
    x, y = q[..., 0], q[..., 1]
    return np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def find_orientation(frame, q0, atol):
    """Offset of the four values in a state vector that equal quaternion q0 (or -q0), or None."""
    for i in range(len(frame) - 3):
        if np.allclose(frame[i:i + 4], q0, atol=atol) or np.allclose(frame[i:i + 4], -q0, atol=atol):
            return i
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="SCENE:POLICY:TAG[,TAG...]")
    ap.add_argument("--object", required=True)
    ap.add_argument("--init", required=True, help="the starts file the runs used")
    ap.add_argument("--threshold", type=float, default=45.0, help="tilt (deg) that counts as tipped over")
    ap.add_argument("--results", default=os.path.join(REPO, "results", "policy_eval_ours"))
    args = ap.parse_args()
    starts = json.load(open(args.init))["init_states"]

    for spec in args.runs:
        scene, policy, tags = spec.split(":")
        rows, offsets = [], set()
        for tag in tags.split(","):
            found = sorted(glob.glob(os.path.join(args.results, ROOTS[policy], "out", scene, f"s15_eval_{policy}_{tag}",
                                                  "*", "*", "results_*")))
            if not found:
                continue
            episodes = json.load(open(os.path.join(found[-1], "eval_results.json")))["episodes"]
            with h5py.File(os.path.join(found[-1], "rollouts.hdf5"), "r") as h:
                demos = sorted((k for k in h["data"] if h["data"][k]["state"].shape[0] > 1),
                               key=lambda k: int(k.split("_")[1]))
                for name, e in zip(demos, episodes):
                    states = h["data"][name]["state"][()]
                    q0 = np.asarray(starts[e["episode_idx"]]["objects"][args.object]["ori"])
                    i = find_orientation(states[0], q0, 5e-3)
                    if i is None:
                        raise SystemExit(f"{spec} {tag} episode {e['episode_idx']}: {args.object}'s start "
                                         f"orientation is not in the recorded state")
                    offsets.add(i)
                    t = tilt_deg(states[:, i:i + 4])
                    rows.append((bool(e["success"]), t.max() > args.threshold, t[-1] > args.threshold))
        if not rows:
            print(f"{spec}: no runs")
            continue
        r = np.array(rows)
        fail = ~r[:, 0]
        print(f"{spec}: {len(r)} episodes, success {r[:, 0].sum()} | tipped at some point {r[:, 1].sum()} "
              f"(failures {r[fail, 1].sum()}/{fail.sum()}, successes {r[~fail, 1].sum()}/{(~fail).sum()}) | "
              f"tipped at the end {r[:, 2].sum()} | state offset {sorted(offsets)}")


if __name__ == "__main__":
    main()
