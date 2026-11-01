# SPDX-License-Identifier: Apache-2.0
"""What the tofu's force does to the arm that holds the knife, from two runs of tofu_og_cut.py.

One run applies the tofu's wrench to the knife, the other (--no-wrench) only measures it; both
record the arm's joint torques and Jacobian every physics step (--efforts). Everything else being
equal, the difference in joint torques is what the arm feels of the tofu. A wrench w on the end
effector takes joint torques J^T w to hold, J the end effector's Jacobian in the robot's base
frame, so the difference is compared with J^T w, w being the tofu's force on the knife and its
moment carried over to the end effector, applied a physics step after the solver computed it.

Example:
    ~/miniforge3/envs/simfoundry/bin/python scripts/cutting/tofu_og_torques.py \\
        results/tofu_og/cook_efforts results/tofu_og/cook_efforts_nowrench
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(run_dir):
    rows = [json.loads(line) for line in open(Path(run_dir) / "arm.jsonl")]
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    return rows, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("with_wrench", type=Path)
    ap.add_argument("without_wrench", type=Path)
    ap.add_argument("--phases", nargs="+", default=["stroke", "press"])
    args = ap.parse_args()

    a, ma = load(args.with_wrench)
    b, mb = load(args.without_wrench)
    assert not ma["no_wrench"] and mb["no_wrench"], "give the run that applied the wrench first"
    n = min(len(a), len(b))
    base_yaw = np.radians(ma["scene_changes"]["robot_yaw_deg"])
    c, s = np.cos(base_yaw), np.sin(base_yaw)
    Rb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rows = []
    for k in range(1, n):
        ra, rb, prev = a[k], b[k], a[k - 1]
        if ra["phase"] not in args.phases or "efforts" not in ra or "fx" not in prev:
            continue
        assert ra["k"] == rb["k"] and ra["phase"] == rb["phase"]
        F = np.array([prev["fx"], prev["fy"], prev["fz"]])  # applied over this step
        M = np.array([prev["mx"], prev["my"], prev["mz"]])  # about the edge midpoint
        edge = np.array([prev["kx"], prev["ky"], prev["kz"]])
        eef = np.asarray(ra["eef"])
        M_eef = M + np.cross(edge - eef, F)
        w = np.concatenate([Rb.T @ F, Rb.T @ M_eef])
        J = np.asarray(ra["jacobian"])
        expected = J.T @ w
        diff = np.asarray(ra["efforts"]) - np.asarray(rb["efforts"])
        rows.append((ra["t"], np.linalg.norm(F), expected, diff))
    if not rows:
        raise SystemExit("no steps with efforts in the given phases")
    E = np.array([r[2] for r in rows])
    D = np.array([r[3] for r in rows])
    loaded = np.array([r[1] for r in rows]) > 0.02
    # Least-squares gain of the measured difference on J^T w: 1 if the arm holds exactly the tofu's wrench,
    # -1 if the efforts are counted the other way round.
    gain = float((E[loaded] * D[loaded]).sum() / max((E[loaded] ** 2).sum(), 1e-30))
    resid = D[loaded] - gain * E[loaded]
    out = dict(
        steps=len(rows), loaded_steps=int(loaded.sum()),
        gain_of_measured_on_JTw=gain,
        corr=float(np.corrcoef(E[loaded].ravel(), D[loaded].ravel())[0, 1]),
        rms_expected_nm=float(np.sqrt((E[loaded] ** 2).mean())),
        rms_measured_nm=float(np.sqrt((D[loaded] ** 2).mean())),
        rms_residual_nm=float(np.sqrt((resid ** 2).mean())),
        rms_measured_unloaded_nm=float(np.sqrt((D[~loaded] ** 2).mean())) if (~loaded).any() else None,
        max_abs_expected_per_joint_nm=np.abs(E).max(axis=0).round(4).tolist(),
        max_abs_measured_per_joint_nm=np.abs(D).max(axis=0).round(4).tolist(),
    )
    print(json.dumps(out, indent=1))
    (args.with_wrench / "torques_vs_no_wrench.json").write_text(json.dumps(out, indent=1))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.array([r[0] for r in rows])
        j = int(np.argmax(np.abs(E).max(axis=0)))
        fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax[0].plot(t, [r[1] for r in rows], color="0.2")
        ax[0].set_ylabel("|F| tofu on knife [N]")
        ax[1].plot(t, E[:, j], label=f"J^T w, joint {j + 1}", color="tab:blue")
        ax[1].plot(t, np.sign(gain) * D[:, j], label="measured difference" + (" (sign flipped)" if gain < 0 else ""),
                   color="tab:orange", lw=0.8)
        ax[1].set_ylabel("torque [N m]")
        ax[1].set_xlabel("time [s]")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(args.with_wrench / "torques_vs_no_wrench.png", dpi=130)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
