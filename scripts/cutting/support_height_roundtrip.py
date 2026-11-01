# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round trip of the support-height log over recorded demonstrations: z -> u -> z at every tick and physics step.

Each demonstration is an ``ep_XXXXX.npz`` written by tofu_policy_episodes.py, read with its folder's ``setup.json`` and,
when present, its ``ep_XXXXX.json``. At each tick the commanded hand pose gives the support height h (the knife edge's
lowest point above the support plane s (the block's bottom, the solver board plane)), the log height u = ln(h + eps) is
taken, and the hand height is solved back from u.
The pass criterion is the error after u is rounded to float32, the dataset's dtype.

Reading a demonstration and choosing its knife (episode JSON, setup grasp, or a fit to the folder) live in
simfoundry/cutting/demonstrations.py, which the converter shares; this script only checks the round trip.

Usage:
    python scripts/cutting/support_height_roundtrip.py --npz PATH [PATH ...] [--out JSON]
    python scripts/cutting/support_height_roundtrip.py --accepted FILE [--count 10] [--out JSON]

Exits with 1 when a round trip is 1 um or more off, or when a pose has its edge at or below the support minus eps.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from simfoundry.cutting.demonstrations import choose_knife, edge_residual_mm, load_demo, rms
from simfoundry.cutting.support_height import hand_z_from_log_height, log_height, support_height

ROUNDTRIP_TOL_M = 1e-6


def round_trip(poses, knife, s, eps):
    """Solves each pose's hand height back from its log height and reports the errors.

    Poses whose edge is at or below the support by eps have no log height; they are left out and counted. The solve
    starts from the same pose with its height set to 0, so the recovered height cannot be the input's own.

    Args:
        poses: (N, 4, 4) world hand poses.
        knife: (4, 4) knife-in-hand transform.
        s: the support plane's height [m].
        eps: the epsilon of the log height [m].

    Returns:
        A dict: ``kept`` and ``left_out`` pose counts, the largest error in float64 and in float32 (``err64_um``,
        ``err32_um``, both in um), and the largest height the solve started from (``start_z_max_m``).
    """
    h = support_height(poses, knife, s)
    keep = h + eps > 0.0
    kept = poses[keep]
    u = log_height(h[keep], eps)
    start = kept.copy()
    start[:, 2, 3] = 0.0
    z64 = hand_z_from_log_height(start, knife, s, u, eps)
    z32 = hand_z_from_log_height(start, knife, s, u.astype(np.float32), eps)
    return {
        "kept": int(keep.sum()),
        "left_out": int((~keep).sum()),
        "err64_um": 1e6 * float(np.max(np.abs(z64 - kept[:, 2, 3]), initial=0.0)),
        "err32_um": 1e6 * float(np.max(np.abs(z32 - kept[:, 2, 3]), initial=0.0)),
        "start_z_max_m": float(np.max(np.abs(start[:, 2, 3]), initial=0.0)),
    }


def check_demo(npz_path, eps, fit_episodes, cache):
    """The report for one demonstration: its knife and source, the edge residual, the heights and both round trips."""
    demo = load_demo(npz_path)
    knife, source, fit = choose_knife(demo, fit_episodes, cache)
    residual = edge_residual_mm(demo, knife)
    h_mm = 1e3 * support_height(demo["E_cmd"], knife, demo["s"])
    cross = np.abs(demo["E_cmd"][:, :3, 3] - demo["cmd_steps"][demo["j"], :3, 3]).max()
    return {
        "npz": str(demo["path"]),
        "knife": source,
        "fit_episodes": None if fit is None else fit["episodes"],
        "fit_rms_mm": None if fit is None else fit["fit_rms_mm"],
        "resid_rms_mm": rms(residual),
        "resid_max_mm": float(np.abs(residual).max()),
        "ticks": len(residual),
        "s_m": demo["s"],
        "h_mm": {"min": float(h_mm.min()), "median": float(np.median(h_mm)), "max": float(h_mm.max())},
        "tick_roundtrip": round_trip(demo["E_cmd"], knife, demo["s"], eps),
        "step_roundtrip": round_trip(demo["cmd_steps"], knife, demo["s"], eps),
        "cross_check_m": float(cross),
    }


def describe(r):
    """One line per demonstration."""
    tick, step = r["tick_roundtrip"], r["step_roundtrip"]
    return (f"{r['npz']} | knife {r['knife']} | edge residual rms {r['resid_rms_mm']:.4f} max {r['resid_max_mm']:.4f} mm | "
            f"ticks {r['ticks']} | h min/median/max {r['h_mm']['min']:.2f}/{r['h_mm']['median']:.2f}/{r['h_mm']['max']:.2f} mm | "
            f"round trip ticks f64 {tick['err64_um']:.4f} f32 {tick['err32_um']:.4f} um, steps f32 {step['err32_um']:.4f} um | "
            f"at or below -eps: ticks {tick['left_out']} steps {step['left_out']} | base cross-check {r['cross_check_m']:.2e} m")


def select_accepted(list_path, count):
    """``count`` archive paths from the list, at evenly spaced positions from its first entry to its last."""
    entries = [line.strip() for line in Path(list_path).read_text().splitlines() if line.strip()]
    positions = np.linspace(0, len(entries) - 1, count).round().astype(int)
    return [entries[i] for i in positions]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Round trip of the support-height log over recorded demonstrations.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz", nargs="+", help="demonstration archives (ep_XXXXX.npz), in the order given")
    source.add_argument("--accepted", help="a list of archive paths, one per line")
    parser.add_argument("--count", type=int, default=10, help="with --accepted: how many, spread evenly over the list")
    parser.add_argument("--eps-mm", type=float, default=1.0, help="eps of the log height [mm]")
    parser.add_argument("--fit-episodes", type=int, default=9, help="how many episodes a folder's fitted knife uses")
    parser.add_argument("--out", help="also write the full result here as JSON")
    args = parser.parse_args(argv)

    paths = args.npz or select_accepted(args.accepted, args.count)
    eps = args.eps_mm * 1e-3
    started = time.time()
    cache = {}
    demos = []
    for path in paths:
        report = check_demo(path, eps, args.fit_episodes, cache)
        demos.append(report)
        print(describe(report), flush=True)
    worst_um = max(max(d["tick_roundtrip"]["err32_um"], d["step_roundtrip"]["err32_um"]) for d in demos)
    left_out = sum(d["tick_roundtrip"]["left_out"] + d["step_roundtrip"]["left_out"] for d in demos)
    code = int(worst_um >= 1e6 * ROUNDTRIP_TOL_M or left_out > 0)
    elapsed = time.time() - started
    print(f"total: {len(demos)} demonstrations, max float32 round trip {worst_um:.4f} um, "
          f"poses at or below -eps {left_out}, exit {code}, {elapsed:.1f} s")
    if args.out:
        result = {"demos": demos, "max_float32_roundtrip_um": worst_um, "left_out": left_out, "exit_code": code,
                  "elapsed_s": elapsed}
        Path(args.out).write_text(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
