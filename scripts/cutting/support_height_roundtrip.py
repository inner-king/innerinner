# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round trip of the support-height log over recorded demonstrations: z -> u -> z at every tick and physics step.

Each demonstration is an ``ep_XXXXX.npz`` written by tofu_policy_episodes.py, read with its folder's ``setup.json`` and,
when present, its ``ep_XXXXX.json``. At each tick the commanded hand pose gives the support height h (the knife edge's
lowest point above the block top s), the log height u = ln(h + eps) is taken, and the hand height is solved back from u.
The pass criterion is the error after u is rounded to float32, the dataset's dtype.

The knife-in-hand transform K comes from the first source that explains the recorded edge heights:
  episode  the episode's own JSON (a generator that writes ``knife_in_eef``);
  setup    the setup's grasp, when its edge residual is under SETUP_TOL_MM;
  fitted   a K fitted to the folder's first ``--fit-episodes`` episodes that the setup does not explain (once per folder).

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
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tofu_policy_episodes as G  # noqa: E402 - pose9: the 9-vector the demonstrations' cmd column is written in

from simfoundry.cutting.support_height import (  # noqa: E402
    fit_knife_in_eef,
    hand_z_from_log_height,
    log_height,
    lowest_edge_z,
    support_height,
)

# provisional: the setup grasp explains its own episode to about 7 um and the other episodes of the 10-27 demos to about
# 100 mm, so any threshold between those two groups gives the same answer; 0.05 mm sits between them with room to spare
SETUP_TOL_MM = 0.05
ROUNDTRIP_TOL_M = 1e-6


def step_match_tolerance_s(t):
    """The largest gap a tick may have to its matched physics step [s].

    The steps' times are stored as float32, so the match can be off by about that format's resolution at the episode's
    time. The 1e-5 s this check first used holds only while that resolution is below it (up to t of about 256 s); at
    t = 367 s one float32 step is 3.05e-5 s, and a physics step is 8.3 ms, so four of those steps still reject any wrong
    match by a factor of about 70.

    Args:
        t: the tick times [s].

    Returns:
        The tolerance [s].
    """
    return 4.0 * float(np.spacing(np.float32(np.abs(t).max())))


def hand_poses(position, xyzw):
    """Poses (N, 4, 4) from positions (N, 3) and xyzw quaternions (N, 4), in whatever frame the inputs are in."""
    P = np.tile(np.eye(4), (len(position), 1, 1))
    P[:, :3, :3] = Rotation.from_quat(xyzw).as_matrix()
    P[:, :3, 3] = position
    return P


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def load_demo(npz_path):
    """The arrays of one demonstration that the round trip uses, with the hand poses in the solver's world frame.

    Args:
        npz_path: an ``ep_XXXXX.npz`` of tofu_policy_episodes.py; its folder holds ``setup.json``.

    Returns:
        A dict: the measured hand poses ``E_act`` and the commanded ones ``E_cmd`` (ticks), the recorded edge height at
        each tick ``edge_z`` (the physics step ``j`` that matches the tick), the world commands per physics step
        ``cmd_steps``, the block top ``s`` [m], the setup's knife ``knife_setup`` and the episode's own ``knife_episode``
        (None when its JSON has none).
    """
    path = Path(npz_path)
    setup = json.loads((path.parent / "setup.json").read_text())
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("z", setup["scene_changes"]["robot_yaw_deg"], degrees=True).as_matrix()
    base[:2, 3] = setup["scene_changes"]["robot_pos"][:2]
    data = np.load(path)
    t = data["t"].astype(float)
    steps = data["steps"].astype(float)
    j = np.abs(steps[None, :, 0] - t[:, None]).argmin(axis=1)
    tolerance = step_match_tolerance_s(t)
    if np.abs(steps[j, 0] - t).max() >= tolerance:
        raise ValueError(f"{path.name}: a tick has no physics step within {tolerance:.2e} s")
    # a step list in reverse order still matches every tick within the tolerance, so the order is checked on its own;
    # no assumption on the tick spacing (it need not be a whole number of physics steps)
    if np.any(np.diff(j) <= 0):
        raise ValueError(f"{path.name}: the matched physics steps do not increase with the ticks")
    state = data["state"].astype(float)
    block = data["block"].astype(float)
    episode_json = path.with_suffix(".json")
    episode_knife = json.loads(episode_json.read_text()).get("knife_in_eef") if episode_json.exists() else None
    return {
        "path": path,
        "E_act": base @ hand_poses(state[:, 9:12], state[:, 12:16]),
        "E_cmd": base @ np.stack([G.pose9(c) for c in data["cmd"].astype(float)]),
        "j": j,
        "edge_z": steps[j, 6],
        "cmd_steps": data["cmd_steps"].astype(float),
        "s": float(block[2] - block[6] / 2.0),
        "knife_setup": np.asarray(setup["grasp"]["knife_in_eef"], float),
        "knife_episode": None if episode_knife is None else np.asarray(episode_knife, float),
    }


def edge_residual_mm(demo, knife):
    """Predicted minus recorded knife-edge height at every tick, in mm."""
    return 1e3 * (lowest_edge_z(demo["E_act"], knife) - demo["edge_z"])


def fitted_knife(folder, knife_setup, fit_episodes):
    """A knife fitted to the folder's first episodes that the setup grasp does not explain.

    Args:
        folder: the demonstration folder (holds the ``ep_XXXXX.npz`` files and ``setup.json``).
        knife_setup: the setup's knife, which the episodes it explains are left out of the fit for.
        fit_episodes: how many episodes, in name order, the fit uses.

    Returns:
        A dict: the fitted ``knife``, the ``episodes`` (names) it was fitted to and the fit's ``fit_rms_mm``.
    """
    eefs, edges, names = [], [], []
    for npz in sorted(folder.glob("ep_*.npz")):
        other = load_demo(npz)
        if rms(edge_residual_mm(other, knife_setup)) < SETUP_TOL_MM:
            continue
        eefs.append(other["E_act"])
        edges.append(other["edge_z"])
        names.append(npz.name)
        if len(names) == fit_episodes:
            break
    if not names:
        raise ValueError(f"{folder}: the setup grasp explains every episode, so there is no knife to fit")
    knife, fit_m = fit_knife_in_eef(np.concatenate(eefs), np.concatenate(edges), knife_setup)
    return {"knife": knife, "episodes": names, "fit_rms_mm": 1e3 * fit_m}


def choose_knife(demo, fit_episodes, cache):
    """The knife for one demonstration, where it came from, and the fit it came from (None unless fitted).

    Args:
        demo: a dict from ``load_demo``.
        fit_episodes: passed to ``fitted_knife``.
        cache: the folder -> fit dict of ``fitted_knife``, shared by the demonstrations of one run.

    Returns:
        A tuple (knife, source, fit).
    """
    if demo["knife_episode"] is not None:
        return demo["knife_episode"], "episode", None
    if rms(edge_residual_mm(demo, demo["knife_setup"])) < SETUP_TOL_MM:
        return demo["knife_setup"], "setup", None
    folder = demo["path"].parent
    if folder not in cache:
        cache[folder] = fitted_knife(folder, demo["knife_setup"], fit_episodes)
    return cache[folder]["knife"], "fitted", cache[folder]


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
