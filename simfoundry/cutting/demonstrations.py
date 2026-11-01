# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reading recorded tofu demonstrations, and choosing each one's knife, shared by the round-trip check and the converter.

A demonstration is an ``ep_XXXXX.npz`` written by scripts/cutting/tofu_policy_episodes.py, read with its folder's
``setup.json`` and, when present, its ``ep_XXXXX.json``. Only numpy is used at import time (scipy is needed only for
``fit_knife_in_eef``, which imports it when it runs), so the converter can use this module in its own environment.
"""

import json
from pathlib import Path

import numpy as np

from simfoundry.cutting.support_height import fit_knife_in_eef, lowest_edge_z

# provisional: the setup grasp explains its own episode to about 7 um and the other episodes of the 10-27 demos to about
# 100 mm, so any threshold between those two groups gives the same answer; 0.05 mm sits between them with room to spare
SETUP_TOL_MM = 0.05


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


def pose_from_rot6(a9):
    """The 4x4 pose of a 9-vector: position ``a9[:3]`` and the first two rotation columns ``a9[3:6]``, ``a9[6:9]``.

    The columns are orthonormalised by Gram-Schmidt. This is the generator's ``pose9``, with the same operations in the
    same order, so the two give bit-identical poses; the converter and the generator must agree exactly.

    Args:
        a9: 9 numbers: a position, then two rotation columns.

    Returns:
        The (4, 4) pose.
    """
    a = np.asarray(a9, dtype=float)
    b1 = a[3:6] / np.linalg.norm(a[3:6])
    b2 = a[6:9] - (b1 @ a[6:9]) * b1
    b2 /= np.linalg.norm(b2)
    P = np.eye(4)
    P[:3, :3] = np.stack([b1, b2, np.cross(b1, b2)], axis=1)
    P[:3, 3] = a[:3]
    return P


def hand_poses(position, xyzw):
    """Poses (N, 4, 4) from positions (N, 3) and xyzw quaternions (N, 4), in whatever frame the inputs are in.

    Why this arithmetic: the quaternions are normalised, then the matrix is written with the diagonal in the
    x^2 - y^2 - z^2 + w^2 form, which is what ``scipy.spatial.transform.Rotation.from_quat(...).as_matrix()`` computes:
    equal to rounding (<= 1e-15 here; bit for bit on the machine this was written on). This form is a convenience, not
    an accuracy condition. The knife fit amplifies a difference of that size along the directions the data does not
    determine, so keeping scipy's form makes the printed round trips come out the same as with scipy. The other usual
    form, cosim's ``quat_xyzw_to_matrix``, differs by up to 7e-16 here. scipy is not imported, so the converter can use
    this without it.

    Args:
        position: (N, 3) positions [m].
        xyzw: (N, 4) quaternions, x, y, z, w order (normalised here).

    Returns:
        The (N, 4, 4) poses.
    """
    position = np.asarray(position, dtype=float)
    q = np.asarray(xyzw, dtype=float)
    q = q / np.sqrt(np.sum(q * q, axis=1))[:, None]
    x, y, z, w = q.T
    P = np.tile(np.eye(4), (len(position), 1, 1))
    P[:, 0, 0] = x * x - y * y - z * z + w * w
    P[:, 1, 1] = -x * x + y * y - z * z + w * w
    P[:, 2, 2] = -x * x - y * y + z * z + w * w
    P[:, 0, 1] = 2 * (x * y - z * w)
    P[:, 1, 0] = 2 * (x * y + z * w)
    P[:, 0, 2] = 2 * (x * z + y * w)
    P[:, 2, 0] = 2 * (x * z - y * w)
    P[:, 1, 2] = 2 * (y * z - x * w)
    P[:, 2, 1] = 2 * (y * z + x * w)
    P[:, :3, 3] = position
    return P


def base_pose(setup):
    """The robot base pose from a demonstration's ``setup.json``: yaw ``robot_yaw_deg`` about z, at ``robot_pos`` (z 0).

    Args:
        setup: the parsed ``setup.json``.

    Returns:
        The (4, 4) base pose in the solver's world frame. The base is level, so its z axis is up.
    """
    yaw = np.radians(setup["scene_changes"]["robot_yaw_deg"])
    c, s = np.cos(yaw), np.sin(yaw)
    B = np.eye(4)
    B[:2, :2] = [[c, -s], [s, c]]
    B[:2, 3] = np.asarray(setup["scene_changes"]["robot_pos"], dtype=float)[:2]
    return B


def rms(values):
    """Root-mean-square of ``values``."""
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

    Raises:
        ValueError: If a tick has no physics step within the tolerance, or the matched steps do not increase.
    """
    path = Path(npz_path)
    setup = json.loads((path.parent / "setup.json").read_text())
    base = base_pose(setup)
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
        "E_cmd": base @ np.stack([pose_from_rot6(c) for c in data["cmd"].astype(float)]),
        "j": j,
        "edge_z": steps[j, 6],
        "cmd_steps": data["cmd_steps"].astype(float),
        "s": float(block[2] - block[6] / 2.0),
        "knife_setup": np.asarray(setup["grasp"]["knife_in_eef"], float),
        "knife_episode": None if episode_knife is None else np.asarray(episode_knife, float),
    }


def edge_residual_mm(demo, knife):
    """Predicted minus recorded knife-edge height at every tick, in mm.

    Args:
        demo: a dict from ``load_demo``.
        knife: the (4, 4) knife-in-hand transform to test.

    Returns:
        (N,) residuals [mm].
    """
    return 1e3 * (lowest_edge_z(demo["E_act"], knife) - demo["edge_z"])


def fitted_knife(folder, knife_setup, fit_episodes):
    """A knife fitted to the folder's first episodes that the setup grasp does not explain.

    Args:
        folder: the demonstration folder (holds the ``ep_XXXXX.npz`` files and ``setup.json``).
        knife_setup: the setup's knife, which the episodes it explains are left out of the fit for.
        fit_episodes: how many episodes, in name order, the fit uses.

    Returns:
        A dict: the fitted ``knife``, the ``episodes`` (names) it was fitted to and the fit's ``fit_rms_mm``.

    Raises:
        ValueError: If the setup grasp explains every episode in the folder, so there is nothing to fit.
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

    The first source that explains the recorded edge heights wins: the episode's own JSON, then the setup's grasp, then
    a knife fitted to the folder (once per folder: ``cache`` keeps it for the other demonstrations of that folder).

    Args:
        demo: a dict from ``load_demo``.
        fit_episodes: passed to ``fitted_knife``.
        cache: the folder -> fit dict of ``fitted_knife``, shared by the demonstrations of one run.

    Returns:
        A tuple (knife, source, fit), with source one of "episode", "setup", "fitted".
    """
    if demo["knife_episode"] is not None:
        return demo["knife_episode"], "episode", None
    if rms(edge_residual_mm(demo, demo["knife_setup"])) < SETUP_TOL_MM:
        return demo["knife_setup"], "setup", None
    folder = demo["path"].parent
    if folder not in cache:
        cache[folder] = fitted_knife(folder, demo["knife_setup"], fit_episodes)
    return cache[folder]["knife"], "fitted", cache[folder]
