# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A synthetic demonstration folder laid out like the recorded ones, for the tests of the reading and round-trip code.

Nothing here reads real data, and nothing needs scipy, so the converter's tests can use it in its own environment.
"""

import json

import numpy as np

from simfoundry.cutting.cosim import matrix_to_quat_xyzw
from simfoundry.cutting.demonstrations import base_pose, pose_from_rot6
from simfoundry.cutting.support_height import lowest_edge_z

TICKS = 40
SPT = 8  # physics steps per tick
BASE_YAW_DEG = 20.0
BASE_POS = [-0.7, -0.06, 0.0]


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotvec_matrix(v):
    """The rotation matrix of a rotation vector, by Rodrigues' formula."""
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    k = v / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _transform(rotvec, translation):
    T = np.eye(4)
    T[:3, :3] = rotvec_matrix(rotvec)
    T[:3, 3] = translation
    return T


K1 = _transform([0.1, 2.1, 0.2], [0.14, 0.0, 0.10])  # the setup's grasp
K2 = K1 @ _transform([0.245, 0.0, 0.0], [0.01, -0.01, 0.025])  # a regrasp: 14 deg about its own x, 2.5 cm along its z


def robot_base():
    """The base of the synthetic folders, from the same setup values they write."""
    return base_pose({"scene_changes": {"robot_pos": BASE_POS, "robot_yaw_deg": BASE_YAW_DEG}})


def base_frame_poses(rng, n):
    """Hand poses in the base frame: knife tilted 20-40 deg, yaw +-15 deg, roll +-5 deg, height 0.012-0.08 m."""
    P = np.tile(np.eye(4), (n, 1, 1))
    yaw = np.radians(rng.uniform(-15, 15, n))
    tilt = np.radians(rng.uniform(20, 40, n))
    roll = np.radians(rng.uniform(-5, 5, n))
    for i in range(n):
        P[i, :3, :3] = _rot_z(yaw[i]) @ _rot_y(tilt[i]) @ _rot_x(roll[i])
    P[:, 0, 3] = 0.3 + rng.uniform(-0.05, 0.05, n)
    P[:, 1, 3] = rng.uniform(-0.05, 0.05, n)
    P[:, 2, 3] = rng.uniform(0.012, 0.08, n)
    return P


def edit_npz(path, **changes):
    """Rewrites one episode archive with some of its arrays replaced."""
    with np.load(path) as archive:
        data = dict(archive)
    data.update(changes)
    np.savez(path, **data)


def write_folder(folder, *, n_eps=12, regrasp_from=1, bias_m=None, below=(), json_knife=(), t0=5.0):
    """A folder laid out like a recorded demonstration set: setup.json and ep_XXXXX.npz, some with an ep_XXXXX.json.

    Episodes from ``regrasp_from`` on hold K2; the others hold K1, the setup's. ``bias_m`` adds a constant error to an
    episode's recorded edge heights. Episodes in ``below`` get a support plane (the block's bottom, the solver board
    plane) at the median edge height, so half their poses have the edge under the support. Episodes in ``json_knife``
    write their knife into their JSON. ``t0`` is the time of the first tick [s]; the physics steps are stored as
    float32, as in the recorded files.
    """
    folder.mkdir(parents=True, exist_ok=True)
    base = robot_base()
    (folder / "setup.json").write_text(json.dumps({
        "scene_changes": {"robot_pos": BASE_POS, "robot_yaw_deg": BASE_YAW_DEG},
        "grasp": {"knife_in_eef": K1.tolist()},
    }))
    rng = np.random.default_rng(0)
    bias_m = bias_m or {}
    for ep in range(n_eps):
        knife = K2 if ep >= regrasp_from else K1
        P = base_frame_poses(rng, TICKS)
        noise = rng.normal(0.0, 1e-4, TICKS) if ep >= 1 else np.zeros(TICKS)  # episode 0 is exact: the setup fits it
        edge = lowest_edge_z(base @ P, knife) + bias_m.get(ep, 0.0) + noise
        cmd_pos = P[:, :3, 3] + rng.uniform(-0.002, 0.002, (TICKS, 3))  # the command misses the hand by mm
        cmd = np.concatenate([cmd_pos, P[:, :3, 0], P[:, :3, 1]], axis=1)
        E_cmd = base @ np.stack([pose_from_rot6(c) for c in cmd])
        s = float(np.median(edge)) if ep in below else float(edge.min()) - 0.005
        block = np.array([0.5, 0.0, s + 0.015, 0.1, 0.1, 0.1, 0.03], dtype=np.float32)
        state = np.zeros((TICKS, 16), np.float32)
        state[:, 9:12] = P[:, :3, 3]
        state[:, 12:16] = [matrix_to_quat_xyzw(R) for R in P[:, :3, :3]]
        steps = np.zeros((TICKS * SPT, 8))
        steps[:, 0] = t0 + 0.1 * np.arange(TICKS * SPT) / SPT
        steps[:, 6] = np.repeat(edge, SPT)
        steps[:, 7] = 0.011
        np.savez(folder / f"ep_{ep:05d}.npz", state=state, cmd=cmd.astype(np.float32),
                 t=t0 + 0.1 * np.arange(TICKS), block=block, cmd_steps=np.repeat(E_cmd, SPT, axis=0).astype(np.float32),
                 steps=steps.astype(np.float32))
        if ep in json_knife:
            (folder / f"ep_{ep:05d}.json").write_text(json.dumps({"knife_in_eef": knife.tolist()}))
    return folder
