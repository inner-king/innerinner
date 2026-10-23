# SPDX-License-Identifier: Apache-2.0
"""A Franka in the Chef'd OmniGibson scene cuts the MPM tofu, the solver running inside Isaac Sim.

Pipeline A built an OmniGibson scene from one frame of the Chef'd clip (stage 14: the Franka, a
board, two plates, a paper towel, a knife and the tofu, all rigid). This script leaves the rigid
tofu out and cuts the MPM tofu in its place, filled and placed as scripts/cutting/tofu_real2sim.py
places it: its level box fitted to its outline in the frame, its bottom on the board's top as the
depth map measures it. The Franka holds a chef's knife whose body has the solver's blade
(og_cosim.write_knife_usd), and the solver is coupled to PhysX in the same process
(simfoundry/cutting/cosim.py, og_cosim.py): after every physics step the tofu advances with the
knife moving as PhysX moved its body, and before the next step the tofu's force and moment act on
that body. The tofu shows as a points prim, a sphere per particle.

Strokes (--stroke):
  cook   the cook's stroke from the clip (real2sim.cook_stroke, as tofu_real2sim.py cuts): the
         blade tilted 30 degrees, pushed forward as it comes down and as it rocks flat on its
         belly, then drawn back towards the handle. The arm follows it with an IK controller on
         its stiff position drives, so the knife goes where it is told and the tofu's force shows
         as the change it makes in the arm's joint torques (compare a --no-wrench run with
         scripts/cutting/tofu_og_torques.py).
  press  a level blade pressed straight down with a set force (--push-force; several make
         several presses, each into a fresh block): a force-limited stroke. The arm runs an
         operational-space (impedance) controller whose goal is put, every step, where its
         spring and the arm's inertia give that force down and hold the blade's pose otherwise;
         the tofu's force on the knife slows it, stalls it or lets it through.

The scene is changed in memory only (scene_edited.json in --out shows how):
- the rigid tofu, the reconstructed knife and the paper towel are left out (--clear: drawing the
  knife back as the cook does, the Panda's hand, 20 cm across its fingers, sweeps the towel off its
  plate);
- the board, whose colliders stand 4.2 cm under the cut where the depth map puts its top at 1.3 cm,
  is levelled and scaled to that height (its colliders, not its visual mesh, are what the knife meets);
- the robot's base moves behind the board, facing the clip's camera, where the cook stands
  (stage 14's base is 0.84 m from the tofu, and nearer on that side the IK stalls; --robot-side);
- its arm takes absolute poses, through IK (cook) or the OSC (press).
The knife waits on the table and is put between the open fingers (gravity off) until assisted
grasping holds it, as repro/fruit_slicing/check_runtime_cut.py does; its weight stays off afterwards,
the arm carrying it as it carries its own links (OmniGibson turns gravity off on a robot's moving
links).

Example (repository root; boots Isaac Sim and needs the GPU, which is shared: see --allow-busy-gpu;
a cook run takes about 140 s on an RTX 3060, of which 96 s cutting):
    PYTHONPATH=$PWD OMNIGIBSON_DATA_PATH=/home/inner/real2sim/SimFoundry/deps/BEHAVIOR-1K/datasets \\
        ~/miniforge3/envs/simfoundry/bin/python scripts/cutting/tofu_og_cut.py --stroke cook \\
        --reference Data/tofu_reference/chefd_side.mp4 --out results/tofu_og/cook
"""

import argparse
import csv
import json
import math
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from simfoundry.cutting.cosim import KnifeCoupling, matrix_to_quat_xyzw, pose_from_axes, quat_xyzw_to_matrix
from simfoundry.cutting.knife import CHEF_KNIFE, KnifePose
from simfoundry.cutting.materials import PRESETS
from simfoundry.cutting.real2sim import cook_stroke, fit_food, load_reconstruction
from simfoundry.cutting.render import CUT_RGB, tofu_look

SCENE_JSON = "s14_og/reconstructed_og_scene.json"
KNIFE = "knife"
#: Where the knife waits on the table before it is put in the hand: clear of the board and the plates.
PARK = (0.55, -0.25)
#: The Panda gripper's command (MultiFingerGripperController, smooth mode): 1 opens it, 0 closes it.
OPEN, CLOSE = 1.0, 0.0
#: The blade's edge stops this far above the board: the arm's drives would push hard on a jam.
BOARD_GAP = 1.0e-3


def gpu_busy_reason():
    """Why the shared GPU should not be used now, or None (nvidia-smi only)."""
    try:
        util = []
        for _ in range(3):
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10, check=True).stdout.split()
            util.append(float(out[0]))
            time.sleep(0.5)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
        return f"nvidia-smi failed: {exc}"
    if max(util) > 10.0:
        return f"GPU utilisation {max(util):.0f}%"
    others = [a for a in apps if a and not any(s in a.lower() for s in ("comfyui", "ollama"))]
    return "other GPU processes: " + "; ".join(others) if others else None


# ---------------------------------------------------------------------------------------------- geometry
def np_(t):
    return t.detach().cpu().numpy().astype(np.float64) if hasattr(t, "detach") else np.asarray(t, np.float64)


def pose_matrix(pos, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_matrix(quat_xyzw)
    T[:3, 3] = pos
    return T


def rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def knife_world(pose: KnifePose, W):
    """4x4 world pose of the knife body (its frame is the solver's knife frame) for a solver-frame pose."""
    T = np.eye(4)
    T[:3, :3] = W[:3, :3] @ np.stack(pose.axes(), axis=1)
    T[:3, 3] = W[:3, :3] @ np.asarray(pose.origin) + W[:3, 3]
    return T


def interpolate(T0, T1, s):
    """Pose between two 4x4 poses: position linear, orientation by the shortest rotation."""
    from scipy.spatial.transform import Rotation

    d = Rotation.from_matrix(T1[:3, :3] @ T0[:3, :3].T).as_rotvec()
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(s * d).as_matrix() @ T0[:3, :3]
    T[:3, 3] = (1.0 - s) * T0[:3, 3] + s * T1[:3, 3]
    return T


def min_jerk(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def rotation_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def angles_deg(T):
    """Yaw, pitch and roll of a knife pose in degrees, as ``KnifePose`` counts them (world frame)."""
    p = pose_from_axes(T[:3, 3], T[:3, 0], T[:3, 1], T[:3, 2])
    return math.degrees(p.yaw), math.degrees(p.pitch), math.degrees(p.roll)


def look_at(eye, target):
    """xyzw quaternion of a USD camera (looking down -z, +y up) at ``eye`` facing ``target``."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return matrix_to_quat_xyzw(np.stack([right, up, -f], axis=1))


# ------------------------------------------------------------------------------------------------ the scene
def edit_scene(scene, *, rec, fitted, robot_pos, robot_yaw, arm_controller, knife_usd, board_hulls, cut_xy,
               clear=()):
    """The stage 14 scene with the changes the module docstring lists, and what was changed.

    ``board_hulls`` are the board's colliders in its link frame (``collider_hulls``): its top where
    the knife meets it is the colliders' top along the cut (``cut_xy``, world), which stands
    a couple of millimetres above its visual mesh.
    """
    init = scene["objects_info"]["init_info"]
    reg = scene["state"]["registry"]["object_registry"]
    knife_name = rec.find("knife").name
    removed = [fitted.food.name, knife_name] + [o.name for o in rec.objects.values()
                                                if any(c in o.category for c in clear)]
    for name in removed:
        init.pop(name, None)
        reg.pop(name, None)
    # The board: levelled (a reconstructed board settles a fraction of a degree off), its height
    # scaled so that its colliders' highest point under the cut is at the depth map's height for
    # its top, and their bottom on the table.
    board = fitted.board.name
    root = reg[board]["root_link"]
    p, R = np.asarray(root["pos"], float), quat_xyzw_to_matrix(root["ori"])
    yaw = math.atan2(R[1, 0], R[0, 0])
    scale0 = np.asarray(init[board]["args"].get("scale", [1.0, 1.0, 1.0]), float)
    hulls = [h * scale0[None, :] for h in board_hulls]
    local_xy = (np.asarray(cut_xy, float) - p[None, :2]) @ rz(yaw)[:2, :2]  # rows: R^T (xy - p)
    from simfoundry.cutting.og_cosim import hull_top

    under = hull_top(hulls, local_xy)
    top, bottom = float(np.nanmax(under)), float(min(h[:, 2].min() for h in hulls))
    s = fitted.board_z / (top - bottom)
    init[board]["args"]["scale"] = [float(scale0[0]), float(scale0[1]), float(scale0[2] * s)]
    reg[board]["root_link"] = {"pos": [float(p[0]), float(p[1]), -s * bottom],
                               "ori": [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)],
                               "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]}
    # The robot: moved within reach; zero velocities (a repositioned object, AGENTS.md rule 3).
    robot = reg["robot0"]
    robot["root_link"] = {"pos": [float(v) for v in robot_pos],
                          "ori": [0.0, 0.0, math.sin(0.5 * robot_yaw), math.cos(0.5 * robot_yaw)],
                          "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]}
    robot["joint_vel"] = [0.0] * len(robot["joint_vel"])
    robot["controllers"]["arm_0"] = {"goal_is_valid": False, "goal": None}
    args = init["robot0"]["args"]
    args["controller_config"]["arm_0"] = arm_controller
    args["obs_modalities"] = ["proprio"]  # no robot cameras to render
    # The knife, lying flat on the table.
    init[KNIFE] = {"class_module": "omnigibson.objects.usd_object", "class_name": "USDObject",
                   "args": {"name": KNIFE, "usd_path": knife_usd, "category": "knife", "fixed_base": False}}
    flat = matrix_to_quat_xyzw(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]))
    reg[KNIFE] = {"is_asleep": False, "non_kin": {},
                  "root_link": {"pos": [PARK[0], PARK[1], 0.012], "ori": flat.tolist(),
                                "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]}}
    return {"removed": removed, "board": board, "board_yaw_deg": math.degrees(yaw),
            "board_scale_z": s, "board_collider_top_before_m": float(p[2] + top),
            "board_collider_top_spread_mm": 1e3 * float(np.nanmax(under) - np.nanmin(under)),
            "board_top_m": fitted.board_z, "robot_pos": [float(v) for v in robot_pos],
            "robot_yaw_deg": math.degrees(robot_yaw)}


def osc_gains(kp, kp_z):
    """The OSC's stiffness per axis of the robot's base frame (x, y, z, then rotations).

    Soft along z, the pushing axis (the base stands upright): the OSC damps critically, kd = 2 sqrt(kp),
    and an arm whose apparent mass at the hand is 6-9 kg, damped so at the stiffness that holds the
    blade's pose, would sink at a millimetre a second. They are set on the controller once it exists
    (``set_osc_gains``): from a list in its config, or with a damping ratio per axis, OmniGibson's OSC
    mixes float64 into its float32 torque kernel, which numba then refuses.
    """
    return [kp, kp, kp_z, kp, kp, kp]


def set_osc_gains(robot, gains):
    ctrl = robot.controllers[f"arm_{robot.default_arm}"]
    ctrl.kp = np.asarray(gains, dtype=np.float32)
    return ctrl


def arm_controller_config(kind, kp, kp_z):
    if kind == "ik":
        # Absolute poses, no smoothing: the stroke is a trajectory to follow, not a teleoperator's deltas.
        return {"name": "InverseKinematicsController", "mode": "absolute_pose", "command_input_limits": None,
                "command_output_limits": None, "smoothing_filter_size": None}
    return {"name": "OperationalSpaceController", "mode": "absolute_pose", "command_input_limits": None,
            "command_output_limits": None, "kp": kp, "damping_ratio": 1.0,
            "decouple_pos_ori": False, "use_gravity_compensation": False, "use_cc_compensation": True}


# ------------------------------------------------------------------------------------------------ the robot
class Arm:
    """End-effector commands through the arm's absolute-pose controller, in the world frame."""

    def __init__(self, robot):
        self.robot = robot
        self.arm = robot.default_arm
        p, q = robot.get_position_orientation()
        self.base = pose_matrix(np_(p), np_(q))
        self.base_inv = np.linalg.inv(self.base)
        self.arm_idx = np_(robot.arm_control_idx[self.arm]).astype(int)

    def eef(self):
        p, q = self.robot.eef_links[self.arm].get_position_orientation()
        return pose_matrix(np_(p), np_(q))

    def action(self, E_world, gripper):
        import torch as th
        from scipy.spatial.transform import Rotation

        E = self.base_inv @ E_world
        rv = Rotation.from_matrix(E[:3, :3]).as_rotvec()
        return th.tensor(np.concatenate([E[:3, 3], rv, [gripper]]), dtype=th.float32)

    def task_inertia(self):
        """The operational-space inertia (6x6, robot base frame) the OSC computes its wrench with."""
        cd = self.robot.get_control_dict()
        M = np_(cd["mass_matrix"])[np.ix_(self.arm_idx, self.arm_idx)]
        J = np_(cd[f"eef_{self.arm}_jacobian_relative"])[:, self.arm_idx]
        return np.linalg.inv(J @ np.linalg.inv(M) @ J.T)

    def jacobian(self):
        cd = self.robot.get_control_dict()
        return np_(cd[f"eef_{self.arm}_jacobian_relative"])[:, self.arm_idx]

    def efforts(self):
        return np_(self.robot.get_joint_efforts())[self.arm_idx]

    def joints(self):
        return np_(self.robot.get_joint_positions())[self.arm_idx]

    def fingers(self):
        return np_(self.robot.get_joint_positions())[np_(self.robot.gripper_control_idx[self.arm]).astype(int)]


class Run:
    """The co-simulation loop: one physics step at a time, a frame every few."""

    def __init__(self, og, env, arm, knife, coupler, cams, out, fps, fields):
        self.og, self.env, self.arm, self.knife, self.coupler = og, env, arm, knife, coupler
        self.cams, self.out, self.fps = cams, out, fps
        self.dt = og.sim.get_physics_dt()
        self.every = max(1, int(round(1.0 / (fps * self.dt))))
        self.k = 0
        self.rows = []
        self.fields = fields
        self.phase = "start"
        self.writers = {}
        self.frame_times = []
        self.points = None
        self.world_from_sim = None
        self.hud = ""
        self.K_in_E = None
        self.reference = None  # frames of the clip, from reference_start, at fps
        self.reference_start = 0.0

    def knife_pose(self):
        from simfoundry.cutting.og_cosim import rigid_body_pose

        return pose_matrix(*rigid_body_pose(self.knife.root_link.prim_path))

    def step(self, E_cmd, gripper, K_cmd=None, extra=None):
        og = self.og
        action = self.arm.action(E_cmd, gripper)
        n0 = len(self.coupler.records) if self.coupler is not None else 0
        t0 = time.perf_counter()
        self.arm.robot.apply_action(action)
        with og.sim.render_on_step(False):
            og.sim.step()
        wall = time.perf_counter() - t0
        self.k += 1
        K = self.knife_pose()
        rec = {"k": self.k, "t": round(self.k * self.dt, 6), "phase": self.phase, "step_wall_s": wall}
        if self.coupler is not None and len(self.coupler.records) > n0:
            r = self.coupler.records[-1]
            rec.update(solver_t=r["solver_t"], advanced=r["advanced"], cut=r["cut"], mpm_wall_s=r["wall_s"],
                       fx=r["force"][0], fy=r["force"][1], fz=r["force"][2],
                       mx=r["moment"][0], my=r["moment"][1], mz=r["moment"][2])
        rec.update(kx=K[0, 3], ky=K[1, 3], kz=K[2, 3])
        rec.update(zip(("kyaw", "kpitch", "kroll"), angles_deg(K)))
        if K_cmd is not None:
            rec.update(zip(("cyaw", "cpitch", "croll"), angles_deg(K_cmd)))
        E = self.arm.eef()
        rec.update(ex=E[0, 3], ey=E[1, 3], ez=E[2, 3], ecx=E_cmd[0, 3], ecy=E_cmd[1, 3], ecz=E_cmd[2, 3],
                   eef_err_mm=1e3 * float(np.linalg.norm(E[:3, 3] - E_cmd[:3, 3])),
                   eef_ang_err_deg=rotation_deg(E[:3, :3], E_cmd[:3, :3]))
        q = self.arm.joints()
        rec.update({f"q{i}": float(v) for i, v in enumerate(q)})
        if self.K_in_E is not None:
            D = np.linalg.inv(self.K_in_E) @ np.linalg.inv(E) @ K
            rec.update(hand_drift_mm=1e3 * float(np.linalg.norm(D[:3, 3])),
                       hand_drift_deg=rotation_deg(D[:3, :3], np.eye(3)))
        if K_cmd is not None:
            rec.update(cx=K_cmd[0, 3], cy=K_cmd[1, 3], cz=K_cmd[2, 3],
                       pos_err_mm=1e3 * float(np.linalg.norm(K[:3, 3] - K_cmd[:3, 3])),
                       ang_err_deg=rotation_deg(K[:3, :3], K_cmd[:3, :3]))
        if "efforts" in self.fields:
            rec["efforts"] = self.arm.efforts().tolist()
        if "jacobian" in self.fields:
            rec["jacobian"] = self.arm.jacobian().tolist()
            rec["eef"] = E[:3, 3].tolist()
        if extra:
            rec.update(extra)
        self.rows.append(rec)
        if self.k % self.every == 0:
            self.frame()
        return K

    def frame(self):
        import imageio.v2 as imageio

        og = self.og
        row = self.rows[-1] if self.rows else {}
        t0 = time.perf_counter()
        if self.points is not None:
            x = self.coupler.coupling.sim.positions()
            self.points.update(x @ self.world_from_sim[:3, :3].T + self.world_from_sim[:3, 3])
        t1 = time.perf_counter()
        og.sim.render()
        t2 = time.perf_counter()
        for name, cam in self.cams.items():
            rgb = np_(cam.get_obs()[0]["rgb"])[:, :, :3].astype(np.uint8)
            rgb = label(rgb, self.hud) if self.hud else rgb
            if name not in self.writers:
                self.writers[name] = imageio.get_writer(str(self.out / f"{name}.mp4"), fps=self.fps,
                                                        codec="libx264", quality=8, macro_block_size=8)
            self.writers[name].append_data(rgb)
            if name == "chefd_view" and self.reference and "traj_t" in row:
                i = min(int(round(row["traj_t"] * self.fps)), len(self.reference) - 1)
                clip_t = self.reference_start + row["traj_t"]
                pair = np.concatenate([label(self.reference[i], f"video  {clip_t:5.2f} s"), rgb], axis=1)
                if "side_by_side" not in self.writers:
                    self.writers["side_by_side"] = imageio.get_writer(
                        str(self.out / "side_by_side.mp4"), fps=self.fps, codec="libx264", quality=8,
                        macro_block_size=8)
                self.writers["side_by_side"].append_data(pair)
            if (self.k // self.every) % (2 * self.fps) == 0:
                (self.out / "frames").mkdir(exist_ok=True)
                imageio.imwrite(self.out / "frames" / f"{name}_{self.k:05d}.png", rgb)
        self.frame_times.append({"points_s": t1 - t0, "render_s": t2 - t1, "read_s": time.perf_counter() - t2})

    def close(self):
        for w in self.writers.values():
            w.close()


def clip_frames(path, start, fps, size, count):
    """``count`` frames of a video from ``start`` [s] at ``fps``, scaled to ``size`` (width, height)."""
    import imageio_ffmpeg

    w, h = size
    gen = imageio_ffmpeg.read_frames(str(path), input_params=["-ss", f"{start:.3f}"],
                                     output_params=["-vf", f"fps={fps},scale={w}:{h}"])
    next(gen)  # metadata
    frames = []
    for raw in gen:
        frames.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
        if len(frames) >= count:
            break
    gen.close()
    return frames


def label(img, text):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, img.shape[0] // 26))
    except OSError:
        font = ImageFont.load_default()
    d.text((12, 8), text, fill=(20, 20, 20), font=font)
    return np.asarray(im)


def plot(rows, out, reference_csv=None):
    """knife_force.png: the tofu's force on the knife over each stroke (and the edge's height for a press)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def col(rs, k):
        return np.array([r.get(k, np.nan) if r.get(k) is not None else np.nan for r in rs], dtype=float)

    cook = [r for r in rows if "traj_t" in r]
    pushes = sorted({r["push_n"] for r in rows if "push_n" in r})
    n = (1 if cook else 0) + (1 if pushes else 0)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.2 * n), squeeze=False)
    ax_i = 0
    if cook:
        ax = axes[ax_i, 0]
        ax_i += 1
        t = col(cook, "traj_t")
        ax.plot(t, col(cook, "fz"), color="tab:blue", lw=0.8, label="Isaac Sim + MPM: Fz")
        ax.plot(t, np.hypot(col(cook, "fx"), col(cook, "fy")), color="tab:orange", lw=0.6,
                label="Isaac Sim + MPM: horizontal |Fx, Fy|")
        if reference_csv is not None and Path(reference_csv).exists():
            ref = list(csv.DictReader(open(reference_csv)))
            ax.plot([float(r["t"]) for r in ref], [float(r["fz"]) for r in ref], color="0.2", lw=1.2,
                    label="tofu_real2sim.py (scripted knife): Fz")
        ax.set_xlabel("stroke time [s]")
        ax.set_ylabel("force of the tofu on the knife [N]")
        ax.legend(fontsize=8)
    if pushes:
        ax = axes[ax_i, 0]
        ax2 = ax.twinx()
        for i, push in enumerate(pushes):
            rs = [r for r in rows if r.get("push_n") == push]
            c = f"C{i}"
            ax.plot(col(rs, "press_t"), col(rs, "edge_mm"), color=c, label=f"push {push:.1f} N: edge height")
            ax2.plot(col(rs, "press_t"), col(rs, "fz"), color=c, lw=0.4, alpha=0.6)
        ax.set_xlabel("press time [s]")
        ax.set_ylabel("edge above the board [mm]")
        ax2.set_ylabel("Fz of the tofu on the knife [N] (thin)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "knife_force.png", dpi=130)
    plt.close(fig)


def grasp_knife(run, knife, robot, arm):
    """Puts the knife's handle between the open fingers, blade down and tip forward, and closes them.

    The knife's gravity is off from here on (see the module docstring). Returns the grasp's record.
    """
    import torch as th
    from scipy.spatial.transform import Rotation
    from simfoundry.cutting.og_cosim import handle_centre

    run.phase = "grasp"
    hold = arm.eef()
    for _ in range(30):
        run.step(hold, OPEN)

    def ends(points):
        out = []
        for gp in points:
            p, q = robot.links[gp.link_name].get_position_orientation()
            out.append(np_(p) + Rotation.from_quat(np_(q)).as_matrix() @ np_(gp.position))
        return np.mean(out, axis=0)

    s, e = ends(robot.assisted_grasp_start_points[arm.arm]), ends(robot.assisted_grasp_end_points[arm.arm])
    centre, y = (s + e) / 2.0, (e - s) / np.linalg.norm(e - s)
    z = np.array([0.0, 0.0, 1.0]) - y[2] * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    if x[:2] @ (centre - arm.base[:3, 3])[:2] < 0.0:
        x, y = -x, -y
    R = np.stack([x, y, z], axis=1)
    placed = centre - R @ handle_centre()
    knife.root_link.disable_gravity()
    knife.set_position_orientation(position=th.tensor(placed, dtype=th.float32),
                                   orientation=th.tensor(matrix_to_quat_xyzw(R), dtype=th.float32))
    knife.set_linear_velocity(th.zeros(3))
    knife.set_angular_velocity(th.zeros(3))
    attached = None
    for i in range(120):
        run.step(hold, CLOSE)
        if robot.is_grasping(arm.arm, candidate_obj=knife) == run.grasp_true:
            attached = i + 1
            break
    for _ in range(30):
        run.step(hold, CLOSE)
    K = run.knife_pose()
    E = arm.eef()
    return {"attached_after_steps": attached, "held": robot.is_grasping(arm.arm, candidate_obj=knife) == run.grasp_true,
            "moved_while_closing_mm": 1e3 * float(np.linalg.norm(K[:3, 3] - placed)),
            "turned_while_closing_deg": rotation_deg(K[:3, :3], R), "fingers_m": arm.fingers().tolist(),
            "knife_in_eef": (np.linalg.inv(E) @ K).tolist()}


def move_knife(run, K_in_E, K_target, gripper, speed=0.15, ang_speed=0.8, settle=0.4, phase="approach"):
    """Takes the held knife to ``K_target`` along a minimum-jerk path, then holds it there for ``settle`` s."""
    run.phase = phase
    K0 = run.knife_pose()
    dist = float(np.linalg.norm(K_target[:3, 3] - K0[:3, 3]))
    ang = math.radians(rotation_deg(K0[:3, :3], K_target[:3, :3]))
    T = max(0.5, dist / speed, ang / ang_speed)
    n = int(math.ceil(T / run.dt))
    E_in_K = np.linalg.inv(K_in_E)
    for i in range(1, n + 1):
        K = interpolate(K0, K_target, min_jerk(i / n))
        run.step(K @ E_in_K, gripper, K_cmd=K)
    for _ in range(int(round(settle / run.dt))):
        run.step(K_target @ E_in_K, gripper, K_cmd=K_target)
    return {"duration_s": n * run.dt, "distance_m": dist}


# ------------------------------------------------------------------------------------------------ strokes
def cook(run, K_in_E, traj, W, lead=0.0, after=1.0):
    """The cook's stroke: the knife pose of the trajectory at every physics step."""
    E_in_K = np.linalg.inv(K_in_E)
    n = int(math.ceil(traj.duration / run.dt))
    cut_t = traj.cut_times[0]
    for i in range(n + 1):
        t = i * run.dt
        run.phase = "hover" if t < cut_t else "stroke"
        K_cmd = knife_world(traj.state(t + lead)[0], W)
        run.hud = f"cook's stroke   t {t:4.2f} s"
        run.step(K_cmd @ E_in_K, CLOSE, K_cmd=K_cmd, extra={"traj_t": t})
    run.phase = "after"
    K_end = knife_world(traj.state(traj.duration)[0], W)
    for _ in range(int(round(after / run.dt))):
        run.step(K_end @ E_in_K, CLOSE, K_cmd=K_end)


def press(run, arm, K_in_E, K_hover, push, board_z, gains, max_time=8.0, lift=0.03, after=1.0):
    """Presses the held knife down with ``push`` newtons through the OSC (see the module docstring).

    The OSC's static wrench on the end effector is Lambda Kp (goal - pose), Lambda its operational-space
    inertia and Kp its stiffness per axis (``gains``). The goal is put every step at the hover pose in
    x, y and orientation, which the OSC's stiffness then holds, plus Kp^-1 Lambda^-1 f, f the push: so
    the wrench is that hold plus exactly f. Solving for the height alone would leave Lambda's cross
    terms to push the blade sideways.
    """
    from scipy.spatial.transform import Rotation

    E_in_K = np.linalg.inv(K_in_E)
    E_hover = K_hover @ E_in_K
    Rb = arm.base[:3, :3]
    t0 = run.k * run.dt
    reached = None
    run.phase = "press"
    while True:
        t = run.k * run.dt - t0
        K = run.knife_pose()
        edge = float(K[2, 3])  # the level blade's straight edge
        if edge <= board_z + BOARD_GAP and reached is None:
            reached = t
        if reached is not None or t > max_time:
            break
        E = arm.eef()
        L = arm.task_inertia()
        # What the hover pose leaves in the base frame, the height left free.
        hold = np.zeros(6)
        hold[:3] = Rb.T @ (E_hover[:3, 3] - E[:3, 3])
        hold[2] = 0.0
        hold[3:] = Rotation.from_matrix(Rb.T @ E_hover[:3, :3] @ E[:3, :3].T @ Rb).as_rotvec()
        f = np.zeros(6)
        f[:3] = Rb.T @ np.array([0.0, 0.0, -push])
        e = hold + np.linalg.solve(L, f) / np.asarray(gains, float)
        z = 2
        goal = np.eye(4)
        goal[:3, 3] = E[:3, 3] + Rb @ e[:3]
        goal[:3, :3] = Rb @ Rotation.from_rotvec(e[3:]).as_matrix() @ Rb.T @ E[:3, :3]
        run.hud = f"press {push:.2f} N   t {t:4.2f} s   edge {1e3 * (edge - board_z):5.1f} mm above the board"
        run.step(goal, CLOSE, extra={"press_t": t, "edge_mm": 1e3 * (edge - board_z), "push_n": push,
                                     "lambda_zz": float(L[z, z])})
    run.phase = "lift"
    K_now = run.knife_pose()
    K_up = K_now.copy()
    K_up[2, 3] += lift
    move_knife(run, K_in_E, K_up, CLOSE, speed=0.05, settle=after, phase="lift")
    return {"reached_board_s": reached, "pressed_s": min(t, max_time)}


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, default=Path("Data/tofu_chefd"), help="Pipeline A scene directory")
    ap.add_argument("--photo", type=Path, default=Path("Data/tofu_reference/chefd_frame_17.9.png"),
                    help="the frame the scene was built from (for the food's outline)")
    ap.add_argument("--stroke", choices=["cook", "press"], default="cook")
    ap.add_argument("--push-force", type=float, nargs="+", default=[0.3, 0.6], help="press: push forces [N]")
    ap.add_argument("--material", default="firm_tofu", choices=sorted(PRESETS))
    ap.add_argument("--fracture-energy", type=float, default=None)
    ap.add_argument("--dx", type=float, default=2.5e-3)
    ap.add_argument("--device", default="cuda:0", help="Warp device of the solver")
    ap.add_argument("--physics-hz", type=float, default=120.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--robot-side", choices=["behind", "left"], default="behind",
                    help="behind the board facing the camera, where the cook stands, or on stage 14's side facing "
                         "+x (there the hand comes so near the base on the draw back that the IK stalls)")
    ap.add_argument("--robot-standoff", type=float, default=0.52,
                    help="the base stands this far from the handle's mean position [m]")
    ap.add_argument("--reference", type=Path, default=None,
                    help="cook: the clip (Data/tofu_reference/chefd_side.mp4), set beside the render from its camera")
    ap.add_argument("--reference-start", type=float, default=4.17, help="clip time [s] at the stroke's start")
    ap.add_argument("--reference-csv", type=Path, default=Path("results/tofu_real2sim/chefd/timeseries.csv"),
                    help="cook: tofu_real2sim.py's force time series, drawn with this run's in knife_force.png")
    ap.add_argument("--kp", type=float, default=150.0, help="press: OSC stiffness [1/s^2]")
    ap.add_argument("--kp-z", type=float, default=10.0, help="press: OSC stiffness along z, the pushing axis [1/s^2]")
    ap.add_argument("--lead", type=float, default=1.0 / 120.0,
                    help="cook: command the stroke this far ahead [s]; the arm's drives trail a pose by a physics step")
    ap.add_argument("--no-wrench", action="store_true", help="measure the tofu's wrench but do not apply it")
    ap.add_argument("--efforts", action="store_true", help="record joint torques and the Jacobian every step")
    ap.add_argument("--clear", nargs="*", default=["paper_towel"],
                    help="categories also left out of the scene: the Panda's hand, 20 cm across its fingers, "
                         "sweeps the paper towel off its plate when it draws the knife back as the cook does")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/tofu_og/cook"))
    args = ap.parse_args()

    if not args.allow_busy_gpu and args.device.startswith("cuda"):
        reason = gpu_busy_reason()
        if reason:
            raise SystemExit(f"the GPU is busy ({reason}); wait, or pass --allow-busy-gpu")
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()

    # The food, fitted and placed as tofu_real2sim.py does it.
    import imageio.v2 as imageio

    rec = load_reconstruction(args.scene)
    H0, W0 = imageio.imread(args.photo).shape[:2]
    fitted = fit_food(rec, (W0, H0))
    place = fitted.placement
    material = PRESETS[args.material]
    if args.fracture_energy is not None:
        material = replace(material, fracture_energy=args.fracture_energy)

    from simfoundry import configure_omnigibson_data_path

    configure_omnigibson_data_path(force=False)
    import omnigibson as og  # noqa: E402 - boots nothing yet
    from omnigibson.macros import gm  # noqa: E402

    from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros  # noqa: E402

    apply_teleop_omnigibson_macros(enable_tr=False)
    gm.ENABLE_OBJECT_STATES = False  # no states or rules are used; they cost time every step
    dt = 1.0 / args.physics_hz
    # One physics step per sim step and no rendering on step: the loop renders a frame every few steps.
    og.launch(physics_dt=dt, rendering_dt=dt, sim_step_dt=dt, device="cpu")
    # Only now the solver: Isaac Sim brings its own Warp (omni.warp.core), and one Warp per process.
    import warp as wp  # noqa: E402
    from omnigibson.controllers import IsGraspingState  # noqa: E402

    from simfoundry.cutting.mpm import MPMCuttingSim, SimConfig  # noqa: E402
    from simfoundry.cutting.og_cosim import (  # noqa: E402
        FoodPoints, PhysicsStepCoupler, collider_hulls, handle_centre, write_knife_usd,
    )
    from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths  # noqa: E402

    config = SimConfig(dx=args.dx)

    def new_sim():
        return MPMCuttingSim(material, place.block_size, knife=CHEF_KNIFE, config=config, device=args.device)

    sim = new_sim()
    S = place.sim_from_world(sim.block_min)
    W = np.linalg.inv(S)
    # The knife's yaw in the solver's frame: tip towards the camera's right, as the cook holds it.
    right_sim = S[:3, :3] @ rec.cam_to_world[:3, 0]
    knife_yaw = 0.0 if right_sim[0] >= 0.0 else math.pi
    traj = cook_stroke(sim, knife_yaw, board_gap=BOARD_GAP)
    lo, hi = sim.block_min, sim.block_max
    centre = 0.5 * (lo + hi)
    if args.stroke == "cook":
        poses = [traj.state(t)[0] for t in np.arange(0.0, traj.duration, 0.05)]
    else:
        # A level blade, its straight part (u from -half_length to half_length - belly) centred over the block.
        g = CHEF_KNIFE
        mid_u = -0.5 * g.belly_length
        e_u = np.array([math.cos(knife_yaw), math.sin(knife_yaw), 0.0])
        hover = KnifePose(np.array([centre[0], centre[1], hi[2] + 0.006]) - mid_u * e_u, yaw=knife_yaw)
        poses = [hover]
    handles = np.array([(knife_world(p, W) @ np.append(handle_centre(), 1.0))[:3] for p in poses])
    if args.robot_side == "behind":
        # Where the cook stands: behind the board, facing the clip's camera (-y).
        robot_pos = np.array([handles[:, 0].mean(), handles[:, 1].mean() + args.robot_standoff, 0.0])
        robot_yaw = -0.5 * math.pi
    else:
        # Stage 14's side of the table, facing +x, the knife pointing away from the robot.
        robot_pos = np.array([handles[:, 0].mean() - args.robot_standoff, handles[:, 1].mean(), 0.0])
        robot_yaw = 0.0

    knife_usd = write_knife_usd(out / "knife" / "chef_knife.usd")
    scene = load_json_with_absolute_usd_paths(str(args.scene / SCENE_JSON))
    kind = "ik" if args.stroke == "cook" else "osc"
    from omnigibson.utils.asset_utils import get_dataset_path  # noqa: E402

    b = scene["objects_info"]["init_info"][fitted.board.name]["args"]
    board_usd = Path(get_dataset_path(b["dataset_name"])) / "objects" / b["category"] / b["model"] / "usd"
    board_usd = board_usd / f"{b['model']}.usd"
    # The cut's line across the block, and a centimetre past either end.
    u = np.linspace(lo[0] - 0.01, hi[0] + 0.01, 25)
    cut_xy = (np.stack([u, np.full_like(u, centre[1]), np.full_like(u, sim.board_z)], axis=1) @ W[:3, :3].T
              + W[:3, 3])[:, :2]
    changes = edit_scene(scene, rec=rec, fitted=fitted, robot_pos=robot_pos, robot_yaw=robot_yaw,
                         arm_controller=arm_controller_config(kind, args.kp, args.kp_z), knife_usd=knife_usd,
                         board_hulls=collider_hulls(board_usd), cut_xy=cut_xy, clear=args.clear)
    (out / "scene_edited.json").write_text(json.dumps(scene, indent=1))
    print("[og_cut] scene changes:", json.dumps(changes), flush=True)

    # Cameras: the clip's own (stage 4 pose, DA3 intrinsics) and a close-up from the front right.
    C = rec.cam_to_world
    cam_quat = matrix_to_quat_xyzw(C[:3, :3] @ np.diag([1.0, -1.0, -1.0]))
    hfov = 2.0 * math.atan(0.5 * rec.calib_size[0] / rec.K[0, 0])
    target = W[:3, :3] @ centre + W[:3, 3]
    az, el = math.radians(-50.0), math.radians(28.0)
    eye = target + 0.36 * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    sensors = []
    for name, pos, quat in (("chefd_view", C[:3, 3], cam_quat), ("closeup", eye, look_at(eye, target))):
        sensors.append({"sensor_type": "VisionSensor", "name": name, "modalities": ["rgb"],
                        "sensor_kwargs": {"image_height": args.height, "image_width": args.width,
                                          "clipping_range": [0.01, 20.0]},
                        "position": [float(v) for v in pos], "orientation": [float(v) for v in quat]})
    hz = args.physics_hz
    env = og.Environment(configs=dict(
        env=dict(action_frequency=hz, rendering_frequency=hz, physics_frequency=hz, external_sensors=sensors),
        scene=dict(type="Scene", scene_file=scene, use_floor_plane=True, floor_plane_visible=True, use_skybox=True,
                   include_robots=True)))
    env.reset()
    cams = {name: env.external_sensors[name] for name in ("chefd_view", "closeup")}
    cams["chefd_view"].focal_length = cams["chefd_view"].horizontal_aperture / (2.0 * math.tan(0.5 * hfov))
    robot = env.robots[0]
    knife = env.scene.object_registry("name", KNIFE)
    arm = Arm(robot)
    if kind == "osc":
        set_osc_gains(robot, osc_gains(args.kp, args.kp_z))
    limits = [np_(robot.joint_lower_limits)[arm.arm_idx].tolist(), np_(robot.joint_upper_limits)[arm.arm_idx].tolist()]
    fields = {"efforts", "jacobian"} if args.efforts else set()

    # What the board's top is where the knife meets it, by ray casts from above along the cut.
    board_hits = []
    for q in cut_xy[::3]:
        hit = og.sim.psqi.raycast_closest((float(q[0]), float(q[1]), 0.3), (0.0, 0.0, -1.0), 0.4)
        if hit["hit"]:
            board_hits.append({"xy": [float(q[0]), float(q[1])], "z": float(hit["position"][2]),
                               "body": str(hit["rigidBody"])})
    print("[og_cut] board under the cut (ray casts, z in mm):",
          [round(1e3 * h["z"], 2) for h in board_hits], flush=True)

    run = Run(og, env, arm, knife, None, cams, out, args.fps, fields)
    run.grasp_true = IsGraspingState.TRUE
    if args.reference is not None and args.stroke == "cook":
        # The clip from the same camera, beside the render: the cook's stroke starts at --reference-start.
        run.reference = clip_frames(args.reference, args.reference_start, args.fps, (args.width, args.height),
                                    int(math.ceil(traj.duration * args.fps)) + 2)
        run.reference_start = args.reference_start
    for _ in range(20):
        run.step(arm.eef(), OPEN)
    grasp = grasp_knife(run, knife, robot, arm)
    K_in_E = np.asarray(grasp["knife_in_eef"])
    run.K_in_E = K_in_E
    print("[og_cut] grasp:", json.dumps({k: v for k, v in grasp.items() if k != "knife_in_eef"}), flush=True)
    if not grasp["held"]:
        raise SystemExit("the gripper did not take the knife")

    results = []
    experiments = [None] if args.stroke == "cook" else list(args.push_force)
    for n_exp, push in enumerate(experiments):
        if n_exp > 0:
            sim = new_sim()
        coupling = KnifeCoupling(sim, S, settle=0.15, linger=1.0)
        coupler = PhysicsStepCoupler(coupling, knife.root_link.prim_path, apply_wrench=not args.no_wrench)
        albedo, _ = tofu_look(sim.x0, sim.skin, cloth=0.0)
        path = f"/World/mpm_tofu_{n_exp}"
        if run.points is not None:
            run.points.points.GetPrim().SetActive(False)
        spacing = args.dx / config.particles_per_axis
        run.points = FoodPoints(og.sim.stage, path, sim.skin, 1.6 * spacing, None, CUT_RGB, albedo=albedo)
        run.world_from_sim = coupling.world_from_sim()
        run.points.update(sim.positions() @ W[:3, :3].T + W[:3, 3])
        run.coupler = coupler
        coupler.start()
        tag = "cook" if push is None else f"press_{push:.2f}N"
        k_start = run.k
        t_exp = time.time()
        if push is None:
            start = knife_world(traj.state(0.0)[0], W)
            above = start.copy()
            above[2, 3] += 0.06
            approach = [move_knife(run, K_in_E, above, CLOSE, settle=0.2),
                        move_knife(run, K_in_E, start, CLOSE, speed=0.05, settle=0.5)]
            cook(run, K_in_E, traj, W, lead=args.lead)
            stroke = {"approach": approach, "trajectory_s": traj.duration}
        else:
            K_hover = knife_world(poses[0], W)
            above = K_hover.copy()
            above[2, 3] += 0.06
            # Slowly, and given time to settle: the OSC is soft along z (osc_gains).
            approach = [move_knife(run, K_in_E, above, CLOSE, settle=0.5),
                        move_knife(run, K_in_E, K_hover, CLOSE, speed=0.02, settle=1.5)]
            stroke = press(run, arm, K_in_E, K_hover, push, W[2, 2] * sim.board_z + W[2, 3],
                           osc_gains(args.kp, args.kp_z))
            stroke["approach"] = approach
            # Out of the way for the next block.
            move_knife(run, K_in_E, above, CLOSE, settle=0.2, phase="away")
        coupler.stop()
        rows = run.rows[k_start:]
        adv = [r for r in rows if r.get("advanced")]
        mpm = np.array([r["mpm_wall_s"] for r in adv]) if adv else np.zeros(1)
        steps = np.array([r["step_wall_s"] for r in rows])
        fz = np.array([r.get("fz", 0.0) for r in rows])
        x = sim.positions()
        groups = sim.piece_groups()[sim.pieces()]
        masses = sorted((round(1e3 * sim.particle_mass * int((groups == g_).sum()), 1) for g_ in np.unique(groups)),
                        reverse=True)
        stroke_rows = [r for r in rows if r["phase"] in ("stroke", "press")]
        res = dict(
            experiment=tag, push_force_n=push, n_cuts=sim.n_cuts,
            cut_complete=sim.cut_complete(0) if sim.n_cuts else False, connected_pieces=sim.connected_pieces(),
            piece_masses_g=masses, ligament_mm=round(1e3 * sim.ligament_height(0, x), 2) if sim.n_cuts else None,
            crack_onset_depth_mm=None if not sim.n_cuts or not np.isfinite(sim.crack_onset_height(0))
            else round(1e3 * (hi[2] - sim.crack_onset_height(0)), 2),
            max_force_z_n=float(fz.max()),
            max_force_n=float(max((math.sqrt(r.get("fx", 0) ** 2 + r.get("fy", 0) ** 2 + r.get("fz", 0) ** 2)
                                   for r in rows), default=0.0)),
            solver_time_s=round(sim.time, 3), physics_steps=len(rows), solver_steps=len(adv),
            idle_steps=len(rows) - len(adv),
            mpm_step_wall_s=dict(mean=float(mpm.mean()), median=float(np.median(mpm)),
                                 p95=float(np.percentile(mpm, 95)), max=float(mpm.max())),
            physics_step_wall_s=dict(
                coupled_mean=float(np.mean([r["step_wall_s"] for r in adv])) if adv else None,
                idle_mean=float(np.mean([r["step_wall_s"] for r in rows if not r.get("advanced")]))
                if len(adv) < len(rows) else None, all_mean=float(steps.mean())),
            tracking=dict(
                pos_err_mm_max=max((r["pos_err_mm"] for r in stroke_rows if "pos_err_mm" in r), default=None),
                pos_err_mm_mean=float(np.mean([r["pos_err_mm"] for r in stroke_rows if "pos_err_mm" in r]))
                if any("pos_err_mm" in r for r in stroke_rows) else None,
                ang_err_deg_max=max((r["ang_err_deg"] for r in stroke_rows if "ang_err_deg" in r), default=None)),
            wall_s=round(time.time() - t_exp, 1), stroke=stroke, coupling_log=coupling.log,
        )
        results.append(res)
        print(f"[og_cut] {tag}: " + json.dumps({k: v for k, v in res.items() if k not in ("stroke", "coupling_log")}),
              flush=True)
        np.savez_compressed(out / f"{tag}_particles.npz", x=x, piece=sim.pieces(), groups=groups, x0=sim.x0,
                            skin=sim.skin)
    run.close()

    # The time series, without the per-step arrays (in a JSON lines file when recorded).
    flat = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in run.rows]
    keys = sorted({k for r in flat for k in r})
    with open(out / "timeseries.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    if fields:
        with open(out / "arm.jsonl", "w") as fh:
            for r in run.rows:
                fh.write(json.dumps({k: r[k] for k in ("k", "t", "phase", "efforts", "jacobian", "eef", "fx", "fy",
                                                       "fz", "mx", "my", "mz", "kx", "ky", "kz") if k in r}) + "\n")
    try:
        plot(run.rows, out, args.reference_csv)
    except Exception as exc:  # a plot is not worth losing the run's numbers over
        print(f"[og_cut] no plot: {exc}", flush=True)
    ft = run.frame_times
    metrics = dict(
        stroke=args.stroke, no_wrench=args.no_wrench, scene=str(args.scene), scene_changes=changes,
        board_under_cut=board_hits, material=material.__dict__, dx=args.dx, particles=sim.n_particles,
        mass_g=round(1e3 * sim.particle_mass * sim.n_particles, 1), solver_dt=sim.dt, physics_dt=dt,
        warp=[wp.__version__, wp.__file__], grasp={k: v for k, v in grasp.items() if k != "knife_in_eef"},
        knife_in_eef=grasp["knife_in_eef"], arm_joint_limits=limits,
        robot_controller=arm_controller_config(kind, args.kp, args.kp_z),
        osc_kp=osc_gains(args.kp, args.kp_z) if kind == "osc" else None,
        results=results,
        frame_cost_s=dict(points_update=float(np.mean([f["points_s"] for f in ft])),
                          render=float(np.mean([f["render_s"] for f in ft])),
                          read=float(np.mean([f["read_s"] for f in ft]))) if ft else None,
        frames=len(ft), total_wall_s=round(time.time() - t_wall, 1),
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    print(json.dumps({k: metrics[k] for k in ("particles", "mass_g", "frame_cost_s", "total_wall_s")}, default=float),
          flush=True)
    os._exit(0)  # OmniGibson's atexit teardown is slow and sometimes hangs; everything is written


if __name__ == "__main__":
    main()
