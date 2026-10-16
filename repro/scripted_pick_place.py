# SPDX-License-Identifier: Apache-2.0
"""Scripted pick-and-place in a SimFoundry-reconstructed OmniGibson scene (no learned policy).

Shows that a reconstructed scene is manipulable: a Franka (IK controller, assisted grasping, as in the
scene file) picks one reconstructed object and puts it into another, and the B-pipeline task's own goal
predicate decides success.

Usage (simfoundry env, headless):
  python repro/scripted_pick_place.py --scene put_marker_in_cup_v2 --task <task.yaml> \
      --pick iter_2 --place iter_1 --mode insert_vertical --out demo.mp4

Modes:
  insert_vertical  top-down grasp at the middle of a lying elongated object, rotate it upright about the
                   finger-closing axis, lower it into the container, release (marker -> cup).
  rim_pinch        top-down pinch of an upright cup's rim wall, carry, release over the container
                   (cup -> bowl).

The pipeline places the robot base arbitrarily for a video shot without a robot (here 0.87 m from the
objects, beyond the Franka's reach); --standoff moves it in front of the objects first, which is the
manual scene-editing step SimFoundry expects between pipelines B and C.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

import imageio
import numpy as np
import torch as th
from PIL import Image

REPO = Path(__file__).resolve().parents[1]  # repro/<this file> -> repository root
sys.path.insert(0, str(REPO))
from simfoundry import CFG_DIR, configure_omnigibson_data_path, import_og_dependencies  # noqa: E402
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths  # noqa: E402

configure_omnigibson_data_path(force=True)
import_og_dependencies()

import omnigibson as og  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402
import omnigibson.lazy as lazy  # noqa: E402
from omnigibson.utils.config_utils import parse_config  # noqa: E402

DATA = REPO / "Data"


def np_(x):
    return x.detach().cpu().numpy().astype(np.float64) if isinstance(x, th.Tensor) else np.asarray(x, np.float64)


def rot_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def angle_between(Ra, Rb):
    return float(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))


def mat_to_axisangle(R):
    return np_(T.quat2axisangle(T.mat2quat(th.tensor(R, dtype=th.float32))))


class Demo:
    def __init__(self, env, video_scale=0.5):
        self.env = env
        self.robot = env.robots[0]
        self.arm = self.robot.default_arm
        self.arm_idx = np_(self.robot.arm_action_idx[self.arm]).astype(int)
        self.grip_idx = np_(self.robot.gripper_action_idx[self.arm]).astype(int)
        self.frames = []
        self.video_scale = video_scale
        self.info = {}
        self.steps = 0
        self.grip_cmd = None
        self.real_cam_sensor = None
        self.masks = []
        self.empty_depth = None

    # ------------------------------------------------------------------ observation / video
    def record(self, obs):
        if self.real_cam_sensor is not None:
            # Foreground = anything nearer than the empty floor. (Instance segmentation would need the
            # BEHAVIOR-1K asset dataset's semantic class list, which is not installed here.)
            o = self.real_cam_sensor.get_obs()[0]
            rgb = np_(o["rgb"])[:, :, :3].astype(np.uint8)
            self.frames.append(rgb)
            if self.empty_depth is not None:
                depth = np_(o["depth_linear"])
                self.masks.append(depth < self.empty_depth - 0.004)
            return
        cams = []
        for name, cam in sorted(obs.get("external", {}).items()):
            if isinstance(cam, dict) and "rgb" in cam:
                cams.append(np_(cam["rgb"])[:, :, :3].astype(np.uint8))
        if not cams:
            return
        h = min(c.shape[0] for c in cams)
        cams = [np.asarray(Image.fromarray(c).resize((int(c.shape[1] * h / c.shape[0]), h))) for c in cams]
        frame = np.concatenate(cams, axis=1)
        if self.video_scale != 1:
            frame = np.asarray(Image.fromarray(frame).resize(
                (int(frame.shape[1] * self.video_scale) // 2 * 2, int(frame.shape[0] * self.video_scale) // 2 * 2)))
        self.frames.append(frame)

    def step(self, arm_cmd, grip):
        action = th.zeros(self.robot.action_dim)
        action[self.arm_idx] = th.tensor(arm_cmd, dtype=th.float32)
        action[self.grip_idx] = float(grip)
        obs, _, terminated, truncated, info = self.env.step(action)
        self.info = info
        self.steps += 1
        self.record(obs)
        return obs

    # ------------------------------------------------------------------ robot state
    def eef(self):
        return np_(self.robot.get_eef_position(self.arm)), np_(T.quat2mat(self.robot.get_eef_orientation(self.arm)))

    def base(self):
        p, q = self.robot.get_position_orientation()
        return np_(p), np_(T.quat2mat(q))

    def finger_axis_world(self):
        links = self.robot.finger_links[self.arm]
        p0 = np_(links[0].get_position_orientation()[0])
        p1 = np_(links[1].get_position_orientation()[0])
        v = p1 - p0
        return v / np.linalg.norm(v)

    def finger_opening(self):
        q = np_(self.robot.get_joint_positions())[np_(self.robot.gripper_control_idx[self.arm]).astype(int)]
        return float(np.sum(np.abs(q)))

    # ------------------------------------------------------------------ motion
    def servo(self, target_p, target_R, grip, max_steps=150, pos_tol=0.004, ang_tol=0.03,
              max_dpos=0.012, max_dori=0.06, label=""):
        bp, bR = self.base()
        for i in range(max_steps):
            p, R = self.eef()
            err = target_p - p
            dpos = bR.T @ err
            n = np.linalg.norm(dpos)
            if n > max_dpos:
                dpos *= max_dpos / n
            dori = mat_to_axisangle(bR.T @ (target_R @ R.T) @ bR)
            a = np.linalg.norm(dori)
            if a > max_dori:
                dori *= max_dori / a
            self.step(np.concatenate([dpos, dori]), grip)
            if np.linalg.norm(err) < pos_tol and a < ang_tol:
                break
        p, R = self.eef()
        q = np_(self.robot.get_joint_positions())[np_(self.robot.arm_control_idx[self.arm]).astype(int)]
        lo = np_(self.robot.joint_lower_limits)[np_(self.robot.arm_control_idx[self.arm]).astype(int)]
        hi = np_(self.robot.joint_upper_limits)[np_(self.robot.arm_control_idx[self.arm]).astype(int)]
        margin = np.minimum(q - lo, hi - q)
        print(f"  [{label}] {i + 1} steps, pos err {np.linalg.norm(target_p - p) * 1000:.1f} mm, "
              f"ang err {np.degrees(np.linalg.norm(mat_to_axisangle(target_R @ R.T))):.1f} deg, "
              f"eef {p.round(3)} target {np.asarray(target_p).round(3)}, min joint margin "
              f"{np.degrees(margin.min()):.0f} deg (joint {int(margin.argmin()) + 1})", flush=True)

    def hold(self, grip, n, label=""):
        for _ in range(n):
            self.step(np.zeros(6), grip)
        if label:
            print(f"  [{label}] held {n} steps, finger opening {self.finger_opening():.4f}", flush=True)



def real_camera(scene, width=1280, hfov_deg=None):
    """World pose and FOV of the video frame stage 3 picked, as a USD camera.

    Stage 4 makes that frame's camera the reference for the OmniGibson world, so its cam2world (OpenCV
    axes: x right, y down, z forward) is already in scene coordinates; DA3 gives the intrinsics.
    """
    d = DATA / scene
    idx = int(json.load(open(d / "s3_ground" / "frame_selection.json"))["selected_idx"])
    T_cv = np.load(d / "s4_frame" / f"image_{idx}_cam2world.npy")
    z = np.load(d / "s2_da" / "da" / "exports" / "npz" / "results.npz")
    K = z["intrinsics"][idx]
    h_da, w_da = z["image"].shape[1:3]
    R_usd = T_cv[:3, :3] @ np.diag([1.0, -1.0, -1.0])      # USD cameras look down -z with +y up
    quat = np_(T.mat2quat(th.tensor(R_usd, dtype=th.float32)))
    aperture = 20.955
    anchor_hfov = 2 * np.arctan(K[0, 2] / K[0, 0])
    # Same pose and aspect, optionally a wider lens: the real frame is then the centred sub-rectangle.
    hfov = np.radians(hfov_deg) if hfov_deg else anchor_hfov
    return {"frame_idx": idx, "position": T_cv[:3, 3].tolist(), "orientation": quat.tolist(),
            "width": width, "height": int(round(width * h_da / w_da)), "aperture": aperture,
            "focal": aperture / 2 / np.tan(hfov / 2), "hfov_deg": float(np.degrees(hfov)),
            "anchor_hfov_deg": float(np.degrees(anchor_hfov))}

def obj_geometry(obj):
    lo, hi = (np_(v) for v in obj.aabb)
    pos, quat = obj.get_position_orientation()
    return {"lo": lo, "hi": hi, "center": (lo + hi) / 2, "R": np_(T.quat2mat(quat)),
            # DatasetObject knows its native box; the USDObject entries of an auto_bg scene state do not,
            # so measure the box in the object's own frame instead.
            "extent_local": (np_(obj.native_bbox) * np_(obj.scale) if hasattr(obj, "native_bbox")
                             else np_(obj.get_base_aligned_bbox(visual=False)[2]))}


def run(args):
    os.chdir(CFG_DIR)
    # --scene-file: the same reconstruction with extra assets, e.g. the auto_bg step-7 scene state whose
    # gs_background is the robot-free Gaussian-splat room (already bridged into this OG world).
    scene_json = load_json_with_absolute_usd_paths(
        args.scene_file or str(DATA / args.scene / "s14_og" / "reconstructed_og_scene.json"))
    gs_bg = "gs_background" in scene_json["objects_info"]["init_info"]
    task_cfg = parse_config(args.task)["og_task_config"]
    cams = parse_config(str(Path(CFG_DIR) / "external_sensors" / f"{args.cams}.yaml"))["external_sensors"]
    real_cam = None
    if args.real_view:
        real_cam = real_camera(args.scene, hfov_deg=args.real_view_hfov)
        # Only the world-fixed copy of the real camera is rendered; the robot-mounted ones are dropped.
        cams = [{"sensor_type": "VisionSensor", "name": "real_view_cam", "modalities": ["rgb", "depth_linear"],
                 "sensor_kwargs": {"image_height": real_cam["height"], "image_width": real_cam["width"],
                                   "clipping_range": [0.01, 100.0]},
                 "position": real_cam["position"], "orientation": real_cam["orientation"]}]

    # ---- put the base in front of the objects (the scene-editing step). Done in the scene JSON before
    # the env exists: teleporting a fixed-base Franka at runtime left the IK controller tracking a base
    # pose the articulation no longer had, and the arm drifted away from every target.
    registry = scene_json["state"]["registry"]["object_registry"]
    p_pick = np.array(registry[args.pick]["root_link"]["pos"])
    p_place = np.array(registry[args.place]["root_link"]["pos"])
    mid = (p_pick + p_place) / 2
    old_base = np.array(registry["robot0"]["root_link"]["pos"])
    new_base = np.array([mid[0] - args.standoff, mid[1], old_base[2]])
    if args.background:
        # A scanned room registered to a canonical frame (light_editor/background_io.py): its sidecar gives
        # the room pose, the support-surface height, and the robot pose it was registered with. Our objects
        # rest on z = 0, so align the room's support surface to z = 0 and keep the room-to-robot offset.
        side = json.load(open(args.background))
        bg_pose = side["pose"]
        surface = float(side["registration"]["support_surface"]["canonical_height_m"])
        bg_robot = side["default_robot"][0]["registry"]["root_link"]["pos"]
        new_base[2] = bg_robot[2] - surface
        bg_pos = [new_base[0] + bg_pose["position"][0] - bg_robot[0],
                  new_base[1] + bg_pose["position"][1] - bg_robot[1],
                  bg_pose["position"][2] - surface]
        usd = str(Path(args.background).with_name(side["usd_path"]))
        scene_json["objects_info"]["init_info"]["mesh_background_0"] = {
            "class_module": "omnigibson.objects.usd_object", "class_name": "USDObject",
            "args": {"name": "mesh_background_0", "usd_path": usd, "category": "mesh_background", "fixed_base": True}}
        registry["mesh_background_0"] = {"is_asleep": False, "non_kin": {},
                                         "root_link": {"pos": bg_pos, "ori": bg_pose["orientation"],
                                                       "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]}}
        print(f"background {side['id']} at {np.round(bg_pos, 3)}, robot base z {new_base[2]:.3f}", flush=True)
    registry["robot0"]["root_link"]["pos"] = new_base.tolist()
    registry["robot0"]["root_link"]["ori"] = [0.0, 0.0, 0.0, 1.0]
    for k in ("lin_vel", "ang_vel"):
        registry["robot0"]["root_link"][k] = [0.0, 0.0, 0.0]

    env = og.Environment(configs={
        "env": {"external_sensors": cams, "action_frequency": 15, "rendering_frequency": 15, "physics_frequency": 120},
        "scene": {"type": "Scene", "scene_file": scene_json, "use_floor_plane": True,
                  # A splat background is visual only: keep the plane for collision, let the splat show the desk.
                  "floor_plane_visible": bool(args.real_view) and not gs_bg,
                  "use_skybox": True, "include_robots": True},
        "task": task_cfg,
    })
    obs, _ = env.reset()
    if gs_bg:
        # NuRec feeds the splat through the RTX tonemapper as radiance. Isaac Sim's default ACES curve
        # (op 6) brightened and saturated it (mean abs error 0.126 against the real frame on background
        # pixels); the linear operator shows the trained colours (0.032) and keeps the lit props natural.
        lazy.carb.settings.get_settings().set("/rtx/post/tonemap/op", 1)
    if real_cam is not None:
        cam = env.external_sensors["real_view_cam"]
        cam.horizontal_aperture = real_cam["aperture"]
        cam.focal_length = real_cam["focal"]
        print(f"real-view camera: frame {real_cam['frame_idx']}, pos {np.round(real_cam['position'], 3)}, "
              f"hfov {real_cam['hfov_deg']:.1f} deg", flush=True)
    demo = Demo(env, video_scale=1.0 if real_cam is not None else 0.5)
    empty_ref = None
    if real_cam is not None:
        sensor = env.external_sensors["real_view_cam"]
        movers = [o for o in env.scene.objects if o.name.startswith("iter_")] + [env.robots[0]]
        for o in movers:
            o.visible = False
        for _ in range(15):
            og.sim.render()
        o_empty = sensor.get_obs()[0]
        empty_ref = np_(o_empty["rgb"])[:, :, :3].astype(np.uint8)
        demo.empty_depth = np_(o_empty["depth_linear"])
        for o in movers:
            o.visible = True
        for _ in range(5):
            og.sim.render()
        demo.real_cam_sensor = sensor
    robot = demo.robot
    pick = env.scene.object_registry("name", args.pick)
    place = env.scene.object_registry("name", args.place)

    gp, gq = obj_geometry(pick), obj_geometry(place)
    bp, _ = demo.base()
    print(f"robot base {old_base.round(3)} -> {bp.round(3)} (requested {new_base.round(3)}); distance to pick "
          f"{np.linalg.norm(gp['center'][:2] - bp[:2]):.3f} m", flush=True)
    new_base = bp

    # ---- which gripper command opens? (the controller takes 0..1; check rather than assume)
    demo.hold(1.0, 15)
    open_at_1 = demo.finger_opening()
    demo.hold(0.0, 15)
    open_at_0 = demo.finger_opening()
    OPEN, CLOSE = (1.0, 0.0) if open_at_1 > open_at_0 else (0.0, 1.0)
    demo.hold(OPEN, 15, "open")
    print(f"gripper: opening {open_at_1:.4f} @1.0, {open_at_0:.4f} @0.0 -> OPEN={OPEN}", flush=True)

    p0, R0 = demo.eef()
    close_w = demo.finger_axis_world()
    close_local = R0.T @ close_w
    # The gripper's approach axis runs from the hand link to the eef link (eef-local +z). The reset pose
    # tilts it ~25 deg off vertical, so it must not be inferred from "world down at reset".
    hand_p = np_(robot.links["panda_hand"].get_position_orientation()[0])
    approach_local = R0.T @ ((p0 - hand_p) / np.linalg.norm(p0 - hand_p))
    print(f"eef reset pos {p0.round(3)}, closing axis local {close_local.round(2)}, "
          f"approach local {approach_local.round(2)}, eef->fingertip {robot.eef_to_fingertip_lengths}", flush=True)

    def R_with(approach_w, closing_w):
        # Rotation mapping the eef's local approach/closing axes onto the requested world directions.
        a_l, c_l = approach_local / np.linalg.norm(approach_local), close_local - close_local.dot(approach_local) * approach_local
        c_l /= np.linalg.norm(c_l)
        b_l = np.cross(a_l, c_l)
        a_w = approach_w / np.linalg.norm(approach_w)
        c_w = closing_w - closing_w.dot(a_w) * a_w
        c_w /= np.linalg.norm(c_w)
        b_w = np.cross(a_w, c_w)
        return np.stack([a_w, c_w, b_w], 1) @ np.stack([a_l, c_l, b_l], 1).T

    down = np.array([0.0, 0.0, -1.0])
    t0 = time.time()
    if args.mode == "insert_vertical":
        ext, R_obj = gp["extent_local"], gp["R"]
        long_w = R_obj[:, int(np.argmax(ext))]
        long_w = np.array([long_w[0], long_w[1], 0.0]); long_w /= np.linalg.norm(long_w)
        closing = np.cross([0, 0, 1.0], long_w)            # fingers close across the short axis
        # The gripper is symmetric, so +closing and -closing are the same grasp but 180 deg apart at the
        # wrist; take the one nearer the current hand pose or IK runs into the last joint's limit.
        _, R_now = demo.eef()
        closing = min((closing, -closing), key=lambda c: angle_between(R_with(down, c), R_now))
        R_grasp = R_with(down, closing)
        c = gp["center"]
        half_len = float(np.max(ext)) / 2
        demo.servo(c + [0, 0, 0.15], R_grasp, OPEN, label="above pick")
        demo.servo(c + [0, 0, args.grasp_dz], R_grasp, OPEN, max_dpos=0.006, label="descend")
        demo.hold(CLOSE, 25, "close")
        grasped = robot.is_grasping(arm=demo.arm, candidate_obj=pick)
        print(f"  grasped: {grasped}", flush=True)
        demo.servo(c + [0, 0, 0.25], R_grasp, CLOSE, label="lift")
        # Upright in two moves. Rotating about the closing axis swings the palm along the object's long
        # axis; with the palm pointing forward the wrist folds back onto the base (joint 4 hit its limit
        # and IK wandered off), so first yaw the object until its long axis runs sideways (closing axis
        # along the forward direction), then tip it up with the palm pointing sideways, choosing the side
        # that keeps the wrist toward the robot's centre line.
        fwd = gq["center"] - new_base; fwd[2] = 0; fwd /= np.linalg.norm(fwd)
        side = np.cross([0, 0, 1.0], fwd)
        hold_p = c + [0, 0, 0.30]
        demo.servo(hold_p, R_grasp, CLOSE, label="raise")
        _, R_now = demo.eef()
        closing2 = min((fwd, -fwd), key=lambda v: angle_between(R_with(down, v), R_now))
        R_yaw = R_with(down, closing2)
        demo.servo(hold_p, R_yaw, CLOSE, max_dori=0.05, max_steps=200, label="yaw long axis sideways")
        p_now, _ = demo.eef()
        palm_dir = side if (p_now - new_base).dot(side) >= 0 else -side
        cands = [rot_axis_angle(closing2, sgn * np.pi / 2) @ R_yaw for sgn in (1, -1)]
        R_up = max(cands, key=lambda R: (R @ approach_local).dot(palm_dir))
        demo.servo(hold_p, R_up, CLOSE, max_dori=0.04, max_steps=250, label="tip upright")
        rim = gq["hi"][2]
        above = np.array([gq["center"][0], gq["center"][1], rim + 0.03 + half_len])
        demo.servo(above + [0, 0, 0.05], R_up, CLOSE, label="over container")
        demo.servo(above, R_up, CLOSE, max_dpos=0.006, label="lower")
        demo.hold(OPEN, 30, "release")
        demo.servo(above + [0, 0, 0.12], R_up, OPEN, label="retreat")
    elif args.mode == "rim_pinch":
        c, lo, hi = gp["center"], gp["lo"], gp["hi"]
        radius = float(min(hi[0] - lo[0], hi[1] - lo[1])) / 2
        radial = gp["center"][:2] - new_base[:2]; radial = np.array([*(radial / np.linalg.norm(radial)), 0.0])
        # Pinch the near wall: fingers straddle it, closing radially.
        grasp_xy = c - radial * (radius - args.wall_inset)
        _, R_now = demo.eef()
        radial_c = min((radial, -radial), key=lambda c: angle_between(R_with(down, c), R_now))
        R_grasp = R_with(down, radial_c)
        top = hi[2]
        demo.servo(np.array([grasp_xy[0], grasp_xy[1], top + 0.12]), R_grasp, OPEN, label="above rim")
        demo.servo(np.array([grasp_xy[0], grasp_xy[1], top - args.rim_depth]), R_grasp, OPEN, max_dpos=0.006,
                   label="descend")
        demo.hold(CLOSE, 25, "close")
        print(f"  grasped: {robot.is_grasping(arm=demo.arm, candidate_obj=pick)}", flush=True)
        lift = np.array([grasp_xy[0], grasp_xy[1], top + 0.20])
        demo.servo(lift, R_grasp, CLOSE, label="lift")
        # Keep the grasp offset relative to the cup centre; heights are set by where the cup's bottom goes.
        offset = grasp_xy - c
        eef_above_bottom = (top - args.rim_depth) - lo[2]
        tgt_xy = np.array([gq["center"][0], gq["center"][1], 0.0]) + np.array([offset[0], offset[1], 0.0])
        over = tgt_xy + [0, 0, gq["hi"][2] + 0.04 + eef_above_bottom]       # bottom 4 cm above the rim
        demo.servo(over, R_grasp, CLOSE, label="over container")
        down_to = tgt_xy + [0, 0, gq["lo"][2] + 0.03 + eef_above_bottom]    # bottom 3 cm above the floor of it
        demo.servo(down_to, R_grasp, CLOSE, max_dpos=0.006, label="lower into")
        over = down_to
        demo.hold(OPEN, 30, "release")
        demo.servo(over + [0, 0, 0.15], R_grasp, OPEN, label="retreat")
    demo.hold(OPEN, 45, "settle")

    done = demo.info.get("done", {})
    success = bool(done.get("success", False))
    pf, qf = obj_geometry(pick), obj_geometry(place)
    print(f"RESULT task_success={success} steps={demo.steps} sim_wall={time.time() - t0:.0f}s "
          f"pick_final={pf['center'].round(3)} place_aabb=[{qf['lo'].round(3)}, {qf['hi'].round(3)}]", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, demo.frames, fps=15, macro_block_size=2)
    if real_cam is not None:
        json.dump({**{k: real_cam[k] for k in ("frame_idx", "hfov_deg", "anchor_hfov_deg", "width", "height",
                                                "position", "orientation")}, "focal": real_cam["focal"],
                   "aperture": real_cam["aperture"]}, open(out.with_suffix(".json"), "w"), indent=1)
        np.savez_compressed(out.with_suffix(".masks.npz"), fg=np.stack(demo.masks), empty_floor=empty_ref)
    print(f"saved {out} ({len(demo.frames)} frames)", flush=True)
    og.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--pick", required=True)
    ap.add_argument("--place", required=True)
    ap.add_argument("--mode", choices=["insert_vertical", "rim_pinch"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--standoff", type=float, default=0.55)
    ap.add_argument("--background", default=None, help="<room>.background.json of a registered mesh background")
    ap.add_argument("--scene-file", default=None,
                    help="scene state JSON to load instead of Data/<scene>/s14_og (e.g. <scene>_scene_state_auto_bg.json)")
    ap.add_argument("--cams", default="sim_franka_teleop", help="external_sensors config name")
    ap.add_argument("--real-view-hfov", type=float, default=None,
                    help="widen the real-view lens (deg); default matches the real camera")
    ap.add_argument("--real-view", action="store_true",
                    help="render only from the real video camera the reconstruction was anchored to")
    ap.add_argument("--grasp-dz", type=float, default=0.0, help="eef height above the object centre at grasp")
    ap.add_argument("--wall-inset", type=float, default=0.004)
    ap.add_argument("--rim-depth", type=float, default=0.025)
    run(ap.parse_args())
