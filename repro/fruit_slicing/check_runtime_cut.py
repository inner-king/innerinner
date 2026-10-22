# SPDX-License-Identifier: Apache-2.0
"""Scripted check of runtime cutting (simfoundry/cutting; repro/README.md, section 15.2). No policy.

A knife (simfoundry/cutting/knife_asset.py) is driven onto each of the four fruits of a scene built from the
authors' ``droid_desk_serve_fruits``, in an environment set up as teleop's is (``apply_teleop_omnigibson_macros()``:
CPU PhysX, CCD, object states and transition rules on; physics 120 Hz, actions and rendering 15 Hz; the Franka in the
scene). The fruits carry ``args.cutting = {"role": "cuttable"}``, the knife ``{"role": "knife", "blade": "blade"}``;
nothing else in the scene changes. One Isaac Sim process per mode:

``cut``: a kinematic knife. Per fruit and knife pose, a stroke: the knife moves in its blade plane, edge first, from
2 cm above the fruit to 4 mm above its bottom at ``--speed``, holds, lifts out, parks; then the pieces settle for
``--settle-s``. Each pose is checked against the scene's other objects first (the knife's colliders at the bottom of
the stroke must stay clear of their bounding boxes) and the scene is reset between strokes. The cuttable objects are
woken on every step the knife moves: a kinematic body moved by setting its pose does not wake a sleeping one, and
PhysX then reports no contact with it. Per cut it records:

- the stall: the wall time of the env step that cut, against the median step, and the rule's own breakdown
  (plane, reading the asset, clipping, cap, hull merge, writing the two files, OmniGibson's remove and add,
  initialization);
- the cut at the knife's pose: the plane the rule used against the blade plane of the pose the script commanded
  at that step, and the distance of every vertex of the pieces' new cut faces from that plane (each face should lie
  at the blade's half thickness plus the margin, on its own side);
- the settling: per piece, its largest and final speeds, how far it moved, and how deep it went below the desk
  (against the uncut fruit's own depth before the stroke); and the piece masses PhysX reports against the parent's;
- three pictures (before, just after, settled) from a camera beside the fruit.

Then presses that must not cut (the handle pushed into each fruit; the flat of the blade pushed onto it), and one
more episode that cuts all four fruits and saves the scene with ``og.sim.save()`` (ground_plane_info added back, as
og.sim.save() drops it), timestamped, never ``_latest``.

``threshold``: the force threshold, which a kinematic knife cannot test (it pushes with whatever force it takes:
hundreds of newtons even at 0.3 mm per step). A dynamic knife on a prismatic rail along its blade's up axis is driven
down with a force capped at each of ``--press-forces``, its edge starting 3 mm above the fruit.

``reload``: loads the saved scene the way teleop and policy evaluation do (``og.Environment``, ``scene_file``),
checks that the pieces come back (names, files and hashes, masses, poses against the uncut props' as a control),
lets them sit, cuts one of them again with the knife (the ``cutting`` entries survive the save), and saves again.

``reset_probe``: where ``env.reset()`` leaves a fruit that a cut removed.

Runs in the simfoundry env with this checkout first on PYTHONPATH, and boots Isaac Sim (about 90 s)::

    PYTHONPATH=$PWD OMNIGIBSON_DATA_PATH=/home/inner/real2sim/SimFoundry/deps/BEHAVIOR-1K/datasets \\
        ~/miniforge3/envs/simfoundry/bin/python repro/fruit_slicing/check_runtime_cut.py cut \\
        --out-dir results/diagnostics/fruit_slicing/runtime_cut
    ... check_runtime_cut.py reload --saved results/diagnostics/fruit_slicing/runtime_cut/<saved>.json \\
        --out-dir results/diagnostics/fruit_slicing/runtime_cut
    ... check_runtime_cut.py threshold --out-dir results/diagnostics/fruit_slicing/runtime_cut/threshold --no-images
"""
import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np

SCENE = ("/home/inner/real2sim/SimFoundry/assets/scenes/DROID/droid_desk_serve_fruits/"
         "droid_desk_serve_fruits_scene_state_latest.json")
FRUITS = ("red_apple_zlgmzo_2", "yellow_banana_yvjcfy_3", "yellow_pear_uckunr_0", "orange_fruit_rofqtq_4")
KNIFE = "knife"
PARK = (0.15, 0.55, 0.25)  # beside the desk, clear of the arm at its reset pose
EDGE_X = 0.06  # knife-frame x of the edge point aimed at the fruit: on the straight part of the edge
RAIL = "/World/cut_check_rail"  # the threshold mode's prismatic joint

#: Knife poses: (yaw of the blade in degrees, offset of the plane from the fruit's centre along the blade normal in
#: m, roll of the blade plane from vertical in degrees). A fruit's k-th stroke takes the k-th (offset, roll) and the
#: first yaw that keeps the knife clear of the other objects and differs from its earlier strokes' by 45 degrees.
OFFSET_ROLL = ((0.0, 0), (0.008, 0), (-0.006, 20))
YAWS = (0, 90, 45, 135, 180, 270, 225, 315)
CANDIDATES = [(yaw, off, roll) for off, roll in OFFSET_ROLL for yaw in YAWS]

#: Knife-frame points aimed at an object, and the axis the knife moves along: the edge cuts down its own blade plane;
#: the handle's underside and the flat of the blade (rolled 90 degrees) press straight down
AIMS = {"edge": ((EDGE_X, 0.0, 0.0), "blade"), "handle": ((-0.067, 0.0, 0.021), "world"),
        "flat": ((0.08, 0.0, 0.0204), "world")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["cut", "threshold", "reload", "reset_probe"])
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--saved", help="reload: the scene the cut mode saved")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fruits", nargs="+", default=list(FRUITS))
    ap.add_argument("--poses", type=int, default=3, help="knife poses per fruit")
    ap.add_argument("--speed", type=float, default=0.05, help="stroke speed (m/s)")
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--press-forces", type=float, nargs="+", default=[2.0, 5.0, 8.0, 15.0, 40.0],
                    help="threshold: push forces of the railed knife (N)")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--no-wake", action="store_true",
                    help="do not wake the cuttable objects while the knife moves (a kinematic knife moved by pose "
                         "does not wake a sleeping fruit, and PhysX then reports no contact with it)")
    args = ap.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    from simfoundry import configure_omnigibson_data_path

    configure_omnigibson_data_path(force=True)

    import omnigibson as og  # noqa: F401 - boots Isaac Sim before the other omnigibson imports
    from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros

    apply_teleop_omnigibson_macros()
    from simfoundry.cutting import rule as cutting  # registers the rule; must come before the scene is built

    cutting.configure(pieces_dir=str(out_dir / "pieces"))
    # Kit brings pxr, which the knife asset is written with before the scene exists; og.Environment checks that the
    # running simulator has its frequencies
    og.launch(physics_dt=1 / 120, rendering_dt=1 / 15, sim_step_dt=1 / 15, device="cpu")
    check = Check(args, out_dir, cutting)
    report = {"cut": check.run_cut, "threshold": check.run_threshold, "reload": check.run_reload,
              "reset_probe": check.run_reset_probe}[args.mode]()
    path = out_dir / f"{args.mode}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(json.dumps(report.get("summary", {}), indent=1, default=float), flush=True)
    print(f"[check] wrote {path}", flush=True)
    os._exit(0)  # OmniGibson's atexit teardown is slow and sometimes hangs; the report is written


# ------------------------------------------------------------------------------------------------------- helpers
def rotation(yaw_deg, roll_deg):
    """Knife orientation: blade along x turned by ``yaw`` about world z, its plane tilted by ``roll`` about x."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("ZX", [yaw_deg, roll_deg], degrees=True)


def look_at(eye, target):
    """xyzw quaternion for a USD camera (looks down -z, +y up) at ``eye`` facing ``target``."""
    from scipy.spatial.transform import Rotation

    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return Rotation.from_matrix(np.stack([right, up, -f], axis=1)).as_quat()


def np_(t):
    return t.detach().cpu().numpy().astype(np.float64) if hasattr(t, "detach") else np.asarray(t, np.float64)


class Check:
    def __init__(self, args, out_dir, cutting):
        self.args, self.out_dir, self.cutting = args, out_dir, cutting
        self.images = not args.no_images
        self.step_times = []
        self.phase = "load"
        self.events = []  # every cut the rule made or refused, with the phase of the script it happened in

    # ---------------------------------------------------------------------------------------------- the scene
    def build_scene(self, dynamic_knife=False):
        """The authors' scene with the fruits marked cuttable and a knife added: kinematic and parked beside the desk,
        or dynamic and lying flat on the floor plane there."""
        from scipy.spatial.transform import Rotation
        from simfoundry.cutting.knife_asset import CUTTING_SPEC, write_knife_usd
        from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

        scene = load_json_with_absolute_usd_paths(self.args.scene)
        init = scene["objects_info"]["init_info"]
        for name in self.args.fruits:
            init[name]["args"]["cutting"] = {"role": "cuttable"}
        knife_usd = write_knife_usd(self.out_dir / "knife" / "knife.usd")
        init[KNIFE] = {"class_module": "omnigibson.objects.usd_object", "class_name": "USDObject",
                       "args": {"name": KNIFE, "usd_path": knife_usd, "category": "knife",
                                "fixed_base": not dynamic_knife, "cutting": dict(CUTTING_SPEC)}}
        if dynamic_knife:
            ground = scene.get("ground_plane_info", {}).get("position", [0, 0, 0])[2]
            root = {"pos": [PARK[0], PARK[1], ground + 0.009], "ori": Rotation.from_euler("X", 90, degrees=True).as_quat().tolist(),
                    "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]}
        else:
            root = {"pos": list(PARK), "ori": Rotation.identity().as_quat().tolist()}
        scene["state"]["registry"]["object_registry"][KNIFE] = {"is_asleep": False, "root_link": root, "non_kin": {}}
        return scene

    def make_env(self, scene_json, warm_steps=10):
        import omnigibson as og
        from simfoundry.utils.ground_plane_utils import apply_ground_plane_info

        cams = [{"sensor_type": "VisionSensor", "name": "cut_cam", "modalities": ["rgb"],
                 "sensor_kwargs": {"image_height": 480, "image_width": 640, "clipping_range": [0.005, 20.0]},
                 "position": [0.3, 0.0, 0.4], "orientation": [0.0, 0.0, 0.0, 1.0]}] if self.images else []
        env = og.Environment(configs=dict(
            env=dict(action_frequency=15, rendering_frequency=15, physics_frequency=120, external_sensors=cams),
            scene=dict(type="Scene", scene_file=scene_json, use_floor_plane=True, floor_plane_visible=True,
                       use_skybox=False, include_robots=True)))
        self.ground = apply_ground_plane_info(scene_json, og.sim.floor_plane)
        self.ground_z = float(scene_json.get("ground_plane_info", {}).get("position", [0, 0, 0])[2])
        env.reset()
        self.env, self.scene = env, env.scene
        self.rule = self.cutting.get_cutting_rule(env.scene)
        assert self.rule is not None, "the cutting rule is not registered: transition rules off?"
        import torch as th

        self.action = th.zeros(env.robots[0].action_dim) if env.robots else None
        for _ in range(warm_steps):
            self.step()
        assert self.rule in env.scene.transition_rule_api.active_rules, "the cutting rule found no knife/cuttable"
        return env

    def step(self):
        n = len(self.rule.cut_log)
        t = time.perf_counter()
        self.env.step(self.action)
        dt = time.perf_counter() - t
        self.step_times.append(dt)
        for r in self.rule.cut_log[n:]:
            self.events.append({"phase": self.phase, "step": len(self.step_times), "cuttable": r["cuttable"],
                                "force_n": r["force_n"], "refused": r.get("refused"),
                                "knife_physx": self.knife_physx_pose(), "pieces": r.get("pieces")})
            print(f"[check] {self.phase}: rule {'refused' if 'refused' in r else 'cut'} {r['cuttable']} at "
                  f"{r['force_n']:.1f} N", flush=True)
        return dt

    def knife_physx_pose(self):
        """Where PhysX has the knife's body, which is what contacts see (OmniGibson moves the Xform above it)."""
        import omnigibson.lazy as lazy

        link = self.obj(KNIFE).root_link
        t = lazy.omni.physx.get_physx_interface().get_rigidbody_transformation(link.prim_path)
        return {"position": list(t["position"]), "rotation_xyzw": list(t["rotation"])} if t.get("ret_val") else None

    def obj(self, name):
        return self.scene.object_registry("name", name)

    def reset(self):
        """env.reset(), a step, the initial state once more, a step. The second load turned out unnecessary: the
        reset_probe mode shows a fruit a cut removed comes back at its saved pose after env.reset(); what stays at the
        scene origin is OmniGibson's aabb of it (hence world_aabb()). The reported runs used this, so it stays."""
        self.env.reset()
        self.step()
        self.scene.load_state(self.scene._initial_file["state"], serialized=False)
        self.step()

    def wake_cuttables(self):
        if self.args.no_wake:
            return
        for o in self.scene.objects:
            if (self.cutting.cutting_spec(o) or {}).get("role") == "cuttable":
                o.wake()

    def asleep(self, name):
        import omnigibson as og
        import omnigibson.lazy as lazy

        o = self.obj(name)
        if o is None:
            return None
        return bool(og.sim.psi.is_sleeping(og.sim.stage_id,
                                           lazy.pxr.PhysicsSchemaTools.sdfPathToInt(o.root_link.prim_path)))

    def snapshot(self, name, target, azimuth_deg):
        """A picture of ``target`` from 28 cm away, 35 degrees up."""
        if not self.images:
            return None
        import imageio.v2 as imageio
        import omnigibson as og
        import torch as th

        cam = self.env.external_sensors["cut_cam"]
        az, el = math.radians(azimuth_deg), math.radians(35)
        eye = np.asarray(target) + 0.28 * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                                                    math.sin(el)])
        cam.set_position_orientation(position=th.tensor(eye, dtype=th.float32),
                                     orientation=th.tensor(look_at(eye, target), dtype=th.float32))
        for _ in range(12):  # the annotator's output trails a moved camera by several renders
            og.sim.render()
        rgb = np_(cam.get_obs()[0]["rgb"])[:, :, :3].astype(np.uint8)
        path = self.out_dir / "images" / f"{name}.png"
        path.parent.mkdir(exist_ok=True)
        imageio.imwrite(path, rgb)
        return str(path)

    # ---------------------------------------------------------------------------------------------- the knife
    def knife_geometry(self):
        from simfoundry.cutting.knife_asset import knife_meshes

        return np.concatenate([V for V, _ in knife_meshes().values()])

    def world_aabb(self, obj):
        """An object's bounding box. A cuttable's comes from its own colliders and pose: Scene.restore() re-adds a
        cut fruit whose collision points (EntityPrim.aabb) stay stale at the scene origin."""
        if (self.cutting.cutting_spec(obj) or {}).get("role") == "cuttable":
            from simfoundry.cutting.geometry import quat_xyzw_to_matrix

            source, _ = self.rule._source(obj)
            pos, quat = (np_(v) for v in obj.get_position_orientation())
            pts = pos + (np.concatenate(source.hulls) * np_(obj.scale)) @ quat_xyzw_to_matrix(quat).T
            return pts.min(0), pts.max(0)
        return tuple(np_(v) for v in obj.aabb)

    def motion(self, fruit_name, pose, aim="edge"):
        """How the knife moves for ``pose``: its orientation, and its origin as a function of the height h of the aim
        point above the object's centre (shifted by the pose's offset along the blade normal) along the motion axis,
        with the h of the object's top and bottom along that axis."""
        lo, hi = self.world_aabb(self.obj(fruit_name))
        center = (lo + hi) / 2
        yaw, offset, roll = pose
        rot = rotation(yaw, roll)
        R = rot.as_matrix()
        point, axis = AIMS[aim]
        up = R[:, 2] if axis == "blade" else np.array([0.0, 0.0, 1.0])
        target = center + offset * R[:, 1]
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        s = (corners - target) @ up

        def origin(h):
            return target + h * up - R @ np.asarray(point)

        return {"quat": rot.as_quat(), "R": R, "origin": origin, "top": s.max(), "bottom": s.min(), "center": center}

    def pose_clear(self, fruit_name, pose, aim="edge", h=None):
        """Whether the knife at the end of its motion stays clear of every object but the fruit (5 mm margin); for a
        press with something other than the edge, also whether its blade stays clear of the fruit."""
        m = self.motion(fruit_name, pose, aim)
        h = m["bottom"] + 0.004 if h is None else h
        pts = m["origin"](h) + self.knife_geometry() @ m["R"].T
        for o in self.scene.objects:
            if o.name in (fruit_name, KNIFE) or o.fixed_base or o in self.env.robots:
                continue
            lo, hi = self.world_aabb(o)
            if np.any(np.all((pts > lo - 0.005) & (pts < hi + 0.005), axis=1)):
                return False
        if aim != "edge":
            from simfoundry.cutting.knife_asset import knife_meshes

            blade = m["origin"](h) + knife_meshes()["blade"][0] @ m["R"].T
            lo, hi = self.world_aabb(self.obj(fruit_name))
            if aim == "handle" and np.any(np.all((blade > lo - 0.002) & (blade < hi + 0.002), axis=1)):
                return False
        return True

    def move_knife(self, position, quat, max_step=0.01, steps=None):
        """Move the kinematic knife in straight steps of at most ``max_step`` (a teleport would sweep through
        whatever lies between)."""
        import torch as th
        from scipy.spatial.transform import Rotation, Slerp

        knife = self.obj(KNIFE)
        p0, q0 = (np_(v) for v in knife.get_position_orientation())
        n = steps or max(1, int(math.ceil(np.linalg.norm(np.asarray(position) - p0) / max_step)))
        slerp = Slerp([0, 1], Rotation.from_quat([q0, quat]))
        for i in range(1, n + 1):
            a = i / n
            knife.set_position_orientation(position=th.tensor(p0 + a * (np.asarray(position) - p0), dtype=th.float32),
                                           orientation=th.tensor(slerp(a).as_quat(), dtype=th.float32))
            self.wake_cuttables()
            self.step()

    # ---------------------------------------------------------------------------------------------- a stroke
    def stroke(self, fruit_name, pose, tag):
        """One stroke through ``fruit_name``; returns its measurements."""
        import omnigibson as og
        import torch as th
        from simfoundry.cutting import geometry as geo

        knife = self.obj(KNIFE)
        m = self.motion(fruit_name, pose)
        quat, origin, center = m["quat"], m["origin"], m["center"]
        h_start, h_end = m["top"] + 0.02, m["bottom"] + 0.004
        rec = {"fruit": fruit_name, "pose": {"yaw_deg": pose[0], "offset_m": pose[1], "roll_deg": pose[2]},
               "tag": tag, "fruit_center": center.tolist(),
               # the control for the pieces' depth below the desk after settling
               "fruit_below_desk_before_m": float(self.ground_z - self.world_aabb(self.obj(fruit_name))[0][2])}
        view_az = pose[0] + 90 + 35
        rec["image_before"] = self.snapshot(f"{tag}_0_before", center, view_az)
        self.phase = f"{tag}:approach"
        # Approach from above, then move down the blade's own up axis so the plane stays put during the stroke
        approach = origin(h_start) + np.array([0, 0, 0.12])
        self.move_knife(approach, quat, max_step=0.02)
        self.move_knife(origin(h_start), quat, max_step=0.02)
        n_log0 = len(self.rule.cut_log)
        self.phase = f"{tag}:stroke"
        rec["knife_at_stroke_start"] = {"commanded": origin(h_start).tolist(), "physx": self.knife_physx_pose()}
        dh = self.args.speed * og.sim.get_sim_step_dt()
        h, steps, forces, asleep = h_start, [], [], []
        cut = None
        while h > h_end - 1e-9:
            h = max(h - dh, h_end)
            knife.set_position_orientation(position=th.tensor(origin(h), dtype=th.float32),
                                           orientation=th.tensor(quat, dtype=th.float32))
            asleep.append(self.asleep(fruit_name))
            self.wake_cuttables()
            dt = self.step()
            peak = {(k.name, c.name): f for (k, c), (f, _) in self.rule.last_window.peak.items()}
            forces.append(round(peak.get((KNIFE, fruit_name), 0.0), 2))
            new = self.rule.cut_log[n_log0:]
            steps.append({"h": h, "step_s": dt, "cut_records": len(new)})
            if new and cut is None:
                cut = {"step": len(steps) - 1, "h": h, "step_s": dt, "record": new[0],
                       "other_records": new[1:]}
                rec["image_cut"] = self.snapshot(f"{tag}_1_cut", center, view_az)
            if h <= h_end:
                break
        rec["knife_at_stroke_end"] = {"commanded": origin(h).tolist(), "physx": self.knife_physx_pose()}
        self.phase = f"{tag}:hold"
        for _ in range(3):
            self.step()
        rec.update(stroke_steps=len(steps), forces_n=forces, asleep_before_step=asleep, h_start=h_start, h_end=h_end)
        self.phase = f"{tag}:lift"
        # Lift out along the blade, then park
        self.move_knife(origin(h_start) + np.array([0, 0, 0.10]), quat, max_step=0.01)
        self.move_knife(np.array(PARK), [0, 0, 0, 1], max_step=0.03)
        rec["unintended_cuts"] = [r["cuttable"] for r in self.rule.cut_log[n_log0:]
                                  if r["cuttable"] != fruit_name and "refused" not in r]
        if cut is None:
            rec["cut"] = None
            rec["max_force_n"] = max(forces) if forces else 0.0
            return rec

        record = cut["record"]
        rec["cut"] = {"step": cut["step"], "edge_height_m": cut["h"], "env_step_s": cut["step_s"],
                      "record": record}
        if "refused" in record:
            return rec

        # The cut plane against the blade plane of the pose commanded at that step
        _, blade_l, _, _ = self.rule._blade(knife)
        commanded, _ = geo.plane_to_world(blade_l, origin(cut["h"]), quat, np_(knife.scale))
        used = geo.Plane.make(record["plane_origin"], record["plane_normal"])
        rec["plane_vs_commanded"] = {
            "angle_deg": math.degrees(math.acos(min(1.0, abs(float(used.normal @ commanded.normal))))),
            "offset_m": abs(float(commanded.signed_distance(used.origin)))}
        # Every vertex of each piece's new cut face, placed at the parent's pose, against that commanded plane
        Rp = geo.quat_xyzw_to_matrix(record["parent_quat"])
        expected = record["half_gap_m"]
        faces = []
        for info in record["piece_files"]:
            source, _ = self.rule._sources[info["usd_path"]]
            gen = record["generation"]
            cap = next(p for p in source.visuals if p.name == f"cut_face_{gen}")
            world = np.asarray(record["parent_pos"]) + (np.asarray(record["parent_scale"]) * cap.points) @ Rp.T
            d = commanded.signed_distance(world)
            faces.append({"piece": info["name"], "vertices": len(d), "distance_mean_m": float(d.mean()),
                          "deviation_max_m": float(np.max(np.abs(np.abs(d) - expected)))})
        rec["cut_faces_vs_commanded"] = {"expected_distance_m": expected, "faces": faces}

        # Settling: the pieces from the cut to the end
        pieces = [self.obj(n) for n in record["pieces"]]
        rec["masses"] = {"parent_kg": record["parent_mass"], "pieces_args_kg": record["masses"],
                         "pieces_physx_kg": [float(p.mass) for p in pieces]}
        start = {p.name: np_(p.get_position_orientation()[0]) for p in pieces}
        speeds = {p.name: [] for p in pieces}
        self.phase = f"{tag}:settle"
        for _ in range(int(round(self.args.settle_s * 15))):
            self.step()
            for p in pieces:
                speeds[p.name].append(float(np.linalg.norm(np_(p.get_linear_velocity()))))
        rec["image_settled"] = self.snapshot(f"{tag}_2_settled", center, view_az)
        settle = {}
        for p in pieces:
            pos = np_(p.get_position_orientation()[0])
            lo, _ = self.world_aabb(p)
            settle[p.name] = {
                "max_speed_mps": max(speeds[p.name]) if speeds[p.name] else None,
                "final_speed_mps": float(np.linalg.norm(np_(p.get_linear_velocity()))),
                "final_angular_rps": float(np.linalg.norm(np_(p.get_angular_velocity()))),
                "moved_m": float(np.linalg.norm(pos - start[p.name])),
                "lowest_below_desk_m": float(self.ground_z - lo[2]),
            }
        rec["settle"] = settle
        rec["settled_plausibly"] = all(
            s["final_speed_mps"] < 0.01 and s["final_angular_rps"] < 0.2 and s["max_speed_mps"] < 1.0
            and s["moved_m"] < 0.15 and s["lowest_below_desk_m"] < 0.003 for s in settle.values())
        return rec

    # ------------------------------------------------------------------------------------------------ presses
    def press(self, fruit_name, pose, aim, depth, speed, tag, stop_on_contact=False, hold_steps=15):
        """Press the knife's ``aim`` point ``depth`` into the top of ``fruit_name`` at ``speed`` (m/s), or until the
        blade first touches it, then hold still; what the rule did meanwhile."""
        import omnigibson as og
        import torch as th

        knife = self.obj(KNIFE)
        m = self.motion(fruit_name, pose, aim)
        quat, origin = m["quat"], m["origin"]
        h_start, h_end = m["top"] + 0.02, m["top"] - depth
        start_pos = np_(self.obj(fruit_name).get_position_orientation()[0])
        self.phase = f"{tag}:approach"
        self.move_knife(origin(h_start) + np.array([0, 0, 0.12]), quat, max_step=0.02)
        self.move_knife(origin(h_start), quat, max_step=0.02)
        n_events, n_log0 = len(self.events), len(self.rule.cut_log)
        self.phase = f"{tag}:press"
        dh = speed * og.sim.get_sim_step_dt()
        h, forces, held = h_start, [], 0
        while True:
            if held == 0:
                h = max(h - dh, h_end)
            knife.set_position_orientation(position=th.tensor(origin(h), dtype=th.float32),
                                           orientation=th.tensor(quat, dtype=th.float32))
            self.wake_cuttables()
            self.step()
            f = {(k.name, c.name): v for (k, c), (v, _) in self.rule.last_window.peak.items()}.get((KNIFE, fruit_name),
                                                                                                   0.0)
            forces.append(round(f, 3))
            if self.obj(fruit_name) is None:
                break  # cut
            if held or (stop_on_contact and f > 0) or h <= h_end:
                held += 1
                if held > hold_steps:
                    break
        self.phase = f"{tag}:lift"
        self.move_knife(origin(h_start) + np.array([0, 0, 0.10]), quat, max_step=0.01)
        self.move_knife(np.array(PARK), [0, 0, 0, 1], max_step=0.03)
        fruit = self.obj(fruit_name)
        records = self.rule.cut_log[n_log0:]
        return {"fruit": fruit_name, "aim": aim, "pose": list(pose), "depth_m": depth, "speed_mps": speed,
                "tag": tag, "forces_n": forces, "max_blade_force_n": max(forces) if forces else 0.0,
                "cut": [r["cuttable"] for r in records if "refused" not in r],
                "refused": [{"cuttable": r["cuttable"], "force_n": r["force_n"], "why": r["refused"]}
                            for r in records if "refused" in r],
                "events": self.events[n_events:],
                "fruit_moved_m": None if fruit is None else
                float(np.linalg.norm(np_(fruit.get_position_orientation()[0]) - start_pos))}

    def gates(self, fruit):
        """Presses with the kinematic knife that must not cut: the handle pushed 1.5 cm into the fruit, and the flat
        of the blade pushed 5 mm into it (the cut it asks for leaves a sliver, which the rule refuses)."""
        out = []
        for aim, rolls, depth, speed in (("handle", (0,), 0.015, 0.05), ("flat", (90, -90), 0.005, 0.05)):
            self.phase = f"{fruit}_{aim}:reset"
            self.reset()
            for _ in range(20):
                self.step()
            pose = next((p for p in ((yaw, 0.0, roll) for roll in rolls for yaw in YAWS)
                         if self.pose_clear(fruit, p, aim, self.motion(fruit, p, aim)["top"] - depth)), None)
            if pose is None:
                out.append({"fruit": fruit, "aim": aim, "skipped": "no clear pose"})
                continue
            rec = self.press(fruit, pose, aim, depth, speed, f"{fruit}_{aim}")
            out.append(rec)
            print(f"[check] gate {aim} on {fruit} pose {pose}: max blade force {rec['max_blade_force_n']:.1f} N, "
                  f"cut {rec['cut']}, refused {[r['why'] for r in rec['refused']]}, fruit moved "
                  f"{(rec['fruit_moved_m'] or 0) * 100:.1f} cm", flush=True)
        return out

    # -------------------------------------------------------------------------------------------- threshold
    def rail(self, force, speed=0.02, damping=5000.0):
        """Put the (dynamic) knife on a prismatic rail along its blade's up axis, anchored to the world where it is,
        with gravity off and a velocity drive pushing it down at ``speed`` with at most ``force`` (N): a hand that
        holds the knife upright and presses with a bounded force. The joint is authored whole before the next step
        (omni.physx ignored a drive added to a joint it had already parsed)."""
        import omnigibson as og
        import omnigibson.lazy as lazy

        pxr = lazy.pxr
        knife = self.obj(KNIFE)
        link = knife.root_link
        pos, quat = (np_(v) for v in knife.get_position_orientation())
        link.prim.GetAttribute("physxRigidBody:disableGravity").Set(True)
        joint = pxr.UsdPhysics.PrismaticJoint.Define(og.sim.stage, RAIL)
        joint.CreateBody1Rel().SetTargets([pxr.Sdf.Path(link.prim_path)])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(pxr.Gf.Vec3f(*pos.tolist()))
        joint.CreateLocalRot0Attr(pxr.Gf.Quatf(float(quat[3]), *map(float, quat[:3])))
        joint.CreateLocalPos1Attr(pxr.Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(pxr.Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        drive = pxr.UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(float(damping))
        drive.CreateTargetVelocityAttr(-float(speed))
        drive.CreateMaxForceAttr(float(force))
        knife.wake()

    def unrail(self):
        import omnigibson as og

        if og.sim.stage.GetPrimAtPath(RAIL):
            og.sim.stage.RemovePrim(RAIL)
        knife = self.obj(KNIFE)
        knife.root_link.prim.GetAttribute("physxRigidBody:disableGravity").Set(False)

    def threshold_press(self, fruit_name, pose, force, tag, steps=45):
        """Rail the knife with its edge 3 mm above the fruit and press with ``force``; what the rule did."""
        import torch as th

        knife = self.obj(KNIFE)
        m = self.motion(fruit_name, pose)
        start = m["origin"](m["top"] + 0.003)
        knife.set_position_orientation(position=th.tensor(start, dtype=th.float32),
                                       orientation=th.tensor(m["quat"], dtype=th.float32))
        knife.set_linear_velocity(th.zeros(3))
        knife.set_angular_velocity(th.zeros(3))
        self.rail(force)
        n_log0 = len(self.rule.cut_log)
        self.phase = f"{tag}:press"
        forces, heights = [], []
        for _ in range(steps):
            self.wake_cuttables()
            knife.wake()
            self.step()
            peak = {(k.name, c.name): v for (k, c), (v, _) in self.rule.last_window.peak.items()}
            forces.append(round(peak.get((KNIFE, fruit_name), 0.0), 3))
            heights.append(float(np_(knife.get_position_orientation()[0])[2]))
            if self.obj(fruit_name) is None:
                break
        records = self.rule.cut_log[n_log0:]
        for _ in range(10):  # the knife drops through the cut to the desk
            self.step()
        self.unrail()
        cut = next((r for r in records if r["cuttable"] == fruit_name and "refused" not in r), None)
        before = [f for f in forces if f > 0]
        return {"fruit": fruit_name, "pose": list(pose), "push_force_n": force, "tag": tag, "forces_n": forces,
                "knife_z_m": heights, "cut": cut is not None,
                "force_at_cut_n": cut["force_n"] if cut else None,
                "steps_to_cut": (len(forces) - 1) if cut else None,
                "blade_force_last_10_median_n": statistics.median(forces[-10:]) if not cut and forces else None,
                "blade_force_max_n": max(forces) if forces else 0.0, "touching_steps": len(before),
                "records": [{"cuttable": r["cuttable"], "force_n": r["force_n"], "refused": r.get("refused")}
                            for r in records]}

    def run_threshold(self):
        """A dynamic knife pressed onto each fruit with a bounded force: below the rule's threshold it must not cut."""
        import omnigibson as og

        self.make_env(self.build_scene(dynamic_knife=True))
        report = {"conditions": {"scene": self.args.scene, "omnigibson": og.__version__,
                                 "min_force_n": self.cutting.SETTINGS.min_force_n,
                                 "press_forces_n": self.args.press_forces}, "presses": []}
        for fruit in self.args.fruits:
            for force in self.args.press_forces:
                self.phase = f"{fruit}_{force}N:reset"
                self.reset()
                for _ in range(20):
                    self.step()
                pose = next((p for p in CANDIDATES[:len(YAWS)] if self.pose_clear(fruit, p)), CANDIDATES[0])
                rec = self.threshold_press(fruit, pose, force, f"{fruit}_{force}N")
                report["presses"].append(rec)
                print(f"[check] press {fruit} with {force} N: cut {rec['cut']} (force at cut {rec['force_at_cut_n']}), "
                      f"blade force max {rec['blade_force_max_n']:.1f} N, last-10 median "
                      f"{rec['blade_force_last_10_median_n']}", flush=True)
        report["summary"] = {f"{p['fruit']} {p['push_force_n']} N": (
            f"cut at {p['force_at_cut_n']:.1f} N after {p['steps_to_cut']} steps" if p["cut"] else
            f"no cut, blade force {p['blade_force_last_10_median_n']} N (max {p['blade_force_max_n']:.1f})")
            for p in report["presses"]}
        return report

    # ------------------------------------------------------------------------------------------- reset probe
    def run_reset_probe(self):
        """Where env.reset() leaves a fruit that a cut had removed: its pose, OmniGibson's aabb, and the bounding box
        of its colliders at its pose, after one env.reset() and again after loading the initial state once more."""
        self.make_env(self.build_scene())
        fruit = self.args.fruits[0]
        self.reset()
        for _ in range(20):
            self.step()
        pose = next(p for p in CANDIDATES if self.pose_clear(fruit, p))
        rec = {"stroke": summarize_stroke(self.stroke(fruit, pose, "probe"))}

        def where(tag):
            o = self.obj(fruit)
            lo, hi = (np_(v) for v in o.aabb)
            clo, chi = self.world_aabb(o)
            rec[tag] = {"position": np_(o.get_position_orientation()[0]).round(4).tolist(),
                        "omnigibson_aabb_center": ((lo + hi) / 2).round(4).tolist(),
                        "collider_aabb_center": ((clo + chi) / 2).round(4).tolist()}
            print(f"[check] {tag}: {rec[tag]}", flush=True)

        self.env.reset()
        where("after_reset_0_steps")
        for _ in range(2):
            self.step()
        where("after_reset_2_steps")
        self.scene.load_state(self.scene._initial_file["state"], serialized=False)
        self.step()
        where("after_reloading_initial_state")
        with open(self.args.scene) as f:
            rec["saved_position"] = json.load(f)["state"]["registry"]["object_registry"][fruit]["root_link"]["pos"]
        return rec

    # -------------------------------------------------------------------------------------------------- cut
    def run_cut(self):
        import omnigibson as og

        scene_json = self.build_scene()
        t = time.perf_counter()
        self.make_env(scene_json)
        report = {"conditions": {"scene": self.args.scene, "load_s": time.perf_counter() - t,
                                 "omnigibson": og.__version__, "settings": vars(self.cutting.SETTINGS).copy(),
                                 "speed_mps": self.args.speed, "physics_dt": og.sim.get_physics_dt(),
                                 "rendering_dt": og.sim.get_rendering_dt(), "ground_plane": self.ground},
                  "strokes": []}
        idle = []
        for _ in range(30):
            idle.append(self.step())
        report["conditions"]["idle_step_median_s"] = statistics.median(idle)
        for fruit in self.args.fruits:
            chosen = []
            for off, roll in OFFSET_ROLL[:self.args.poses]:
                pose = next((p for p in ((yaw, off, roll) for yaw in YAWS)
                             if all(abs((p[0] - c[0] + 180) % 360 - 180) >= 45 for c in chosen)
                             and self.pose_clear(fruit, p)), None)
                if pose is not None:
                    chosen.append(pose)
            report.setdefault("poses", {})[fruit] = chosen
            for k, pose in enumerate(chosen):
                self.phase = f"{fruit}_p{k}:reset"
                self.reset()
                rec_reset = {o: np_(self.obj(o).get_position_orientation()[0]).tolist() for o in self.args.fruits}
                for _ in range(20):  # settle, and let the knife re-arm (rearm_s)
                    self.step()
                rec = self.stroke(fruit, pose, f"{fruit}_p{k}")
                rec["fruit_positions_after_reset"] = rec_reset
                report["strokes"].append(rec)
                print(f"[check] {fruit} pose {pose}: " + summarize_stroke(rec), flush=True)

        report["gates"] = [g for fruit in self.args.fruits for g in self.gates(fruit)]

        # One episode that cuts every fruit, then a save
        self.phase = "episode:reset"
        self.reset()
        for _ in range(20):
            self.step()
        episode = []
        for fruit in self.args.fruits:
            # the trials' first pose for this fruit, unless the pieces of the fruits cut before it are in the way now
            tried = [tuple(s["pose"].values()) for s in report["strokes"] if s["fruit"] == fruit]
            pose = next((p for p in tried + CANDIDATES if self.pose_clear(fruit, p)), None)
            if pose is None:
                continue
            rec = self.stroke(fruit, pose, f"all_{fruit}")
            episode.append(rec)
            print(f"[check] episode {fruit}: " + summarize_stroke(rec), flush=True)
            for _ in range(15):
                self.step()
        report["episode"] = episode
        report["events"] = self.events
        report["save"] = self.save("cut")
        report["summary"] = summarize(report, self.step_times)
        return report

    def save(self, tag):
        """og.sim.save() with ground_plane_info put back, under a timestamped name; what came out."""
        import omnigibson as og

        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"droid_desk_runtime_cutting_scene_state_{tag}_{stamp}.json"
        out = {"path": str(path)}
        t = time.perf_counter()
        og.sim.save(json_paths=[str(path)])
        out["save_s"] = time.perf_counter() - t
        with open(path) as f:
            doc = json.load(f)
        with open(self.args.scene if tag == "cut" else self.args.saved) as f:
            doc["ground_plane_info"] = json.load(f).get("ground_plane_info")
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)
        init = doc["objects_info"]["init_info"]
        reg = doc["state"]["registry"]["object_registry"]
        pieces = {n: a["args"] for n, a in init.items() if a["args"].get("cutting", {}).get("parent")}
        out.update(objects=len(init), objects_with_state=len(reg), fruits_left=[f for f in FRUITS if f in init],
                   pieces={n: {"usd_path": a["usd_path"], "expected_file_hash": a.get("expected_file_hash"),
                               "mass": a.get("mass"), "cutting": a.get("cutting")} for n, a in pieces.items()},
                   knife_cutting=init.get(KNIFE, {}).get("args", {}).get("cutting"))
        return out

    # ------------------------------------------------------------------------------------------------ reload
    def run_reload(self):
        import hashlib
        import omnigibson as og
        from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

        saved = load_json_with_absolute_usd_paths(self.args.saved)
        init = saved["objects_info"]["init_info"]
        reg = saved["state"]["registry"]["object_registry"]
        piece_names = sorted(n for n, a in init.items() if a["args"].get("cutting", {}).get("parent"))
        t = time.perf_counter()
        self.make_env(saved, warm_steps=0)
        report = {"saved": self.args.saved, "load_s": time.perf_counter() - t, "pieces_in_file": piece_names}
        names = {o.name for o in self.scene.objects}
        report["pieces_loaded"] = [n for n in piece_names if n in names]
        report["fruits_loaded"] = [f for f in FRUITS if f in names]
        # The uncut props are the control: this load path moves objects whether or not they were cut (section 15.1)
        controls = [o.name for o in self.scene.objects if o.name in reg and o.name not in piece_names
                    and not o.fixed_base and o not in self.env.robots]

        def gap(n):
            pos = np_(self.obj(n).get_position_orientation()[0])
            return 1e3 * float(np.linalg.norm(pos - np.asarray(reg[n]["root_link"]["pos"])))

        at_load = {n: gap(n) for n in report["pieces_loaded"] + controls}
        for _ in range(10):
            self.step()
        checks = {}
        for n in report["pieces_loaded"]:
            o, a = self.obj(n), init[n]["args"]
            with open(a["usd_path"], "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
            checks[n] = {"hash_matches": digest == a["expected_file_hash"],
                         "mass_args_kg": a["mass"], "mass_physx_kg": float(o.mass),
                         "load_gap_mm": at_load[n], "gap_after_10_steps_mm": gap(n),
                         "generation": a["cutting"]["generation"]}
        report["controls"] = {n: {"load_gap_mm": at_load[n], "gap_after_10_steps_mm": gap(n)} for n in controls}
        start = {n: np_(self.obj(n).get_position_orientation()[0]) for n in report["pieces_loaded"]}
        for _ in range(45):
            self.step()
        for n in report["pieces_loaded"]:
            checks[n]["drift_3s_mm"] = 1e3 * float(np.linalg.norm(np_(self.obj(n).get_position_orientation()[0])
                                                                  - start[n]))
        report["pieces"] = checks
        # Cut a piece again: the largest of the first fruit's
        candidates = [n for n in report["pieces_loaded"] if n.startswith(f"half_{self.args.fruits[0]}")]
        target = max(candidates, key=lambda n: init[n]["args"]["mass"]) if candidates else None
        report["recut"] = None
        if target is not None:
            pose = next((p for p in CANDIDATES if self.pose_clear(target, p)), CANDIDATES[0])
            report["recut"] = self.stroke(target, pose, f"recut_{target}")
            print(f"[check] recut {target}: " + summarize_stroke(report["recut"]), flush=True)
        report["save"] = self.save("reload")
        report["summary"] = {
            "pieces_loaded": f"{len(report['pieces_loaded'])}/{len(piece_names)}",
            "fruits_loaded": report["fruits_loaded"],
            "hashes_match": all(c["hash_matches"] for c in checks.values()),
            "mass_max_abs_err_kg": max(abs(c["mass_args_kg"] - c["mass_physx_kg"]) for c in checks.values())
            if checks else None,
            "load_gap_max_mm": max(c["load_gap_mm"] for c in checks.values()) if checks else None,
            "gap_after_10_steps_max_mm": max(c["gap_after_10_steps_mm"] for c in checks.values()) if checks else None,
            "controls_gap_after_10_steps_mm": {n: round(c["gap_after_10_steps_mm"], 2)
                                               for n, c in report["controls"].items()},
            "drift_3s_max_mm": max(c["drift_3s_mm"] for c in checks.values()) if checks else None,
            "recut": summarize_stroke(report["recut"]) if report["recut"] else None,
            "resave_pieces": len(report["save"]["pieces"]),
            "resave_generations": sorted({p["cutting"]["generation"] for p in report["save"]["pieces"].values()}),
        }
        return report


def summarize_stroke(rec):
    cut = rec.get("cut")
    if cut is None:
        return f"no cut (max force {rec.get('max_force_n', 0):.1f} N)"
    r = cut["record"]
    if "refused" in r:
        return f"refused at {r['force_n']:.1f} N: {r['refused']}"
    t = r["timings"]
    s = rec.get("settle", {})
    return (f"cut at step {cut['step']} ({r['force_n']:.1f} N), env step {cut['env_step_s'] * 1e3:.0f} ms, rule "
            f"{t['rule_step_s'] * 1e3:.0f} ms (geometry {1e3 * (t['clip_hulls_s'] + t['merge_hulls_s'] + t['clip_visual_s'] + t['cap_s']):.0f}, "
            f"write {t['write_s'] * 1e3:.0f}, remove {t['remove_s'] * 1e3:.0f}, add {t['add_s'] * 1e3:.0f}, "
            f"init {t['initialize_s'] * 1e3:.0f}), plane {rec['plane_vs_commanded']['angle_deg']:.3f} deg "
            f"{rec['plane_vs_commanded']['offset_m'] * 1e3:.3f} mm, faces dev "
            f"{max(f['deviation_max_m'] for f in rec['cut_faces_vs_commanded']['faces']) * 1e3:.3f} mm, masses "
            f"{[round(m, 4) for m in rec['masses']['pieces_physx_kg']]}/{rec['masses']['parent_kg']:.3f}, settled "
            f"{rec.get('settled_plausibly')} (max speed {max((v['max_speed_mps'] for v in s.values()), default=0):.2f} "
            f"m/s, moved {max((v['moved_m'] for v in s.values()), default=0) * 100:.1f} cm, below desk "
            f"{max((v['lowest_below_desk_m'] for v in s.values()), default=0) * 1e3:.1f} mm)"
            + (f", also cut {rec['unintended_cuts']}" if rec.get("unintended_cuts") else ""))


def summarize(report, step_times):
    strokes = [s for s in report["strokes"] + report.get("episode", [])]
    made = [s for s in strokes if s.get("cut") and "refused" not in s["cut"]["record"]]
    med = statistics.median

    def stat(key):
        vals = [s["cut"]["record"]["timings"][key] for s in made if key in s["cut"]["record"]["timings"]]
        return {"median_ms": 1e3 * med(vals), "max_ms": 1e3 * max(vals)} if vals else None

    per_fruit = {}
    for s in report["strokes"]:
        per_fruit.setdefault(s["fruit"], []).append(
            "cut" if s.get("cut") and "refused" not in s["cut"]["record"] else
            ("refused" if s.get("cut") else "no cut"))
    idle = report["conditions"]["idle_step_median_s"]
    return {
        "strokes": len(strokes), "cuts": len(made), "per_fruit": per_fruit,
        "env_step_idle_median_ms": 1e3 * idle,
        "env_step_with_cut": {"median_ms": 1e3 * med([s["cut"]["env_step_s"] for s in made]),
                              "max_ms": 1e3 * max(s["cut"]["env_step_s"] for s in made)} if made else None,
        "rule_step": stat("rule_step_s"),
        **{k: stat(f"{k}_s") for k in ("plane", "read_source", "clip_hulls", "merge_hulls", "clip_visual", "cap",
                                       "write", "remove", "add", "initialize", "place")},
        "force_at_cut_n": [round(s["cut"]["record"]["force_n"], 1) for s in made],
        "plane_angle_max_deg": max((s["plane_vs_commanded"]["angle_deg"] for s in made), default=None),
        "plane_offset_max_m": max((s["plane_vs_commanded"]["offset_m"] for s in made), default=None),
        "cut_face_deviation_max_m": max((f["deviation_max_m"] for s in made
                                         for f in s["cut_faces_vs_commanded"]["faces"]), default=None),
        "mass_conservation_max_err_kg": max((abs(sum(s["masses"]["pieces_physx_kg"]) - s["masses"]["parent_kg"])
                                             for s in made), default=None),
        "settled_plausibly": f"{sum(bool(s.get('settled_plausibly')) for s in made)}/{len(made)}",
        "below_desk_mm": {
            "fruits_before_cut_max": 1e3 * max((s["fruit_below_desk_before_m"] for s in strokes), default=0.0),
            "pieces_settled_max": 1e3 * max((v["lowest_below_desk_m"] for s in made for v in s["settle"].values()),
                                            default=0.0)},
        "unintended_cuts": [s["unintended_cuts"] for s in strokes if s.get("unintended_cuts")],
        "gates": [{"fruit": g["fruit"], "aim": g["aim"], "skipped": g.get("skipped"),
                   "max_blade_force_n": g.get("max_blade_force_n"), "cut": g.get("cut"),
                   "refused": [r["why"] for r in g.get("refused", [])], "fruit_moved_m": g.get("fruit_moved_m")}
                  for g in report.get("gates", [])],
        "save": {k: report["save"][k] for k in ("path", "save_s", "objects", "objects_with_state", "fruits_left")}
        | {"pieces": len(report["save"]["pieces"])},
    }


if __name__ == "__main__":
    main()
