# SPDX-License-Identifier: Apache-2.0
"""Episodes for a tofu-cutting policy in the co-simulation: scripted demonstrations and open-loop replays.

Builds on scripts/cutting/tofu_og_cut.py (same scene edit, MPM tofu, held knife, IK arm), but one process
runs many episodes. Each episode (seed = episode id) takes a fresh solver block, shifted and turned on the
board by a seeded random amount, with the knife starting from a seeded random pose above the board:

  expert  the cook's stroke (real2sim.cook_stroke) planned on that block, its tilt and descent speed drawn
          per episode, then the knife lifted out;
  blind   the knife commands of one recorded expert episode, replayed whatever the block's pose is (the
          baseline a policy has to beat: it shows how much of the task needs seeing the tofu);
  policy  a trained policy, asked for the next end-effector pose at every tick by a policy server
          (policy_server.py, in its own environment) over a local socket; the arm moves there linearly
          over the tick's physics steps.

Every 1/--policy-hz s the episode records what a policy sees and does: a fixed camera on the board and one
on the hand (RGB), the arm's joints, fingers and end-effector pose (robot base frame), and the end-effector
pose commanded at that tick. At the end the solver judges the cut. Outputs in --out:
  ep_<seed>.npz   images_front, images_wrist (T, H, W, 3) uint8, state (T, 16), cmd (T, 9: position and
                  the first two rotation columns, base frame), block (7: centre, yaw, size; privileged),
                  cmd_steps (N, 4, 4) world end-effector command at every physics step (for blind replays),
                  steps (N, 10) per physics step: t, knife force xyz [N], edge over the board top [mm], tracking
                  error [mm], then the edge's lowest point z, the board top under it, its x, y [m, world]
  ep_<seed>.json  the draw and the outcome, the episode's place in its process, the knife in the hand
  episodes.jsonl  one outcome line per episode

Example (repository root, Isaac Sim boots; see tofu_og_cut.py for the data path):
    PYTHONPATH=$PWD python scripts/cutting/tofu_policy_episodes.py --mode expert --seeds 0 5 --out results/eps
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tofu_og_cut as T  # noqa: E402 - the co-simulation's scene edit, arm and stroke helpers

from simfoundry.cutting.cosim import KnifeCoupling, matrix_to_quat_xyzw, pose_from_axes  # noqa: E402
from simfoundry.cutting.knife import CHEF_KNIFE  # noqa: E402
from simfoundry.cutting.materials import PRESETS  # noqa: E402
from simfoundry.cutting.real2sim import OBJECTS_INFO, cook_stroke, fit_food, load_reconstruction  # noqa: E402
from simfoundry.cutting.render import CUT_RGB, tofu_look  # noqa: E402


def draw(seed, a):
    """The episode's random choices, from its seed alone (``--nominal``: none, the cook's own stroke)."""
    if a.nominal:
        return {"shift_m": [0.0, 0.0], "yaw_deg": 0.0, "start_m": [0.0, 0.0, 0.0], "tilt_deg": 30.0, "descend_m_s": 0.08}
    rng = np.random.default_rng(seed)
    return {"shift_m": rng.uniform(-a.shift, a.shift, 2).tolist(),
            "yaw_deg": float(rng.uniform(-a.yaw_deg, a.yaw_deg)),
            "start_m": rng.uniform(-a.start_jitter, a.start_jitter, 3).tolist(),
            "tilt_deg": float(rng.uniform(*a.tilt_deg)),
            "descend_m_s": float(rng.uniform(*a.descend))}


def level_board(rec, scene_reg):
    """Lays the board flat by its own shape, for an asset whose frame is not z-up.

    tofu_og_cut.edit_scene levels a board by keeping only the yaw of its root frame, which assumes the asset's
    z is the board's normal; the LiDAR kitchen's board asset has its normal along a slanted local axis, so that
    stood it on edge. Here the board turns by the smallest rotation that makes its thinnest axis (PCA of its
    vertices) vertical, about its centroid, and drops to rest on z = 0. The reconstruction's board mesh moves
    with it (put_on_board reads it). Returns the new root pose (position, xyzw quaternion).
    """
    board = rec.find("board")
    v = board.mesh.vertices
    c = v.mean(axis=0)
    n = np.linalg.svd(v - c, full_matrices=False)[2][2]
    n = n if n[2] >= 0.0 else -n
    k = np.cross(n, [0.0, 0.0, 1.0])
    sin, cos = np.linalg.norm(k), float(n[2])
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]]) / max(sin, 1e-12)
    R_align = np.eye(3) + sin * K + (1.0 - cos) * K @ K
    v2 = (v - c) @ R_align.T + c
    dz = -v2[:, 2].min()
    board.mesh.vertices = v2 + np.array([0.0, 0.0, dz])
    root = scene_reg[board.name]["root_link"]
    R0 = T.quat_xyzw_to_matrix(np.asarray(root["ori"], float))
    pos = R_align @ (np.asarray(root["pos"], float) - c) + c + np.array([0.0, 0.0, dz])
    return pos, matrix_to_quat_xyzw(R_align @ R0)


def put_on_board(rec, fitted, across, board_z=None):
    """The fitted food moved onto the cutting board: centred on it, bottom on its top, long side along ``across``.

    For a scene whose food lies elsewhere (the LiDAR kitchen's tofu sits in a tray). The board's top is the
    depth model's median height over the board's own mask (real2sim.surface_height), as fit_food measures it
    near the food; the food keeps the size fitted where it lay.
    """
    from simfoundry.cutting.real2sim import surface_height

    board = rec.find("board")
    if board_z is None:
        board_z = surface_height(rec, board.name)
    v = board.mesh.vertices
    c = np.array([*(0.5 * (v[:, :2].min(axis=0) + v[:, :2].max(axis=0))), board_z])
    psi = math.atan2(across[1], across[0])
    L = fitted.placement.block_size
    # FoodPlacement.corner is the box's lower corner in the world turned by -yaw (real2sim.place_food).
    corner = T.rz(-psi) @ c - np.array([0.5 * L[0], 0.5 * L[1], 0.0])
    return replace(fitted, board=board, board_z=board_z, placement=replace(fitted.placement, yaw=psi, corner=corner))


def planar(centre, shift, yaw):
    """World map that turns by ``yaw`` about ``centre`` (vertical axis) and then shifts in the plane."""
    G = np.eye(4)
    G[:3, :3] = T.rz(yaw)
    G[:3, 3] = centre - T.rz(yaw) @ centre + np.array([shift[0], shift[1], 0.0])
    return G


class BoardTop:
    """Height of the board's collider tops under world (x, y): the surface the knife can actually meet.

    The hulls are put in the world once (the board does not move), so any orientation works, including a board
    whose own frame is not z-up. Hull planes are precomputed (og_cosim.hull_top rebuilds them every call).
    """

    def __init__(self, hulls, scale, pos, quat):
        from scipy.spatial import ConvexHull

        p = np.asarray(pos, float)
        R = T.quat_xyzw_to_matrix(np.asarray(quat, float))
        self.planes = []
        for h in hulls:
            w = (np.asarray(h, float) * np.asarray(scale, float)[None, :]) @ R.T + p
            eq = ConvexHull(w).equations
            self.planes.append((eq[eq[:, 2] > 1e-9], eq[eq[:, 2] < -1e-9], eq[np.abs(eq[:, 2]) <= 1e-9]))

    def __call__(self, xy):
        xy = np.asarray(xy, float)[:2]
        top = -np.inf
        for up, down, side in self.planes:
            z_top = (-(up[:, 3] + up[:, :2] @ xy) / up[:, 2]).min()
            z_bot = (-(down[:, 3] + down[:, :2] @ xy) / down[:, 2]).max()
            if z_top >= z_bot and (not len(side) or np.all(side[:, :2] @ xy + side[:, 3] <= 1e-12)):
                top = max(top, z_top)
        return top if np.isfinite(top) else float("nan")


def level_board(rec, scene_reg):
    """Lays the board flat by its own shape, for an asset whose frame is not z-up.

    tofu_og_cut.edit_scene levels a board by keeping only the yaw of its root frame, which assumes the asset's
    z is the board's normal; the LiDAR kitchen's board asset has its normal along a slanted local axis, so that
    stood it on edge. Here the board turns by the smallest rotation that makes its thinnest axis (PCA of its
    vertices) vertical, about its centroid, and drops to rest on z = 0. The reconstruction's board mesh moves
    with it (put_on_board reads it). Returns the new root pose (position, xyzw quaternion).
    """
    board = rec.find("board")
    v = board.mesh.vertices
    c = v.mean(axis=0)
    n = np.linalg.svd(v - c, full_matrices=False)[2][2]
    n = n if n[2] >= 0.0 else -n
    k = np.cross(n, [0.0, 0.0, 1.0])
    sin, cos = np.linalg.norm(k), float(n[2])
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]]) / max(sin, 1e-12)
    R_align = np.eye(3) + sin * K + (1.0 - cos) * K @ K
    v2 = (v - c) @ R_align.T + c
    dz = -v2[:, 2].min()
    board.mesh.vertices = v2 + np.array([0.0, 0.0, dz])
    root = scene_reg[board.name]["root_link"]
    R0 = T.quat_xyzw_to_matrix(np.asarray(root["ori"], float))
    pos = R_align @ (np.asarray(root["pos"], float) - c) + c + np.array([0.0, 0.0, dz])
    return pos, matrix_to_quat_xyzw(R_align @ R0)


def put_on_board(rec, fitted, across, board_z=None):
    """The fitted food moved onto the cutting board: centred on it, bottom on its top, long side along ``across``.

    For a scene whose food lies elsewhere (the LiDAR kitchen's tofu sits in a tray). The board's top is the
    depth model's median height over the board's own mask (real2sim.surface_height), as fit_food measures it
    near the food; the food keeps the size fitted where it lay.
    """
    from simfoundry.cutting.real2sim import surface_height

    board = rec.find("board")
    if board_z is None:
        board_z = surface_height(rec, board.name)
    v = board.mesh.vertices
    c = np.array([*(0.5 * (v[:, :2].min(axis=0) + v[:, :2].max(axis=0))), board_z])
    psi = math.atan2(across[1], across[0])
    L = fitted.placement.block_size
    # FoodPlacement.corner is the box's lower corner in the world turned by -yaw (real2sim.place_food).
    corner = T.rz(-psi) @ c - np.array([0.5 * L[0], 0.5 * L[1], 0.0])
    return replace(fitted, board=board, board_z=board_z, placement=replace(fitted.placement, yaw=psi, corner=corner))


def planar(centre, shift, yaw):
    """World map that turns by ``yaw`` about ``centre`` (vertical axis) and then shifts in the plane."""
    G = np.eye(4)
    G[:3, :3] = T.rz(yaw)
    G[:3, 3] = centre - T.rz(yaw) @ centre + np.array([shift[0], shift[1], 0.0])
    return G


class BoardTop:
    """Height of the board's collider tops under world (x, y): the surface the knife can actually meet.

    The board is levelled by ``edit_scene`` (a yaw only), so its link frame differs from the world by a turn
    about z and a shift. Hull planes are precomputed once (og_cosim.hull_top rebuilds them every call).
    """

    def __init__(self, hulls, scale, pos, quat):
        from scipy.spatial import ConvexHull

        self.p = np.asarray(pos, float)
        self.R = T.quat_xyzw_to_matrix(np.asarray(quat, float))
        self.planes = []
        for h in hulls:
            eq = ConvexHull(np.asarray(h, float) * np.asarray(scale, float)[None, :]).equations
            self.planes.append((eq[eq[:, 2] > 1e-9], eq[eq[:, 2] < -1e-9], eq[np.abs(eq[:, 2]) <= 1e-9]))

    def __call__(self, xy):
        loc = self.R[:2, :2].T @ (np.asarray(xy, float)[:2] - self.p[:2])
        top = -np.inf
        for up, down, side in self.planes:
            z_top = (-(up[:, 3] + up[:, :2] @ loc) / up[:, 2]).min()
            z_bot = (-(down[:, 3] + down[:, :2] @ loc) / down[:, 2]).max()
            if z_top >= z_bot and (not len(side) or np.all(side[:, :2] @ loc + side[:, 3] <= 1e-12)):
                top = max(top, z_top)
        return self.p[2] + top if np.isfinite(top) else float("nan")


def capped_coupler(base_cls, f_max, m_max):
    """PhysicsStepCoupler whose wrench on the knife is capped: a solver blow-up must not throw the arm around.

    A policy that jerks the knife can make the explicit MPM step diverge (forces of 1e16 N were seen); passed to
    PhysX unchanged, that wrecked the robot and every later episode of the process. Non-finite wrenches are dropped.
    """

    class Capped(base_cls):
        def _post_step(self, dt):
            super()._post_step(dt)
            if self.wrench is None:
                return
            force, point, moment = (np.asarray(w, float) for w in self.wrench)
            if not (np.all(np.isfinite(force)) and np.all(np.isfinite(moment)) and np.all(np.isfinite(point))):
                self.wrench = None
                self.blown = True
                return
            nf, nm = np.linalg.norm(force), np.linalg.norm(moment)
            if nf > f_max or nm > m_max:
                self.blown = self.blown or nf > 10.0 * f_max
                force = force * min(1.0, f_max / max(nf, 1e-12))
                moment = moment * min(1.0, m_max / max(nm, 1e-12))
                self.wrench = (force, point, moment)

    Capped.blown = False
    return Capped


def limit_step(E_prev, E_tgt, max_m, max_deg):
    """The target moved toward from E_prev by at most max_m and max_deg (a policy's jump is cut, not refused)."""
    d = E_tgt[:3, 3] - E_prev[:3, 3]
    n = float(np.linalg.norm(d))
    ang = T.rotation_deg(E_prev[:3, :3], E_tgt[:3, :3])
    s = min(1.0, max_m / max(n, 1e-12), max_deg / max(ang, 1e-12))
    return T.interpolate(E_prev, E_tgt, s) if s < 1.0 else E_tgt


def rot6(R):
    return np.concatenate([R[:3, 0], R[:3, 1]])


def pose9(a):
    """4x4 pose from position and the first two rotation columns (Gram-Schmidt)."""
    a = np.asarray(a, dtype=float)
    b1 = a[3:6] / np.linalg.norm(a[3:6])
    b2 = a[6:9] - (b1 @ a[6:9]) * b1
    b2 /= np.linalg.norm(b2)
    P = np.eye(4)
    P[:3, :3] = np.stack([b1, b2, np.cross(b1, b2)], axis=1)
    P[:3, 3] = a[:3]
    return P


def pack(a):
    """An array as plain types, so the two environments' numpy versions never meet in a pickle."""
    a = np.ascontiguousarray(a)
    return (a.dtype.str, a.shape, a.tobytes())


class EpisodeRun(T.Run):
    """The co-simulation loop of tofu_og_cut.py, recording policy ticks instead of writing videos."""

    def __init__(self, *args, wrist_offset=None, **kw):
        super().__init__(*args, **kw)
        self.rec = None  # list of ticks while an episode records, else None
        self.cmd = None
        self.cmd_steps = None
        self.edge_min = math.inf  # lowest edge point above the board's collider top under it [m]
        self.S = None
        self.W = None
        self.board_z = None
        self.board_top = None
        self.wrist_offset = wrist_offset
        self.auto = True  # record a tick every 1/fps s inside step(); the policy mode captures itself
        self.overview = []  # (t, image) of the wide camera, when there is one
        self.overview_every = 0  # --overview-fps: a wide frame every this many physics steps while recording (0: off)
        self.overview_dir = None
        self.overview_n = 0
        self.tracker = None  # --track-objects

    def step(self, E_cmd, gripper, K_cmd=None, extra=None):
        self.cmd = E_cmd  # before the step: Run.step records a tick inside it
        K = super().step(E_cmd, gripper, K_cmd=K_cmd, extra=extra)
        if self.cmd_steps is not None:
            self.cmd_steps.append(np.asarray(E_cmd, np.float32))
        if self.S is not None:
            Ks = self.S @ K
            p = pose_from_axes(Ks[:3, 3], Ks[:3, 0], Ks[:3, 1], Ks[:3, 2])
            e = self.W[:3, :3] @ CHEF_KNIFE.lowest_edge_point(p, 100) + self.W[:3, 3]
            top = self.board_top(e[:2])
            # x and y too, so a corrected board-top measure (the LiDAR board's is wrong) can be re-applied later
            self.rows[-1]["edge_x"], self.rows[-1]["edge_y"] = float(e[0]), float(e[1])
            self.rows[-1]["edge_z"] = float(e[2])
            self.rows[-1]["top_z"] = float(top)
            if np.isfinite(top):
                self.edge_min = min(self.edge_min, float(e[2] - top))
                self.rows[-1]["clear_mm"] = 1e3 * float(e[2] - top)
        if self.rec is not None and self.overview_every and self.k % self.overview_every == 0:
            self.wide_frame()
        if self.rec is not None and self.tracker is not None:
            self.tracker.step(self.k * self.dt, self.dt, self.k)
        return K

    def wide_frame(self):
        """One wide-camera frame between the policy's ticks (--overview-fps), written at once as a JPEG named by its
        time. The tofu's points are moved to the solver's state first: otherwise they only move at the ticks."""
        import imageio.v2 as imageio

        if self.points is not None:
            x = self.coupler.coupling.sim.positions()
            self.points.update(x @ self.world_from_sim[:3, :3].T + self.world_from_sim[:3, 3])
        self.og.sim.render()
        im = T.np_(self.cams["overview"].get_obs()[0]["rgb"])[:, :, :3].astype(np.uint8)
        imageio.imwrite(self.overview_dir / f"f{self.overview_n:05d}_t{self.k * self.dt:08.3f}.jpg", im, quality=92)
        self.overview_n += 1

    def frame(self):
        if self.rec is not None and self.auto:
            self.capture()

    def cuts_done(self):
        """Cuts finished so far: the cuts the coupling began, less the one under way (its edge has not yet risen back
        out of the food, the coupling not re-armed). The same count a policy-mode episode can be given."""
        if self.coupler is None:
            return 0
        cpl = self.coupler.coupling
        return int(cpl.sim.n_cuts - (0 if cpl.armed else 1))

    def capture(self):
        """Renders the two cameras and records a tick: what the policy sees, and the command now."""
        import torch as th

        t0 = time.perf_counter()
        if self.points is not None:
            x = self.coupler.coupling.sim.positions()
            self.points.update(x @ self.world_from_sim[:3, :3].T + self.world_from_sim[:3, 3])
        E = self.arm.eef()
        C = E @ self.wrist_offset
        self.cams["wrist"].set_position_orientation(position=th.tensor(C[:3, 3], dtype=th.float32),
                                                    orientation=th.tensor(matrix_to_quat_xyzw(C[:3, :3]),
                                                                          dtype=th.float32))
        t1 = time.perf_counter()
        self.og.sim.render()
        t2 = time.perf_counter()
        imgs = {n: T.np_(c.get_obs()[0]["rgb"])[:, :, :3].astype(np.uint8) for n, c in self.cams.items()}
        Eb = self.arm.base_inv @ E
        Cb = self.arm.base_inv @ self.cmd
        state = np.concatenate([self.arm.joints(), self.arm.fingers(), Eb[:3, 3], matrix_to_quat_xyzw(Eb[:3, :3])])
        if "overview" in imgs:
            self.overview.append((self.k * self.dt, imgs["overview"]))
        self.rec.append({"front": imgs["front"], "wrist": imgs["wrist"], "state": state.astype(np.float32),
                         "cmd": np.concatenate([Cb[:3, 3], rot6(Cb[:3, :3])]).astype(np.float32),
                         "t": self.k * self.dt, "phase": self.phase, "cuts_done": self.cuts_done()})
        self.frame_times.append({"points_s": t1 - t0, "render_s": t2 - t1, "read_s": time.perf_counter() - t2})
        return self.rec[-1]


class ObjectTracker:
    """--track-objects: chosen scene objects at every physics step while an episode records — pose, velocity, the
    contact force on each from the robot, the knife, the board, the floor and anything else (PhysX contact reports,
    impulse / dt), and every other step the nearest robot link or knife (gap between collision boundary points).
    For "why does the plate move when the knife barely touches it" (2026-10-28); written next to each episode."""

    PARTS = ("robot", "knife", "board", "floor", "other")

    def __init__(self, env, robot, knife, board, names):
        self.objs = []
        for n in names:
            o = env.scene.object_registry("name", n)
            if o is None:  # a category instead of a name
                o = next((x for x in env.scene.objects if getattr(x, "category", None) == n), None)
            if o is None:
                print(f"[eps] track: no object {n}", flush=True)
            else:
                self.objs.append(o)
        self.robot_path, self.knife_path, self.board_path = robot.prim_path, knife.prim_path, board.prim_path
        self.links = [(f"robot:{k}", lk) for k, lk in robot.links.items()] + [("knife", knife.root_link)]
        self.rows = []

    def part(self, path):
        for name, prefix in (("robot", self.robot_path), ("knife", self.knife_path), ("board", self.board_path)):
            if path.startswith(prefix):
                return name
        return "floor" if "ground_plane" in path else "other"

    def describe(self):
        """What PhysX holds for each tracked object, its links' bound physics materials, and the floor's."""
        from pxr import PhysxSchema, UsdPhysics, UsdShade

        def material(prim):
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            if not mat:
                return None
            p = mat.GetPrim()
            api, px = UsdPhysics.MaterialAPI(p), PhysxSchema.PhysxMaterialAPI(p)
            return {"path": str(p.GetPath()), "static_friction": api.GetStaticFrictionAttr().Get(),
                    "dynamic_friction": api.GetDynamicFrictionAttr().Get(), "restitution": api.GetRestitutionAttr().Get(),
                    "friction_combine": px.GetFrictionCombineModeAttr().Get() if p.HasAPI(PhysxSchema.PhysxMaterialAPI)
                    else None}

        out = {}
        for o in self.objs:
            lo, hi = (T.np_(v) for v in o.aabb)
            out[o.name] = {"category": o.category, "mass_kg": float(o.mass), "aabb_extent_m": (hi - lo).tolist(),
                           "aabb_lo": lo.tolist(), "aabb_hi": hi.tolist(),
                           "links": {ln: {"mass_kg": float(lk.mass),
                                          "collision_meshes": {mn: material(m.prim) for mn, m in lk.collision_meshes.items()}}
                                     for ln, lk in o.links.items()}}
        import omnigibson as og
        from pxr import Usd

        floor = {}
        gp = og.sim.stage.GetPrimAtPath("/World/ground_plane")
        if gp and gp.IsValid():
            for p in Usd.PrimRange(gp):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    floor[str(p.GetPath())] = material(p)
        scene_prim = og.sim._physics_context.get_current_physics_scene_prim()
        out["_floor_colliders"] = floor
        out["_physics_scene_default_material"] = material(scene_prim) if scene_prim else None
        return out

    def step(self, t, dt, k):
        row = {"t": t, "pos": [], "quat": [], "lin": [], "ang": [], "force": [], "near": []}
        near = k % 2 == 0
        if near:
            pts = []
            for name, lk in self.links:
                q = lk.collision_boundary_points_world
                if q is not None:
                    pts.append((name, q))
        for o in self.objs:
            p, q = o.get_position_orientation()
            row["pos"].append(T.np_(p))
            row["quat"].append(T.np_(q))
            row["lin"].append(T.np_(o.get_linear_velocity()))
            row["ang"].append(T.np_(o.get_angular_velocity()))
            f = {pt: np.zeros(3) for pt in self.PARTS}
            me = o.root_link.prim_path
            for c in o.root_link.contact_list():
                other = c.body1 if c.body0 == me else c.body0
                imp = c.impulse  # a structured (x, y, z) record from the contact sensor
                imp = np.array([imp["x"], imp["y"], imp["z"]], float) if getattr(imp, "dtype", None) is not None \
                    and imp.dtype.names else np.asarray(imp, float)
                f[self.part(str(other))] += (1.0 if c.body0 == me else -1.0) * imp
            row["force"].append([float(np.linalg.norm(f[pt])) / dt for pt in self.PARTS])
            if near:
                import torch as th

                mine = o.collision_points_world
                best = (np.inf, "")
                for name, q in pts:
                    d = float(th.cdist(mine, q).min())
                    if d < best[0]:
                        best = (d, name)
                row["near"].append(best)
            else:
                row["near"].append((np.nan, ""))
        self.rows.append(row)

    def save(self, path):
        if not self.rows:
            return
        names = sorted({n for r in self.rows for _, n in r["near"] if n})
        np.savez_compressed(path, objects=np.array([o.name for o in self.objs]), parts=np.array(self.PARTS),
                            t=np.array([r["t"] for r in self.rows]),
                            pos=np.array([r["pos"] for r in self.rows], np.float64),
                            quat=np.array([r["quat"] for r in self.rows], np.float64),
                            lin=np.array([r["lin"] for r in self.rows], np.float32),
                            ang=np.array([r["ang"] for r in self.rows], np.float32),
                            force=np.array([r["force"] for r in self.rows], np.float32),
                            near_dist=np.array([[d for d, _ in r["near"]] for r in self.rows], np.float32),
                            near_link=np.array([[names.index(n) if n else -1 for _, n in r["near"]] for r in self.rows]),
                            link_names=np.array(names))
        self.rows = []


def judge(sim, run, board_pose0, board):
    """The episode's outcome from the solver's state and the board's displacement."""
    groups = sim.piece_groups()[sim.pieces()]
    counts = np.array([int((groups == g).sum()) for g in np.unique(groups)])
    frac = float(np.sort(counts)[-2] / counts.sum()) if len(counts) >= 2 else 0.0
    smallest = float(counts.min() / counts.sum()) if len(counts) >= 2 else 0.0
    p1 = T.np_(board.get_position_orientation()[0])
    out = {"n_cuts": int(sim.n_cuts), "cut_complete": bool(sim.cut_complete(0)) if sim.n_cuts else False,
           "connected_pieces": int(sim.connected_pieces()),
           "piece_masses_g": sorted((round(1e3 * sim.particle_mass * int(c), 1) for c in counts), reverse=True),
           "second_piece_frac": round(frac, 4), "smallest_piece_frac": round(smallest, 4),
           "ligament_mm": round(1e3 * sim.ligament_height(0, sim.positions()), 2) if sim.n_cuts else None,
           "edge_min_above_board_mm": round(1e3 * run.edge_min, 2),
           "board_moved_mm": round(1e3 * float(np.linalg.norm(p1 - board_pose0)), 2)}
    return out


def success(o, a):
    return bool(o["cut_complete"] and o["connected_pieces"] >= 2 and o["second_piece_frac"] >= a.min_piece
                and o["edge_min_above_board_mm"] >= -a.max_dig_mm and o["board_moved_mm"] <= a.max_board_move_mm
                and o["held"] and o["hand_drift_deg"] <= a.max_hand_drift_deg)


def dice_expert(run, sim, W, K_in_E, counts, plan, a, lead):
    """--mode dice: counts[0] parallel press cuts one way, then counts[1] with the blade turned a quarter.

    A one-armed robot cannot turn the tofu as a cook does, so the blade turns instead. ``plan`` (dice_knife_plan) gives
    each direction's yaw (which side the tip points to) and where along the straight edge the food's middle sits, so
    that the 20 cm blade and the hand clear the other objects. Each cut is planned from the particles as they are when
    it starts (the knife can push pieces): the food's extent across the blade, split evenly. The blade is level and
    pressed straight down (KnifeTrajectory.strokes, one stroke each, hovering 12 mm over the food so that the coupling
    begins a new cut every stroke).
    """
    from simfoundry.cutting.knife import KnifeTrajectory

    span = edge_span(sim.knife_geometry)
    log = []
    for d, (k_n, (yaw, u_t, _)) in enumerate(zip(counts, plan)):
        e_u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        axis = int(np.argmax(np.abs([-math.sin(yaw), math.cos(yaw)])))  # cuts are spaced along the blade's normal
        for i in range(1, k_n + 1):
            x = sim.positions()
            lo, hi = np.percentile(x, 0.2, axis=0), np.percentile(x, 99.8, axis=0)  # a stray particle moves no cut
            c = 0.5 * (lo + hi)
            c[axis] = lo[axis] + i * (hi[axis] - lo[axis]) / (k_n + 1)
            # the food as it is now along the blade must lie on its usable span: move the planned place the least
            along = (x[:, :2] - c[:2]) @ e_u[:2]
            a_lo, a_hi = np.percentile(along, 0.2), np.percentile(along, 99.8)
            u_lo, u_hi = span[0] - a_lo, span[1] - a_hi  # u_t such that u_t + along stays inside the span
            u_cut = float(np.clip(u_t, u_lo, u_hi)) if u_lo <= u_hi else 0.5 * (u_lo + u_hi)
            o = c[:2] - u_cut * e_u[:2]
            traj = KnifeTrajectory.strokes([(o[0], o[1], yaw)], top_z=float(hi[2]), board_z=sim.board_z, start_time=0.0,
                                           clearance=0.012, board_gap=T.BOARD_GAP, descend_speed=a.dice_descend,
                                           settle=0.05)
            start = T.knife_world(traj.state(0.0)[0], W)
            if i == 1:  # a new direction: up, turn in the air, down beside the first cut
                up = run.knife_pose()
                up[2, 3] = max(up[2, 3], start[2, 3] + 0.04)
                T.move_knife(run, K_in_E, up, T.CLOSE, speed=0.10, settle=0.1, phase="lift")
                above = start.copy()
                above[2, 3] += 0.04
                T.move_knife(run, K_in_E, above, T.CLOSE, ang_speed=a.approach_ang_speed, settle=0.2, phase="turn")
            T.move_knife(run, K_in_E, start, T.CLOSE, speed=0.05, ang_speed=a.approach_ang_speed, settle=0.2)
            n0 = sim.n_cuts
            T.cook(run, K_in_E, traj, W, lead=lead, after=0.2)
            ew = run.knife_pose()
            log.append({"direction": d, "index": i, "yaw_deg": round(math.degrees(yaw), 1), "at_m": round(float(c[axis]), 4),
                        "axis": axis, "food_along_edge_m": round(u_cut, 4),
                        "began_cut": sim.n_cuts > n0, "cut": sim.n_cuts - 1 if sim.n_cuts > n0 else None,
                        "knife_end_z_m": round(float(ew[2, 3]), 4)})
    up = run.knife_pose()
    up[2, 3] += 0.08
    T.move_knife(run, K_in_E, up, T.CLOSE, speed=0.10, settle=0.2, phase="lift")
    return {"counts": list(counts), "directions": [{"yaw_deg": round(math.degrees(y), 1), "food_along_edge_m": round(u, 4),
                                                    "clearance_m": round(c, 4)} for y, u, c in plan], "strokes": log}


HAND_RADIUS = 0.045  # the gripper around the handle's centre, for clearance [m]
HEEL_MARGIN = 0.010  # food kept this far from the blade's heel end, where the edge stops [m]
BELLY_RISE_OK = 0.5e-3  # the curved part of the edge may cover food where it has risen no more than this [m]


def edge_span(g):
    """Where along the blade (u from the edge's midpoint) food can lie and still be cut through by a level press: from
    HEEL_MARGIN inside the heel to where the belly has risen BELLY_RISE_OK (a cut counts as through within the bottom
    layer of particles, so a fraction of a millimetre of rise is still a through cut)."""
    end = g.half_length - g.belly_length
    if g.belly_length > 0.0 and g.tip_rise > 0.0:
        end += g.belly_length * math.sqrt(BELLY_RISE_OK / g.tip_rise)
    return -g.half_length + HEEL_MARGIN, end


def dice_obstacles(env, skip):
    """World AABBs (lo, hi) of the scene's objects a knife could hit: all but the names in ``skip`` (robot, knife, board,
    food) and backgrounds (an object whose footprint is over four times the board's: a scanned counter or room)."""
    board = env.scene.object_registry("name", skip[2])
    blo, bhi = (T.np_(v) for v in board.aabb)
    area_b = float(np.prod(bhi[:2] - blo[:2]))
    out = []
    for ob in env.scene.objects:
        if ob.name in skip:
            continue
        try:
            lo, hi = (T.np_(v) for v in ob.aabb)
        except Exception:  # no collision points (visual only): nothing to hit
            continue
        if float(np.prod(hi[:2] - lo[:2])) > 4.0 * area_b:
            continue
        out.append((ob.name, lo, hi))
    return out


def dice_clearance(W, sim, c, yaw, u_t, obstacles):
    """Clearance [m] of the knife at the bottom of a press cut through ``c`` (solver frame) with the food's middle
    ``u_t`` along the blade from the edge's midpoint: the blade between edge and spine against every obstacle whose
    height range it shares, and the hand (a ball round the handle) against all. Negative: they overlap."""
    from simfoundry.cutting.og_cosim import handle_centre

    g = sim.knife_geometry
    e_u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    e_v = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    o = np.array([c[0], c[1], sim.board_z]) - u_t * e_u
    us = np.linspace(-g.half_length, g.half_length, 41)
    lo_s = o + us[:, None] * e_u + g.edge_height(us)[:, None] * np.array([0, 0, 1.0])
    hi_s = o + us[:, None] * e_u + g.spine_height(us)[:, None] * np.array([0, 0, 1.0])
    hc = handle_centre()
    hand = o + hc[0] * e_u + hc[1] * e_v + hc[2] * np.array([0, 0, 1.0])
    to_w = lambda p: p @ W[:3, :3].T + W[:3, 3]  # noqa: E731
    lo_w, hi_w, hand_w = to_w(lo_s), to_w(hi_s), to_w(hand[None])[0]
    best = np.inf
    for _, a, b in obstacles:
        gap_xy = np.maximum(np.maximum(a[None, :2] - lo_w[:, :2], lo_w[:, :2] - b[None, :2]), 0.0)
        d = np.linalg.norm(gap_xy, axis=1)
        shares = (lo_w[:, 2] < b[2] + 0.005) & (hi_w[:, 2] > a[2] - 0.005)
        if shares.any():
            best = min(best, float(d[shares].min()) if (d[shares] > 0).all() else -1.0)
        gh = np.maximum(np.maximum(a - hand_w, hand_w - b), 0.0)
        best = min(best, float(np.linalg.norm(gh)) - HAND_RADIUS)
    return best


def dice_knife_plan(W, sim, counts, obstacles, base_xy, yaw_a):
    """Per cutting direction, the tip's side (yaw or yaw + pi) and the food's place along the straight edge that
    keep the knife clearest of other objects over all of that direction's cuts (planned on the block as it lies);
    ties go to the handle nearer the robot. Returns [(yaw, u_t, clearance)] for the two directions."""
    from simfoundry.cutting.og_cosim import handle_centre

    g = sim.knife_geometry
    lo, hi = sim.block_min, sim.block_max
    straight = edge_span(g)
    plan = []
    for d, (k_n, yaw0) in enumerate(((counts[0], yaw_a), (counts[1], yaw_a + math.pi / 2))):
        e_u0 = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
        axis = int(np.argmax(np.abs([-math.sin(yaw0), math.cos(yaw0)])))
        # the first direction's cuts open gaps about a blade's thickness wide across this blade: the food is longer here
        spread = counts[0] * g.spine_thickness if d == 1 else 0.0
        half_food = 0.5 * float(np.abs(e_u0[:2]) @ (hi - lo)[:2]) + 0.5 * spread
        u_lo, u_hi = straight[0] + half_food, straight[1] - half_food  # the food stays under the straight edge
        cands = []
        for yaw in (yaw0, yaw0 + math.pi):
            for u_t in np.linspace(u_lo, u_hi, 5) if u_hi > u_lo else [0.5 * (u_lo + u_hi)]:
                cl = np.inf
                for i in range(1, k_n + 1):
                    c = 0.5 * (lo + hi)
                    c[axis] = lo[axis] + i * (hi[axis] - lo[axis]) / (k_n + 1)
                    cl = min(cl, dice_clearance(W, sim, c, yaw, u_t, obstacles))
                h = np.array([*(0.5 * (lo + hi))[:2], 0.0]) + (handle_centre()[0] - u_t) * np.array([math.cos(yaw), math.sin(yaw), 0])
                near = float(np.linalg.norm((W[:3, :3] @ h + W[:3, 3])[:2] - base_xy))
                cands.append((round(cl, 3), -near, yaw, float(u_t)))
        cl, _, yaw, u_t = max(cands)
        plan.append((math.atan2(math.sin(yaw), math.cos(yaw)), u_t, cl))
    return plan


def dice_place(W_of, sim, counts, obstacles, base_xy, yaw_a, board_lo, board_hi, need=0.02, step=0.01):
    """Where on the board to put the food so that both directions' knives clear other objects by ``need``: the
    planned (draw's) place if it does, else the nearest place on a grid over the board that does (the food kept on
    the board with a 1 cm margin), else the place with the most clearance. ``W_of(delta)`` is the episode's map with
    an extra shift delta (world xy). Returns (delta, plan, clear): clear is False when no place on the board clears
    (a scene too crowded for this knife), and then the plan is the most clearing one, for the record."""
    lo, hi = sim.block_min, sim.block_max
    corners = np.array([[x, y, lo[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
    best = None
    span = np.arange(-0.15, 0.15 + 1e-9, step)
    for dx, dy in sorted(((a, b) for a in span for b in span), key=lambda p: math.hypot(*p)):
        W = W_of(np.array([dx, dy]))
        cw = corners @ W[:3, :3].T + W[:3, 3]
        if (cw[:, :2] < board_lo[:2] + 0.01).any() or (cw[:, :2] > board_hi[:2] - 0.01).any():
            continue
        plan = dice_knife_plan(W, sim, counts, obstacles, base_xy, yaw_a)
        cl = min(p[2] for p in plan)
        if cl >= need:
            return np.array([dx, dy]), plan, True
        if best is None or cl > best[2]:
            best = (np.array([dx, dy]), plan, cl)
    if best is None:  # the food does not fit on the board at all
        return np.zeros(2), dice_knife_plan(W_of(np.zeros(2)), sim, counts, obstacles, base_xy, yaw_a), False
    return best[0], best[1], False


def dice_outcome(sim, counts, block_min, block_max):
    """Pieces of a diced block against the grid it was meant to make: per-cut completion, the pieces' masses against
    the ideal (the block's mass over the grid's count), and particles that left the block's footprint."""
    x = sim.positions()
    groups = sim.piece_groups()[sim.pieces()]
    n = np.bincount(groups)
    n = n[n > 0]
    expected = (counts[0] + 1) * (counts[1] + 1)
    ideal = n.sum() / expected
    ratio = np.sort(n / ideal)[::-1]
    real = ratio[ratio >= 0.1]  # crumbs under a tenth of a piece are counted apart
    lo, hi = np.asarray(block_min, float), np.asarray(block_max, float)
    out_xy = (x[:, 0] < lo[0] - 0.03) | (x[:, 0] > hi[0] + 0.03) | (x[:, 1] < lo[1] - 0.03) | (x[:, 1] > hi[1] + 0.03)
    below = x[:, 2] < sim.board_z - 1e-3
    complete = [bool(sim.cut_complete(c)) for c in range(sim.n_cuts)]
    return {"expected_pieces": int(expected), "planned_cuts": int(sum(counts)), "n_cuts": int(sim.n_cuts),
            "cut_complete": complete, "all_cuts_complete": bool(sim.n_cuts == sum(counts) and all(complete)),
            "pieces": int(len(n)), "pieces_over_tenth": int(len(real)),
            "piece_mass_g": [round(1e3 * sim.particle_mass * int(v), 2) for v in np.sort(n)[::-1]],
            "mass_ratio_to_ideal": [round(float(r), 3) for r in ratio],
            "mass_cv": round(float(real.std() / real.mean()), 4) if len(real) else None,
            "ratio_min": round(float(real.min()), 3) if len(real) else None,
            "ratio_max": round(float(real.max()), 3) if len(real) else None,
            "crumbs_mass_frac": round(float(ratio[ratio < 0.1].sum() / expected), 4),
            "escaped_frac": round(float((out_xy | below).mean()), 5)}


def dice_success(o, dz, moved_mm, a):
    """Draft judgement (proposed 2026-10-28, not yet agreed). basic: every planned stroke began a cut, every cut is
    through, and at least 90 % of the grid's pieces exist. strict: basic, every piece 0.5-1.5 times the ideal mass and
    no crumbs over 2 % of the block, under 0.5 % of the particles out of the footprint, the knife held and turned
    <= 5 deg in the hand, the board moved <= 2 mm, and no other object moved more than 10 mm."""
    basic = bool(dz["all_cuts_complete"] and dz["pieces_over_tenth"] >= 0.9 * dz["expected_pieces"])
    strict = bool(basic and dz["ratio_min"] is not None and dz["ratio_min"] >= 0.5 and dz["ratio_max"] <= 1.5
                  and dz["crumbs_mass_frac"] <= 0.02 and dz["escaped_frac"] <= 0.005 and o["held"]
                  and o["hand_drift_deg"] <= a.max_hand_drift_deg and o["board_moved_mm"] <= a.max_board_move_mm
                  and (moved_mm is None or moved_mm <= 10.0))
    return basic, strict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, default=Path("Data/tofu_chefd"))
    ap.add_argument("--photo", type=Path, default=Path("Data/tofu_reference/chefd_frame_17.9.png"))
    ap.add_argument("--mode", choices=["expert", "blind", "policy", "dice"], default="expert",
                    help="dice: the expert dices the block (--dice K1 K2 press cuts one way, then the other)")
    ap.add_argument("--dice", type=int, nargs=2, default=[3, 3], metavar=("K1", "K2"))
    ap.add_argument("--dice-descend", type=float, default=0.04, help="dice: the press cut's speed [m/s]")
    ap.add_argument("--dice-plan-only", action="store_true", help="dice: write each seed's placement and knife plan, no episode")
    ap.add_argument("--dice-skip-unclear", action="store_true", help="dice: skip a seed whose knife plan does not clear (demos)")
    ap.add_argument("--dice-eval", action="store_true",
                    help="with --mode policy: a dicing episode (same food placement as the dice demos, dice outcome); the "
                         "policy is sent the finished-cut count too")
    ap.add_argument("--floor-ref", choices=["collider", "solver"], default="collider",
                    help="--edge-floor-mm against the board's collider top, or against the tofu solver's board plane "
                         "(the LiDAR board's collider-top measure is wrong). A rule that knows the board's height.")
    ap.add_argument("--policy-port", type=int, default=6000, help="policy: the policy server's port")
    ap.add_argument("--episode-s", type=float, default=12.0, help="policy: episode length [s]")
    ap.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute",
                    help="policy: the action is the hand pose to reach, or its change from the hand's pose now")
    ap.add_argument("--edge-floor-mm", type=float, default=None,
                    help="policy: a RULE, not the policy: lift any commanded pose whose knife edge would go below the "
                         "board's collider top + this [mm] (it knows the board's height; report it apart)")
    ap.add_argument("--max-step-m", type=float, default=0.03, help="policy: largest hand move per tick [m]")
    ap.add_argument("--max-step-deg", type=float, default=20.0, help="policy: largest hand turn per tick [deg]")
    ap.add_argument("--wrench-cap", type=float, nargs=2, default=[50.0, 5.0],
                    help="largest force [N] and moment [N m] of the tofu passed to PhysX")
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 5], help="first seed and count")
    ap.add_argument("--blind-from", type=Path, default=None, help="blind: the expert episode npz to replay")
    ap.add_argument("--nominal", action="store_true", help="no random draw: the block where it was fitted, the cook's stroke")
    ap.add_argument("--material", default="firm_tofu", choices=sorted(PRESETS))
    ap.add_argument("--dx", type=float, default=4.0e-3)
    ap.add_argument("--physics-hz", type=float, default=120.0)
    ap.add_argument("--policy-hz", type=float, default=10.0)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--track-objects", nargs="*", default=[],
                    help="object names (or categories) to log at every physics step with their contacts (ep_*_track.npz)")
    ap.add_argument("--diag-objects", nargs="*", default=[],
                    help="DIAGNOSIS ONLY, never a scene setting: objects whose mass / friction the next two options change")
    ap.add_argument("--diag-mass-scale", type=float, default=1.0)
    ap.add_argument("--diag-friction", type=float, default=None, help="static and dynamic friction of their materials")
    ap.add_argument("--overview-fps", type=float, default=0.0,
                    help="with --overview: also write wide frames at this rate of simulated time (JPEG, for videos)")
    ap.add_argument("--overview", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="add a wide third camera that sees the whole robot and the board (pictures only, not policy input)")
    ap.add_argument("--warmup-renders", type=int, default=15,
                    help="renders before an episode's first tick, so DLSS's history holds the new block (3 left ghosts)")
    ap.add_argument("--physx-enhanced-determinism", action="store_true",
                    help="set PhysX's enhanced determinism on the physics scene before it is simulated (repeatability test)")
    ap.add_argument("--shift", type=float, default=0.02, help="block shift on the board, +- [m]")
    ap.add_argument("--yaw-deg", type=float, default=20.0, help="block turn, +- [deg]")
    ap.add_argument("--start-jitter", type=float, default=0.02, help="knife start, +- per axis [m]")
    ap.add_argument("--tilt-deg", type=float, nargs=2, default=[20.0, 40.0])
    ap.add_argument("--descend", type=float, nargs=2, default=[0.05, 0.10], help="descent speed range [m/s]")
    ap.add_argument("--robot-standoff", type=float, default=0.52)
    ap.add_argument("--robot-side", choices=["behind", "stage14"], default="behind",
                    help="behind: behind the board facing the clip's camera (tofu_og_cut.py); stage14: on the line from "
                         "where stage 14 put the robot to the food, facing the food (a top-down capture has no 'behind')")
    ap.add_argument("--fit-support", default="board", help="what the food rests on in the frame (fit_food's board)")
    ap.add_argument("--food-on-board", action="store_true", help="move the fitted food onto the cutting board")
    ap.add_argument("--food-shape", choices=["box", "hull", "mesh"], default="box",
                    help="the solid the solver fills (real2sim.fit_food); mesh is the scene's food as reconstructed")
    ap.add_argument("--no-photo-fit", action="store_true", help="keep the scene's food size (fit_food fit=False)")
    ap.add_argument("--mass-from-scene", action="store_true",
                    help="density = the scene's mass for the food (stage 11, the VL model's estimate) over the solid's "
                         "volume, so the solver's food weighs what the scene says")
    ap.add_argument("--food-height", type=float, default=None,
                    help="block height [m] instead of the fitted one (a setting while the scene's thickness is open)")
    ap.add_argument("--food-mass-g", type=float, default=None,
                    help="the food's mass [g]: with --food-density the box's height is mass / (density x the fitted "
                         "footprint), so a mass estimate and a density set the thickness instead of a hand-picked one")
    ap.add_argument("--food-density", type=float, default=None,
                    help="the food's density [kg/m3] (stage 11's VL answer for the LiDAR tofu: 1100); else the preset's")
    ap.add_argument("--food-mass-source", default=None, help="where the food's mass came from, recorded with each episode")
    ap.add_argument("--food-thickness-mm", type=float, default=None,
                    help="the food's measured thickness [mm]: the box's height, the footprint staying as fitted (the "
                         "mass follows from the density)")
    ap.add_argument("--food-thickness-source", default=None, help="where --food-thickness-mm came from, recorded")
    ap.add_argument("--food-density-source", default=None, help="where --food-density came from, recorded")
    ap.add_argument("--level-board", action="store_true",
                    help="lay the board flat by its thinnest axis (an asset whose frame is not z-up); its top = mesh top")
    ap.add_argument("--approach-ang-speed", type=float, default=0.4,
                    help="largest turn rate on the way to the stroke [rad/s]: faster, the IK falls behind by a centimetre")
    ap.add_argument("--wrist", type=float, nargs=3, default=[0.06, 0.0, -0.10],
                    help="wrist camera in the end-effector frame [m]; it looks along the fingers")
    ap.add_argument("--min-piece", type=float, default=0.2)
    ap.add_argument("--max-dig-mm", type=float, default=1.0)
    ap.add_argument("--max-board-move-mm", type=float, default=2.0)
    ap.add_argument("--max-hand-drift-deg", type=float, default=5.0)
    ap.add_argument("--clear", nargs="*", default=["paper_towel"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dicing = args.mode == "dice" or args.dice_eval  # the expert dices, or a policy is judged on dicing
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()

    import imageio.v2 as imageio

    rec = load_reconstruction(args.scene)
    H0, W0_ = imageio.imread(args.photo).shape[:2]
    if args.food_height is not None and args.food_shape != "box":
        raise SystemExit("--food-height sets a box's height; it has no meaning for --food-shape " + args.food_shape)
    if args.food_mass_g is not None and (args.food_density is None or args.food_shape != "box"
                                         or args.food_height is not None or args.mass_from_scene):
        raise SystemExit("--food-mass-g sets a box's height: it needs --food-density, --food-shape box, and neither "
                         "--food-height nor --mass-from-scene")
    if args.food_thickness_mm is not None and (args.food_shape != "box" or args.food_height is not None
                                               or args.food_mass_g is not None or args.mass_from_scene):
        raise SystemExit("--food-thickness-mm sets a box's height: it needs --food-shape box, and none of --food-height, "
                         "--food-mass-g, --mass-from-scene")
    fitted = fit_food(rec, (W0_, H0), board=args.fit_support, shape=args.food_shape, fit=not args.no_photo_fit)
    if args.robot_side == "stage14":
        scene0 = json.loads((args.scene / T.SCENE_JSON).read_text())
        r0 = np.asarray(scene0["state"]["registry"]["object_registry"]["robot0"]["root_link"]["pos"], float)
        target = rec.find("board").mesh.vertices.mean(axis=0) if args.food_on_board else fitted.shape.vertices.mean(axis=0)
        facing = (target[:2] - r0[:2]) / np.linalg.norm(target[:2] - r0[:2])
        across = np.array([-facing[1], facing[0]])  # the robot's left: the knife's tip points there
    board_pose = None
    if args.level_board:
        scene0 = json.loads((args.scene / T.SCENE_JSON).read_text())
        board_pose = level_board(rec, scene0["state"]["registry"]["object_registry"])
    if args.food_on_board:
        top = float(rec.find("board").mesh.vertices[:, 2].max()) if args.level_board else None
        fitted = put_on_board(rec, fitted, across, board_z=top)
    if args.food_height is not None:  # the bottom stays on the board, the block grows or shrinks upwards
        bs = np.array(fitted.placement.block_size, float)
        bs[2] = args.food_height
        fitted = replace(fitted, placement=replace(fitted.placement, block_size=bs))
    if args.food_mass_g is not None:  # the bottom stays on the board, the footprint stays as fitted
        bs = np.array(fitted.placement.block_size, float)
        bs[2] = 1e-3 * args.food_mass_g / (args.food_density * bs[0] * bs[1])
        fitted = replace(fitted, placement=replace(fitted.placement, block_size=bs))
    if args.food_thickness_mm is not None:  # the bottom stays on the board, the footprint stays as fitted
        bs = np.array(fitted.placement.block_size, float)
        bs[2] = 1e-3 * args.food_thickness_mm
        fitted = replace(fitted, placement=replace(fitted.placement, block_size=bs))
    place = fitted.placement
    material = PRESETS[args.material]
    if args.food_density is not None:
        material = replace(material, density=args.food_density)
    food_volume = float(np.prod(place.block_size)) if args.food_shape == "box" else float(place.local_mesh.volume)
    stage11 = json.loads((args.scene / OBJECTS_INFO).read_text())
    stage11_mass = next((float(e["mass"]) for e in stage11.values() if e.get("name") == fitted.food.name), None)
    if args.mass_from_scene:
        material = replace(material, density=stage11_mass / food_volume)
    food_params = {
        "mass_kg": material.density * food_volume,
        "mass_source": args.food_mass_source or ("stage11" if args.mass_from_scene else "density x solid volume"),
        "thickness_m": float(place.block_size[2]),
        "thickness_source": args.food_thickness_source or (
            "--food-height" if args.food_height is not None else "mass / density / footprint"
            if args.food_mass_g is not None else "--food-thickness-mm" if args.food_thickness_mm is not None
            else "photo fit" if not args.no_photo_fit else "scene mesh"),
        "density": material.density,
        "density_source": args.food_density_source or (
            "stage11 mass / solid volume" if args.mass_from_scene else "--food-density"
            if args.food_density is not None else f"preset {args.material}"),
        "footprint_m": place.block_size[:2].tolist(), "stage11_mass_kg": stage11_mass, "shape": args.food_shape}
    print(f"[eps] food: {args.food_shape}, {np.round(1e3 * place.block_size, 1).tolist()} mm, {1e6 * food_volume:.1f} cm3, "
          f"density {material.density:.0f} kg/m3, {1e3 * material.density * food_volume:.1f} g", flush=True)

    from simfoundry import configure_omnigibson_data_path

    configure_omnigibson_data_path(force=False)
    import omnigibson as og  # noqa: E402
    from omnigibson.macros import gm  # noqa: E402

    from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros  # noqa: E402

    apply_teleop_omnigibson_macros(enable_tr=False)
    gm.ENABLE_OBJECT_STATES = False
    dt = 1.0 / args.physics_hz
    og.launch(physics_dt=dt, rendering_dt=dt, sim_step_dt=dt, device="cpu")
    if args.physx_enhanced_determinism:
        # Identical runs part before the policy acts (evals_det, 19:11). A scene-creation flag, so it is set while the
        # simulator is still stopped; the thread count (/persistent/physics/numThreads) is left alone, because Kit
        # may write /persistent settings back to the config every Isaac Sim on this machine reads.
        from pxr import PhysxSchema  # noqa: E402

        scene_prim = og.sim._physics_context.get_current_physics_scene_prim()
        PhysxSchema.PhysxSceneAPI.Apply(scene_prim).CreateEnableEnhancedDeterminismAttr(True)
    import warp as wp  # noqa: E402,F401 - after og.launch (Kit's own Warp)
    from omnigibson.controllers import IsGraspingState  # noqa: E402

    from simfoundry.cutting.mpm import MPMCuttingSim, SimConfig  # noqa: E402
    from simfoundry.cutting.og_cosim import FoodPoints, PhysicsStepCoupler, collider_hulls, handle_centre, write_knife_usd  # noqa: E402,E501
    from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths  # noqa: E402
    from omnigibson.utils.asset_utils import get_dataset_path  # noqa: E402

    config = SimConfig(dx=args.dx)
    Coupler = capped_coupler(PhysicsStepCoupler, *args.wrench_cap)

    def new_sim():
        return MPMCuttingSim(material, place.block_size, knife=CHEF_KNIFE, config=config, device="cuda:0",
                             inside=None if args.food_shape == "box" else place.inside)

    sim = new_sim()
    S0 = place.sim_from_world(sim.block_min)
    W0 = np.linalg.inv(S0)
    if args.robot_side == "stage14":
        left_sim = S0[:3, :3] @ np.array([across[0], across[1], 0.0])
        knife_yaw = 0.0 if left_sim[0] >= 0.0 else math.pi
    else:
        right_sim = S0[:3, :3] @ rec.cam_to_world[:3, 0]
        knife_yaw = 0.0 if right_sim[0] >= 0.0 else math.pi
    lo, hi = sim.block_min, sim.block_max
    centre_w = W0[:3, :3] @ (0.5 * (lo + hi)) + W0[:3, 3]
    traj0 = cook_stroke(sim, knife_yaw, board_gap=T.BOARD_GAP)
    poses = [traj0.state(t)[0] for t in np.arange(0.0, traj0.duration, 0.05)]
    handles = np.array([(T.knife_world(p, W0) @ np.append(handle_centre(), 1.0))[:3] for p in poses])
    if args.robot_side == "stage14":
        robot_pos = np.array([*(handles[:, :2].mean(axis=0) - args.robot_standoff * facing), 0.0])
        robot_yaw = math.atan2(facing[1], facing[0])
    else:
        robot_pos = np.array([handles[:, 0].mean(), handles[:, 1].mean() + args.robot_standoff, 0.0])
        robot_yaw = -0.5 * math.pi

    knife_usd = write_knife_usd(out / "knife" / "chef_knife.usd")
    scene = load_json_with_absolute_usd_paths(str(args.scene / T.SCENE_JSON))
    b = scene["objects_info"]["init_info"][fitted.board.name]["args"]
    board_usd = Path(get_dataset_path(b["dataset_name"])) / "objects" / b["category"] / b["model"] / "usd"
    board_usd = board_usd / f"{b['model']}.usd"
    u = np.linspace(lo[0] - 0.01, hi[0] + 0.01, 25)
    cut_xy = (np.stack([u, np.full_like(u, 0.5 * (lo[1] + hi[1])), np.full_like(u, sim.board_z)], axis=1)
              @ W0[:3, :3].T + W0[:3, 3])[:, :2]
    scale0 = list(scene["objects_info"]["init_info"][fitted.board.name]["args"].get("scale", [1.0, 1.0, 1.0]))
    changes = T.edit_scene(scene, rec=rec, fitted=fitted, robot_pos=robot_pos, robot_yaw=robot_yaw,
                           arm_controller=T.arm_controller_config("ik", 150.0, 10.0), knife_usd=knife_usd,
                           board_hulls=collider_hulls(board_usd), cut_xy=cut_xy, clear=args.clear)
    if board_pose is not None:  # edit_scene's yaw-only levelling and z scale do not hold for this board: undo them
        scene["objects_info"]["init_info"][fitted.board.name]["args"]["scale"] = scale0
        scene["state"]["registry"]["object_registry"][fitted.board.name]["root_link"] = {
            "pos": [float(x) for x in board_pose[0]], "ori": [float(x) for x in board_pose[1]],
            "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]}
        changes["board_levelled_by_shape"] = {"pos": [float(x) for x in board_pose[0]], "scale": scale0}
    # Cameras: one fixed on the board (where tofu_og_cut.py puts its close-up, a little further back, aimed
    # at the block's nominal centre and never moved), one carried by the hand (posed every tick).
    az, el = math.radians(-50.0), math.radians(32.0)
    eye = centre_w + 0.45 * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    sensors = [{"sensor_type": "VisionSensor", "name": n, "modalities": ["rgb"],
                "sensor_kwargs": {"image_height": args.img, "image_width": args.img, "clipping_range": [0.01, 20.0]},
                "position": [float(v) for v in p], "orientation": [float(v) for v in q]}
               for n, p, q in (("front", eye, T.look_at(eye, centre_w)), ("wrist", eye, T.look_at(eye, centre_w)))]
    if args.overview:
        # From the side of the robot-to-food line, a little in front, high enough to see the whole arm.
        v = (robot_pos[:2] - centre_w[:2]) / np.linalg.norm(robot_pos[:2] - centre_w[:2])
        side = np.array([-v[1], v[0]])
        target_o = np.array([*(0.55 * centre_w[:2] + 0.45 * robot_pos[:2]), 0.25])
        d = side - 0.45 * v
        eye_o = target_o + np.array([*(1.7 * d / np.linalg.norm(d)), 0.75])
        sensors.append({"sensor_type": "VisionSensor", "name": "overview", "modalities": ["rgb"],
                        "sensor_kwargs": {"image_height": args.overview[1], "image_width": args.overview[0],
                                          "clipping_range": [0.01, 20.0]},
                        "position": [float(x) for x in eye_o], "orientation": [float(x) for x in T.look_at(eye_o, target_o)]})
    hz = args.physics_hz
    env = og.Environment(configs=dict(
        env=dict(action_frequency=hz, rendering_frequency=hz, physics_frequency=hz, external_sensors=sensors),
        scene=dict(type="Scene", scene_file=scene, use_floor_plane=True, floor_plane_visible=True, use_skybox=True,
                   include_robots=True)))
    env.reset()
    if args.physx_enhanced_determinism:  # still set once the scene is loaded and playing?
        sp = og.sim._physics_context.get_current_physics_scene_prim()
        print("[eps] physx enhanced determinism:", sp.GetPath(),
              PhysxSchema.PhysxSceneAPI(sp).GetEnableEnhancedDeterminismAttr().Get(), flush=True)
    cams = {n: env.external_sensors[n] for n in ("front", "wrist") + (("overview",) if args.overview else ())}
    robot = env.robots[0]
    knife = env.scene.object_registry("name", T.KNIFE)
    board = env.scene.object_registry("name", fitted.board.name)
    arm = T.Arm(robot)
    bp, bq = board.get_position_orientation()
    board_pose0 = (bp.clone(), bq.clone())
    lo_b, hi_b = (T.np_(v) for v in board.aabb)
    c_b = 0.5 * (lo_b + hi_b)
    board_diag = {"pos": T.np_(bp).tolist(), "quat": T.np_(bq).tolist(), "aabb_lo": lo_b.tolist(), "aabb_hi": hi_b.tolist(),
                  "collider_top_at_aabb_centre": board_top(c_b[:2]) if "board_top" in dir() else None}
    board_top = BoardTop(collider_hulls(board_usd), scene["objects_info"]["init_info"][fitted.board.name]["args"]["scale"],
                         T.np_(bp), T.np_(bq))
    board_diag["collider_top_at_aabb_centre"] = board_top(c_b[:2])
    print("[eps] board:", json.dumps(board_diag), flush=True)
    # The wrist camera looks along the end effector's z (out of the fingers), its image up along -x.
    wrist = np.eye(4)
    wrist[:3, :3] = np.stack([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], axis=1)
    wrist[:3, 3] = args.wrist
    run = EpisodeRun(og, env, arm, knife, None, cams, out, args.policy_hz, set(), wrist_offset=wrist)
    run.grasp_true = IsGraspingState.TRUE
    for _ in range(20):
        run.step(arm.eef(), T.OPEN)
    grasp = T.grasp_knife(run, knife, robot, arm)
    K_in_E = np.asarray(grasp["knife_in_eef"])
    run.K_in_E = K_in_E
    print("[eps] grasp:", json.dumps({k: v for k, v in grasp.items() if k != "knife_in_eef"}), flush=True)
    if not grasp["held"]:
        raise SystemExit("the gripper did not take the knife")
    (out / "setup.json").write_text(json.dumps({"scene_changes": changes, "grasp": grasp, "args": vars(args),
                                                "robot_pos": robot_pos.tolist(), "dx": args.dx,
                                                "block_size": place.block_size.tolist(),
                                                "food_volume_cm3": 1e6 * food_volume, "food_params": food_params,
                                                "particle_mass_sum_g": 1e3 * sim.particle_mass * sim.n_particles},
                                               indent=1, default=str))
    if args.track_objects:
        run.tracker = ObjectTracker(env, robot, knife, board, args.track_objects)
        loaded = {"as_loaded": run.tracker.describe()}
        if args.diag_objects:  # DIAGNOSIS ONLY: changed in this process, never written into the scene
            for n in args.diag_objects:
                o = env.scene.object_registry("name", n)
                if args.diag_mass_scale != 1.0:
                    o.root_link.mass = float(o.root_link.mass) * args.diag_mass_scale
                if args.diag_friction is not None:
                    for mat in (o._link_physics_materials or {}).values():
                        mat.set_static_friction(args.diag_friction)
                        mat.set_dynamic_friction(args.diag_friction)
            loaded["after_diagnosis_change"] = run.tracker.describe()
            loaded["diagnosis"] = {"objects": args.diag_objects, "mass_scale": args.diag_mass_scale,
                                   "friction": args.diag_friction, "note": "진단, 장면 설정 아님"}
        if args.overview:  # to mark tracked objects in the wide frames: world point x -> [x 1] V P (row vectors)
            og.sim.render()
            cp = cams["overview"].camera_parameters
            loaded["overview_camera"] = {"view": T.np_(cp["cameraViewTransform"]).reshape(4, 4).tolist(),
                                         "projection": T.np_(cp["cameraProjection"]).reshape(4, 4).tolist(),
                                         "resolution": [int(v) for v in cp["renderProductResolution"]]}
        (out / "objects.json").write_text(json.dumps(loaded, indent=1, default=str))
        print("[eps] tracked objects:", [o.name for o in run.tracker.objs], flush=True)
    if dicing:
        obstacles = dice_obstacles(env, [robot.name, knife.name, board.name, fitted.food.name])
        board_lo, board_hi = (T.np_(v) for v in board.aabb)
        others0 = [(env.scene.object_registry("name", n), env.scene.object_registry("name", n).get_position_orientation())
                   for n, _, _ in obstacles]
        print("[eps] dice obstacles:", [n for n, _, _ in obstacles], flush=True)
        if run.tracker is None:  # every object the knife could hit, without naming any
            run.tracker = ObjectTracker(env, robot, knife, board, [n for n, _, _ in obstacles])
    blind = np.load(args.blind_from) if args.mode == "blind" else None
    conn = None
    if args.mode == "policy":
        from multiprocessing.connection import Client

        conn = Client(("127.0.0.1", args.policy_port), authkey=b"tofu")
    home0 = T.knife_world(traj0.state(0.0)[0], W0)
    home0[2, 3] += 0.08
    # The arm's IK takes one damped step from wherever the joints are and holds nothing in its null space,
    # so a 7-joint arm's posture wanders from episode to episode (one process drifted into a posture where 6 of
    # 7 strokes ploughed). Every episode starts from this posture, reached once from the robot's reset pose.
    T.move_knife(run, K_in_E, home0, T.CLOSE, ang_speed=args.approach_ang_speed, settle=0.5, phase="home")
    q_ref, E_ref = arm.joints(), arm.eef()

    first, count = args.seeds
    log = open(out / "episodes.jsonl", "a")
    points_n = 0
    for seed in range(first, first + count):
        d = draw(seed, args)
        if args.dice_plan_only:  # the planning step alone: where the food goes and how the knife is held, no episode
            sim_p = new_sim()
            W_of = lambda delta: planar(centre_w, np.asarray(d["shift_m"]) + delta, math.radians(d["yaw_deg"])) @ W0  # noqa
            delta, kplan, ok = dice_place(W_of, sim_p, args.dice, obstacles, arm.base[:2, 3], knife_yaw, board_lo, board_hi)
            lo_b, hi_b = sim_p.block_min, sim_p.block_max
            corners = np.array([[x, y, lo_b[2]] for x in (lo_b[0], hi_b[0]) for y in (lo_b[1], hi_b[1])])
            Wp = W_of(delta)
            plan_rec = {"seed": seed, "draw": d, "plan_clear": bool(ok), "food_shift_mm": [round(1e3 * float(v), 1) for v in delta],
                        "directions": [{"yaw_deg": round(math.degrees(y), 1), "food_along_edge_m": round(u, 4),
                                        "clearance_m": round(c, 4)} for y, u, c in kplan],
                        "food_corners_world": (corners @ Wp[:3, :3].T + Wp[:3, 3]).round(4).tolist(),
                        "board_aabb": [board_lo.round(4).tolist(), board_hi.round(4).tolist()],
                        "obstacles": [[n, a.round(4).tolist(), b.round(4).tolist()] for n, a, b in obstacles]}
            (out / f"plan_{seed:05d}.json").write_text(json.dumps(plan_rec, indent=1))
            print("[eps] dice plan:", json.dumps({k: plan_rec[k] for k in ("seed", "plan_clear", "food_shift_mm", "directions")}),
                  flush=True)
            del sim_p
            continue
        t_ep = time.time()
        # How far the last episode moved the knife in the hand (logged), then the same start for every episode, the
        # first included: let go, put the arm back in the reference posture, take the knife again. grasp_knife builds
        # the knife's frame from the fingers where the hand is, so the setup grasp (taken where the robot stands
        # after loading) holds the knife 30 deg off this one: a first episode that kept it started with the hand
        # turned 30 deg from every later episode and from 176 of the 180 LiDAR demos, and the image policy missed.
        D = np.linalg.inv(K_in_E) @ np.linalg.inv(arm.eef()) @ run.knife_pose()
        slip_mm, slip_deg = 1e3 * float(np.linalg.norm(D[:3, 3])), T.rotation_deg(D[:3, :3], np.eye(3))
        import torch as th

        for _ in range(10):
            run.step(arm.eef(), T.OPEN)
        idx = th.tensor(arm.arm_idx, dtype=th.long)
        robot.set_joint_positions(th.tensor(q_ref, dtype=th.float32), indices=idx)
        robot.set_joint_velocities(th.zeros(len(arm.arm_idx)), indices=idx)
        for _ in range(10):
            run.step(E_ref, T.OPEN)
        # the board too: a hard strike in the last episode may have pushed it
        board.set_position_orientation(position=board_pose0[0], orientation=board_pose0[1])
        board.set_linear_velocity(th.zeros(3))
        board.set_angular_velocity(th.zeros(3))
        grasp = T.grasp_knife(run, knife, robot, arm)
        K_in_E = np.asarray(grasp["knife_in_eef"])
        run.K_in_E = K_in_E
        regrasp = {"slip_mm": round(slip_mm, 2), "slip_deg": round(slip_deg, 2),
                   "posture_err_rad": round(float(np.abs(arm.joints() - q_ref).max()), 4)}
        if dicing:  # the other objects too: a knife sweeping over the board can push them
            for ob, (p_o, q_o) in others0:
                ob.set_position_orientation(position=p_o, orientation=q_o)
                ob.set_linear_velocity(th.zeros(3))
                ob.set_angular_velocity(th.zeros(3))
        sim = new_sim()
        W = planar(centre_w, d["shift_m"], math.radians(d["yaw_deg"])) @ W0
        if dicing:  # the food where both directions' knives clear the other objects
            W_of = lambda delta: planar(centre_w, np.asarray(d["shift_m"]) + delta, math.radians(d["yaw_deg"])) @ W0  # noqa
            dice_delta, dice_kplan, dice_ok = dice_place(W_of, sim, args.dice, obstacles, arm.base[:2, 3], knife_yaw,
                                                         board_lo, board_hi)
            W = W_of(dice_delta)
            dice_plan = {"counts": list(args.dice), "food_shift_mm": [round(1e3 * float(v), 1) for v in dice_delta],
                         "plan_clear": bool(dice_ok), "directions": [{"yaw_deg": round(math.degrees(y), 1),
                                                                      "food_along_edge_m": round(u, 4), "clearance_m": round(c, 4)}
                                                                     for y, u, c in dice_kplan]}
            print(f"[eps] dice place: shift {np.round(1e3 * dice_delta, 1)} mm, plan {dice_kplan}", flush=True)
            if args.dice_skip_unclear and not dice_ok:  # demos: no knife plan clears the other objects here
                (out / f"skipped_{seed:05d}.json").write_text(json.dumps({"seed": seed, "reason": "plan_clear false",
                                                                          "draw": d}))
                print(f"[eps] seed {seed}: skipped, no clear dice plan", flush=True)
                continue
        S = np.linalg.inv(W)
        traj = cook_stroke(sim, knife_yaw, tilt=math.radians(d["tilt_deg"]), descend_speed=d["descend_m_s"],
                           board_gap=T.BOARD_GAP)
        # To the episode's start pose before the block exists to the arm (not recorded).
        if blind is not None:  # where the replayed episode started, so its first command is no jump
            home = blind["cmd_steps"][0].astype(np.float64) @ K_in_E
        else:
            home = home0.copy()
            home[:3, 3] += np.asarray(d["start_m"])
        T.move_knife(run, K_in_E, home, T.CLOSE, ang_speed=args.approach_ang_speed, settle=0.3, phase="home")
        coupling = KnifeCoupling(sim, S, settle=0.15, linger=1.0)
        coupler = Coupler(coupling, knife.root_link.prim_path, apply_wrench=True)
        albedo, _ = tofu_look(sim.x0, sim.skin, cloth=0.0)
        if run.points is not None:
            run.points.points.GetPrim().SetActive(False)
        run.points = FoodPoints(og.sim.stage, f"/World/mpm_tofu_{points_n}", sim.skin,
                                1.6 * args.dx / config.particles_per_axis, None, CUT_RGB, albedo=albedo)
        points_n += 1
        run.world_from_sim = coupling.world_from_sim()
        run.points.update(sim.positions() @ W[:3, :3].T + W[:3, 3])
        run.coupler = coupler
        run.S, run.W, run.board_z, run.edge_min, run.board_top = S, W, sim.board_z, math.inf, board_top
        # OmniGibson renders with DLSS (simulator.py: /rtx/post/dlss/execMode), which blends earlier frames: after
        # the block is swapped, the first few renders show it half transparent or show the last episode's pieces.
        for _ in range(args.warmup_renders):
            og.sim.render()
        board_p0 = T.np_(board.get_position_orientation()[0])
        coupler.start()
        run.rec, run.cmd_steps = [], []
        run.rows = []
        if args.overview and args.overview_fps > 0:  # dense wide frames for a video of this episode
            run.overview_every = max(1, round(args.physics_hz / args.overview_fps))
            run.overview_dir = out / "overview_frames" / f"ep{seed:05d}"
            run.overview_dir.mkdir(parents=True, exist_ok=True)
            run.overview_n = 0
        k0 = 0
        if args.mode == "expert":
            start = T.knife_world(traj.state(0.0)[0], W)
            above = start.copy()
            above[2, 3] += 0.06
            T.move_knife(run, K_in_E, above, T.CLOSE, ang_speed=args.approach_ang_speed, settle=0.2)
            T.move_knife(run, K_in_E, start, T.CLOSE, speed=0.05, ang_speed=args.approach_ang_speed, settle=0.3)
            T.cook(run, K_in_E, traj, W, lead=1.0 / hz, after=0.5)
            up = run.knife_pose()
            up[2, 3] += 0.08
            T.move_knife(run, K_in_E, up, T.CLOSE, speed=0.10, settle=0.2, phase="lift")
        elif args.mode == "dice":
            dice_plan.update(dice_expert(run, sim, W, K_in_E, args.dice, dice_kplan, args, 1.0 / hz))
        elif args.mode == "blind":
            for E_cmd in blind["cmd_steps"]:
                run.step(E_cmd.astype(np.float64), T.CLOSE)
        else:
            conn.send({"cmd": "reset"})
            conn.recv()
            run.auto = False
            run.phase = "policy"
            blk_now = np.concatenate([W[:3, :3] @ (0.5 * (sim.block_min + sim.block_max)) + W[:3, 3],
                                      [math.atan2(W[1, 0], W[0, 0])], sim.block_max - sim.block_min])
            E_prev = run.cmd.copy()
            floor_hits = 0
            for _ in range(int(round(args.episode_s * args.policy_hz))):
                tick = run.capture()
                conn.send({"cmd": "act", "state": pack(tick["state"]), "env_state": pack(blk_now.astype(np.float32)),
                           "front": pack(tick["front"]), "wrist": pack(tick["wrist"]), "cuts_done": int(tick["cuts_done"])})
                a = conn.recv()
                a9 = np.frombuffer(a[2], dtype=a[0]).reshape(-1)
                if len(a9) == 8:  # a policy trained without the near-constant r21 column: refill it, Gram-Schmidt does the rest
                    a9 = np.append(a9, 0.0)
                if args.action_mode == "delta":  # position change in the base frame, turn in the hand's frame
                    Eb = arm.base_inv @ arm.eef()
                    D = pose9(a9)
                    T_b = np.eye(4)
                    T_b[:3, :3] = Eb[:3, :3] @ D[:3, :3]
                    T_b[:3, 3] = Eb[:3, 3] + D[:3, 3]
                    E_tgt = arm.base @ T_b
                else:
                    E_tgt = arm.base @ pose9(a9)
                E_tgt = limit_step(E_prev, E_tgt, args.max_step_m, args.max_step_deg)
                if args.edge_floor_mm is not None:
                    Kt = S @ (E_tgt @ K_in_E)
                    kp = pose_from_axes(Kt[:3, 3], Kt[:3, 0], Kt[:3, 1], Kt[:3, 2])
                    ew = W[:3, :3] @ CHEF_KNIFE.lowest_edge_point(kp, 100) + W[:3, 3]
                    top = board_top(ew[:2]) if args.floor_ref == "collider" else float(W[2, 3] + sim.board_z)
                    if np.isfinite(top) and ew[2] < top + 1e-3 * args.edge_floor_mm:
                        E_tgt = E_tgt.copy()
                        E_tgt[2, 3] += top + 1e-3 * args.edge_floor_mm - ew[2]
                        floor_hits += 1
                if coupler.blown:
                    break
                for j in range(1, run.every + 1):
                    E = T.interpolate(E_prev, E_tgt, j / run.every)
                    run.step(E, T.CLOSE)
                E_prev = E_tgt
            run.auto = True
        coupler.stop()
        rows = run.rows[k0:]
        fmag = np.array([math.sqrt(r.get("fx", 0) ** 2 + r.get("fy", 0) ** 2 + r.get("fz", 0) ** 2) for r in rows])
        i_max = int(np.argmax(fmag)) if len(fmag) else 0
        o = judge(sim, run, board_p0, board)
        held = robot.is_grasping(arm.arm, candidate_obj=knife) == run.grasp_true
        drift = max((r.get("hand_drift_deg", 0.0) for r in rows), default=0.0)
        adv = [r for r in rows if r.get("advanced")]
        o.update(held=bool(held), hand_drift_deg=round(float(drift), 3), sim_blown=bool(coupler.blown),
                 floor_hits=int(floor_hits) if args.mode == "policy" else None,
                 max_force_n=round(float(max((math.sqrt(r.get("fx", 0) ** 2 + r.get("fy", 0) ** 2 + r.get("fz", 0) ** 2)
                                               for r in rows), default=0.0)), 4),
                 max_force_at=dict(t=round(rows[i_max]["t"] - rows[0]["t"], 3), phase=rows[i_max]["phase"],
                                   clear_mm=round(rows[i_max].get("clear_mm", float("nan")), 3)) if rows else None,
                 physics_steps=len(rows), solver_steps=len(adv), ticks=len(run.rec),
                 mpm_step_ms=round(1e3 * float(np.mean([r["mpm_wall_s"] for r in adv])), 2) if adv else None,
                 wall_s=round(time.time() - t_ep, 1))
        o["success"] = success(o, args)
        if dicing:
            moved = None
            if run.tracker is not None and run.tracker.rows:
                p0, p1 = np.array(run.tracker.rows[0]["pos"]), np.array([r["pos"] for r in run.tracker.rows])
                far = 1e3 * np.linalg.norm(p1[:, :, :2] - p0[None, :, :2], axis=2).max(axis=0)
                o["other_objects_moved_mm"] = {ob.name: round(float(v), 2) for ob, v in zip(run.tracker.objs, far)}
                moved = float(far.max())
            o["dice"] = dice_outcome(sim, args.dice, sim.block_min, sim.block_max)
            o["dice_plan"] = dice_plan
            o["dice_basic"], o["dice_strict"] = dice_success(o, o["dice"], moved, args)
            o["dice_strict"] = bool(o["dice_strict"] and dice_plan["plan_clear"])  # an unclear plan is never strict
            o["sim_time_s"] = round(len(rows) * run.dt, 2)
        # order_in_process: where the episode ran in its process, so a table can show whether anything depends on it
        info = {"seed": seed, "mode": args.mode, "draw": d, "outcome": o, "regrasp": regrasp,
                "order_in_process": seed - first, "start_grasp": "regrasp",
                "knife_in_eef": K_in_E.tolist(), "board_pos0": board_p0.tolist(), "food_params": food_params}
        R = run.rec
        blk = np.concatenate([W[:3, :3] @ (0.5 * (sim.block_min + sim.block_max)) + W[:3, 3],
                              [math.atan2(W[1, 0], W[0, 0])], sim.block_max - sim.block_min])
        np.savez_compressed(out / f"ep_{seed:05d}.npz", images_front=np.stack([r["front"] for r in R]),
                            images_wrist=np.stack([r["wrist"] for r in R]), state=np.stack([r["state"] for r in R]),
                            cmd=np.stack([r["cmd"] for r in R]), t=np.array([r["t"] for r in R]),
                            block=blk.astype(np.float32), cmd_steps=np.stack(run.cmd_steps),
                            steps=np.array([[r["t"], r.get("fx", 0.0), r.get("fy", 0.0), r.get("fz", 0.0),
                                             r.get("clear_mm", np.nan), r.get("pos_err_mm", np.nan),
                                             r.get("edge_z", np.nan), r.get("top_z", np.nan),
                                             r.get("edge_x", np.nan), r.get("edge_y", np.nan)] for r in rows],
                                           dtype=np.float32),
                            step_phase=np.array([r["phase"] for r in rows]),
                            tick_phase=np.array([r.get("phase", "") for r in R]),
                            tick_cuts_done=np.array([r.get("cuts_done", 0) for r in R], np.int16),
                            **({"final_x_world": (sim.positions() @ W[:3, :3].T + W[:3, 3]).astype(np.float32),
                                "final_piece": sim.piece_groups()[sim.pieces()].astype(np.int16)}
                               if dicing else {}))
        (out / f"ep_{seed:05d}.json").write_text(json.dumps(info, indent=1))
        if run.tracker is not None:
            run.tracker.save(out / f"ep_{seed:05d}_track.npz")
        log.write(json.dumps(info) + "\n")
        log.flush()
        print(f"[eps] seed {seed}: " + json.dumps(o), flush=True)
        if run.overview:
            (out / "overview").mkdir(exist_ok=True)
            t0o = run.overview[0][0]
            for t_o, im in run.overview[::5]:
                imageio.imwrite(out / "overview" / f"ep{seed:05d}_t{t_o - t0o:05.1f}.png", im)
            run.overview = []
        if seed - first < 2:
            for n in ("front", "wrist"):
                for i in np.linspace(0, len(R) - 1, 6).astype(int):
                    (out / "frames").mkdir(exist_ok=True)
                    imageio.imwrite(out / "frames" / f"ep{seed:05d}_{n}_{i:03d}.png", R[i][n])
        run.rec, run.cmd_steps, run.S = None, None, None
        run.rows = []  # one episode at a time in memory
        del coupling, coupler, sim
        gc.collect()
    log.close()
    print(f"[eps] done {count} episodes in {time.time() - t_wall:.0f} s", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
