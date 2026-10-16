# SPDX-License-Identifier: Apache-2.0
"""Build a fruit-slicing OmniGibson scene from SimFoundry's published `droid_desk_serve_fruits` scene.

Run in the `3dgrut` env (has pxr + trimesh + scipy; the simfoundry env has no standalone pxr):
    ~/miniforge3/envs/3dgrut/bin/python repro/fruit_slicing/make_slicing_scene.py

The published fruits (apple, pear, orange, banana) are single rigid USD meshes with no "half" models, so
OmniGibson's own Sliceable/SlicingRule (which swaps a whole object for two pre-modelled halves from the
BEHAVIOR-1K dataset) cannot be used as-is. This script makes the missing halves and mirrors that rule:

1. Plan one vertical cut per fruit, in the world frame of the scene, choosing the cut plane and the knife's
   heading so that the knife and gripper at the bottom of the stroke stay clear of the other objects and the
   gripper stays within the Franka's reach (banana: roughly across its long axis; round fruits: any plane
   through the centre).
2. Split the fruit's visual mesh by that plane (faces assigned by centroid side, so the original welded
   topology, face-varying UVs and smooth shading are kept; triangles are ~0.3 mm so the edge is not visible),
   close each half with a flat cap whose UVs are polar (boundary -> unit circle) so a procedurally painted
   flesh texture puts the peel rim exactly along the cut outline, and build convex collision hulls
   (slab-wise along the principal axis, so a curved banana half is not one fat hull).
3. Write each half as a USD that is a copy of the original asset (same link / material / physics layout)
   with the geometry replaced. Each half keeps the whole fruit's local frame, so placing it at the whole
   fruit's pose reproduces its part of the fruit exactly; mass is split by hull volume.
4. Write a chef's knife (blade, bolster, handle) and a fixed knife block with a slot that holds the knife
   on its edge, so a top-down grasp already holds it in cutting orientation.
5. Write the scene state JSON (the original scene + 8 parked, hidden halves + knife + block) and the cut
   plan JSON that the demo (slice_fruits_demo.py) reads.
"""
import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

SCENES = Path(__file__).resolve().parents[2] / "assets" / "scenes" / "DROID"  # repro/<dir>/<this file> -> repo
SRC = SCENES / "droid_desk_serve_fruits"
SRC_JSON = SRC / "droid_desk_serve_fruits_scene_state_latest.json"
OUT = SCENES / "droid_desk_fruit_slicing"
SCENE_NAME = "droid_desk_fruit_slicing"

FRUITS = {  # scene object name -> flesh style
    "yellow_banana_yvjcfy_3": "banana",
    "yellow_pear_uckunr_0": "pear",
    "red_apple_zlgmzo_2": "apple",
    "orange_fruit_rofqtq_4": "orange",
}
DESK_Z = -0.02  # the scene's ground plane, on which every object rests

# ---- knife geometry. Knife frame: x handle -> tip, y blade normal, z up; edge at z = 0 at the bolster (x = 0).
BLADE_LEN, BLADE_H = 0.16, 0.045
SPINE_T, EDGE_T = 0.0022, 0.0004
BOLSTER_LEN = 0.012
HANDLE_LEN, HANDLE_H, HANDLE_T = 0.11, 0.024, 0.016
GRASP_X = -(BOLSTER_LEN + HANDLE_LEN / 2)  # eef target: handle centre
GRASP_Z = BLADE_H - HANDLE_H / 2
KNIFE_MASS = 0.18

# ---- knife block (fixed): base plate + two walls; the blade stands in the slot between them, edge down.
BLOCK_BASE = (0.10, 0.08, 0.012)       # x, y, z size
BLOCK_WALL = (0.09, 0.022, 0.035)      # each wall
BLOCK_SLOT = 0.006
BLOCK_BLADE_X = 0.075                  # knife-frame x of the block centre (the walls hold blade x 0.03..0.12)

ROBOT_REACH = (0.30, 0.66)             # allowed horizontal eef distance from the robot base at the cut


def quat_xyzw_to_R(q):
    return Rotation.from_quat(q).as_matrix()


def R_to_quat_xyzw(R):
    return Rotation.from_matrix(R).as_quat().tolist()


# ----------------------------------------------------------------------------------------------------- USD io
def find_visual_mesh(stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and "/visuals/" in str(prim.GetPath()):
            return prim
    raise RuntimeError("no visual mesh")


def read_visual(usd_path):
    stage = Usd.Stage.Open(str(usd_path))
    prim = find_visual_mesh(stage)
    # Root, link and mesh transforms are identity in these assets (checked); assert rather than assume.
    for p in (prim, prim.GetParent().GetParent(), stage.GetDefaultPrim()):
        M = UsdGeom.Xformable(p).GetLocalTransformation()
        assert np.allclose(np.array(M), np.eye(4), atol=1e-6), f"non-identity transform on {p.GetPath()}"
    m = UsdGeom.Mesh(prim)
    pts = np.asarray(m.GetPointsAttr().Get(), np.float64)
    counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
    assert (counts == 3).all()
    idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
    st = np.asarray(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").ComputeFlattened(), np.float64).reshape(-1, 3, 2)
    return pts, idx, st


def add_identity_xform_ops(xf):
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 1, 1))


def define_omnipbr(stage, path, texture=None, color=(0.8, 0.8, 0.8), metallic=0.0, roughness=0.5):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/{Path(path).name}")
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if texture:
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    return mat


def define_mesh(stage, path, V, F, purpose="default", st=None, normals=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(F.reshape(-1), np.int32)))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*V.min(0).tolist()), Gf.Vec3f(*V.max(0).tolist())]))
    m.CreatePurposeAttr(purpose)
    if st is not None:
        pv = UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                             UsdGeom.Tokens.faceVarying if len(st) == F.size
                                                             else UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(st, np.float32)))
    if normals is not None:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals, np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    add_identity_xform_ops(m)
    return m


def define_collider(stage, path, pts):
    hull = ConvexHull(pts)
    V = pts[hull.vertices]
    remap = {v: i for i, v in enumerate(hull.vertices)}
    F = np.vectorize(remap.get)(hull.simplices)
    m = define_mesh(stage, path, V, F, purpose="guide")
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    return hull.volume


# --------------------------------------------------------------------------------------------- mesh splitting
def plane_basis(n, up_hint):
    u = up_hint - up_hint.dot(n) * n
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def make_cap(pts, idx, o, n, outward, up_hint):
    """Flat cap over the cross-section: a fan from the section centroid to every section segment.

    Fruit cross-sections are star-shaped about their centroid, so the fan tiles the section without
    chaining the (possibly gappy) segment soup into loops. UVs are polar: centroid -> (0.5, 0.5), section
    boundary -> unit circle, angle measured from the fruit's up axis projected into the plane.
    """
    mesh = trimesh.Trimesh(pts, idx, process=False)
    mesh.merge_vertices()
    segs = trimesh.intersections.mesh_plane(mesh, n, o)            # (m, 2, 3)
    segs = segs[np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1) > 1e-7]
    mid = segs.mean(1)
    w = np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1)
    g = (mid * w[:, None]).sum(0) / w.sum()
    g = g - (g - o).dot(n) * n
    u, v = plane_basis(n, up_hint)
    a, b = segs[:, 0], segs[:, 1]
    flip = np.cross(a - g, b - g) @ outward < 0
    a2 = np.where(flip[:, None], b, a)
    b2 = np.where(flip[:, None], a, b)
    m = len(segs)
    V = np.concatenate([g[None], a2, b2])
    F = np.stack([np.zeros(m, int), 1 + np.arange(m), 1 + m + np.arange(m)], 1)
    rel = V - g
    ang = np.arctan2(rel @ v, rel @ u)
    rho = np.ones(len(V))
    rho[0] = 0.0
    st = np.stack([0.5 + 0.5 * rho * np.cos(ang), 0.5 + 0.5 * rho * np.sin(ang)], 1)
    # Fan centre must not take the angle of an arbitrary direction: per-triangle the centre UV is (0.5, 0.5)
    # for every triangle, so it is shared (vertex interpolation) without seams.
    normals = np.repeat(outward[None], m, 0)
    return V, F, st, normals, g, segs


def slab_hulls(P, max_slabs=5):
    """Convex pieces of a half: slices along its principal axis (curved banana halves need several)."""
    c = P.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((P - c).T))
    axis = evecs[:, -1]
    elong = math.sqrt(evals[-1] / max(evals[-2], 1e-12))
    k = 1 if elong < 1.7 else int(min(max_slabs, round(elong * 1.4)))
    t = (P - c) @ axis
    edges = np.linspace(t.min(), t.max(), k + 1)
    pad = 0.002
    pieces = []
    for i in range(k):
        sel = P[(t >= edges[i] - pad) & (t <= edges[i + 1] + pad)]
        if len(sel) >= 8:
            pieces.append(sel)
    return pieces


# -------------------------------------------------------------------------------------------- flesh textures
def flesh_texture(style, size=1024, seed=0):
    """Procedural cut-surface texture in the cap's polar UVs (r = 1 is the peel line, phi = 0 the fruit axis)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    su = (xx + 0.5) / size
    sv = 1.0 - (yy + 0.5) / size                                  # USD st has v up
    du, dv = (su - 0.5) * 2, (sv - 0.5) * 2
    r = np.sqrt(du ** 2 + dv ** 2)
    phi = np.arctan2(dv, du)
    noise = rng.normal(0, 1, (size // 8, size // 8))
    noise = np.asarray(Image.fromarray(((noise - noise.min()) / np.ptp(noise) * 255).astype(np.uint8))
                       .resize((size, size), Image.BICUBIC), np.float64) / 255 - 0.5
    fine = rng.normal(0, 1, (size, size)) * 0.5

    def col(c):
        return np.broadcast_to(np.asarray(c, np.float64), (size, size, 3)).copy()

    def blend(img, mask, c):
        mask = np.clip(mask, 0, 1)[..., None]
        return img * (1 - mask) + np.asarray(c, np.float64) * mask

    def smooth_step(x, w):
        return np.clip(0.5 + x / w, 0, 1)

    # point coordinates along / across the fruit axis
    ax, cx = du, dv
    if style in ("apple", "pear"):
        base = (243, 235, 203) if style == "apple" else (240, 234, 200)
        img = col(base) + noise[..., None] * 14 + fine[..., None] * 3
        # slightly translucent ring under the peel
        img = blend(img, smooth_step(r - 0.90, 0.04) * 0.55, (226, 214, 160))
        # core: an elongated, slightly darker region along the fruit axis with a fibrous outline
        a_ax, a_cx = (0.46, 0.24) if style == "apple" else (0.40, 0.17)
        q = np.sqrt((ax / a_ax) ** 2 + (cx / a_cx) ** 2)
        img = blend(img, smooth_step(1.0 - q, 0.15) * 0.55, (228, 210, 150))
        img = blend(img, np.exp(-((q - 1.0) / 0.035) ** 2) * 0.55, (205, 185, 120))
        # vascular bundle dots on a ring
        for k in range(10):
            t = 2 * np.pi * k / 10
            d = np.sqrt((du - 0.62 * np.cos(t)) ** 2 + (dv - 0.62 * np.sin(t)) ** 2)
            img = blend(img, smooth_step(0.018 - d, 0.01) * 0.6, (214, 196, 135))
        # two seeds near the centre, pointed along the axis
        for s in (-1, 1):
            q2 = np.sqrt(((ax - 0.02) / 0.10) ** 2 + ((cx - s * 0.075) / 0.045) ** 2)
            img = blend(img, smooth_step(1 - q2, 0.25), (88, 52, 26))
        if style == "pear":  # stone-cell grit
            grit = (rng.random((size, size)) > 0.996).astype(float)
            img = blend(img, grit * 0.5 * (r < 0.9), (205, 190, 140))
        peel = (150, 22, 28) if style == "apple" else (196, 172, 62)
        img = blend(img, smooth_step(r - 0.955, 0.02), peel)
    elif style == "orange":
        img = col((247, 150, 35)) + noise[..., None] * 18 + fine[..., None] * 6
        nseg = 11
        seg_phase = (phi % (2 * np.pi / nseg)) / (2 * np.pi / nseg)
        membrane = np.exp(-((np.minimum(seg_phase, 1 - seg_phase) * 2 * np.pi * r / nseg) / 0.010) ** 2)
        img = blend(img, membrane * (r < 0.86) * 0.85, (252, 214, 160))
        # juice vesicles: radial streaks
        streak = 0.5 + 0.5 * np.sin(phi * 140 + noise * 6)
        img = blend(img, streak * 0.15 * (r < 0.85), (255, 175, 70))
        img = blend(img, smooth_step(0.10 - r, 0.04), (252, 222, 170))          # central pith
        img = blend(img, smooth_step(r - 0.86, 0.02), (252, 236, 205))          # albedo (white pith)
        img = blend(img, smooth_step(r - 0.94, 0.015), (232, 112, 18))          # flavedo (peel)
    elif style == "banana":
        img = col((246, 238, 206)) + noise[..., None] * 10 + fine[..., None] * 3
        lobes = 0.5 + 0.5 * np.cos(3 * phi)
        img = blend(img, smooth_step(0.28 * (0.75 + 0.25 * lobes) - r, 0.06) * 0.45, (232, 220, 175))
        for k in range(3):
            t = 2 * np.pi * k / 3 + 0.5
            d = np.sqrt((du - 0.11 * np.cos(t)) ** 2 + (dv - 0.11 * np.sin(t)) ** 2)
            img = blend(img, smooth_step(0.03 - d, 0.015), (70, 52, 36))
        img = blend(img, smooth_step(r - 0.93, 0.02), (236, 222, 160))
        img = blend(img, smooth_step(r - 0.96, 0.015), (222, 186, 40))
    else:
        raise ValueError(style)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------------------------------------- planning
def world_points(pts, pose, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    sel = pts[rng.choice(len(pts), min(n, len(pts)), replace=False)]
    p, R = pose
    return sel @ R.T + p


def seg_point_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def knife_block_pose(block_xy):
    """World pose of the knife standing in the block (knife +x = world +x, edge on the base plate)."""
    bx, by = block_xy
    origin = np.array([bx - BLOCK_BLADE_X, by, DESK_Z + BLOCK_BASE[2] + 0.0005])
    return origin, np.eye(3)


def plan_cut(name, style, P_world, obstacles, base_xy, knife_heading0):
    """Choose cut plane + knife heading for one fruit.

    For every candidate horizontal plane normal n and knife heading x (= +/- z x n): the blade must span the
    section with margin; the gripper sits at the handle centre, (s - GRASP_X) behind the cut point.
    Score = clearance of gripper/knife from other objects at the bottom of the stroke, then reach and how far
    the wrist has to turn from the heading the knife was grasped at.
    """
    c = (P_world.min(0) + P_world.max(0)) / 2
    z = np.array([0, 0, 1.0])
    cands = []
    if style == "banana":
        evals, evecs = np.linalg.eigh(np.cov((P_world[:, :2] - P_world[:, :2].mean(0)).T))
        a = np.array([*evecs[:, -1], 0.0])
        base_t = math.atan2(a[1], a[0])
        for dth in np.radians(np.arange(-35, 36, 5)):
            for t in np.arange(-0.04, 0.0401, 0.01):
                cands.append((base_t + dth, c + a * t))
    else:
        for th in np.radians(np.arange(0, 180, 7.5)):
            cands.append((th, c))
    best = None
    for th, o in cands:
        n = np.array([math.cos(th), math.sin(th), 0.0])
        S = P_world[np.abs((P_world - o) @ n) < 0.0025]
        if len(S) < 30:
            continue
        for sgn in (1, -1):
            x = sgn * np.cross(z, n)
            proj = (S - o) @ x
            lo, hi = proj.min(), proj.max()
            half = (hi - lo) / 2
            q = o + x * (lo + hi) / 2                     # cut point: middle of the section along the blade
            s = half + 0.02                               # blade position of the cut point (from bolster)
            if BLADE_LEN - s < half + 0.012:
                continue
            eef = q - x * (s - GRASP_X)
            tip = q + x * (BLADE_LEN - s)
            reach = np.linalg.norm(eef[:2] - base_xy)
            if not (ROBOT_REACH[0] <= reach <= ROBOT_REACH[1]):
                continue
            turn = math.degrees(math.acos(np.clip(x[:2] @ knife_heading0[:2], -1, 1)))
            if turn > 125:
                continue
            clear = []
            for oname, O, infl in obstacles:
                d_knife = seg_point_dist(O[:, :2], eef[:2], tip[:2]).min() - 0.012 - infl
                d_grip = np.linalg.norm(O[:, :2] - eef[:2], axis=1).min() - 0.065 - infl
                clear.append((min(d_knife, d_grip), oname))
            d_self = np.linalg.norm(P_world[:, :2] - eef[:2], axis=1).min() - 0.05
            clear.append((d_self, "self"))
            cmin, cname = min(clear)
            score = min(cmin, 0.06) - 0.0003 * turn - 0.02 * max(0, reach - 0.6)
            if best is None or score > best["score"]:
                best = dict(score=score, clearance=cmin, limiting=cname, n=n, x=x, o=o, q=q, s=s, eef=eef,
                            reach=reach, turn=turn, half=half, theta=math.degrees(th))
    if best is None:
        raise RuntimeError(f"no feasible cut for {name}")
    print(f"  {name:24s} plane normal {best['n'][:2].round(3)} knife heading {best['x'][:2].round(3)} "
          f"(turn {best['turn']:.0f} deg) cut pt {best['q'].round(3)} section half-width {best['half']*100:.1f} cm, "
          f"eef reach {best['reach']:.3f} m, clearance {best['clearance']*100:.1f} cm (limit: {best['limiting']})",
          flush=True)
    return best


# --------------------------------------------------------------------------------------------- asset writers
def write_half(src_usd, dst_usd, category, model, skin_V, skin_F, skin_st, cap, hulls_pts, mass, flesh_png):
    dst_usd.parent.mkdir(parents=True, exist_ok=True)
    Sdf.Layer.FindOrOpen(str(src_usd)).Export(str(dst_usd))
    stage = Usd.Stage.Open(str(dst_usd))
    root = stage.GetDefaultPrim()
    for p in list(stage.GetPseudoRoot().GetChildren()):
        if p.GetName() == "colliders":                       # stray importer leftover outside the asset
            stage.RemovePrim(p.GetPath())
    root.GetAttribute("ig:category").Set(category)
    root.GetAttribute("ig:model").Set(model)
    link = next(c for c in root.GetChildren() if c.HasAPI(UsdPhysics.RigidBodyAPI))
    vis = find_visual_mesh(stage)
    m = UsdGeom.Mesh(vis)
    m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(skin_V, np.float32)))
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(skin_F), 3, np.int32)))
    m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.ascontiguousarray(skin_F.reshape(-1), np.int32)))
    UsdGeom.PrimvarsAPI(vis).GetPrimvar("st").Set(
        Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(skin_st.reshape(-1, 2), np.float32)))
    m.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*skin_V.min(0).tolist()), Gf.Vec3f(*skin_V.max(0).tolist())]))

    looks = root.GetPath().AppendChild("Looks")
    flesh = define_omnipbr(stage, str(looks.AppendChild("flesh")), texture="../material/flesh.png", roughness=0.35)
    cap_V, cap_F, cap_st, cap_N = cap
    cm = define_mesh(stage, str(vis.GetParent().GetPath().AppendChild("cut_face")), cap_V, cap_F, st=cap_st,
                     normals=cap_N)
    UsdShade.MaterialBindingAPI.Apply(cm.GetPrim()).Bind(flesh)

    coll = link.GetPath().AppendChild("collisions")
    for c in list(stage.GetPrimAtPath(coll).GetChildren()):
        stage.RemovePrim(c.GetPath())
    for i, P in enumerate(hulls_pts):
        define_collider(stage, str(coll.AppendChild(f"collision_{i}")), P)
    link.GetAttribute("physics:mass").Set(float(mass))
    for prop in ("physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes"):
        link.RemoveProperty(prop)                            # let PhysX derive them from the new hulls
    allV = np.concatenate([skin_V, cap_V])
    root.GetAttribute("ig:nativeBB").Set(Gf.Vec3f(*(allV.max(0) - allV.min(0)).tolist()))
    stage.GetRootLayer().Save()

    mat_dir = dst_usd.parent.parent / "material"
    mat_dir.mkdir(parents=True, exist_ok=True)
    skin_png = Path(src_usd).parent.parent / "material" / "material_0.png"
    tgt = mat_dir / "material_0.png"
    if tgt.exists():
        tgt.unlink()
    try:
        os.link(skin_png, tgt)                                # hard link: no second copy of a 20 MB texture
    except OSError:
        shutil.copy2(skin_png, tgt)
    flesh_png.save(mat_dir / "flesh.png")
    misc = dst_usd.parent.parent / "misc"
    misc.mkdir(exist_ok=True)
    json.dump({"meta_links": {}, "link_tags": {}, "object_parts": [], "bbox_size": (allV.max(0) - allV.min(0)).tolist(),
               "orientations": [], "openable_joint_ids": {}}, open(misc / "metadata.json", "w"))


def write_rigid_usd(dst_usd, model, category, parts, materials, collider_pts, mass):
    """Single-link rigid asset in the same layout as the SimFoundry objects (root / Looks / link / visuals,collisions)."""
    dst_usd.parent.mkdir(parents=True, exist_ok=True)
    if dst_usd.exists():
        dst_usd.unlink()
    stage = Usd.Stage.CreateNew(str(dst_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{model}")
    stage.SetDefaultPrim(root.GetPrim())
    add_identity_xform_ops(root)
    root.GetPrim().CreateAttribute("ig:category", Sdf.ValueTypeNames.String).Set(category)
    root.GetPrim().CreateAttribute("ig:model", Sdf.ValueTypeNames.String).Set(model)
    UsdGeom.Scope.Define(stage, f"/{model}/Looks")
    mats = {k: define_omnipbr(stage, f"/{model}/Looks/{k}", **v) for k, v in materials.items()}
    link = UsdGeom.Xform.Define(stage, f"/{model}/base_link")
    add_identity_xform_ops(link)
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(float(mass))
    UsdGeom.Xform.Define(stage, f"/{model}/base_link/visuals")
    allV = []
    for name, V, F, mat in parts:
        # Unwelded triangles -> flat shading, which is what machined/box parts should look like.
        Vu = V[F.reshape(-1)]
        Fu = np.arange(len(Vu)).reshape(-1, 3)
        mesh = define_mesh(stage, f"/{model}/base_link/visuals/{name}", Vu, Fu)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mats[mat])
        allV.append(V)
    UsdGeom.Xform.Define(stage, f"/{model}/base_link/collisions")
    for i, P in enumerate(collider_pts):
        define_collider(stage, f"/{model}/base_link/collisions/collision_{i}", P)
    allV = np.concatenate(allV)
    root.GetPrim().CreateAttribute("ig:nativeBB", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3f(*(allV.max(0) - allV.min(0)).tolist()))
    stage.GetRootLayer().Save()
    return allV


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def knife_meshes():
    """Chef's knife: wedge-section blade with a rising edge and falling spine toward the tip."""
    xs = np.linspace(0, BLADE_LEN, 33)
    t = xs / BLADE_LEN
    edge_z = np.where(t < 0.6, 0.0, 0.016 * ((t - 0.6) / 0.4) ** 2)
    spine_z = np.where(t < 0.5, BLADE_H, BLADE_H - (BLADE_H - 0.016) * ((t - 0.5) / 0.5) ** 1.3)
    thick_s = SPINE_T * (1 - 0.6 * t)
    V, F = [], []
    for i, x in enumerate(xs):
        V += [[x, -EDGE_T / 2, edge_z[i]], [x, EDGE_T / 2, edge_z[i]],
              [x, thick_s[i] / 2, spine_z[i]], [x, -thick_s[i] / 2, spine_z[i]]]
    V = np.array(V)
    for i in range(len(xs) - 1):
        a, b = 4 * i, 4 * (i + 1)
        for k in range(4):
            k2 = (k + 1) % 4
            F += [[a + k, b + k, b + k2], [a + k, b + k2, a + k2]]
    F += [[0, 2, 1], [0, 3, 2]]
    e = 4 * (len(xs) - 1)
    F += [[e, e + 1, e + 2], [e, e + 2, e + 3]]
    blade = (V, np.array(F))
    bolster = box([-BOLSTER_LEN, -0.006, 0.012], [0.0, 0.006, BLADE_H])
    handle = box([-BOLSTER_LEN - HANDLE_LEN, -HANDLE_T / 2, BLADE_H - HANDLE_H],
                 [-BOLSTER_LEN, HANDLE_T / 2, BLADE_H])
    rivets = []
    for rx in (-BOLSTER_LEN - 0.02, -BOLSTER_LEN - HANDLE_LEN / 2, -BOLSTER_LEN - HANDLE_LEN + 0.02):
        for side in (-1, 1):
            y0, y1 = sorted((side * HANDLE_T / 2, side * (HANDLE_T / 2 + 0.0006)))
            rivets.append(box([rx - 0.0035, y0, GRASP_Z - 0.0035], [rx + 0.0035, y1, GRASP_Z + 0.0035]))
    return blade, bolster, handle, rivets


# ------------------------------------------------------------------------------------------------------ main
def main(args):
    scene = json.load(open(SRC_JSON))
    init = scene["objects_info"]["init_info"]
    reg = scene["state"]["registry"]["object_registry"]
    base_xy = np.array(reg["robot0"]["root_link"]["pos"][:2])

    def pose(name):
        rl = reg[name]["root_link"]
        return np.array(rl["pos"]), quat_xyzw_to_R(rl["ori"])

    def usd_of(name):
        return SRC / init[name]["args"]["usd_path"]

    meshes, wpts = {}, {}
    for name in [k for k in reg if k not in ("robot0", "mesh_background_0")]:
        pts, idx, st = read_visual(usd_of(name))
        meshes[name] = (pts, idx, st)
        wpts[name] = world_points(pts, pose(name), n=6000 if name in FRUITS else 3000)
    print("objects (world AABB):")
    for k, P in wpts.items():
        print(f"  {k:30s} {P.min(0).round(3)} .. {P.max(0).round(3)}")

    # ---- knife block placement: free desk spot to the robot's right, handle within easy reach
    block_xy = np.array(args.block_xy)
    k_origin, k_R = knife_block_pose(block_xy)
    blade, bolster, handle, rivets = knife_meshes()
    knife_V_local = np.concatenate([blade[0], bolster[0], handle[0]])
    knife_w = knife_V_local @ k_R.T + k_origin
    bb_lo = [block_xy[0] - BLOCK_BASE[0] / 2, block_xy[1] - BLOCK_BASE[1] / 2, DESK_Z]
    bb_hi = [block_xy[0] + BLOCK_BASE[0] / 2, block_xy[1] + BLOCK_BASE[1] / 2,
             DESK_Z + BLOCK_BASE[2] + BLOCK_WALL[2]]
    block_w = np.array([[x, y, z] for x in np.linspace(bb_lo[0], bb_hi[0], 8) for y in np.linspace(bb_lo[1], bb_hi[1], 6)
                        for z in (bb_lo[2], bb_hi[2])])
    for k, P in wpts.items():
        d = min(np.linalg.norm(P[:, None, :2] - knife_w[None, :, :2], axis=2).min(),
                np.linalg.norm(P[:, None, :2] - block_w[None, :, :2], axis=2).min())
        print(f"  knife+block clearance to {k}: {d*100:.1f} cm")
    grasp_w = k_R @ np.array([GRASP_X, 0, GRASP_Z]) + k_origin
    print(f"knife grasp point {grasp_w.round(3)}, reach {np.linalg.norm(grasp_w[:2] - base_xy):.3f} m")

    # ---- plan the cuts
    print("cut plan:")
    plan = {"scene": SCENE_NAME, "desk_z": DESK_Z,
            "knife": {"name": "chef_knife_0", "blade_len": BLADE_LEN, "blade_h": BLADE_H, "grasp_local": [GRASP_X, 0, GRASP_Z],
                      "spine_t": SPINE_T},
            "block": {"name": "knife_block_0", "xy": block_xy.tolist()}, "fruits": []}
    for name, style in FRUITS.items():
        obstacles = [(k, P, 0.02 if k in FRUITS else 0.0) for k, P in wpts.items() if k != name]
        obstacles.append(("knife_block", block_w, 0.0))
        best = plan_cut(name, style, wpts[name], obstacles, base_xy, np.array([1.0, 0, 0]))
        p, R = pose(name)
        plan["fruits"].append({
            "name": name, "style": style,
            "normal_local": (R.T @ best["n"]).tolist(), "origin_local": (R.T @ (best["o"] - p)).tolist(),
            "cut_point_local": (R.T @ (best["q"] - p)).tolist(), "heading_sign_local_ref": None,
            "knife_heading_world": best["x"].tolist(), "normal_world": best["n"].tolist(),
            "blade_s": best["s"], "section_half_width": best["half"], "clearance": best["clearance"],
            "halves": [f"{name}_half0", f"{name}_half1"]})
    if args.plan_only:
        return

    # ---- halves
    OUT.mkdir(parents=True, exist_ok=True)
    new_init, new_reg = {}, {}
    park = 0
    for fp in plan["fruits"]:
        name, style = fp["name"], fp["style"]
        pts, idx, st = meshes[name]
        n_l = np.asarray(fp["normal_local"])
        n_l /= np.linalg.norm(n_l)
        o_l = np.asarray(fp["origin_local"])
        cat = init[name]["args"]["category"]
        model = Path(init[name]["args"]["usd_path"]).stem
        # fruit axis in its local frame = longest extent for pear/apple (stem axis), used to orient the core
        ext = pts.max(0) - pts.min(0)
        up_hint = np.eye(3)[int(np.argmax(ext))] if style in ("pear", "banana") else np.eye(3)[2]
        d_face = (pts[idx].mean(1) - o_l) @ n_l
        tex = flesh_texture(style, seed=hash(name) % 1000)
        vols = []
        halves = []
        for side, mask in ((0, d_face >= 0), (1, d_face < 0)):
            outward = -n_l if side == 0 else n_l
            F = idx[mask]
            used, inv = np.unique(F, return_inverse=True)
            skin_V, skin_F, skin_st = pts[used], inv.reshape(-1, 3), st[mask]
            cap_V, cap_F, cap_st, cap_N, g, _ = make_cap(pts, idx, o_l, n_l, outward, up_hint)
            P_all = np.concatenate([skin_V, cap_V])
            pieces = slab_hulls(P_all)
            vols.append(sum(ConvexHull(P).volume for P in pieces))
            halves.append((skin_V, skin_F, skin_st, (cap_V, cap_F, cap_st, cap_N), pieces))
        mass_total = float(Usd.Stage.Open(str(usd_of(name))).GetDefaultPrim().GetChildren()[1]
                           .GetAttribute("physics:mass").Get() or 0.15)
        for side, (skin_V, skin_F, skin_st, cap, pieces) in enumerate(halves):
            hname = f"{name}_half{side}"
            hmodel = f"{model}h{side}"
            dst = OUT / "objects" / f"half_{cat}" / hmodel / "usd" / f"{hmodel}.usd"
            mass = mass_total * vols[side] / sum(vols)
            write_half(usd_of(name), dst, f"half_{cat}", hmodel, skin_V, skin_F, skin_st, cap, pieces, mass, tex)
            print(f"  wrote {hname}: {len(skin_F)} skin tris, {len(cap[1])} cap tris, {len(pieces)} hulls, "
                  f"mass {mass*1000:.0f} g -> {dst.relative_to(SCENES)}", flush=True)
            new_init[hname] = {"class_module": "omnigibson.objects.usd_object", "class_name": "USDObject",
                               "args": {"name": hname, "usd_path": str(dst.relative_to(OUT)), "category": f"half_{cat}"}}
            new_reg[hname] = {"is_asleep": False, "non_kin": {},
                              "root_link": {"pos": [8.0 + 0.4 * park, 8.0, DESK_Z + 0.06], "ori": [0, 0, 0, 1.0],
                                            "lin_vel": [0.0] * 3, "ang_vel": [0.0] * 3}}
            park += 1

    # ---- knife + block
    knife_usd = OUT / "objects" / "chef_knife" / "knife01" / "usd" / "knife01.usd"
    parts = [("blade", *blade, "steel"), ("bolster", *bolster, "steel"), ("handle", *handle, "handle")]
    parts += [(f"rivet_{i}", V, F, "steel") for i, (V, F) in enumerate(rivets)]
    write_rigid_usd(knife_usd, "knife01", "chef_knife", parts,
                    {"steel": dict(color=(0.78, 0.79, 0.81), metallic=1.0, roughness=0.22),
                     "handle": dict(color=(0.16, 0.09, 0.05), metallic=0.0, roughness=0.45)},
                    [blade[0], np.concatenate([bolster[0], handle[0]])], KNIFE_MASS)
    bx, by, bz = BLOCK_BASE
    wx, wy, wz = BLOCK_WALL
    base_box = box([-bx / 2, -by / 2, 0], [bx / 2, by / 2, bz])
    wall_l = box([-wx / 2, BLOCK_SLOT / 2, bz], [wx / 2, BLOCK_SLOT / 2 + wy, bz + wz])
    wall_r = box([-wx / 2, -BLOCK_SLOT / 2 - wy, bz], [wx / 2, -BLOCK_SLOT / 2, bz + wz])
    block_usd = OUT / "objects" / "knife_block" / "block01" / "usd" / "block01.usd"
    write_rigid_usd(block_usd, "block01", "knife_block",
                    [("base", *base_box, "wood"), ("wall_left", *wall_l, "wood"), ("wall_right", *wall_r, "wood")],
                    {"wood": dict(color=(0.62, 0.43, 0.25), roughness=0.6)},
                    [base_box[0], wall_l[0], wall_r[0]], 1.0)
    new_init["chef_knife_0"] = {"class_module": "omnigibson.objects.usd_object", "class_name": "USDObject",
                                "args": {"name": "chef_knife_0", "usd_path": str(knife_usd.relative_to(OUT)),
                                         "category": "chef_knife"}}
    new_reg["chef_knife_0"] = {"is_asleep": False, "non_kin": {},
                               "root_link": {"pos": k_origin.tolist(), "ori": R_to_quat_xyzw(k_R),
                                             "lin_vel": [0.0] * 3, "ang_vel": [0.0] * 3}}
    new_init["knife_block_0"] = {"class_module": "omnigibson.objects.usd_object", "class_name": "USDObject",
                                 "args": {"name": "knife_block_0", "usd_path": str(block_usd.relative_to(OUT)),
                                          "category": "knife_block", "fixed_base": True}}
    new_reg["knife_block_0"] = {"is_asleep": False, "non_kin": {},
                                "root_link": {"pos": [block_xy[0], block_xy[1], DESK_Z], "ori": [0, 0, 0, 1.0],
                                              "lin_vel": [0.0] * 3, "ang_vel": [0.0] * 3}}

    # ---- scene JSON: original objects keep their assets (paths re-rooted), new ones appended
    for k, v in init.items():
        if "usd_path" in v["args"]:
            v["args"]["usd_path"] = os.path.relpath(SRC / v["args"]["usd_path"], OUT)
    init.update(new_init)
    reg.update(new_reg)
    out_json = OUT / f"{SCENE_NAME}_scene_state.json"
    json.dump(scene, open(out_json, "w"), indent=2)
    json.dump(plan, open(OUT / "cut_plan.json", "w"), indent=2)
    print(f"wrote {out_json}\nwrote {OUT / 'cut_plan.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--block-xy", type=float, nargs=2, default=[0.37, -0.25])
    main(ap.parse_args())
