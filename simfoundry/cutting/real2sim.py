# SPDX-License-Identifier: Apache-2.0
"""A cutting simulation set up from a SimFoundry Pipeline A reconstruction.

Pipeline A turns one frame of a video into textured meshes in metres (stage 11), their poses in a
Z-up frame whose origin lies on the table (stage 12, or stage 14 once OmniGibson has settled
them), and the camera that took the frame (stage 4 pose, stage 2 intrinsics). This module reads
those and hands the MPM solver what it needs: the food's shape as an occupancy test, the board it
rests on, the colour of its skin, and the camera, so a simulated cut can be drawn from the video's
own viewpoint and set beside it.

The steps, in the order ``scripts/cutting/tofu_real2sim.py`` takes them:

* ``load_reconstruction`` reads the scene; ``stage5_mask`` maps an object's mask back onto the
  frame.
* ``surface_height`` reads the board's top from the depth map's own point cloud, and
  ``fit_to_silhouette`` resizes and moves the food on it until its outline matches the frame:
  a small object's size and distance are the loosest part of a single-view reconstruction.
* ``level_box`` replaces a box-shaped food by its box. ``place_food`` turns the world about Z
  so the food lines up with the solver's axes, long side along x;
  ``FoodPlacement.sim_from_world`` maps the world into the solver's frame once the solver has
  said where its box sits.
* ``photo_albedo`` gives the food's particles the colours the frame shows, less the
  renderer's light.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Where Pipeline A puts what this module reads, relative to the scene directory.
OBJECTS_INFO = "s11_sim/scene_objects_info.json"
POSE_FILES = ("s14_og/settled_poses.json", "s12_physics/pb_scene_poses.json")
DEPTH_RESULTS = "s2_da/da/exports/npz/results.npz"


@dataclass
class ReconstructedObject:
    """One object of the scene: its stage 11 mesh, posed in the world frame."""

    name: str
    category: str
    mesh: "trimesh.Trimesh"  # noqa: F821 - textured, in world coordinates [m]


@dataclass
class Reconstruction:
    """What Pipeline A recovered from one frame, in its Z-up world frame (table top at z = 0).

    Attributes:
        scene_dir: The scene's directory under the pipeline's root.
        objects: Objects by name (``iter_<k>``).
        cam_to_world: 4x4 pose of the camera, OpenCV axes (x right, y down, z forward).
        K: Intrinsic matrix of the camera, for an image of ``calib_size`` pixels.
        calib_size: (width, height) that ``K`` belongs to: the depth model's input.
        pose_file: Which file the poses came from.
        img_idx: The input frame the scene was built from.
    """

    scene_dir: Path
    objects: dict
    cam_to_world: np.ndarray
    K: np.ndarray
    calib_size: tuple
    pose_file: str
    img_idx: int = 0

    def find(self, query: str) -> ReconstructedObject:
        """The one object whose category contains ``query`` (case-insensitive)."""
        hits = [o for o in self.objects.values() if query.lower() in o.category.lower()]
        if len(hits) != 1:
            cats = sorted(o.category for o in self.objects.values())
            raise KeyError(f"{len(hits)} objects match {query!r}; categories: {cats}")
        return hits[0]


def _quat_xyzw_to_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_reconstruction(scene_dir, img_idx=None) -> Reconstruction:
    """Reads a finished Pipeline A scene (stages 2, 4, 11 and 12 or 14).

    Args:
        scene_dir: ``<root_dir>/<scene_name>``.
        img_idx: Which input frame the scene was built from; by default the one stage 5
            recorded in ``s5_scene/decomposition_frame.json``, else 0.
    """
    import trimesh

    scene_dir = Path(scene_dir)
    if img_idx is None:
        frame = scene_dir / "s5_scene" / "decomposition_frame.json"
        img_idx = int(json.loads(frame.read_text())["img_idx"]) if frame.exists() else 0
    info = json.loads((scene_dir / OBJECTS_INFO).read_text())
    pose_file = next((p for p in POSE_FILES if (scene_dir / p).exists()), None)
    if pose_file is None:
        raise FileNotFoundError(f"no object poses in {scene_dir} (looked for {POSE_FILES})")
    poses = json.loads((scene_dir / pose_file).read_text())
    objects = {}
    for entry in info.values():
        name, cat, model = entry["name"], entry["category"], entry["model"]
        if name not in poses:
            continue
        obj = scene_dir / "s11_sim" / "objects" / cat / model / "urdf" / "shape" / "visual" / f"{cat}_{model}_link_textured_mesh.obj"
        mesh = trimesh.load(obj, process=False, force="mesh")
        pos, quat = poses[name]
        T = np.eye(4)
        T[:3, :3] = _quat_xyzw_to_matrix(quat)
        T[:3, 3] = pos
        mesh.apply_transform(T)
        objects[name] = ReconstructedObject(name=name, category=cat, mesh=mesh)
    cam_to_world = np.load(scene_dir / "s4_frame" / f"image_{img_idx}_cam2world.npy")
    depth = np.load(scene_dir / DEPTH_RESULTS)
    h, w = depth["image"][img_idx].shape[:2]
    return Reconstruction(
        scene_dir=scene_dir,
        objects=objects,
        cam_to_world=cam_to_world,
        K=np.asarray(depth["intrinsics"][img_idx], dtype=float),
        calib_size=(w, h),
        pose_file=pose_file,
        img_idx=img_idx,
    )


def footprint_yaw(points) -> float:
    """Yaw of the long side of the smallest rectangle around the points' footprint [rad].

    The answer is folded into [-pi/2, pi/2): a block has no front or back.
    """
    from scipy.spatial import ConvexHull

    xy = np.asarray(points, dtype=float)[:, :2]
    hull = xy[ConvexHull(xy).vertices]
    best = None
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        d = b - a
        if np.linalg.norm(d) < 1e-12:
            continue
        th = math.atan2(d[1], d[0])
        c, s = math.cos(th), math.sin(th)
        r = hull @ np.array([[c, -s], [s, c]])
        ext = r.max(axis=0) - r.min(axis=0)
        if best is None or ext[0] * ext[1] < best[0]:
            best = (ext[0] * ext[1], th if ext[0] >= ext[1] else th + math.pi / 2.0)
    th = best[1]
    return (th + math.pi / 2.0) % math.pi - math.pi / 2.0


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class FoodPlacement:
    """The food turned into the solver's box frame.

    Attributes:
        block_size: Size of the box around the food [m], for ``MPMCuttingSim``.
        yaw: Yaw of the food's long side in the world [rad].
        corner: World point that lands on the box's lower corner, after the yaw.
        local_mesh: The food in the box frame (lower corner at the origin), watertight.
    """

    block_size: np.ndarray
    yaw: float
    corner: np.ndarray
    local_mesh: "trimesh.Trimesh"  # noqa: F821

    def inside(self, pts) -> np.ndarray:
        """Which box-frame points lie in the food (``MPMCuttingSim``'s ``inside``)."""
        return self.local_mesh.contains(np.asarray(pts, dtype=float))

    def sim_from_world(self, block_min) -> np.ndarray:
        """4x4 map from the world to a solver whose box starts at ``block_min``."""
        T = np.eye(4)
        T[:3, :3] = _rz(-self.yaw)
        T[:3, 3] = np.asarray(block_min, dtype=float) - self.corner
        return T


def place_food(mesh_world, yaw=None, convex=False) -> FoodPlacement:
    """Puts a reconstructed food mesh into an axis-aligned box frame.

    Args:
        mesh_world: The food in the world frame (``ReconstructedObject.mesh``).
        yaw: Yaw to align with x [rad]; the long side of the footprint if None.
        convex: Use the mesh's convex hull. For food that is convex anyway (a block of tofu)
            this undoes what occlusion does to a single-view reconstruction: whatever hid part
            of the food in the frame (the knife, a finger) is carved out of the mesh.
    """
    m = mesh_world.copy()
    if convex:
        m = m.convex_hull
    else:
        # The textured mesh repeats vertices along UV seams; welded, it is closed again.
        m.merge_vertices(merge_tex=True, merge_norm=True)
        if not m.is_watertight:
            m.fill_holes()
    if not m.is_watertight:
        raise ValueError("the food mesh is not watertight, so it has no inside")
    if yaw is None:
        yaw = footprint_yaw(m.vertices)
    R = _rz(-yaw)
    v = m.vertices @ R.T
    lo, hi = v.min(axis=0), v.max(axis=0)
    local = m.copy()
    local.vertices = v - lo
    return FoodPlacement(block_size=hi - lo, yaw=float(yaw), corner=lo, local_mesh=local)


def level_box(mesh_world, yaw=None):
    """The level box around a mesh, turned to its footprint's long side: a block of food as a block.

    A generated mesh of something box-shaped carries the generator's habits (chamfered edges, a
    face leaning a degree or two). Filled with a particle lattice, such a face comes out
    terraced, and every terrace step shows as a line in the render. Its box keeps the
    reconstruction's proportions, place and turn.

    Args:
        mesh_world: The food in the world frame.
        yaw: Yaw of the box [rad]; the footprint's long side if None.
    """
    import trimesh

    v = np.asarray(mesh_world.vertices, dtype=float)
    if yaw is None:
        yaw = footprint_yaw(v)
    R = _rz(-yaw)
    local = v @ R.T
    lo, hi = local.min(axis=0), local.max(axis=0)
    box = trimesh.creation.box(extents=hi - lo)
    box.apply_translation(0.5 * (lo + hi))
    box.apply_transform(np.block([[R.T, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    return box


def stage5_mask(scene_dir, name, size) -> np.ndarray:
    """An object's stage 5 mask (``s5_scene/removal_mask/<name>.png``) in the frame's pixels.

    Stage 5 works on the frame padded to its own aspect ratio (centred, as
    ``processing_utils.pad_image_to_ratio`` does it, sizes truncated) and resized; this undoes
    both without importing the pipeline's utilities.

    Args:
        scene_dir: The scene directory.
        name: The object's name (``iter_<k>``).
        size: (width, height) of the frame.

    Returns:
        (height, width) bool mask.
    """
    import cv2
    from PIL import Image

    m = np.asarray(Image.open(Path(scene_dir) / "s5_scene" / "removal_mask" / f"{name}.png").convert("L"))
    w, h = size
    ratio = m.shape[1] / m.shape[0]
    new_w, new_h = (w, int(w / ratio)) if w / h > ratio else (int(h * ratio), h)
    full = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, left = (new_h - h) // 2, (new_w - w) // 2
    return full[top:top + h, left:left + w] > 127


def surface_height(rec: Reconstruction, name: str, columns=None) -> float:
    """Height of an object's top surface, read from the depth model's own point cloud [m].

    Stage 4 keeps that cloud in the world frame, one point per pixel of the depth image
    (``s4_frame/image_<k>_pc_raw_rotated.ply``). The median height over the object's stage 5
    mask, eroded to stay off its edges, is a measurement of where food on a board rests; a
    reconstructed board mesh can be several times too thick.

    Args:
        rec: The reconstruction.
        name: The object (``iter_<k>``), e.g. the board.
        columns: Optional (x0, x1) range of depth-image columns to use, e.g. around the food.
    """
    import cv2
    import open3d as o3d

    w, h = rec.calib_size
    cloud = rec.scene_dir / "s4_frame" / f"image_{rec.img_idx}_pc_raw_rotated.ply"
    pts = np.asarray(o3d.io.read_point_cloud(str(cloud)).points).reshape(h, w, 3)
    m = cv2.erode(stage5_mask(rec.scene_dir, name, (w, h)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if columns is not None:
        keep = np.zeros_like(m)
        keep[:, max(0, int(columns[0])):int(columns[1])] = True
        m &= keep
    if m.sum() < 20:
        raise ValueError(f"too few depth pixels on {name}")
    return float(np.median(pts[m][:, 2]))


def fit_to_silhouette(mesh_world, cam, mask, valid=None, z_bottom=None):
    """Resizes and moves a convex food mesh until its outline in the frame matches its mask.

    Single-view reconstruction leaves a small object's size and distance loose: they come from
    the few pixels of a depth map it covers, smeared at its edges. Its outline in the frame is
    sharp. The food stays level, its bottom on ``z_bottom`` (the board's top), so a size and a
    place on the board are all that is fitted: nearer means lower in the frame and larger.

    Args:
        mesh_world: The food in the world frame; only its convex hull is used.
        cam: ``render.Camera`` of the frame, in the world frame.
        mask: (H, W) bool mask of the food in the frame.
        valid: (H, W) bool, where the food would be seen if it were there; pixels covered by
            something in front of it (the knife) are left out of the comparison.
        z_bottom: Height of the surface it rests on; its current bottom if None.

    Returns:
        (T, info): the 4x4 similarity to apply to the food's meshes, and the fitted scale,
        shift [m] and silhouette IoU before and after.
    """
    import cv2
    from scipy.optimize import minimize

    V = np.asarray(mesh_world.convex_hull.vertices, dtype=float)
    zmin = float(V[:, 2].min())
    base = np.array([V[:, 0].mean(), V[:, 1].mean(), zmin])
    dz = (zmin if z_bottom is None else float(z_bottom)) - zmin
    if valid is None:
        valid = np.ones_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    pad = int(max(xs.ptp(), ys.ptp()))
    H, W = mask.shape
    x0, x1, y0, y1 = max(0, xs.min() - pad), min(W, xs.max() + pad), max(0, ys.min() - pad), min(H, ys.max() + pad)
    M, Vd = mask[y0:y1, x0:x1], valid[y0:y1, x0:x1]

    def moved(p):
        return base + math.exp(p[0]) * (V - base) + np.array([p[1], p[2], dz])

    def outline(p):
        px, py, _ = cam.project(moved(p))
        hull = cv2.convexHull(np.stack([px - x0, py - y0], axis=1).astype(np.float32))
        img = np.zeros(M.shape, np.uint8)
        cv2.fillConvexPoly(img, np.round(hull * 16.0).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
        return img.astype(bool)

    def iou(p):
        P = outline(p)
        return float((P & M & Vd).sum() / max(((P | M) & Vd).sum(), 1))

    # Start from the outline's width and centre: scale to the mask's width, then move the food
    # along the board until its centre lies on the ray through the mask's centre.
    p = np.zeros(3)
    iou0 = iou(p)
    for _ in range(3):
        P = outline(p)
        pys, pxs = np.nonzero(P)
        if len(pxs):
            p[0] += math.log(max(xs.ptp(), 1) / max(pxs.ptp(), 1))
        c = moved(p).mean(axis=0)
        ray = cam.rays[int(np.clip(ys.mean(), 0, H - 1)), int(np.clip(xs.mean(), 0, W - 1))]
        t = (c[2] - cam.eye[2]) / ray[2]
        target = cam.eye + t * ray
        p[1:] += (target - c)[:2]
    res = minimize(lambda q: 1.0 - iou(q), p, method="Nelder-Mead",
                   options=dict(initial_simplex=[p, p + [0.04, 0, 0], p + [0, 0.004, 0], p + [0, 0, 0.004]],
                                xatol=1e-4, fatol=1e-4, maxiter=400))
    q = res.x
    s = math.exp(q[0])
    T = np.eye(4)
    T[:3, :3] *= s
    T[:3, 3] = base - s * base + np.array([q[1], q[2], dz])
    return T, dict(scale=s, shift_m=[float(q[1]), float(q[2]), dz], iou_before=iou0, iou_after=iou(q))


def photo_albedo(renderer, x, piece, photo, seen_mask, gloss):
    """Albedo of the food's particles, read from the photograph the renderer's camera took.

    Each particle the renderer would draw takes the mean, over the pixels it covers, of the
    photo's colour divided by the light the renderer puts there (``render.food_light`` with
    the renderer's own screen-space normals). Drawn at the start, the food then looks as it
    does in the frame, pores and all, at the particles' resolution.

    Args:
        renderer: ``render.Renderer`` whose camera matches the photo pixel for pixel.
        x: (N, 3) particle positions as drawn (the initial ones).
        piece: (N,) piece index of each particle.
        photo: (H, W, 3) uint8 photograph, the renderer's size.
        seen_mask: (H, W) bool: where the food itself shows in the photo (not something in
            front of it).
        gloss: (N,) gloss of each particle, as the renderer will use it.

    Returns:
        (albedo (N, 3) in [0, 1], seen (N,) bool); albedo is 0 where unseen.
    """
    from simfoundry.cutting.render import SPEC_GAIN, food_light

    _, near, nrm, _ = renderer.food_pixels(x, piece)
    use = (near >= 0) & seen_mask
    ids = near[use]
    diffuse, spec = food_light(renderer.style, nrm[use], renderer.cam.forward)
    colour = photo[..., :3][use].astype(float) / 255.0
    per_pixel = (colour - SPEC_GAIN * gloss[ids][:, None] * spec[:, None]) / diffuse[:, None]
    sums = np.zeros((len(x), 3))
    counts = np.zeros(len(x))
    np.add.at(sums, ids, per_pixel)
    np.add.at(counts, ids, 1.0)
    seen = counts > 0
    albedo = np.zeros((len(x), 3))
    albedo[seen] = np.clip(sums[seen] / counts[seen][:, None], 0.0, 1.0)
    return albedo, seen


def board_top(mesh_world, near_xy=None, radius=0.05) -> float:
    """Height of a board's top face: its highest vertices, near ``near_xy`` if given [m]."""
    v = np.asarray(mesh_world.vertices, dtype=float)
    if near_xy is not None:
        close = np.linalg.norm(v[:, :2] - np.asarray(near_xy, dtype=float)[None, :], axis=1) < radius
        if close.sum() >= 20:
            v = v[close]
    return float(np.percentile(v[:, 2], 98.0))
