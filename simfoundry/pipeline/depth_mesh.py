# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A textured background mesh from the canonical frame's depth map and stage 5's clean plate.

The pipeline reconstructs the objects of a scene and leaves its surroundings out: the support surface
is a plane and everything else is missing. The depth map of the canonical frame already measures those
surroundings, and stage 5 repaints the frame with every detected object removed (the *clean plate*).
This module joins the two. Each depth pixel becomes a vertex, neighbouring pixels are joined into
triangles, and the clean plate is the texture.

Five decisions carry the result, each of which the tests pin:

- **Object pixels move onto the surface they stood on.** The depth under an object is the object, not the
  background that was behind it. Each object mask is replaced by where its rays meet the plane the object
  stood on: the horizontal plane fitted to the measured ring around that object (`mask_surface: local`), or
  z = 0 (`mask_surface: plane`) when the ring does not fit one. A ring that mixes two heights falls back to
  z = 0, so an object standing on something higher than the support plane (a pot on a cooktop) keeps the
  known limit that this module cannot see that surface.
- **Plane pixels stay on the plane.** A pixel within `plane_snap_tol_m` of z = 0 becomes a vertex where its
  ray meets z = 0, so depth noise does not move it. `plane_snap_below_m` and `plane_snap_above_m` set the two
  sides of that band apart (both null keeps the symmetric band). A triangle whose three vertices are all plane pixels (the
  plane layer) is judged only by the loose `plane_max_edge_ratio`; it is never judged by the relief limit,
  so a flat counter top cannot crack along a mask edge.
- **Relief triangles are judged tightly.** Every other triangle must have edges not much longer than one
  pixel's footprint at that depth (`relief_max_edge_ratio`). Joining two pixels on either side of a depth jump
  (a bottle against the wall) would pull a rubber sheet between them, and a surface seen at grazing incidence
  theta stretches by 1/cos(theta), so 4.0 keeps surfaces up to about 75 degrees.
- **Edges and crumbs are trimmed.** `border_trim_px` drops the outer pixels, where the frame has no data, and
  connected pieces of fewer than `min_component_triangles` triangles are dropped (the sawtooth fragments a cut leaves).
- **The file is Y-up.** glTF fixes Y as up, and the light editor's mesh import (`asset_import.prepare_mesh`,
  up axis `auto`) rotates a glTF by +90 degrees about X to get a Z-up scene. A Z-up world written out
  unchanged would lie on its side after that import, so `export_glb` writes the rotation that import undoes.

Frame mapping: stage 5 works on the source frame padded to its image model's aspect ratio and resized
(`source_padded_resized.png`), so the clean plate and the removal masks are in that frame, not on the
depth grid. `padded_geometry` reproduces `pad_image_to_ratio` on the depth grid to find the part of the
padded frame that is the depth grid, the same mapping stage 8 applies with `unpad_image`.

This module needs only numpy, PIL, scipy and trimesh, so it runs in the light editor's environment too.
"""

from __future__ import annotations

import math
import numbers
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

#: Provisional defaults; each is a stage setting (`s_bg_mesh.*`) and none is fitted to more than one scene.
DEFAULT_PARAMS: dict[str, Any] = {
    # Pixels (on the depth grid) the object masks grow by, so the soft edge of a mask and its small
    # misalignment with the depth map do not leave a rim of object depth around the plane patch.
    "mask_dilate_px": 3,
    # Which surface a masked pixel moves onto: "local" reads it from the measured ring around each object (a
    # plane fitted to the ring if it is one flat surface, else a horizontal plane at the ring's dominant height;
    # z = 0 only for a ring too small to read); "plane" uses z = 0 for every object. Provisional.
    "mask_surface": "local",
    # Pixels of the ring around an object mask that define the surface under it. Provisional.
    "ring_px": 6,
    # A ring with fewer measured points than this falls back to z = 0. Provisional; no measurement behind it.
    "min_ring_points": 30,
    # A local surface steeper than this (hypot of dz/dx and dz/dy, so 0.05 is about 3 degrees) falls back to
    # z = 0: a ring that is not a flat floor is not the floor. Provisional.
    "max_local_tilt": 0.05,
    # Median absolute residual (metres) of the ring about its fitted plane. Above it the ring is not one surface
    # (a cooktop next to a counter), so the object is filled at the ring's dominant height instead. Provisional.
    "max_local_mad_m": 0.005,
    # Pixels from each frame edge whose vertices and triangles are dropped. There is no data past the frame,
    # and its edge pixels are the least accurate. Provisional.
    "border_trim_px": 4,
    # Edge length over one pixel's footprint at that depth, for relief triangles (those with a vertex that is
    # not on the plane). 4.0 keeps a surface up to about 75 degrees off the viewing direction and still drops
    # depth jumps (a bottle against the wall behind it is a jump of tens of ratios). 2.0 removes the sawtooth at
    # a counter's front edge but grows the largest dropped piece on the rerun copy from 3423 to 10003 triangles;
    # 4.0 is no worse than the 894363c setting on any count there. Provisional.
    "relief_max_edge_ratio": 4.0,
    # The same measure for plane-layer triangles, only as a safety net against a far grazing edge: the layer
    # is flat by construction, so this is loose. Provisional.
    "plane_max_edge_ratio": 8.0,
    # Connected pieces of the triangulation with fewer triangles than this are dropped: the sawtooth
    # fragments a depth cut leaves at a counter edge. Provisional; no measurement behind it.
    "min_component_triangles": 200,
    # Metres. One rough measurement of +8 mm on one counter region is the whole basis for 1 cm.
    "plane_snap_tol_m": 0.01,
    # Metres under and over z = 0 for the snap band (see plane_snap_mask). None takes plane_snap_tol_m on that side,
    # so both None is the symmetric band above. The candidate values for these two come from one scene, one capture
    # and one sensor, so they stay out of the defaults; the config and the README say what they do.
    "plane_snap_below_m": None,
    "plane_snap_above_m": None,
    # Metres in the camera frame; None keeps every pixel with a positive depth.
    "z_far": None,
}

#: Applying this to a Z-up world point gives its glTF (Y-up) position: (x, y, z) -> (x, z, -y). It is
#: the inverse of the +90 degrees about X that the light editor applies to a glTF on import.
WORLD_TO_GLTF = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, -1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
)


# --------------------------------------------------------------------------------------
# Frame mapping
# --------------------------------------------------------------------------------------

def padded_geometry(grid_hw: tuple[int, int], frame_hw: tuple[int, int]) -> dict[str, Any]:
    """Where the depth grid sits inside the padded frame stage 5 works in.

    Mirrors `simfoundry.utils.processing_utils.pad_image_to_ratio` applied to an image of the depth
    grid's size with the frame's aspect ratio as the target, including its integer truncation.

    Args:
        grid_hw: (height, width) of the depth grid.
        frame_hw: (height, width) of stage 5's padded frame (a clean plate or a removal mask).

    Returns:
        dict: `padded_hw` (the depth grid padded to the frame's ratio), `pad` ((delta_w, delta_h) as
        `pad_image_to_ratio` returns it) and `box` ((left, top, right, bottom) in frame pixels: the part
        of the frame that is the depth grid).
    """
    h, w = grid_hw
    frame_h, frame_w = frame_hw
    target_ratio = frame_w / frame_h
    if w / h == target_ratio:
        new_h, new_w = h, w
    elif w / h > target_ratio:
        new_h, new_w = int(w / target_ratio), w
    else:
        new_h, new_w = h, int(h * target_ratio)
    delta_w, delta_h = new_w - w, new_h - h
    left, top = delta_w // 2, delta_h // 2
    right, bottom = delta_w - left, delta_h - top
    box = (
        int(round(frame_w * left / new_w)),
        int(round(frame_h * top / new_h)),
        frame_w - int(round(frame_w * right / new_w)),
        frame_h - int(round(frame_h * bottom / new_h)),
    )
    return {"padded_hw": (new_h, new_w), "pad": (delta_w, delta_h), "box": box}


def crop_to_grid_frame(frame: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    """The part of a padded frame that is the depth grid, at the frame's own resolution."""
    left, top, right, bottom = padded_geometry(grid_hw, frame.shape[:2])["box"]
    return frame[top:bottom, left:right]


def frame_mask_to_grid(mask: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    """A boolean mask in stage 5's padded frame, on the depth grid (nearest neighbour)."""
    cropped = crop_to_grid_frame(np.asarray(mask) > 0, grid_hw).astype(np.uint8) * 255
    resized = Image.fromarray(cropped).resize((grid_hw[1], grid_hw[0]), Image.NEAREST)
    return np.asarray(resized) > 127


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------

def pixel_rays(grid_hw: tuple[int, int], K: np.ndarray) -> np.ndarray:
    """Camera-frame ray (z = 1) through each pixel, shape (H, W, 3).

    Pixel indices are used as coordinates without a half-pixel offset, the convention of
    `compute_point_cloud_from_depth`, so the mesh and the stage-4 point cloud agree exactly.
    """
    h, w = grid_hw
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).astype(np.float64)
    return uv1 @ np.linalg.inv(np.asarray(K, dtype=np.float64)).T


def to_world(points_cam: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    """Apply a 4x4 camera-to-world transform to (..., 3) points."""
    cam2world = np.asarray(cam2world, dtype=np.float64)
    return points_cam @ cam2world[:3, :3].T + cam2world[:3, 3]


def to_camera(points_world: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    """Inverse of `to_world` (the transform is a rigid motion, so its inverse is the transpose)."""
    cam2world = np.asarray(cam2world, dtype=np.float64)
    return (points_world - cam2world[:3, 3]) @ cam2world[:3, :3]


def plane_intersections(
    rays_world: np.ndarray, centre: np.ndarray, height: float = 0.0, min_slope: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Where rays from `centre` meet the horizontal plane z = `height`.

    Args:
        rays_world: (..., 3) ray directions in the world frame (any length).
        centre: the camera centre in the world frame.
        height: the plane's z.
        min_slope: rays whose |dz| is below this are parallel to the plane.

    Returns:
        tuple: the (..., 3) intersection points (undefined where invalid) and a boolean array that is
        True where the ray does reach the plane in front of the camera.
    """
    return plane_intersections_general(rays_world, centre, 0.0, 0.0, height, min_slope)


def plane_intersections_general(
    rays_world: np.ndarray, centre: np.ndarray, a=0.0, b=0.0, c=0.0, min_slope: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Where rays from `centre` meet the surface z = a x + b y + c (a plane; a = b = 0 is horizontal at z = c).

    `a`, `b` and `c` may be scalars or arrays that broadcast with the rays, so each pixel can have its own plane.
    The denominator is dz - a dx - b dy, the rate at which the ray's height changes relative to the surface.

    Returns:
        tuple: the (..., 3) intersection points (undefined where invalid) and a boolean array that is True where
        the ray does reach the surface in front of the camera.
    """
    dx, dy, dz = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]
    denominator = dz - a * dx - b * dy
    numerator = a * centre[0] + b * centre[1] + c - centre[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = numerator / denominator
    valid = (np.abs(denominator) > min_slope) & np.isfinite(t) & (t > 0)
    t = np.where(valid, t, 0.0)
    return centre + rays_world * t[..., None], valid


def grid_triangles(
    valid: np.ndarray,
    points: np.ndarray,
    z_cam: np.ndarray,
    focal_px: float,
    max_edge_ratio: float,
    plane_mask: np.ndarray | None = None,
    plane_max_edge_ratio: float | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Two triangles per pixel quad, minus the invalid and the stretched ones.

    The winding makes a surface that faces the camera face outwards. Seen from above, as in a tabletop
    capture, the support plane therefore has +z normals in the Z-up world.

    A triangle whose three vertices are all on the plane (`plane_mask`) is judged by `plane_max_edge_ratio`
    and every other triangle by `max_edge_ratio`. The plane layer is flat, so it can carry a looser limit
    without letting a depth jump through; sharing one limit would crack the plane wherever a mask edge
    moves its pixels onto z = 0 at a grazing angle.

    Args:
        valid: (H, W) bool, vertices that exist.
        points: (H, W, 3) vertex positions (any frame; only distances are used).
        z_cam: (H, W) each vertex's depth along the camera axis.
        focal_px: pixel focal length in pixels (the mean of fx and fy).
        max_edge_ratio: the longest allowed edge of a relief triangle, in units of one pixel's footprint at that depth.
        plane_mask: (H, W) bool, vertices on the support plane; None means there are none.
        plane_max_edge_ratio: the same limit for plane-layer triangles; None uses `max_edge_ratio`.

    Returns:
        tuple: (n, 3) vertex indices into the flattened grid, and counts of what was dropped.
    """
    h, w = valid.shape
    index = np.arange(h * w).reshape(h, w)
    a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    triangles = np.concatenate([
        np.stack([a, c, b], axis=-1).reshape(-1, 3),
        np.stack([b, c, d], axis=-1).reshape(-1, 3),
    ])
    n_total = len(triangles)

    flat_valid = valid.reshape(-1)
    ok = flat_valid[triangles].all(axis=1)
    n_invalid = int((~ok).sum())

    flat_plane = np.zeros(h * w, dtype=bool) if plane_mask is None else np.asarray(plane_mask, dtype=bool).reshape(-1)
    on_plane = flat_plane[triangles].all(axis=1)
    limit = np.where(on_plane, max_edge_ratio if plane_max_edge_ratio is None else plane_max_edge_ratio, max_edge_ratio)

    flat_points = points.reshape(-1, 3)
    flat_z = z_cam.reshape(-1)
    worst = np.zeros(len(triangles))
    for i, j in ((0, 1), (1, 2), (0, 2)):
        u, v = triangles[:, i], triangles[:, j]
        # An edge between diagonal neighbours spans sqrt(2) pixels; the others span one.
        steps = np.where(
            np.abs((u // w) - (v // w)) + np.abs((u % w) - (v % w)) == 2, np.sqrt(2.0), 1.0,
        )
        footprint = 0.5 * (flat_z[u] + flat_z[v]) / focal_px * steps
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.linalg.norm(flat_points[u] - flat_points[v], axis=1) / footprint
        worst = np.maximum(worst, np.where(np.isfinite(ratio), ratio, np.inf))
    stretched = ok & (worst > limit)
    ok &= ~stretched
    return triangles[ok], {
        "triangles_total": n_total,
        "triangles_dropped_invalid": n_invalid,
        "triangles_dropped_stretched": int(stretched.sum()),
        "triangles_dropped_stretched_plane": int((stretched & on_plane).sum()),
        "triangles_dropped_stretched_relief": int((stretched & ~on_plane).sum()),
    }


def snap_to_plane(points: np.ndarray, tol: float, height: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Set points within `tol` of the plane z = `height` onto it. Returns the points and the snapped mask."""
    snapped = np.abs(points[..., 2] - height) < tol
    out = points.copy()
    out[..., 2] = np.where(snapped, height, out[..., 2])
    return out, snapped


def plane_snap_mask(z: np.ndarray, below: float, above: float) -> np.ndarray:
    """Which heights snap onto the support plane z = 0: those with `-below < z < above`, strict at both ends.

    Both ends are strict on purpose: with `below == above == tol` the mask is `np.abs(z) < tol` for every input,
    NaN and infinities included, so the symmetric band is the old one bit for bit. A zero side turns that side off,
    and it leaves out z = 0 itself, as a zero symmetric tolerance does.

    Args:
        z: heights in metres, after the object pixels have moved.
        below: how far under the plane a height may be and still snap, in metres.
        above: how far over the plane a height may be and still snap, in metres.

    Returns:
        np.ndarray: a boolean mask shaped like `z`.
    """
    z = np.asarray(z, dtype=np.float64)
    return (z > -below) & (z < above)


def fit_plane_robust(points: np.ndarray, iterations: int = 4):
    """Least-squares plane z = a x + b y + c, with the points that do not fit dropped and refitted.

    A ring of measured surface around an object can catch a neighbouring object, so each round drops points
    whose residual lies further than 3 robust standard deviations (1.4826 x MAD, at least 2 mm) from the
    median residual, then refits on what is left.

    Args:
        points: (N, 3) world points.
        iterations: refit rounds.

    Returns:
        tuple: (a, b, c, mad) with mad the median absolute residual of the kept points, or None with fewer than
        three points (or fewer than three kept).
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return None
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    keep = np.ones(len(points), dtype=bool)
    for _ in range(iterations):
        if keep.sum() < 3:
            return None
        A = np.column_stack([x[keep], y[keep], np.ones(keep.sum())])
        (a, b, c), *_ = np.linalg.lstsq(A, z[keep], rcond=None)
        resid = z - (a * x + b * y + c)
        centre = np.median(resid[keep])
        spread = np.median(np.abs(resid[keep] - centre))
        keep = np.abs(resid - centre) <= max(3 * 1.4826 * spread, 0.002)
    if keep.sum() < 3:
        return None
    A = np.column_stack([x[keep], y[keep], np.ones(keep.sum())])
    (a, b, c), *_ = np.linalg.lstsq(A, z[keep], rcond=None)
    resid = z[keep] - (a * x[keep] + b * y[keep] + c)
    return float(a), float(b), float(c), float(np.median(np.abs(resid)))


def dominant_height(z: np.ndarray, band_m: float = 0.005) -> float:
    """The median of the most populated `band_m` band of heights.

    A ring can hold two surfaces at once (a sink beside a counter). The median of all its heights then lands
    between them, at a height neither surface has, so the wider surface is found by counting per band instead.
    A tie goes to the lower band.
    """
    z = np.asarray(z, dtype=np.float64)
    bands = np.floor(z / band_m).astype(np.int64)
    values, counts = np.unique(bands, return_counts=True)
    return float(np.median(z[bands == values[np.argmax(counts)]]))


def local_support_planes(mask: np.ndarray, points: np.ndarray, usable: np.ndarray, params: dict[str, Any]):
    """The surface each object stood on, read from the measured ring around it.

    Each 8-connected piece of the mask is one object. The ring is the `ring_px` pixels around that piece, outside
    every mask, with a measured depth. In order:

    - fewer than `min_ring_points` ring points: the support plane z = 0, since the ring says nothing about the surface;
    - a fitted plane z = a x + b y + c that is flat enough (`max_local_tilt`) and fits the ring (`max_local_mad_m`):
      that plane (kind 1);
    - otherwise a horizontal plane at the ring's dominant height (`dominant_height`), kind 2. The support plane is
      not used here: a counter 1 cm above z = 0 with a mask over it would otherwise leave a 1 cm step at the mask edge.

    With `mask_surface` "plane" every object is on z = 0 (kind 0).

    Args:
        mask: (H, W) bool, the object masks on the depth grid (already dilated).
        points: (H, W, 3) measured world points.
        usable: (H, W) bool, pixels with a measured depth and not masked.
        params: the settings (`mask_surface`, `ring_px`, `min_ring_points`, `max_local_tilt`, `max_local_mad_m`).

    Returns:
        tuple: coefficients a, b, c (each (H, W), the plane of each masked pixel; zero outside the masks) and `kind`
        (H, W) int8: 0 support plane, 1 fitted local plane, 2 horizontal plane at the ring's dominant height.
    """
    from scipy.ndimage import binary_dilation, label

    h, w = mask.shape
    a = np.zeros((h, w))
    b = np.zeros((h, w))
    c = np.zeros((h, w))
    kind = np.zeros((h, w), dtype=np.int8)
    if params["mask_surface"] != "local":
        return a, b, c, kind
    structure = np.ones((3, 3), dtype=bool)
    pieces, n_pieces = label(mask, structure=structure)
    for i in range(1, n_pieces + 1):
        piece = pieces == i
        grown = binary_dilation(piece, structure=structure, iterations=int(params["ring_px"])) if params["ring_px"] > 0 else piece
        ring = grown & ~mask & usable
        ring_points = points[ring]
        if len(ring_points) < int(params["min_ring_points"]):
            continue
        fit = fit_plane_robust(ring_points)
        if fit is not None:
            fa, fb, fc, mad = fit
            if np.hypot(fa, fb) <= float(params["max_local_tilt"]) and mad <= float(params["max_local_mad_m"]):
                a[piece], b[piece], c[piece] = fa, fb, fc
                kind[piece] = 1
                continue
        c[piece] = dominant_height(ring_points[:, 2])
        kind[piece] = 2
    return a, b, c, kind


def drop_small_components(triangles: np.ndarray, n_vertices: int, min_triangles: int) -> tuple[np.ndarray, int]:
    """Drop the connected pieces of a triangulation that have fewer than `min_triangles` triangles.

    Pieces are joined through shared vertices. A cut through a sawtooth edge leaves crumbs of a few triangles
    that are not part of the surface.

    Returns:
        tuple: the kept triangles and the number of triangles dropped.
    """
    if min_triangles <= 0 or len(triangles) == 0:
        return triangles, 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_vertices, n_vertices))
    n_components, labels = connected_components(graph, directed=False)
    piece_of_triangle = labels[triangles[:, 0]]
    sizes = np.bincount(piece_of_triangle, minlength=n_components)
    keep = sizes[piece_of_triangle] >= min_triangles
    return triangles[keep], int((~keep).sum())


# --------------------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------------------

def build_background_mesh(
    depth: np.ndarray,
    K: np.ndarray,
    cam2world: np.ndarray,
    plate: np.ndarray,
    object_mask: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
):
    """The textured background mesh, in the stage-4 world (Z-up, metres, support plane at z = 0).

    Args:
        depth: (H, W) metric depth along the camera axis; pixels with depth <= 0 or not finite have no vertex.
        K: 3x3 intrinsics of that depth grid.
        cam2world: 4x4 transform from the camera frame to the world (`image_<idx>_cam2world.npy`).
        plate: stage 5's clean plate, (h, w, 3) uint8, in the padded frame.
        object_mask: boolean (h, w) union of the removal masks in the same padded frame, or None.
        params: overrides of `DEFAULT_PARAMS`.

    Returns:
        tuple: the `trimesh.Trimesh` (with texture and UVs), the texture as a PIL image, and a dict of
        counts and settings for `stage_info.json`.
    """
    import trimesh
    from scipy.ndimage import binary_dilation

    p = {**DEFAULT_PARAMS, **(params or {})}
    if p["mask_surface"] not in ("local", "plane"):
        raise ValueError(f"s_bg_mesh.mask_surface must be 'local' or 'plane', got {p['mask_surface']!r}")
    for key in ("plane_snap_below_m", "plane_snap_above_m"):
        value = p[key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"s_bg_mesh.{key} must be null or a finite number of metres >= 0, got {value!r}")
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    cam2world = np.asarray(cam2world, dtype=np.float64)
    h, w = depth.shape

    rays_cam = pixel_rays((h, w), K)
    points_cam = rays_cam * depth[..., None]
    points = to_world(points_cam, cam2world)
    rays_world = rays_cam @ cam2world[:3, :3].T
    centre = cam2world[:3, 3]

    in_range = np.isfinite(depth) & (depth > 0)
    if p["z_far"] is not None:
        in_range &= depth <= float(p["z_far"])

    n_masked = 0
    n_substituted_invalid = 0
    n_fitted = 0
    n_ring_level = 0
    depth_cam = depth
    if object_mask is not None and np.asarray(object_mask).any():
        mask = frame_mask_to_grid(object_mask, (h, w))
        if p["mask_dilate_px"]:
            mask = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=int(p["mask_dilate_px"]))
        n_masked = int(mask.sum())
        slope_a, slope_b, slope_c, kind = local_support_planes(mask, points, in_range & ~mask, p)
        plane_points, hits = plane_intersections_general(rays_world, centre, slope_a, slope_b, slope_c)
        replace = mask & hits
        points = np.where(replace[..., None], plane_points, points)
        z_new = to_camera(plane_points, cam2world)[..., 2]
        depth_cam = np.where(replace, z_new, depth)
        in_range = np.where(mask, hits & ((z_new <= float(p["z_far"])) if p["z_far"] is not None else True), in_range)
        n_substituted_invalid = int((mask & ~hits).sum())
        n_fitted = int((replace & (kind == 1)).sum())
        n_ring_level = int((replace & (kind == 2)).sum())

    # Pixels from the frame edge have no neighbours on one side, and the edge of a measured depth map is its
    # least accurate part, so they go with their triangles.
    trim = int(p["border_trim_px"])
    n_before_trim = int(in_range.sum())
    if trim > 0:
        in_range = in_range.copy()
        in_range[:trim] = False
        in_range[-trim:] = False
        in_range[:, :trim] = False
        in_range[:, -trim:] = False
    n_trimmed = n_before_trim - int(in_range.sum())

    # A plane pixel is one within the snap band of z = 0 after the object pixels have moved. Its triangles
    # form the plane layer, judged by the loose plane limit; the rest are relief, judged by the tight one. A side
    # left null takes the symmetric tolerance, so the defaults keep the band they always had.
    tol = float(p["plane_snap_tol_m"])
    below = tol if p["plane_snap_below_m"] is None else float(p["plane_snap_below_m"])
    above = tol if p["plane_snap_above_m"] is None else float(p["plane_snap_above_m"])
    plane_flag = in_range & plane_snap_mask(points[..., 2], below, above)
    focal = 0.5 * (K[0, 0] + K[1, 1])
    triangles, counts = grid_triangles(
        in_range, points, depth_cam, focal, float(p["relief_max_edge_ratio"]),
        plane_mask=plane_flag, plane_max_edge_ratio=float(p["plane_max_edge_ratio"]),
    )
    triangles, n_small = drop_small_components(triangles, h * w, int(p["min_component_triangles"]))
    on_plane_triangle = plane_flag.reshape(-1)[triangles].all(axis=1)

    # The heights before the snap, so each snapped vertex is counted by the side it came from.
    pre_snap_z = points[..., 2].copy()
    # A plane vertex is where its ray meets z = 0. That depends on the pixel and the pose, not on the depth
    # noise, so the plane does not wobble with the measurement. Vertices are shared by both layers, so each
    # is one point and no seam opens between them.
    on_plane, hits_plane = plane_intersections(rays_world, centre)
    use = plane_flag & hits_plane
    points = np.where(use[..., None], on_plane, points)
    points[..., 2] = np.where(plane_flag & ~hits_plane, 0.0, points[..., 2])

    used = np.unique(triangles)
    plane_used = plane_flag.reshape(-1)[used]
    snapped_from_below = int((plane_used & (pre_snap_z.reshape(-1)[used] < 0)).sum())
    snapped_from_above = int((plane_used & (pre_snap_z.reshape(-1)[used] > 0)).sum())
    remap = np.full(h * w, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    rows, cols = np.divmod(used, w)
    # Pixel centres sit at integer coordinates (the convention of the unprojection), so a pixel's
    # texture coordinate is (index + 0.5) / size. trimesh's v runs from the bottom of the image.
    uv = np.stack([(cols + 0.5) / w, 1.0 - (rows + 0.5) / h], axis=1)

    texture = Image.fromarray(np.ascontiguousarray(crop_to_grid_frame(np.asarray(plate), (h, w))))
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True,
    )
    mesh = trimesh.Trimesh(
        vertices=points.reshape(-1, 3)[used],
        faces=remap[triangles],
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
        process=False,
    )
    bounds = mesh.bounds if len(mesh.vertices) else np.zeros((2, 3))
    n_replaced = n_masked - n_substituted_invalid
    stats = {
        "grid_hw": [h, w],
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        **counts,
        "triangles_dropped_small_components": n_small,
        "triangles_plane_layer": int(on_plane_triangle.sum()),
        "triangles_relief": int((~on_plane_triangle).sum()),
        "vertices_trimmed_border": n_trimmed,
        "object_pixels_replaced_by_plane": n_replaced,
        "object_pixels_without_plane_hit": n_substituted_invalid,
        "object_pixels_local_surface": n_fitted,
        "object_pixels_ring_level": n_ring_level,
        "object_pixels_support_plane": n_replaced - n_fitted - n_ring_level,
        "vertices_snapped_to_plane": int(plane_flag.reshape(-1)[used].sum()),
        "vertices_snapped_from_below": snapped_from_below,
        "vertices_snapped_from_above": snapped_from_above,
        "plane_snap_below_m_used": below,
        "plane_snap_above_m_used": above,
        "bounds_min_m": [float(x) for x in bounds[0]],
        "bounds_max_m": [float(x) for x in bounds[1]],
        "texture_hw": [texture.height, texture.width],
        "params": {k: p[k] for k in DEFAULT_PARAMS},
    }
    return mesh, texture, stats


def export_glb(mesh, path: str | os.PathLike) -> None:
    """Write a Z-up world mesh as a spec-conforming (Y-up) GLB.

    Args:
        mesh: a `trimesh.Trimesh` in the Z-up world.
        path: output `.glb` path.
    """
    out = mesh.copy()
    out.apply_transform(WORLD_TO_GLTF)
    out.export(str(path), file_type="glb")


# --------------------------------------------------------------------------------------
# Stage 5 outputs
# --------------------------------------------------------------------------------------

_ITER_PNG = re.compile(r"^iter_(\d+)\.png$")


def find_stage5_inputs(stage5_dir: str | os.PathLike) -> dict[str, Any]:
    """The clean plate and removal masks stage 5 left in `stage5_dir`.

    The clean plate is the repainted frame of the last iteration (every object removed up to there).
    A masks list holds each removal mask of that iteration or an earlier one. Iteration numbers can
    have gaps, so files are listed, not counted. Without any `post_object_removal` image (no object
    was removed) the source frame itself is the plate and there are no masks.

    Args:
        stage5_dir: `Data/<scene>/s5_scene`.

    Returns:
        dict: `plate` (path), `masks` (paths, ascending), `iteration` (int, or None for the source frame).

    Raises:
        FileNotFoundError: when neither a repainted frame nor `source_padded_resized.png` exists.
    """
    stage5_dir = Path(stage5_dir)
    done = _iteration_files(stage5_dir / "post_object_removal")
    if done:
        last = max(done)
        masks = _iteration_files(stage5_dir / "removal_mask")
        return {
            "plate": str(done[last]),
            "masks": [str(masks[i]) for i in sorted(masks) if i <= last],
            "iteration": last,
        }
    source = stage5_dir / "source_padded_resized.png"
    if not source.is_file():
        raise FileNotFoundError(
            f"No clean plate under {stage5_dir}: neither post_object_removal/iter_<n>.png nor "
            "source_padded_resized.png exists. Run stage 5 first."
        )
    return {"plate": str(source), "masks": [], "iteration": None}


def _iteration_files(directory: Path) -> dict[int, Path]:
    if not directory.is_dir():
        return {}
    found = {}
    for name in os.listdir(directory):
        match = _ITER_PNG.match(name)
        if match:
            found[int(match.group(1))] = directory / name
    return found
