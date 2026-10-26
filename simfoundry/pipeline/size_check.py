# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 8 places a mesh by one scale and one pose; this checks its size against the depth.

Stage 8 fits a single scalar (pre_scale_factor times the CPD scale). Nothing compares the result with the object
as the depth saw it, so a mesh 14 % too wide or a third too thin passes silently. This module measures two things
on both sides, in the camera frame where the placed mesh and the depth cloud already live:

- the footprint: the two sides of the minimum-area rectangle of the points projected on the plane perpendicular
  to gravity, longer side first;
- the height: the extent of the mesh along gravity, against the depth's height above the surface it stands on.

Both are reported. Only a warning is raised by default; `decide_height_fit` says when the height may be stretched
to match the depth, and `fit_height` does the stretch. Nothing here reads or writes files.

The height's floor is itself measured, and it scatters: a height that is off by less than that scatter is noise, not
a size error. So a height only counts as off when it is beyond both `size_warn_ratio` of the depth and
`height_noise_sigmas` times the floor's robust spread (`floor_spread_m`). When the floor is read from a neighbouring
object's pixels, the spread is that of those pixels and the floor is their most populated height bin
(`dominant_floor`), not their median, which their walls would pull.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

#: A surface needs this many points around the object to be read as "the surface it stands on".
DEFAULT_MIN_RING_POINTS = 50
#: Width of the height bins `dominant_floor` counts in. Provisional: 5 mm, the order of the spreads (1 to 5 mm) that the
#: lidar run's flat counter rings measured. Not tuned on any object.
FLOOR_BIN_M = 0.005


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _plane_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to `up`."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, helper)
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def footprint_sides(points, up) -> tuple[float, float] | None:
    """The two sides of the minimum-area rectangle of `points` projected on the plane perpendicular to `up`.

    The rectangle is found in the plane, so its sides do not depend on the axes the camera happens to use. They are
    returned longer side first, which pairs the sides of two footprints by rank.

    Returns:
        (long, short) in the points' units, or None when there are fewer than three points or they are collinear.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return None
    up = _unit(up)
    e1, e2 = _plane_basis(up)
    planar = np.stack([points @ e1, points @ e2], axis=1).astype(np.float32)
    _, (w, h), _ = cv2.minAreaRect(planar)
    if min(w, h) < 1e-9:
        return None
    return tuple(sorted((float(w), float(h)), reverse=True))


def _facing_camera(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Boolean per face: True where the face looks at the camera, which sits at the origin.

    A face looks at the camera when its normal points back towards the origin, i.e. the dot product of the normal
    with the face's centre is negative.
    """
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centres = triangles.mean(axis=1)
    return np.einsum("ij,ij->i", normals, centres) < 0


def visible_vertices(vertices, faces) -> np.ndarray:
    """The mesh's vertices that belong to at least one face looking at the camera (points as an (N, 3) array).

    Only these are comparable with a depth cloud, which sees the camera-facing surface and nothing behind it.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    visible = faces[_facing_camera(vertices, faces)]
    return vertices[np.unique(visible)] if len(visible) else np.empty((0, 3))


def robust_spread(values) -> float:
    """The robust standard deviation of `values`: 1.4826 times their median absolute deviation.

    The factor makes it the standard deviation for normal values. It is a median, so a few stray floor points do not
    widen it.
    """
    values = np.asarray(values, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def dominant_floor(heights) -> float:
    """The median of the most populated `FLOOR_BIN_M`-wide height bin; a tie goes to the lower bin.

    Used when the points include a neighbouring object's pixels: the surface under this object is their bulk, while
    their median is pulled by the neighbour's walls and edges.
    """
    heights = np.asarray(heights, dtype=np.float64)
    bins = np.floor(heights / FLOOR_BIN_M).astype(np.int64)
    populated, counts = np.unique(bins, return_counts=True)
    chosen = populated[np.argmax(counts)]
    return float(np.median(heights[bins == chosen]))


def _footprint_report(mesh_points: np.ndarray, depth_points: np.ndarray, up: np.ndarray, warn: float) -> dict[str, Any]:
    mesh_sides = footprint_sides(mesh_points, up)
    depth_sides = footprint_sides(depth_points, up)
    if mesh_sides is None:
        return {"skipped": "the mesh's footprint is degenerate"}
    if depth_sides is None:
        return {"skipped": "the depth footprint has fewer than three points or is degenerate"}
    ratios = [mesh_sides[0] / depth_sides[0], mesh_sides[1] / depth_sides[1]]
    return {
        "mesh_sides_m": [float(s) for s in mesh_sides],
        "depth_sides_m": [float(s) for s in depth_sides],
        "ratios": [float(r) for r in ratios],
        "max_deviation": float(max(abs(r - 1.0) for r in ratios)),
    }


def _height_report(mesh_points: np.ndarray, depth_points: np.ndarray, ring_points: np.ndarray, up: np.ndarray,
                   settings: dict[str, Any], fallback_points: np.ndarray | None = None) -> dict[str, Any]:
    mesh_up = mesh_points @ up
    mesh_m = float(mesh_up.max() - mesh_up.min())
    if mesh_m <= 0.0:
        return {"skipped": "the mesh has no extent along the up direction"}
    ring_up = np.asarray(ring_points, dtype=np.float64).reshape(-1, 3) @ up
    min_ring = int(settings.get("min_ring_points", DEFAULT_MIN_RING_POINTS))
    if len(ring_up) >= min_ring:
        # The median, not the minimum: a few depth returns below the surface must not move the floor.
        floor_points, floor, source = ring_up, float(np.median(ring_up)), "ring"
    else:
        # The object's own surroundings with no neighbour excluded: when a neighbour fills the ring, the surface under
        # this object is that neighbour's visible surface (the tofu on the tray in the lidar run).
        fallback_up = np.empty(0)
        if fallback_points is not None:
            fallback_up = np.asarray(fallback_points, dtype=np.float64).reshape(-1, 3) @ up
        if not settings.get("floor_fallback", False) or len(fallback_up) < min_ring:
            return {"skipped": f"only {len(ring_up)} points around the object to read the surface from, "
                               f"need {min_ring}"}
        floor_points, floor, source = fallback_up, dominant_floor(fallback_up), "neighbour_masks"
    top = float(np.percentile(np.asarray(depth_points, dtype=np.float64) @ up, settings["height_percentile"]))
    depth_m = top - floor
    if depth_m <= 0.0:
        return {"skipped": "the depth shows no height above the surface"}
    ratio = mesh_m / depth_m
    return {
        "mesh_m": mesh_m,
        "depth_m": depth_m,
        "ratio": ratio,
        "deviation": float(abs(ratio - 1.0)),
        "deviation_m": float(abs(mesh_m - depth_m)),
        "floor_m": floor,
        "floor_spread_m": robust_spread(floor_points),
        "floor_source": source,
        "top_m": top,
    }


def _height_exceeds(height: dict[str, Any], ratio: float, sigmas: float | None) -> bool:
    """Whether a comparable height is off by more than `ratio` of the depth AND by more than the floor's own noise.

    The two limits together are the same as the larger of them: the mismatch must clear `ratio * depth_m` and also
    `sigmas * floor_spread_m`. `sigmas` None is the ratio alone, as before the floor's spread was measured. A mismatch
    inside the floor's own scatter cannot be told apart from that scatter, so it is not reported.
    """
    if not height["deviation"] > ratio:
        return False
    if sigmas is None:
        return True
    return height["deviation_m"] > sigmas * height["floor_spread_m"]


def size_report(mesh_vertices, mesh_faces, depth_points, ring_points, up, settings: dict[str, Any],
                fallback_points=None) -> dict[str, Any]:
    """Compare the placed mesh with the depth: footprint, height, and which of the two are over the threshold.

    Args:
        mesh_vertices: (N, 3) vertices of the placed mesh, camera frame, metres.
        mesh_faces: (F, 3) vertex indices of that mesh.
        depth_points: (M, 3) the object's depth points, camera frame.
        ring_points: (K, 3) depth points on the surface just around the object (outside every removed mask).
        up: the world up direction in the camera frame (need not be a unit vector).
        settings: `size_warn_ratio` (float; a deviation above it warns), `visible_only` (compare only the
            camera-facing part of the mesh), `height_percentile` (the depth height's percentile),
            `min_ring_points` (optional; default DEFAULT_MIN_RING_POINTS), `height_noise_sigmas` (optional; the
            floor's spreads a height must also clear, None for the ratio alone), `floor_fallback` (optional; read the
            floor from `fallback_points` when the ring is too thin).
        fallback_points: (J, 3) depth points around the object with no neighbour excluded, for the fallback floor.

    Returns:
        dict: `footprint` and `height`, each with its numbers or a `skipped` reason, and `warnings`, a list of
        "footprint" and/or "height". Only Python floats and lists, so it serialises to JSON.
    """
    up = _unit(up)
    warn = float(settings["size_warn_ratio"])
    vertices = np.asarray(mesh_vertices, dtype=np.float64)
    faces = np.asarray(mesh_faces, dtype=np.int64)
    footprint_points = visible_vertices(vertices, faces) if settings["visible_only"] else vertices
    footprint = _footprint_report(footprint_points, np.asarray(depth_points, dtype=np.float64), up, warn)
    height = _height_report(vertices, np.asarray(depth_points, dtype=np.float64),
                            np.asarray(ring_points, dtype=np.float64), up, settings, fallback_points)
    warnings = []
    if "max_deviation" in footprint and footprint["max_deviation"] > warn:
        warnings.append("footprint")
    if "deviation" in height and _height_exceeds(height, warn, settings.get("height_noise_sigmas")):
        warnings.append("height")
    return {"footprint": footprint, "height": height, "warnings": warnings}


def decide_height_fit(report: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Whether the mesh's height should be stretched to the depth's, and why not if it should not.

    Stretching only ever changes the height, so it can only fix a height mismatch; a footprint mismatch is left for a
    person to look at. It is applied only when it is enabled, the height could be compared, the mismatch is above
    `threshold` and above the floor's noise (the same test as the warning), and the factor needed is inside
    [min_scale, max_scale]. Outside those limits the mismatch is more likely a wrong depth or a wrong mesh than a
    stretched mesh, so the height is left as it is.

    Returns:
        dict: `applied` (bool), `scale` (the factor, or None), `skipped` (the reason, or None when applied).
    """
    if not settings["enabled"]:
        return {"applied": False, "scale": None, "skipped": "height_fit is disabled"}
    height = report["height"]
    if "skipped" in height:
        return {"applied": False, "scale": None, "skipped": f"height not comparable: {height['skipped']}"}
    if not height["deviation"] > settings["threshold"]:
        return {"applied": False, "scale": None,
                "skipped": f"height deviation {height['deviation']:.3f} within the threshold {settings['threshold']}"}
    if not _height_exceeds(height, settings["threshold"], settings.get("height_noise_sigmas")):
        return {"applied": False, "scale": None,
                "skipped": f"height deviation {height['deviation_m']:.4f} m within the floor's noise "
                           f"({settings['height_noise_sigmas']} x {height['floor_spread_m']:.4f} m)"}
    scale = height["depth_m"] / height["mesh_m"]
    if not settings["min_scale"] <= scale <= settings["max_scale"]:
        return {"applied": False, "scale": None,
                "skipped": f"scale {scale:.3f} outside [{settings['min_scale']}, {settings['max_scale']}]"}
    return {"applied": True, "scale": float(scale), "skipped": None}


def fit_height(vertices, up, scale: float) -> np.ndarray:
    """Stretch a mesh along `up` by `scale`, keeping its lowest point (along `up`) where it is.

    Points move only along `up`, so the footprint and every other direction are unchanged, and the extent along
    `up` becomes `scale` times what it was.

    Args:
        vertices: (N, 3), in the frame where `up` is the up direction (the canonical frame).
        up: the up direction in that frame.
        scale: the factor for the height.

    Returns:
        (N, 3) new vertices.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    direction = _unit(up)
    along = vertices @ direction
    base = along.min()
    return vertices + (scale - 1.0) * (along - base)[:, None] * direction[None, :]


_GROW_KERNEL = np.ones((3, 3), dtype=np.uint8)


def _grow(mask, px: int) -> np.ndarray:
    """`mask` grown by `px` pixels in every direction (a 3 by 3 square per step)."""
    mask = np.asarray(mask, dtype=bool)
    if px <= 0:
        return mask.copy()
    return cv2.dilate(mask.astype(np.uint8), _GROW_KERNEL, iterations=int(px)).astype(bool)


def ring_mask(own_px, all_px, ring_px: int, gap_px: int) -> np.ndarray:
    """The pixels that show the surface around one object: close to it, and clear of every object mask.

    A neighbouring object's mask, or the edge of any mask, shows that object's depth, not the surface the object
    stands on. So the ring is the object's own mask grown by `ring_px`, minus every mask grown by `gap_px`. The gap
    also keeps the ring off the edges of the object's own mask, which are mixed with the object in the depth.

    Args:
        own_px: (H, W) bool, this object's removal mask on the depth grid.
        all_px: (H, W) bool, the union of every removal mask on that grid (own included).
        ring_px: how far from the object the ring reaches, in pixels.
        gap_px: how far every mask's edge is kept clear of the ring, in pixels (0 keeps the ring off the masks only).

    Returns:
        (H, W) bool, the ring.
    """
    grown = _grow(own_px, ring_px)
    everything = np.asarray(all_px, dtype=bool)
    return grown & ~_grow(everything, gap_px)
