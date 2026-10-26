# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A textured background mesh from the canonical frame's depth map and stage 5's clean plate.

The pipeline reconstructs the objects of a scene and leaves its surroundings out: the support surface
is a plane and everything else is missing. The depth map of the canonical frame already measures those
surroundings, and stage 5 repaints the frame with every detected object removed (the *clean plate*).
This module joins the two. Each depth pixel becomes a vertex, neighbouring pixels are joined into
triangles, and the clean plate is the texture.

Four decisions carry the result, each of which the tests pin:

- **Object pixels are moved onto the support plane.** The depth under an object is the object, not the
  background that was behind it. Every object stood on the support plane (z = 0 in the stage-4 world),
  so a pixel under an object mask is replaced by where its ray meets that plane. That is wrong for an
  object that stood on something higher than the support plane (a pot on a cooktop), which is a known
  limit, not something this module can see.
- **Triangles that stretch across a depth jump are dropped.** Joining two pixels on either side of an
  object's silhouette would pull a rubber sheet between them. A triangle is kept only if each of its
  edges is not much longer than one pixel's footprint at that depth (`max_edge_ratio`), which a surface
  facing the camera satisfies with ratio about 1 and one seen at grazing incidence theta with 1/cos(theta).
- **Vertices within `plane_snap_tol_m` of the support plane are set onto it.** Objects are placed at
  z = 0, so a background that floats a few millimetres above or sinks below the plane would hover or
  bury them.
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
    # Edge length over one pixel's footprint at that depth. 4 keeps a surface up to about 75 degrees off
    # the viewing direction and drops depth jumps (a bottle against the wall behind it is a jump of
    # tens of ratios). No measurement behind the value.
    "max_edge_ratio": 4.0,
    # Metres. One rough measurement of +8 mm on one counter region is the whole basis for 1 cm.
    "plane_snap_tol_m": 0.01,
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
    dz = rays_world[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (height - centre[2]) / dz
    valid = (np.abs(dz) > min_slope) & np.isfinite(t) & (t > 0)
    t = np.where(valid, t, 0.0)
    return centre + rays_world * t[..., None], valid


def grid_triangles(
    valid: np.ndarray,
    points: np.ndarray,
    z_cam: np.ndarray,
    focal_px: float,
    max_edge_ratio: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Two triangles per pixel quad, minus the invalid and the stretched ones.

    The winding makes a surface that faces the camera face outwards. Seen from above, as in a tabletop
    capture, the support plane therefore has +z normals in the Z-up world.

    Args:
        valid: (H, W) bool, vertices that exist.
        points: (H, W, 3) vertex positions (any frame; only distances are used).
        z_cam: (H, W) each vertex's depth along the camera axis.
        focal_px: pixel focal length in pixels (the mean of fx and fy).
        max_edge_ratio: the longest allowed edge, in units of one pixel's footprint at that depth.

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
    stretched = ok & (worst > max_edge_ratio)
    ok &= ~stretched
    return triangles[ok], {
        "triangles_total": n_total,
        "triangles_dropped_invalid": n_invalid,
        "triangles_dropped_stretched": int(stretched.sum()),
    }


def snap_to_plane(points: np.ndarray, tol: float, height: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Set points within `tol` of the plane z = `height` onto it. Returns the points and the snapped mask."""
    snapped = np.abs(points[..., 2] - height) < tol
    out = points.copy()
    out[..., 2] = np.where(snapped, height, out[..., 2])
    return out, snapped


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
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    cam2world = np.asarray(cam2world, dtype=np.float64)
    h, w = depth.shape

    rays_cam = pixel_rays((h, w), K)
    points_cam = rays_cam * depth[..., None]
    points = to_world(points_cam, cam2world)
    centre = cam2world[:3, 3]

    in_range = np.isfinite(depth) & (depth > 0)
    if p["z_far"] is not None:
        in_range &= depth <= float(p["z_far"])

    n_masked = 0
    n_substituted_invalid = 0
    if object_mask is not None and np.asarray(object_mask).any():
        mask = frame_mask_to_grid(object_mask, (h, w))
        if p["mask_dilate_px"]:
            mask = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=int(p["mask_dilate_px"]))
        n_masked = int(mask.sum())
        rays_world = rays_cam @ cam2world[:3, :3].T
        plane_points, hits = plane_intersections(rays_world, centre)
        replace = mask & hits
        points = np.where(replace[..., None], plane_points, points)
        z_new = to_camera(plane_points, cam2world)[..., 2]
        depth_cam = np.where(replace, z_new, depth)
        in_range = np.where(mask, hits & ((z_new <= float(p["z_far"])) if p["z_far"] is not None else True), in_range)
        n_substituted_invalid = int((mask & ~hits).sum())
    else:
        depth_cam = depth

    focal = 0.5 * (K[0, 0] + K[1, 1])
    triangles, counts = grid_triangles(in_range, points, depth_cam, focal, float(p["max_edge_ratio"]))

    points, snapped = snap_to_plane(points, float(p["plane_snap_tol_m"]))

    used = np.unique(triangles)
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
    stats = {
        "grid_hw": [h, w],
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        **counts,
        "object_pixels_replaced_by_plane": n_masked - n_substituted_invalid,
        "object_pixels_without_plane_hit": n_substituted_invalid,
        "vertices_snapped_to_plane": int(snapped.reshape(-1)[used].sum()),
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
