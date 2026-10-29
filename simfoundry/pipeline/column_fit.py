# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit a placed mesh's vertical profile to the depth, column by column (stage 8, off by default).

`size_check.fit_height` stretches a whole mesh by one factor along gravity, so it cannot mend a mesh whose shape along
gravity is wrong. Hunyuan made the lidar run's tofu a bowed plate: 18.9 mm at its thickest but about 11 mm on average
(volume over area), half the 23 mm the lidar measured, and its height ratio against the depth (0.82) is inside the
height check's 0.25, so nothing touched it. Its stage 11 mass (density times volume) came out half the real one.

This module treats the space above the support as columns. In each column it reads three heights: where the mesh's
top and bottom cross it (vertical rays), and the depth's top. Every vertex keeps its fraction of the way from the
mesh's bottom to its top, and that fraction is laid between the support and the depth's top. The bottom then rests on
the support, the top follows the depth, and the triangles, uv and texture stay as they were.

That is only right for a mesh a vertical line crosses once on the way in and once on the way out (a "height field":
a block, a board, a fruit lying down). It is refused for:

- a mesh many columns of which a vertical line crosses more or fewer than twice (a handle, an overhang, a mesh with
  holes): `min_two_hit_share`;
- a container, whose top surface sinks inside its rim (a tray, a bowl, a pot): `max_depression`. Its depth top inside
  the rim is its floor or its contents, which the column fit would wrongly take for the object's top;
- an object whose depth top the sensor cannot resolve: fewer than `min_top_cells` columns keep a depth reading once
  the reading's own rim is dropped (`top_erode_cells`: depth pixels at an object's edge mix the object and what is
  behind it);
- an object the depth already agrees with: the mean column height over the support differs by less than
  `threshold` from the mesh's (the same role as the height fit's threshold).

All heights are along `up`, in whatever frame the vertices and points share (stage 8: the camera frame). Nothing here
reads or writes files. The constants are provisional, from one lidar scene (eight objects); see the README.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from scipy import ndimage

from simfoundry.pipeline.size_check import _plane_basis, _unit

#: The settings and their defaults. `enabled` False keeps stage 8 as it was (no call, no record).
DEFAULT_COLUMN_FIT: dict[str, Any] = {
    "enabled": False,
    # Column width (m). 3 mm: below the lidar's 3.6 mm pixel at 0.7 m, above the 2.25 mm grid stage 2 resamples it to.
    "cell_m": 0.003,
    # Share of the mesh's columns a vertical line must cross exactly twice.
    "min_two_hit_share": 0.9,
    # A container's rim stands above its inside by more than this share of the object's height.
    "max_depression": 0.3,
    # Depth columns dropped at the rim of the depth's own footprint before the top is read.
    "top_erode_cells": 2,
    # Depth columns the top must keep after that.
    "min_top_cells": 20,
    # The mesh's columns must be about equally thick (25th over 75th percentile of the inner columns' thickness): a
    # block, a slab or a board, bowed or not. A dome, a ball or a lying cylinder thins towards its edge, and filling its
    # columns down to the support would turn it into a pedestal.
    "min_thickness_uniformity": 0.7,
    # The support read around the object and the depth stage 5 left under it once removed must agree within this share
    # of the depth's height, or the surface the object stands on is not known (a knife on a board edge).
    "max_support_disagreement": 0.3,
    # Mean column height over the support: depth against mesh, the relative difference that triggers the fit.
    "threshold": 0.25,
}

#: Two crossings closer than this along a ray are one (a ray through an edge or a vertex hits two triangles).
_SAME_HIT_M = 2e-4


def resolve_settings(cfg_section) -> dict[str, Any]:
    """The column fit's settings: the defaults overridden by the given section (a dict, a Hydra node or None)."""
    settings = dict(DEFAULT_COLUMN_FIT)
    if cfg_section:
        for key in DEFAULT_COLUMN_FIT:
            if key in cfg_section and cfg_section[key] is not None:
                settings[key] = cfg_section[key]
    settings["enabled"] = bool(settings["enabled"])
    for key in ("cell_m", "min_two_hit_share", "max_depression", "threshold", "min_thickness_uniformity",
                "max_support_disagreement"):
        settings[key] = float(settings[key])
        if not settings[key] > 0:
            raise ValueError(f"s8_pose.column_fit.{key} must be > 0, got {settings[key]!r}")
    for key in ("top_erode_cells", "min_top_cells"):
        value = settings[key]
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"s8_pose.column_fit.{key} must be an integer >= 0, got {value!r}")
        settings[key] = int(value)
    return settings


def _frame(points: np.ndarray, up: np.ndarray) -> np.ndarray:
    """(N, 3) points as (a, b, h): two coordinates across `up` and the height along it."""
    e1, e2 = _plane_basis(up)
    return np.stack([points @ e1, points @ e2, points @ up], axis=1)


def mesh_columns(vertices, faces, up, cell_m: float, origin=None, shape=None) -> dict[str, Any]:
    """Where vertical lines cross the mesh: per column the number of crossings, the highest and the lowest.

    Args:
        vertices: (N, 3) mesh vertices.
        faces: (F, 3) triangle indices.
        up: the gravity direction (unit or not).
        cell_m: column width (m).
        origin, shape: the grid's corner (a, b) and size; by default the mesh's footprint.

    Returns:
        dict: origin, shape, cell_m, hits (int array), top and bottom (float arrays, NaN where no crossing).
    """
    up = _unit(up)
    v = np.asarray(vertices, dtype=np.float64)
    abh = _frame(v, up)
    if origin is None:
        origin = abh[:, :2].min(axis=0)
        shape = tuple(int(n) for n in np.ceil((abh[:, :2].max(axis=0) - origin) / cell_m) + 1)
    origin = np.asarray(origin, dtype=np.float64)
    ia, ib = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    centres = origin + (np.stack([ia.ravel(), ib.ravel()], axis=1) + 0.5) * cell_m
    e1, e2 = _plane_basis(up)
    start = abh[:, 2].max() + 0.01
    origins = centres[:, :1] * e1 + centres[:, 1:] * e2 + start * up
    directions = np.repeat(-up[None], len(origins), axis=0)
    mesh = trimesh.Trimesh(v, np.asarray(faces), process=False)
    # trimesh's pure-Python intersector, not the default one: with embree installed the default returned one crossing
    # per ray although multiple_hits was asked for (the lidar run's tofu: 1 instead of 2), which reads every mesh as
    # crossed once. The Python one is slower and the same everywhere.
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, ray_index, _ = intersector.intersects_location(origins, directions, multiple_hits=True)
    heights = locations @ up
    hits = np.zeros(len(origins), dtype=np.int64)
    top = np.full(len(origins), np.nan)
    bottom = np.full(len(origins), np.nan)
    order = np.lexsort((-heights, ray_index))
    ray_index, heights = ray_index[order], heights[order]
    for r in np.unique(ray_index):
        h = heights[ray_index == r]
        distinct = h[np.concatenate([[True], np.abs(np.diff(h)) > _SAME_HIT_M])]
        hits[r], top[r], bottom[r] = len(distinct), distinct[0], distinct[-1]
    return {"origin": origin, "shape": tuple(shape), "cell_m": float(cell_m), "hits": hits.reshape(shape),
            "top": top.reshape(shape), "bottom": bottom.reshape(shape)}


def depression_ratio(top: np.ndarray, support_m: float) -> float | None:
    """How far a top surface sinks inside its rim, as a share of its height above the support.

    Read on the depth's top, not on the mesh: a tray or a bowl shows its rim above its floor to the sensor, while a
    Hunyuan plate that came out bowed (the lidar run's tofu, the board) is flat to the sensor and is exactly what the
    fit should mend. The rim is the columns within two of the footprint's edge, the inside the columns at least a
    quarter of the footprint's largest inner distance from it. A block, a board or a fruit gives about 0 or less (the
    sensor's rim mixes in the lower surface around); a tray or a bowl gives most of its height. None when the footprint
    has no inside.
    """
    inside = np.isfinite(top)
    if not inside.any():
        return None
    footprint = ndimage.binary_fill_holes(ndimage.binary_closing(inside, iterations=1))
    dist = ndimage.distance_transform_edt(np.pad(footprint, 1))[1:-1, 1:-1]
    rim = inside & (dist <= 2)
    core = inside & (dist >= max(3.0, 0.25 * dist.max()))
    if not rim.any() or not core.any():
        return None
    height = np.percentile(top[inside], 95) - support_m
    if height <= 0:
        return None
    return float((np.percentile(top[rim], 90) - np.median(top[core])) / height)


#: Kernel width (m) of the height density `lowest_mode` reads, and the share of the highest peak a lower peak needs.
_MODE_BANDWIDTH_M = 0.0015
_MODE_MIN_SHARE = 0.25


def lowest_mode(heights) -> float | None:
    """The lowest well-populated height among `heights`: the lowest peak of their smoothed density that reaches a
    quarter of the highest one.

    For an object resting on a neighbour (the tofu in its tray), the band around it holds the surface it rests on and,
    above it, the neighbour's walls and whatever else stands there; the support is the lowest of these surfaces, and a
    density peak does not depend on where bin edges fall (the 5 mm bins of `size_check.dominant_floor` picked the
    tray floor at 12 mm or a 18 mm shelf of the same band depending on the frame the heights were counted in). None when
    there are no heights.
    """
    h = np.asarray(heights, dtype=np.float64).ravel()
    h = h[np.isfinite(h)]
    if len(h) == 0:
        return None
    grid = np.arange(h.min() - 3 * _MODE_BANDWIDTH_M, h.max() + 3 * _MODE_BANDWIDTH_M, _MODE_BANDWIDTH_M / 3)
    density = np.zeros_like(grid)
    for chunk in np.array_split(h, max(1, len(h) // 2000)):
        density += np.exp(-0.5 * ((grid[:, None] - chunk[None]) / _MODE_BANDWIDTH_M) ** 2).sum(axis=1)
    peaks = np.flatnonzero((density[1:-1] >= density[:-2]) & (density[1:-1] > density[2:])) + 1
    if len(peaks) == 0:
        return float(np.median(h))
    strong = peaks[density[peaks] >= _MODE_MIN_SHARE * density[peaks].max()]
    return float(grid[strong.min()])


def depth_top(points, up, columns: dict[str, Any], erode_cells: int) -> np.ndarray:
    """The depth's top per column (median height of the points in it), NaN where none, its own rim dropped."""
    up = _unit(up)
    abh = _frame(np.asarray(points, dtype=np.float64).reshape(-1, 3), up)
    idx = np.floor((abh[:, :2] - columns["origin"]) / columns["cell_m"]).astype(np.int64)
    shape = columns["shape"]
    ok = (idx[:, 0] >= 0) & (idx[:, 0] < shape[0]) & (idx[:, 1] >= 0) & (idx[:, 1] < shape[1])
    top = np.full(shape, np.nan)
    if not ok.any():
        return top
    flat = idx[ok, 0] * shape[1] + idx[ok, 1]
    h = abh[ok, 2]
    order = np.argsort(flat, kind="stable")
    flat, h = flat[order], h[order]
    cells, starts = np.unique(flat, return_index=True)
    for cell, chunk in zip(cells, np.split(h, starts[1:])):
        top.flat[cell] = np.median(chunk)
    if erode_cells > 0:
        # The points are about as far apart as a column is wide, so the occupied columns have pin holes; erode the
        # footprint they cover (holes closed), not the occupied columns themselves, or the holes eat the inside away.
        footprint = ndimage.binary_fill_holes(ndimage.binary_closing(np.isfinite(top), iterations=1))
        keep = ndimage.binary_erosion(footprint, iterations=erode_cells, border_value=0)
        top[~keep] = np.nan
    return top


def _fill_nearest(values: np.ndarray) -> np.ndarray:
    """Every NaN cell takes the value of the nearest finite one."""
    valid = np.isfinite(values)
    if valid.all():
        return values
    _, (ia, ib) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return values[ia, ib]


def _sample(grid: np.ndarray, columns: dict[str, Any], ab: np.ndarray) -> np.ndarray:
    """Bilinear values of a filled grid at points (a, b)."""
    coords = (ab - columns["origin"]) / columns["cell_m"] - 0.5
    return ndimage.map_coordinates(grid, [coords[:, 0], coords[:, 1]], order=1, mode="nearest")


def column_fit(vertices, faces, depth_points, up, support_m: float, settings: dict[str, Any], under_points=None
               ) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Lay each vertex between the support and the depth's top, keeping its place between the mesh's bottom and top.

    Args:
        vertices: (N, 3) placed mesh vertices (the frame of `depth_points`).
        faces: (F, 3) triangles.
        depth_points: (M, 3) the object's depth points.
        up: the gravity direction in that frame.
        support_m: the height (along `up`) of the surface the object stands on: the size check's floor.
        settings: `resolve_settings` output.
        under_points: (J, 3) the depth stage 5 left inside the object's mask once the object was removed (what was
            painted in its place); None skips the support agreement test.

    Returns:
        (new vertices or None when refused, record). The record always holds the numbers the decision used.
    """
    up = _unit(up)
    v = np.asarray(vertices, dtype=np.float64)
    columns = mesh_columns(v, faces, up, settings["cell_m"])
    crossed = columns["hits"] > 0
    record: dict[str, Any] = {"applied": False, "support_m": float(support_m), "cell_m": settings["cell_m"],
                              "mesh_columns": int(crossed.sum())}
    if not crossed.any():
        return None, {**record, "skipped": "no vertical line crosses the mesh"}
    record["two_hit_share"] = float((columns["hits"][crossed] == 2).mean())
    if record["two_hit_share"] < settings["min_two_hit_share"]:
        return None, {**record, "skipped": f"not a height field: {record['two_hit_share']:.2f} of the columns are "
                                           f"crossed exactly twice, need {settings['min_two_hit_share']}"}
    two = columns["hits"] == 2
    # A container sinks inside its rim both in the mesh and to the sensor. Either alone is not enough: Hunyuan bowed the
    # lidar run's tofu into a curled plate (mesh 0.23) that the sensor sees flat (0.03), and the sensor's own rim mixes
    # in the lower surface around an object.
    record["mesh_depression"] = depression_ratio(np.where(two, columns["top"], np.nan), float(np.nanmin(columns["bottom"])))
    record["depth_depression"] = depression_ratio(depth_top(depth_points, up, columns, 0), support_m)
    if all(d is not None and d > settings["max_depression"] for d in (record["mesh_depression"], record["depth_depression"])):
        return None, {**record, "skipped": f"a container: its top sinks inside its rim by {record['mesh_depression']:.2f} "
                                           f"(mesh) and {record['depth_depression']:.2f} (depth) of its height, "
                                           f"both over {settings['max_depression']}"}
    thickness = np.where(two, columns["top"] - columns["bottom"], np.nan)
    footprint = ndimage.binary_fill_holes(crossed)
    inner = two & (ndimage.distance_transform_edt(np.pad(footprint, 1))[1:-1, 1:-1] >= 3)
    if inner.any():
        record["thickness_uniformity"] = float(np.percentile(thickness[inner], 25) / np.percentile(thickness[inner], 75))
        if record["thickness_uniformity"] < settings["min_thickness_uniformity"]:
            return None, {**record, "skipped": f"not slab-like: its inner columns' thickness quartiles are "
                                               f"{record['thickness_uniformity']:.2f} apart, need "
                                               f"{settings['min_thickness_uniformity']}"}
    top = depth_top(depth_points, up, columns, settings["top_erode_cells"])
    top[~two] = np.nan
    record["depth_columns"] = int(np.isfinite(top).sum())
    if record["depth_columns"] < settings["min_top_cells"]:
        return None, {**record, "skipped": f"only {record['depth_columns']} columns keep a depth top, "
                                           f"need {settings['min_top_cells']}"}
    # The mesh's own thickness per column against the depth's height over the support: a mesh placed by its top can
    # float, and its top alone would then agree with the depth while its volume does not (the tofu: top 22.5 mm over
    # the tray floor, thickness about 11 mm).
    both = two & np.isfinite(top)
    mesh_mean = float(np.mean(columns["top"][both] - columns["bottom"][both]))
    depth_mean = float(np.mean(top[both] - support_m))
    record.update(mesh_mean_thickness_m=mesh_mean, depth_mean_height_m=depth_mean,
                  mesh_mean_top_over_support_m=float(np.mean(columns["top"][both] - support_m)))
    if depth_mean <= 0:
        return None, {**record, "skipped": "the depth shows no height above the support"}
    if under_points is not None and len(under_points):
        record["support_under_m"] = float(np.median(np.asarray(under_points, dtype=np.float64).reshape(-1, 3) @ up))
        record["support_disagreement"] = float(abs(record["support_under_m"] - support_m) / depth_mean)
        if record["support_disagreement"] > settings["max_support_disagreement"]:
            return None, {**record, "skipped": f"the support is not known: around the object it reads "
                                               f"{support_m:.4f} m, under it {record['support_under_m']:.4f} m, "
                                               f"{record['support_disagreement']:.2f} of the depth's height apart"}
    record["relative_difference"] = float(abs(mesh_mean - depth_mean) / depth_mean)
    if record["relative_difference"] < settings["threshold"]:
        return None, {**record, "skipped": f"the depth agrees: mean column heights differ by "
                                           f"{record['relative_difference']:.2f}, under {settings['threshold']}"}
    mesh_top = _fill_nearest(np.where(two, columns["top"], np.nan))
    mesh_bottom = _fill_nearest(np.where(two, columns["bottom"], np.nan))
    target_top = ndimage.gaussian_filter(_fill_nearest(top), sigma=1.0)
    abh = _frame(v, up)
    t, b, h = (_sample(g, columns, abh[:, :2]) for g in (mesh_top, mesh_bottom, target_top))
    span = t - b
    frac = np.where(span > 1e-4, (abh[:, 2] - b) / np.where(span > 1e-4, span, 1.0), (abh[:, 2] >= (t + b) / 2) * 1.0)
    frac = np.clip(frac, 0.0, 1.0)
    new_h = support_m + frac * np.maximum(h - support_m, 0.0)
    out = v + (new_h - abh[:, 2])[:, None] * up
    record["applied"] = True
    return out, record
