# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Align a background splat to a scene by the appearance of the counter top.

A background video (phone, DA3 poses) and a scene (ARKit LiDAR frames) are separate captures, so they share no frame:
the 6th stage cannot chain them through camera centres the way its `single` and `umeyama` modes do. This module aligns
the two point clouds instead, from what the counter top looks like:

1. The largest plane of the seed cloud is the counter; the normal points towards the cameras and z = 0 on the plane.
2. Within a band of the plane, a top-view luminance map is built for both clouds.
3. For every yaw (1 degree) an FFT correlation finds a rough shift, which is then searched cell by cell and scored by
   a masked normalised cross-correlation (NCC) over the cells both clouds observe.
4. The best yaw and shift are refined on a 0.5 cm grid in 0.1 degree steps.

It is a port of research 1's prototype `align_v4.py`. The numerical conventions are the prototype's, except the points
listed in the brief: a seeded numpy RANSAC instead of open3d (which has no seed), `np.bincount` instead of `np.add.at`,
the fine stage on a 0.5 cm grid, a finite-depth filter, and the peak, overlap and scale records. Those are the only
differences; anything else here should match the prototype's behaviour exactly.

The functions in this module take arrays and return dictionaries. Only `da3_world_points` and
`scene_points_from_background_mesh` read files, and nothing here writes one; the 6th stage writes the record.
"""

import math
import numbers
from collections.abc import Mapping

import numpy as np
import trimesh
from omegaconf import OmegaConf
from PIL import Image

from simfoundry.pipeline import depth_mesh
from simfoundry.pipeline.upright_from_fit import rotation_between

SCHEMA = "bg_align/1"

# Rec. 601 luma. The prototype's choice; the colour statistics only need a single channel that tracks brightness.
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])

# RANSAC on the seed cloud is O(iterations x points). A fixed stride keeps it deterministic and bounded.
MAX_RANSAC_POINTS = 200_000

# The coarse search steps two cells at a time, then refines around the best with a one-cell step. Fixed, as in the
# prototype: a setting would invite tuning the search to one scene.
COARSE_STEP_CELLS = 2
REFINE_RADIUS_CELLS = 2

# The scale check is a record only. Nine scales over +/-4 percent, in integer steps so no float accumulates.
SCALE_STEP = 0.01
SCALE_HALF_STEPS = 4

# Keys the 6th stage reads itself. They name files, so `resolve_settings` ignores them rather than validating numbers.
PATH_KEYS = ("seed_ply", "scene_mesh", "scene_texture")

DEFAULT_SETTINGS = {
    "cell_m": 0.01,
    "fine_cell_m": 0.005,
    "band_m": 0.03,
    "margin_m": 1.2,
    "yaw_step_deg": 1.0,
    "coarse_search_cells": 10,
    "fine_yaw_range_deg": 1.0,
    "fine_yaw_step_deg": 0.1,
    "fine_shift_cells": 4,
    "pixel_stride": 2,
    "plane_threshold_m": 0.01,
    "plane_iterations": 2000,
    "plane_seed": 0,
    "min_plane_share": 0.05,
    "min_overlap": 0.5,
    "min_ncc": 0.5,
    "min_peak_gap": 0.05,
    "peak_window_deg": 10.0,
    "n_peaks_recorded": 8,
    "scale_warn": 0.01,
    "scale_min_gain": 0.02,
}

_INT_KEYS = frozenset({
    "coarse_search_cells", "fine_shift_cells", "pixel_stride", "plane_iterations", "plane_seed", "n_peaks_recorded",
})


def _above(low, inclusive=False):
    return (lambda v: v >= low) if inclusive else (lambda v: v > low)


def _between(low, high):
    return lambda v: low <= v <= high


# key -> (predicate, text for the error message). The text is what the message says the value must be.
_RULES = {
    "cell_m": (_above(0.0), "a positive number"),
    "fine_cell_m": (_above(0.0), "a positive number"),
    "band_m": (_above(0.0), "a positive number"),
    "margin_m": (_above(0.0, inclusive=True), "a number >= 0"),
    "yaw_step_deg": (lambda v: 0.0 < v <= 360.0, "a number in (0, 360]"),
    "coarse_search_cells": (_above(1, inclusive=True), "an integer >= 1"),
    "fine_yaw_range_deg": (_above(0.0, inclusive=True), "a number >= 0"),
    "fine_yaw_step_deg": (_above(0.0), "a positive number"),
    "fine_shift_cells": (_above(0, inclusive=True), "an integer >= 0"),
    "pixel_stride": (_above(1, inclusive=True), "an integer >= 1"),
    "plane_threshold_m": (_above(0.0), "a positive number"),
    "plane_iterations": (_above(1, inclusive=True), "an integer >= 1"),
    "plane_seed": (_above(0, inclusive=True), "an integer >= 0"),
    "min_plane_share": (lambda v: 0.0 < v <= 1.0, "a number in (0, 1]"),
    "min_overlap": (_between(0.0, 1.0), "a number in [0, 1]"),
    "min_ncc": (_between(-1.0, 1.0), "a number in [-1, 1]"),
    "min_peak_gap": (_above(0.0, inclusive=True), "a number >= 0"),
    "peak_window_deg": (_between(0.0, 180.0), "a number in [0, 180]"),
    "n_peaks_recorded": (_above(1, inclusive=True), "an integer >= 1"),
    "scale_warn": (_above(0.0), "a positive number"),
    "scale_min_gain": (_above(0.0, inclusive=True), "a number >= 0"),
}


def _as_mapping(section):
    """The settings as a plain dict; accepts a dict, a Hydra/OmegaConf node or None."""
    if section is None:
        return {}
    if OmegaConf.is_config(section):
        return dict(OmegaConf.to_container(section, resolve=True))
    if isinstance(section, Mapping):
        return dict(section)
    raise TypeError(f"s6_bridge.appearance must be a mapping or null, got {type(section).__name__}")


def _validated(key, value):
    predicate, text = _RULES[key]
    message = f"s6_bridge.appearance.{key} must be {text}, got {value!r}"
    # bool is an int in Python; a yes/no switch must not pass as a count or a threshold.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(message)
    number = float(value)
    if not math.isfinite(number) or not predicate(number):
        raise ValueError(message)
    if key in _INT_KEYS:
        if not number.is_integer():
            raise ValueError(message)
        return int(number)
    return number


def resolve_settings(section):
    """The appearance settings with defaults filled in, validated and typed.

    The path keys in `PATH_KEYS` are ignored here: the 6th stage resolves those itself. Any other key that is not in
    `DEFAULT_SETTINGS` is an error, so a misspelt key cannot fall back silently to a default.

    Args:
        section: a dict, an OmegaConf node, or None (all defaults).

    Returns:
        dict: every key of `DEFAULT_SETTINGS`, as Python ints for the count keys and floats otherwise.

    Raises:
        ValueError: for an unknown key, a value that is not a finite number in its range, or a bool.
        TypeError: when `section` is not a mapping.
    """
    raw = _as_mapping(section)
    unknown = sorted(set(raw) - set(DEFAULT_SETTINGS) - set(PATH_KEYS))
    if unknown:
        raise ValueError(f"s6_bridge.appearance has unknown keys {unknown}; the known keys are {sorted(DEFAULT_SETTINGS)}")
    return {key: _validated(key, raw.get(key, default)) for key, default in DEFAULT_SETTINGS.items()}


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _rotation2(yaw_deg):
    """The 2x2 rotation about z by `yaw_deg`, applied to row vectors as `points @ rotation.T`."""
    angle = np.radians(yaw_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin], [sin, cos]])


def _apply(transform, points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ transform[:3, :3].T + transform[:3, 3]


def fit_gravity_plane(points, camera_centres, settings):
    """The largest plane of a seed cloud, with its normal turned towards the cameras.

    RANSAC is seeded (`plane_seed`), so the same input gives the same plane. The prototype used open3d's
    `segment_plane`, which has no seed. Clouds above `MAX_RANSAC_POINTS` are sampled with a fixed stride for the
    RANSAC only; the inlier count and share are then taken over all points.

    Args:
        points: (N, 3) seed points in metres.
        camera_centres: (F, 3) camera centres of the source frames, on the side the counter is seen from.
        settings: as for `resolve_settings`.

    Returns:
        dict or None: `normal` (unit, 3 floats), `offset` (so that normal . p + offset = 0 on the plane), `inliers`,
        `share` (inliers over all finite points), `rms_m` (of the inliers' distances) and `n_points`. None when there
        are fewer than three points or fewer than `min_plane_share` of them lie on the best plane.
    """
    cfg = resolve_settings(settings)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    n_points = len(pts)
    if n_points < 3:
        return None
    cams = np.asarray(camera_centres, dtype=np.float64).reshape(-1, 3)
    if len(cams) == 0:
        raise ValueError("fit_gravity_plane needs at least one camera centre to choose the normal's side")
    stride = int(math.ceil(n_points / MAX_RANSAC_POINTS))
    fit_pts = pts[::stride] if stride > 1 else pts
    threshold = cfg["plane_threshold_m"]

    rng = np.random.default_rng(cfg["plane_seed"])
    triples = rng.integers(0, len(fit_pts), size=(cfg["plane_iterations"], 3))
    best_count, best_normal, best_offset = -1, None, None
    for i, j, k in triples:
        if len({i, j, k}) < 3:
            continue
        normal = np.cross(fit_pts[j] - fit_pts[i], fit_pts[k] - fit_pts[i])
        length = np.linalg.norm(normal)
        if length < 1e-12:  # collinear: no plane through these three points
            continue
        normal = normal / length
        offset = -normal @ fit_pts[i]
        count = int(np.count_nonzero(np.abs(fit_pts @ normal + offset) < threshold))
        # Strictly greater: on a tie the earlier triple wins, which keeps the result reproducible.
        if count > best_count:
            best_count, best_normal, best_offset = count, normal, offset
    if best_normal is None:
        return None

    # Refit by SVD on the best plane's inliers, so the plane is not left at the three-point estimate.
    inliers = fit_pts[np.abs(fit_pts @ best_normal + best_offset) < threshold]
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vt[-1]
    offset = -normal @ centroid
    if np.median(cams @ normal + offset) < 0:
        normal, offset = -normal, -offset

    distance = np.abs(pts @ normal + offset)
    on_plane = distance < threshold
    count = int(on_plane.sum())
    share = count / n_points
    if count == 0 or share < cfg["min_plane_share"]:
        return None
    return {
        "normal": [float(v) for v in normal],
        "offset": float(offset),
        "inliers": count,
        "share": float(share),
        "rms_m": float(np.sqrt(np.mean(distance[on_plane] ** 2))),
        "n_points": n_points,
    }


def gravity_transform(normal, offset):
    """The rigid transform that puts a plane on z = 0 with its normal along +z.

    Args:
        normal: the plane normal (any length; it is normalised).
        offset: the plane offset for that normal (normal . p + offset = 0 on the plane).

    Returns:
        np.ndarray: (4, 4) rigid transform (rotation determinant +1).
    """
    length = np.linalg.norm(np.asarray(normal, dtype=np.float64))
    unit = _unit(normal)
    distance = float(offset) / length
    rotation = rotation_between(unit, [0.0, 0.0, 1.0])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    # The plane's point nearest the origin, -distance * unit, must land on the origin: translate by distance * R unit.
    transform[:3, 3] = distance * (rotation @ unit)
    return transform


def luminance(colours):
    """Brightness of RGB colours, as a weighted sum of the channels.

    Args:
        colours: (N, 3) colours, 0-255 floats or uint8.

    Returns:
        np.ndarray: (N,) brightness on the same scale.
    """
    return np.asarray(colours, dtype=np.float64).reshape(-1, 3) @ LUMA_WEIGHTS


def top_view_map(xy, values, origin, shape, cell_m):
    """Mean of `values` per square cell of a top-down grid.

    Points outside the grid, and points with a non-finite coordinate or value, are dropped. A cell with no point is
    NaN. The mean is taken with `np.bincount` (the prototype's `np.add.at` is too slow at 360 yaws).

    Args:
        xy: (N, 2) positions in metres.
        values: (N,) values per point.
        origin: (2,) the grid's lower-left corner.
        shape: (n0, n1) cells along x and y.
        cell_m: cell size in metres.

    Returns:
        np.ndarray: (n0, n1) float64, NaN where the cell is empty.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n0, n1 = int(shape[0]), int(shape[1])
    finite = np.all(np.isfinite(xy), axis=1) & np.isfinite(values)
    xy, values = xy[finite], values[finite]
    ix = np.floor((xy[:, 0] - origin[0]) / cell_m).astype(np.int64)
    iy = np.floor((xy[:, 1] - origin[1]) / cell_m).astype(np.int64)
    inside = (ix >= 0) & (ix < n0) & (iy >= 0) & (iy < n1)
    flat = ix[inside] * n1 + iy[inside]
    sums = np.bincount(flat, weights=values[inside], minlength=n0 * n1)
    counts = np.bincount(flat, minlength=n0 * n1)
    out = np.full(n0 * n1, np.nan)
    filled = counts > 0
    out[filled] = sums[filled] / counts[filled]
    return out.reshape(n0, n1)


def masked_ncc(target_map, source_map, di, dj, min_overlap_cells):
    """Normalised cross-correlation of two maps, shifted by (di, dj) cells, over the cells both observe.

    The slices follow the prototype: target[a, b] pairs with source[a2, b2], where a2 is a shifted back by di. Each side
    is normalised over the overlap by (value - mean) / (std + 1e-9), and the score is the mean of the product.

    Args:
        target_map: (n0, n1) map, NaN where empty.
        source_map: (n0, n1) map, NaN where empty.
        di: shift along the first axis, in cells.
        dj: shift along the second axis, in cells.
        min_overlap_cells: fewest overlapping cells that still give a score.

    Returns:
        tuple or None: (score, overlap_cells), or None when the overlap is below `min_overlap_cells`, or when the shift is
        as large as the map (nothing to compare).
    """
    target = np.asarray(target_map, dtype=np.float64)
    source = np.asarray(source_map, dtype=np.float64)
    n0, n1 = target.shape
    if abs(di) >= n0 or abs(dj) >= n1:
        return None
    a = slice(max(di, 0), n0 + min(di, 0))
    b = slice(max(dj, 0), n1 + min(dj, 0))
    a2 = slice(max(-di, 0), n0 + min(-di, 0))
    b2 = slice(max(-dj, 0), n1 + min(-dj, 0))
    t, s = target[a, b], source[a2, b2]
    both = np.isfinite(t) & np.isfinite(s)
    overlap = int(both.sum())
    if overlap == 0 or overlap < min_overlap_cells:
        return None
    x = t[both]
    y = s[both]
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    return float((x * y).mean()), overlap


def _grid(xy, cell_m, margin_m):
    """The grid that covers the points plus a margin on every side: (origin, (n0, n1))."""
    origin = np.array([xy[:, 0].min() - margin_m, xy[:, 1].min() - margin_m])
    extent = np.ptp(xy, axis=0) + 2.0 * margin_m
    return origin, (int(math.ceil(extent[0] / cell_m)), int(math.ceil(extent[1] / cell_m)))


def _centred(grid_map):
    """The map with its mean removed and empty cells at zero: the FFT correlation's input."""
    return np.where(np.isfinite(grid_map), grid_map - np.nanmean(grid_map), 0.0)


def _signed_shift(index, size):
    """The shift that an FFT peak at `index` stands for: the correlation is circular, so past half the grid it is negative."""
    return index if index < size // 2 else index - size


def _circular_gap(a_deg, b_deg):
    gap = abs(a_deg - b_deg) % 360.0
    return min(gap, 360.0 - gap)


def _peak_yaws(yaw_scores, window_deg):
    """The yaws whose score is at least that of every yaw within `window_deg` on either side (around the circle)."""
    peaks = []
    for yaw, score in yaw_scores.items():
        rivals = [other for other in yaw_scores if other != yaw and _circular_gap(other, yaw) <= window_deg]
        if all(yaw_scores[other] <= score for other in rivals):
            peaks.append(yaw)
    return peaks


def second_peak_score(yaw_scores, best_yaw, window_deg):
    """The score of the highest local maximum other than the best yaw's own.

    A local maximum is a yaw that no other yaw within `window_deg` beats. The shoulder of a broad peak is not one, so a
    single wide peak gives -1 (the largest possible gap), and a second peak far from the best gives its score. A yaw
    with no score is simply absent from `yaw_scores` and takes no part.

    Args:
        yaw_scores: {yaw_deg: score} for the yaws that have a score.
        best_yaw: the best yaw, a key of `yaw_scores`.
        window_deg: the half-width of the window, in degrees (circular).

    Returns:
        float: the second peak's score, or -1.0 when the best is the only peak.
    """
    rivals = [yaw_scores[yaw] for yaw in _peak_yaws(yaw_scores, window_deg) if _circular_gap(yaw, best_yaw) > 0]
    return float(max(rivals)) if rivals else -1.0


def _unsearched():
    return {
        "yaw_deg": None,
        "shift_xy": None,
        "ncc": None,
        "overlap": None,
        "yaw_curve": [],
        "peaks": [],
        "second_peak_ncc": None,
        "peak_gap": None,
        "scale_best": None,
        "scale_curve": [],
        "warnings": [],
    }


def _best_on_grid(target_map, source_map, centre, radius, step, min_overlap_cells):
    """The best (score, di, dj, overlap) on a grid of shifts around `centre`, first best on ties."""
    best = None
    for di in range(centre[0] - radius, centre[0] + radius + 1, step):
        for dj in range(centre[1] - radius, centre[1] + radius + 1, step):
            hit = masked_ncc(target_map, source_map, di, dj, min_overlap_cells)
            if hit is not None and (best is None or hit[0] > best[0]):
                best = (hit[0], di, dj, hit[1])
    return best


def search_alignment(target_xy, target_lum, source_xy, source_lum, settings):
    """The yaw and shift that best align two gravity-aligned top views, with the peak and scale records.

    The inputs are the points already within the band (z = 0 plane, up = +z), as xy and brightness. The coarse search
    covers every yaw at `yaw_step_deg` on a `cell_m` grid; the fine search covers +/- `fine_yaw_range_deg` at
    `fine_yaw_step_deg` on a `fine_cell_m` grid. The scale check then re-scores the final alignment at nine scales,
    for the record only: the transform itself always keeps scale 1.

    Args:
        target_xy: (N, 2) scene points within the band.
        target_lum: (N,) their brightness.
        source_xy: (M, 2) source points, after the gravity transform and within the band.
        source_lum: (M,) their brightness.
        settings: as for `resolve_settings`.

    Returns:
        dict: `yaw_deg` and `shift_xy` (the final rotation and translation of the source xy; None when nothing matched),
        `ncc` and `overlap` (fraction of the target's cells, None when nothing matched), `yaw_curve` ([[yaw, score], ...]
        for the yaws with a score), `peaks` (the best local maxima, highest first), `second_peak_ncc` and `peak_gap`
        (best coarse score minus the second peak's, from the same curve), `scale_best` and `scale_curve` ([[scale, ncc],
        ...]), and `warnings`.
    """
    cfg = resolve_settings(settings)
    target = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    source = np.asarray(source_xy, dtype=np.float64).reshape(-1, 2)
    target_values = np.asarray(target_lum, dtype=np.float64).reshape(-1)
    source_values = np.asarray(source_lum, dtype=np.float64).reshape(-1)
    result = _unsearched()
    if len(target) == 0 or len(source) == 0:
        return result

    cell = cfg["cell_m"]
    origin, shape = _grid(target, cell, cfg["margin_m"])
    target_map = top_view_map(target, target_values, origin, shape, cell)
    n_target = int(np.isfinite(target_map).sum())
    if n_target == 0:
        return result
    min_cells = cfg["min_overlap"] * n_target
    target_fft = np.fft.rfft2(_centred(target_map))
    centre = np.median(target, axis=0)

    # Coarse: every yaw. The FFT gives a rough shift, the grid around it gives the score.
    yaw_scores, yaw_shifts, yaw_overlap = {}, {}, {}
    for step in range(int(round(360.0 / cfg["yaw_step_deg"]))):
        yaw = step * cfg["yaw_step_deg"]
        rotated = source @ _rotation2(yaw).T
        base = centre - np.median(rotated, axis=0)
        source_map = top_view_map(rotated + base, source_values, origin, shape, cell)
        if not np.isfinite(source_map).any():
            continue
        corr = np.fft.irfft2(target_fft * np.conj(np.fft.rfft2(_centred(source_map))), s=shape)
        i, j = np.unravel_index(np.argmax(corr), corr.shape)
        ci = _signed_shift(i, shape[0])
        cj = _signed_shift(j, shape[1])
        best = _best_on_grid(target_map, source_map, (ci, cj), cfg["coarse_search_cells"], COARSE_STEP_CELLS, min_cells)
        if best is None:
            continue
        score, bi, bj, overlap = best
        for di in range(bi - REFINE_RADIUS_CELLS, bi + REFINE_RADIUS_CELLS + 1):
            for dj in range(bj - REFINE_RADIUS_CELLS, bj + REFINE_RADIUS_CELLS + 1):
                hit = masked_ncc(target_map, source_map, di, dj, min_cells)
                if hit is not None and hit[0] > score:
                    score, bi, bj, overlap = hit[0], di, dj, hit[1]
        yaw_scores[yaw] = score
        yaw_overlap[yaw] = overlap
        yaw_shifts[yaw] = base + np.array([bi, bj]) * cell
    if not yaw_scores:
        return result

    result["yaw_curve"] = [[float(yaw), float(yaw_scores[yaw])] for yaw in sorted(yaw_scores)]
    best_yaw = min(yaw_scores, key=lambda yaw: (-yaw_scores[yaw], yaw))
    coarse_best = yaw_scores[best_yaw]
    second = second_peak_score(yaw_scores, best_yaw, cfg["peak_window_deg"])
    ranked = sorted(_peak_yaws(yaw_scores, cfg["peak_window_deg"]), key=lambda yaw: (-yaw_scores[yaw], yaw))
    result["peaks"] = [
        {
            "yaw_deg": float(yaw),
            "ncc": float(yaw_scores[yaw]),
            "shift_m": [float(yaw_shifts[yaw][0]), float(yaw_shifts[yaw][1]), 0.0],
        }
        for yaw in ranked[: cfg["n_peaks_recorded"]]
    ]
    result["second_peak_ncc"] = second
    result["peak_gap"] = float(coarse_best - second)

    # Fine: a 0.5 cm grid, 0.1 degree yaw steps around the coarse best, shifts within a few cells of the coarse shift.
    fine_cell = cfg["fine_cell_m"]
    _, fine_shape = _grid(target, fine_cell, cfg["margin_m"])
    fine_target = top_view_map(target, target_values, origin, fine_shape, fine_cell)
    n_fine = int(np.isfinite(fine_target).sum())
    fine_min = cfg["min_overlap"] * n_fine
    window = cfg["fine_shift_cells"]
    k_max = int(round(cfg["fine_yaw_range_deg"] / cfg["fine_yaw_step_deg"]))
    coarse_shift = yaw_shifts[best_yaw]
    fine = None
    for k in range(-k_max, k_max + 1):
        # An integer k keeps the yaws exact; adding 0.1 repeatedly would drift.
        yaw = best_yaw + k * cfg["fine_yaw_step_deg"]
        rotated = source @ _rotation2(yaw).T
        base = centre - np.median(rotated, axis=0)
        source_map = top_view_map(rotated + base, source_values, origin, fine_shape, fine_cell)
        if not np.isfinite(source_map).any():
            continue
        centre_cells = np.rint((coarse_shift - base) / fine_cell).astype(int)
        for di in range(centre_cells[0] - window, centre_cells[0] + window + 1):
            for dj in range(centre_cells[1] - window, centre_cells[1] + window + 1):
                hit = masked_ncc(fine_target, source_map, di, dj, fine_min)
                if hit is not None and (fine is None or hit[0] > fine["ncc"]):
                    fine = {
                        "ncc": hit[0],
                        "overlap": hit[1],
                        "yaw": yaw,
                        "shift": base + np.array([di, dj]) * fine_cell,
                    }
    if fine is not None:
        final_yaw, final_shift = fine["yaw"], fine["shift"]
        final_ncc, final_overlap = fine["ncc"], fine["overlap"] / n_fine
    else:
        # No candidate of the fine window clears the overlap bound, so the coarse alignment is what the record keeps.
        # Refusing here would report "no overlap", but the coarse search did find overlap; the score is what decides.
        final_yaw, final_shift = best_yaw, coarse_shift
        final_ncc, final_overlap = coarse_best, yaw_overlap[best_yaw] / n_target

    result["yaw_deg"] = float(final_yaw)
    result["shift_xy"] = [float(v) for v in final_shift]
    result["ncc"] = float(final_ncc)
    result["overlap"] = float(final_overlap)

    # Scale check, for the record. Scale the final alignment about its median and re-search the shift around it.
    moved = source @ _rotation2(final_yaw).T + final_shift
    pivot = np.median(moved, axis=0)
    scale_curve = []
    for step in range(-SCALE_HALF_STEPS, SCALE_HALF_STEPS + 1):
        scale = 1.0 + step * SCALE_STEP
        scaled_map = top_view_map(pivot + scale * (moved - pivot), source_values, origin, fine_shape, fine_cell)
        hit = _best_on_grid(fine_target, scaled_map, (0, 0), window, 1, fine_min)
        scale_curve.append([scale, float(hit[0]) if hit is not None else -1.0])
    best_scale, best_scale_ncc = max(scale_curve, key=lambda point: point[1])
    ncc_at_one = dict(scale_curve)[1.0]
    result["scale_best"] = float(best_scale)
    result["scale_curve"] = scale_curve
    gain = best_scale_ncc - ncc_at_one
    if abs(best_scale - 1.0) > cfg["scale_warn"] and gain > cfg["scale_min_gain"]:
        result["warnings"].append(
            f"scale check: NCC is higher by {gain:.3f} at scale {best_scale:.2f} than at 1.00; "
            "the transform keeps scale 1 and this is recorded only"
        )
    return result


def _height_residual(target_xy, target_z, moved_xy, source_z, settings):
    """Median absolute height difference over the fine cells that both clouds fill, after the final alignment."""
    cfg = resolve_settings(settings)
    cell = cfg["fine_cell_m"]
    origin, shape = _grid(target_xy, cell, cfg["margin_m"])
    target_height = top_view_map(target_xy, target_z, origin, shape, cell)
    source_height = top_view_map(moved_xy, source_z, origin, shape, cell)
    both = np.isfinite(target_height) & np.isfinite(source_height)
    if not both.any():
        return None
    return float(np.median(np.abs(target_height[both] - source_height[both])))


def decide(record, settings):
    """Accept or refuse an alignment record, and list every reason it was refused.

    Args:
        record: the dict `align_clouds` builds (plane, ncc, overlap, peak_gap).
        settings: as for `resolve_settings`.

    Returns:
        tuple: (accepted, reasons). `reasons` holds any of `no_gravity_plane`, `no_overlap`, `low_ncc`, `ambiguous`.
    """
    cfg = resolve_settings(settings)
    if record.get("plane") is None:
        return False, ["no_gravity_plane"]
    reasons = []
    ncc, overlap, gap = record.get("ncc"), record.get("overlap"), record.get("peak_gap")
    if ncc is None or overlap is None or overlap < cfg["min_overlap"]:
        reasons.append("no_overlap")
    if ncc is not None and ncc < cfg["min_ncc"]:
        reasons.append("low_ncc")
    if gap is not None and gap < cfg["min_peak_gap"]:
        reasons.append("ambiguous")
    return not reasons, reasons


def align_clouds(target_points, target_colours, source_points, source_colours, source_camera_centres, settings, *,
                 seed_points):
    """Align the source (DA3) cloud to the target (scene) cloud, end to end, and return the alignment record.

    The gravity plane comes from `seed_points` only: the seed cloud is what the splat was trained from, and its largest
    plane is the counter. The brightness maps use every source point and colour (all frames), as the prototype does.

    Args:
        target_points: (N, 3) scene points, Z-up, the counter at z = 0.
        target_colours: (N, 3) scene colours, 0-255.
        source_points: (M, 3) DA3 world points of all frames.
        source_colours: (M, 3) DA3 pixel colours, 0-255.
        source_camera_centres: (F, 3) DA3 camera centres, used to choose the plane normal's side.
        settings: as for `resolve_settings`.
        seed_points: (K, 3) the seed cloud, keyword-only, used for the gravity plane only.

    Returns:
        dict: the record described in the brief (schema `bg_align/1`). `M_src_to_scene` is a 4x4 list when accepted and
        None otherwise. The keys `inputs`, `sidecar_written` and `stale_sidecar_present` are placeholders here; the 6th
        stage fills them in, because they describe files.
    """
    cfg = resolve_settings(settings)
    plane = fit_gravity_plane(seed_points, source_camera_centres, cfg)
    record = {
        "schema": SCHEMA,
        "accepted": False,
        "reasons": [],
        "warnings": [],
        "settings": cfg,
        "inputs": None,
        "plane": plane,
        "n_target_points": None,
        "n_source_points": None,
        "peaks": [],
        "yaw_curve": [],
        "second_peak_ncc": None,
        "peak_gap": None,
        "yaw_deg": None,
        "shift_m": None,
        "ncc": None,
        "overlap": None,
        "scale": 1.0,
        "scale_best": None,
        "scale_curve": [],
        "height_residual_m": None,
        "M_src_to_scene": None,
        "sidecar_written": False,
        "stale_sidecar_present": False,
    }
    if plane is not None:
        gravity = gravity_transform(plane["normal"], plane["offset"])
        target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
        target_rgb = np.asarray(target_colours, dtype=np.float64).reshape(-1, 3)
        source = _apply(gravity, source_points)
        source_rgb = np.asarray(source_colours, dtype=np.float64).reshape(-1, 3)
        band = cfg["band_m"]
        target_in = np.abs(target[:, 2]) < band
        source_in = np.abs(source[:, 2]) < band
        record["n_target_points"] = int(target_in.sum())
        record["n_source_points"] = int(source_in.sum())
        search = search_alignment(
            target[target_in, :2], luminance(target_rgb[target_in]),
            source[source_in, :2], luminance(source_rgb[source_in]),
            cfg,
        )
        record["warnings"] = list(search["warnings"])
        record["peaks"] = search["peaks"]
        record["yaw_curve"] = search["yaw_curve"]
        record["second_peak_ncc"] = search["second_peak_ncc"]
        record["peak_gap"] = search["peak_gap"]
        record["yaw_deg"] = search["yaw_deg"]
        record["ncc"] = search["ncc"]
        record["overlap"] = search["overlap"]
        record["scale_best"] = search["scale_best"]
        record["scale_curve"] = search["scale_curve"]
        if search["yaw_deg"] is not None:
            shift = np.asarray(search["shift_xy"], dtype=np.float64)
            record["shift_m"] = [float(shift[0]), float(shift[1]), 0.0]
            moved = source[source_in, :2] @ _rotation2(search["yaw_deg"]).T + shift
            record["height_residual_m"] = _height_residual(
                target[target_in, :2], target[target_in, 2], moved, source[source_in, 2], cfg,
            )
    record["accepted"], record["reasons"] = decide(record, cfg)
    if record["accepted"]:
        transform = np.eye(4)
        transform[:2, :2] = _rotation2(record["yaw_deg"])
        transform[:2, 3] = record["shift_m"][:2]
        record["M_src_to_scene"] = (transform @ gravity).tolist()
    return record


def da3_world_points(npz_path, stride=2):
    """World points, their colours and camera centres from a DA3 results file.

    Pixels are taken every `stride` rows and columns, and only where the depth is finite and positive: a pixel without a
    depth has no place in the world. The extrinsics are world-to-camera, as DA3 writes them.

    Args:
        npz_path: path to DA3's results.npz (extrinsics, intrinsics, depth, image).
        stride: pixel step in both image directions.

    Returns:
        tuple: (points (N, 3) world metres, colours (N, 3) 0-255, camera centres (F, 3)).
    """
    with np.load(npz_path) as data:
        extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
        depths = data["depth"]
        images = data["image"]
    points, colours = [], []
    for ext, intr, depth, image in zip(extrinsics, intrinsics, depths, images):
        rows, cols = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z = depth[rows, cols].astype(np.float64)
        keep = np.isfinite(z) & (z > 0)
        rows, cols, z = rows[keep], cols[keep], z[keep]
        cam = np.stack([(cols - intr[0, 2]) / intr[0, 0] * z, (rows - intr[1, 2]) / intr[1, 1] * z, z], axis=-1)
        points.append((cam - ext[:, 3]) @ ext[:, :3])
        colours.append(image[rows, cols].astype(np.float64))
    centres = -np.einsum("nji,nj->ni", extrinsics[:, :, :3], extrinsics[:, :, 3])
    if not points:
        return np.zeros((0, 3)), np.zeros((0, 3)), centres
    return np.concatenate(points), np.concatenate(colours), centres


def scene_points_from_background_mesh(glb_path, texture_path):
    """The scene's background mesh as Z-up points, with the colour of the texel under each vertex.

    The mesh is stored Y-up (the depth-mesh export), so its vertices go back through the inverse of
    `depth_mesh.WORLD_TO_GLTF`. The colour is the nearest texel, with v flipped as glTF has it.

    Args:
        glb_path: the background mesh (.glb), with UV coordinates.
        texture_path: the image the mesh's UV coordinates point into.

    Returns:
        tuple: (points (N, 3) Z-up metres, colours (N, 3) 0-255).

    Raises:
        ValueError: when the mesh has no UV coordinates.
    """
    mesh = trimesh.load(str(glb_path), force="mesh", process=False)
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        raise ValueError(f"background mesh has no UV coordinates, so its texture colours cannot be read: {glb_path}")
    back = np.linalg.inv(depth_mesh.WORLD_TO_GLTF)
    points = np.asarray(mesh.vertices, dtype=np.float64) @ back[:3, :3].T + back[:3, 3]
    texture = np.asarray(Image.open(texture_path).convert("RGB"))
    height, width = texture.shape[:2]
    uv = np.asarray(uv, dtype=np.float64)
    rows = np.clip(((1.0 - uv[:, 1]) * height).astype(int), 0, height - 1)
    cols = np.clip((uv[:, 0] * width).astype(int), 0, width - 1)
    return points, texture[rows, cols].astype(np.float64)
