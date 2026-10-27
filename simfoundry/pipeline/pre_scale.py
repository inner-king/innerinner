# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 8's pre-scale factor (`s8_pose.pre_scale_method`; `current` by default).

Stage 8 shrinks the mesh by `pre_scale_factor`, the ratio of two oriented-box diagonals, before CPD fits it. The current
method takes both boxes from Open3D's PCA-based `get_oriented_bounding_box`: of the source, and of a Poisson-disk sample
of the mesh. For a flat, nearly square object the PCA axes can turn from one sample to the next, so the diagonal moves
with the sample (the lidar run's tray moved by 17.8 %, the board by 11.0 %). The minimal-volume box of the mesh's
vertices does not depend on a sample, and moved by at most 0.4 % on the same meshes.

`obb_minimal` takes the minimal-volume boxes of the source and of the mesh's vertices. A box is trusted only when its
extent is finite and positive and its rotation is orthogonal (`R^T R = I` to 1e-6). A side whose box is not trusted, or
whose computation raises, takes the PCA diagonal that `current` would give that side (for the mesh side, the Poisson
sample's), and the fallback is recorded. Open3D's own PCA raises on a flat or a very small cloud too; there the numpy PCA
of the same points is used, and a cloud with no extent at all is an error, because no scale can be read from it.

Nothing else changes: the Poisson target is still sampled and scaled by the factor, and the voxel size stays on the PCA
source diagonal, so the two methods differ in the factor alone.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import open3d as o3d

PRE_SCALE_METHODS = ("current", "obb_minimal")
#: Largest entry of R^T R - I that a minimal box's rotation may have and still be trusted.
ORTHOGONAL_TOL = 1e-6


@dataclass(frozen=True)
class PreScale:
    """The factor, the two diagonals it is the ratio of, and the sides that fell back with the reason."""

    factor: float
    source_diag: float
    mesh_diag: float
    method: str
    fallback: dict[str, str] = field(default_factory=dict)


def validate_pre_scale_method(value) -> str:
    """The configured method, which must be exactly one of PRE_SCALE_METHODS.

    Raises:
        ValueError: for anything else, including another letter case or surrounding spaces.
    """
    if not isinstance(value, str) or value not in PRE_SCALE_METHODS:
        raise ValueError(f"s8_pose.pre_scale_method must be one of {', '.join(PRE_SCALE_METHODS)}; got {value!r}")
    return value


def pca_diagonal(pcd) -> float:
    """The diagonal of Open3D's PCA-based oriented box, as stage 8 has always taken it."""
    return np.linalg.norm(pcd.get_oriented_bounding_box().extent)


def _first_line(exc: Exception) -> str:
    """The first line of an Open3D error: its Qhull and Open3D messages run to many lines, and the record keeps one."""
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__


def minimal_diagonal(pcd) -> tuple[float | None, str | None]:
    """The diagonal of the minimal-volume oriented box, or None with the reason when the box cannot be trusted.

    Open3D's minimal box can come out with a non-orthogonal rotation (it warns "Invalid oriented bounding box" and the
    rotation is wrong), with a zero or non-finite extent, or it can raise on degenerate input (a flat cloud, fewer than
    four points, an empty cloud). None of those is used: the caller takes the PCA diagonal for that side instead.

    Args:
        pcd: an Open3D point cloud.

    Returns:
        (diagonal, None) for a trusted box, else (None, reason).
    """
    try:
        box = pcd.get_minimal_oriented_bounding_box()
    except Exception as exc:  # Open3D raises on degenerate input; any such case is a fallback, not a stop
        return None, f"minimal box raised: {_first_line(exc)}"
    extent = np.asarray(box.extent, dtype=np.float64)
    rotation = np.asarray(box.R, dtype=np.float64)
    if not (np.all(np.isfinite(extent)) and np.all(extent > 0.0)):
        return None, f"minimal box extent {extent.tolist()} is not finite and positive"
    if not np.all(np.isfinite(rotation)):
        return None, "minimal box rotation is not finite"
    deviation = float(np.abs(rotation.T @ rotation - np.eye(3)).max())
    if deviation > ORTHOGONAL_TOL:
        return None, f"minimal box rotation is not orthogonal (max |R^T R - I| = {deviation:.2e})"
    return float(np.linalg.norm(extent)), None


def numpy_pca_diagonal(points) -> float | None:
    """The diagonal of the box along the principal axes of `points`, computed in numpy.

    Used only where Open3D's own PCA cannot run. None when there is no extent to scale by (no points, or all points
    at one place).
    """
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        return None
    centred = x - x.mean(axis=0)
    _, axes = np.linalg.eigh(centred.T @ centred)
    diagonal = float(np.linalg.norm(np.ptp(centred @ axes, axis=0)))
    return diagonal if np.isfinite(diagonal) and diagonal > 0.0 else None


def pca_fallback_diagonal(pcd, side: str) -> tuple[float, str]:
    """The diagonal a side takes when its minimal box is not trusted: the PCA diagonal `current` would give it.

    Open3D's PCA box is used when it runs. Where it raises or gives no extent (a flat cloud, three points), the numpy PCA
    of the same points is used instead.

    Args:
        pcd: the point cloud of that side (for the mesh side, the Poisson sample that `current` takes its PCA box from).
        side: "source" or "mesh", for the error message.

    Returns:
        (diagonal, how): the diagonal and a short text of where it came from.

    Raises:
        ValueError: when the cloud has no extent at all, so no scale can be read from it.
    """
    try:
        diagonal = float(pca_diagonal(pcd))
        if np.isfinite(diagonal) and diagonal > 0.0:
            return diagonal, "the PCA box"
        note = f"the PCA box gave diagonal {diagonal}"
    except Exception as exc:  # Open3D's PCA raises on flat, very small and empty clouds as its minimal box does
        note = f"the PCA box raised: {_first_line(exc)}"
    numeric = numpy_pca_diagonal(np.asarray(pcd.points))
    if numeric is None:
        raise ValueError(f"the {side} cloud has no extent to scale by ({len(pcd.points)} point(s)); {note}")
    return numeric, f"the numpy PCA of the same points ({note})"


def select_pre_scale(method, source, mesh, target) -> PreScale:
    """The pre-scale factor of stage 8 under `method`, with the two diagonals it is the ratio of.

    Args:
        method: `current` or `obb_minimal` (validated here).
        source: the source point cloud (the real-world partial cloud; `obb_minimal` drops its non-finite points again).
        mesh: the raw mesh read from file (`o3d.geometry.TriangleMesh`); `obb_minimal` uses its vertices.
        target: the Poisson-disk sample of the mesh that CPD is fitted to; `current` takes its PCA diagonal.

    Returns:
        PreScale: the factor (source diagonal over mesh diagonal) and both diagonals. `fallback` names each side that
        took its PCA diagonal instead, with the reason; it is empty when no side fell back, and always empty for
        `current`, whose values are the ones stage 8 has always computed.
    """
    method = validate_pre_scale_method(method)
    if method == "current":
        source_diag = pca_diagonal(source)
        mesh_diag = pca_diagonal(target)
        return PreScale(factor=source_diag / mesh_diag, source_diag=source_diag, mesh_diag=mesh_diag, method=method)
    source = source.remove_non_finite_points()
    mesh_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(mesh.vertices))).remove_non_finite_points()
    fallback = {}
    source_diag, reason = minimal_diagonal(source)
    if reason is not None:
        source_diag, how = pca_fallback_diagonal(source, "source")
        fallback["source"] = f"{reason}; replaced by {how}"
    mesh_diag, reason = minimal_diagonal(mesh_points)
    if reason is not None:
        mesh_diag, how = pca_fallback_diagonal(target, "mesh")
        fallback["mesh"] = f"{reason}; replaced by {how}"
    return PreScale(factor=source_diag / mesh_diag, source_diag=source_diag, mesh_diag=mesh_diag, method=method,
                    fallback=fallback)


def pre_scale_record(method, fallback) -> dict[str, Any]:
    """The info keys the method adds to stage 8's output. `current` adds none, so its files are unchanged.

    Args:
        method: the configured `s8_pose.pre_scale_method`.
        fallback: the `fallback` dict of the chosen PreScale.

    Returns:
        dict: `pre_scale_method`, plus `pre_scale_fallback` (each side that fell back, and why) when a side fell back.
    """
    method = validate_pre_scale_method(method)
    if method == "current":
        return {}
    record = {"pre_scale_method": method}
    if fallback:
        record["pre_scale_fallback"] = dict(fallback)
    return record
