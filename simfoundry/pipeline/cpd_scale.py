# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 8's optional limit on the CPD scale (`s8_pose.cpd_scale_limit`, off by default).

Stage 8 scales the mesh by `pre_scale_factor` and then lets CPD choose a further scale m, carried as `tf_z_up.scale`.
With a limit r set, each scaled fit whose m lies outside [1 - r, 1 + r] is moved to the nearest bound and refitted
there by a fixed-scale rigid CPD; fits inside the bounds are left as they were. The fits are then pooled with the
unscaled ones and the lowest `dist` wins, as before.

Known issue, recorded and not fixed: `dist` is measured in the target frame after the source was scaled by 1/m, so the
dists of fits with different m are not in one unit. A unit correction (dist * m^2) was tried and withdrawn; see
b9_scale/report_pre_ko.md (research 2).
"""

import copy
import math
from typing import Any

import numpy as np
import open3d as o3d
from probreg import cpd, transformation

from simfoundry.utils.faiss_utils import l2_search


def validate_cpd_scale_limit(value) -> float | None:
    """The configured limit r, checked: None (off), or a float strictly between 0 and 1.

    Args:
        value: the value of `s8_pose.cpd_scale_limit`.

    Returns:
        None when the limit is off, else `value` as a float.

    Raises:
        ValueError: for anything else. Zero is refused on purpose: it would not mean "unscaled fits only" (those are kept
            whatever the limit is), it would mean no scaled fits at all, and the config spells "off" as null.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"s8_pose.cpd_scale_limit must be null or a float in (0, 1), got {value!r}")
    if not 0.0 < value < 1.0:
        hint = " (zero is not a limit: use null to turn it off)" if value == 0.0 else ""
        raise ValueError(f"s8_pose.cpd_scale_limit must be null or a float in (0, 1), got {value!r}{hint}")
    return float(value)


def cpd_scale_of(tf) -> float:
    """The scalar scale of a transform. probreg keeps it as a numpy scalar; the stage reads it as one."""
    return float(np.asarray(tf.scale).ravel()[0])


def clamp_target(m: float, limit: float) -> float | None:
    """The bound a scale m is moved to, or None when m is inside [1 - limit, 1 + limit].

    The bounds themselves count as inside: a fit exactly at 1 +/- limit is not refitted.

    Args:
        m: the CPD scale of a fit.
        limit: the validated limit r.

    Returns:
        The nearest of 1 - limit and 1 + limit when m is outside them, else None.
    """
    low, high = 1.0 - limit, 1.0 + limit
    if m < low:
        return low
    if m > high:
        return high
    return None


def _fit_entry(result, tf_cum, target, tf_y_up) -> dict[str, Any]:
    """A fit in `sample_fits`' format, from its cumulative transform (source to target frame).

    `dist` is `sample_fits`' definition: the sum of the nearest distances from the target's points to the result's.
    `tf_y_up` is made the way `sample_fits` makes it, the cumulative transform times the constant and then inverted.
    """
    dists, _ = l2_search(np.array(target.points), np.array(result.points), 1)
    return {
        "result": result,
        "dist": np.sum(dists),
        "tf_z_up": tf_cum.inverse(),
        "tf_y_up": (tf_cum * tf_y_up).inverse(),
    }


def clamp_refit(entry: dict[str, Any], m_c: float, target, tf_y_up) -> dict[str, Any]:
    """The fit `entry` again, held at the scale m_c, in `sample_fits`' format.

    The aligned source (`result`, in the target frame) is scaled about its centroid by k = (1/m_c) / (its current scale),
    which moves it to the size that m_c stands for. A rigid CPD with the scale held fixed (`update_scale=False`) then
    refits its rotation and translation to the target. Scaling about the centroid leaves the source where it was, so the
    refit only corrects its pose. The `dist` of the result is measured after the scaling; see the module docstring.

    Args:
        entry: a scaled fit from `sample_fits` (`update_scale=True`, `return_inverse=True`).
        m_c: the scale to hold the fit at, normally one of the bounds from `clamp_target`.
        target: the target point cloud that `sample_fits` was given.
        tf_y_up: the stage's module-level `tf_y_up`, the constant that `sample_fits` multiplies in.

    Returns:
        dict: `result`, `dist`, `tf_z_up` (with scale m_c) and `tf_y_up`.
    """
    tf_cum = entry["tf_z_up"].inverse()  # source to target frame: what `sample_fits` inverts for `tf_z_up`
    k = (1.0 / m_c) / cpd_scale_of(tf_cum)
    result = entry["result"]
    centre = np.asarray(result.points).mean(0)
    moved = copy.deepcopy(result)
    moved.points = o3d.utility.Vector3dVector((np.asarray(result.points) - centre) * k + centre)
    tf_scale = transformation.RigidTransformation(rot=np.eye(3), t=(1.0 - k) * centre, scale=k)
    tf_param, _, _ = cpd.registration_cpd(
        moved, copy.deepcopy(target), tf_type_name="rigid", use_color=False, use_cuda=False, update_scale=False
    )
    refit = copy.deepcopy(moved)
    refit.points = tf_param.transform(refit.points)
    return _fit_entry(refit, tf_param * tf_scale * tf_cum, target, tf_y_up)


def apply_cpd_scale_limit(scaled, limit, target, tf_y_up) -> list[dict[str, Any]]:
    """The scaled fits with the limit applied: each fit outside the bounds clamped and refitted, the rest unchanged.

    With `limit` None the list is returned as it is, with no extra keys, so the stage's outputs are those it gave before
    the limit existed. With a limit set, every fit returned records `cpd_scale_unclamped` (its m before the limit) and
    `cpd_scale_clamped` (whether it was refitted).

    Args:
        scaled: the scaled fits from `sample_fits`, after the stage's prune of degenerate scales.
        limit: the configured `s8_pose.cpd_scale_limit`: None, or a float in (0, 1).
        target: the target point cloud that `sample_fits` was given.
        tf_y_up: the stage's module-level `tf_y_up`.

    Returns:
        list of dict: the fits, in the order given.
    """
    limit = validate_cpd_scale_limit(limit)
    if limit is None:
        return scaled
    fits = []
    for entry in scaled:
        m = cpd_scale_of(entry["tf_z_up"])
        m_c = clamp_target(m, limit)
        if m_c is None:
            fits.append({**entry, "cpd_scale_unclamped": m, "cpd_scale_clamped": False})
        else:
            refit = clamp_refit(entry, m_c, target, tf_y_up)
            fits.append({**refit, "cpd_scale_unclamped": m, "cpd_scale_clamped": True})
    return fits


def select_top_fit(unscaled, scaled, limit, target, tf_y_up) -> dict[str, Any]:
    """The fit stage 8 keeps: the lowest `dist` among the unscaled fits and the limited scaled fits.

    The stage pools and sorts inline, because its sorted list is used after the choice. This is the same rule, so the
    tests can check the choice without running the stage.
    """
    pooled = list(unscaled) + list(apply_cpd_scale_limit(scaled, limit, target, tf_y_up))
    return sorted(pooled, key=lambda fit: fit["dist"])[0]


def cpd_scale_limit_record(limit, top_fit) -> dict[str, Any]:
    """The info keys the limit adds to stage 8's output, and none when the limit is off.

    `cpd_scale_unclamped` is the chosen fit's m before the limit. A fit that was never scaled has its own scale (1.0).
    The `cpd_scale` key of the same file is the m after the limit.

    Args:
        limit: the configured `s8_pose.cpd_scale_limit`.
        top_fit: the chosen fit.

    Returns:
        dict: `cpd_scale_limit`, `cpd_scale_clamped` and `cpd_scale_unclamped`, or {} when the limit is off.
    """
    limit = validate_cpd_scale_limit(limit)
    if limit is None:
        return {}
    return {
        "cpd_scale_limit": limit,
        "cpd_scale_clamped": bool(top_fit.get("cpd_scale_clamped", False)),
        "cpd_scale_unclamped": float(top_fit.get("cpd_scale_unclamped", cpd_scale_of(top_fit["tf_z_up"]))),
    }
