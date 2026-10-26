# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compares an upsampled object image with the stage-5 crop it was made from.

Stage 6 sends the object crop (RGBA, alpha = the object mask) to an image model and gets an upsampled picture
back. When the model changes the object (its corners are cut, a hole is filled in, it is redrawn at another
size), the silhouette stops matching the crop. The numbers here say which way it changed.

The thresholds are settings (`s6_upsample.reference_check`) and are provisional: one scene and 15 objects, and
only `min_iou` has any evidence behind it. This module needs only numpy, PIL and scipy.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

# Two alpha images whose aspect ratios differ by more than this are compared anyway, but the result says so.
ASPECT_TOLERANCE = 0.02
# Alpha above this counts as object. Soft edges of a cut-out sit near the middle of the range.
DEFAULT_ALPHA_THRESHOLD = 127
ACTIONS = ("warn", "revert")


def _binary(alpha, threshold):
    return np.asarray(alpha) > threshold


def _hole_fraction(mask):
    """Area of the holes in a silhouette, as a fraction of its filled area (0 for an empty silhouette)."""
    filled = binary_fill_holes(mask)
    filled_area = int(filled.sum())
    if filled_area == 0:
        return 0.0
    return float((filled & ~mask).sum()) / filled_area


def silhouette_metrics(reference_alpha, upsampled_alpha, alpha_threshold=DEFAULT_ALPHA_THRESHOLD):
    """Compares two alpha images of the same object.

    The reference is resized to the upsampled size with nearest neighbour, so both silhouettes are on one grid.
    An empty silhouette has no hole fraction and no area ratio.

    Args:
        reference_alpha: (h, w) alpha of the stage-5 crop.
        upsampled_alpha: (H, W) alpha of the upsampled picture.
        alpha_threshold: alpha above this is object.

    Returns:
        dict: `iou`, `ref_area_frac`, `up_area_frac`, `area_ratio` (None when the reference is empty),
        `ref_hole_frac`, `up_hole_frac`, `hole_change` (signed: positive when holes appeared), and
        `aspect_mismatch`. Every value is a plain Python number, bool or None, so the dict serialises to JSON.
    """
    ref = _binary(reference_alpha, alpha_threshold)
    up = _binary(upsampled_alpha, alpha_threshold)
    ref_h, ref_w = ref.shape
    up_h, up_w = up.shape
    aspect_mismatch = abs((up_w / up_h) / (ref_w / ref_h) - 1.0) > ASPECT_TOLERANCE
    # Each area is a fraction of its own picture's frame; the reference is resized below only to compare silhouettes.
    ref_frame_area = ref.size
    ref_own_area = int(ref.sum())
    if ref.shape != up.shape:
        resized = Image.fromarray(ref.astype(np.uint8) * 255).resize((up_w, up_h), Image.NEAREST)
        ref = np.asarray(resized) > 127

    union = int((ref | up).sum())
    inter = int((ref & up).sum())
    iou = 1.0 if union == 0 else float(inter) / union
    ref_area = int(ref.sum())
    up_area = int(up.sum())
    ref_hole = _hole_fraction(ref)
    up_hole = _hole_fraction(up)
    return {
        "iou": iou,
        "ref_area_frac": float(ref_own_area) / ref_frame_area,
        "up_area_frac": float(up_area) / up.size,
        "area_ratio": None if ref_area == 0 else float(up_area) / ref_area,
        "ref_hole_frac": ref_hole,
        "up_hole_frac": up_hole,
        "hole_change": up_hole - ref_hole,
        "aspect_mismatch": bool(aspect_mismatch),
    }


def evaluate_flags(metrics, thresholds):
    """The checks that fail, as names. A threshold that is None skips its check.

    Args:
        metrics: the dict from `silhouette_metrics`.
        thresholds: the dict from `resolve_thresholds`.

    Returns:
        list[str]: "low_iou", "hole_changed" and "area_changed" for each check that fails.
    """
    flags = []
    min_iou = thresholds.get("min_iou")
    if min_iou is not None and metrics["iou"] < min_iou:
        flags.append("low_iou")
    max_hole = thresholds.get("max_hole_change")
    if max_hole is not None and abs(metrics["hole_change"]) > max_hole:
        flags.append("hole_changed")
    area_range = thresholds.get("area_ratio_range")
    ratio = metrics["area_ratio"]
    if area_range is not None and ratio is not None and not (area_range[0] <= ratio <= area_range[1]):
        flags.append("area_changed")
    return flags


def _alpha(rgba):
    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"expected an RGBA image, got shape {array.shape}")
    return array[..., 3]


def check_pair(reference_rgba, upsampled_rgba, thresholds):
    """`silhouette_metrics` on two RGBA images, plus the flags the thresholds raise.

    Args:
        reference_rgba: (h, w, 4) uint8, the stage-5 crop as padded for the model.
        upsampled_rgba: (H, W, 4) uint8, the upsampled picture after background removal.
        thresholds: the dict from `resolve_thresholds`.

    Returns:
        dict: the metrics of `silhouette_metrics` and `flags`.
    """
    metrics = silhouette_metrics(_alpha(reference_rgba), _alpha(upsampled_rgba))
    metrics["flags"] = evaluate_flags(metrics, thresholds)
    return metrics


def revert_to_reference(reference_rgba, out_rgb_fpath, out_rgba_fpath):
    """Writes the stage-5 crop in place of the upsampled picture, as RGB and as RGBA.

    The RGB file takes the colour channels as they are; it is not composited on white, so the crop's transparent
    padding comes back as whatever colour the padding was, and the RGBA file keeps the alpha.

    Args:
        reference_rgba: (h, w, 4) uint8, the stage-5 crop as padded for the model.
        out_rgb_fpath: where the RGB picture goes (the stage writes `upsampled/<name>.png`).
        out_rgba_fpath: where the RGBA picture goes (the stage writes `upsampled/<name>_transparent.png`).
    """
    rgba = np.ascontiguousarray(np.asarray(reference_rgba, dtype=np.uint8))
    Image.fromarray(np.ascontiguousarray(rgba[..., :3])).save(out_rgb_fpath)
    Image.fromarray(rgba).save(out_rgba_fpath)


def run_reference_check(reference_rgba, upsampled_rgb_fpath, transparent_fpath, thresholds):
    """Checks one upsampled picture against its crop and, under action "revert", writes the crop back when flagged.

    The background-removed picture is read from `transparent_fpath`. The revert is the only write, and it happens
    only when `reverted` is True: a warning is not a change.

    Args:
        reference_rgba: (h, w, 4) uint8, the stage-5 crop as padded for the model.
        upsampled_rgb_fpath: where the RGB upsampled picture is (the stage writes `upsampled/<name>.png`).
        transparent_fpath: the background-removed picture (the stage writes `upsampled/<name>_transparent.png`).
        thresholds: the dict from `resolve_thresholds`.

    Returns:
        dict: the result of `check_pair`, plus `reverted` (True only when the crop was written back).
    """
    upsampled = np.asarray(Image.open(transparent_fpath).convert("RGBA"))
    check = check_pair(reference_rgba, upsampled, thresholds)
    check["reverted"] = bool(check["flags"]) and thresholds["action"] == "revert"
    if check["reverted"]:
        revert_to_reference(reference_rgba, upsampled_rgb_fpath, transparent_fpath)
    return check


def resolve_thresholds(section):
    """The thresholds from `s6_upsample.reference_check`, in the form `evaluate_flags` reads.

    Args:
        section: the config section (an OmegaConf node or a dict) with `action`, `min_iou`, `max_hole_change`
            and `area_ratio_range`.

    Returns:
        dict: the same keys, with the range as a tuple (or None) and the numbers as floats (or None).

    Raises:
        ValueError: when `action` is neither "warn" nor "revert".
    """
    action = section.get("action", "warn")
    if action not in ACTIONS:
        raise ValueError(f"s6_upsample.reference_check.action must be one of {ACTIONS}, got {action!r}")
    min_iou = section.get("min_iou", None)
    max_hole = section.get("max_hole_change", None)
    area_range = section.get("area_ratio_range", None)
    return {
        "action": action,
        "min_iou": None if min_iou is None else float(min_iou),
        "max_hole_change": None if max_hole is None else float(max_hole),
        "area_ratio_range": None if area_range is None else tuple(float(v) for v in list(area_range)),
    }
