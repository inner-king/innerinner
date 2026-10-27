# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two stage-6 options that keep the object's own colours: infill over the occluded part only, and a colour match.

Stage 6 sends the object crop to an image model twice: once to paint the hidden part (infill) and once to upsample
the result. The model repaints the visible surface too, so a cream tofu can come back orange-brown. Both options are
settings, and both are off in the defaults:

- `infill_mode: occluded_only` keeps the crop's visible pixels and takes the infill output only where the object was
  hidden (or where the model added pixels). Its seam is feathered over `infill_feather_px` pixels.
  `infill_rim_trim_px` then takes the infill's outer rim out of the alpha (`trim_infill_rim`).
- `color_match.enabled` moves the upsampled picture's colour statistics (CIELAB, per channel) onto the crop's visible
  surface. It is skipped when the model painted more than `max_filled_fraction` of the object, because the visible
  surface's statistics would then pull the painted part towards the visible colour as well.

Three CIE76 distances are reported for each picture, and they answer different questions:

- `delta_e_*`: the distance between the mean colours of each picture's own object pixels (alpha > 127). This is the
  definition of the reference script that measures the stage-6 shift, and the headline number.
- `delta_e_visible_*`: the same distance over the visible surface alone, the upsampled pixels the crop also showed.
- `delta_e_pixel_*`: the mean per-pixel distance over that same visible surface, with the crop resized to the picture.

The own-mask distance includes any painted part, so it can be larger than the visible-surface one. The transform's
statistics use the visible surface, so the mask they come from is not the mask the headline distance is taken over.
"""

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

from simfoundry.pipeline.upsample_check import ASPECT_TOLERANCE, DEFAULT_ALPHA_THRESHOLD, silhouette_metrics

INFILL_MODES = ("full", "occluded_only")
# A channel whose spread is this small has no spread to match; scaling it would only amplify noise.
SIGMA_EPS = 1e-6
COEFFICIENT_KEYS = ("mu_picture", "mu_reference", "sd_picture", "sd_reference", "ratio")


def resolve_infill_mode(value):
    """`s6_upsample.infill_mode`, checked against the two modes.

    Raises:
        ValueError: for any other value.
    """
    if value not in INFILL_MODES:
        raise ValueError(f"s6_upsample.infill_mode must be one of {INFILL_MODES}, got {value!r}")
    return value


def resolve_color_match(section):
    """`s6_upsample.color_match` in the form `color_match_files` reads.

    Raises:
        ValueError: when `strength` is outside [0, 1], the spread range is not 0 < low < high, or
            `max_filled_fraction` is set and is outside [0, 1].
    """
    strength = float(section.get("strength", 1.0))
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"s6_upsample.color_match.strength must be in [0, 1], got {strength}")
    lo, hi = (float(v) for v in list(section.get("std_ratio_range", [0.5, 2.0])))
    if not 0.0 < lo < hi:
        raise ValueError(f"s6_upsample.color_match.std_ratio_range must be 0 < low < high, got {[lo, hi]}")
    max_filled = section.get("max_filled_fraction", None)
    if max_filled is not None and not 0.0 <= float(max_filled) <= 1.0:
        raise ValueError(f"s6_upsample.color_match.max_filled_fraction must be null or in [0, 1], got {max_filled}")
    return {
        "strength": strength,
        "match_std": bool(section.get("match_std", True)),
        "std_ratio_range": (lo, hi),
        "max_filled_fraction": None if max_filled is None else float(max_filled),
    }


def _rgb_to_lab(rgb_uint8):
    rgb = np.ascontiguousarray(np.asarray(rgb_uint8, dtype=np.float32) / 255.0)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def _lab_to_rgb_uint8(lab):
    rgb = cv2.cvtColor(np.ascontiguousarray(np.asarray(lab, dtype=np.float32)), cv2.COLOR_LAB2RGB)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _lanczos(rgb_uint8, size_wh):
    return np.asarray(Image.fromarray(np.asarray(rgb_uint8, dtype=np.uint8)).resize(size_wh, Image.LANCZOS))


def _nearest_mask(mask, size_wh):
    resized = Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255).resize(size_wh, Image.NEAREST)
    return np.asarray(resized) > DEFAULT_ALPHA_THRESHOLD


# --------------------------------------------------------------------------------------
# Infill over the occluded part only
# --------------------------------------------------------------------------------------

# The seam can only sit where the infill paints something the crop does not show. The outline the model draws around the
# object is a thin ring, a few pixels wide, and taken as seam it dims the crop's own edge. So the added region is opened
# with a square of 2 * SEAM_OPEN_RADIUS + 1 pixels first: a ring narrower than that goes, a part the model filled in (a
# hole, the back of a lid) is wider and stays. Fixed on purpose: a setting would invite tuning it per object.
SEAM_OPEN_RADIUS = 2


def opened_added_region(visible, infill_mask):
    """The part of the infill the crop does not show, with the thin outline ring removed (see `SEAM_OPEN_RADIUS`).

    Args:
        visible: (H, W) bool, the crop's visible pixels at the infill's resolution.
        infill_mask: (H, W) bool, the infill's object pixels.

    Returns:
        np.ndarray: (H, W) bool, the opened added region.
    """
    added = np.asarray(infill_mask, dtype=bool) & ~np.asarray(visible, dtype=bool)
    square = np.ones((2 * SEAM_OPEN_RADIUS + 1,) * 2, dtype=bool)
    return binary_dilation(binary_erosion(added, structure=square), structure=square)


def seam_weight(visible, real_added, feather):
    """The crop's share of each pixel, at the infill's resolution.

    Outside the visible part the weight is 0: there the picture is the infill. Inside it, the weight is 1 - near, where
    `near` is the opened added region grown by `feather` pixels and blurred with sigma = feather / 2. So the crop keeps
    its own colour wherever nothing the infill added is close, and the two blend only where they meet. With feather 0 the
    seam is hard: `near` is the added region itself.

    Args:
        visible: (H, W) bool, the crop's visible pixels.
        real_added: (H, W) bool, the opened added region (see `opened_added_region`).
        feather: seam width in pixels; 0 for a hard seam.

    Returns:
        np.ndarray: (H, W) float64 in [0, 1].
    """
    if feather == 0:
        near = np.asarray(real_added, dtype=np.float64)
    else:
        grown = binary_dilation(real_added, iterations=feather)
        near = gaussian_filter(grown.astype(np.float64), sigma=feather / 2.0)
    return np.where(visible, np.clip(1.0 - near, 0.0, 1.0), 0.0)


def composite_visible_over_infill(crop_rgba, infill_rgba, feather_px):
    """The crop's visible pixels over the infill output, with a seam only where the infill filled something in.

    Done at the infill's resolution: the crop's RGB is resized with Lanczos, its alpha with bilinear and then binarised
    at the alpha threshold. The seam is where the infill paints what the crop does not show, opened to drop the outline
    ring (`opened_added_region`); the crop keeps its own colour across the rest of its visible part, so an outline the
    infill draws around the object cannot reach the crop's edge (the dark rim it would carry). The colour is
    `w * crop + (1 - w) * infill` with `seam_weight` for `w`, and the alpha is the maximum of the two alphas, so the
    visible surface is always part of the object.

    The two must have the same aspect ratio (`ASPECT_TOLERANCE`); otherwise the infill is returned unchanged and the
    reason is recorded.

    Args:
        crop_rgba: (h, w, 4) uint8, the stage-5 crop as padded for the model.
        infill_rgba: (H, W, 4) uint8, the infill output after background removal.
        feather_px: seam width in pixels of the infill resolution.

    Returns:
        tuple: the (H, W, 4) uint8 picture and an info dict: `composited`, `reason` (None when composited),
        `visible_fraction` (visible pixels over the composite's alpha; None when not composited), `alignment_iou` (the
        two silhouettes' IoU, computed whether or not the composition ran) and `seam_added_pixels` (the opened added
        region's pixel count; None when not composited).
    """
    crop = np.asarray(crop_rgba, dtype=np.uint8)
    infill = np.asarray(infill_rgba, dtype=np.uint8)
    feather = int(feather_px)
    if feather < 0:
        raise ValueError(f"infill_feather_px must be >= 0, got {feather}")
    h_c, w_c = crop.shape[:2]
    h_i, w_i = infill.shape[:2]
    alignment_iou = silhouette_metrics(crop[..., 3], infill[..., 3])["iou"]
    if abs((w_i / h_i) / (w_c / h_c) - 1.0) > ASPECT_TOLERANCE:
        return infill.copy(), {
            "composited": False,
            "reason": f"aspect ratio differs: crop {w_c}x{h_c}, infill {w_i}x{h_i}",
            "visible_fraction": None,
            "alignment_iou": float(alignment_iou),
            "seam_added_pixels": None,
        }
    rgb_in = _lanczos(crop[..., :3], (w_i, h_i))
    alpha_in = np.asarray(Image.fromarray(crop[..., 3]).resize((w_i, h_i), Image.BILINEAR))
    visible = alpha_in > DEFAULT_ALPHA_THRESHOLD
    real_added = opened_added_region(visible, infill[..., 3] > DEFAULT_ALPHA_THRESHOLD)
    weight = seam_weight(visible, real_added, feather)
    blended = weight[..., None] * rgb_in.astype(np.float64) + (1.0 - weight[..., None]) * infill[..., :3].astype(np.float64)
    rgb = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    alpha = np.maximum(visible.astype(np.uint8) * 255, infill[..., 3]).astype(np.uint8)
    out = np.dstack([rgb, alpha])
    object_pixels = int((alpha > DEFAULT_ALPHA_THRESHOLD).sum())
    return out, {
        "composited": True,
        "reason": None,
        "visible_fraction": float(visible.sum()) / object_pixels if object_pixels else None,
        "alignment_iou": float(alignment_iou),
        "seam_added_pixels": int(real_added.sum()),
    }


def composite_infill_file(crop_rgba, infilled_fpath, feather_px):
    """Composites the crop over the infill file in place, when the two can be composited.

    Args:
        crop_rgba: (h, w, 4) uint8, the stage-5 crop as padded for the model.
        infilled_fpath: the infill output after background removal (`infilled/<name>.png`); overwritten only when
            the composition ran.
        feather_px: seam width in pixels of the infill resolution.

    Returns:
        dict: the `composite_visible_over_infill` info.
    """
    infill = np.asarray(Image.open(infilled_fpath).convert("RGBA"))
    out, info = composite_visible_over_infill(crop_rgba, infill, feather_px)
    if info["composited"]:
        Image.fromarray(out).save(infilled_fpath)
    return info


# --------------------------------------------------------------------------------------
# Rim trim: the infill's outer edge, taken out of a composite's alpha
# --------------------------------------------------------------------------------------

def resolve_rim_trim_px(value):
    """`s6_upsample.infill_rim_trim_px`: whole infill-resolution pixels to take off the infill's outer rim (0 = off).

    Raises:
        ValueError: for a negative or fractional value.
    """
    if not float(value).is_integer() or value < 0:
        raise ValueError(f"s6_upsample.infill_rim_trim_px must be a whole number >= 0, got {value}")
    return int(value)


def trim_infill_rim(crop_rgba, composite_rgba, rim_trim_px):
    """Takes the infill's outer rim out of the alpha of a composite made by `composite_visible_over_infill`.

    The composite's object is the visible surface together with the infill's object pixels, so eroding that union by
    `rim_trim_px` removes the band along the infill's outer edge. The visible surface is then put back: it is never
    cut, even at the image border, where the erosion counts the outside as empty. Only the alpha changes, to 0 or 255;
    the colour stays as the composite made it. Zero returns the composite unchanged.

    Args:
        crop_rgba: (h, w, 4) uint8, the stage-5 crop as padded for the model.
        composite_rgba: (H, W, 4) uint8, the composite at the infill's resolution.
        rim_trim_px: pixels to trim, at the infill's resolution (see `resolve_rim_trim_px`).

    Returns:
        tuple: the (H, W, 4) uint8 picture and the number of object pixels removed.
    """
    rim = resolve_rim_trim_px(rim_trim_px)
    composite = np.asarray(composite_rgba, dtype=np.uint8)
    if rim == 0:
        return composite.copy(), 0
    crop = np.asarray(crop_rgba, dtype=np.uint8)
    h, w = composite.shape[:2]
    # The surface `composite_visible_over_infill` used: the crop's alpha, bilinear to the infill's size.
    visible = np.asarray(Image.fromarray(crop[..., 3]).resize((w, h), Image.BILINEAR)) > DEFAULT_ALPHA_THRESHOLD
    union = composite[..., 3] > DEFAULT_ALPHA_THRESHOLD
    kept = visible | binary_erosion(union, iterations=rim)
    out = composite.copy()
    out[..., 3] = kept.astype(np.uint8) * 255
    return out, int((union & ~kept).sum())


def trim_infill_rim_file(crop_rgba, composite_fpath, rim_trim_px):
    """Trims a composited infill file in place (see `trim_infill_rim`). A zero setting writes nothing.

    Returns:
        dict: `rim_trim_px` (the setting) and `rim_trimmed_pixels` (object pixels removed).
    """
    rim = resolve_rim_trim_px(rim_trim_px)
    composite = np.asarray(Image.open(composite_fpath).convert("RGBA"))
    out, trimmed = trim_infill_rim(crop_rgba, composite, rim)
    if rim:
        Image.fromarray(out).save(composite_fpath)
    return {"rim_trim_px": rim, "rim_trimmed_pixels": trimmed}


# --------------------------------------------------------------------------------------
# Colour match in CIELAB
# --------------------------------------------------------------------------------------

def lab_affine_coefficients(picture_rgb, picture_mask, reference_rgb, reference_mask, match_std, std_ratio_range):
    """Per-channel mean and spread of both pictures over their visible masks, and the spread ratio.

    The ratio is `sigma_reference / sigma_picture` clipped to `std_ratio_range`, and 1 when the spread is not
    matched or the picture's spread is about zero.

    Returns:
        dict: `mu_picture`, `mu_reference`, `sd_picture`, `sd_reference` and `ratio`, each a length-3 array (L, a, b).

    Raises:
        ValueError: when either mask has no pixel.
    """
    # Float64 here: the float32 mean of a flat picture is off by about one ulp, which reads as a spread of ~1e-6 and
    # would stretch the ratio to its clip, where a flat picture has no spread to scale at all.
    pic_lab = _rgb_to_lab(picture_rgb)[np.asarray(picture_mask, dtype=bool)].astype(np.float64)
    ref_lab = _rgb_to_lab(reference_rgb)[np.asarray(reference_mask, dtype=bool)].astype(np.float64)
    if len(pic_lab) == 0 or len(ref_lab) == 0:
        raise ValueError("colour match needs visible pixels in both pictures")
    mu_pic, mu_ref = pic_lab.mean(axis=0), ref_lab.mean(axis=0)
    sd_pic, sd_ref = pic_lab.std(axis=0), ref_lab.std(axis=0)
    lo, hi = std_ratio_range
    ratio = np.ones(3)
    if match_std:
        for channel in range(3):
            if sd_pic[channel] > SIGMA_EPS:
                ratio[channel] = np.clip(sd_ref[channel] / sd_pic[channel], lo, hi)
    return {"mu_picture": mu_pic, "mu_reference": mu_ref, "sd_picture": sd_pic, "sd_reference": sd_ref, "ratio": ratio}


def apply_lab_affine(rgb, coefficients, strength):
    """Moves every pixel of a picture: `out' = (out - mu_picture) * ratio + mu_reference`, blended by `strength`.

    Strength 0 returns the picture unchanged. The Lab round trip is exact to a rounding step, so the result is
    clipped to 0-255 uint8.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    if strength == 0.0:
        return rgb.copy()
    lab = _rgb_to_lab(rgb)
    shifted = (lab - coefficients["mu_picture"]) * coefficients["ratio"] + coefficients["mu_reference"]
    return _lab_to_rgb_uint8((1.0 - strength) * lab + strength * shifted)


def mean_delta_e(picture_rgb, picture_mask, reference_rgb, reference_mask):
    """CIE76 distance between the mean Lab colours of two pictures, each taken over its own mask."""
    pic = _rgb_to_lab(picture_rgb)[np.asarray(picture_mask, dtype=bool)].mean(axis=0)
    ref = _rgb_to_lab(reference_rgb)[np.asarray(reference_mask, dtype=bool)].mean(axis=0)
    return float(np.linalg.norm(pic - ref))


def pixel_delta_e(picture_rgb, picture_mask, reference_rgb):
    """Mean per-pixel CIE76 over the picture's visible mask, with the reference resized to the picture (Lanczos)."""
    height, width = picture_rgb.shape[:2]
    ref_up = _lanczos(reference_rgb, (width, height))
    diff = _rgb_to_lab(picture_rgb) - _rgb_to_lab(ref_up)
    per_pixel = np.linalg.norm(diff, axis=-1)
    mask = np.asarray(picture_mask, dtype=bool)
    return float(per_pixel[mask].mean()) if mask.any() else None


def filled_vs_visible_delta_e(picture_rgb, filled_mask, visible_mask):
    """CIE76 between the mean Lab of the painted part and of the visible surface, both inside one picture.

    This is how far the colour the model painted sits from the colour it kept on the visible surface. None when
    either part is empty.
    """
    filled = np.asarray(filled_mask, dtype=bool)
    visible = np.asarray(visible_mask, dtype=bool)
    if not filled.any() or not visible.any():
        return None
    lab = _rgb_to_lab(picture_rgb)
    return float(np.linalg.norm(lab[filled].mean(axis=0) - lab[visible].mean(axis=0)))


def skip_reason(filled_fraction, max_filled_fraction):
    """Why the colour match is skipped, or None when it runs.

    A picture whose painted part is larger than the limit keeps its own colours: the visible surface's statistics
    would otherwise pull the painted part towards the visible colour. A fraction equal to the limit still runs, and a
    null limit never skips.

    Returns:
        str or None: the reason, in the form the stage records it.
    """
    if max_filled_fraction is None or filled_fraction <= max_filled_fraction:
        return None
    return f"filled fraction {filled_fraction:.3f} exceeds {max_filled_fraction:.3f}"


def _visible_distances(picture_rgb, visible_mask, reference_rgb):
    """Mean-colour and per-pixel CIE76 over the visible surface, with the reference on the picture's grid.

    Returns:
        tuple: the two distances, both None when the visible surface is empty.
    """
    if not visible_mask.any():
        return None, None
    height, width = picture_rgb.shape[:2]
    reference_up = _lanczos(reference_rgb, (width, height))
    return (
        mean_delta_e(picture_rgb, visible_mask, reference_up, visible_mask),
        pixel_delta_e(picture_rgb, visible_mask, reference_rgb),
    )


def match_color_lab(picture_rgb, picture_mask, reference_rgb, reference_mask, strength, match_std, std_ratio_range):
    """The picture moved towards the reference's colour statistics, with the visible-surface distances.

    The transform's statistics come from the two masks, which are the visible surface. The headline distance that
    `color_match_files` reports is taken over each object's own alpha instead, so the two masks differ on purpose.

    Args:
        picture_rgb: (H, W, 3) uint8, the upsampled picture.
        picture_mask: (H, W) bool, the picture's visible surface (its alpha, within the crop's visible mask).
        reference_rgb: (h, w, 3) uint8, the stage-5 crop's colour.
        reference_mask: (h, w) bool, the crop's visible mask.
        strength: 0 keeps the picture, 1 moves it fully.
        match_std: also match each channel's spread.
        std_ratio_range: (low, high) clip for the spread ratio.

    Returns:
        tuple: the matched (H, W, 3) uint8 picture and the info dict: `strength`, `match_std`,
        `delta_e_visible_before/after` and `delta_e_pixel_before/after` (both over `picture_mask`), and `coefficients`
        (the output of `lab_affine_coefficients`, to apply the same transform to another picture).
    """
    coefficients = lab_affine_coefficients(picture_rgb, picture_mask, reference_rgb, reference_mask, match_std,
                                           std_ratio_range)
    matched = apply_lab_affine(picture_rgb, coefficients, strength)
    before_visible, before_pixel = _visible_distances(picture_rgb, picture_mask, reference_rgb)
    after_visible, after_pixel = _visible_distances(matched, picture_mask, reference_rgb)
    return matched, {
        "strength": float(strength),
        "match_std": bool(match_std),
        "delta_e_visible_before": before_visible,
        "delta_e_visible_after": after_visible,
        "delta_e_pixel_before": before_pixel,
        "delta_e_pixel_after": after_pixel,
        "coefficients": coefficients,
    }


def _distances(picture_rgb, reference_rgb, cutout, crop, visible, filled):
    """Every distance the record reports for one picture. Each is None when the mask it needs is empty.

    `cutout` and `crop` are the two objects' own masks (the headline distance); `visible` and `filled` split the
    cutout by whether the crop showed the pixel.
    """
    visible_de, pixel_de = _visible_distances(picture_rgb, visible, reference_rgb)
    own = mean_delta_e(picture_rgb, cutout, reference_rgb, crop) if cutout.any() and crop.any() else None
    return {
        "delta_e": own,
        "delta_e_visible": visible_de,
        "delta_e_pixel": pixel_de,
        "filled_vs_visible_delta_e": filled_vs_visible_delta_e(picture_rgb, filled, visible),
    }


def _color_match_report(settings, filled_fraction, before, after, coefficients, skipped):
    report = {
        "strength": settings["strength"],
        "match_std": settings["match_std"],
        "filled_fraction": filled_fraction,
        "delta_e_before": before["delta_e"],
        "delta_e_after": after["delta_e"],
        "delta_e_visible_before": before["delta_e_visible"],
        "delta_e_visible_after": after["delta_e_visible"],
        "delta_e_pixel_before": before["delta_e_pixel"],
        "delta_e_pixel_after": after["delta_e_pixel"],
        "filled_vs_visible_delta_e_before": before["filled_vs_visible_delta_e"],
        "filled_vs_visible_delta_e_after": after["filled_vs_visible_delta_e"],
        "skipped": skipped,
    }
    for key in COEFFICIENT_KEYS:
        report[key] = None if coefficients is None else [float(v) for v in coefficients[key]]
    return report


def color_match_files(reference_rgba, transparent_fpath, rgb_fpath, settings):
    """Colour-matches one upsampled object: both of its files get the same transform, or neither is written.
    Only the pixels the object covers (alpha > 0) take the transform, in both files.

    Two masks are in play, and they answer different questions. The transform's statistics use the visible surface:
    the upsampled object's pixels that the crop also showed, against the crop's own pixels. The headline distance
    (`delta_e_*`) compares each object's own alpha instead, as the reference script does. The painted part (object
    pixels the crop did not show) is reported as a fraction of the upsampled object's pixels, and the match is skipped
    when that fraction is above `max_filled_fraction`.

    Args:
        reference_rgba: (h, w, 4) uint8, the stage-5 crop as padded for the model.
        transparent_fpath: the background-removed picture (`upsampled/<name>_transparent.png`).
        rgb_fpath: the RGB picture (`upsampled/<name>.png`).
        settings: the dict from `resolve_color_match`.

    Returns:
        dict: the headline and visible-surface distances before and after (None when a mask is empty),
        `filled_fraction`, `filled_vs_visible_delta_e_before/after`, the transform's coefficients (None when skipped),
        and `skipped` (None when the match ran, else the reason).
    """
    transparent = np.asarray(Image.open(transparent_fpath).convert("RGBA"))
    rgb_file = np.asarray(Image.open(rgb_fpath).convert("RGB"))
    reference = np.asarray(reference_rgba, dtype=np.uint8)
    height, width = transparent.shape[:2]
    picture_rgb = transparent[..., :3]
    reference_rgb = reference[..., :3]
    cutout = transparent[..., 3] > DEFAULT_ALPHA_THRESHOLD
    crop = reference[..., 3] > DEFAULT_ALPHA_THRESHOLD
    crop_up = _nearest_mask(crop, (width, height))
    covered = transparent[..., 3] > 0
    visible = cutout & crop_up
    filled = cutout & ~crop_up
    cutout_pixels = int(cutout.sum())
    filled_fraction = float(filled.sum()) / cutout_pixels if cutout_pixels else None
    if not cutout.any():
        skipped = "no object pixels in the upsampled picture"
    elif not visible.any():
        skipped = "the upsampled picture and the crop share no visible pixels"
    else:
        skipped = skip_reason(filled_fraction, settings["max_filled_fraction"])
    before = _distances(picture_rgb, reference_rgb, cutout, crop, visible, filled)
    if skipped is not None:
        return _color_match_report(settings, filled_fraction, before, {key: None for key in before}, None, skipped)
    matched_t, info = match_color_lab(picture_rgb, visible, reference_rgb, crop, settings["strength"],
                                      settings["match_std"], settings["std_ratio_range"])
    # Only the pixels the object covers take the transform, so the background keeps its colour in both files.
    matched_t = np.where(covered[..., None], matched_t, picture_rgb)
    matched_rgb = apply_lab_affine(rgb_file, info["coefficients"], settings["strength"])
    matched_rgb = np.where(covered[..., None], matched_rgb, rgb_file)
    after = _distances(matched_t, reference_rgb, cutout, crop, visible, filled)
    Image.fromarray(np.dstack([matched_t, transparent[..., 3]])).save(transparent_fpath)
    Image.fromarray(matched_rgb).save(rgb_fpath)
    return _color_match_report(settings, filled_fraction, before, after, info["coefficients"], None)
