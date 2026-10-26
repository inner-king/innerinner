# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 6's infill-over-occluded composite and colour match (simfoundry/pipeline/upsample_color.py).

The arithmetic runs on the CPU with synthetic masks and pictures. The stage needs a GPU, so its wiring is read from its
source with ast, and the default settings are read from the config file.
"""

import ast
import json
import pathlib
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image
from scipy.ndimage import binary_erosion

from simfoundry.pipeline.upsample_color import (
    _lab_to_rgb_uint8,
    _rgb_to_lab,
    apply_lab_affine,
    color_match_files,
    composite_infill_file,
    composite_visible_over_infill,
    filled_vs_visible_delta_e,
    lab_affine_coefficients,
    match_color_lab,
    pixel_delta_e,
    resolve_color_match,
    resolve_infill_mode,
    resolve_rim_trim_px,
    skip_reason,
    trim_infill_rim,
    trim_infill_rim_file,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGE = ROOT / "scripts/pipeline/A_reconstruction/stages/6_upsample_object_images.py"
CONFIG = ROOT / "scripts/cfg/real2sim_cfg.yaml"
SHIFT = (10.0, 5.0, -3.0)
CREAM = (200, 180, 150)
PAINTED = (30, 60, 200)
# Inside the gamut after SHIFT. PAINTED sits at the gamut edge, and the 0-255 clip would then change its Lab.
MUTED = (110, 120, 150)


# ---------------------------------------------------------------------------
# Synthetic pictures and reference colours
# ---------------------------------------------------------------------------

def _texture(h, w, seed=0):
    return np.random.default_rng(seed).integers(60, 200, size=(h, w, 3), dtype=np.uint8)


def _crop(h, w, box, rgb):
    """An RGBA picture whose object is `box` = (top, left, bottom, right), coloured `rgb` (a colour or an array)."""
    out = np.zeros((h, w, 4), dtype=np.uint8)
    top, left, bottom, right = box
    out[top:bottom, left:right, :3] = rgb
    out[top:bottom, left:right, 3] = 255
    return out


def _solid(h, w, rgb, alpha=255):
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = alpha
    return out


def _lanczos(rgb, h, w):
    return np.asarray(Image.fromarray(rgb).resize((w, h), Image.LANCZOS))


def _mean_lab(rgb, mask):
    lab = cv2.cvtColor(np.asarray(rgb, dtype=np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return lab[mask].mean(axis=0)


def _lab_of(rgb):
    pixel = np.array([[rgb]], dtype=np.float32) / 255.0
    return cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0, 0]


def _shifted(rgb, shift):
    """`rgb` moved by a uniform Lab offset, rounded back to 8-bit RGB the way the stage would store it."""
    lab = cv2.cvtColor(np.asarray(rgb, dtype=np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lab = lab + np.asarray(shift, dtype=np.float32)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def _write_pair(folder, crop, transparent, rgb):
    folder.mkdir(parents=True, exist_ok=True)
    transparent_path = folder / "upsampled_transparent.png"
    rgb_path = folder / "upsampled.png"
    Image.fromarray(transparent).save(transparent_path)
    Image.fromarray(rgb).save(rgb_path)
    return transparent_path, rgb_path


# ---------------------------------------------------------------------------
# Infill over the occluded part only
# ---------------------------------------------------------------------------

def test_visible_interior_keeps_the_crop_colour_not_the_infill():
    crop = _crop(40, 40, (10, 10, 30, 30), _texture(20, 20))
    out, info = composite_visible_over_infill(crop, _solid(80, 80, (20, 240, 30)), feather_px=3)

    expected = _lanczos(crop[..., :3], 80, 80)
    # The weight is 1 once the eroded visible surface and the blur's reach (4 sigma) both lie inside the object:
    # 9 px at feather 3, so the centre of a 20 px square at infill resolution is exact.
    inner = np.s_[30:50, 30:50]
    assert info["composited"] is True
    assert np.abs(out[inner][..., :3].astype(int) - expected[inner].astype(int)).max() <= 1


def test_a_hole_takes_the_infill_colour_and_becomes_object():
    crop = _crop(40, 40, (2, 2, 38, 38), CREAM)
    crop[14:26, 14:26] = 0  # the hole, alpha 0
    out, _ = composite_visible_over_infill(crop, _solid(80, 80, PAINTED), feather_px=3)

    hole = out[36:44, 36:44]  # the hole is rows and columns 28 to 52 at infill resolution
    assert (hole[..., :3] == PAINTED).all()
    assert (hole[..., 3] == 255).all()
    band = out[12:20, 12:20, :3]  # the visible band, away from its edges
    expected = _lanczos(crop[..., :3], 80, 80)[12:20, 12:20]
    assert np.abs(band.astype(int) - expected.astype(int)).max() <= 1


def test_infill_cannot_change_the_visible_interior_of_a_solid_object():
    crop = _crop(40, 40, (10, 10, 30, 30), _texture(20, 20, seed=1))
    noise = np.dstack([np.random.default_rng(2).integers(0, 256, (80, 80, 3), dtype=np.uint8),
                       np.full((80, 80), 255, dtype=np.uint8)])
    expected = _lanczos(crop[..., :3], 80, 80)[30:50, 30:50].astype(int)
    for infill in (_solid(80, 80, (250, 10, 10)), noise):
        out, _ = composite_visible_over_infill(crop, infill, feather_px=3)
        assert np.abs(out[30:50, 30:50, :3].astype(int) - expected).max() <= 1


def test_composite_runs_at_the_infill_resolution():
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    out, info = composite_visible_over_infill(crop, _solid(80, 80, PAINTED), feather_px=3)

    assert out.shape == (80, 80, 4)
    assert info["composited"] is True
    assert info["reason"] is None
    assert 0.0 < info["visible_fraction"] <= 1.0


def test_aspect_mismatch_beyond_tolerance_is_refused_and_recorded():
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    infill = _solid(120, 80, PAINTED)  # 2:3 against a square crop
    out, info = composite_visible_over_infill(crop, infill, feather_px=3)

    assert info["composited"] is False
    assert "aspect" in info["reason"]
    assert info["visible_fraction"] is None
    assert np.array_equal(out, infill)
    assert isinstance(info["alignment_iou"], float)


def test_aspect_just_inside_the_tolerance_is_composited_and_the_direction_matters():
    # 149/152 is 1.97% off the square crop, inside the 2% tolerance. The inverse 152/149 is 2.01% off, outside it.
    crop = _crop(152, 152, (30, 30, 120, 120), CREAM)
    out, info = composite_visible_over_infill(crop, _solid(152, 149, PAINTED), feather_px=3)

    assert info["composited"] is True
    assert out.shape == (152, 149, 4)


def test_feather_zero_is_a_hard_seam_and_feather_blends_inside_the_object():
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    expected = _lanczos(crop[..., :3], 80, 80)
    hard, _ = composite_visible_over_infill(crop, _solid(80, 80, PAINTED), feather_px=0)
    soft, _ = composite_visible_over_infill(crop, _solid(80, 80, PAINTED), feather_px=3)

    # The first visible column at infill resolution is 20 and the last hidden one is 19.
    assert np.array_equal(hard[40, 20, :3], expected[40, 20])
    assert np.array_equal(hard[40, 19, :3], PAINTED)
    # With feather 3 the crop's weight at its edge is about 0.05, so the edge pixel is mostly painted colour.
    assert not np.array_equal(soft[40, 20, :3], hard[40, 20, :3])
    to_painted = np.abs(soft[40, 20, :3].astype(int) - PAINTED).sum()
    to_crop = np.abs(soft[40, 20, :3].astype(int) - expected[40, 20]).sum()
    assert to_painted < to_crop


def test_feather_width_changes_how_many_pixels_are_blended():
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    infill = _solid(80, 80, PAINTED)
    expected = _lanczos(crop[..., :3], 80, 80).astype(int)

    def blended_count(feather):
        out, _ = composite_visible_over_infill(crop, infill, feather_px=feather)
        mixed = np.abs(out[..., :3].astype(int) - expected).sum(axis=-1) > 0
        mixed &= np.abs(out[..., :3].astype(int) - np.asarray(PAINTED)).sum(axis=-1) > 0
        return int(mixed.sum())

    assert blended_count(6) > blended_count(2) > 0


def test_a_semi_transparent_edge_of_the_crop_is_not_visible():
    # Alpha 100 is below the threshold, so the halo is not visible surface and takes the infill colour, not the crop's.
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    crop[6:34, 6:34, :3] = CREAM
    crop[6:34, 6:34, 3] = 100
    crop[10:30, 10:30, 3] = 255
    out, _ = composite_visible_over_infill(crop, _solid(80, 80, PAINTED), feather_px=3)

    # Row 14 is the halo at infill resolution, 6 px from the visible surface, where the weight is exactly 0.
    assert np.array_equal(out[14, 40, :3], PAINTED)


def test_alpha_is_the_larger_of_the_visible_mask_and_the_infill_alpha():
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    infill = _solid(80, 80, PAINTED, alpha=0)
    infill[:, 12:18, 3] = 100  # outside the visible surface, below the alpha threshold
    infill[:, 70:76, 3] = 255  # outside the visible surface, opaque
    out, _ = composite_visible_over_infill(crop, infill, feather_px=3)

    assert (out[30:50, 30:50, 3] == 255).all()  # the visible surface stays object under a transparent infill
    assert (out[:, 12:18, 3] == 100).all()
    assert (out[:, 70:76, 3] == 255).all()
    assert (out[:, 0:12, 3] == 0).all()


def test_info_serialises_to_json_in_both_branches():
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    _, composited = composite_visible_over_infill(crop, _solid(80, 80, PAINTED), feather_px=3)
    _, refused = composite_visible_over_infill(crop, _solid(120, 80, PAINTED), feather_px=3)

    for info in (composited, refused):
        assert json.loads(json.dumps(info)) == info
        assert type(info["composited"]) is bool


def test_infill_file_is_overwritten_only_when_the_composition_ran(tmp_path):
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    composed = tmp_path / "composed.png"
    Image.fromarray(_solid(80, 80, PAINTED)).save(composed)
    info = composite_infill_file(crop, composed, 3)
    expected, _ = composite_visible_over_infill(crop, _solid(80, 80, PAINTED), 3)

    assert info["composited"] is True
    assert np.array_equal(np.asarray(Image.open(composed)), expected)

    refused = tmp_path / "refused.png"
    painted = _solid(120, 80, PAINTED)
    Image.fromarray(painted).save(refused)
    info = composite_infill_file(crop, refused, 3)

    assert info["composited"] is False
    assert np.array_equal(np.asarray(Image.open(refused)), painted)


# ---------------------------------------------------------------------------
# Colour match: the transform
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("match_std", [False, True])
def test_full_strength_undoes_a_uniform_lab_shift_and_half_strength_undoes_half(match_std):
    reference = np.random.default_rng(3).integers(110, 190, size=(48, 48, 3), dtype=np.uint8)
    picture = _shifted(reference, SHIFT)
    mask = np.ones((48, 48), dtype=bool)

    _, full = match_color_lab(picture, mask, reference, mask, 1.0, match_std, (0.5, 2.0))
    _, half = match_color_lab(picture, mask, reference, mask, 0.5, match_std, (0.5, 2.0))
    untouched, none = match_color_lab(picture, mask, reference, mask, 0.0, match_std, (0.5, 2.0))

    assert full["delta_e_visible_before"] == pytest.approx(np.linalg.norm(SHIFT), abs=0.5)
    assert full["delta_e_visible_after"] < 0.25
    assert half["delta_e_visible_after"] == pytest.approx(0.5 * full["delta_e_visible_before"], abs=0.3)
    assert np.array_equal(untouched, picture)
    assert none["delta_e_visible_after"] == pytest.approx(none["delta_e_visible_before"])


def test_statistics_are_lab_means_over_the_masked_pixels_only():
    rng = np.random.default_rng(4)
    picture = rng.integers(0, 256, (30, 30, 3), dtype=np.uint8)
    reference = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
    picture_mask = rng.random((30, 30)) > 0.5
    reference_mask = rng.random((20, 20)) > 0.3
    coefficients = lab_affine_coefficients(picture, picture_mask, reference, reference_mask, True, (0.5, 2.0))

    assert np.allclose(coefficients["mu_picture"], _mean_lab(picture, picture_mask), atol=1e-4)
    assert np.allclose(coefficients["mu_reference"], _mean_lab(reference, reference_mask), atol=1e-4)
    assert np.allclose(coefficients["sd_reference"], cv2.cvtColor(
        reference.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)[reference_mask].std(axis=0), atol=1e-4)


def test_pixels_outside_the_masks_do_not_enter_the_statistics():
    rng = np.random.default_rng(5)
    picture = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    reference = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    picture_mask = np.zeros((24, 24), dtype=bool)
    picture_mask[:, :12] = True
    reference_mask = picture_mask.copy()
    base = lab_affine_coefficients(picture, picture_mask, reference, reference_mask, True, (0.5, 2.0))

    picture[:, 12:] = 255  # the painted part, outside the visible surface
    reference[:, 12:] = 0  # the crop's padding
    changed = lab_affine_coefficients(picture, picture_mask, reference, reference_mask, True, (0.5, 2.0))

    for key in base:
        assert np.array_equal(base[key], changed[key])


def test_match_std_off_keeps_the_spread_ratio_at_one():
    rng = np.random.default_rng(6)
    picture = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    reference = rng.integers(40, 220, (24, 24, 3), dtype=np.uint8)
    mask = np.ones((24, 24), dtype=bool)

    coefficients = lab_affine_coefficients(picture, mask, reference, mask, False, (0.5, 2.0))

    assert np.array_equal(coefficients["ratio"], [1.0, 1.0, 1.0])


def test_spread_ratio_is_clipped_to_the_configured_range_in_both_directions():
    rng = np.random.default_rng(7)
    flat = (120 + rng.integers(0, 2, (40, 40, 3))).astype(np.uint8)  # a spread of one level
    wide = rng.integers(40, 220, (40, 40, 3), dtype=np.uint8)
    mask = np.ones((40, 40), dtype=bool)

    widen = lab_affine_coefficients(flat, mask, wide, mask, True, (0.5, 2.0))["ratio"]
    narrow = lab_affine_coefficients(wide, mask, flat, mask, True, (0.5, 2.0))["ratio"]

    assert np.all(widen == 2.0)
    assert np.all(narrow == 0.5)


def test_a_flat_picture_has_no_spread_to_scale_and_moves_to_the_reference_colour():
    flat = np.full((16, 16, 3), 90, dtype=np.uint8)
    other = np.random.default_rng(8).integers(40, 220, (16, 16, 3), dtype=np.uint8)
    mask = np.ones((16, 16), dtype=bool)

    coefficients = lab_affine_coefficients(flat, mask, other, mask, True, (0.5, 2.0))
    matched = apply_lab_affine(flat, coefficients, 1.0)

    assert np.array_equal(coefficients["ratio"], [1.0, 1.0, 1.0])
    assert np.linalg.norm(_mean_lab(matched, mask) - _mean_lab(other, mask)) < 1.0


def test_matching_leaves_the_shape_dtype_and_alpha_of_both_files_alone(tmp_path):
    rng = np.random.default_rng(9)
    crop = _crop(40, 40, (8, 8, 32, 32), rng.integers(60, 200, (24, 24, 3), dtype=np.uint8))
    transparent = _solid(80, 80, (0, 0, 0), alpha=0)
    transparent[16:64, 16:64, :3] = rng.integers(40, 230, (48, 48, 3), dtype=np.uint8)
    transparent[16:64, 16:64, 3] = 255
    transparent_path, rgb_path = _write_pair(tmp_path, crop, transparent, transparent[..., :3].copy())

    color_match_files(crop, transparent_path, rgb_path, resolve_color_match({}))
    matched = np.asarray(Image.open(transparent_path))
    matched_rgb = np.asarray(Image.open(rgb_path))

    assert matched.shape == transparent.shape and matched.dtype == np.uint8
    assert np.array_equal(matched[..., 3], transparent[..., 3])
    assert matched_rgb.shape == (80, 80, 3) and matched_rgb.dtype == np.uint8
    assert np.array_equal(matched_rgb, matched[..., :3])  # the same transform reached both files


# ---------------------------------------------------------------------------
# Colour match: the reported distances, the filled part and the limit
# ---------------------------------------------------------------------------

def _filled_case():
    """A 20 px crop with a 10 px visible square of cream, and a cutout of the same size.

    The cutout's visible square drifted by SHIFT, and rows 0 to 4 were painted blue where the crop showed nothing.
    The cutout has 200 object pixels, 100 of them painted: a filled fraction of 0.5. A frame denominator would give
    0.25 and a visible-surface one would give 1.0.
    """
    crop = _crop(20, 20, (5, 5, 15, 15), CREAM)
    transparent = np.zeros((20, 20, 4), dtype=np.uint8)
    transparent[..., :3] = MUTED
    transparent[5:15, 5:15, :3] = _shifted(np.full((1, 1, 3), CREAM, dtype=np.uint8), SHIFT)[0, 0]
    transparent[5:15, 5:15, 3] = 255
    transparent[0:5, :, 3] = 255
    return crop, transparent


def test_the_report_gives_the_headline_visible_and_painted_distances(tmp_path):
    crop, transparent = _filled_case()
    transparent_path, rgb_path = _write_pair(tmp_path, crop, transparent, transparent[..., :3].copy())
    report = color_match_files(crop, transparent_path, rgb_path, resolve_color_match({}))

    rgb = transparent[..., :3]
    cutout = transparent[..., 3] > 127
    crop_mask = crop[..., 3] > 127
    visible = cutout & crop_mask
    filled = cutout & ~crop_mask
    lab_crop = _mean_lab(crop[..., :3], crop_mask)
    lab_visible = _mean_lab(rgb, visible)
    lab_filled = _mean_lab(rgb, filled)

    assert report["skipped"] is None
    assert report["filled_fraction"] == pytest.approx(0.5)
    # Headline: each object's own alpha, as the reference script measures it.
    assert report["delta_e_before"] == pytest.approx(np.linalg.norm(_mean_lab(rgb, cutout) - lab_crop), abs=1e-3)
    # The drift is uniform, so the visible surface is undone by the transform and the painted part keeps its gap.
    assert report["delta_e_visible_before"] == pytest.approx(np.linalg.norm(lab_visible - lab_crop), abs=1e-3)
    assert report["delta_e_visible_after"] < 0.5
    assert report["filled_vs_visible_delta_e_before"] == pytest.approx(np.linalg.norm(lab_filled - lab_visible), abs=1e-3)
    assert report["filled_vs_visible_delta_e_after"] == pytest.approx(report["filled_vs_visible_delta_e_before"], abs=0.5)
    # After the mean-only shift, the cutout's mean is the crop's plus half the painted part's gap.
    assert report["delta_e_after"] == pytest.approx(0.5 * report["filled_vs_visible_delta_e_before"], abs=0.5)


def test_a_fully_visible_picture_has_no_painted_part_to_measure(tmp_path):
    crop, transparent = _filled_case()
    transparent[0:5, :, 3] = 0
    transparent_path, rgb_path = _write_pair(tmp_path, crop, transparent, transparent[..., :3].copy())
    report = color_match_files(crop, transparent_path, rgb_path, resolve_color_match({}))

    assert report["filled_fraction"] == 0.0
    assert report["filled_vs_visible_delta_e_before"] is None
    assert report["filled_vs_visible_delta_e_after"] is None


@pytest.mark.parametrize("limit, applied", [(0.3, False), (0.5, True), (0.7, True), (None, True)])
def test_the_limit_skips_above_it_and_runs_at_or_below_it(tmp_path, limit, applied):
    crop, transparent = _filled_case()
    transparent_path, rgb_path = _write_pair(tmp_path, crop, transparent, transparent[..., :3].copy())
    before = (transparent_path.read_bytes(), rgb_path.read_bytes())

    report = color_match_files(crop, transparent_path, rgb_path, resolve_color_match({"max_filled_fraction": limit}))

    assert report["filled_fraction"] == pytest.approx(0.5)
    if applied:
        assert report["skipped"] is None
        assert transparent_path.read_bytes() != before[0]
        assert rgb_path.read_bytes() != before[1]
        assert report["delta_e_after"] is not None
    else:
        assert report["skipped"] == "filled fraction 0.500 exceeds 0.300"
        assert (transparent_path.read_bytes(), rgb_path.read_bytes()) == before
        assert report["delta_e_after"] is None
        assert report["mu_picture"] is None
        assert report["delta_e_before"] is not None


@pytest.mark.parametrize("fraction, limit, expected", [
    (0.9, None, None),
    (0.5, 0.5, None),
    (0.3, 0.5, None),
    (0.6, 0.5, "filled fraction 0.600 exceeds 0.500"),
])
def test_skip_reason_is_inclusive_at_the_limit_and_never_skips_without_one(fraction, limit, expected):
    assert skip_reason(fraction, limit) == expected


def test_an_empty_cutout_or_a_cutout_away_from_the_crop_is_skipped_not_fatal(tmp_path):
    crop, _ = _filled_case()
    empty = np.zeros((20, 20, 4), dtype=np.uint8)
    empty_path, empty_rgb = _write_pair(tmp_path / "empty", crop, empty, empty[..., :3].copy())
    report = color_match_files(crop, empty_path, empty_rgb, resolve_color_match({}))
    assert report["skipped"] == "no object pixels in the upsampled picture"
    assert report["filled_fraction"] is None

    away = np.zeros((20, 20, 4), dtype=np.uint8)
    away[0:4, 0:4] = (255, 0, 0, 255)
    away_path, away_rgb = _write_pair(tmp_path / "away", crop, away, away[..., :3].copy())
    before = (away_path.read_bytes(), away_rgb.read_bytes())
    report = color_match_files(crop, away_path, away_rgb, resolve_color_match({}))
    assert report["skipped"] == "the upsampled picture and the crop share no visible pixels"
    assert (away_path.read_bytes(), away_rgb.read_bytes()) == before


def test_painted_part_distance_is_none_when_nothing_was_painted():
    picture = np.full((8, 8, 3), 100, dtype=np.uint8)
    visible = np.ones((8, 8), dtype=bool)

    assert filled_vs_visible_delta_e(picture, np.zeros((8, 8), dtype=bool), visible) is None


def test_the_limit_is_validated_and_defaults_to_no_limit():
    assert resolve_color_match({})["max_filled_fraction"] is None
    assert resolve_color_match({"max_filled_fraction": 0.4})["max_filled_fraction"] == 0.4
    for bad in (1.5, -0.1):
        with pytest.raises(ValueError):
            resolve_color_match({"max_filled_fraction": bad})


def test_infill_mode_is_validated():
    assert resolve_infill_mode("full") == "full"
    with pytest.raises(ValueError):
        resolve_infill_mode("occluded")


# ---------------------------------------------------------------------------
# Stage wiring and defaults
# ---------------------------------------------------------------------------

def _stage_main():
    tree = ast.parse(STAGE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return tree, main


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name]


def _positional_names(call):
    return [a.id if isinstance(a, ast.Name) else None for a in call.args]


def _parents(root):
    parents = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_ifs(node, parents):
    found = []
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            found.append(node)
    return found


def _in_body(call, guard):
    return any(sub is call for stmt in guard.body for sub in ast.walk(stmt))


def test_stage_imports_the_colour_helpers_at_module_level():
    tree = ast.parse(STAGE.read_text())
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "simfoundry.pipeline.upsample_color"
        for alias in node.names
    }

    assert {"color_match_files", "composite_infill_file", "resolve_color_match", "resolve_infill_mode"} <= imported


def test_the_new_settings_are_read_before_the_image_loop():
    _, main = _stage_main()
    loop = next(n for n in main.body if isinstance(n, ast.For) and getattr(n.target, "id", None) == "filename")
    first = {}
    for node in ast.walk(main):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    first[target.id] = min(first.get(target.id, node.lineno), node.lineno)

    for name in ("infill_mode", "infill_occluded", "infill_feather_px", "infill_composites",
                 "color_match_enabled", "color_match_settings", "color_matches"):
        assert first[name] < loop.lineno, name


def test_the_composite_runs_in_the_infill_guard_after_background_removal_and_before_upsampling():
    _, main = _stage_main()
    parents = _parents(main)
    (call,) = _calls(main, "composite_infill_file")
    guards = _enclosing_ifs(call, parents)

    assert _positional_names(call) == ["input_img_padded", "infilled_img_fpath", "infill_feather_px"]
    assert guards and isinstance(guards[0].test, ast.Name) and guards[0].test.id == "infill_occluded"
    assert _in_body(call, guards[0])
    removal = [c for c in _calls(main, "remove_background") if _positional_names(c)[:1] == ["infilled_img_fpath"]]
    assert len(removal) == 1 and removal[0].lineno < call.lineno
    upsample = [
        c for c in _calls(main, "call_image_editing_model")
        if any(k.arg == "task" and isinstance(k.value, ast.Constant) and k.value.value == "upsample" for k in c.keywords)
    ]
    assert len(upsample) == 1 and call.lineno < upsample[0].lineno


def test_the_colour_match_runs_when_enabled_and_not_reverted_between_the_check_and_validity():
    _, main = _stage_main()
    parents = _parents(main)
    (call,) = _calls(main, "color_match_files")
    guard = _enclosing_ifs(call, parents)[0]
    guard_names = {n.id for n in ast.walk(guard.test) if isinstance(n, ast.Name)}
    guard_strings = {n.value for n in ast.walk(guard.test) if isinstance(n, ast.Constant)}

    assert _positional_names(call) == ["input_img_padded", "transparent_fpath", "upsampled_img_fpath", "color_match_settings"]
    assert _in_body(call, guard)
    assert "color_match_enabled" in guard_names
    assert "reverted" in guard_strings
    (check,) = _calls(main, "run_reference_check")
    (validity,) = [
        n for n in ast.walk(main)
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "check_valid" and n.lineno > call.lineno
    ]
    assert check.lineno < call.lineno < validity.lineno


def test_each_record_key_is_written_only_when_its_option_is_on():
    _, main = _stage_main()
    conditionals = [n for n in ast.walk(main) if isinstance(n, ast.IfExp) and isinstance(n.body, ast.Dict)]
    for test_name, key in (("infill_occluded", "infill_composites"), ("color_match_enabled", "color_matches")):
        matches = [
            n for n in conditionals
            if isinstance(n.test, ast.Name) and n.test.id == test_name
            and any(isinstance(k, ast.Constant) and k.value == key for k in n.body.keys)
        ]
        assert len(matches) == 1, key


def test_the_colour_record_keeps_both_definitions_and_the_skip_reason():
    _, main = _stage_main()
    (record,) = [
        n for n in ast.walk(main)
        if isinstance(n, ast.Tuple) and any(isinstance(e, ast.Constant) and e.value == "skipped" for e in n.elts)
    ]
    keys = {e.value for e in record.elts if isinstance(e, ast.Constant)}

    assert {
        "delta_e_before", "delta_e_after", "delta_e_visible_before", "delta_e_visible_after",
        "delta_e_pixel_before", "delta_e_pixel_after", "filled_fraction", "filled_vs_visible_delta_e_before",
        "filled_vs_visible_delta_e_after", "strength", "match_std", "skipped",
    } <= keys


def test_the_stage_code_defaults_match_the_config_defaults():
    # Keyed by the section each read comes from: `enabled` is read from both reference_check and color_match.
    _, main = _stage_main()
    defaults = {}
    for call in ast.walk(main):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "get"
                and len(call.args) > 1 and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[1], ast.Constant)):
            receiver = call.func.value
            section = receiver.attr if isinstance(receiver, ast.Attribute) else getattr(receiver, "id", None)
            defaults[(section, call.args[0].value)] = call.args[1].value

    assert defaults[("s6_upsample", "infill_mode")] == "full"
    assert defaults[("s6_upsample", "infill_feather_px")] == 3
    assert defaults[("color_match_cfg", "enabled")] is False


def test_the_module_defaults_are_full_strength_with_spread_and_no_limit():
    settings = resolve_color_match({})

    assert settings["strength"] == 1.0
    assert settings["match_std"] is True
    assert settings["std_ratio_range"] == (0.5, 2.0)
    assert settings["max_filled_fraction"] is None


def test_the_config_defaults_keep_both_options_off():
    section = OmegaConf.load(CONFIG).s6_upsample
    colour = section.color_match

    assert section.infill is False
    assert section.infill_mode == "full"
    assert section.infill_feather_px == 3
    assert colour.enabled is False
    assert colour.strength == 1.0
    assert colour.match_std is True
    assert list(colour.std_ratio_range) == [0.5, 2.0]
    assert colour.max_filled_fraction is None
    assert resolve_color_match(colour)["max_filled_fraction"] is None


# ---------------------------------------------------------------------------
# Round-1 review: the mutants the review spec kept alive
# ---------------------------------------------------------------------------

def _weight_row_out():
    """A 100 px white crop with a 60 px visible square at [20, 80), over an opaque black infill (feather 3).

    The crop's RGB is white everywhere, so its Lanczos resample is white, and each red value is 255 times the seam
    weight of the row.
    """
    crop = np.zeros((100, 100, 4), dtype=np.uint8)
    crop[..., :3] = 255
    crop[20:80, 20:80, 3] = 255
    infill = np.zeros((100, 100, 4), dtype=np.uint8)
    infill[..., 3] = 255
    out, _ = composite_visible_over_infill(crop, infill, feather_px=3)
    return out


# Red values along row 50 from the closed-form seam weight: a normalised Gaussian (sigma 1.5, radius 6) over the 3 px
# erosion of the square, which starts at x = 20. The 0.5 crossing lies between x = 22 (0.367) and x = 23 (0.633).
EXPECTED_RED = {17: 0, 18: 0, 19: 2, 20: 11, 21: 39, 22: 94, 23: 161, 24: 216, 25: 244, 26: 253, 27: 255, 28: 255, 29: 255}


def test_the_seam_weight_is_pinned_so_a_wider_or_narrower_erosion_is_caught():
    out = _weight_row_out()

    for x, red in EXPECTED_RED.items():
        assert int(out[50, x, 0]) == red, x
    assert (out[50, 29:71, :3] == 255).all()  # exactly 1 from depth 9 to 3 px short of the right edge
    assert (out[50, :17, :3] == 0).all()      # the black infill, beyond the seam


def test_alpha_127_is_not_visible_and_alpha_128_is():
    # Same-size crop and infill, so the alpha reaches the threshold unresampled. The centre is far from the block edge.
    infill = _solid(40, 40, (0, 0, 0))
    for alpha, expected in ((127, 0), (128, 255)):
        crop = _solid(40, 40, (255, 255, 255), alpha=0)
        crop[5:35, 5:35, 3] = alpha
        out, _ = composite_visible_over_infill(crop, infill, feather_px=3)
        assert out[20, 20, 0] == expected, alpha


def test_the_visible_mask_follows_the_bilinear_alpha_not_the_nearest_one():
    # Alternating partial (100) and opaque (255) columns: at 2x, bilinear mixing lifts the partial columns over the
    # threshold while nearest keeps them under it (2480 against 1280 visible pixels at 80 x 80).
    alpha = np.zeros((40, 40), dtype=np.uint8)
    alpha[:, 12:28] = np.where(np.arange(16) % 2 == 0, 100, 255)
    crop = np.zeros((40, 40, 4), dtype=np.uint8)
    crop[..., :3] = 255
    crop[..., 3] = alpha
    _, info = composite_visible_over_infill(crop, _solid(80, 80, (0, 0, 0)), feather_px=3)

    # With an opaque infill the object is the whole 80 x 80 picture, so the fraction is the visible count over 6400.
    bilinear = np.asarray(Image.fromarray(alpha).resize((80, 80), Image.BILINEAR)) > 127
    assert bilinear.sum() == 2480
    assert info["visible_fraction"] == pytest.approx(bilinear.sum() / 6400)


def test_visible_fraction_is_the_visible_count_over_the_union_with_the_infill():
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    infill = _solid(80, 80, PAINTED, alpha=0)
    infill[:, :40, 3] = 255  # the infill is larger than the visible surface
    _, info = composite_visible_over_infill(crop, infill, feather_px=3)

    visible = np.asarray(Image.fromarray(crop[..., 3]).resize((80, 80), Image.BILINEAR)) > 127
    union = visible | (infill[..., 3] > 127)
    assert info["visible_fraction"] == pytest.approx(visible.sum() / union.sum())
    assert 0.0 < info["visible_fraction"] < 1.0


def test_the_aspect_rule_is_relative_so_three_point_oh_against_three_point_oh_five_is_composited():
    crop = _crop(100, 300, (10, 10, 90, 290), CREAM)  # 3.0 to 1
    out, info = composite_visible_over_infill(crop, _solid(100, 305, PAINTED), feather_px=3)  # 3.05, 1.7% off
    assert info["composited"] is True
    assert out.shape == (100, 305, 4)

    _, refused = composite_visible_over_infill(crop, _solid(100, 310, PAINTED), feather_px=3)  # 3.1, 3.3% off
    assert refused["composited"] is False

    thin = _crop(100, 10, (10, 1, 90, 9), CREAM)  # 0.1 to 1
    _, refused_thin = composite_visible_over_infill(thin, _solid(200, 21, PAINTED), feather_px=3)  # 0.105, 5% off
    assert refused_thin["composited"] is False


def test_a_negative_feather_is_refused():
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    with pytest.raises(ValueError):
        composite_visible_over_infill(crop, _solid(80, 80, PAINTED), feather_px=-1)


def test_a_refused_composite_leaves_the_infill_file_byte_for_byte(tmp_path):
    # Saved at compression level 0, so a re-save at the default level would change the bytes.
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    infill = np.dstack([np.random.default_rng(11).integers(0, 256, (120, 80, 3), dtype=np.uint8),
                        np.full((120, 80), 255, dtype=np.uint8)])
    path = tmp_path / "infilled.png"
    Image.fromarray(infill).save(path, compress_level=0)
    before = path.read_bytes()

    info = composite_infill_file(crop, path, 3)

    assert info["composited"] is False
    assert path.read_bytes() == before


def test_strength_zero_returns_the_picture_exactly_because_a_lab_round_trip_drifts():
    rng = np.random.default_rng(12)
    picture = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=bool)
    coefficients = lab_affine_coefficients(picture, mask, picture, mask, True, (0.5, 2.0))

    # The round trip alone moves some pixels by a level, which is why strength 0 needs its own early return.
    assert (_lab_to_rgb_uint8(_rgb_to_lab(picture)) != picture).any()
    assert np.array_equal(apply_lab_affine(picture, coefficients, 0.0), picture)


def test_the_pixel_distance_resamples_the_reference_with_lanczos():
    # The picture is the Lanczos upscale of an 8 px gradient, so the distance is 0 when the reference is resampled the
    # same way. A nearest-neighbour reference is about 5 Lab units away on average.
    reference = np.zeros((8, 8, 3), dtype=np.uint8)
    for x in range(8):
        reference[:, x] = (32 * x, 255 - 32 * x, 128)
    picture = np.asarray(Image.fromarray(reference).resize((64, 64), Image.LANCZOS))
    mask = np.ones((64, 64), dtype=bool)

    assert pixel_delta_e(picture, mask, reference) < 0.5
    nearest = np.asarray(Image.fromarray(reference).resize((64, 64), Image.NEAREST))
    assert float(np.linalg.norm(_rgb_to_lab(picture) - _rgb_to_lab(nearest), axis=-1).mean()) > 2.0


@pytest.mark.parametrize("strength", [0.0, 1.0])
def test_strength_bounds_are_inclusive(strength):
    assert resolve_color_match({"strength": strength})["strength"] == strength


@pytest.mark.parametrize("strength", [-0.1, 1.1])
def test_strength_outside_the_bounds_is_refused(strength):
    with pytest.raises(ValueError):
        resolve_color_match({"strength": strength})


@pytest.mark.parametrize("spread", [(0.5, 0.5), (2.0, 0.5), (0.0, 2.0)])
def test_the_spread_range_needs_zero_below_low_below_high(spread):
    with pytest.raises(ValueError):
        resolve_color_match({"std_ratio_range": list(spread)})


def test_a_limit_of_one_is_allowed_and_above_one_is_refused():
    assert resolve_color_match({"max_filled_fraction": 1.0})["max_filled_fraction"] == 1.0
    with pytest.raises(ValueError):
        resolve_color_match({"max_filled_fraction": 1.1})


def test_the_match_handles_a_crop_and_a_cutout_that_are_not_square(tmp_path):
    # A 40 x 80 crop (h x w) with a visible block, and an 80 x 160 cutout: the block at twice the size, plus a painted
    # strip on the right that the crop did not show. Painted fraction 960 / 4800 = 0.2.
    crop = np.zeros((40, 80, 4), dtype=np.uint8)
    crop[8:32, 10:50] = (*CREAM, 255)
    cutout = np.zeros((80, 160, 4), dtype=np.uint8)
    cutout[16:64, 20:100] = (120, 160, 200, 255)
    cutout[16:64, 120:140] = (*PAINTED, 255)
    transparent_path, rgb_path = _write_pair(tmp_path, crop, cutout, cutout[..., :3].copy())

    report = color_match_files(crop, transparent_path, rgb_path, resolve_color_match({}))

    assert report["skipped"] is None
    assert report["filled_fraction"] == pytest.approx(0.2)
    assert report["delta_e_visible_before"] > 3.0
    assert report["delta_e_visible_after"] < 0.5


def test_only_the_pixels_the_object_covers_take_the_transform_in_both_files(tmp_path):
    crop = _crop(20, 20, (5, 5, 15, 15), CREAM)
    transparent = np.zeros((20, 20, 4), dtype=np.uint8)
    transparent[..., :3] = 255  # a white background at alpha 0
    transparent[4:16, 4:16, :3] = MUTED
    transparent[4:16, 4:16, 3] = 128  # a partial edge around the surface
    transparent[5:15, 5:15, 3] = 255
    transparent_path, rgb_path = _write_pair(tmp_path, crop, transparent, transparent[..., :3].copy())

    color_match_files(crop, transparent_path, rgb_path, resolve_color_match({}))
    matched = np.asarray(Image.open(transparent_path))
    matched_rgb = np.asarray(Image.open(rgb_path))

    background = transparent[..., 3] == 0
    edge = transparent[..., 3] == 128
    assert np.array_equal(matched[..., 3], transparent[..., 3])
    assert np.array_equal(matched[background, :3], transparent[background, :3])
    assert np.array_equal(matched_rgb[background], transparent[background, :3])
    assert (matched[edge, :3] != transparent[edge, :3]).any()
    assert np.array_equal(matched_rgb, matched[..., :3])  # both files got the same transform


# ---------------------------------------------------------------------------
# Rim trim (s6_upsample.infill_rim_trim_px)
# ---------------------------------------------------------------------------

def _ring_case(width):
    """A 20 px visible square in a 40 px crop, with a dark infill band `width` px wide around it, at the same size."""
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    infill = _solid(40, 40, (20, 20, 20), alpha=0)
    infill[10 - width:30 + width, 10 - width:30 + width, 3] = 255
    return crop, infill


def test_zero_rim_trim_returns_the_composite_unchanged_and_writes_nothing(tmp_path):
    crop, infill = _ring_case(width=2)
    composite, _ = composite_visible_over_infill(crop, infill, feather_px=3)
    path = tmp_path / "composite.png"
    Image.fromarray(composite).save(path)
    before = path.read_bytes()

    out, removed = trim_infill_rim(crop, composite, 0)
    record = trim_infill_rim_file(crop, path, 0)

    assert np.array_equal(out, composite)
    assert removed == 0
    assert record == {"rim_trim_px": 0, "rim_trimmed_pixels": 0}
    assert path.read_bytes() == before


def test_a_two_pixel_dark_band_is_gone_at_two_and_its_inner_pixel_survives_at_one():
    crop, infill = _ring_case(width=2)
    composite, _ = composite_visible_over_infill(crop, infill, feather_px=3)
    visible = np.zeros((40, 40), dtype=bool)
    visible[10:30, 10:30] = True
    band = (composite[..., 3] > 127) & ~visible
    assert band.sum() == 176

    two, removed_two = trim_infill_rim(crop, composite, 2)
    assert removed_two == 176
    assert (two[..., 3][band] == 0).all()
    assert (two[..., 3][visible] == 255).all()
    assert np.array_equal(two[..., :3], composite[..., :3])  # colour is never trimmed

    one, removed_one = trim_infill_rim(crop, composite, 1)
    inner = np.zeros((40, 40), dtype=bool)
    inner[9:31, 9:31] = True
    inner &= ~visible
    assert removed_one == 92
    assert (one[..., 3][inner] == 255).all()
    assert (one[..., 3][band & ~inner] == 0).all()
    assert np.array_equal(one[..., :3], composite[..., :3])


def test_the_rim_is_counted_in_infill_pixels_not_crop_pixels():
    # A 40 px crop and an 80 px infill. A 4 px band of infill pixels around the visible square: two trims remove the
    # two outer rings (368 pixels) and leave the two rings nearest the surface (336). The band is 704 pixels.
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    infill = _solid(80, 80, (20, 20, 20), alpha=0)
    infill[16:64, 16:64, 3] = 255
    composite, _ = composite_visible_over_infill(crop, infill, feather_px=3)
    visible = np.zeros((80, 80), dtype=bool)
    visible[20:60, 20:60] = True
    band = (composite[..., 3] > 127) & ~visible
    assert band.sum() == 704

    trimmed, removed = trim_infill_rim(crop, composite, 2)

    assert removed == 368
    assert (trimmed[..., 3][band] == 255).sum() == 336


def test_a_thick_filled_block_loses_only_its_outer_rim_and_stays_joined_to_the_surface():
    crop = _crop(40, 40, (10, 10, 30, 30), CREAM)
    infill = _solid(40, 40, (20, 20, 20), alpha=0)
    infill[10:30, 30:36, 3] = 255  # a 6 px block beyond the surface's right edge
    composite, _ = composite_visible_over_infill(crop, infill, feather_px=3)
    visible = np.zeros((40, 40), dtype=bool)
    visible[10:30, 10:30] = True
    union = composite[..., 3] > 127
    kept = visible | binary_erosion(union, iterations=2)

    trimmed, removed = trim_infill_rim(crop, composite, 2)

    assert removed == int((union & ~kept).sum())
    assert np.array_equal(trimmed[..., 3] > 127, kept)
    assert (trimmed[20, 30:34, 3] == 255).all()  # the block's inner columns stay, joined to the surface
    assert (trimmed[20, 34:36, 3] == 0).all()    # its two outer columns go
    assert (trimmed[10:30, 10:30, 3] == 255).all()


def test_the_surface_at_the_image_edge_is_never_trimmed():
    # Outside the image counts as empty for the erosion, so the surface would be eroded at the border without the OR.
    crop = _crop(40, 40, (0, 0, 20, 20), CREAM)
    composite, _ = composite_visible_over_infill(crop, _solid(40, 40, (20, 20, 20)), feather_px=3)

    trimmed, removed = trim_infill_rim(crop, composite, 2)

    assert (trimmed[0:20, 0:20, 3] == 255).all()
    assert removed > 0  # the infill's own edge along the frame border is trimmed


def test_a_negative_or_fractional_rim_trim_is_refused():
    for bad in (-1, 2.5):
        with pytest.raises(ValueError):
            resolve_rim_trim_px(bad)
    crop, infill = _ring_case(width=2)
    composite, _ = composite_visible_over_infill(crop, infill, feather_px=3)
    with pytest.raises(ValueError):
        trim_infill_rim(crop, composite, -1)


def test_the_trim_record_serialises_to_json(tmp_path):
    crop, infill = _ring_case(width=2)
    composite, _ = composite_visible_over_infill(crop, infill, feather_px=3)
    path = tmp_path / "composite.png"
    Image.fromarray(composite).save(path)

    record = trim_infill_rim_file(crop, path, 2)

    assert record == {"rim_trim_px": 2, "rim_trimmed_pixels": 176}
    assert json.loads(json.dumps(record)) == record


def test_the_rim_trim_is_off_by_default_in_the_config_and_the_stage():
    section = OmegaConf.load(CONFIG).s6_upsample
    assert section.infill_rim_trim_px == 0
    assert resolve_rim_trim_px(section.infill_rim_trim_px) == 0

    _, main = _stage_main()
    (read,) = [n for n in ast.walk(main) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "resolve_rim_trim_px"]
    default = read.args[0].args[1]  # cfg.s6_upsample.get("infill_rim_trim_px", 0)
    assert isinstance(default, ast.Constant) and default.value == 0


# ---------------------------------------------------------------------------
# Stage wiring, read from the source
# ---------------------------------------------------------------------------

def test_the_composite_switch_needs_both_the_infill_switch_and_occluded_only_mode():
    _, main = _stage_main()
    (assign,) = [
        n for n in main.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "infill_occluded" for t in n.targets)
    ]
    code = compile(ast.Expression(body=assign.value), "infill_occluded", "eval")
    for infill, mode, expected in [(True, "full", False), (True, "occluded_only", True),
                                   (False, "occluded_only", False), (False, "full", False)]:
        namespace = {"bool": bool, "infill_mode": mode,
                     "cfg": SimpleNamespace(s6_upsample=SimpleNamespace(infill=infill))}
        assert eval(code, {}, namespace) is expected, (infill, mode)


def test_the_composite_and_colour_records_are_keyed_by_the_image_name():
    _, main = _stage_main()
    for name in ("infill_composites", "color_matches"):
        subscripts = [n for n in ast.walk(main) if isinstance(n, ast.Subscript) and getattr(n.value, "id", None) == name]
        assert subscripts, name
        assert all(isinstance(s.slice, ast.Name) and s.slice.id == "img_name" for s in subscripts), name


def test_each_setting_is_read_from_its_own_key_in_the_stage():
    _, main = _stage_main()
    reads = {}
    for node in main.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        keys = {
            call.args[0].value
            for call in ast.walk(node.value)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "get"
            and call.args and isinstance(call.args[0], ast.Constant)
            and isinstance(call.func.value, ast.Attribute) and call.func.value.attr == "s6_upsample"
        }
        if keys:
            reads[node.targets[0].id] = keys
    assert reads["color_match_cfg"] == {"color_match"}
    assert reads["infill_mode"] == {"infill_mode"}
    assert reads["infill_feather_px"] == {"infill_feather_px"}
    assert reads["infill_rim_trim_px"] == {"infill_rim_trim_px"}


def test_the_rim_trim_runs_only_on_a_composite_that_was_made_and_is_recorded_beside_it():
    _, main = _stage_main()
    parents = _parents(main)
    (call,) = _calls(main, "trim_infill_rim_file")
    (composite,) = _calls(main, "composite_infill_file")
    guard = _enclosing_ifs(call, parents)[0]

    assert _in_body(call, guard)
    assert any(isinstance(n, ast.Constant) and n.value == "composited" for n in ast.walk(guard.test))
    assert composite.lineno < call.lineno
    strings = {n.value for n in ast.walk(main) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {"rim_trim_px", "rim_trimmed_pixels"} <= strings


def test_the_rim_record_is_merged_into_the_composite_record_of_the_same_image():
    _, main = _stage_main()
    updates = [
        c for c in ast.walk(main)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "update"
        and isinstance(c.func.value, ast.Subscript) and getattr(c.func.value.value, "id", None) == "infill_composites"
    ]
    assert len(updates) == 1
    assert isinstance(updates[0].func.value.slice, ast.Name) and updates[0].func.value.slice.id == "img_name"
