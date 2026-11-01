# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the appearance alignment (`bg_align`) and its wiring in the 6th stage.

Everything is synthetic: a counter top with a brightness boundary, four squares and a sunken patch outside the band. The
source is the same pattern sampled at other points, with noise, and moved by a known rigid transform. No GPU, open3d or
real scene is needed. The synthetic runs use a 0.3 m margin where the default is 1.2 m, only to keep the grids small.
"""

import ast
import json
import pathlib
import subprocess

import numpy as np
import pytest
import trimesh
import yaml
from PIL import Image

from simfoundry.pipeline import bg_align
from simfoundry.pipeline import depth_mesh

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGE_REL = "scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/6_bridge_bg_splat_to_og.py"
STAGE = ROOT / STAGE_REL
CONFIG = ROOT / "scripts/cfg/auto_bg.yaml"
README = ROOT / "scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/README.md"
# The commit this branch starts from. The stage's other modes must still read as they did there.
BASE = "8c54932"

TEST_SETTINGS = {"margin_m": 0.3}

# (x0, y0, x1, y1, brightness) in scene metres. The boundary at x = 0.1 and the squares are placed asymmetrically.
ASYMMETRIC = (
    (0.25, 0.10, 0.40, 0.25, 40.0),
    (0.55, 0.30, 0.70, 0.45, 250.0),
    (0.85, 0.12, 0.95, 0.22, 150.0),
    (0.70, 0.02, 0.78, 0.08, 20.0),
)
# Symmetric under a half turn about (0.6, 0.3): every square has a mirror image with the same brightness.
SYMMETRIC = (
    (0.25, 0.10, 0.40, 0.25, 40.0),
    (0.80, 0.35, 0.95, 0.50, 40.0),
    (0.55, 0.30, 0.70, 0.45, 250.0),
    (0.50, 0.15, 0.65, 0.30, 250.0),
)
PROBES = np.array([[0.325, 0.175, 0.0], [0.625, 0.375, 0.0], [0.9, 0.17, 0.0], [0.74, 0.05, 0.0]])

# (seed, yaw, tilt, tilt axis, translation): the known transform from the scene to the DA3 world.
CASES = [
    (1, 37.3, 25.0, 40.0, (0.31, -0.22, 0.15)),
    (2, 120.0, 10.0, 200.0, (-0.2, 0.25, 0.1)),
    (3, 300.0, 35.0, 300.0, (0.05, 0.1, -0.2)),
]


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _rodrigues(axis, angle):
    axis = _unit(axis)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _rz(deg):
    angle = np.radians(deg)
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def _counter(x, y, symmetric=False):
    if symmetric:
        value = np.full(np.shape(x), 120.0)
        squares = SYMMETRIC
    else:
        value = np.where(x < 0.1, 200.0, 90.0)
        squares = ASYMMETRIC
    for x0, y0, x1, y1, level in squares:
        value = np.where((x >= x0) & (x < x1) & (y >= y0) & (y < y1), level, value)
    return value


def _forward(yaw, tilt, axis, translation):
    """The known transform from the scene frame to the DA3 world: a tilted plane, a yaw, and a translation."""
    rotation = _rodrigues((np.cos(np.radians(axis)), np.sin(np.radians(axis)), 0.0), np.radians(tilt)) @ _rz(yaw)
    return rotation, np.asarray(translation, dtype=np.float64)


def _scene(symmetric=False):
    """The counter at z = 0 on a 5 mm grid, and a sunken patch at z = -0.08 that the band must ignore."""
    gx, gy = np.meshgrid(np.arange(0.0, 1.2, 0.005), np.arange(0.0, 0.6, 0.005), indexing="ij")
    x, y = gx.ravel(), gy.ravel()
    points = np.column_stack([x, y, np.zeros_like(x)])
    colours = np.repeat(_counter(x, y, symmetric)[:, None], 3, axis=1)
    sx, sy = np.meshgrid(np.arange(0.4, 0.6, 0.005), np.arange(0.1, 0.3, 0.005), indexing="ij")
    sunk = np.column_stack([sx.ravel(), sy.ravel(), np.full(sx.size, -0.08)])
    points = np.vstack([points, sunk])
    colours = np.vstack([colours, np.full((len(sunk), 3), 255.0)])
    return points, colours


def _source(forward, rng, symmetric=False, x_max=1.2, n=30000):
    """The same pattern sampled at other points, with 1 mm height noise and 2-level brightness noise, in the DA3 world."""
    rotation, translation = forward
    x = rng.uniform(0.0, x_max, n)
    y = rng.uniform(0.0, 0.6, n)
    scene = np.column_stack([x, y, rng.normal(0.0, 0.001, n)])
    brightness = np.clip(_counter(x, y, symmetric) + rng.normal(0.0, 2.0, n), 0.0, 255.0)
    return scene @ rotation.T + translation, np.repeat(brightness[:, None], 3, axis=1)


def _cameras(forward, rng):
    """Four camera centres half a metre above the counter, on the side the normal points to."""
    rotation, translation = forward
    xy = rng.uniform(0.0, 1.0, (4, 2))
    above = np.column_stack([xy[:, 0], xy[:, 1], np.full(4, 0.5)])
    return above @ rotation.T + translation


def _align(forward, seed=1, symmetric=False, x_max=1.2, settings=None, source_colour=None):
    rng = np.random.default_rng(seed)
    target, target_rgb = _scene(symmetric)
    source, source_rgb = _source(forward, rng, symmetric, x_max=x_max)
    if source_colour is not None:
        source_rgb = source_colour(rng, len(source))
    cameras = _cameras(forward, rng)
    seed_points = source[rng.choice(len(source), 5000, replace=False)]
    return bg_align.align_clouds(
        target, target_rgb, source, source_rgb, cameras, settings or TEST_SETTINGS, seed_points=seed_points,
    )


def _errors(record, forward):
    """The worst position error over the four probes (metres) and the tilt of the plane's normal (degrees)."""
    rotation, translation = forward
    matrix = np.asarray(record["M_src_to_scene"])
    da3 = PROBES @ rotation.T + translation
    estimate = da3 @ matrix[:3, :3].T + matrix[:3, 3]
    position = np.linalg.norm(estimate - PROBES, axis=1).max()
    normal = matrix[:3, :3] @ (rotation @ np.array([0.0, 0.0, 1.0]))
    tilt = np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))
    return position, tilt


def _plane_cloud(rng, normal, offset, n=4000, outliers=0.4, noise=0.001):
    """Points on the plane n . p + offset = 0, with height noise, and a share of random outliers in a box."""
    basis_a = _unit(np.cross(normal, [1.0, 0.0, 0.0]))
    basis_b = np.cross(normal, basis_a)
    base = -offset * normal
    a, b = rng.uniform(-0.5, 0.5, (2, n))
    on_plane = base + a[:, None] * basis_a + b[:, None] * basis_b + rng.normal(0.0, noise, n)[:, None] * normal
    count = int(n * outliers / (1.0 - outliers))
    scatter = rng.uniform(-0.6, 0.6, (count, 3))
    return np.vstack([on_plane, scatter])


# --------------------------------------------------------------------------------------------------------------------
# The whole alignment on the synthetic counter
# --------------------------------------------------------------------------------------------------------------------

RECORD_KEYS = {
    "schema", "accepted", "reasons", "warnings", "settings", "inputs", "plane", "n_target_points", "n_source_points",
    "peaks", "yaw_curve", "second_peak_ncc", "peak_gap", "yaw_deg", "shift_m", "ncc", "overlap", "scale", "scale_best",
    "scale_curve", "height_residual_m", "M_src_to_scene", "sidecar_written", "stale_sidecar_present",
}


@pytest.mark.parametrize("seed, yaw, tilt, axis, translation", CASES)
def test_the_counter_is_recovered_within_half_a_centimetre_and_half_a_degree(seed, yaw, tilt, axis, translation):
    forward = _forward(yaw, tilt, axis, translation)
    record = _align(forward, seed)

    assert record["accepted"] and record["reasons"] == []
    position, tilt_error = _errors(record, forward)
    assert position <= 0.005
    assert tilt_error <= 0.5
    assert record["scale"] == 1.0
    assert 0.99 <= record["scale_best"] <= 1.01
    assert record["n_target_points"] == 28800  # the sunken patch is outside the band and must not be counted
    assert record["settings"] == bg_align.resolve_settings(TEST_SETTINGS)
    assert set(record) == RECORD_KEYS
    json.dumps(record, sort_keys=True)


def test_the_same_inputs_give_the_same_record_byte_for_byte():
    forward = _forward(*CASES[0][1:])
    first = json.dumps(_align(forward, 1), sort_keys=True)
    second = json.dumps(_align(forward, 1), sort_keys=True)
    assert first == second


def test_scattered_points_have_no_gravity_plane_and_no_matrix():
    rng = np.random.default_rng(9)
    seed_cube = rng.uniform(0.0, 1.0, (5000, 3))
    forward = _forward(*CASES[0][1:])
    target, target_rgb = _scene()
    source, source_rgb = _source(forward, rng)
    record = bg_align.align_clouds(
        target, target_rgb, source, source_rgb, _cameras(forward, rng), TEST_SETTINGS, seed_points=seed_cube,
    )
    assert record["plane"] is None
    assert record["accepted"] is False
    assert record["reasons"] == ["no_gravity_plane"]
    assert record["M_src_to_scene"] is None


def test_a_source_covering_less_than_half_the_counter_is_refused_for_no_overlap():
    forward = _forward(*CASES[0][1:])
    record = _align(forward, 1, x_max=0.5)
    assert record["accepted"] is False
    assert "no_overlap" in record["reasons"]
    assert record["M_src_to_scene"] is None


def test_brightness_that_is_pure_noise_is_refused_for_low_ncc():
    forward = _forward(*CASES[0][1:])

    def noise(rng, n):
        return np.repeat(rng.uniform(0.0, 255.0, (n, 1)), 3, axis=1)

    record = _align(forward, 1, source_colour=noise)
    assert record["accepted"] is False
    assert "low_ncc" in record["reasons"]
    assert record["M_src_to_scene"] is None


def test_a_counter_with_a_half_turn_symmetry_is_refused_as_ambiguous():
    forward = _forward(*CASES[0][1:])
    record = _align(forward, 1, symmetric=True)
    assert record["accepted"] is False
    assert "ambiguous" in record["reasons"]
    assert record["M_src_to_scene"] is None


# --------------------------------------------------------------------------------------------------------------------
# Gravity plane
# --------------------------------------------------------------------------------------------------------------------

def test_the_gravity_plane_is_recovered_and_its_normal_turns_to_the_cameras():
    rng = np.random.default_rng(4)
    normal = _unit([0.2, -0.3, 1.0])
    point = np.array([0.1, -0.2, 0.3])
    offset = -normal @ point
    points = _plane_cloud(rng, normal, offset)
    above = point + 0.5 * normal + rng.normal(0.0, 0.01, (3, 3))
    below = point - 0.5 * normal + rng.normal(0.0, 0.01, (3, 3))

    up = bg_align.fit_gravity_plane(points, above, {})
    down = bg_align.fit_gravity_plane(points, below, {})

    assert np.abs(np.array(up["normal"]) - normal).max() <= 1e-3
    assert abs(up["offset"] - offset) <= 1e-3
    assert np.abs(np.array(down["normal"]) + normal).max() <= 1e-3
    assert abs(down["offset"] + offset) <= 1e-3
    assert up["share"] > 0.5 and up["rms_m"] <= 0.002 and up["n_points"] == len(points)


def test_a_different_seed_finds_the_same_plane():
    rng = np.random.default_rng(5)
    normal = _unit([-0.1, 0.25, 1.0])
    points = _plane_cloud(rng, normal, 0.0)
    cameras = np.array([[0.0, 0.0, 0.5]])
    zero = bg_align.fit_gravity_plane(points, cameras, {"plane_seed": 0})
    five = bg_align.fit_gravity_plane(points, cameras, {"plane_seed": 5})
    assert np.abs(np.array(zero["normal"]) - np.array(five["normal"])).max() <= 1e-3


def test_the_seeded_ransac_repeats_exactly_for_the_same_settings():
    rng = np.random.default_rng(6)
    normal = _unit([0.3, 0.1, 1.0])
    points = _plane_cloud(rng, normal, 0.05, n=3000, outliers=0.6, noise=0.003)
    cameras = np.array([[0.0, 0.0, 0.5]])
    first = bg_align.fit_gravity_plane(points, cameras, {"plane_iterations": 200})
    second = bg_align.fit_gravity_plane(points, cameras, {"plane_iterations": 200})
    assert first is not None
    assert first == second


def test_a_cloud_without_a_plane_gives_none():
    rng = np.random.default_rng(7)
    assert bg_align.fit_gravity_plane(rng.uniform(0.0, 1.0, (5000, 3)), np.array([[0.5, 0.5, 2.0]]), {}) is None
    assert bg_align.fit_gravity_plane(np.zeros((2, 3)), np.array([[0.0, 0.0, 1.0]]), {}) is None


def test_the_gravity_transform_puts_the_plane_on_z_zero_with_its_normal_up_as_a_rigid_motion():
    rng = np.random.default_rng(8)
    for normal in ([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.3, -0.5, 0.8]):
        unit = _unit(normal)
        offset = 0.42
        transform = bg_align.gravity_transform(normal, offset)
        rotation = transform[:3, :3]
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        # The plane is normal . p + offset = 0 with the normal as given, so its point nearest the origin is -offset n / |n|^2.
        origin_point = -offset * np.asarray(normal, dtype=np.float64) / np.dot(normal, normal)
        assert np.allclose(bg_align._apply(transform, origin_point), 0.0, atol=1e-12)
        draws = rng.normal(size=(20, 3))
        in_plane = origin_point + draws - (draws @ unit)[:, None] * unit
        assert np.allclose(bg_align._apply(transform, in_plane)[:, 2], 0.0, atol=1e-12)
        assert np.allclose(rotation @ unit, [0.0, 0.0, 1.0], atol=1e-12)


# --------------------------------------------------------------------------------------------------------------------
# Search pieces
# --------------------------------------------------------------------------------------------------------------------

def test_top_view_map_is_the_mean_per_cell_and_drops_what_is_outside_or_not_finite():
    rng = np.random.default_rng(12)
    xy = rng.uniform(-0.2, 1.0, (500, 2))
    values = rng.uniform(0.0, 255.0, 500)
    xy[3] = [np.nan, 0.1]
    values[7] = np.nan
    origin, shape, cell = np.array([0.0, 0.0]), (12, 9), 0.1

    expected = np.full(shape, np.nan)
    sums, counts = {}, {}
    for (x, y), value in zip(xy, values):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(value)):
            continue
        i, j = int(np.floor((x - origin[0]) / cell)), int(np.floor((y - origin[1]) / cell))
        if 0 <= i < shape[0] and 0 <= j < shape[1]:
            sums[(i, j)] = sums.get((i, j), 0.0) + value
            counts[(i, j)] = counts.get((i, j), 0) + 1
    for key in sums:
        expected[key] = sums[key] / counts[key]

    got = bg_align.top_view_map(xy, values, origin, shape, cell)
    np.testing.assert_allclose(got, expected, equal_nan=True, rtol=1e-12)


def test_masked_ncc_is_the_plain_correlation_of_the_overlap_and_shifts_the_right_way():
    rng = np.random.default_rng(13)
    target = rng.normal(size=(20, 20))
    source = rng.normal(size=(20, 20))
    target[0, 0] = np.nan
    di, dj = 3, -2
    # target[a, b] pairs with source[a2, b2]: rows 3.. with rows ..16, columns ..17 with columns 2...
    paired_t = target[3:, :18].ravel()
    paired_s = source[:17, 2:].ravel()
    keep = np.isfinite(paired_t) & np.isfinite(paired_s)
    expected = np.corrcoef(paired_t[keep], paired_s[keep])[0, 1]

    score, overlap = bg_align.masked_ncc(target, source, di, dj, 1)
    assert score == pytest.approx(expected, abs=1e-6)
    assert overlap == int(keep.sum())

    # A map that is another map shifted by (2, 1) matches at (2, 1), not at (-2, -1).
    shifted = np.full((20, 20), np.nan)
    shifted[2:, 1:] = source[:-2, :-1]
    assert bg_align.masked_ncc(shifted, source, 2, 1, 1)[0] == pytest.approx(1.0, abs=1e-6)
    assert bg_align.masked_ncc(shifted, source, -2, -1, 1)[0] < 0.9


def test_masked_ncc_takes_the_overlap_bound_inclusively_and_refuses_a_shift_past_the_map():
    rng = np.random.default_rng(14)
    target = rng.normal(size=(10, 10))
    source = rng.normal(size=(10, 10))
    overlap = bg_align.masked_ncc(target, source, 0, 0, 1)[1]
    assert overlap == 100
    assert bg_align.masked_ncc(target, source, 0, 0, 100) is not None
    assert bg_align.masked_ncc(target, source, 0, 0, 101) is None
    assert bg_align.masked_ncc(target, source, 10, 0, 1) is None
    assert bg_align.masked_ncc(target, source, 0, -10, 1) is None


def test_luminance_is_the_weighted_sum_of_the_channels():
    colours = np.array([[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0], [10.0, 20.0, 30.0]])
    expected = [0.299 * 255, 0.587 * 255, 0.114 * 255, 0.299 * 10 + 0.587 * 20 + 0.114 * 30]
    np.testing.assert_allclose(bg_align.luminance(colours), expected, rtol=1e-12)


def test_the_fft_peak_past_half_the_grid_is_a_negative_shift():
    assert bg_align._signed_shift(3, 10) == 3
    assert bg_align._signed_shift(5, 10) == -5
    assert bg_align._signed_shift(9, 10) == -1


def _texture_points(seed=15, half=0.15, step=0.005):
    rng = np.random.default_rng(seed)
    blocks = rng.uniform(0.0, 255.0, (60, 60))
    gx, gy = np.meshgrid(np.arange(0.0, 2 * half, step), np.arange(0.0, 2 * half, step), indexing="ij")
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    values = blocks[np.floor(xy[:, 0] / 0.01).astype(int), np.floor(xy[:, 1] / 0.01).astype(int)]
    return xy, values


@pytest.mark.parametrize("yaw", [0.0, 30.0])
def test_the_search_recovers_a_known_rotation_and_translation_on_a_small_patch(yaw):
    translation = np.array([0.04, -0.02])
    target, target_lum = _texture_points()
    # A source that is the target moved back: rotate by -yaw and shift by -translation, then the search has to undo it.
    source = (target - translation) @ _rotation_about_z(-yaw).T

    result = bg_align.search_alignment(target, target_lum, source, target_lum, {"margin_m": 0.05})

    assert result["yaw_deg"] == pytest.approx(yaw, abs=0.2)
    assert np.abs(np.array(result["shift_xy"]) - translation).max() <= 0.005


def _rotation_about_z(deg):
    return _rz(deg)[:2, :2]


def test_the_coarse_search_steps_two_cells_across_plus_and_minus_ten_then_refines_by_one():
    calls = []
    real = bg_align.masked_ncc

    def recorder(target_map, source_map, di, dj, minimum):
        calls.append((di, dj))
        return real(target_map, source_map, di, dj, minimum)

    target, target_lum = _texture_points()
    bg_align.masked_ncc = recorder
    try:
        bg_align.search_alignment(target, target_lum, target, target_lum, {"margin_m": 0.05, "yaw_step_deg": 360.0})
    finally:
        bg_align.masked_ncc = real

    coarse, refine = calls[:121], calls[121:146]
    rows = sorted({di for di, _ in coarse})
    cols = sorted({dj for _, dj in coarse})
    assert len(rows) == len(cols) == 11
    assert set(np.diff(rows)) == {2} and set(np.diff(cols)) == {2}
    refine_rows = sorted({di for di, _ in refine})
    refine_cols = sorted({dj for _, dj in refine})
    assert len(refine_rows) == len(refine_cols) == 5
    assert set(np.diff(refine_rows)) == {1} and set(np.diff(refine_cols)) == {1}


def test_second_peak_is_the_highest_local_maximum_other_than_the_best():
    def gauss(yaw, centre, sigma, height):
        gap = min(abs(yaw - centre) % 360.0, 360.0 - abs(yaw - centre) % 360.0)
        return height * np.exp(-gap ** 2 / (2 * sigma ** 2))

    single = {float(y): gauss(y, 0.0, 6.0, 0.95) for y in range(360)}
    assert bg_align.second_peak_score(single, 0.0, 10.0) == -1.0

    two = {float(y): max(gauss(y, 0.0, 6.0, 0.95), gauss(y, 180.0, 6.0, 0.83)) for y in range(360)}
    assert bg_align.second_peak_score(two, 0.0, 10.0) == pytest.approx(0.83)

    missing = {y: score for y, score in two.items() if y not in range(1, 10)}
    assert bg_align.second_peak_score(missing, 0.0, 10.0) == pytest.approx(0.83)

    # The best sits at 355 and a shoulder at 0 is higher than the true second peak, but the window crosses the
    # 0/360 boundary, so that shoulder is not a peak and the second peak is the 175 one.
    wrap = {float(y): max(gauss(y, 355.0, 8.0, 0.95), gauss(y, 175.0, 3.0, 0.60)) for y in range(360)}
    assert wrap[0.0] > 0.6
    assert bg_align.second_peak_score(wrap, 355.0, 10.0) == pytest.approx(0.60, abs=1e-9)

    # A window of zero makes every yaw a peak, so the broad curve has a near-zero gap: that is what the window stops.
    assert bg_align.second_peak_score(single, 0.0, 0.0) > 0.9


# --------------------------------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------------------------------

def test_the_defaults_are_the_settings_and_path_keys_are_ignored():
    assert bg_align.resolve_settings(None) == bg_align.DEFAULT_SETTINGS
    resolved = bg_align.resolve_settings({"seed_ply": "/somewhere/seed.ply", "scene_mesh": None, "cell_m": 0.02})
    assert resolved["cell_m"] == 0.02
    assert "seed_ply" not in resolved


@pytest.mark.parametrize(
    "key, value",
    [
        ("cell_m", float("nan")),
        ("cell_m", -0.01),
        ("cell_m", 0.0),
        ("cell_m", True),
        ("cell_m", "0.01"),
        ("cell_m", None),
        ("coarse_search_cells", 2.5),
        ("min_overlap", 1.5),
        ("pixel_stride", 0),
        ("min_peak_gap", -0.1),
    ],
)
def test_a_value_that_is_not_a_finite_number_in_range_is_refused(key, value):
    with pytest.raises(ValueError, match=f"s6_bridge.appearance.{key}"):
        bg_align.resolve_settings({key: value})


def test_an_unknown_key_is_an_error_not_a_default():
    with pytest.raises(ValueError, match="unknown keys"):
        bg_align.resolve_settings({"cell_size": 0.01})


# --------------------------------------------------------------------------------------------------------------------
# Input helpers
# --------------------------------------------------------------------------------------------------------------------

def test_da3_points_come_back_from_the_depth_with_world_to_camera_extrinsics(tmp_path):
    rng = np.random.default_rng(16)
    frames, height, width = 2, 6, 8
    extrinsics = np.zeros((frames, 3, 4), dtype=np.float32)
    intrinsics = np.zeros((frames, 3, 3), dtype=np.float32)
    depth = rng.uniform(1.0, 2.0, (frames, height, width)).astype(np.float32)
    depth[0, 1, 2] = 0.0
    depth[1, 4, 5] = np.nan
    image = rng.integers(0, 256, (frames, height, width, 3), dtype=np.uint8)
    for k in range(frames):
        basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        extrinsics[k, :, :3] = basis
        extrinsics[k, :, 3] = rng.normal(0.0, 0.3, 3)
        intrinsics[k] = [[5.0, 0.0, 3.5], [0.0, 5.0, 2.5], [0.0, 0.0, 1.0]]
    path = tmp_path / "results.npz"
    np.savez(path, extrinsics=extrinsics, intrinsics=intrinsics, depth=depth, image=image)

    points, colours, centres = bg_align.da3_world_points(path, stride=1)

    expected_points, expected_colours = [], []
    for k in range(frames):
        rotation = extrinsics[k, :, :3].astype(np.float64)
        translation = extrinsics[k, :, 3].astype(np.float64)
        f, cx, cy = 5.0, 3.5, 2.5
        for v in range(height):
            for u in range(width):
                z = float(depth[k, v, u])
                if not (np.isfinite(z) and z > 0):
                    continue
                cam = np.array([(u - cx) / f * z, (v - cy) / f * z, z])
                expected_points.append(np.linalg.inv(rotation) @ (cam - translation))
                expected_colours.append(image[k, v, u])
    np.testing.assert_allclose(points, np.array(expected_points), atol=1e-6)
    np.testing.assert_array_equal(colours, np.array(expected_colours, dtype=float))
    for k in range(frames):
        rotation = extrinsics[k, :, :3].astype(np.float64)
        translation = extrinsics[k, :, 3].astype(np.float64)
        assert np.allclose(centres[k], -np.linalg.inv(rotation) @ translation, atol=1e-6)


def _write_textured_glb(path, z_up, uv, texture):
    vertices = z_up @ depth_mesh.WORLD_TO_GLTF[:3, :3].T + depth_mesh.WORLD_TO_GLTF[:3, 3]
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=Image.fromarray(texture.astype(np.uint8)))
    trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], visual=visual, process=False).export(str(path))


def test_mesh_points_go_back_to_z_up_and_take_the_nearest_texel_with_v_flipped(tmp_path):
    z_up = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    uv = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
    texture = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]])
    path = tmp_path / "background_mesh.glb"
    _write_textured_glb(path, z_up, uv, texture)
    png = tmp_path / "background_texture.png"
    Image.fromarray(texture.astype(np.uint8)).save(png)

    points, colours = bg_align.scene_points_from_background_mesh(path, png)

    order = np.lexsort((points[:, 1], points[:, 0]))
    np.testing.assert_allclose(points[order], z_up[np.lexsort((z_up[:, 1], z_up[:, 0]))], atol=1e-6)
    expected = {(0.0, 0.0): [0, 0, 255], (1.0, 0.0): [255, 255, 255], (1.0, 1.0): [0, 255, 0], (0.0, 1.0): [255, 0, 0]}
    for point, colour in zip(points, colours):
        assert colour.tolist() == expected[(round(point[0]), round(point[1]))]


def test_a_mesh_without_uv_coordinates_is_refused(tmp_path):
    path = tmp_path / "plain.glb"
    trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False).export(str(path))
    png = tmp_path / "unused.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(png)
    with pytest.raises(ValueError, match="UV"):
        bg_align.scene_points_from_background_mesh(path, png)


# --------------------------------------------------------------------------------------------------------------------
# Config, README and stage wiring
# --------------------------------------------------------------------------------------------------------------------

def test_the_config_appearance_block_has_the_module_defaults():
    section = yaml.safe_load(CONFIG.read_text())["s6_bridge"]["appearance"]
    assert set(section) == set(bg_align.DEFAULT_SETTINGS) | set(bg_align.PATH_KEYS)
    for key, value in bg_align.DEFAULT_SETTINGS.items():
        assert section[key] == value, key
    for key in bg_align.PATH_KEYS:
        assert section[key] is None, key


def test_the_judgement_thresholds_are_marked_provisional_in_the_config():
    lines = CONFIG.read_text().splitlines()
    for key in ("min_plane_share", "min_ncc", "min_peak_gap", "peak_window_deg", "scale_warn", "scale_min_gain"):
        line = next(text for text in lines if text.strip().startswith(f"{key}:"))
        assert "provisional" in line, key


def test_the_readme_section_states_the_switch_its_limits_and_the_keys():
    text = README.read_text()
    assert "## Cross-capture alignment (`bridge_mode=appearance`)" in text
    section = text.split("## Cross-capture alignment (`bridge_mode=appearance`)", 1)[1].split("\n## ", 1)[0]
    for phrase in (
        "off by default", "one scene", "counter top", "scale", "provisional",
        "min_ncc", "min_peak_gap", "min_plane_share", "peak_window_deg", "align.json", "pose.json",
    ):
        assert phrase in section, phrase
    assert "recommend" not in section.lower()


def _stage_tree(source=None):
    return ast.parse(source if source is not None else STAGE.read_text())


def _mode_literal(node):
    """The string that an `if mode == "..."` test compares against, or None for any other node."""
    if not isinstance(node, ast.If):
        return None
    test = node.test
    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "mode"
            and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq) and isinstance(test.comparators[0], ast.Constant)):
        return test.comparators[0].value
    return None


def _chain(tree):
    """The `if single / elif umeyama / elif appearance` chain, keyed by mode."""
    node = next(n for n in ast.walk(tree) if _mode_literal(n) == "single")
    branches = {}
    while isinstance(node, ast.If):
        branches[_mode_literal(node)] = node
        node = node.orelse[0] if node.orelse and isinstance(node.orelse[0], ast.If) else None
    return branches


def _auto(tree):
    return next(n for n in ast.walk(tree) if _mode_literal(n) == "auto")


def _unparse(statements):
    return "\n".join(ast.unparse(statement) for statement in statements)


def _base_source():
    result = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{BASE}:{STAGE_REL}"], capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"the base commit {BASE} is not available here: {result.stderr.strip()}")
    return result.stdout


def test_the_appearance_mode_is_one_branch_and_the_other_modes_read_as_they_did_at_the_base():
    tree = _stage_tree()
    base = _stage_tree(_base_source())
    chain, base_chain = _chain(tree), _chain(base)

    assert set(chain) == {"single", "umeyama", "appearance"}
    for value in ("single", "umeyama"):
        assert _unparse(chain[value].body) == _unparse(base_chain[value].body), value
    assert _unparse(chain["appearance"].orelse) == _unparse(base_chain["umeyama"].orelse)
    assert _unparse(_auto(tree).body) == _unparse(_auto(base).body)
    assert "appearance" not in _unparse(_auto(tree).body)


def test_the_sidecar_payload_and_its_write_read_as_they_did_at_the_base():
    tree = _stage_tree()
    base = _stage_tree(_base_source())

    def payload(root):
        return next(n for n in ast.walk(root) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "sidecar_payload" for t in n.targets))

    def sidecar_write(root):
        return next(n for n in ast.walk(root) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func) == "sidecar.write_text")

    assert ast.unparse(payload(tree).value) == ast.unparse(payload(base).value)
    assert ast.unparse(sidecar_write(tree)) == ast.unparse(sidecar_write(base))


def _imports_bg_align(node):
    return isinstance(node, ast.ImportFrom) and any(alias.name == "bg_align" for alias in node.names)


def test_the_appearance_module_is_imported_only_inside_its_branch():
    tree = _stage_tree()
    assert not [node for node in tree.body if _imports_bg_align(node)]
    assert any(_imports_bg_align(node) for node in ast.walk(_chain(tree)["appearance"]))


def test_the_alignment_is_called_with_the_seed_as_a_keyword_and_the_record_is_written_before_any_refusal():
    tree = _stage_tree()
    appearance = _chain(tree)["appearance"]
    calls = [node for node in ast.walk(appearance) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "bg_align.align_clouds"]
    assert len(calls) == 1
    assert [keyword.arg for keyword in calls[0].keywords] == ["seed_points"]

    rejection = next(node for node in ast.walk(appearance)
                     if isinstance(node, ast.If) and "accepted" in ast.unparse(node.test))
    exit_line = min(node.lineno for node in ast.walk(rejection)
                    if isinstance(node, ast.Call) and ast.unparse(node.func) == "sys.exit")
    record_line = min(node.lineno for node in ast.walk(appearance)
                      if isinstance(node, ast.Call) and ast.unparse(node.func) == "align_record_path.write_text")
    sidecar_line = min(node.lineno for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and ast.unparse(node.func) == "sidecar.write_text")
    assert record_line < exit_line
    assert record_line < sidecar_line


def test_the_alignment_record_key_is_added_only_in_the_appearance_mode():
    tree = _stage_tree()
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                           and t.slice.value == "alignment_record" for t in node.targets)]
    assert len(assignments) == 1
    appearance_ifs = [node for node in ast.walk(tree) if _mode_literal(node) == "appearance"]
    assert any(assignments[0] in list(ast.walk(node)) for node in appearance_ifs)


def test_the_stale_sidecar_flag_comes_from_a_file_check():
    tree = _stage_tree()
    targets = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                       and t.slice.value == "stale_sidecar_present" for t in node.targets)]
    assert len(targets) == 1
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "exists"
               for node in ast.walk(targets[0].value))


def test_the_unknown_mode_still_refuses_with_the_same_message():
    assert "Unknown bridge mode" in STAGE.read_text()
