# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The height_fit safeguards (B16): the rod rule, the axis-alignment rule and the thickness floor, on synthetic meshes.

Nothing here reads research data. The table of research 1's rows is computed from the stage-8 meshes and reported,
not tested. Every mesh a test relies on is built from the shape it names (a box, a tube, an ellipsoid).
"""

import ast
import json
import pathlib
import re

import numpy as np
import pytest
from omegaconf import OmegaConf

from simfoundry.pipeline import size_check as sc

REPO = pathlib.Path(__file__).resolve().parents[1]
STAGE8 = REPO / "scripts/pipeline/A_reconstruction/stages/8_match_object_poses.py"
CFG = REPO / "scripts/cfg/real2sim_cfg.yaml"
UP = np.array([0.0, 0.0, 1.0])
SAFEGUARD_KEYS = ("skip_rod_like", "rod_min_elongation", "rod_min_roundness", "min_axis_alignment",
                  "min_result_extent_m")
RECORD_KEYS = {"axis_cos", "short_over_mid", "long_over_mid", "result_extent_m"}


def _rotation(axis, degrees):
    """The rotation matrix about `axis` by `degrees` (right-handed; Rodrigues' formula)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(degrees)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _box(lengths):
    """The eight corners of a box with edges `lengths` along x, y and z, centred on the origin."""
    half = np.asarray(lengths, dtype=float) / 2.0
    return np.array([[sx * half[0], sy * half[1], sz * half[2]]
                     for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])


def _tube(length, semi_a, semi_b, n_along=21, n_round=24):
    """Points on an elliptic tube along x: `length` long, semi-axes `semi_a` (y) and `semi_b` (z), with rings at both
    ends and in the middle, so its extents are exact."""
    along = np.linspace(-length / 2, length / 2, n_along)
    around = np.linspace(0.0, 2 * np.pi, n_round, endpoint=False)
    return np.array([[x, semi_a * np.cos(a), semi_b * np.sin(a)] for x in along for a in around])


def _ellipsoid(semi_axes, n=14):
    """Points on an ellipsoid with the given semi-axes along x, y and z, on a regular grid of angles."""
    polar = np.linspace(0.0, np.pi, n)
    azimuth = np.linspace(0.0, 2 * np.pi, 2 * n, endpoint=False)
    a, b, c = semi_axes
    return np.array([[a * np.sin(p) * np.cos(q), b * np.sin(p) * np.sin(q), c * np.cos(p)]
                     for p in polar for q in azimuth])


def _pen(roll_deg=90.0):
    """A pen: an ellipse of 7.5 by 5.3 mm across, 126.6 mm long, its long axis tilted 13 degrees and rolled about it."""
    return _tube(0.1266, 0.00375, 0.00265) @ _rotation([1, 0, 0], roll_deg).T @ _rotation([0, 1, 0], 13.0).T


def _height_m(points, up=UP):
    return float(np.ptp(np.asarray(points) @ up))


def _settings(**overrides):
    """The height_fit settings with the stretch rules on and every safeguard off, unless overridden."""
    settings = {"enabled": True, "threshold": 0.25, "min_scale": 0.1, "max_scale": 2.0, "height_noise_sigmas": None,
                "skip_rod_like": False, "rod_min_elongation": 3.0, "rod_min_roundness": 0.6,
                "min_axis_alignment": None, "min_result_extent_m": None}
    settings.update(overrides)
    return settings


def _report(mesh_m, depth_m, floor_spread_m=0.001):
    """A height report as size_report writes it, for a mesh of height `mesh_m` against a depth of `depth_m`."""
    return {"height": {"mesh_m": mesh_m, "depth_m": depth_m, "ratio": mesh_m / depth_m,
                       "deviation": abs(mesh_m / depth_m - 1.0), "deviation_m": abs(mesh_m - depth_m),
                       "floor_m": 0.0, "floor_spread_m": floor_spread_m, "floor_source": "ring", "top_m": depth_m},
            "footprint": {}, "warnings": []}


def _same_profile(left, right):
    assert left["undefined"] == right["undefined"]
    if left["undefined"] is None:
        assert left["extents"] == pytest.approx(right["extents"], rel=1e-9, abs=1e-15)
        assert left["short_over_mid"] == pytest.approx(right["short_over_mid"], rel=1e-9)
        assert left["long_over_mid"] == pytest.approx(right["long_over_mid"], rel=1e-9)
        assert left["abs_cos"] == pytest.approx(right["abs_cos"], rel=1e-9, abs=1e-12)


# 1. the profile of a known shape ---------------------------------------------------------------------------------------

def test_a_box_gives_its_extents_and_the_ratios_of_its_edges():
    profile = sc.mesh_axis_profile(_box((0.30, 0.10, 0.02)), UP)
    assert profile["undefined"] is None
    assert profile["extents"] == pytest.approx([0.02, 0.10, 0.30], abs=1e-12)
    assert profile["short_over_mid"] == pytest.approx(0.2, abs=1e-9)
    assert profile["long_over_mid"] == pytest.approx(3.0, abs=1e-9)
    assert profile["abs_cos"] == pytest.approx(1.0, abs=1e-9)  # up along the thin edge
    along_long_edge = sc.mesh_axis_profile(_box((0.30, 0.10, 0.02)), np.array([1.0, 0.0, 0.0]))
    assert along_long_edge["abs_cos"] == pytest.approx(0.0, abs=1e-9)


# 2. invariance ----------------------------------------------------------------------------------------------------------

def test_the_profile_is_unchanged_by_rotation_translation_scale_order_and_the_sign_of_up():
    rng = np.random.default_rng(7)
    points = _pen(roll_deg=40.0)
    base = sc.mesh_axis_profile(points, UP)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    moved = 2.5 * points @ rotation.T + np.array([0.3, -0.2, 0.7])
    scaled = sc.mesh_axis_profile(moved, rotation @ UP)  # up turned with the mesh
    assert scaled["extents"] == pytest.approx([2.5 * e for e in base["extents"]], rel=1e-9)
    assert scaled["short_over_mid"] == pytest.approx(base["short_over_mid"], rel=1e-9)
    assert scaled["long_over_mid"] == pytest.approx(base["long_over_mid"], rel=1e-9)
    assert scaled["abs_cos"] == pytest.approx(base["abs_cos"], rel=1e-9)
    _same_profile(sc.mesh_axis_profile(points[rng.permutation(len(points))], UP), base)
    _same_profile(sc.mesh_axis_profile(points, 5.0 * UP), base)
    _same_profile(sc.mesh_axis_profile(points, -UP), base)


# 3. degenerate input ----------------------------------------------------------------------------------------------------

def test_degenerate_input_is_undefined_not_an_error_and_serialises():
    cases = {
        "three points": (_box((1.0, 1.0, 1.0))[:3], UP, "only 3 vertices"),
        "identical points": (np.ones((4, 3)), UP, "middle extent is zero"),
        "nan": (np.vstack([_box((1.0, 2.0, 3.0)), [[np.nan, 0.0, 0.0]]]), UP, "non-finite vertices"),
        "inf": (np.vstack([_box((1.0, 2.0, 3.0)), [[np.inf, 0.0, 0.0]]]), UP, "non-finite vertices"),
        "zero up": (_box((1.0, 2.0, 3.0)), np.zeros(3), "zero up direction"),
    }
    for name, (points, up, reason) in cases.items():
        profile = sc.mesh_axis_profile(points, up)
        assert profile["undefined"].startswith(reason), name
        assert profile["extents"] is None and profile["short_over_mid"] is None, name
        assert profile["long_over_mid"] is None and profile["abs_cos"] is None, name
        json.dumps(profile)
    cube = _box((1.0, 1.0, 1.0))
    assert sc.mesh_axis_profile(cube, UP) == sc.mesh_axis_profile(cube, UP)  # the tie is broken the same way
    json.dumps(sc.mesh_axis_profile(cube, UP))
    with pytest.raises(ValueError):
        sc.mesh_axis_profile(np.zeros((5, 2)), UP)


# 4. the rod rule, the main condition ----------------------------------------------------------------------------------

def test_a_thin_round_rod_is_skipped_and_a_flat_blade_or_a_lump_is_stretched():
    settings = _settings(skip_rod_like=True)
    cylinder = _tube(0.126, 0.004, 0.004) @ _rotation([0, 1, 0], 8.0).T
    pen = _pen(roll_deg=0.0)
    for name, points in {"cylinder": cylinder, "pen": pen}.items():
        report = _report(_height_m(points), 1.8 * _height_m(points))
        out = sc.decide_height_fit(report, settings, vertices=points, up=UP)
        assert out["applied"] is False and out["scale"] is None, name
        assert out["skipped"].startswith("rod-like"), name
        assert out["long_over_mid"] > 3.0 and out["short_over_mid"] >= 0.6, name
        assert set(out) == {"applied", "scale", "skipped", "axis_cos", "short_over_mid", "long_over_mid"}, name
    stretched = {
        "knife blade (314 x 54 x 18 mm)": _box((0.314, 0.054, 0.018)),
        "onion-like slab (107 x 103 x 44 mm)": _box((0.107, 0.103, 0.044)),
        "round lump (long/mid 1.04, short/mid 0.66)": _ellipsoid((0.0535, 0.0515, 0.034)),
        "slab (long/mid 1.2, short/mid 0.05)": _box((0.12, 0.10, 0.005)),
    }
    for name, points in stretched.items():
        report = _report(_height_m(points), 1.8 * _height_m(points))
        out = sc.decide_height_fit(report, settings, vertices=points, up=UP)
        assert out["applied"] is True and out["skipped"] is None, name
        assert out["scale"] == pytest.approx(1.8, rel=1e-9), name


# 5. the rod limits: inclusive at the limit, exclusive just above ------------------------------------------------------

def test_the_rod_limits_skip_at_the_limit_and_stretch_just_above_it():
    points = _pen(roll_deg=0.0)
    profile = sc.mesh_axis_profile(points, UP)
    report = _report(_height_m(points), 1.8 * _height_m(points))
    long_over, short_over = profile["long_over_mid"], profile["short_over_mid"]
    at_elongation = sc.decide_height_fit(report, _settings(skip_rod_like=True, rod_min_elongation=long_over),
                                         vertices=points, up=UP)
    just_above = sc.decide_height_fit(
        report, _settings(skip_rod_like=True, rod_min_elongation=float(np.nextafter(long_over, np.inf))),
        vertices=points, up=UP)
    assert at_elongation["applied"] is False and just_above["applied"] is True
    at_roundness = sc.decide_height_fit(report, _settings(skip_rod_like=True, rod_min_roundness=short_over),
                                        vertices=points, up=UP)
    just_above = sc.decide_height_fit(
        report, _settings(skip_rod_like=True, rod_min_roundness=float(np.nextafter(short_over, np.inf))),
        vertices=points, up=UP)
    assert at_roundness["applied"] is False and just_above["applied"] is True


# 6. the three safeguards: independent, in order, and the alignment cases --------------------------------------------

def test_each_safeguard_refuses_on_its_own_and_the_first_failing_one_gives_the_reason():
    pen = _pen(roll_deg=90.0)  # its shortest axis lies horizontal: |cos| 0
    report = _report(_height_m(pen), 0.35 * _height_m(pen))
    thinnest_after = sc.mesh_axis_profile(sc.fit_height(pen, UP, 0.35), UP)["extents"][0]
    rod = sc.decide_height_fit(report, _settings(skip_rod_like=True), vertices=pen, up=UP)
    align = sc.decide_height_fit(report, _settings(min_axis_alignment=0.9), vertices=pen, up=UP)
    thick = sc.decide_height_fit(report, _settings(min_result_extent_m=1.5 * thinnest_after),
                                 vertices=pen, up=UP)
    assert rod["skipped"].startswith("rod-like")
    assert align["skipped"].startswith("shortest axis not vertical")
    assert thick["skipped"].startswith("stretched mesh would be")
    everything = sc.decide_height_fit(
        report, _settings(skip_rod_like=True, min_axis_alignment=0.9, min_result_extent_m=1.5 * thinnest_after),
        vertices=pen, up=UP)
    assert everything["skipped"].startswith("rod-like")  # the rod rule is first
    assert {"axis_cos", "short_over_mid", "long_over_mid", "result_extent_m"} <= set(everything)


def test_the_alignment_rule_refuses_a_round_section_and_a_standing_tube_and_stretches_a_lying_plate():
    marker = _tube(0.126, 0.006, 0.0056) @ _rotation([0, 1, 0], 8.0).T  # short/mid 0.93: round enough
    out = sc.decide_height_fit(_report(_height_m(marker), 0.79 * _height_m(marker)),
                               _settings(min_axis_alignment=0.9), vertices=marker, up=UP)
    assert out["skipped"].startswith("shortest axis not determinable")
    standing = _tube(0.126, 0.0037, 0.0026) @ _rotation([0, 1, 0], 90.0).T  # the long axis along z
    out = sc.decide_height_fit(_report(_height_m(standing), 1.8 * _height_m(standing)),
                               _settings(min_axis_alignment=0.9), vertices=standing, up=UP)
    assert out["skipped"].startswith("shortest axis not vertical") and out["axis_cos"] < 0.05
    plate = _box((0.1, 0.08, 0.005))
    out = sc.decide_height_fit(_report(_height_m(plate), 1.8 * _height_m(plate)),
                               _settings(min_axis_alignment=0.9), vertices=plate, up=UP)
    assert out["applied"] is True and out["axis_cos"] == pytest.approx(1.0, abs=1e-9)


def test_alignment_is_strict_on_the_short_axis_and_inclusive_on_the_angle():
    box = _box((9.0, 10.0, 10.0))  # short/mid exactly 0.9: not above the limit, so the rule does not refuse it
    report = _report(_height_m(box, np.array([1.0, 0.0, 0.0])), 1.8 * _height_m(box, np.array([1.0, 0.0, 0.0])))
    out = sc.decide_height_fit(report, _settings(min_axis_alignment=0.9), vertices=box,
                               up=np.array([1.0, 0.0, 0.0]))
    assert out["applied"] is True and out["short_over_mid"] == pytest.approx(0.9, abs=1e-12)
    diagonal = np.array([1.0, 0.0, 1.0])
    limit = sc.mesh_axis_profile(box, diagonal)["abs_cos"]
    out = sc.decide_height_fit(report, _settings(min_axis_alignment=limit), vertices=box, up=diagonal)
    assert out["applied"] is True  # |cos| equal to the limit is applied


# 7. the thickness floor ---------------------------------------------------------------------------------------------

def test_the_thickness_floor_compares_the_stretched_thinnest_extent_strictly():
    plate = _box((0.100, 0.080, 0.010))  # a 10 mm plate, stretched by 0.25 to 2.5 mm
    report = _report(_height_m(plate), 0.25 * _height_m(plate))
    stretched = sc.mesh_axis_profile(sc.fit_height(plate, UP, 0.25), UP)["extents"][0]
    refused = sc.decide_height_fit(report, _settings(min_result_extent_m=0.004), vertices=plate, up=UP)
    assert refused["applied"] is False
    assert refused["skipped"] == f"stretched mesh would be {stretched * 1000:.1f} mm thick (minimum 4.0 mm)"
    assert refused["result_extent_m"] == pytest.approx(stretched, rel=1e-9)
    assert sc.decide_height_fit(report, _settings(min_result_extent_m=0.002), vertices=plate, up=UP)["applied"]
    assert sc.decide_height_fit(report, _settings(min_result_extent_m=stretched), vertices=plate, up=UP)["applied"]


def test_a_safeguard_on_needs_the_mesh_and_up_even_where_the_default_rules_skip_the_object():
    plate = _box((0.100, 0.080, 0.010))
    within_threshold = _report(_height_m(plate), 1.05 * _height_m(plate))  # skipped by the default rules
    with pytest.raises(ValueError):
        sc.decide_height_fit(within_threshold, _settings(min_result_extent_m=0.004))
    stretched = _report(_height_m(plate), 0.25 * _height_m(plate))
    with pytest.raises(ValueError):
        sc.decide_height_fit(stretched, _settings(min_result_extent_m=0.004), up=UP)
    assert sc.decide_height_fit(stretched, _settings(enabled=False, min_result_extent_m=0.004)) == {
        "applied": False, "scale": None, "skipped": "height_fit is disabled"}


# 8. the stretch and the profile commute with a rigid move and a uniform scale -------------------------------------

def test_the_profile_of_a_stretch_is_the_same_whether_it_is_taken_before_or_after_the_frame_move():
    rng = np.random.default_rng(3)
    points = _pen(roll_deg=30.0)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    offset = np.array([0.4, -0.1, 0.2])
    scale_all = 1.7
    moved_up = rotation @ UP
    for stretch in (0.35, 1.6):
        left = sc.mesh_axis_profile(sc.fit_height(scale_all * points @ rotation.T + offset, moved_up, stretch),
                                    moved_up)
        right = sc.mesh_axis_profile(scale_all * sc.fit_height(points, UP, stretch) @ rotation.T + offset, moved_up)
        assert left["extents"] == pytest.approx(right["extents"], abs=1e-9)
        assert left["abs_cos"] == pytest.approx(right["abs_cos"], abs=1e-9)


def test_the_rod_rule_reads_3_and_0_6_when_its_keys_are_absent():
    settings = {"enabled": True, "threshold": 0.25, "min_scale": 0.1, "max_scale": 2.0, "height_noise_sigmas": None,
                "skip_rod_like": True}
    slender = _tube(0.028, 0.005, 0.0045)  # long/mid 2.8, short/mid 0.9: not elongated enough to be a rod
    report = _report(_height_m(slender), 1.8 * _height_m(slender))
    assert sc.decide_height_fit(report, settings, vertices=slender, up=UP)["applied"] is True
    flat_round = _tube(0.05, 0.005, 0.0025)  # long/mid 5, short/mid 0.5: not round enough to be a rod
    report = _report(_height_m(flat_round), 1.8 * _height_m(flat_round))
    assert sc.decide_height_fit(report, settings, vertices=flat_round, up=UP)["applied"] is True


def test_the_shortest_axis_is_the_shortest_extent_not_the_smallest_variance():
    xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(-0.25, 0.25, 11))
    grid = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    outliers = np.array([[0.0, 0.0, 0.6], [0.0, 0.0, -0.6]])  # a wide z extent, but little variance along z
    profile = sc.mesh_axis_profile(np.vstack([grid, outliers]), UP)
    assert profile["extents"][0] == pytest.approx(0.5, abs=1e-9)
    assert profile["abs_cos"] == pytest.approx(0.0, abs=1e-9)


def test_the_rod_and_alignment_reasons_name_their_numbers_and_thresholds_exactly():
    pen = _pen(roll_deg=0.0)
    profile = sc.mesh_axis_profile(pen, UP)
    report = _report(_height_m(pen), 1.8 * _height_m(pen))
    rod = sc.decide_height_fit(report, _settings(skip_rod_like=True), vertices=pen, up=UP)
    assert rod["skipped"] == (f"rod-like (long/mid {profile['long_over_mid']:.2f} >= 3, "
                              f"short/mid {profile['short_over_mid']:.2f} >= 0.6): "
                              "a vertical stretch would flatten its section")
    other = sc.decide_height_fit(report, _settings(skip_rod_like=True, rod_min_elongation=2.5,
                                                   rod_min_roundness=0.5), vertices=pen, up=UP)
    assert other["skipped"] == (f"rod-like (long/mid {profile['long_over_mid']:.2f} >= 2.5, "
                                f"short/mid {profile['short_over_mid']:.2f} >= 0.5): "
                                "a vertical stretch would flatten its section")
    flat = _pen(roll_deg=90.0)
    flat_profile = sc.mesh_axis_profile(flat, UP)
    not_vertical = sc.decide_height_fit(_report(_height_m(flat), 1.8 * _height_m(flat)),
                                        _settings(min_axis_alignment=0.9), vertices=flat, up=UP)
    assert not_vertical["skipped"] == f"shortest axis not vertical (|cos| {flat_profile['abs_cos']:.2f})"
    marker = _tube(0.126, 0.006, 0.0056) @ _rotation([0, 1, 0], 8.0).T
    marker_profile = sc.mesh_axis_profile(marker, UP)
    undetermined = sc.decide_height_fit(_report(_height_m(marker), 1.8 * _height_m(marker)),
                                        _settings(min_axis_alignment=0.9), vertices=marker, up=UP)
    assert undetermined["skipped"] == (f"shortest axis not determinable "
                                       f"(short/mid {marker_profile['short_over_mid']:.2f})")


def test_a_thickness_safeguard_on_three_points_is_undefined_not_an_error():
    three = _box((1.0, 1.0, 1.0))[:3]
    out = sc.decide_height_fit(_report(0.1, 0.035), _settings(min_result_extent_m=0.004), vertices=three, up=UP)
    assert out["applied"] is False and out["scale"] is None
    assert out["skipped"] == "shape undefined (only 3 vertices, need 4)"
    assert out["result_extent_m"] is None


def test_no_vertices_at_all_is_undefined_not_an_error():
    for empty in (np.zeros((0, 3)), np.array([])):
        profile = sc.mesh_axis_profile(empty, UP)
        assert profile["undefined"] == "only 0 vertices, need 4"
        assert profile["extents"] is None and profile["abs_cos"] is None
        json.dumps(profile)


def test_a_missing_vertices_or_up_is_refused_by_the_safeguard_check_itself():
    plate = _box((0.100, 0.080, 0.010))
    stretched = _report(_height_m(plate), 0.25 * _height_m(plate))
    for safeguard in ({"skip_rod_like": True}, {"min_axis_alignment": 0.9}, {"min_result_extent_m": 0.004}):
        settings = _settings(**safeguard)
        with pytest.raises(ValueError, match="safeguard"):
            sc.decide_height_fit(stretched, settings, vertices=plate, up=None)
        with pytest.raises(ValueError, match="safeguard"):
            sc.decide_height_fit(stretched, settings, vertices=None, up=UP)
        with pytest.raises(ValueError, match="safeguard"):
            sc.decide_height_fit(stretched, settings)


def test_alignment_refuses_an_angle_just_past_its_limit():
    box = _box((9.0, 10.0, 10.0))  # short/mid exactly 0.9
    diagonal = np.array([1.0, 0.0, 1.0])
    limit = sc.mesh_axis_profile(box, diagonal)["abs_cos"]
    report = _report(_height_m(box, diagonal), 1.8 * _height_m(box, diagonal))
    just_past = float(np.nextafter(limit, 1.0))
    out = sc.decide_height_fit(report, _settings(min_axis_alignment=just_past), vertices=box, up=diagonal)
    assert out["applied"] is False
    assert out["skipped"] == f"shortest axis not vertical (|cos| {limit:.2f})"


# 9. every safeguard off: the decision is the one it was before the safeguards -------------------------------------

def test_with_every_safeguard_off_the_decision_is_the_one_it_was_before_the_safeguards():
    points = _pen(roll_deg=40.0)
    before = {"enabled": True, "threshold": 0.25, "min_scale": 0.1, "max_scale": 2.0, "height_noise_sigmas": 2.0}
    off = {**before, "skip_rod_like": False, "rod_min_elongation": 3.0, "rod_min_roundness": 0.6,
           "min_axis_alignment": None, "min_result_extent_m": None}
    off_as_none = {**before, "skip_rod_like": None, "min_axis_alignment": None, "min_result_extent_m": None}
    mesh_m = _height_m(points)
    outcomes = set()
    for i in range(20):
        depth = mesh_m * (0.3, 0.6, 0.8, 1.0, 1.4, 2.4)[i % 6]
        report = _report(mesh_m * (1.0 + 0.01 * i), depth, floor_spread_m=0.001 * (1 + i % 3))
        expected = sc.decide_height_fit(report, before)
        outcomes.add(expected["skipped"].split(" ")[0] if expected["skipped"] else "applied")
        assert sc.decide_height_fit(report, off) == expected
        assert sc.decide_height_fit(report, off_as_none) == expected
        assert sc.decide_height_fit(report, off, vertices=points, up=UP) == expected
        assert not RECORD_KEYS & set(expected)
    assert len(outcomes) >= 3  # the reports cover applied, refused and skipped decisions


# 10. the record ---------------------------------------------------------------------------------------------------

def test_the_record_holds_only_the_numbers_of_the_safeguards_that_are_on_and_serialises():
    pen = _pen(roll_deg=90.0)
    report = _report(_height_m(pen), 0.35 * _height_m(pen))
    rod_only = sc.decide_height_fit(report, _settings(skip_rod_like=True), vertices=pen, up=UP)
    thickness_only = sc.decide_height_fit(report, _settings(min_result_extent_m=0.004), vertices=pen, up=UP)
    assert set(rod_only) == {"applied", "scale", "skipped", "axis_cos", "short_over_mid", "long_over_mid"}
    assert set(thickness_only) == {"applied", "scale", "skipped", "result_extent_m"}
    for out in (rod_only, thickness_only):
        assert all(v is None or isinstance(v, float) for k, v in out.items() if k not in ("applied", "skipped"))
        json.dumps(out)
    three = sc.decide_height_fit(_report(0.1, 0.035), _settings(skip_rod_like=True),
                                 vertices=_box((1.0, 1.0, 1.0))[:3], up=UP)
    assert three["skipped"].startswith("shape undefined (only 3 vertices")
    assert three["axis_cos"] is None and three["long_over_mid"] is None


# 11. validation ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("key", "bad"), [
    ("skip_rod_like", 1), ("skip_rod_like", 0), ("skip_rod_like", "true"), ("skip_rod_like", []),
    ("skip_rod_like", 2.0),
    ("rod_min_elongation", 1.0), ("rod_min_elongation", 0.5), ("rod_min_elongation", float("nan")),
    ("rod_min_elongation", float("inf")), ("rod_min_elongation", True), ("rod_min_elongation", "3"),
    ("rod_min_elongation", None),
    ("rod_min_roundness", 0), ("rod_min_roundness", -0.1), ("rod_min_roundness", 1.5),
    ("rod_min_roundness", float("nan")), ("rod_min_roundness", True), ("rod_min_roundness", "0.6"),
    ("rod_min_roundness", None),
    ("min_axis_alignment", 0), ("min_axis_alignment", -0.1), ("min_axis_alignment", 1.5),
    ("min_axis_alignment", float("nan")), ("min_axis_alignment", True), ("min_axis_alignment", "0.97"),
    ("min_axis_alignment", [0.97]),
    ("min_result_extent_m", 0), ("min_result_extent_m", -0.001), ("min_result_extent_m", float("nan")),
    ("min_result_extent_m", float("inf")), ("min_result_extent_m", True), ("min_result_extent_m", "0.004"),
])
def test_a_bad_safeguard_value_is_refused_by_name_at_start_up(key, bad):
    with pytest.raises(ValueError, match=key):
        sc.validate_height_fit_safeguards({key: bad})


@pytest.mark.parametrize(("key", "good"), [
    ("skip_rod_like", True), ("skip_rod_like", False), ("skip_rod_like", None),
    ("rod_min_elongation", 1.5), ("rod_min_elongation", 3.0),
    ("rod_min_roundness", 0.6), ("rod_min_roundness", 1.0),
    ("min_axis_alignment", None), ("min_axis_alignment", 0.9), ("min_axis_alignment", 1.0),
    ("min_result_extent_m", None), ("min_result_extent_m", 0.004),
])
def test_a_valid_safeguard_value_passes(key, good):
    sc.validate_height_fit_safeguards({key: good})


def test_the_default_safeguards_pass_validation():
    sc.validate_height_fit_safeguards({})


# 12. the stage-8 wiring, read from its source ---------------------------------------------------------------------

def _stage8_main():
    tree = ast.parse(STAGE8.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Name) and n.func.id == name)
                 or (isinstance(n.func, ast.Attribute) and n.func.attr == name))]


def test_stage_8_reads_the_safeguard_keys_with_the_defaults_the_yaml_gives():
    main = _stage8_main()
    settings = next(n.value for n in ast.walk(main) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "height_fit_settings" for t in n.targets))
    read = {key.value: value for key, value in zip(settings.keys, settings.values)
            if isinstance(key, ast.Constant) and key.value in SAFEGUARD_KEYS}
    assert set(read) == set(SAFEGUARD_KEYS)
    yaml_block = OmegaConf.to_container(OmegaConf.load(CFG).s8_pose.height_fit)
    for name in SAFEGUARD_KEYS:
        call = read[name]
        assert isinstance(call, ast.Call) and call.args[0].value == name, name
        assert ast.literal_eval(call.args[1]) == yaml_block[name], name


def test_stage_8_validates_the_safeguards_once_outside_every_loop_and_before_the_decision():
    main = _stage8_main()
    parent = {child: node for node in ast.walk(main) for child in ast.iter_child_nodes(node)}
    imported = {a.name for n in ast.parse(STAGE8.read_text()).body if isinstance(n, ast.ImportFrom)
                and n.module == "simfoundry.pipeline.size_check" for a in n.names}
    assert "validate_height_fit_safeguards" in imported  # the validator is imported with the other size-check names
    validates = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "validate_height_fit_safeguards"]
    assert len(validates) == 1
    call = validates[0]
    assert ast.unparse(call.args[0]) == "height_fit_settings" and not call.keywords
    node = call
    while node is not main:
        node = parent[node]
        assert not isinstance(node, (ast.For, ast.While)), "the validator runs inside a loop"
    decisions = _calls(main, "decide_height_fit")
    assert len(decisions) == 1
    assert call.lineno < decisions[0].lineno


def test_stage_8_calls_decide_height_fit_once_with_the_placed_mesh_and_the_up_direction():
    main = _stage8_main()
    calls = _calls(main, "decide_height_fit")
    assert len(calls) == 1
    call = calls[0]
    assert [ast.dump(a) for a in call.args] == [
        ast.dump(ast.parse("size_check_record", mode="eval").body),
        ast.dump(ast.parse("height_fit_settings", mode="eval").body),
    ]
    keywords = {k.arg: k.value for k in call.keywords}
    assert set(keywords) == {"vertices", "up"}
    assert ast.dump(keywords["vertices"]) == ast.dump(ast.parse("np.asarray(mesh_updated.vertices)", mode="eval").body)
    assert ast.dump(keywords["up"]) == ast.dump(ast.parse("up_cam", mode="eval").body)


# 13. the yaml -----------------------------------------------------------------------------------------------------------

def test_the_yaml_carries_the_five_safeguards_off_with_candidate_notes():
    block = OmegaConf.to_container(OmegaConf.load(CFG).s8_pose.height_fit)
    assert {k: block[k] for k in SAFEGUARD_KEYS} == {
        "skip_rod_like": False, "rod_min_elongation": 3.0, "rod_min_roundness": 0.6,
        "min_axis_alignment": None, "min_result_extent_m": None,
    }
    text = CFG.read_text()
    lines = text.splitlines()
    for key in SAFEGUARD_KEYS:
        line = next(line for line in lines if line.lstrip().startswith(f"{key}:"))
        assert "candidate" in line, key
    assert "one table of 24 rows of 9 objects in 3 scenes" in text
    assert re.search("[가-힣]", text) is None  # the configuration comments are in English
