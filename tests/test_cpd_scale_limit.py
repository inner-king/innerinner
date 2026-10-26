# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional CPD scale limit (`s8_pose.cpd_scale_limit`), on synthetic data (CPU, no GPU).

Stage 8 is imported with importlib for its `sample_fits` and its `tf_y_up` constant. The synthetic cloud is a box, and its
source is the top half of the box (z > 0.02) scaled by a factor and shifted. On such a partial source CPD's scale estimate
comes out about 1.1 times the factor (measured: m 1.44 for factor 1.3, 1.05 for 0.95, 0.885 for 0.8), so the factors
below are picked for where the estimate lands, and each test asserts that landing before it relies on it.
"""

import ast
import copy
import importlib.util
import os
import pathlib
import sys

import numpy as np
import open3d as o3d
import pytest
from probreg import cpd, transformation

from simfoundry.pipeline import cpd_scale as cs
from simfoundry.utils.faiss_utils import l2_search

REPO = pathlib.Path(__file__).resolve().parents[1]
STAGE8 = REPO / "scripts/pipeline/A_reconstruction/stages/8_match_object_poses.py"
CFG = REPO / "scripts/cfg/real2sim_cfg.yaml"
README = REPO / "scripts/pipeline/README.md"

LIMIT = 0.1
FACTOR_BEYOND = 1.3  # CPD m about 1.44: beyond 1 + LIMIT
FACTOR_INSIDE = 0.95  # CPD m about 1.05: inside [1 - LIMIT, 1 + LIMIT]
FACTOR_BELOW = 0.8  # CPD m about 0.885: below 1 - LIMIT


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    """No GPU for the whole module: `l2_search` reads ENFORCE_FAISS_GPU at each call, so the switch must stay set."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ENFORCE_FAISS_GPU", "0")
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        yield


@pytest.fixture(scope="module")
def stage8(cpu_only):
    """The stage module, imported on the CPU.

    Importing the stage runs `bootstrap_hydra_workdir`, which changes the working directory to the Hydra config folder and
    appends to sys.path. Both are put back here: the tests that run after this module find their files by paths relative
    to the repository root.
    """
    cwd, path = os.getcwd(), list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("stage8_cpd_scale_limit", STAGE8)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
        sys.path[:] = path
    return module


@pytest.fixture(scope="module")
def target():
    """600 points on a box 0.10 x 0.06 x 0.04 m standing on z = 0, so its top half is z > 0.02."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.10, height=0.06, depth=0.04)
    o3d.utility.random.seed(0)
    return mesh.sample_points_uniformly(600)


def _top_half_source(target, factor):
    """The top half of the target, scaled by `factor` about its centre and moved by a small shift."""
    points = np.asarray(target.points)
    top = points[points[:, 2] > 0.02]
    centre = top.mean(0)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector((top - centre) * factor + centre + np.array([0.01, -0.005, 0.0]))
    return source


def _sample_fits(stage8, source, target, update_scale):
    np.random.seed(0)
    return stage8.sample_fits(source, target, 20, update_scale=update_scale, return_inverse=True)


@pytest.fixture(scope="module")
def fits_by_factor(stage8, target):
    """(unscaled, scaled) `sample_fits` results for a factor, computed once per factor for the whole module."""
    cache = {}

    def fits(factor):
        if factor not in cache:
            source = _top_half_source(target, factor)
            cache[factor] = (_sample_fits(stage8, source, target, False), _sample_fits(stage8, source, target, True))
        return cache[factor]

    return fits


def _same_transform(a, b):
    return (np.allclose(np.asarray(a.rot), np.asarray(b.rot), atol=1e-9)
            and np.allclose(np.asarray(a.t), np.asarray(b.t), atol=1e-9)
            and cs.cpd_scale_of(a) == pytest.approx(cs.cpd_scale_of(b), abs=1e-9))


def _stage_main():
    tree = ast.parse(STAGE8.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def _assignments_to(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


def test_a_scaled_fit_beyond_the_limit_is_held_at_the_bound_and_the_chosen_fit_is_within_it(stage8, target, fits_by_factor):
    unscaled, scaled = fits_by_factor(FACTOR_BEYOND)
    assert max(cs.cpd_scale_of(f["tf_z_up"]) for f in scaled) > 1 + LIMIT  # the case: CPD asks for more than the bound
    clamped = [f for f in cs.apply_cpd_scale_limit(scaled, LIMIT, target, stage8.tf_y_up) if f["cpd_scale_clamped"]]
    assert clamped
    for fit in clamped:
        assert cs.cpd_scale_of(fit["tf_z_up"]) == pytest.approx(1 + LIMIT, abs=1e-9)  # the refit holds the bound (1e-9)
        assert fit["cpd_scale_unclamped"] > 1 + LIMIT
    chosen = cs.select_top_fit(unscaled, scaled, LIMIT, target, stage8.tf_y_up)
    assert cs.cpd_scale_of(chosen["tf_z_up"]) <= 1 + LIMIT + 1e-9


def test_the_refit_is_the_fixed_scale_cpd_about_the_centroid_as_restated_here(stage8, target, fits_by_factor):
    _, scaled = fits_by_factor(FACTOR_BEYOND)
    entry = next(f for f in scaled if cs.cpd_scale_of(f["tf_z_up"]) > 1 + LIMIT)
    fit = cs.clamp_refit(entry, 1 + LIMIT, target, stage8.tf_y_up)
    # The same steps, written out: scale about the centroid to the size of 1 / m_c, a fixed-scale rigid CPD, then compose.
    tf_cum = entry["tf_z_up"].inverse()
    k = (1.0 / (1 + LIMIT)) / cs.cpd_scale_of(tf_cum)
    centre = np.asarray(entry["result"].points).mean(0)
    moved = copy.deepcopy(entry["result"])
    moved.points = o3d.utility.Vector3dVector((np.asarray(entry["result"].points) - centre) * k + centre)
    tf_param, _, _ = cpd.registration_cpd(moved, copy.deepcopy(target), tf_type_name="rigid", use_color=False,
                                          use_cuda=False, update_scale=False)
    assert not np.allclose(np.asarray(tf_param.t), 0.0, atol=1e-6)  # the refit moves something, so skipping it would show
    tf_scale = transformation.RigidTransformation(rot=np.eye(3), t=(1.0 - k) * centre, scale=k)
    assert _same_transform(fit["tf_z_up"], (tf_param * tf_scale * tf_cum).inverse())


def test_a_fit_inside_the_bounds_is_left_as_it_was_and_the_choice_is_the_unlimited_one(stage8, target, fits_by_factor):
    unscaled, scaled = fits_by_factor(FACTOR_INSIDE)
    assert all(1 - LIMIT <= cs.cpd_scale_of(f["tf_z_up"]) <= 1 + LIMIT for f in scaled)  # the case: nothing to clamp
    limited = cs.apply_cpd_scale_limit(scaled, LIMIT, target, stage8.tf_y_up)
    assert not any(f["cpd_scale_clamped"] for f in limited)
    for before, after in zip(scaled, limited):
        assert _same_transform(after["tf_z_up"], before["tf_z_up"])
        assert after["dist"] == before["dist"]
    with_limit = cs.select_top_fit(unscaled, scaled, LIMIT, target, stage8.tf_y_up)
    without = cs.select_top_fit(unscaled, scaled, None, target, stage8.tf_y_up)
    assert with_limit["dist"] == without["dist"]
    assert _same_transform(with_limit["tf_z_up"], without["tf_z_up"])


def test_with_the_limit_off_the_fits_come_back_as_they_are_and_the_choice_is_the_stage_pooling(stage8, target, fits_by_factor):
    unscaled, scaled = fits_by_factor(FACTOR_BEYOND)
    assert cs.apply_cpd_scale_limit(scaled, None, target, stage8.tf_y_up) is scaled
    assert not any("cpd_scale_clamped" in f for f in scaled)  # no key is added when the limit is off
    chosen = cs.select_top_fit(unscaled, scaled, None, target, stage8.tf_y_up)
    stage_choice = sorted(unscaled + scaled, key=lambda x: x["dist"])[0]  # the stage's own lines
    assert chosen is stage_choice


def test_a_scaled_fit_below_the_lower_bound_is_moved_up_to_it(stage8, target, fits_by_factor):
    _, scaled = fits_by_factor(FACTOR_BELOW)
    assert any(cs.cpd_scale_of(f["tf_z_up"]) < 1 - LIMIT for f in scaled)  # the case
    clamped = [f for f in cs.apply_cpd_scale_limit(scaled, LIMIT, target, stage8.tf_y_up) if f["cpd_scale_clamped"]]
    assert clamped
    for fit in clamped:
        assert cs.cpd_scale_of(fit["tf_z_up"]) == pytest.approx(1 - LIMIT, abs=1e-9)
        assert fit["cpd_scale_unclamped"] < 1 - LIMIT


@pytest.mark.parametrize(("m", "expected"), [
    (1.0, None),
    (1.0 + LIMIT, None),  # the upper bound itself is inside
    (1.0 + LIMIT + 1e-9, 1.0 + LIMIT),
    (1.0 - LIMIT, None),  # the lower bound itself is inside
    (1.0 - LIMIT - 1e-9, 1.0 - LIMIT),
    (0.8, 1.0 - LIMIT),  # below the lower bound: moved up to it
    (1.3, 1.0 + LIMIT),
])
def test_the_bounds_are_inside_and_a_scale_just_outside_them_is_moved_to_the_nearest(m, expected):
    got = cs.clamp_target(m, LIMIT)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("value", [-0.1, 0.0, 0, False, True, 1.0, 1.5, "0.1", float("nan"), float("inf")])
def test_the_limit_must_be_a_float_strictly_between_zero_and_one(value):
    with pytest.raises(ValueError):
        cs.validate_cpd_scale_limit(value)


def test_zero_is_refused_with_the_pointer_to_null():
    with pytest.raises(ValueError, match="use null"):
        cs.validate_cpd_scale_limit(0.0)


def test_an_out_of_range_limit_is_refused_when_the_fits_are_limited_too(stage8, target):
    with pytest.raises(ValueError):
        cs.apply_cpd_scale_limit([], 1.5, target, stage8.tf_y_up)


@pytest.mark.parametrize("value", [0.1, 0.5, 0.999])
def test_a_float_in_the_open_interval_and_null_are_accepted(value):
    assert cs.validate_cpd_scale_limit(value) == value
    assert cs.validate_cpd_scale_limit(None) is None


def test_a_clamped_fit_has_the_sample_fits_keys_and_its_tf_y_up_in_the_sample_fits_form(stage8, target, fits_by_factor):
    _, scaled = fits_by_factor(FACTOR_BEYOND)
    tf_y_up = stage8.tf_y_up
    # sample_fits inverts (cumulative * constant), so its tf_y_up is the constant's inverse times its tf_z_up.
    assert _same_transform(scaled[0]["tf_y_up"], tf_y_up.inverse() * scaled[0]["tf_z_up"])
    clamped = next(f for f in cs.apply_cpd_scale_limit(scaled, LIMIT, target, tf_y_up) if f["cpd_scale_clamped"])
    assert {"result", "dist", "tf_z_up", "tf_y_up"} <= set(clamped)
    assert _same_transform(clamped["tf_y_up"], tf_y_up.inverse() * clamped["tf_z_up"])


def test_the_dist_of_a_clamped_fit_is_the_sample_fits_distance_to_the_target(stage8, target, fits_by_factor):
    _, scaled = fits_by_factor(FACTOR_BEYOND)
    clamped = next(f for f in cs.apply_cpd_scale_limit(scaled, LIMIT, target, stage8.tf_y_up) if f["cpd_scale_clamped"])
    dists, _ = l2_search(np.array(target.points), np.array(clamped["result"].points), 1)
    assert clamped["dist"] == pytest.approx(float(np.sum(dists)), rel=1e-12)


def test_the_info_keys_come_only_with_a_limit_and_record_the_scale_before_the_limit(stage8, target, fits_by_factor):
    unscaled, scaled = fits_by_factor(FACTOR_BEYOND)
    assert cs.cpd_scale_limit_record(None, scaled[0]) == {}
    clamped = next(f for f in cs.apply_cpd_scale_limit(scaled, LIMIT, target, stage8.tf_y_up) if f["cpd_scale_clamped"])
    record = cs.cpd_scale_limit_record(LIMIT, clamped)
    assert record == {"cpd_scale_limit": LIMIT, "cpd_scale_clamped": True,
                      "cpd_scale_unclamped": clamped["cpd_scale_unclamped"]}
    assert record["cpd_scale_unclamped"] > 1 + LIMIT
    assert cs.cpd_scale_of(clamped["tf_z_up"]) == pytest.approx(1 + LIMIT, abs=1e-9)  # the scale after the limit
    assert cs.cpd_scale_of(unscaled[0]["tf_z_up"]) == pytest.approx(1.0, abs=1e-9)
    assert cs.cpd_scale_limit_record(LIMIT, unscaled[0]) == {"cpd_scale_limit": LIMIT, "cpd_scale_clamped": False,
                                                             "cpd_scale_unclamped": pytest.approx(1.0, abs=1e-9)}


def test_the_limit_is_applied_after_the_prune_and_before_the_fits_are_pooled():
    main = _stage_main()
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "apply_cpd_scale_limit"]
    assert len(calls) == 1
    call = calls[0]
    assert [ast.unparse(a) for a in call.args] == ["info_scaled_sorted", "cpd_scale_limit", "target", "tf_y_up"]
    assigned = [n for n in _assignments_to(main, "info_scaled_sorted") if n.value is call]
    assert len(assigned) == 1
    prune = [n for n in _assignments_to(main, "info_scaled_sorted")
             if ast.unparse(n.value).startswith("[info for info in info_scaled_sorted if")]
    merge = _assignments_to(main, "all_info")
    assert len(prune) == 1 and len(merge) == 1
    assert prune[0].lineno < call.lineno < merge[0].lineno


def test_the_limit_comes_from_the_config_through_one_name_with_a_null_default():
    reads = _assignments_to(_stage_main(), "cpd_scale_limit")
    assert len(reads) == 1
    assert ast.unparse(reads[0].value) == "cfg.s8_pose.get('cpd_scale_limit', None)"
    default = reads[0].value.args[1]
    assert isinstance(default, ast.Constant) and default.value is None


def test_the_stage_pools_and_sorts_as_before_and_writes_the_keys_from_the_chosen_fit():
    main = _stage_main()
    assert [ast.unparse(n.value) for n in _assignments_to(main, "all_info")] == ["info_unscaled_sorted + info_scaled_sorted"]
    assert "sorted(all_info, key=lambda x: x['dist'])" in [ast.unparse(n.value) for n in _assignments_to(main, "info_sorted")]
    assert "info_sorted[0]" in [ast.unparse(n.value) for n in _assignments_to(main, "top_info")]
    updates = [c for c in ast.walk(main) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
               and c.func.attr == "update" and ast.unparse(c.func.value) == "out_info"]
    assert len(updates) == 1
    assert [ast.unparse(a) for a in updates[0].args] == ["cpd_scale_limit_record(cpd_scale_limit, top_info)"]


def test_the_chosen_fit_is_logged_when_the_limit_changed_it():
    guards = [n for n in ast.walk(_stage_main()) if isinstance(n, ast.If)
              and ast.unparse(n.test) == "top_info.get('cpd_scale_clamped', False)"]
    assert len(guards) == 1
    logged = [c for c in ast.walk(guards[0]) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
              and c.func.attr == "info"]
    assert logged and "clamped to" in ast.unparse(logged[0]) and "(limit" in ast.unparse(logged[0])


def test_the_config_default_is_null_and_is_the_stage_default():
    omegaconf = pytest.importorskip("omegaconf")
    yaml = pytest.importorskip("yaml")
    assert omegaconf.OmegaConf.load(str(CFG)).s8_pose.get("cpd_scale_limit", None) is None
    assert yaml.safe_load(CFG.read_text())["s8_pose"]["cpd_scale_limit"] is None


def test_the_readme_names_the_keys_and_the_withdrawn_correction():
    text = README.read_text()
    assert "cpd_scale_limit" in text and "cpd_scale_clamped" in text and "cpd_scale_unclamped" in text
    assert "report_pre_ko.md" in text
