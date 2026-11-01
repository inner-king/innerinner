# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vegetable presets, tissue kinds, names and the setting warnings of the MPM cutting solver (materials.py, mpm.py).

The numbers are the request's (final_veg/solver/briefs/R5_presets_warnings_ko.md) written out here as literals, so that a
changed table value or threshold in the module fails a test. The warning tests build small blocks and take a few steps.
CPU only.
"""

import json
import math
import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from simfoundry.cutting import materials as M  # noqa: E402
from simfoundry.cutting import mpm as MP  # noqa: E402
from simfoundry.cutting.materials import FIRM_TOFU, PRESETS, VEGETABLE_RANGES  # noqa: E402
from simfoundry.cutting.mpm import HOLD_KAPPA, MPMCuttingSim, SimConfig, SolverWarning  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLOCK = (0.04, 0.025, 0.03)  # 4 x 2.5 x 3 cm; the demo's firm tofu block is 8 x 5 x 3 cm and shares the 3 cm height

# The request's table: (E [Pa], nu, rho [kg/m^3], G_c [J/m^2], knife friction).
TABLE = {
    "carrot": (6.0e6, 0.35, 1020.0, 389.0, 0.41),
    "cucumber": (1.1e6, 0.35, 995.0, 125.0, 0.36),
    "zucchini": (1.8e6, 0.35, 960.0, 705.0, 0.36),
    "onion": (1.8e6, 0.35, 1075.0, 451.0, 0.477),
}
# The request's ranges: E [Pa], G_c [J/m^2], rho [kg/m^3], nu.
RANGES = {
    "carrot": dict(E=(4.6e6, 9.9e6), G=(344.0, 440.0), rho=(1003.0, 1040.0), nu=(0.24, 0.45)),
    "cucumber": dict(E=(0.85e6, 1.4e6), G=(60.0, 238.0), rho=(980.0, 1010.0), nu=(0.24, 0.45)),
    "zucchini": dict(E=(0.38e6, 5.3e6), G=(622.0, 800.0), rho=(910.0, 1010.0), nu=(0.24, 0.45)),
    "onion": dict(E=(1.4e6, 2.2e6), G=(302.0, 675.0), rho=(1040.0, 1110.0), nu=(0.24, 0.45)),
}
# The request's names, lower case except the Korean ones (matching is case-insensitive).
SYNONYMS = {
    "carrot": ["carrot", "Daucus carota", "당근"],
    "cucumber": ["cucumber", "Cucumis sativus", "오이"],
    "zucchini": ["zucchini", "courgette", "Cucurbita pepo", "애호박"],
    "onion": ["onion", "Allium cepa", "양파"],
    "firm_tofu": ["tofu", "bean curd", "두부"],
}
TISSUE = ["root_tuber", "watery_fruit", "bulb_scale", "gel"]
# The request's kinds and members: root_tuber is carrot, watery_fruit cucumber and zucchini, bulb_scale onion, gel firm tofu.
KIND_MEMBERS = {
    "root_tuber": ["carrot"],
    "watery_fruit": ["cucumber", "zucchini"],
    "bulb_scale": ["onion"],
    "gel": ["firm_tofu"],
}


def _geomean(values):
    return math.prod(values) ** (1.0 / len(values))


def _members(kind):
    return [PRESETS[name] for name in KIND_MEMBERS[kind]]


def _sim(material=FIRM_TOFU, block=BLOCK, **config):
    return MPMCuttingSim(material, block, config=SimConfig(**config), device="cpu")


class no_solver_warning:
    """Fails the test if a SolverWarning is raised inside the block (the warnings are recorded here, not shown)."""

    def __enter__(self):
        self._catch = warnings.catch_warnings(record=True)
        self._log = self._catch.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc, tb):
        self._catch.__exit__(exc_type, exc, tb)
        if exc_type is None:
            raised = [str(w.message) for w in self._log if issubclass(w.category, SolverWarning)]
            if raised:
                pytest.fail(f"unexpected SolverWarning: {raised}")
        return False


# ---- presets and the common rule ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(TABLE))
def test_the_vegetable_presets_are_the_request_table(name):
    E, nu, rho, G, friction = TABLE[name]
    m = PRESETS[name]
    assert m.youngs_modulus == pytest.approx(E, rel=1e-12)
    assert m.poisson_ratio == pytest.approx(nu, rel=1e-12)
    assert m.density == pytest.approx(rho, rel=1e-12)
    assert m.fracture_energy == pytest.approx(G, rel=1e-12)
    assert m.knife_friction == pytest.approx(friction, rel=1e-12)


def test_the_four_tofu_presets_are_still_there_with_their_values():
    assert {"firm_tofu", "silken_tofu", "extra_firm_tofu", "pressed_tofu"} <= set(PRESETS)
    assert PRESETS["firm_tofu"] is FIRM_TOFU
    assert FIRM_TOFU.youngs_modulus == pytest.approx(26.4e3)
    assert FIRM_TOFU.fracture_energy == pytest.approx(3.0)


@pytest.mark.parametrize("name", sorted(TABLE))
def test_every_vegetable_follows_the_common_rule(name):
    m = PRESETS[name]
    assert m.viscosity == 2.0
    assert m.relaxed_modulus == 0.0 and m.relaxation_time == 0.0  # elastic: the request gives no relaxation value
    assert m.tensile_strength == pytest.approx(m.youngs_modulus / 10.0)  # the tofu rule, E / 10
    assert m.initiation_length == pytest.approx(2.0e-3)
    assert m.board_friction == 0.5 and m.contact_friction == 0.3


@pytest.mark.parametrize("name", sorted(RANGES))
def test_the_vegetable_ranges_are_the_request_ranges(name):
    r = RANGES[name]
    mine = VEGETABLE_RANGES[name]
    assert mine["youngs_modulus"] == pytest.approx(r["E"], rel=1e-12)
    assert mine["fracture_energy"] == pytest.approx(r["G"], rel=1e-12)
    assert mine["density"] == pytest.approx(r["rho"], rel=1e-12)
    assert mine["poisson_ratio"] == pytest.approx(r["nu"], rel=1e-12)


@pytest.mark.parametrize("name", sorted(TABLE))
def test_each_preset_lies_inside_its_vegetable_range(name):
    ranges = VEGETABLE_RANGES[name]
    m = PRESETS[name]
    for field in ("youngs_modulus", "fracture_energy", "density", "poisson_ratio"):
        low, high = ranges[field]
        assert low <= getattr(m, field) <= high, field


# ---- names -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("canonical", sorted(SYNONYMS))
def test_each_synonym_finds_its_preset_whatever_the_case(canonical):
    target = FIRM_TOFU if canonical == "firm_tofu" else PRESETS[canonical]
    for name in SYNONYMS[canonical]:
        assert M.material_for_name(name) is target
        assert M.material_for_name(name.upper()) is target
        assert M.material_for_name(name.lower()) is target


@pytest.mark.parametrize("name", ["tomato", "potato", "", "root_tuber", "watery_fruit", "firm_tofu", "tofu-ish"])
def test_an_unknown_name_gives_none_and_is_not_guessed(name):
    assert M.material_for_name(name) is None


# ---- tissue kinds --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", TISSUE)
def test_a_kind_is_the_geometric_mean_of_its_members_for_E_G_and_friction(kind):
    members = _members(kind)
    m, _ = M.tissue_material(kind)
    assert m.youngs_modulus == pytest.approx(_geomean([x.youngs_modulus for x in members]), rel=1e-12)
    assert m.fracture_energy == pytest.approx(_geomean([x.fracture_energy for x in members]), rel=1e-12)
    assert m.knife_friction == pytest.approx(_geomean([x.knife_friction for x in members]), rel=1e-12)


@pytest.mark.parametrize("kind", TISSUE)
def test_a_kind_is_the_arithmetic_mean_of_its_members_for_density_and_nu(kind):
    members = _members(kind)
    m, _ = M.tissue_material(kind)
    assert m.density == pytest.approx(sum(x.density for x in members) / len(members), rel=1e-12)
    assert m.poisson_ratio == pytest.approx(sum(x.poisson_ratio for x in members) / len(members), rel=1e-12)


def test_the_watery_fruit_example_of_the_request():
    m, note = M.tissue_material("watery_fruit")
    assert m.youngs_modulus == pytest.approx(math.sqrt(1.1e6 * 1.8e6), rel=1e-9)  # 1.407 MPa
    assert m.fracture_energy == pytest.approx(math.sqrt(125.0 * 705.0), rel=1e-9)  # 296.9
    assert note == "대신함(watery_fruit, cucumber, zucchini)"


def test_the_kind_rule_is_geometric_or_arithmetic_as_stated_not_by_coincidence(monkeypatch):
    # The table's members share nu and friction, so a swapped mean passes on them. Synthetic members tell the rules apart.
    a = replace(PRESETS["cucumber"], youngs_modulus=1.0e6, fracture_energy=100.0, density=900.0, poisson_ratio=0.30,
                knife_friction=0.30)
    b = replace(PRESETS["zucchini"], youngs_modulus=4.0e6, fracture_energy=400.0, density=1100.0, poisson_ratio=0.45,
                knife_friction=0.60)
    monkeypatch.setitem(M._MEMBERS, "cucumber", a)
    monkeypatch.setitem(M._MEMBERS, "zucchini", b)
    m, _ = M.tissue_material("watery_fruit")
    assert m.youngs_modulus == pytest.approx(2.0e6, rel=1e-12)  # geometric: sqrt(1e6 * 4e6)
    assert m.fracture_energy == pytest.approx(200.0, rel=1e-12)  # geometric: sqrt(100 * 400)
    assert m.knife_friction == pytest.approx(math.sqrt(0.30 * 0.60), rel=1e-12)  # geometric
    assert m.density == pytest.approx(1000.0, rel=1e-12)  # arithmetic
    assert m.poisson_ratio == pytest.approx(0.375, rel=1e-12)  # arithmetic


@pytest.mark.parametrize("kind", TISSUE)
def test_the_note_names_the_kind_and_what_stands_in(kind):
    m, note = M.tissue_material(kind)
    assert m.name == kind
    assert note == f"대신함({kind}, {', '.join(KIND_MEMBERS[kind])})"


def test_the_tissue_kinds_have_the_request_members():
    assert {kind: list(members) for kind, members in M.TISSUE_KINDS.items()} == KIND_MEMBERS


def test_the_spread_is_two_point_six_and_the_draw_ranges_follow_it():
    assert M.TISSUE_SPREAD == 2.6
    for kind in TISSUE:
        m, _ = M.tissue_material(kind)
        ranges = M.tissue_range(kind)
        assert ranges["youngs_modulus"] == pytest.approx((m.youngs_modulus / 2.6, m.youngs_modulus * 2.6), rel=1e-12)
        assert ranges["fracture_energy"] == pytest.approx((m.fracture_energy / 2.6, m.fracture_energy * 2.6), rel=1e-12)


def test_the_other_ranges_of_the_vegetable_kinds_are_the_request_values():
    assert M.TISSUE_OTHER_RANGES["poisson_ratio"] == pytest.approx((0.24, 0.45))
    assert M.TISSUE_OTHER_RANGES["density"] == pytest.approx((910.0, 1110.0))  # the union of the four ranges
    assert M.TISSUE_OTHER_RANGES["knife_friction"] == pytest.approx((0.30, 0.60))
    for kind in ("root_tuber", "watery_fruit", "bulb_scale"):
        ranges = M.tissue_range(kind)
        assert ranges["density"] == pytest.approx((910.0, 1110.0))
        assert ranges["poisson_ratio"] == pytest.approx((0.24, 0.45))
        assert ranges["knife_friction"] == pytest.approx((0.30, 0.60))


def test_gel_is_the_exception_with_nu_and_friction_fixed_and_the_tofu_density_range():
    ranges = M.tissue_range("gel")
    assert ranges["poisson_ratio"] == pytest.approx((0.40, 0.40))
    assert ranges["knife_friction"] == pytest.approx((0.30, 0.30))
    assert ranges["density"] == pytest.approx((1034.0, 1083.0))


@pytest.mark.parametrize("kind", ["root_tuber", "watery_fruit", "bulb_scale"])
def test_a_vegetable_kind_stays_elastic(kind):
    m, _ = M.tissue_material(kind)
    assert m.relaxed_modulus == 0.0 and m.relaxation_time == 0.0


def test_gel_keeps_firm_tofus_relaxation_so_it_is_the_same_tofu_model():
    m, _ = M.tissue_material("gel")
    assert m.relaxed_modulus == pytest.approx(FIRM_TOFU.relaxed_modulus, rel=1e-9)
    assert m.relaxation_time == pytest.approx(FIRM_TOFU.relaxation_time, rel=1e-12)
    assert m.viscosity == FIRM_TOFU.viscosity


# ---- drawing ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(TABLE))
def test_a_drawn_vegetable_stays_in_its_ranges_and_keeps_the_common_rule(name):
    rng = np.random.default_rng(1)
    ranges = VEGETABLE_RANGES[name]
    for _ in range(300):
        m = M.sample_material(name, rng)
        for field in ("youngs_modulus", "fracture_energy", "density", "poisson_ratio"):
            low, high = ranges[field]
            assert low <= getattr(m, field) <= high, field
        assert m.tensile_strength == pytest.approx(m.youngs_modulus / 10.0)
        assert m.knife_friction == PRESETS[name].knife_friction  # the table has no range for it: the preset's value
        assert m.initiation_length == pytest.approx(2.0e-3)
        assert m.relaxed_modulus == 0.0


def test_the_same_seed_gives_the_same_draw_and_another_seed_a_different_one():
    a = M.sample_material("carrot", np.random.default_rng(7))
    b = M.sample_material("carrot", np.random.default_rng(7))
    c = M.sample_material("carrot", np.random.default_rng(8))
    assert a == b
    assert a != c
    kind_a, _ = M.sample_tissue("watery_fruit", np.random.default_rng(7))
    kind_b, _ = M.sample_tissue("watery_fruit", np.random.default_rng(7))
    assert kind_a == kind_b


def test_the_draws_are_log_uniform_in_mean_and_variance():
    rng = np.random.default_rng(3)
    low, high = VEGETABLE_RANGES["carrot"]["youngs_modulus"]
    logs = np.log([M.sample_material("carrot", rng).youngs_modulus for _ in range(10000)])
    expected_mean = 0.5 * (math.log(low) + math.log(high))
    expected_var = (math.log(high) - math.log(low)) ** 2 / 12.0
    assert logs.mean() == pytest.approx(expected_mean, abs=0.02)
    assert logs.var() == pytest.approx(expected_var, rel=0.05)


@pytest.mark.parametrize("field", ["youngs_modulus", "fracture_energy", "density", "poisson_ratio"])
@pytest.mark.parametrize("name", sorted(TABLE))
def test_a_vegetable_is_drawn_log_uniformly_in_each_property(name, field, monkeypatch):
    # A wide range makes a log-uniform draw and a uniform one differ clearly in the mean and variance of the logs.
    low, high = 1.0e-2, 1.0e2
    monkeypatch.setitem(VEGETABLE_RANGES[name], field, (low, high))
    rng = np.random.default_rng(11)
    logs = np.log([getattr(M.sample_material(name, rng), field) for _ in range(5000)])
    assert logs.mean() == pytest.approx(0.5 * (math.log(low) + math.log(high)), abs=0.15)
    assert logs.var() == pytest.approx((math.log(high) - math.log(low)) ** 2 / 12.0, rel=0.06)


@pytest.mark.parametrize("kind", ["root_tuber", "watery_fruit", "bulb_scale"])
def test_a_kind_draws_E_and_G_log_uniformly_and_the_other_properties_uniformly(kind, monkeypatch):
    # Wide ranges make a log-uniform property show in the middle of the logs and a uniform one in the middle of the range.
    uniform = {"poisson_ratio": (0.1, 0.9), "density": (100.0, 900.0), "knife_friction": (0.1, 0.9)}
    for field, bounds in uniform.items():
        monkeypatch.setitem(M.TISSUE_OTHER_RANGES, field, bounds)
    rng = np.random.default_rng(13)
    draws = [M.sample_tissue(kind, rng)[0] for _ in range(10000)]
    m, _ = M.tissue_material(kind)
    for field in ("youngs_modulus", "fracture_energy"):
        logs = np.log([getattr(d, field) for d in draws])
        assert logs.mean() == pytest.approx(math.log(getattr(m, field)), abs=0.03)
    for field, (low, high) in uniform.items():
        values = np.array([getattr(d, field) for d in draws])
        assert values.mean() == pytest.approx(0.5 * (low + high), abs=0.01 * (high - low))


def test_gel_draws_E_and_G_log_uniformly_and_its_density_uniformly():
    rng = np.random.default_rng(17)
    draws = [M.sample_tissue("gel", rng)[0] for _ in range(20000)]
    m, _ = M.tissue_material("gel")
    for field in ("youngs_modulus", "fracture_energy"):
        logs = np.log([getattr(d, field) for d in draws])
        assert logs.mean() == pytest.approx(math.log(getattr(m, field)), abs=0.03)
    density = np.array([d.density for d in draws])
    assert density.mean() == pytest.approx(0.5 * (1034.0 + 1083.0), abs=0.5)  # uniform on 1034-1083: the middle


def test_a_drawn_kind_stays_in_its_ranges_and_the_gel_keeps_its_fixed_values():
    rng = np.random.default_rng(5)
    for kind in TISSUE:
        ranges = M.tissue_range(kind)
        for _ in range(200):
            m, note = M.sample_tissue(kind, rng)
            assert ranges["youngs_modulus"][0] <= m.youngs_modulus <= ranges["youngs_modulus"][1]
            assert ranges["fracture_energy"][0] <= m.fracture_energy <= ranges["fracture_energy"][1]
            assert ranges["density"][0] <= m.density <= ranges["density"][1]
            assert ranges["poisson_ratio"][0] <= m.poisson_ratio <= ranges["poisson_ratio"][1]
            assert ranges["knife_friction"][0] <= m.knife_friction <= ranges["knife_friction"][1]
            assert m.tensile_strength == pytest.approx(m.youngs_modulus / 10.0)
            assert note == M.tissue_material(kind)[1]
            if kind == "gel":
                assert m.poisson_ratio == 0.40 and m.knife_friction == 0.30
                assert m.relaxed_modulus / m.youngs_modulus == pytest.approx(
                    FIRM_TOFU.relaxed_modulus / FIRM_TOFU.youngs_modulus, rel=1e-9)
                assert m.relaxation_time == pytest.approx(FIRM_TOFU.relaxation_time, rel=1e-12)


# ---- setting warnings ----------------------------------------------------------------------------------------------------

def test_the_warning_class_is_a_user_warning():
    assert issubclass(SolverWarning, UserWarning)


def test_the_thresholds_are_the_request_values():
    assert MP.PI_LIMIT == 0.157
    assert MP.SIDE_MARGIN_CELLS == 5.0
    assert MP.TOP_MARGIN_CELLS == 7.0
    assert MP.WALL_BAND_CELLS == 4.5  # 3 cells of wall and 1.5 cells of B-spline reach
    assert HOLD_KAPPA == 0.0059


def test_pi_is_strict_at_its_limit():
    # G = 0.0785 J/m^2 with E = 1 Pa and dx = 0.5 m gives Pi = 0.157 exactly (no warning); 1e-6 more gives a warning.
    block = (0.5, 0.5, 0.5)
    at = dict(dx=0.5, side_margin=5 * 0.5, top_margin=7 * 0.5)
    exact = replace(FIRM_TOFU, youngs_modulus=1.0, density=1.0, fracture_energy=0.0785)
    with no_solver_warning():
        sim = _sim(exact, block=block, **at)
    assert not any(w.startswith("Pi") for w in sim.warnings)
    over = replace(exact, fracture_energy=0.0785 * (1.0 + 1e-6))
    with pytest.warns(SolverWarning, match="Pi = G_c"):
        sim = _sim(over, block=block, **at)
    assert any(w.startswith("Pi") for w in sim.warnings)


def test_pi_warns_above_0_157_and_not_below_it():
    dx, E = 4e-3, FIRM_TOFU.youngs_modulus
    below = replace(FIRM_TOFU, fracture_energy=0.99 * 0.157 * E * dx)  # Pi = G / (E dx) = 0.99 x 0.157
    above = replace(FIRM_TOFU, fracture_energy=1.01 * 0.157 * E * dx)
    with no_solver_warning():
        assert _sim(below, dx=dx).warnings == []
    with pytest.warns(SolverWarning, match="Pi = G_c"):
        sim = _sim(above, dx=dx)
    assert len(sim.warnings) == 1 and sim.warnings[0].startswith("Pi = G_c / (E dx) = ")


def test_the_demo_firm_tofu_single_setup_gives_no_setting_warning():
    # The demo's defaults: dx 2.5 mm, default margins, hold none. Pi 0.045, rho g H / E 0.0118 (the request's values).
    dx = 2.5e-3
    pi = FIRM_TOFU.fracture_energy / (FIRM_TOFU.youngs_modulus * dx)
    kappa = FIRM_TOFU.density * 9.81 * BLOCK[2] / FIRM_TOFU.youngs_modulus
    assert pi == pytest.approx(0.045, abs=5e-4)
    assert kappa == pytest.approx(0.0118, abs=5e-5)
    with no_solver_warning():
        sim = _sim(FIRM_TOFU, dx=dx)
    assert sim.warnings == []


def test_no_hold_warns_for_a_light_block_and_not_for_firm_tofu_at_1_5_cm():
    g, H, dx = 9.81, 0.015, 4e-3
    assert 1060.0 * g * H / FIRM_TOFU.youngs_modulus > HOLD_KAPPA  # firm tofu's 1.5 cm block: 0.005908, just above
    with no_solver_warning():
        assert _sim(FIRM_TOFU, block=(0.04, 0.025, H), dx=dx).warnings == []
    light = replace(FIRM_TOFU, density=0.5 * HOLD_KAPPA * FIRM_TOFU.youngs_modulus / (g * H))
    with pytest.warns(SolverWarning, match="no hold"):
        sim = _sim(light, block=(0.04, 0.025, H), dx=dx)
    assert len(sim.warnings) == 1 and sim.warnings[0].startswith("no hold")


def test_no_hold_is_strict_below_hold_kappa_and_silent_at_it():
    # Exact in binary: density 1, E 1, H 0.25, gravity HOLD_KAPPA / 0.25, so rho g H / E equals HOLD_KAPPA exactly.
    rig = replace(FIRM_TOFU, youngs_modulus=1.0, density=1.0, fracture_energy=1.0e-6)  # Pi stays far below its limit
    at = dict(block=(0.1, 0.1, 0.25), dx=0.05, side_margin=5 * 0.05, top_margin=7 * 0.05, gravity=HOLD_KAPPA / 0.25)
    sim = _sim(rig, **at)
    assert not any(w.startswith("no hold") for w in sim.warnings)
    below = dict(at, gravity=0.999 * HOLD_KAPPA / 0.25)
    with pytest.warns(SolverWarning, match="no hold"):
        sim = _sim(rig, **below)
    assert any(w.startswith("no hold") for w in sim.warnings)


def test_a_pressed_block_does_not_warn_about_no_hold():
    light = replace(FIRM_TOFU, density=500.0)
    with no_solver_warning():
        sim = _sim(light, block=(0.04, 0.025, 0.015), dx=4e-3, hold="top_pressure")
    assert not any(w.startswith("no hold") for w in sim.warnings)


@pytest.mark.parametrize("side, warns", [(5 * 4e-3, False), (5 * 4e-3 - 1e-6, True)])
def test_side_margin_warns_below_five_dx_and_not_at_it(side, warns):
    config = dict(dx=4e-3, side_margin=side, top_margin=7 * 4e-3)
    if warns:
        with pytest.warns(SolverWarning, match="side margin"):
            sim = _sim(FIRM_TOFU, **config)
    else:
        with no_solver_warning():
            sim = _sim(FIRM_TOFU, **config)
    assert any(w.startswith("the empty space is too narrow (side margin") for w in sim.warnings) == warns


@pytest.mark.parametrize("top, warns", [(7 * 4e-3, False), (7 * 4e-3 - 1e-6, True)])
def test_top_margin_warns_below_seven_dx_and_not_at_it(top, warns):
    config = dict(dx=4e-3, side_margin=5 * 4e-3, top_margin=top)
    if warns:
        with pytest.warns(SolverWarning, match="top margin"):
            sim = _sim(FIRM_TOFU, **config)
    else:
        with no_solver_warning():
            sim = _sim(FIRM_TOFU, **config)
    assert any("top margin" in w for w in sim.warnings) == warns


def test_the_test_blocks_jig_margins_give_one_margin_warning_naming_both():
    # The tests' CONFIG: side 12 mm and top 16 mm at 4 mm. Both are short, and the kind is reported once.
    with pytest.warns(SolverWarning, match="margin"):
        sim = _sim(FIRM_TOFU, block=(0.032, 0.024, 0.016), dx=4e-3, side_margin=0.012, top_margin=0.016)
    assert len(sim.warnings) == 1
    assert "side margin" in sim.warnings[0] and "top margin" in sim.warnings[0]


def test_a_kind_is_reported_once_per_simulation_however_often_it_is_hit():
    with pytest.warns(SolverWarning, match="margin"):
        sim = _sim(FIRM_TOFU, block=(0.032, 0.024, 0.016), dx=4e-3, side_margin=0.012, top_margin=0.016)
    n = len(sim.warnings)
    sim._warn_once("margin", "again")
    sim._warn_once("margin", "again")
    assert len(sim.warnings) == n


def _place(sim, index, xyz):
    pts = sim.x.numpy().astype(np.float32).copy()
    pts[index] = xyz
    sim.x = wp.array(pts, dtype=wp.vec3, device="cpu")


def _mid_block(sim):
    return [0.5 * (sim.block_min[i] + sim.block_max[i]) for i in range(3)]


def test_a_particle_within_4_5_cells_of_a_side_sets_the_wall_warning_once():
    dx = 4e-3
    sim = _sim(FIRM_TOFU, dx=dx)
    _place(sim, 0, [4.4 * dx, 0.5 * sim.domain_size[1], _mid_block(sim)[2]])
    with pytest.warns(SolverWarning, match="wall band") as record:
        sim.advance(sim.time + sim.dt)
        sim.advance(sim.time + sim.dt)
    assert sum("wall band" in w for w in sim.warnings) == 1
    assert sum("wall band" in str(r.message) for r in record) == 1


@pytest.mark.parametrize("face", ["low_x", "high_x", "low_y", "high_y", "top"])
def test_each_side_and_the_top_is_a_wall_band(face):
    dx = 4e-3
    sim = _sim(FIRM_TOFU, dx=dx)
    Lx, Ly, Lz = (float(v) for v in sim.domain_size)
    xyz = [0.5 * Lx, 0.5 * Ly, _mid_block(sim)[2]]
    if face == "low_x":
        xyz[0] = 4.4 * dx
    elif face == "high_x":
        xyz[0] = Lx - 4.4 * dx
    elif face == "low_y":
        xyz[1] = 4.4 * dx
    elif face == "high_y":
        xyz[1] = Ly - 4.4 * dx
    else:
        xyz[2] = Lz - 0.75 * dx
    _place(sim, 0, xyz)
    with pytest.warns(SolverWarning, match="wall band"):
        sim.advance(sim.time + sim.dt)


def test_a_particle_just_outside_4_5_cells_does_not_set_the_wall_warning():
    dx = 4e-3
    sim = _sim(FIRM_TOFU, dx=dx)
    _place(sim, 0, [4.6 * dx, 0.5 * sim.domain_size[1], _mid_block(sim)[2]])
    with no_solver_warning():
        sim.advance(sim.time + sim.dt)
    assert not any("wall band" in w for w in sim.warnings)


def test_the_board_is_not_a_wall():
    dx = 4e-3
    sim = _sim(FIRM_TOFU, dx=dx)
    _place(sim, 0, [0.5 * sim.domain_size[0], 0.5 * sim.domain_size[1], sim.board_z + 0.5e-3])
    with no_solver_warning():
        sim.advance(sim.time + sim.dt)
    assert not any("wall band" in w for w in sim.warnings)


def test_the_flag_is_cleared_after_each_advance():
    dx = 4e-3
    sim = _sim(FIRM_TOFU, dx=dx)
    _place(sim, 0, [4.4 * dx, 0.5 * sim.domain_size[1], _mid_block(sim)[2]])
    with pytest.warns(SolverWarning, match="wall band"):
        sim.advance(sim.time + sim.dt)
    assert int(sim.wall_band.numpy()[0]) == 0


def test_the_default_single_setup_stays_silent_through_a_short_run():
    sim = _sim(FIRM_TOFU, dx=2.5e-3)
    with no_solver_warning():
        sim.advance(sim.time + 0.02)
    assert sim.warnings == []


def test_the_sweep_writes_its_warnings_into_the_json(tmp_path):
    def run(name, *extra):
        out = tmp_path / f"{name}.json"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "cutting" / "solver_sweep.py"), "--block", "0.032", "0.024", "0.016",
             "--dx", "0.004", "--clearance", "0.003", "--start-time", "0.08", "--stop-at-board", *extra, "--out", str(out)],
            capture_output=True, text=True,
            env={"PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
            timeout=900,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        return json.loads(out.read_text())

    loud = run("loud", "--gc", "1000")  # G_c 1000 J/m^2 at 4 mm: Pi = 9.5, far above its limit
    assert any(w.startswith("Pi = G_c") for w in loud["warnings"])
    quiet = run("quiet")  # the sweep's defaults for this block: nothing to warn about
    assert quiet["warnings"] == []  # the key is there even when the list is empty


def test_the_demo_writes_its_warnings_into_metrics_json(tmp_path):
    # A 2 cm block at 4 mm with G_c 1000 J/m^2: the demo runs in about 15 s and warns about Pi and the wall band.
    out = tmp_path / "demo"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "cutting" / "tofu_cutting_demo.py"), "--block", "0.02", "0.02", "0.008",
         "--dx", "0.004", "--fracture-energy", "1000", "--no-video", "--no-overlay", "--png-every", "1000", "--out", str(out)],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    metrics = json.loads((out / "metrics.json").read_text())
    assert any(w.startswith("Pi = G_c") for w in metrics["warnings"])
