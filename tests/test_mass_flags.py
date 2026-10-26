# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""simfoundry.utils.mass_estimation.mass_flags, and stage 11's wiring of it (no GPU: the stage is read as source)."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from simfoundry.utils import mass_estimation as me  # noqa: E402

STAGE11 = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "A_reconstruction" / "stages" / \
    "11_make_objects_sim_ready.py"

# A geometry whose mass in kg equals its density in kg/m^3: 1 m^3 is 1e6 cm^3.
UNIT_VOLUME = 1e6


def geo(volume_cm3=UNIT_VOLUME):
    return {"volume_cm3": volume_cm3, "area_cm2": 1.0, "hull_cm3": 1.0, "fill": None, "extent_cm": [1.0, 1.0, 1.0]}


def flags_for(material, mass_kg, volume_cm3=UNIT_VOLUME, typical_g=None, material_g=None):
    return me.mass_flags(mass_kg, typical_g, material_g, material, geo(volume_cm3))


# (family, keywords, (lo, hi) in kg/m^3), as the brief gives the table
CASES = [
    ("steel", ["steel", "iron", "stainless"], (7000, 8100)),
    ("aluminum", ["aluminum", "aluminium"], (2500, 2900)),
    ("copper", ["copper", "brass", "bronze"], (7400, 9000)),
    ("glass", ["glass"], (2200, 2800)),
    ("ceramic", ["ceramic", "porcelain", "stoneware", "clay"], (1800, 2900)),
    ("foam", ["foam", "sponge"], (10, 400)),
    ("rubber", ["rubber", "silicone"], (900, 1600)),
    ("plastic", ["plastic", "polymer", "acrylic", "nylon", "abs", "polypropylene", "polyethylene", "pvc", "pet"],
     (850, 2300)),
    ("wood", ["wood", "bamboo"], (150, 1300)),
    ("paper", ["paper", "cardboard"], (200, 1300)),
    ("stone", ["stone", "concrete", "marble", "granite"], (1800, 3100)),
    ("fabric", ["fabric", "cloth", "textile", "leather"], (100, 1600)),
    ("food", ["fruit", "vegetable", "tofu", "meat", "fish", "cheese", "bread", "dough", "food"], (250, 1300)),
    ("liquid", ["water", "juice", "oil", "liquid"], (700, 1450)),
]
KEYWORD_CASES = [(kw, family, lo, hi) for family, kws, (lo, hi) in CASES for kw in kws]


# --- clamp -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("material_g, expected", [
    (100, "low"), (249.9, "low"), (250, None), (300, None), (1000, None), (1000.1, "high"), (1500, "high"),
])
def test_clamp_side_against_typical_500(material_g, expected):
    # 250 and 1000 are exactly typical/2 and typical*2: combine_masses leaves them alone, so no flag.
    assert flags_for(None, 1.0, typical_g=500, material_g=material_g)["clamped"] == expected


@pytest.mark.parametrize("typical_g, material_g", [(None, 100), (500, None), (None, None), (0, 100), (500, 0)])
def test_no_clamp_without_both_estimates(typical_g, material_g):
    # A zero counts as no estimate, as in combine_masses, which then returns the other value unclamped.
    assert flags_for(None, 1.0, typical_g=typical_g, material_g=material_g)["clamped"] is None


def test_clamp_and_density_flags_are_reported_together():
    flags = flags_for({"material": "steel"}, 1.0, typical_g=500, material_g=100)
    assert flags["clamped"] == "low"
    assert flags["flags"] == ["clamped_low", "density_low"]


# --- density ---------------------------------------------------------------------------------------------------------

def test_density_is_mass_over_enclosed_volume_in_kg_per_m3():
    # 0.25 kg in 100 cm^3 (1e-4 m^3) is 2500 kg/m^3, the lower edge of aluminium's range: inside, not flagged.
    flags = flags_for({"material": "aluminum"}, 0.25, volume_cm3=100.0)
    assert flags["density_kg_m3"] == pytest.approx(2500.0)
    assert flags["density_family"] == "aluminum"
    assert flags["density_outside_range"] is False
    assert flags["flags"] == []


@pytest.mark.parametrize("keyword, family, lo, hi", KEYWORD_CASES)
def test_each_keyword_names_its_family_and_range(keyword, family, lo, hi):
    mid = (lo + hi) / 2
    inside = flags_for({"material": keyword}, mid)
    assert inside["density_family"] == family
    assert inside["density_range_kg_m3"] == [lo, hi]
    assert inside["density_checked"] is True
    assert inside["density_outside_range"] is False
    assert inside["flags"] == []
    assert flags_for({"material": keyword}, lo - 1)["flags"] == ["density_low"]
    assert flags_for({"material": keyword}, hi + 1)["flags"] == ["density_high"]


@pytest.mark.parametrize("keyword, family, lo, hi", KEYWORD_CASES)
def test_range_bounds_count_as_inside(keyword, family, lo, hi):
    for density in (lo, hi):
        flags = flags_for({"material": keyword}, density)
        assert flags["density_outside_range"] is False
        assert flags["flags"] == []


@pytest.mark.parametrize("material, family", [
    ("stainless steel pot", "steel"),
    ("STEEL", "steel"),
    ("Cast Iron", "steel"),
    ("plastic-coated rubber", "rubber"),  # rubber is listed before plastic, so it wins the tie
    ("foam rubber", "foam"),  # foam is listed before rubber
    ("plastic foam", "foam"),
])
def test_first_listed_family_wins_and_matching_ignores_case(material, family):
    assert flags_for({"material": material}, 1.0)["density_family"] == family


@pytest.mark.parametrize("material, family", [
    ("wooden", "wood"),  # a keyword starts a word
    ("steelwork", "steel"),
    ("carpet", None),  # "pet" is inside a word, not a word of its own
    ("foil", None),  # "oil" is inside a word
    ("environment", None),  # "iron" is inside a word
    ("absorbent", None),  # "abs" starts a word but is not a whole word
    ("boiled", None),
    ("PET bottle", "plastic"),  # three letters or fewer are whole words
    ("ABS plastic", "plastic"),
    ("oil", "liquid"),
    ("olive oil", "liquid"),
])
def test_keywords_match_whole_words_at_word_starts(material, family):
    assert flags_for({"material": material}, 1.0)["density_family"] == family


@pytest.mark.parametrize("material", [{"material": "unobtainium"}, {"material": None}, {"material": ""}, None])
def test_unknown_or_missing_material_uses_the_global_range(material):
    flags = flags_for(material, 1000.0)
    assert flags["density_family"] is None
    assert flags["density_range_kg_m3"] == [50, 12000]
    assert flags["density_checked"] is True
    assert flags["flags"] == []


def test_global_range_flags_the_implausible_ends():
    assert flags_for({"material": "unobtainium"}, 49.0)["flags"] == ["density_low"]
    assert flags_for({"material": "unobtainium"}, 50.0)["flags"] == []
    assert flags_for({"material": "unobtainium"}, 12000.0)["flags"] == []
    assert flags_for({"material": "unobtainium"}, 12001.0)["flags"] == ["density_high"]


# --- shell and missing volume ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("form", ["shell", "Shell", "shell, thin wall"])
def test_shell_is_not_density_checked(form):
    # 0.25 kg over 100 cm^3 is 2500 kg/m^3, under steel's range: it would be flagged if it were checked.
    flags = me.mass_flags(0.25, None, None, {"material": "steel", "form": form}, geo(volume_cm3=100.0))
    assert flags["density_checked"] is False
    assert flags["density_outside_range"] is None
    assert flags["flags"] == []


def test_shell_keeps_its_clamp():
    flags = me.mass_flags(0.25, 500, 100, {"material": "steel", "form": "shell"}, geo(volume_cm3=100.0))
    assert flags["clamped"] == "low"
    assert flags["density_checked"] is False
    assert flags["flags"] == ["clamped_low"]


@pytest.mark.parametrize("volume_cm3", [None, 0.0])
def test_no_enclosed_volume_means_no_density(volume_cm3):
    # None is an open mesh; 0.0 is a degenerate one. Neither may divide by zero, and the clamp still stands.
    flags = me.mass_flags(0.25, 500, 100, {"material": "steel"}, geo(volume_cm3=volume_cm3))
    assert flags["density_kg_m3"] is None
    assert flags["density_checked"] is False
    assert flags["density_outside_range"] is None
    assert flags["clamped"] == "low"
    assert flags["flags"] == ["clamped_low"]


# --- flag order and content ------------------------------------------------------------------------------------------

def test_clamp_flag_comes_before_density_flag():
    # Clamped low to 250 g, then 0.25 kg over 10 cm^3 is 25000 kg/m^3, over steel's range.
    flags = me.mass_flags(0.25, 500, 100, {"material": "steel"}, geo(volume_cm3=10.0))
    assert flags["flags"] == ["clamped_low", "density_high"]


def test_clamp_high_with_density_low():
    flags = me.mass_flags(1.0, 500, 1500, {"material": "steel"}, geo())
    assert flags["flags"] == ["clamped_high", "density_low"]


def test_result_is_json_serialisable_with_plain_python_types():
    flags = me.mass_flags(np.float64(0.25), np.float64(500), np.float32(100), {"material": "steel"},
                          {"volume_cm3": np.float64(100.0), "area_cm2": 1.0, "hull_cm3": 1.0, "fill": None,
                           "extent_cm": [1.0, 1.0, 1.0]})
    json.dumps(flags)
    assert type(flags["density_kg_m3"]) is float
    assert type(flags["density_checked"]) is bool
    assert type(flags["density_outside_range"]) is bool
    assert not [v for v in flags.values() if isinstance(v, np.generic)]


# --- estimate_rigid_mass integration ---------------------------------------------------------------------------------

class _AnsweringVLM:
    """Answers the typical question and the material question with the given dicts, told apart by their text."""

    def __init__(self, typical, material):
        self.typical, self.material = typical, material

    def __call__(self, prompt, **kwargs):
        return "material" if "describe how it is built" in prompt else "typical"

    def get_result_text(self, result):
        answer = self.material if result == "material" else self.typical
        return f"```json\n{json.dumps(answer)}\n```"


def _estimate(typical, material, tm):
    obb = np.array([0.1, 0.1, 0.1])  # the unit box: 1000 cm^3
    return me.estimate_rigid_mass(_AnsweringVLM(typical, material), "a pot", "img.png", tm, obb)


def test_integration_low_clamp_on_a_closed_box():
    tm = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    # Plastic at 1.0 g/cm^3 over 1000 cm^3 is 1000 g; the typical answer 2.5 kg puts the half-way point at 1250 g.
    estimate = _estimate({"mass": 2.5, "friction": 0.4}, {"material": "plastic", "density": 1.0, "form": "solid",
                                                           "friction": 0.3}, tm)
    flags = me.mass_flags(estimate["mass_kg"], estimate["typical_g"], estimate["material_g"], estimate["material"],
                          estimate["geometry"])
    assert estimate["mass_kg"] == pytest.approx(1.25)
    assert flags["clamped"] == "low"
    assert flags["density_family"] == "plastic"
    assert flags["density_kg_m3"] == pytest.approx(1250.0)
    assert flags["flags"] == ["clamped_low"]


def test_integration_high_clamp_with_steel_density_low():
    tm = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    # Steel at 7.8 g/cm^3 is 7800 g; the typical 0.5 kg caps it at 1000 g, which is 1000 kg/m^3: under steel's range.
    estimate = _estimate({"mass": 0.5, "friction": 0.4}, {"material": "steel", "density": 7.8, "form": "solid"}, tm)
    flags = me.mass_flags(estimate["mass_kg"], estimate["typical_g"], estimate["material_g"], estimate["material"],
                          estimate["geometry"])
    assert estimate["mass_kg"] == pytest.approx(1.0)
    assert flags["flags"] == ["clamped_high", "density_low"]


def test_integration_open_sheet_is_not_density_checked():
    sheet = trimesh.Trimesh(vertices=[[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]], faces=[[0, 1, 2], [0, 2, 3]],
                            process=False)
    estimate = _estimate({"mass": 2.5, "friction": 0.4}, {"material": "plastic", "density": 1.0, "form": "solid"}, sheet)
    assert estimate["geometry"]["volume_cm3"] is None
    flags = me.mass_flags(estimate["mass_kg"], estimate["typical_g"], estimate["material_g"], estimate["material"],
                          estimate["geometry"])
    assert flags["density_checked"] is False
    assert flags["density_kg_m3"] is None
    assert flags["clamped"] == "low"


# --- stage 11 wiring (read as source; the stage imports Isaac-side modules and cannot run here) ----------------------

def _stage11():
    tree = ast.parse(STAGE11.read_text(encoding="utf-8"), str(STAGE11))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "import_rigid_scene_object")
    parents = {child: node for node in ast.walk(function) for child in ast.iter_child_nodes(node)}
    return tree, function, parents


def _ancestor_ifs(node, parents):
    found = []
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            found.append(node)
    return found


def _compares(expr):
    return {ast.unparse(node) for node in ast.walk(expr) if isinstance(node, ast.Compare)}


def test_stage11_imports_mass_flags_at_module_level():
    tree, _, _ = _stage11()
    imported = [alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
                and node.module == "simfoundry.utils.mass_estimation" for alias in node.names]
    assert "mass_flags" in imported


def test_mass_flags_is_called_only_under_the_estimate_mass_guard():
    _, function, parents = _stage11()
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "mass_flags"]
    assert len(calls) == 1
    guards = [i for i in _ancestor_ifs(calls[0], parents) if "estimate['mass_kg'] is not None" in _compares(i.test)]
    assert len(guards) == 1
    assert "estimate is not None" in _compares(guards[0].test)


def test_mass_flags_result_is_stored_under_out_mass_flags():
    _, function, _ = _stage11()
    stores = [a for a in ast.walk(function) if isinstance(a, ast.Assign) and len(a.targets) == 1
              and isinstance(a.targets[0], ast.Subscript) and isinstance(a.targets[0].slice, ast.Constant)
              and a.targets[0].slice.value == "mass_flags"]
    assert len(stores) == 1
    assert ast.unparse(stores[0].targets[0].value) == "out"
    assert isinstance(stores[0].value, ast.Call) and stores[0].value.func.id == "mass_flags"


def test_mass_flags_receives_the_mass_that_was_written_and_the_estimate_fields_in_order():
    _, function, _ = _stage11()
    call = next(n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "mass_flags")
    assert not call.keywords
    assert [ast.unparse(arg) for arg in call.args] == [
        "mass", "estimate['typical_g']", "estimate['material_g']", "estimate['material']", "estimate['geometry']"]
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "mass"
    for arg in call.args[1:]:
        assert isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name) and arg.value.id == "estimate"
        assert isinstance(arg.slice, ast.Constant) and isinstance(arg.slice.value, str)


def test_mass_is_assigned_only_from_the_estimate_or_the_direct_question():
    _, function, _ = _stage11()
    stores = [n for n in ast.walk(function) if isinstance(n, ast.Name) and n.id == "mass"
              and isinstance(n.ctx, ast.Store)]
    assert len(stores) == 2
    from_estimate = [a for a in ast.walk(function) if isinstance(a, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "mass" for t in a.targets)]
    assert len(from_estimate) == 1
    assert ast.unparse(from_estimate[0].value) == "estimate['mass_kg']"


def test_flagged_mass_is_logged_as_a_warning():
    _, function, _ = _stage11()
    flagged = [n for n in ast.walk(function) if isinstance(n, ast.If) and "flags" in ast.unparse(n.test)]
    assert len(flagged) == 1
    warnings = [c for c in ast.walk(flagged[0]) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "warning" and ast.unparse(c.func.value) == "logger"]
    assert len(warnings) == 1
