# SPDX-License-Identifier: Apache-2.0
"""The CPU-only helpers of scripts/cutting/food (the food-cutting robot episodes): slice thickness, per-episode thickness
plans, the gripper column of the LeRobot export, the scene looks, the knife force cap and the rigid hull fallback.
None of them needs Isaac Sim, OmniGibson or a GPU."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cutting" / "food"))
sys.path.insert(0, str(ROOT / "scripts" / "cutting" / "policy"))

SP = 0.835e-3  # particle spacing [m]


def _slab(x0, x1, ny=12, nz=12):
    xs = np.arange(x0 + SP / 2, x1, SP)
    g = np.stack(np.meshgrid(xs, np.arange(ny) * SP, np.arange(nz) * SP, indexing="ij"), -1).reshape(-1, 3)
    return g


def test_column_thickness_ignores_a_few_particles_of_the_next_layer():
    fre = importlib.import_module("food_robot_episodes")
    slab = _slab(0.0, 3.0e-3)
    stray = np.array([[3.0e-3 + 1.2e-3, 0.0, 0.0], [3.0e-3 + 1.2e-3, SP, 0.0]])  # two particles of the next layer
    P = np.concatenate([slab, stray])
    extent = P[:, 0].max() - P[:, 0].min() + SP
    col = fre.col_thickness(P, SP)
    assert extent > 4.0e-3  # the old judge (x extent) is pulled past 4 mm by two particles
    assert abs(col - 3.0e-3) < 0.5e-3  # the column median stays at the slab's thickness


def test_slice_report_judges_by_the_column_thickness_and_keeps_the_extent():
    fre = importlib.import_module("food_robot_episodes")
    body = _slab(-20.0e-3, 0.0)
    slab = _slab(0.0, 3.0e-3)
    stray = np.array([[-1.2e-3, 0.0, 0.0]])  # one body-side particle that went with the slice
    x = np.concatenate([body, slab, stray])
    lab = np.concatenate([np.zeros(len(body), int), np.ones(len(slab), int), [1]])
    rep = fre.slice_report(x, lab, [0.0], SP, [3.0])
    assert len(rep) == 1 and rep[0]["separated"]
    assert rep[0]["thick_ok"]
    assert rep[0]["thickness_extent_mm"] > rep[0]["thickness_mm"]


def test_thickness_plan_is_chosen_per_seed_and_fits_the_food(tmp_path):
    fre = importlib.import_module("food_robot_episodes")
    shape = tmp_path / "stub"
    (tmp_path / "stub.json").write_text(json.dumps({"bounds": [[0.0, -0.01, 0.0], [0.040, 0.01, 0.02]]}))
    argv = ["--veg", "cucumber", "--shape", str(shape), "--plan", "3,3,3,3", "--wet", "1"]
    plans = {fre.plan_with_thickness(argv, "3,5,8", 4, s)[5] for s in range(12)}
    assert plans == {"3,3,3,3", "5,5,5,5", "8,8,8"}  # 40 mm food, 16 mm kept for the hand: three 8 mm cuts at most
    assert fre.plan_with_thickness(argv, "3,5,8", 4, 7) == fre.plan_with_thickness(argv, "3,5,8", 4, 7)
    assert argv[5] == "3,3,3,3"  # the caller's list is not changed


def test_gripper_column_from_phases_or_recorded_commands():
    conv = importlib.import_module("food_npz_to_lerobot")
    cmd = np.zeros((6, 9), np.float32)
    phases = np.array(["approach", "descend", "close", "lift", "release", "retreat"])
    g = conv.grip_column({"cmd": cmd, "tick_phase": phases}, {"mode": "expert_move"})
    assert g.tolist() == [1, 1, 0, 0, 1, 1]
    assert conv.grip_column({"cmd": cmd, "tick_phase": phases}, {"mode": "expert_plan"}).tolist() == [0] * 6  # the knife is held
    rec = np.array([1, 0, 0, 0, 0, 1], np.float32)
    assert conv.grip_column({"cmd": cmd, "tick_phase": phases, "grip_cmd": rec}, {"mode": "expert_move"}).tolist() == rec.tolist()


def test_scene_looks(tmp_path, monkeypatch):
    tex = tmp_path / "tex"
    tex.mkdir()
    idx = {f"{k}_{i}": {"file": f"{k}_{i}.png", "size_m": [1, 1], "kind": k} for k in ("board", "counter", "wall") for i in range(3)}
    idx["board_white_plastic"] = {"file": "w.png", "size_m": [0.34, 0.24], "kind": "board"}
    (tex / "index.json").write_text(json.dumps(idx))
    monkeypatch.setenv("FOOD_LOOK_TEXTURES", str(tex))
    monkeypatch.setenv("FOOD_LOOK_HDRI", str(tmp_path / "no_hdri"))
    fl = importlib.reload(importlib.import_module("food_look"))
    assert fl.sample_look("base", 0) is None
    real = fl.sample_look("real", 0)
    assert real["board"]["tex"] == "board_white_plastic" and real["kind"] == "real"
    a, b = fl.sample_look("random", 3), fl.sample_look("random", 3)
    assert a == b  # a seed gives the same look
    c = fl.sample_look("random", 4)
    assert a != c
    for lk in (a, c):
        assert idx[lk["board"]["tex"]]["kind"] == "board"
        assert 2700 <= lk["ceiling"]["kelvin"] <= 6500
        assert all(abs(v) <= 0.02 for v in lk["cam"]["front_dpos"])
    json.dumps(a)  # stored in the episode's json


def test_force_cap_presets_and_admittance():
    fc = importlib.import_module("force_cap")
    assert fc.parse_cap("elrobot_motor") == pytest.approx(18.6)
    assert fc.parse_cap("none") is None and fc.parse_cap(None) is None
    assert fc.parse_cap("12") == pytest.approx(12.0)
    with pytest.raises(ValueError):
        fc.parse_cap("strong")


def test_hull_falls_back_on_flat_points():
    fh = importlib.import_module("food_handoff")
    rng = np.random.default_rng(0)
    flat = np.c_[rng.uniform(-1, 1, (50, 2)), np.zeros(50)] * 1e-3  # all points in one plane: plain qhull refuses
    V, F = fh._hull(flat)
    assert len(V) >= 3 and len(F) >= 1
