# SPDX-License-Identifier: Apache-2.0
"""Pressing the top skin of a block (SimConfig.hold = "top_pressure"): which particles are pressed, how hard they push,
and that the default presses nothing. CPU at coarse resolution."""

import importlib.util
import json
import math
import subprocess
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from simfoundry.cutting.knife import KnifePose, KnifeTrajectory  # noqa: E402
from simfoundry.cutting.materials import FIRM_TOFU  # noqa: E402
from simfoundry.cutting.mpm import HOLD_KAPPA, HOLD_RESERVED, MPMCuttingSim, SimConfig  # noqa: E402

DX = 4e-3
BLOCK = (0.032, 0.024, 0.016)
ACROSS = -math.pi / 2.0  # blade along y: its normal e_v is the x axis
TOP_LAYER = 16 * 12  # the 32 x 24 mm top at 2 mm particle spacing
ROOT = Path(__file__).resolve().parents[1]


def _sim(material=FIRM_TOFU, **kwargs):
    config = SimConfig(dx=DX, side_margin=0.012, top_margin=0.016, **kwargs)
    return MPMCuttingSim(material, BLOCK, config=config, device="cpu")


def _top_layer_mask(sim):
    """Particles whose start lies in the top lattice layer: within one spacing (dx / 2) of the top, jitter included."""
    return sim.x0[:, 2] > sim.block_max[2] - DX / 2


def test_a_box_presses_its_top_layer_and_nothing_else():
    sim = _sim(hold="top_pressure")
    pressed = sim._hold_flag.numpy().astype(bool)
    assert sim.hold_top_count == TOP_LAYER
    assert pressed.sum() == TOP_LAYER
    assert np.array_equal(pressed, _top_layer_mask(sim))


def test_the_band_beside_the_blade_is_left_out_while_the_knife_is_on():
    sim = _sim(hold="top_pressure")
    centre = 0.5 * (sim.block_min + sim.block_max)
    origin = np.array([centre[0], centre[1], sim.block_max[2] + 0.01])
    pose = KnifePose(origin, ACROSS)
    sim.set_knife(pose)
    x = sim.x.numpy().astype(float)
    distance = np.abs((x - origin) @ pose.axes()[1])  # the blade's plane: |(x - origin) . e_v|, as the kernel reads it
    expected = int(np.sum(_top_layer_mask(sim) & (distance >= sim.hold_band)))
    sim._apply_hold(1.0e-4)
    assert expected == 120  # 6 of the 16 columns (1.5 dx = 6 mm either side of the blade) are in the band
    assert sim.hold_count_last == expected
    assert sim.hold_count_last == TOP_LAYER - 72


def test_no_band_without_a_knife():
    sim = _sim(hold="top_pressure")
    sim._apply_hold(1.0e-4)
    assert sim.hold_count_last == TOP_LAYER


def test_a_particle_on_the_blade_plane_is_pressed_when_the_band_is_zero():
    # Strict edge: |distance| < band. The particle sits exactly on the plane, so with band 0 it must still be pressed.
    sim = _sim(hold="top_pressure", hold_band=0.0)
    first = int(np.flatnonzero(sim._hold_flag.numpy())[0])
    origin = sim.x.numpy()[first].astype(float)
    sim.set_knife(KnifePose(origin, ACROSS))
    sim._apply_hold(1.0e-4)
    assert sim.hold_count_last == TOP_LAYER


def test_the_band_defaults_to_one_and_a_half_cells():
    assert _sim(hold="top_pressure").hold_band == pytest.approx(1.5 * DX, rel=1e-12)


def test_the_pressing_force_is_the_pressure_times_the_top_area_and_points_down():
    sim = _sim(hold="top_pressure")
    p = HOLD_KAPPA * FIRM_TOFU.youngs_modulus
    assert sim.hold_pressure == pytest.approx(p, rel=1e-12)
    before = sim.grid_v.numpy().sum(axis=(0, 1, 2, 3)).astype(float)
    sim._apply_hold(1.0e-4)
    area = BLOCK[0] * BLOCK[1]
    assert sim.hold_force_last == pytest.approx(p * area, rel=0.02)
    # The momentum added to the grid sums to -F dt in z and nothing sideways (the B-spline weights sum to one).
    added = sim.grid_v.numpy().sum(axis=(0, 1, 2, 3)).astype(float) - before
    assert added[0] == pytest.approx(0.0, abs=1e-12)
    assert added[1] == pytest.approx(0.0, abs=1e-12)
    assert added[2] == pytest.approx(-p * area * 1.0e-4, rel=1e-4)


def test_each_pressed_particle_pushes_the_grid_of_its_own_piece():
    # After a cut the pressed skin can belong to two pieces. Each particle's push must land on its own piece's grid,
    # not on the sum of all pieces, which is all the test above can see.
    sim = _sim(hold="top_pressure")
    pressed = np.flatnonzero(sim._hold_flag.numpy())
    piece = np.zeros(sim.n_particles, dtype=np.int32)
    piece[pressed[::2]] = 1  # every other pressed particle belongs to piece 1
    sim.piece = wp.array(piece, dtype=int, device="cpu")
    sim._apply_hold(1.0e-4)
    per_piece = sim.grid_v.numpy().astype(float).sum(axis=(1, 2, 3))  # (pieces, 3): momentum summed over each piece's nodes
    n1 = int(np.count_nonzero(piece[pressed] == 1))
    n0 = len(pressed) - n1
    assert n0 > 0 and n1 > 0
    push = sim.hold_particle_force * 1.0e-4  # momentum one pressed particle gives, F dt
    assert per_piece[1, 2] == pytest.approx(-n1 * push, rel=1e-4)
    assert per_piece[0, 2] == pytest.approx(-n0 * push, rel=1e-4)
    assert np.allclose(per_piece[:, :2], 0.0, atol=1e-12)
    assert np.allclose(per_piece[2:], 0.0, atol=1e-12)


def test_the_default_presses_nothing_and_substeps_do_not_press():
    sim = _sim()
    assert sim.hold_top_count == 0
    assert not sim._hold_flag.numpy().any()
    assert sim.hold_pressure == 0.0
    sim._apply_hold(1.0e-4)
    assert sim.hold_count_last == 0
    assert sim.hold_force_last == 0.0
    sim.substep(sim.dt)
    assert sim.hold_steps == 1  # only the explicit call above


def test_each_substep_of_a_pressed_block_applies_the_press_once():
    sim = _sim(hold="top_pressure")
    sim.substep(sim.dt)
    assert sim.hold_steps == 1
    assert sim.hold_count_last == TOP_LAYER
    assert sim.hold_force_last == pytest.approx(sim.hold_pressure * BLOCK[0] * BLOCK[1], rel=0.02)


def test_the_step_counter_sums_every_pressed_step():
    sim = _sim(hold="top_pressure")
    per_step = sim.hold_pressure * BLOCK[0] * BLOCK[1]
    for _ in range(3):
        sim._apply_hold(1.0e-4)
    assert sim.hold_steps == 3
    assert sim.hold_force_sum == pytest.approx(3 * per_step, rel=0.02)


def test_the_pressure_is_kappa_times_the_fast_load_modulus_unless_given():
    assert HOLD_KAPPA == 0.0059
    # rho g H / E for firm tofu, 4 x 2.5 x 1.5 cm under its own weight: the value the constant is taken from.
    assert HOLD_KAPPA == pytest.approx(1060.0 * 9.81 * 0.015 / 26400.0, rel=0.01)
    # Firm tofu is viscoelastic: the pressure uses the fast-load modulus (26.4 kPa), not the relaxed one (6.6 kPa).
    assert _sim(hold="top_pressure").hold_pressure == pytest.approx(0.0059 * 26.4e3, rel=1e-9)
    assert _sim(hold="top_pressure", hold_pressure=1234.0).hold_pressure == 1234.0


@pytest.mark.parametrize("name", sorted(HOLD_RESERVED))
def test_a_reserved_hold_name_refuses_to_build(name):
    with pytest.raises(NotImplementedError, match="reserved: waiting for the robot setup decision"):
        _sim(hold=name)


def test_an_unknown_hold_name_is_a_value_error_that_lists_the_modes():
    with pytest.raises(ValueError, match="top_pressure"):
        _sim(hold="topp")


def _shape_top_skin_count(inside, block, spacing):
    """(pressed, skin only) of a shape, counted one lattice point at a time: skin if a neighbour one spacing along an
    axis is outside the solid, and pressed if the skin point's neighbour one spacing up is outside as well."""
    counts = [max(1, int(round(s / sp))) for s, sp in zip(block, spacing)]
    axes = [(np.arange(c) + 0.5) * (s / c) for c, s in zip(counts, block)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    def outside(q):
        return bool(q[0] < 0.0 or q[1] < 0.0 or q[2] < 0.0 or q[0] > block[0] or q[1] > block[1] or q[2] > block[2]
                    or not inside(q[None])[0])

    pressed = skin_only = 0
    for p in grid:
        if not inside(p[None])[0]:
            continue
        skin = False
        for a in range(3):
            for s in (-1.0, 1.0):
                q = p.copy()
                q[a] += s * spacing[a]
                skin = skin or outside(q)
        up = p.copy()
        up[2] += spacing[2]
        skin_only += int(skin)
        pressed += int(skin and outside(up))
    return pressed, skin_only


def test_a_shaped_solid_presses_its_skin_point_whose_neighbour_above_is_outside():
    # A T: a wide base 7.5 mm high with an 8 mm wide column on top. The base's own top is pressed where the column is
    # not above it, and the column's top is pressed too.
    def inside(p):
        base = p[:, 2] < 0.0075
        column = (np.abs(p[:, 0] - 0.016) < 0.008) & (p[:, 2] < 0.016)
        return base | column

    config = SimConfig(dx=DX, side_margin=0.012, top_margin=0.016, hold="top_pressure")
    sim = MPMCuttingSim(FIRM_TOFU, BLOCK, config=config, device="cpu", inside=inside)
    spacing = np.array([BLOCK[0] / 16, BLOCK[1] / 12, BLOCK[2] / 8])
    pressed, skin_only = _shape_top_skin_count(inside, BLOCK, spacing)
    assert sim.hold_top_count == pressed
    assert pressed < skin_only  # the above-outside rule is what leaves the side faces unpressed


def test_the_default_presses_nothing_even_for_a_shape():
    def inside(p):
        return p[:, 2] < 0.0075

    sim = MPMCuttingSim(FIRM_TOFU, BLOCK, config=SimConfig(dx=DX, side_margin=0.012, top_margin=0.016),
                        device="cpu", inside=inside)
    assert sim.hold_top_count == 0


@pytest.mark.parametrize("script", ["scripts/cutting/tofu_cutting_demo.py"])
def test_the_demo_takes_the_hold_options_and_refuses_a_reserved_name(script, tmp_path):
    helped = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True, text=True,
                            env={"PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                                 "HOME": str(Path.home())}, timeout=600)
    assert helped.returncode == 0, helped.stderr[-2000:]
    assert "--hold-pressure" in helped.stdout
    assert "top_pressure" in helped.stdout  # argparse may wrap the usage line, so the option is not matched whole
    refused = subprocess.run([sys.executable, str(ROOT / script), "--hold", "plate", "--out", str(tmp_path / "out")],
                             capture_output=True, text=True,
                             env={"PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                                  "HOME": str(Path.home())}, timeout=600)
    assert refused.returncode != 0
    assert "reserved: waiting for the robot setup decision" in refused.stderr


def _sweep_module():
    spec = importlib.util.spec_from_file_location("solver_sweep", ROOT / "scripts" / "cutting" / "solver_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sweep_records_the_hold_options_only_when_they_are_used():
    sweep = _sweep_module()
    plain = dict(E=1.0e6, out="run.json", hold="none", hold_pressure=None, hold_band=None)
    assert sorted(sweep.replay_args(Namespace(**plain))) == ["E", "out"]  # a press-free run writes the old JSON
    pressed = dict(plain, hold="top_pressure", hold_pressure=1234.0, hold_band=0.0)
    recorded = sweep.replay_args(Namespace(**pressed))
    assert recorded["hold"] == "top_pressure" and recorded["hold_pressure"] == 1234.0 and recorded["hold_band"] == 0.0


def test_the_sweep_takes_the_hold_options_on_its_command_line():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "cutting" / "solver_sweep.py"), "--help"],
                          capture_output=True, text=True,
                          env={"PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                               "HOME": str(Path.home())}, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    for flag in ("--hold-pressure", "--hold-band", "top_pressure"):
        assert flag in proc.stdout


def _sweep_run(tmp_path, name, *extra):
    out = tmp_path / f"{name}.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "cutting" / "solver_sweep.py"), "--block", "0.02", "0.02", "0.008",
         "--dx", "0.004", "--clearance", "0.003", "--start-time", "0.08", "--stop-at-board", *extra, "--out", str(out)],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(out.read_text())


def test_the_sweep_press_changes_the_cut_and_the_band_changes_the_press(tmp_path):
    # A 2 cm block: the 1.5 dx band leaves some of its top pressed, so the press and the band both show in the ratio.
    plain = _sweep_run(tmp_path, "none")
    pressed = _sweep_run(tmp_path, "press", "--hold", "top_pressure")
    narrow = _sweep_run(tmp_path, "band0", "--hold", "top_pressure", "--hold-band", "0.0")
    unloaded = _sweep_run(tmp_path, "zero", "--hold", "top_pressure", "--hold-pressure", "0.0")
    assert abs(pressed["force_ratio"] - plain["force_ratio"]) > 1e-4
    assert abs(narrow["force_ratio"] - pressed["force_ratio"]) > 1e-4
    assert unloaded["force_ratio"] == pytest.approx(plain["force_ratio"], abs=1e-12)  # zero pressure is no press
