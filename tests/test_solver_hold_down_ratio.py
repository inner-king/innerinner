# SPDX-License-Identifier: Apache-2.0
"""Pressing removes the bounce of a 1.1 MPa block on a coarse grid (research 2's request R2, criterion 5(iv)).

Two full cuts, about 20 s each on the CPU, so this file is kept apart from the quick hold-down checks. The conditions
and the expectation are research 2's: an elastic block of 3.2 x 2.4 x 1.6 cm on a 4 mm grid, the knife at 5 cm/s from
3 mm above the top from 0.08 s on, stopped 0.05 s after it reaches the board. The grid margins are the SimConfig
defaults that research 2 used: with the 12 and 16 mm margins of the other test files the unpressed block does not
bounce and the force ratio stays near 0.73 instead of 0.37.
"""

import math

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from simfoundry.cutting.knife import KnifeTrajectory  # noqa: E402
from simfoundry.cutting.materials import SoftSolidMaterial  # noqa: E402
from simfoundry.cutting.mpm import MPMCuttingSim, SimConfig  # noqa: E402

DX = 4e-3
BLOCK = (0.032, 0.024, 0.016)
ACROSS = -math.pi / 2.0  # blade along y: its normal e_v is the x axis
MATERIAL = SoftSolidMaterial(
    name="research2_elastic", youngs_modulus=1.1e6, poisson_ratio=0.35, density=995.0, viscosity=2.0,
    fracture_energy=125.0, relaxed_modulus=0.0, relaxation_time=0.0, tensile_strength=1.1e5,
)


def _first_time(traj, level, step=1.0e-3):
    """The first time the knife's lowest point is at or below ``level``. The straight blade's lowest point is its origin."""
    ts = np.arange(0.0, traj.duration, step)
    z = np.array([traj.state(t)[0].origin[2] for t in ts])
    hit = np.flatnonzero(z <= level)
    assert len(hit), "the knife never reaches that level"
    lo, hi = ts[max(hit[0] - 1, 0)], ts[hit[0]]
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if traj.state(mid)[0].origin[2] <= level:
            hi = mid
        else:
            lo = mid
    return hi


def _cut(hold):
    """(force ratio, pre-board crack fraction) of one stroke, both as research 2 defines them."""
    sim = MPMCuttingSim(MATERIAL, BLOCK, config=SimConfig(dx=DX, hold=hold), device="cpu")
    centre = 0.5 * (sim.block_min + sim.block_max)
    top = sim.block_max[2]
    board = sim.board_z
    height = top - board  # H, the block's height above the board
    traj = KnifeTrajectory.strokes([(centre[0] + 1.0e-3, centre[1], ACROSS)], top_z=top, board_z=board,
                                   start_time=0.08, clearance=0.003, descend_speed=0.05, dwell=0.05, settle=0.0)
    enter = _first_time(traj, board + sim.config.dilation * DX + 0.5e-3)  # within (dilation dx + 0.5 mm) of the board
    stop = _first_time(traj, board + 0.3e-3) + 0.05
    times = sorted({min(stop, k * 0.005) for k in range(1, int(stop / 0.005) + 2)} | {enter, stop})
    rows = []
    for t in times:
        f, _ = sim.advance(t, traj)
        origin_z = traj.state(t)[0].origin[2]
        rows.append((top - origin_z, float(f[2])))
    window = [f for (d0, _), (d1, f) in zip(rows[:-1], rows[1:]) if 0.002 <= d1 <= height - 0.002 and d1 > d0]
    assert window, "the knife never reached the window of the force ratio"
    ratio = float(np.mean(window)) / (MATERIAL.fracture_energy * BLOCK[1])
    pre = sim.pre_board_ratio(0)  # R1's definition: read at the first step the edge is within dilation dx + 0.5 mm
    assert pre is not None, "the edge never came within dilation dx + 0.5 mm of the board"
    return ratio, pre


def test_pressing_removes_the_bounce_on_a_coarse_grid():
    ratio_none, pre_none = _cut("none")
    ratio_press, pre_press = _cut("top_pressure")
    assert ratio_press - ratio_none >= 0.20, (ratio_none, ratio_press)
    assert pre_press >= 0.95, (pre_none, pre_press)
