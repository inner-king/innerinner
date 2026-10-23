# SPDX-License-Identifier: Apache-2.0
"""Physics checks for the MPM knife-cutting solver (simfoundry/cutting), on the CPU at coarse resolution."""

import math
from dataclasses import replace

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from simfoundry.cutting.knife import KnifePose, KnifeTrajectory  # noqa: E402
from simfoundry.cutting.materials import FIRM_TOFU, PRESETS, SILKEN_TOFU  # noqa: E402
from simfoundry.cutting.mpm import MPMCuttingSim, SimConfig  # noqa: E402

DX = 4e-3
BLOCK = (0.032, 0.024, 0.016)
CONFIG = SimConfig(dx=DX, side_margin=0.012, top_margin=0.016)
ACROSS = -math.pi / 2.0  # blade along y, cutting plane x = const


def _sim(material=FIRM_TOFU):
    return MPMCuttingSim(material, BLOCK, config=CONFIG, device="cpu")


def _stroke(sim, depth=None, x_offset=1.0e-3, speed=0.08):
    """One vertical stroke through the block centre (or only ``depth`` into it), then a lift."""
    c = 0.5 * (sim.block_min + sim.block_max)
    top = sim.block_max[2]
    bottom = sim.board_z if depth is None else top - depth - 0.3e-3
    return KnifeTrajectory.strokes(
        [(c[0] + x_offset, c[1], ACROSS)], top_z=top, board_z=bottom,
        start_time=0.05, clearance=0.006, descend_speed=speed, settle=0.1,
    )


def _run(sim, traj, frame=0.01):
    """Advances through the whole trajectory and returns the knife force per frame."""
    forces = []
    t = 0.0
    while t < traj.duration - 1e-9:
        t = min(t + frame, traj.duration)
        f, _ = sim.advance(t, traj)
        forces.append((t, f))
    return forces


def test_materials_are_consistent():
    for m in PRESETS.values():
        assert m.shear_modulus > 0.0 and m.lame_lambda > 0.0
        assert 0.0 < m.poisson_ratio < 0.5
        assert math.isclose(
            m.youngs_modulus, 2.0 * m.shear_modulus * (1.0 + m.poisson_ratio), rel_tol=1e-12
        )
    assert SILKEN_TOFU.youngs_modulus < FIRM_TOFU.youngs_modulus
    assert SILKEN_TOFU.wave_speed < FIRM_TOFU.wave_speed


def test_stroke_keeps_the_cut_plane_while_sawing():
    traj = KnifeTrajectory.strokes(
        [(0.05, 0.04, ACROSS)], top_z=0.03, board_z=0.01, saw_amplitude=0.01, saw_frequency=2.0
    )
    (t_cut,) = traj.cut_times
    pose0, lin0, _ = traj.state(t_cut)
    assert pose0.origin[2] == pytest.approx(0.03 + 0.012)
    _, e_v, _ = KnifePose(origin=pose0.origin, yaw=ACROSS).axes()
    for t in np.linspace(t_cut, t_cut + 0.3, 7):
        pose, lin, _ = traj.state(t)
        # Sawing slides the blade along its edge: the plane (normal e_v) stays where it was.
        assert np.dot(pose.origin - pose0.origin, e_v) == pytest.approx(0.0, abs=1e-12)
        assert lin[2] < 0.0


def test_block_rests_on_the_board():
    sim = _sim()
    x0 = sim.positions()
    sim.advance(0.15)
    x = sim.positions()
    assert not np.isnan(x).any()
    # Gravity sags 1.6 cm of firm tofu by a fraction of a millimetre, no more.
    assert np.abs(x - x0).max() < 1.0e-3
    assert x[:, 2].min() > sim.board_z - 0.25 * DX


@pytest.fixture(scope="module")
def cut_once():
    """A block after one full stroke through it, and the knife force per frame."""
    sim = _sim()
    traj = _stroke(sim)
    forces = _run(sim, traj)
    return sim, forces


def test_one_stroke_cuts_the_block_in_two_that_stay_apart(cut_once):
    sim, forces = cut_once
    assert sim.n_pieces == 2
    assert sim.cut_complete(0)
    assert sim.connected_pieces() == 2
    x, pc = sim.positions(), sim.pieces()
    assert not np.isnan(x).any()
    frac = (pc == 0).mean()
    assert 0.35 < frac < 0.65
    # No interpenetration across the cut once the knife is out and the halves settled.
    cut_x = 0.5 * (sim.block_min[0] + sim.block_max[0]) + 1.0e-3
    left = x[pc == int(np.argmin([x[pc == p, 0].mean() for p in (0, 1)])), 0]
    right = x[pc == int(np.argmax([x[pc == p, 0].mean() for p in (0, 1)])), 0]
    assert left.max() < cut_x + 0.5 * DX
    assert right.min() > cut_x - 0.5 * DX
    # The tofu resists the knife: an upward force while the edge is in it.
    assert max(f[2] for _, f in forces) > 0.05
    # And nothing fell through the board.
    assert x[:, 2].min() > sim.board_z - 0.5 * DX


def test_cut_pieces_export_as_closed_meshes(cut_once, tmp_path):
    pytest.importorskip("skimage")
    pytest.importorskip("trimesh")
    from simfoundry.cutting.export import export_pieces

    sim, _ = cut_once
    spacing = DX / CONFIG.particles_per_axis
    summary = export_pieces(sim.positions(), sim.pieces(), spacing, tmp_path, density=FIRM_TOFU.density)
    assert len(summary) == 2
    for entry in summary:
        assert entry["watertight"]
        assert (tmp_path / entry["file"]).exists()
        # The surface encloses about the volume the particles stand for.
        assert entry["volume_cm3"] == pytest.approx(entry["particle_volume_cm3"], rel=0.25)


def test_a_shallow_press_dents_without_cutting():
    tough = replace(FIRM_TOFU, fracture_energy=200.0)
    sim = _sim(tough)
    traj = _stroke(sim, depth=3.0e-3)
    forces = _run(sim, traj)
    assert sim.crack_front(0).min() > 999.0
    assert not sim.cut_complete(0)
    assert max(f[2] for _, f in forces) > 0.0


def test_a_partial_cut_stops_with_the_knife():
    sim = _sim()
    depth = 8.0e-3
    traj = _stroke(sim, depth=depth)
    _run(sim, traj)
    assert sim.n_pieces == 2  # split into two fields, but still one block below the slit
    assert not sim.cut_complete(0)
    assert sim.connected_pieces() == 1
    o = sim._cut_frames[0][0]
    front = sim.crack_front(0)
    lowest = front.min() + o[2] - sim.board_z  # height of the deepest crack tip above the board
    block_h = BLOCK[2]
    # The crack ran to about the knife's depth, not further: a dilated edge plus one crack lead.
    assert block_h - depth - 2.0 * DX < lowest < block_h - depth + DX


def test_a_tougher_tofu_needs_more_force_to_cut():
    means = []
    for gc in (1.0, 8.0):
        sim = _sim(replace(FIRM_TOFU, fracture_energy=gc))
        traj = _stroke(sim)
        (t_cut,) = traj.cut_times
        forces = _run(sim, traj)
        top = sim.block_max[2]
        t_in = t_cut + (traj.state(t_cut)[0].origin[2] - top) / 0.08
        t_out = t_in + (top - sim.board_z) / 0.08
        cutting = [f[2] for t, f in forces if t_in + 0.02 < t < t_out - 0.02]
        means.append(np.mean(cutting))
        assert sim.cut_complete(0)
    assert means[1] > 1.2 * means[0]
