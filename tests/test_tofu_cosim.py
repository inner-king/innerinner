# SPDX-License-Identifier: Apache-2.0
"""The co-simulation's side of the knife coupling (simfoundry/cutting/cosim.py), on the CPU without Isaac Sim."""

import math

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from simfoundry.cutting.cosim import (  # noqa: E402
    KnifeCoupling, matrix_to_quat_xyzw, pose_from_axes, quat_xyzw_to_matrix,
)
from simfoundry.cutting.knife import KnifePose, KnifeTrajectory  # noqa: E402
from simfoundry.cutting.materials import FIRM_TOFU  # noqa: E402
from simfoundry.cutting.mpm import MPMCuttingSim, SimConfig  # noqa: E402

DX = 4e-3
BLOCK = (0.032, 0.024, 0.016)
CONFIG = SimConfig(dx=DX, side_margin=0.012, top_margin=0.016)
ACROSS = -math.pi / 2.0
H = 1.0 / 120.0  # a physics step


def _world_frame(yaw=math.radians(35.0), shift=(0.4, -0.2, 0.75)):
    """A world in which the solver's box is turned and moved (sim_from_world)."""
    c, s = math.cos(yaw), math.sin(yaw)
    W = np.eye(4)
    W[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    W[:3, 3] = shift
    return np.linalg.inv(W)


def _body_pose(pose_sim, sim_from_world):
    """World pose (position, xyzw) of a knife body whose frame is the solver's knife frame."""
    W = np.linalg.inv(sim_from_world)
    R = W[:3, :3] @ np.stack(pose_sim.axes(), axis=1)
    return W[:3, :3] @ pose_sim.origin + W[:3, 3], matrix_to_quat_xyzw(R)


def _press(sim, t):
    """A level blade over the block's middle, down at 8 cm/s from 6 mm above it after 50 ms."""
    c = 0.5 * (sim.block_min + sim.block_max)
    hover, bottom = sim.block_max[2] + 0.006, sim.board_z + 0.3e-3
    z = hover - 0.08 * min(max(t - 0.05, 0.0), (hover - bottom) / 0.08)
    return KnifePose(np.array([c[0] + 1.0e-3, c[1], z]), ACROSS)


def test_pose_and_quaternion_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        yaw, pitch, roll = rng.uniform(-3.1, 3.1), rng.uniform(-1.4, 1.4), rng.uniform(-3.1, 3.1)
        p = KnifePose(np.zeros(3), yaw, pitch, roll)
        q = pose_from_axes(p.origin, *p.axes())
        for a, b in zip(p.axes(), q.axes()):
            assert np.allclose(a, b, atol=1e-9)
        R = np.stack(p.axes(), axis=1)
        assert np.allclose(quat_xyzw_to_matrix(matrix_to_quat_xyzw(R)), R, atol=1e-9)


def test_coupled_knife_cuts_as_the_solver_does_in_its_own_frame():
    """Driving the solver through a turned, moved world gives the forces of driving it directly."""
    S = _world_frame()
    direct = MPMCuttingSim(FIRM_TOFU, BLOCK, config=CONFIG, device="cpu")
    coupled = MPMCuttingSim(FIRM_TOFU, BLOCK, config=CONFIG, device="cpu")
    link = KnifeCoupling(coupled, S, auto_cut=False, settle=0.0, linger=0.0)
    steps = int(0.45 / H)
    prev = _press(direct, 0.0)
    begin = int(round(0.05 / H))
    worst = 0.0
    peak = 0.0
    for k in range(1, steps + 1):
        pose = _press(direct, k * H)
        traj = KnifeTrajectory.linear(direct.time, direct.time + H, prev, pose, begins_cut=k == begin + 1)
        f_direct, _ = direct.advance(direct.time + H, traj)
        prev = pose
        step = link.step(H, *_body_pose(pose, S), begin_cut=k == begin + 1)
        f_coupled = link.R @ step.force  # back in the solver's frame
        worst = max(worst, float(np.abs(f_coupled - f_direct).max()))
        peak = max(peak, float(f_direct[2]))
    assert peak > 0.02  # the knife did cut, against some resistance
    assert worst < 1e-3 * peak
    assert coupled.cut_complete(0) and direct.cut_complete(0)
    assert coupled.connected_pieces() == direct.connected_pieces() == 2


def test_a_stroke_begins_its_cut_by_itself_and_the_solver_rests_while_the_knife_is_away():
    S = _world_frame()
    sim = MPMCuttingSim(FIRM_TOFU, BLOCK, config=CONFIG, device="cpu")
    link = KnifeCoupling(sim, S, settle=0.05, linger=0.05)
    c = 0.5 * (sim.block_min + sim.block_max)
    away = KnifePose(np.array([c[0] + 1.0e-3, c[1], sim.block_max[2] + 0.10]), ACROSS)
    cuts, advanced = [], []
    for k in range(30):  # 0.25 s away: the solver settles the block, then waits
        advanced.append(link.step(H, *_body_pose(away, S)).advanced)
    assert advanced[0] and not advanced[-1]
    t_idle = sim.time
    for k in range(int(0.5 / H)):
        step = link.step(H, *_body_pose(_press(sim, 0.05 + k * H), S))
        if step.cut is not None:
            cuts.append((k, step.cut))
    assert t_idle < 0.06 + H
    assert len(cuts) == 1  # one stroke, one cut, begun within the last few millimetres
    k0 = cuts[0][0]
    hover = sim.block_max[2] + 0.006
    assert hover - 0.08 * k0 * H < sim.block_max[2] + link.cut_height + 1e-9
    assert sim.cut_complete(0) and sim.connected_pieces() == 2


def test_knife_asset_has_the_solvers_blade():
    """The knife PhysX moves is the blade the solver cuts with, closed, and weighs what a chef's knife does."""
    import trimesh

    from simfoundry.cutting.knife import CHEF_KNIFE
    from simfoundry.cutting.og_cosim import handle_centre, knife_mass_properties, knife_parts

    g = CHEF_KNIFE
    parts = knife_parts(g)
    V, F = parts["blade"]
    blade = trimesh.Trimesh(V, F, process=False)
    assert blade.is_watertight and blade.volume > 0.0
    rings = V[:-1].reshape(-1, 6, 3)  # one cross-section per ring, the tip last on its own
    u = rings[:, 0, 0]
    assert np.allclose(rings[:, 0, 2], g.edge_height(u)) and np.allclose(rings[:, 1, 2], g.edge_height(u))
    assert np.allclose(rings[:, 3, 2], g.spine_height(u)) and np.allclose(rings[:, 4, 2], g.spine_height(u))
    assert np.allclose(V[-1], [g.half_length, 0.0, g.edge_height(g.half_length)])
    assert u.min() == pytest.approx(-g.half_length)
    mass, com, moments, axes = knife_mass_properties(g)
    assert 0.15 < mass < 0.25
    assert -g.half_length - 0.03 < com[0] < -0.03  # balanced between the bolster and the blade's middle
    assert abs(axes[:, int(np.argmin(moments))][0]) > 0.99  # spins most easily about its length
    lo, hi = parts["handle"][0].min(axis=0), parts["handle"][0].max(axis=0)
    assert np.all(handle_centre(g) > lo) and np.all(handle_centre(g) < hi)


def test_top_of_convex_colliders():
    from simfoundry.cutting.og_cosim import hull_top

    cube = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    slab = cube * [1.0, 1.0, 0.1] + [0.5, 0.0, 1.0]
    top = hull_top([cube, slab], [[0.25, 0.5], [0.75, 0.5], [1.25, 0.5], [3.0, 3.0]])
    assert np.allclose(top[:3], [1.0, 1.1, 1.1]) and np.isnan(top[3])
    wedge = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 2], [1, 1, 2]], float)
    assert hull_top([wedge], [[0.5, 0.5]])[0] == pytest.approx(1.5)
