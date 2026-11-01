# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subgrid contact (SimConfig.face_contact = "subgrid"): the material face of each side, the touch rule, the face normal,
the required speed and the push, checked on values worked out by hand; then a few steps of the whole solver on a small
block at 4 mm. CPU only."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from simfoundry.cutting.knife import KnifePose  # noqa: E402
from simfoundry.cutting.materials import SoftSolidMaterial  # noqa: E402
from simfoundry.cutting.mpm import (  # noqa: E402
    FACE_OPEN,
    KnifeState,
    MPMCuttingSim,
    SimConfig,
    _face_normal,
    _face_p2g,
    _face_push,
    _face_required_at,
    _face_slope,
    _face_touch,
    _grid_update,
)

wp.init()

ROOT = Path(__file__).resolve().parents[1]
DX = 4e-3
BLOCK = (0.032, 0.024, 0.016)
ACROSS = -math.pi / 2.0  # blade along y: its normal e_v is the x axis
MATERIAL = SoftSolidMaterial(
    name="face_test", youngs_modulus=1.1e6, poisson_ratio=0.35, density=995.0, viscosity=2.0,
    fracture_energy=125.0, relaxed_modulus=0.0, relaxation_time=0.0, tensile_strength=1.1e5,
    board_friction=0.5, knife_friction=0.3,
)
EDGE, SPINE, BEVEL = 0.1e-3, 0.95e-3, 3.2e-3  # the 15-degree knife of the R3 brief


@wp.kernel
def _slope_kernel(knife: KnifeState, kw: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = _face_slope(knife, kw[i])


@wp.kernel
def _normal_kernel(knife: KnifeState, s: wp.array(dtype=float), kw: wp.array(dtype=float),
                   out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = _face_normal(knife, s[i], kw[i])


@wp.kernel
def _touch_kernel(knife: KnifeState, kv: wp.array(dtype=float), kw: wp.array(dtype=float),
                  d: wp.array(dtype=float), reach: float, out: wp.array(dtype=int)):
    i = wp.tid()
    out[i] = _face_touch(knife, kv[i], kw[i], d[i], reach)


@wp.kernel
def _required_kernel(knife: KnifeState, r: wp.array(dtype=wp.vec3), h: wp.array(dtype=float),
                     d: wp.array(dtype=float), dx: float, out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = _face_required_at(knife, r[i], h[i], d[i], dx)


@wp.kernel
def _push_kernel(vel: wp.array(dtype=wp.vec3), kvel: wp.array(dtype=wp.vec3), nrm: wp.array(dtype=wp.vec3),
                 req: wp.array(dtype=float), friction: float, out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = _face_push(vel[i], kvel[i], nrm[i], req[i], friction)


def _knife(edge=EDGE, spine=SPINE, bevel=BEVEL, *, origin=(0.0, 0.0, 0.0), lin=(0.0, 0.0, 0.0),
           ang=(0.0, 0.0, 0.0), friction=0.0, dilation=3.0e-3):
    """A knife on the frame of ACROSS: e_v along x (the normal), e_w up the blade, e_u along the edge."""
    k = KnifeState()
    k.origin = wp.vec3(*origin)
    k.e_u = wp.vec3(0.0, 1.0, 0.0)
    k.e_v = wp.vec3(1.0, 0.0, 0.0)
    k.e_w = wp.vec3(0.0, 0.0, 1.0)
    k.lin_vel = wp.vec3(*lin)
    k.ang_vel = wp.vec3(*ang)
    k.half_length = 0.05
    k.height = 0.046
    k.edge_thickness = edge
    k.spine_thickness = spine
    k.bevel_height = bevel
    k.friction = friction
    k.dilation = dilation
    k.enabled = 1
    k.tip_rise = 0.0
    k.belly_length = 0.0
    k.spine_taper = 0.0
    return k


def _call(kernel, n, inputs, out_dtype, knife=None):
    """Runs a test kernel on n entries; ``inputs`` are the arrays in kernel order (the knife, if any, comes first)."""
    out = wp.zeros(n, dtype=out_dtype, device="cpu")
    args = ([knife] if knife is not None else []) + list(inputs) + [out]
    wp.launch(kernel, dim=n, inputs=args, device="cpu")
    return out.numpy()


def _arr(values, dtype):
    """A device array of float32 values (the solver's precision): ``float`` or ``wp.vec3`` entries."""
    return wp.array(np.asarray(values, dtype=np.float32), dtype=dtype, device="cpu")


def test_the_bevel_slope_is_the_half_thickness_slope_and_zero_above_the_bevel():
    knife = _knife()
    kw = np.array([-1.0e-3, 0.0, 1.0e-3, 3.1e-3, BEVEL, 5.0e-3])
    got = _call(_slope_kernel, len(kw), [_arr(kw, float)], float, knife)
    slope = (SPINE - EDGE) / (2.0 * BEVEL)  # h' = dh/dkw on the bevel, (spine - edge) / (2 bevel)
    expected = np.array([0.0, slope, slope, slope, 0.0, 0.0])  # below the edge, at the bevel's top and above: 0
    assert np.allclose(got, expected, rtol=1e-6, atol=0.0), (got, expected)


def test_the_face_normal_points_into_the_material_and_leans_only_on_the_bevel():
    knife = _knife()
    kw = np.array([1.0e-3, 1.0e-3, 5.0e-3, 5.0e-3])
    s = np.array([1.0, -1.0, 1.0, -1.0])
    got = _call(_normal_kernel, len(s), [_arr(s, float), _arr(kw, float)], wp.vec3, knife)
    slope = (SPINE - EDGE) / (2.0 * BEVEL)
    norm = math.sqrt(1.0 + slope * slope)
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0, rtol=1e-6)
    # On the bevel: (s e_v - h' e_w) / sqrt(1 + h'^2); the e_w part is the wedge.
    assert got[0] == pytest.approx([1.0 / norm, 0.0, -slope / norm], rel=1e-6, abs=1e-9)
    assert got[1] == pytest.approx([-1.0 / norm, 0.0, -slope / norm], rel=1e-6, abs=1e-9)
    # Above the bevel h' = 0: the normal is s e_v exactly, with no wedge part.
    assert got[2] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert got[3] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)


def test_a_face_touches_only_inside_the_real_half_thickness_above_the_edge_and_within_reach():
    knife = _knife()
    h0 = 0.5 * EDGE  # the half-thickness at the edge
    reach = 4.4e-3
    # The boundaries are tested 1e-6 m (or 1e-7 m) off, not on: the solver's knife is float32.
    kv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, h0 + reach - 1.0e-6, h0 + reach + 1.0e-6, 0.0])
    kw = np.array([0.0, 0.0, 0.0, 0.0, -1.0e-4, 0.0, 0.0, 2.0e-3])
    d = np.array([h0 - 1.0e-6, h0 + 1.0e-7, h0 + 1.5e-3, FACE_OPEN, 0.0, 0.0, 0.0, 0.3e-3])
    got = _call(_touch_kernel, len(d), [_arr(kv, float), _arr(kw, float), _arr(d, float), reach], int, knife)
    # inside h: touch; just outside h: no; inside the dilation but outside h: no; no particle: no; below the edge: no;
    # within reach: touch; just outside the reach: no; on the bevel with the face inside h(2 mm): touch.
    assert list(got) == [1, 0, 0, 0, 0, 1, 0, 1]
    # Exactly on the surface is not touching (strict): the solver's float32 values of h and d are the same here.
    on_surface = _call(_touch_kernel, 1, [_arr([0.0], float), _arr([0.0], float), _arr([h0], float), reach], int,
                       knife)
    assert list(on_surface) == [0]


def test_the_required_speed_closes_the_gap_in_one_cell_and_is_zero_for_a_stopped_blade():
    h0 = 0.5 * EDGE
    r0 = np.zeros((1, 3))
    cases = [
        (_knife(lin=(0.0, 0.0, -0.05)), r0, h0, 0.0, DX, (h0 - 0.0) * 0.05 / DX),
        (_knife(lin=(0.0, 0.0, -0.05)), r0, h0, h0, DX, 0.0),  # the face at the blade's surface: continuous at 0
        (_knife(), r0, h0, 0.0, DX, 0.0),  # a stopped blade
        (_knife(ang=(0.0, 0.0, 1.0)), np.array([[0.01, 0.0, 0.0]]), h0, 0.0, DX, h0 * 0.01 / DX),  # v = w x r
        (_knife(lin=(0.0, 0.0, -0.05)), r0, h0, 0.0, 2.0 * DX, h0 * 0.05 / (2.0 * DX)),  # one cell is dx
    ]
    for knife, r, h, d, dx, want in cases:
        got = _call(_required_kernel, 1, [_arr(r, wp.vec3), _arr([h], float), _arr([d], float), dx], float, knife)
        assert got[0] == pytest.approx(want, rel=1e-6, abs=1e-15)


def test_a_push_is_nothing_when_the_normal_speed_is_at_least_the_required():
    vel = np.array([[0.6, 0.0, 0.0], [0.5, 0.0, 0.0]])
    kvel = np.zeros((2, 3))
    nrm = np.tile([1.0, 0.0, 0.0], (2, 1))
    req = np.array([0.5, 0.5])  # vn = 0.6 >= 0.5 and vn = 0.5 >= 0.5
    got = _call(_push_kernel, 2, [_arr(vel, wp.vec3), _arr(kvel, wp.vec3), _arr(nrm, wp.vec3), _arr(req, float), 0.3],
                wp.vec3)
    assert np.array_equal(got, vel.astype(np.float32))  # the solver's float32: unchanged, not rounded


def test_a_push_is_applied_when_the_normal_speed_is_between_zero_and_the_required():
    # vn = 0.2 is not below 0 but is below req = 0.5: the face must still be pushed out to 0.5.
    vel = np.array([[0.2, 0.03, 0.0]])
    out = _call(_push_kernel, 1, [_arr(vel, wp.vec3), _arr(np.zeros((1, 3)), wp.vec3),
                                  _arr([[1.0, 0.0, 0.0]], wp.vec3), _arr([0.5], float), 0.0], wp.vec3)[0]
    assert out == pytest.approx([0.5, 0.03, 0.0], rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("req", [0.0, 0.002])
def test_friction_cannot_reverse_the_tangential_speed(req):
    # Friction that can take away all the tangential speed (mu (req - vn) > |t|) stops it. The friction factor must not go
    # below zero, or the tangential speed would come back reversed.
    vel = np.array([[-0.01, 0.003, 0.0]])  # approaching (vn = -0.01) with a tangential speed of 0.003
    kvel = np.zeros((1, 3))
    nrm = np.array([[1.0, 0.0, 0.0]])
    out = _call(_push_kernel, 1, [_arr(vel, wp.vec3), _arr(kvel, wp.vec3), _arr(nrm, wp.vec3), _arr([req], float), 1.0],
                wp.vec3)[0]
    assert out == pytest.approx([req, 0.0, 0.0], abs=1e-9), "the tangential speed is gone, not reversed"


def test_a_push_sets_the_normal_speed_to_the_required_and_cuts_friction_by_the_normal_impulse():
    vel = np.array([[-0.01, 0.03, 0.0]] * 3)
    kvel = np.zeros((3, 3))
    nrm = np.tile([1.0, 0.0, 0.0], (3, 1))
    req = np.array([0.002, 0.002, 0.0])
    friction = np.array([0.0, 0.3, 0.3])
    rows = []
    for k in range(3):
        rows.append(_call(_push_kernel, 1, [_arr(vel[k:k + 1], wp.vec3), _arr(kvel[k:k + 1], wp.vec3),
                                           _arr(nrm[k:k + 1], wp.vec3), _arr(req[k:k + 1], float), float(friction[k])],
                          wp.vec3)[0])
    # Frictionless: the normal speed becomes req, the tangential speed is kept.
    assert rows[0] == pytest.approx([0.002, 0.03, 0.0], rel=1e-6, abs=1e-12)
    # Friction 0.3 cuts the tangential speed by 0.3 (req - vn) / |t| = 0.3 * 0.012 / 0.03 (normal impulse req - vn).
    assert rows[1] == pytest.approx([0.002, 0.03 * (1.0 - 0.3 * 0.012 / 0.03), 0.0], rel=1e-6, abs=1e-12)
    # With req = 0 the same rule is the node's rule: the cut uses -vn = 0.01.
    assert rows[2] == pytest.approx([0.0, 0.03 * (1.0 - 0.3 * 0.01 / 0.03), 0.0], rel=1e-6, abs=1e-12)


def test_a_stationary_face_under_a_descending_knife_moves_out_at_the_wedge_speed_and_pushes_the_knife_up():
    # The sign test asked for in the brief. The face is exactly on the blade's surface (d = h), so the required speed
    # is 0 and the material must keep up with the blade's normal motion. The knife moves down at v. The material is
    # at rest, so its outward normal speed must be the knife's normal speed, v h' / sqrt(1 + h'^2) (the brief's
    # "h' v" is this value up to the factor sqrt(1 + h'^2), which is 1 to 0.1 % at this knife's h').
    v = 0.05
    kw_bevel, kw_spine = 1.0e-3, 5.0e-3
    slope = (SPINE - EDGE) / (2.0 * BEVEL)
    norm = math.sqrt(1.0 + slope * slope)
    for kw, want_out in ((kw_bevel, v * slope / norm), (kw_spine, 0.0)):
        knife = _knife(lin=(0.0, 0.0, -v))
        h = 0.5 * (EDGE + (SPINE - EDGE) * min(kw / BEVEL, 1.0))
        n = _call(_normal_kernel, 1, [_arr([1.0], float), _arr([kw], float)], wp.vec3, knife)[0]
        req = _call(_required_kernel, 1, [_arr(np.zeros((1, 3)), wp.vec3), _arr([h], float), _arr([h], float), DX],
                    float, knife)[0]
        assert req == 0.0
        kvel = np.array([[0.0, 0.0, -v]])
        out = _call(_push_kernel, 1, [_arr(np.zeros((1, 3)), wp.vec3), _arr(kvel, wp.vec3), _arr([n], wp.vec3),
                                      _arr([req], float), 0.0], wp.vec3)[0]
        outward = float(np.dot(out, n))
        assert outward == pytest.approx(want_out, rel=1e-6, abs=1e-12), (kw, outward, want_out)
        if kw == kw_bevel:
            # The e_v part: the material slides along the face, so its e_v speed is s v h' / (1 + h'^2).
            assert out[0] == pytest.approx(v * slope / (1.0 + slope * slope), rel=1e-6)
            # The material is pushed down (-e_w) by the wedge, so the knife is pushed up: the vertical force is positive.
            assert out[2] < 0.0
        else:
            assert np.allclose(out, 0.0, atol=1e-12)  # above the bevel the face is parallel to the motion: no wedge


def test_the_empty_node_sentinel_is_far_above_any_stored_shortfall():
    # The guard in _grid_update tells an empty node by a stored value above 0.5e30, so the sentinel must lie above it.
    assert FACE_OPEN >= 1.0e30


def _half_thickness_by_hand(w):
    """h(w) of the test blade, written out here and not imported: the edge's half-thickness plus the bevel's share of the
    spine's, with w clamped to the bevel."""
    return 0.5 * (EDGE + (SPINE - EDGE) * min(max(w / BEVEL, 0.0), 1.0))


def _half_thickness_of(knife, w):
    """h(w) of any blade, from its state, by hand (the same formula as mpm._half_thickness)."""
    s = min(max(w / knife.bevel_height, 0.0), 1.0)
    return 0.5 * (knife.edge_thickness + (knife.spine_thickness - knife.edge_thickness) * s)


def _edge_height_by_hand(knife, u):
    """The edge's height above its straight part at u along the blade, by hand (the same formula as mpm._edge_w)."""
    if knife.belly_length <= 0.0:
        return 0.0
    s = min(max((u - (knife.half_length - knife.belly_length)) / knife.belly_length, 0.0), 1.0)
    return knife.tip_rise * s * s


def _face_reference(x, piece, field_side, knife, reach_bound, half_spacing, dx, shape):
    """The face array at every node, by hand: the least of −δ_p over the particles p of side s in the node's support, where
    δ_p = h(kw_p) − (s (x_p − o) . e_v − half_spacing) and kw_p = (x_p − o) . e_w − edge(u_p), u_p = (x_p − o) . e_u, for
    particles within reach_bound of the blade's plane. Positions and dot products are float32 as in the solver, and the
    stencil is floored after the − 0.5 of x * fl32(1/dx), so a particle on a cell boundary lands in the same cells as in
    P2G."""
    ref = np.full((2,) + tuple(shape), FACE_OPEN, dtype=np.float64)
    x32 = np.asarray(x, dtype=np.float32)
    origin32 = np.asarray([float(knife.origin[i]) for i in range(3)], dtype=np.float32)
    e32 = {name: np.asarray([float(getattr(knife, name)[i]) for i in range(3)], dtype=np.float32)
           for name in ("e_u", "e_v", "e_w")}
    inv_dx = np.float32(1.0 / np.float32(dx))  # the kernel's 1.0 / dx, with dx itself a float32
    for p in range(len(x32)):
        s = field_side[piece[p]]
        if s == 0:
            continue
        r = x32[p] - origin32
        d = float(np.dot(r, e32["e_v"]))
        if abs(d) > reach_bound:
            continue
        kw = float(np.dot(r, e32["e_w"])) - _edge_height_by_hand(knife, float(np.dot(r, e32["e_u"])))
        value = s * d - half_spacing - _half_thickness_of(knife, kw)
        b = np.floor(x32[p] * inv_dx - np.float32(0.5)).astype(int)
        idx = 1 if s > 0 else 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    node = (idx, b[0] + i, b[1] + j, b[2] + k)
                    ref[node] = min(ref[node], value)
    return ref


def test_the_face_scatter_keeps_the_least_minus_shortfall_of_each_side_on_the_27_nodes():
    origin = np.array([0.016, 0.012, 0.0])
    e_v = np.array([1.0, 0.0, 0.0])
    knife = _knife(origin=tuple(origin))
    y, z = 0.012, 0.008
    # Off the cell boundaries, so the stencil does not depend on rounding.
    x = np.array([
        [origin[0] + 2.1e-3, y, z],  # side +1, face at 2.1 mm - half spacing
        [origin[0] + 5.3e-3, y, z],  # side +1, behind it: the min keeps 2.1 mm
        [origin[0] - 3.2e-3, y, z],  # side -1, the other side
        [origin[0] + 1.1e-3, y, z],  # side 0: piece not in the cut, skipped
        [origin[0] + 21.0e-3, y, z],  # just beyond the bound (19.5 mm), inside the grid: skipped by the bound
    ])
    piece = np.array([0, 0, 1, 2, 0], dtype=np.int32)
    field_side = np.array([1, -1, 0], dtype=np.int32)  # piece 0: side +1, piece 1: side -1, piece 2: not in the cut
    half_spacing = DX / (2.0 * 2)
    shape = (12, 12, 12)
    reach_bound = 0.5 * max(EDGE, SPINE) + 3.0e-3 + 4.0 * DX
    want = _face_reference(x, piece, field_side, knife, reach_bound, half_spacing, DX, shape)
    face = wp.array(np.full((2,) + shape, FACE_OPEN, dtype=np.float32), dtype=float, device="cpu")
    wp.launch(_face_p2g, dim=len(x), inputs=[wp.array(x.astype(np.float32), dtype=wp.vec3, device="cpu"),
                                             wp.array(piece, dtype=int, device="cpu"),
                                             wp.array(field_side, dtype=int, device="cpu"), knife, reach_bound,
                                             half_spacing, DX, face], device="cpu")
    got = face.numpy().astype(np.float64)
    open_nodes = want >= FACE_OPEN
    assert np.array_equal(got >= FACE_OPEN, open_nodes), "nodes outside every support must keep FACE_OPEN"
    assert np.allclose(got[~open_nodes], want[~open_nodes], rtol=1e-6, atol=1e-9)
    # Every particle sits 8 mm above the edge, above the bevel, so h = spine / 2 for all of them. The stored value is then
    # the face's distance minus the half spacing minus h, and the least of the +1 side is its nearest face (2.1 mm).
    h8 = _half_thickness_by_hand(0.008)
    assert got[1].min() == pytest.approx(2.1e-3 - half_spacing - h8, abs=2e-9)
    assert got[0].min() == pytest.approx(3.2e-3 - half_spacing - h8, abs=2e-9)


def _cut_sim(mode, *, cut=True, particles_per_axis=2):
    config = SimConfig(dx=DX, side_margin=0.012, top_margin=0.016, face_contact=mode,
                       particles_per_axis=particles_per_axis)
    sim = MPMCuttingSim(MATERIAL, BLOCK, config=config, device="cpu")
    centre = 0.5 * (sim.block_min + sim.block_max)
    origin = np.array([centre[0], centre[1], sim.block_max[2] - 1.0e-3])
    sim.set_knife(KnifePose(origin, ACROSS))
    if cut:
        sim.begin_cut()
    return sim


def test_the_intact_block_is_the_same_in_both_face_modes():
    # Without a cut no node is separated, so the face push is never used: both modes must agree bit for bit, with the
    # knife pressing into the intact top (the node push of kinds 0 and 1).
    runs = {}
    for mode in ("node", "subgrid"):
        sim = _cut_sim(mode, cut=False)
        for _ in range(30):
            sim.substep(sim.dt)
        runs[mode] = (sim.positions().copy(), sim.v.numpy().copy())
    assert np.array_equal(runs["node"][0], runs["subgrid"][0])
    assert np.array_equal(runs["node"][1], runs["subgrid"][1])


@pytest.mark.parametrize("particles_per_axis", [2, 3])
def test_the_face_array_is_refilled_each_step_and_holds_the_least_minus_shortfall_of_each_side(particles_per_axis):
    # The half spacing is dx / (2 ppa): 2 particles per axis (the default) and 3 (the solver study's) must both give it.
    sim = _cut_sim("subgrid", particles_per_axis=particles_per_axis)
    assert sim.face_min.shape == (2,) + tuple(sim.grid_shape)
    x0 = sim.x.numpy().astype(np.float64)
    piece = sim.piece.numpy()
    field_side = sim.field_side.numpy()
    knife = sim._knife
    reach_bound = 0.5 * max(knife.edge_thickness, knife.spine_thickness) + knife.dilation + 4.0 * sim.dx
    want = _face_reference(x0, piece, field_side, knife, reach_bound,
                           sim.dx / (2.0 * sim.config.particles_per_axis), sim.dx, sim.grid_shape)
    sim.substep(sim.dt)
    got = sim.face_min.numpy().astype(np.float64)
    open_nodes = want >= FACE_OPEN
    assert np.array_equal(got >= FACE_OPEN, open_nodes)
    assert np.allclose(got[~open_nodes], want[~open_nodes], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("offset", [-0.5e-3, 0.3e-3], ids=["0.5mm_inside", "0.3mm_outside"])
def test_the_solver_reach_bound_writes_a_particle_inside_it_and_not_one_beyond(offset):
    # The solver's bound is 0.5 max(edge, spine) + dilation + 4 dx, with the dilation 0.75 dx of the config: 20.0 mm for this
    # knife at 4 mm. A side +1 particle 0.5 mm inside it (19.5 mm from the blade's plane) must write its face into its 27
    # nodes; one 0.3 mm beyond must not. A bound from the thinner end (19.15 mm) or without the dilation (17.0 mm) drops the
    # inside particle, and no other particle of the block lies in that band.
    sim = _cut_sim("subgrid")
    knife = sim._knife
    origin = np.array([float(knife.origin[i]) for i in range(3)])
    e_v = np.array([float(knife.e_v[i]) for i in range(3)])
    bound = 0.5 * max(float(knife.edge_thickness), float(knife.spine_thickness)) + float(knife.dilation) + 4.0 * sim.dx
    x = sim.x.numpy().astype(np.float64)
    piece = sim.piece.numpy()
    field_side = sim.field_side.numpy()
    p = next(i for i in range(len(x)) if field_side[piece[i]] == 1)
    x[p] += (bound + offset - float((x[p] - origin) @ e_v)) * e_v  # along the blade's normal, same height and station
    sim.x = wp.array(x.astype(np.float32), dtype=wp.vec3, device="cpu")
    want = _face_reference(x, piece, field_side, knife, bound,
                           sim.dx / (2.0 * sim.config.particles_per_axis), sim.dx, sim.grid_shape)
    sim.substep(sim.dt)
    got = sim.face_min.numpy().astype(np.float64)
    open_nodes = want >= FACE_OPEN
    assert np.array_equal(got >= FACE_OPEN, open_nodes)
    assert np.allclose(got[~open_nodes], want[~open_nodes], rtol=1e-5, atol=1e-7)
    if offset < 0.0:
        cell = np.floor(x[p] / sim.dx - 0.5).astype(int)
        stencil = got[1, cell[0]:cell[0] + 3, cell[1]:cell[1] + 3, cell[2]:cell[2] + 3]
        assert (stencil < FACE_OPEN).all(), "the inside particle must reach all 27 of its nodes"


def test_a_short_subgrid_cut_stays_finite_and_pushes_the_cracked_faces():
    # A stroke that starts its own cut (advance begins it), as the solver sweep runs it.
    from simfoundry.cutting.knife import KnifeTrajectory

    config = SimConfig(dx=DX, side_margin=0.012, top_margin=0.016, face_contact="subgrid")
    sim = MPMCuttingSim(MATERIAL, BLOCK, config=config, device="cpu")
    centre = 0.5 * (sim.block_min + sim.block_max)
    traj = KnifeTrajectory.strokes([(centre[0] + 1.0e-3, centre[1], ACROSS)], top_z=sim.block_max[2],
                                   board_z=sim.board_z, start_time=0.08, clearance=0.003, descend_speed=0.05,
                                   dwell=0.05, settle=0.0)
    sim.advance(0.35, traj)
    assert np.isfinite(sim.positions()).all()
    kind_side = sim.kind_side.numpy()
    assert np.isfinite(kind_side).all()
    assert kind_side[2] > 0.0  # kind 2: the faces of a crack were pushed


def test_the_face_mode_changes_a_cut_of_a_small_block():
    # The solver must pass the face mode through to the push: a cut in the subgrid mode is not the node mode's cut.
    from simfoundry.cutting.knife import KnifeTrajectory

    runs = {}
    for mode in ("node", "subgrid"):
        config = SimConfig(dx=DX, side_margin=0.012, top_margin=0.016, face_contact=mode)
        sim = MPMCuttingSim(MATERIAL, BLOCK, config=config, device="cpu")
        centre = 0.5 * (sim.block_min + sim.block_max)
        traj = KnifeTrajectory.strokes([(centre[0] + 1.0e-3, centre[1], ACROSS)], top_z=sim.block_max[2],
                                       board_z=sim.board_z, start_time=0.08, clearance=0.003, descend_speed=0.05,
                                       dwell=0.05, settle=0.0)
        forces = [np.asarray(sim.advance(t, traj)[0], float) for t in np.arange(0.01, 0.25, 0.01)]
        runs[mode] = np.array(forces)
    assert np.abs(runs["subgrid"] - runs["node"]).max() > 1e-3


def test_the_sweep_records_the_face_options_only_when_they_are_used():
    import importlib.util
    from argparse import Namespace

    spec = importlib.util.spec_from_file_location("solver_sweep", ROOT / "scripts" / "cutting" / "solver_sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    plain = dict(E=1.0e6, out="run.json", hold="none", hold_pressure=None, hold_band=0.0, face_contact="node",
                 edge_t=None, spine_t=None, bevel_h=None, side_margin=None, top_margin=None, kind_series=False)
    assert sorted(sweep.replay_args(Namespace(**plain))) == ["E", "out"]  # a node run writes the old JSON
    faced = sweep.replay_args(Namespace(**dict(plain, face_contact="subgrid", edge_t=1.0e-4, kind_series=True,
                                               side_margin=0.005)))
    assert faced["face_contact"] == "subgrid" and faced["edge_t"] == 1.0e-4 and faced["kind_series"] is True
    assert faced["side_margin"] == 0.005 and "spine_t" not in faced and "top_margin" not in faced


def test_the_sweep_takes_the_face_options_and_its_rows_change_with_them(tmp_path):
    # A 2 cm block at 4 mm: the face mode, the knife's edge thickness and the side margin each change the cut, and the
    # per-kind series appears only when asked for.
    def run(name, *extra):
        out = tmp_path / f"{name}.json"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "cutting" / "solver_sweep.py"), "--block", "0.02", "0.02", "0.008",
             "--dx", "0.004", "--clearance", "0.003", "--start-time", "0.08", "--stop-at-board", *extra,
             "--out", str(out)],
            capture_output=True, text=True,
            env={"PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
            timeout=900,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        return json.loads(out.read_text())

    node = run("node", "--hold", "top_pressure", "--kind-series")
    subgrid = run("subgrid", "--hold", "top_pressure", "--kind-series", "--face-contact", "subgrid")
    thin = run("thin", "--hold", "top_pressure", "--kind-series", "--face-contact", "subgrid", "--edge-t", "0.0001",
               "--spine-t", "0.00095", "--bevel-h", "0.0032")
    margin = run("margin", "--hold", "top_pressure", "--kind-series", "--face-contact", "subgrid",
                 "--side-margin", "0.005")
    assert subgrid["args"]["face_contact"] == "subgrid" and "face_contact" not in node["args"]
    assert thin["args"]["edge_t"] == pytest.approx(1.0e-4) and "edge_t" not in subgrid["args"]
    assert margin["args"]["side_margin"] == pytest.approx(0.005)
    assert "kind_side" in node["rows"][-1] and "kind_impulse" in subgrid["rows"][-1]
    assert abs(subgrid["force_ratio"] - node["force_ratio"]) > 1e-6  # the face mode changes the cut
    assert abs(thin["force_ratio"] - subgrid["force_ratio"]) > 1e-6  # so do the knife's thicknesses
    assert abs(margin["force_ratio"] - subgrid["force_ratio"]) > 1e-6  # and the side walls
    plain = run("plain", "--hold", "top_pressure")
    assert "kind_side" not in plain["rows"][-1]  # the series is only added when asked for
    # Each knife dimension changes the cut on its own: a variant of the thin knife that differs in one value.
    variants = {
        "edge": ("--edge-t", "0.0003", "--spine-t", "0.00095", "--bevel-h", "0.0032"),
        "spine": ("--edge-t", "0.0001", "--spine-t", "0.002", "--bevel-h", "0.0032"),
        "bevel": ("--edge-t", "0.0001", "--spine-t", "0.00095", "--bevel-h", "0.02"),
    }
    for name, knife in variants.items():
        other = run(f"knife_{name}", "--hold", "top_pressure", "--kind-series", "--face-contact", "subgrid", *knife)
        assert abs(other["force_ratio"] - thin["force_ratio"]) > 1e-6, name


def _sweep_json(tmp_path, name, *extra):
    out = tmp_path / f"{name}.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "cutting" / "solver_sweep.py"), "--block", "0.02", "0.02", "0.008",
         "--dx", "0.004", "--clearance", "0.003", "--start-time", "0.08", "--stop-at-board", "--hold", "top_pressure",
         "--face-contact", "subgrid", *extra, "--out", str(out)],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(out.read_text())


def test_the_sweep_sends_the_top_margin_to_the_top_and_the_side_margin_to_the_sides(tmp_path):
    # The same 2 cm block as above. A run with --top-margin alone must differ from one with --side-margin of the same value
    # (a sweep that wrote the top value into the side margin would make the two equal), and the top margin reaches the cut.
    base = _sweep_json(tmp_path, "base")
    top = _sweep_json(tmp_path, "top", "--top-margin", "0.012")
    side = _sweep_json(tmp_path, "side", "--side-margin", "0.012")
    assert top["args"]["top_margin"] == pytest.approx(0.012) and "side_margin" not in top["args"]
    assert abs(top["force_ratio"] - side["force_ratio"]) > 1e-6
    assert abs(top["force_ratio"] - base["force_ratio"]) > 1e-6


def test_an_unknown_face_contact_is_refused():
    with pytest.raises(ValueError, match="face_contact"):
        MPMCuttingSim(MATERIAL, BLOCK, config=SimConfig(dx=DX, face_contact="plate"), device="cpu")


def test_the_node_mode_keeps_a_placeholder_face_array():
    sim = _cut_sim("node")
    assert sim.face_min.shape == (2, 1, 1, 1)


def _expected_push(v, kvel, nrm, req, friction):
    """The node velocity after the push, as float64 numbers: the relative normal speed goes to req where it is below req,
    and the tangential part is cut by Coulomb friction with the normal impulse (req - vn)."""
    rel = v - kvel
    vn = float(rel @ nrm)
    if vn < req:
        relt = rel - vn * nrm
        rtl = float(np.linalg.norm(relt))
        if rtl > 1.0e-12:
            relt = relt * max(0.0, 1.0 - friction * (req - vn) / rtl)
        return kvel + relt + req * nrm
    return v


def _expected_face_node(kv, kw, d, v, lin, ang, friction):
    """The subgrid branch of _grid_update for a node of side +1, by hand (float64): (new velocity, required speed)."""
    r = np.array([kv, 0.0, kw])
    kvel = np.array(lin, float) + np.cross(np.array(ang, float), r)
    edge, spine, bevel, dilation = EDGE, SPINE, BEVEL, 3.0e-3
    reach = dilation + DX
    if kw < 0.0:
        return v, 0.0
    h = 0.5 * (edge + (spine - edge) * min(kw / bevel, 1.0))
    slope = (spine - edge) / (2.0 * bevel) if kw < bevel else 0.0
    nrm = np.array([1.0, 0.0, -slope]) / math.sqrt(1.0 + slope * slope)
    if abs(kv) < h + reach and d < h:
        req = (h - d) * float(np.linalg.norm(kvel)) / DX
        return _expected_push(v, kvel, nrm, req, friction), req
    return v, 0.0


def _one_node_full(*, kv, kw, v, d=None, face_on, separated=True, lin=(0.0, 0.0, -0.05), ang=(0.0, 0.0, 0.0),
                   friction=0.0, face=None):
    """Runs the knife branch of _grid_update on one node of one piece (side +1, a cut in progress), with the node at kv,
    kw from the blade's edge. Its material face is d, the distance from the blade's plane (subgrid), or ``face`` is the
    whole face array, as _face_p2g leaves it. Returns a dict with the node velocity after the step and every knife book:
    the knife impulse and moment, and the kind-2 work, impulse and side impulse."""
    n, idx = 12, (5, 5, 5)
    xi = np.array(idx, dtype=float) * DX
    origin = xi - np.array([kv, 0.0, kw])  # so that the node's offset from the edge is (kv, 0, kw)
    knife = _knife(origin=tuple(origin), lin=lin, ang=ang, friction=friction)
    grid_m = np.zeros((1, n, n, n), np.float32)
    grid_m[(0,) + idx] = 1.0
    grid_v = np.zeros((1, n, n, n, 3), np.float32)
    grid_v[(0,) + idx] = v
    if face is None:
        # The kernel adds the node's own half-thickness to the value it reads, so the array holds d − h(kw); no particle
        # (FACE_OPEN) stays as it is.
        face = np.full((2, n, n, n), FACE_OPEN, np.float32)
        if d < 0.5e30:
            face[(1,) + idx] = d - _half_thickness_by_hand(kw)
    front = np.full((1, 1), -1.0e3 if separated else 1.0e3, np.float32)  # below the front: separated; above: not
    origin32 = origin.astype(np.float32).reshape(1, 3)
    knife_impulse = wp.zeros(1, dtype=wp.vec3, device="cpu")
    knife_moment = wp.zeros(1, dtype=wp.vec3, device="cpu")
    kind_work = wp.zeros(3, dtype=float, device="cpu")
    kind_impulse = wp.zeros(3, dtype=wp.vec3, device="cpu")
    kind_side = wp.zeros(3, dtype=float, device="cpu")
    grid_m_w = wp.array(grid_m, dtype=float, device="cpu")
    grid_v_w = wp.array(grid_v, dtype=wp.vec3, device="cpu")
    grid_g_w = wp.zeros((1, n, n, n), dtype=wp.vec3, device="cpu")
    wp.launch(
        _grid_update, dim=(n, n, n),
        inputs=[
            grid_m_w, grid_v_w, grid_g_w, 1, DX, 1.0e-4, wp.vec3(0.0, 0.0, 0.0),
            wp.zeros(1, dtype=int, device="cpu"), wp.zeros(1, dtype=int, device="cpu"),
            wp.zeros(1, dtype=int, device="cpu"), 0,
            wp.array(origin32, dtype=wp.vec3, device="cpu"),
            wp.array(np.array([[0.0, 1.0, 0.0]], np.float32), dtype=wp.vec3, device="cpu"),
            wp.array(np.array([[0.0, 0.0, 1.0]], np.float32), dtype=wp.vec3, device="cpu"),
            wp.array(front, dtype=float, device="cpu"), -0.5 * DX, 1,
            knife, 0, wp.array(np.array([1], np.int32), dtype=int, device="cpu"),
            0.0, -1, 0.0, n, n, n,
            knife_impulse, knife_moment, wp.zeros(1, dtype=float, device="cpu"), kind_work, kind_impulse, kind_side,
            wp.array(face, dtype=float, device="cpu"), face_on,
        ],
        device="cpu",
    )
    return dict(
        v=grid_v_w.numpy()[(0,) + idx].astype(np.float64),
        knife_impulse=knife_impulse.numpy()[0].astype(np.float64),
        knife_moment=knife_moment.numpy()[0].astype(np.float64),
        kind_work=kind_work.numpy().astype(np.float64),
        kind_impulse=kind_impulse.numpy().astype(np.float64),
        kind_side=kind_side.numpy().astype(np.float64),
    )


def _one_node_branch(**kwargs):
    """The velocity and the knife's book entries of _one_node_full, as one tuple."""
    run = _one_node_full(**kwargs)
    return (run["v"], run["knife_impulse"], run["kind_work"], run["kind_impulse"], run["kind_side"])


def _one_node_from_particles(*, kv, kw, v, particles, lin=(0.0, 0.0, -0.05), ang=(0.0, 0.0, 0.0), friction=0.0):
    """The knife branch on the node of _one_node_full, with its face array filled by _face_p2g from particles of side +1
    (piece 0). Each particle is (d, kw_p): its distance from the blade's plane and its height above the edge, so it sits at
    the blade's origin + (d, 0, kw_p)."""
    n, idx = 12, (5, 5, 5)
    origin = np.array(idx, dtype=float) * DX - np.array([kv, 0.0, kw])
    knife = _knife(origin=tuple(origin), lin=lin, ang=ang, friction=friction)
    x = np.array([origin + np.array([d, 0.0, k]) for d, k in particles], dtype=np.float32)
    face_min = wp.array(np.full((2, n, n, n), FACE_OPEN, np.float32), dtype=float, device="cpu")
    wp.launch(_face_p2g, dim=len(x), inputs=[wp.array(x, dtype=wp.vec3, device="cpu"),
                                             wp.array(np.zeros(len(x), np.int32), dtype=int, device="cpu"),
                                             wp.array(np.array([1], np.int32), dtype=int, device="cpu"),
                                             knife, 0.05, DX / 4.0, DX, face_min], device="cpu")
    return _one_node_full(kv=kv, kw=kw, v=v, face_on=1, lin=lin, ang=ang, friction=friction, face=face_min.numpy())


def test_the_knife_branch_pushes_a_face_that_has_come_inside_the_real_half_thickness():
    kv, kw, v = 0.0, 1.0e-3, np.array([0.0, 0.0, 0.0])
    h = 0.5 * (EDGE + (SPINE - EDGE) * (kw / BEVEL))
    d = h - 5.0e-5  # inside the blade's half-thickness
    lin = (0.0, 0.0, -0.05)
    out, impulse, kwork, kimp, kside = _one_node_branch(kv=kv, kw=kw, v=v, d=d, face_on=1, lin=lin)
    want, req = _expected_face_node(kv, kw, d, v, lin, (0.0, 0.0, 0.0), 0.0)
    assert req == pytest.approx((h - d) * 0.05 / DX, rel=1e-9)
    assert np.allclose(out, want, rtol=1e-4, atol=1e-7), (out, want)
    assert not np.allclose(out, v), "the face must be pushed"
    # The knife impulse is the material's momentum change (mass 1), and kind 2 counts it.
    assert np.allclose(impulse, out - v, rtol=1e-4, atol=1e-7)
    assert kside[2] == pytest.approx(abs(float((out - v) @ np.array([1.0, 0.0, 0.0]))), rel=1e-4, abs=1e-9)
    assert kwork[2] == pytest.approx(float((out - v) @ np.array([0.0, 0.0, -0.05])), rel=1e-4, abs=1e-9)


@pytest.mark.parametrize("ang", [(0.0, 0.0, 0.0), (0.0, 10.0, 0.0)], ids=["still", "spinning"])
def test_a_single_face_contact_books_its_impulse_work_side_and_moment_once(ang):
    # The only node with mass is one cracked node, so the step holds exactly one push. Its books must all agree with
    # the push: the impulse dp, the work dp . kvel, the side impulse |dp . e_v| and the moment r x dp (r from the knife's
    # origin to the node), all of them kind 2 (a face of a crack). A spinning knife has a different kvel at each node,
    # so the work must use this node's kvel.
    kv, kw = 0.0, 1.0e-3
    h = 0.5 * (EDGE + (SPINE - EDGE) * (kw / BEVEL))
    d = h - 5.0e-5
    lin = (0.0, 0.0, -0.05)
    run = _one_node_full(kv=kv, kw=kw, v=np.zeros(3), d=d, face_on=1, lin=lin, ang=ang)
    dp = run["v"] - np.zeros(3)  # the mass is 1
    r = np.array([kv, 0.0, kw])
    kvel = np.array(lin) + np.cross(np.array(ang), r)
    assert np.any(dp), "the push must have happened"
    assert np.allclose(run["kind_impulse"][2], dp, rtol=1e-4, atol=1e-12)
    assert not run["kind_impulse"][:2].any(), "only kind 2 is booked"
    assert run["kind_work"][2] == pytest.approx(float(dp @ kvel), rel=1e-4, abs=1e-12)
    assert not run["kind_work"][:2].any()
    assert run["kind_side"][2] == pytest.approx(abs(float(dp @ np.array([1.0, 0.0, 0.0]))), rel=1e-4, abs=1e-12)
    assert not run["kind_side"][:2].any()
    assert np.allclose(run["knife_impulse"], dp, rtol=1e-4, atol=1e-12)
    # The node offset r is formed in float32 (the solver's precision), so the x and z components carry ~3e-11 of
    # rounding against values of ~1e-5: the absolute tolerance sits well above that and well below the physics.
    assert np.allclose(run["knife_moment"], np.cross(r, dp), rtol=1e-3, atol=1e-9)


@pytest.mark.parametrize("case", ["outside_h", "no_particle", "below_edge"])
def test_the_knife_branch_leaves_a_node_alone_when_its_face_does_not_touch(case):
    kv, kw, v = 0.0, 1.0e-3, np.array([-0.01, 0.0, 0.0])
    h = 0.5 * (EDGE + (SPINE - EDGE) * (kw / BEVEL))
    d = {"outside_h": h + 5.0e-5, "no_particle": FACE_OPEN, "below_edge": 0.0}[case]
    if case == "below_edge":
        kw = -1.0e-4
    out, impulse, kwork, kimp, kside = _one_node_branch(kv=kv, kw=kw, v=v, d=d, face_on=1)
    assert np.array_equal(out.astype(np.float32), v.astype(np.float32)), (case, out)
    assert not impulse.any() and not kside.any()


def test_a_node_that_is_not_separated_is_the_same_in_both_modes():
    # Above the front a node is not cracked: the intact rule applies in both modes, bit for bit.
    kv, kw, v = 2.5e-5, 1.0e-3, np.array([-0.01, 0.0, 0.0])
    node_out, *_ = _one_node_branch(kv=kv, kw=kw, v=v, d=FACE_OPEN, face_on=0, separated=False)
    sub_out, *_ = _one_node_branch(kv=kv, kw=kw, v=v, d=FACE_OPEN, face_on=1, separated=False)
    assert np.array_equal(node_out, sub_out)
    assert not np.array_equal(node_out, v.astype(np.float32)), "the intact rule pushes this node"


def test_the_intact_band_reaches_one_dilation_beyond_the_real_half_thickness():
    # Above the front, the intact rule (not the face rule) pushes a node within h + dilation of the blade's plane. The
    # cases sit 0.5 % either side of that edge.
    kw = 1.0e-3
    h = 0.5 * (EDGE + (SPINE - EDGE) * (kw / BEVEL))
    dilation = 3.0e-3  # the knife's dilation, as _knife sets it
    v = np.array([-0.01, 0.0, 0.0])
    inside, *_ = _one_node_branch(kv=h + 0.995 * dilation, kw=kw, v=v, d=FACE_OPEN, face_on=0, separated=False)
    outside, *_ = _one_node_branch(kv=h + 1.005 * dilation, kw=kw, v=v, d=FACE_OPEN, face_on=0, separated=False)
    assert not np.array_equal(inside, v.astype(np.float32)), "inside h + dilation the node is pushed"
    assert np.array_equal(outside, v.astype(np.float32)), "outside h + dilation it is not"


def test_a_cracked_node_uses_its_node_position_in_the_node_mode_and_its_face_in_the_subgrid_mode():
    # The node is inside the blade (s kv < h) but its material face is not (d >= h): the node mode pushes the node,
    # the subgrid mode leaves it, because the face is what meets the blade there.
    kv, kw, v = 2.5e-5, 1.0e-3, np.array([-0.01, 0.0, 0.0])
    h = 0.5 * (EDGE + (SPINE - EDGE) * (kw / BEVEL))
    assert kv < h
    node_out, *_ = _one_node_branch(kv=kv, kw=kw, v=v, d=FACE_OPEN, face_on=0)
    assert not np.array_equal(node_out, v.astype(np.float32))
    sub_out, *_ = _one_node_branch(kv=kv, kw=kw, v=v, d=FACE_OPEN, face_on=1)
    assert np.array_equal(sub_out.astype(np.float32), v.astype(np.float32))


def test_the_required_speed_in_the_branch_uses_the_knife_velocity_at_the_node():
    kv, kw, v = 0.0, 1.0e-3, np.array([0.0, 0.0, 0.0])
    h = 0.5 * (EDGE + (SPINE - EDGE) * (kw / BEVEL))
    d = h - 5.0e-5
    lin, ang = (0.0, 0.0, -0.05), (0.0, 10.0, 0.0)  # the spin adds 0.01 m/s in x at this node
    out, *_ = _one_node_branch(kv=kv, kw=kw, v=v, d=d, face_on=1, lin=lin, ang=ang)
    want, req = _expected_face_node(kv, kw, d, v, lin, ang, 0.0)
    kvel_speed = float(np.linalg.norm(np.array([0.01, 0.0, -0.05])))
    assert req == pytest.approx((h - d) * kvel_speed / DX, rel=1e-9)
    assert np.allclose(out, want, rtol=1e-4, atol=1e-7), (out, want)


def test_a_stopped_blade_pushes_only_where_the_material_approaches_the_blade():
    kv, kw = 0.0, 1.0e-3
    h = 0.5 * (EDGE + (SPINE - EDGE) * (kw / BEVEL))
    for v, approaching in ((np.array([-0.01, 0.0, 0.0]), True), (np.array([0.01, 0.0, 0.0]), False)):
        out, *_ = _one_node_branch(kv=kv, kw=kw, v=v, d=h - 5.0e-5, face_on=1, lin=(0.0, 0.0, 0.0))
        want, req = _expected_face_node(kv, kw, h - 5.0e-5, v, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        assert req == 0.0
        assert np.allclose(out, want, rtol=1e-4, atol=1e-7), (v, out, want)
        if not approaching:
            assert np.array_equal(out.astype(np.float32), v.astype(np.float32))


def test_above_the_bevel_the_face_is_pushed_along_e_v_and_not_leaning():
    kv, kw = 0.0, 5.0e-3
    h = 0.5 * SPINE
    d = h - 1.0e-5
    out, *_ = _one_node_branch(kv=kv, kw=kw, v=np.zeros(3), d=d, face_on=1, lin=(0.0, 0.0, -0.05))
    # h' = 0 here: the normal is e_v, the gap is h - d = 1e-5 m, the required speed is (h - d) * 0.05 / dx, and the
    # push is all along e_v (the material stays at rest along the face, which is parallel to the motion).
    assert out[1] == pytest.approx(0.0, abs=1e-9) and out[2] == pytest.approx(0.0, abs=1e-9)
    assert out[0] == pytest.approx((h - d) * 0.05 / DX, rel=1e-4)


def test_a_less_spread_face_at_a_lower_height_does_not_raise_the_shortfall_of_the_node_above_it():
    # Two faces of side +1 in the support of one node 4 mm above the edge. A thin particle low down (h = 0.116 mm at 0.5 mm)
    # has its face 0.10 mm from the plane, so its own shortfall is 0.016 mm. A particle at the node's height (h = 0.475 mm)
    # has its face 0.30 mm away, so its shortfall is 0.175 mm. Each particle is compared with the half-thickness at its own
    # height, so the node keeps 0.175 mm and its face distance is 0.30 mm. Comparing every particle with the node's height
    # would keep the low face's 0.10 mm and push the node by 0.375 mm.
    kv, kw, v = 0.0, 4.0e-3, np.zeros(3)
    lin = (0.0, 0.0, -0.05)
    run = _one_node_from_particles(kv=kv, kw=kw, v=v, particles=[(1.10e-3, 0.5e-3), (1.30e-3, 4.0e-3)], lin=lin)
    want, req = _expected_face_node(kv, kw, 0.30e-3, v, lin, (0.0, 0.0, 0.0), 0.0)
    assert req > 0.0
    assert np.allclose(run["v"], want, rtol=1e-4, atol=1e-7), (run["v"], want)
    low_face_only, _ = _expected_face_node(kv, kw, 0.10e-3, v, lin, (0.0, 0.0, 0.0), 0.0)
    assert not np.allclose(low_face_only, want, rtol=1e-3, atol=1e-7), "the low face would have changed the push"


def test_the_shortfall_of_each_particle_uses_the_raised_edge_where_it_stands():
    # A blade whose edge rises to a 2 mm tip over its last 20 mm. A particle over the belly (u = 40 mm, the edge 0.5 mm up)
    # is compared with the half-thickness 0.4 mm above that raised edge; one on the straight part at the same depth is
    # compared with the half-thickness 0.9 mm above the edge.
    origin = np.array([0.016, 0.012, 0.012])
    knife = _knife(origin=tuple(origin))
    knife.tip_rise = 2.0e-3
    knife.belly_length = 20.0e-3
    x = np.array([
        [origin[0] + 2.1e-3, origin[1] + 40.0e-3, origin[2] + 0.9e-3],
        [origin[0] + 2.1e-3, origin[1], origin[2] + 0.9e-3],
    ])
    piece = np.array([0, 1], dtype=np.int32)
    field_side = np.array([1, 1], dtype=np.int32)
    half_spacing = DX / (2.0 * 2)
    shape = (2, 16, 16, 16)
    reach_bound = 0.5 * max(EDGE, SPINE) + 3.0e-3 + 4.0 * DX
    want = _face_reference(x, piece, field_side, knife, reach_bound, half_spacing, DX, shape[1:])
    face = wp.array(np.full(shape, FACE_OPEN, dtype=np.float32), dtype=float, device="cpu")
    wp.launch(_face_p2g, dim=len(x), inputs=[wp.array(x.astype(np.float32), dtype=wp.vec3, device="cpu"),
                                             wp.array(piece, dtype=int, device="cpu"),
                                             wp.array(field_side, dtype=int, device="cpu"), knife, reach_bound,
                                             half_spacing, DX, face], device="cpu")
    got = face.numpy().astype(np.float64)
    open_nodes = want >= FACE_OPEN
    assert np.array_equal(got >= FACE_OPEN, open_nodes)
    assert np.allclose(got[~open_nodes], want[~open_nodes], rtol=1e-6, atol=1e-9)
    # The stencils: the belly particle's cells are x 4-6, y 12-14, z 2-4; the straight particle's are x 4-6, y 2-4, z 2-4.
    belly = got[1, 4:7, 12:15, 2:5]
    straight = got[1, 4:7, 2:5, 2:5]
    assert belly.min() == pytest.approx(2.1e-3 - half_spacing - _half_thickness_by_hand(0.4e-3), abs=2e-9)
    assert straight.min() == pytest.approx(2.1e-3 - half_spacing - _half_thickness_by_hand(0.9e-3), abs=2e-9)


def test_a_face_exactly_at_the_real_half_thickness_is_not_touched():
    # Contact needs δ > 0 (d < h): with the face on the blade's surface the node is left alone, so an approaching node keeps
    # its speed.
    kv, kw = 0.0, 1.0e-3
    v = np.array([-0.01, 0.0, 0.0])
    out, *_ = _one_node_branch(kv=kv, kw=kw, v=v, d=_half_thickness_by_hand(kw), face_on=1)
    assert np.array_equal(out.astype(np.float32), v.astype(np.float32)), out


def test_a_node_exactly_on_the_edge_height_is_touched_when_its_face_is_inside():
    kv, kw, v = 0.0, 0.0, np.array([-0.01, 0.0, 0.0])
    d = _half_thickness_by_hand(0.0) - 5.0e-6
    out, *_ = _one_node_branch(kv=kv, kw=kw, v=v, d=d, face_on=1)
    want, req = _expected_face_node(kv, kw, d, v, (0.0, 0.0, -0.05), (0.0, 0.0, 0.0), 0.0)
    assert req > 0.0
    assert np.allclose(out, want, rtol=1e-4, atol=1e-7), (out, want)
    assert not np.array_equal(out.astype(np.float32), v.astype(np.float32))


# The solver before face_contact existed (commit 01a3475), on the stroke below: research 2's elastic material, a block of
# 3.2 x 2.4 x 1.6 cm at 4 mm, the press on, for 0.4 s. A mutant that changes the node path moves these numbers.
GOLDEN_MATERIAL = SoftSolidMaterial(
    name="research2_elastic", youngs_modulus=1.1e6, poisson_ratio=0.35, density=995.0, viscosity=2.0,
    fracture_energy=125.0, relaxed_modulus=0.0, relaxation_time=0.0, tensile_strength=1.1e5,
)
GOLDEN_NODE = dict(samples=40, sum_forces=83.54656832799061, sum_positions=5657.782192999497,
                   fz_1=-0.0002538590251788264, fz_20=0.04061651125084605, fz_last=0.0139351104735399,
                   x_last=-0.014893824118189748, max_abs=13.938635587692248)


def test_the_node_path_reproduces_the_solver_before_the_option():
    from simfoundry.cutting.knife import KnifeTrajectory

    sim = MPMCuttingSim(GOLDEN_MATERIAL, BLOCK, config=SimConfig(dx=DX, hold="top_pressure", hold_band=0.0),
                        device="cpu")
    centre = 0.5 * (sim.block_min + sim.block_max)
    traj = KnifeTrajectory.strokes([(centre[0] + 1.0e-3, centre[1], ACROSS)], top_z=sim.block_max[2],
                                   board_z=sim.board_z, start_time=0.08, clearance=0.003, descend_speed=0.05,
                                   dwell=0.05, settle=0.0)
    forces, positions = [], []
    t = 0.0
    while t < min(0.4, traj.duration) - 1e-9:
        t = min(t + 0.01, 0.4)
        f, _ = sim.advance(t, traj)
        forces.append(np.asarray(f, dtype=np.float64))
        positions.append(sim.positions().astype(np.float64))
    forces, positions = np.array(forces), np.array(positions)
    g = GOLDEN_NODE
    assert len(forces) == g["samples"]
    assert forces.sum() == pytest.approx(g["sum_forces"], rel=1e-9)
    assert positions.sum() == pytest.approx(g["sum_positions"], rel=1e-9)
    assert forces[1, 2] == pytest.approx(g["fz_1"], rel=1e-9, abs=1e-15)
    assert forces[20, 2] == pytest.approx(g["fz_20"], rel=1e-9)
    assert forces[-1, 2] == pytest.approx(g["fz_last"], rel=1e-9)
    assert forces[-1, 0] == pytest.approx(g["x_last"], rel=1e-9)
    assert np.abs(forces).max() == pytest.approx(g["max_abs"], rel=1e-9)
