# SPDX-License-Identifier: Apache-2.0
"""Knife cutting of a soft solid (tofu) with a multi-field MLS-MPM solver in NVIDIA Warp.

How a cut works here, and why:

* The material is a set of MPM particles (moving least squares MPM, Hu et al., SIGGRAPH
  2018) with a fixed-corotated elastic law and a small Kelvin-Voigt viscosity.
* A blade is far thinner than a grid cell. On one shared grid the material on both sides
  of the blade shares nodes, so the two sides stay glued however far the blade goes - the
  problem Hu et al.'s CPIC solves. Here every piece of material owns its own grid field
  instead. When a stroke starts, each piece under the blade's plane is split by side into
  two pieces.
* Below the crack front the two sides are still one material: their nodal velocities are
  merged there, which is exactly single-field MPM. Above the front they are two bodies with
  frictional contact between them (Bardenhagen et al., 2000), so a finished cut never
  heals. CPIC heals once the blade leaves, which BladeMaster (arXiv 2609.27342) fixes with
  permanent per-particle side labels; the piece split here is the same idea.
* The crack front advances by an energy balance, the one Atkins and Williams & Patel
  (Interface Focus 6, 2016) write for cutting: it runs as far as the work of the knife's push
  on the intact material pays for, at G_c per unit of new crack area. Steady cutting then
  takes a push of G_c times the length of edge in the food, whatever the grid. A local
  criterion, the strain energy density under the edge reaching G_c / dx (a crack band),
  cannot do this: under a blade dilated to the grid that density levels off at a value that
  does not depend on dx, so on a fine grid the knife only ploughs.
* Before a stroke's first crack the push must pay for an initiation length of crack
  (``SoftSolidMaterial.initiation_length``), so a straight press dents the food before it
  gives way. The last layer pinched between the edge and the board parts as well.
* Slicing helps: where the edge also slides along itself, the energy per area is divided by
  1 + xi^2, xi being the ratio of the edge's sliding to its pushing speed (Atkins, Xu and
  Jeronimidis 2004: the work of the sliding force goes into the cut too). A blade pressed
  straight down cuts with the full G_c; a cook's press-and-slide stroke, or a tilted blade
  coming down, with less push and less denting. The sliding speed follows its envelope: it
  rises at once and decays over a fraction of a second. Taken instant by instant, every
  reversal of a sawing stroke would stop the slicing and reload the edge from rest, and
  sawing would cut worse than pressing. xi counts up to ``SimConfig.slice_push_max``; past
  it the push hardly changes (the Chef'd stroke takes the same force with a cap of 10).
* A partly cut piece can tear on its own, knife or no knife. The front of every unfinished cut
  drops a cell at a time wherever its uncut ligament is pulled open harder than the tofu holds:
  the opening stress at the crack tip, by beam theory from the tension and bending the bond
  carries across the cut, reaches the tensile strength, or the faces next to the tip have
  opened further than a cohesive crack with that strength and G_c can bridge (2 G_c /
  sigma_t). Tofu fails at about 10 % stretch, which puts E G_c / sigma_t^2 near a centimetre,
  the size of what a cut leaves: there the ligament's strength decides, not the stress
  concentration at the tip. A slit left alone carries nothing open across it and stays.
* The knife pushes on each field from that field's own side of the blade. Where the
  material is still intact the blade is dilated to most of a grid cell, because a 0.3 mm
  edge falls between nodes and would push on nothing.

The force of the material on the knife is the momentum the knife boundary condition
removes, summed over nodes, so a robot simulator can apply it to the tool.
"""

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from simfoundry.cutting.knife import KnifeGeometry, KnifePose
from simfoundry.cutting.materials import SoftSolidMaterial

NO_CRACK = wp.constant(1.0e3)
MASS_EPS = wp.constant(1.0e-14)
# The contact and bonding bookkeeping keeps one bit per piece in an int32.
MAX_PIECES = 30


@wp.struct
class KnifeState:
    origin: wp.vec3
    e_u: wp.vec3
    e_v: wp.vec3
    e_w: wp.vec3
    lin_vel: wp.vec3
    ang_vel: wp.vec3
    half_length: float
    height: float
    edge_thickness: float
    spine_thickness: float
    bevel_height: float
    friction: float
    dilation: float
    enabled: int
    tip_rise: float
    belly_length: float
    spine_taper: float


@wp.func
def _bspline(fx: float, i: int):
    if i == 0:
        return 0.5 * (1.5 - fx) * (1.5 - fx)
    if i == 1:
        return 0.75 - (fx - 1.0) * (fx - 1.0)
    return 0.5 * (fx - 0.5) * (fx - 0.5)


@wp.func
def _bspline_grad(fx: float, i: int):
    if i == 0:
        return fx - 1.5
    if i == 1:
        return -2.0 * (fx - 1.0)
    return fx - 0.5


@wp.func
def _edge_w(k: KnifeState, u: float):
    """Height of the edge above its straight part at u along the blade (the belly)."""
    if k.belly_length <= 0.0:
        return 0.0
    s = wp.clamp((u - (k.half_length - k.belly_length)) / k.belly_length, 0.0, 1.0)
    return k.tip_rise * s * s


@wp.func
def _edge_slope(k: KnifeState, u: float):
    """d(edge height)/du at u."""
    if k.belly_length <= 0.0:
        return 0.0
    s = wp.clamp((u - (k.half_length - k.belly_length)) / k.belly_length, 0.0, 1.0)
    return 2.0 * k.tip_rise * s / k.belly_length


@wp.func
def _spine_w(k: KnifeState, u: float):
    """Height of the spine above the edge's straight part at u."""
    if k.spine_taper <= 0.0:
        return k.height
    s = wp.clamp((u - (k.half_length - k.spine_taper)) / k.spine_taper, 0.0, 1.0)
    return k.tip_rise + (k.height - k.tip_rise) * (1.0 - s * s)


@wp.func
def _half_thickness(k: KnifeState, w: float):
    s = wp.clamp(w / k.bevel_height, 0.0, 1.0)
    return 0.5 * (k.edge_thickness + (k.spine_thickness - k.edge_thickness) * s)


@wp.func
def _stress(F: wp.mat33, C: wp.mat33, mu: float, lam: float, eta: float):
    """Kirchhoff stress and strain energy density of fixed-corotated elasticity plus viscosity."""
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)
    # Keep U and V proper rotations so that R = U V^T is the polar rotation even for an
    # inverted element; flipping a column together with sig[2] leaves U diag(sig) V^T intact.
    if wp.determinant(U) < 0.0:
        U = wp.mat33(U[0, 0], U[0, 1], -U[0, 2], U[1, 0], U[1, 1], -U[1, 2], U[2, 0], U[2, 1], -U[2, 2])
        sig = wp.vec3(sig[0], sig[1], -sig[2])
    if wp.determinant(V) < 0.0:
        V = wp.mat33(V[0, 0], V[0, 1], -V[0, 2], V[1, 0], V[1, 1], -V[1, 2], V[2, 0], V[2, 1], -V[2, 2])
        sig = wp.vec3(sig[0], sig[1], -sig[2])
    R = U @ wp.transpose(V)
    J = wp.determinant(F)
    tau = 2.0 * mu * (F - R) @ wp.transpose(F) + wp.identity(n=3, dtype=float) * (lam * J * (J - 1.0))
    tau = tau + eta * J * (C + wp.transpose(C))
    d0 = sig[0] - 1.0
    d1 = sig[1] - 1.0
    d2 = sig[2] - 1.0
    psi = mu * (d0 * d0 + d1 * d1 + d2 * d2) + 0.5 * lam * (J - 1.0) * (J - 1.0)
    return tau, psi


@wp.func
def _maxwell_stress(Fn: wp.mat33, mu_n: float):
    """Kirchhoff stress and energy density of the deviatoric Hencky spring of the Maxwell branch."""
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(Fn, U, sig, V)
    e0 = wp.log(wp.max(wp.abs(sig[0]), 1.0e-4))
    e1 = wp.log(wp.max(wp.abs(sig[1]), 1.0e-4))
    e2 = wp.log(wp.max(wp.abs(sig[2]), 1.0e-4))
    m = (e0 + e1 + e2) / 3.0
    d = wp.vec3(e0 - m, e1 - m, e2 - m)
    tau = U @ wp.diag(2.0 * mu_n * d) @ wp.transpose(U)
    return tau, mu_n * wp.dot(d, d)


@wp.kernel
def _clear_grid(
    grid_m: wp.array4d(dtype=float), grid_v: wp.array4d(dtype=wp.vec3), grid_g: wp.array4d(dtype=wp.vec3)
):
    f, i, j, k = wp.tid()
    grid_m[f, i, j, k] = 0.0
    grid_v[f, i, j, k] = wp.vec3(0.0)
    grid_g[f, i, j, k] = wp.vec3(0.0)


@wp.kernel
def _p2g(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    F: wp.array(dtype=wp.mat33),
    piece: wp.array(dtype=int),
    Fn: wp.array(dtype=wp.mat33),
    pmass: float,
    pvol: float,
    mu: float,
    lam: float,
    eta: float,
    mu_n: float,
    dx: float,
    dt: float,
    grid_m: wp.array4d(dtype=float),
    grid_v: wp.array4d(dtype=wp.vec3),
    grid_g: wp.array4d(dtype=wp.vec3),
    psi_out: wp.array(dtype=float),
):
    p = wp.tid()
    inv_dx = 1.0 / dx
    xp = x[p]
    f = piece[p]
    tau, psi = _stress(F[p], C[p], mu, lam, eta)
    if mu_n > 0.0:
        tau_n, psi_n = _maxwell_stress(Fn[p], mu_n)
        tau = tau + tau_n
        psi = psi + psi_n
    psi_out[p] = psi
    affine = (-dt * pvol * 4.0 * inv_dx * inv_dx) * tau + pmass * C[p]
    mv = pmass * v[p]
    gx = xp * inv_dx
    bx = int(wp.floor(gx[0] - 0.5))
    by = int(wp.floor(gx[1] - 0.5))
    bz = int(wp.floor(gx[2] - 0.5))
    fx = gx[0] - float(bx)
    fy = gx[1] - float(by)
    fz = gx[2] - float(bz)
    for i in range(3):
        wx = _bspline(fx, i)
        gwx = _bspline_grad(fx, i)
        for j in range(3):
            wy = _bspline(fy, j)
            gwy = _bspline_grad(fy, j)
            for kk in range(3):
                wz = _bspline(fz, kk)
                gwz = _bspline_grad(fz, kk)
                weight = wx * wy * wz
                dpos = wp.vec3(float(i) - fx, float(j) - fy, float(kk) - fz) * dx
                wp.atomic_add(grid_m, f, bx + i, by + j, bz + kk, weight * pmass)
                wp.atomic_add(grid_v, f, bx + i, by + j, bz + kk, weight * (mv + affine @ dpos))
                # Sum of m_p grad N_i(x_p): points out of the body, used as the contact normal.
                grad = wp.vec3(gwx * wy * wz, wx * gwy * wz, wx * wy * gwz) * (pmass * inv_dx)
                wp.atomic_add(grid_g, f, bx + i, by + j, bz + kk, grad)


@wp.func
def _edge_at_bin(knife: KnifeState, o: wp.vec3, u: wp.vec3, ub: float):
    """Where the edge crosses ``u = ub`` in a cut's frame (o: knife origin from the cut's origin).

    Returns the distance along the blade from the origin, huge for an edge across the plane,
    and the edge point relative to the knife origin (on the belly, raised above the straight
    edge's line).
    """
    denom = wp.dot(knife.e_u, u)
    if wp.abs(denom) < 1.0e-6:
        return 1.0e9, wp.vec3(0.0)
    goal = ub - wp.dot(o, u)
    s = goal / denom
    # On the belly the edge is raised along e_w, which a pitched blade tilts along the cut: up to
    # 1 cm at the tip of a 30-degree blade. The straight edge's crossing would put the bins under
    # the tip beyond the blade, never cracked while the tip presses on them. Newton steps find
    # where the curved edge itself crosses the bin.
    lift = wp.dot(knife.e_w, u)
    for _ in range(4):
        slope = denom + _edge_slope(knife, s) * lift
        if wp.abs(slope) < 1.0e-6:
            break
        s = s - (s * denom + _edge_w(knife, s) * lift - goal) / slope
    return s, s * knife.e_u + _edge_w(knife, s) * knife.e_w


@wp.func
def _edge_normal(knife: KnifeState, s: float):
    """Unit normal of the edge within the blade's plane at s, pointing from the edge into the blade."""
    return wp.normalize(knife.e_w - _edge_slope(knife, s) * knife.e_u)


@wp.func
def _crack_reach(knife: KnifeState, s: float, u: wp.vec3, w: wp.vec3, dx: float, lead: float):
    """How far below the edge, in a bin one cell wide at s, the crack may run.

    Past the dilated edge, a rounded nose that is taller in the cut's plane where the edge
    slopes (a pitched blade, the belly), by ``lead``: a crack held behind the nose would leave
    the nose pressing on intact material however fast the crack runs, and that push would
    set the force instead of G_c.
    """
    t = knife.e_u + _edge_slope(knife, s) * knife.e_w
    slope = wp.abs(wp.dot(t, w)) / wp.max(wp.abs(wp.dot(t, u)), 1.0e-6)
    r0 = 0.5 * knife.edge_thickness + knife.dilation
    return r0 * wp.sqrt(1.0 + slope * slope) + 0.5 * dx * slope + lead


@wp.kernel
def _slice_factors(
    knife: KnifeState,
    c: int,
    cut_o: wp.array(dtype=wp.vec3),
    cut_u: wp.array(dtype=wp.vec3),
    umin: float,
    dx: float,
    slice_push: int,
    slide: wp.array(dtype=float),
    slide_alpha: float,
    xi_max: float,
    factor: wp.array(dtype=float),
):
    """Energy per area of each bin's crack, as a fraction of G_c.

    1 / (1 + xi^2) where the edge pushes into the material; 0 where it lifts or holds still,
    which cuts nothing, however much work was done before.
    """
    b = wp.tid()
    o = knife.origin - cut_o[c]
    s, arm = _edge_at_bin(knife, o, cut_u[c], umin + (float(b) + 0.5) * dx)
    factor[b] = 0.0
    if wp.abs(s) > knife.half_length:
        return
    tangent = wp.normalize(knife.e_u + _edge_slope(knife, s) * knife.e_w)
    vel = knife.lin_vel + wp.cross(knife.ang_vel, arm)
    # The sliding speed's envelope: it rises at once and decays over SimConfig.slice_averaging.
    along = wp.abs(wp.dot(vel, tangent))
    if along >= slide[b]:
        slide[b] = along
    else:
        slide[b] = slide[b] + slide_alpha * (along - slide[b])
    push = -wp.dot(vel, _edge_normal(knife, s))
    if push <= 0.0:
        return
    f = 1.0
    if slice_push == 1:
        xi = wp.min(slide[b] / push, xi_max)
        f = 1.0 / (1.0 + xi * xi)
    factor[b] = f


@wp.kernel
def _crack_need(
    front: wp.array2d(dtype=float),
    top: wp.array(dtype=float),
    width: wp.array(dtype=float),
    knife: KnifeState,
    c: int,
    cut_o: wp.array(dtype=wp.vec3),
    cut_u: wp.array(dtype=wp.vec3),
    cut_w: wp.array(dtype=wp.vec3),
    umin: float,
    dx: float,
    lead: float,
    board_z: float,
    factor: wp.array(dtype=float),
    fracture_energy: float,
    initiation_length: float,
    initiated: wp.array(dtype=int),
    need: wp.array(dtype=float),
):
    """Sums the fracture energy of the crack that would run from the front to its reach.

    Before the stroke's first crack, that of an initiation length of crack instead.
    """
    b = wp.tid()
    o = knife.origin - cut_o[c]
    s, arm = _edge_at_bin(knife, o, cut_u[c], umin + (float(b) + 0.5) * dx)
    if wp.abs(s) > knife.half_length or width[b] <= 0.0 or factor[b] <= 0.0:
        return
    if knife.origin[2] + arm[2] - board_z <= knife.dilation:
        return  # pinched against the board: parts for nothing (see _advance_crack)
    target = wp.dot(o + arm, cut_w[c]) - _crack_reach(knife, s, cut_u[c], cut_w[c], dx, lead)
    front_eff = wp.min(front[c, b], top[b])
    if target < front_eff:
        depth = front_eff - target
        if initiated[c] == 0:
            depth = initiation_length
        wp.atomic_add(need, 0, fracture_energy * width[b] * depth * factor[b])


@wp.kernel
def _advance_crack(
    front: wp.array2d(dtype=float),
    top: wp.array(dtype=float),
    width: wp.array(dtype=float),
    knife: KnifeState,
    c: int,
    cut_o: wp.array(dtype=wp.vec3),
    cut_u: wp.array(dtype=wp.vec3),
    cut_w: wp.array(dtype=wp.vec3),
    umin: float,
    dx: float,
    lead: float,
    board_z: float,
    factor: wp.array(dtype=float),
    initiation_length: float,
    work: wp.array(dtype=float),
    need: wp.array(dtype=float),
    initiated: wp.array(dtype=int),
    first_crack: wp.array(dtype=float),
):
    """Runs the crack as far as this step's work on the intact material pays for."""
    b = wp.tid()
    o = knife.origin - cut_o[c]
    s, arm = _edge_at_bin(knife, o, cut_u[c], umin + (float(b) + 0.5) * dx)
    if wp.abs(s) > knife.half_length:
        return
    w_edge = wp.dot(o + arm, cut_w[c])
    # The last layer pinched between the edge and the board parts as well.
    gap = knife.origin[2] + arm[2] - board_z
    if gap <= knife.dilation:
        if w_edge < front[c, b]:
            # Well below the board: the nodes just under it still carry mass from the bottom
            # layer, and a front level with them would leave the pieces bonded there.
            front[c, b] = wp.min(front[c, b], w_edge - gap - 3.0 * dx)
        return
    if width[b] <= 0.0 or factor[b] <= 0.0 or need[0] <= 0.0:
        return
    target = w_edge - _crack_reach(knife, s, cut_u[c], cut_w[c], dx, lead)
    front_eff = wp.min(front[c, b], top[b])
    if target >= front_eff:
        return
    if initiated[c] == 0:
        # The material first gives way once the push has paid for an initiation length of
        # crack, which opens at once; until then the knife only presses.
        if work[0] < need[0]:
            return
        # How deep the edge's lowest point in the material had pressed when it gave way.
        wp.atomic_min(first_crack, c, w_edge)
        front[c, b] = wp.max(target, front_eff - initiation_length)
        return
    # Every bin the edge presses into goes the same fraction of its way: the edge is one blade.
    f = wp.min(1.0, work[0] / need[0])
    front[c, b] = front_eff - f * (front_eff - target)


@wp.kernel
def _settle_crack_work(
    work: wp.array(dtype=float), need: wp.array(dtype=float), initiated: wp.array(dtype=int), c: int
):
    if need[0] > 0.0 and (initiated[c] == 1 or work[0] >= need[0]):
        # Spent on the crack. Work beyond what it needed went into pressing and is not kept:
        # banked, it would let the knife cut on later with no push at all.
        initiated[c] = 1
        work[0] = 0.0
    elif need[0] <= 0.0:
        # Nothing to cut (the edge is off the material, or lifting): this pressing pays for nothing.
        work[0] = 0.0
    need[0] = 0.0


@wp.kernel
def _ligament_load(
    grid_m: wp.array4d(dtype=float),
    grid_v: wp.array4d(dtype=wp.vec3),
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
    pair_cut: wp.array(dtype=int),
    pair_side: wp.array(dtype=int),
    n_pairs: int,
    cut_o: wp.array(dtype=wp.vec3),
    cut_u: wp.array(dtype=wp.vec3),
    cut_v: wp.array(dtype=wp.vec3),
    cut_w: wp.array(dtype=wp.vec3),
    front: wp.array2d(dtype=float),
    umin: float,
    nbins: int,
    dx: float,
    dt: float,
    lig_n: wp.array2d(dtype=float),
    lig_nw: wp.array2d(dtype=float),
    lig_top: wp.array2d(dtype=float),
):
    """Sums, per crack-front bin, the force the bond below each front carries across its cut.

    It is the impulse _grid_update's merge is about to give the two fields at each bonded node,
    over dt, along the cut's normal: positive where the bond holds the sides together. Summed
    with its moment, it is the tension and bending that the uncut ligament transmits.
    """
    i, j, k = wp.tid()
    xi = wp.vec3(float(i), float(j), float(k)) * dx
    for q in range(n_pairs):
        a = pair_a[q]
        b = pair_b[q]
        ma = grid_m[a, i, j, k]
        mb = grid_m[b, i, j, k]
        if ma > MASS_EPS and mb > MASS_EPS:
            c = pair_cut[q]
            rc = xi - cut_o[c]
            bb = int(wp.floor((wp.dot(rc, cut_u[c]) - umin) / dx))
            if bb >= 0 and bb < nbins:
                w = wp.dot(rc, cut_w[c])
                if w < front[c, bb]:
                    # Gravity reaches both fields alike and drops out of the difference.
                    rel = grid_v[a, i, j, k] / ma - grid_v[b, i, j, k] / mb
                    f = -float(pair_side[q]) * wp.dot(rel, cut_v[c]) * (ma * mb / (ma + mb)) / dt
                    wp.atomic_add(lig_n, c, bb, f)
                    wp.atomic_add(lig_nw, c, bb, f * w)
                    wp.atomic_max(lig_top, c, bb, w)


@wp.kernel
def _tip_opening(
    x: wp.array(dtype=wp.vec3),
    x0: wp.array(dtype=wp.vec3),
    piece: wp.array(dtype=int),
    field_cut_side: wp.array2d(dtype=int),
    cut_live: wp.array(dtype=int),
    n_cuts: int,
    cut_o: wp.array(dtype=wp.vec3),
    cut_u: wp.array(dtype=wp.vec3),
    cut_v: wp.array(dtype=wp.vec3),
    cut_w: wp.array(dtype=wp.vec3),
    front: wp.array2d(dtype=float),
    umin: float,
    nbins: int,
    dx: float,
    layer: float,
    open_sum: wp.array3d(dtype=float),
    open_cnt: wp.array3d(dtype=float),
):
    """Sums, per bin, how far each face of an unfinished cut moved along its normal just above the front.

    The faces are the particles that started within ``layer`` of the plane, one particle layer
    on each side; the difference of the two sides' mean displacements is the crack's opening
    next to its tip.
    """
    p = wp.tid()
    f = piece[p]
    xp = x[p]
    for c in range(n_cuts):
        s = field_cut_side[f, c]
        if s == 0 or cut_live[c] == 0:
            continue
        if wp.abs(wp.dot(x0[p] - cut_o[c], cut_v[c])) >= layer:
            continue
        rel = xp - cut_o[c]
        b = int(wp.floor((wp.dot(rel, cut_u[c]) - umin) / dx))
        if b < 0 or b >= nbins:
            continue
        fr = front[c, b]
        w = wp.dot(rel, cut_w[c])
        if fr < 0.5 * NO_CRACK and w >= fr and w < fr + dx:
            k = 0
            if s > 0:
                k = 1
            wp.atomic_add(open_sum, c, b, k, wp.dot(xp - x0[p], cut_v[c]))
            wp.atomic_add(open_cnt, c, b, k, 1.0)


@wp.func
def _knife_cutting_bin(
    knife: KnifeState,
    active_cut: int,
    c: int,
    b: int,
    cut_o: wp.array(dtype=wp.vec3),
    cut_u: wp.array(dtype=wp.vec3),
    cut_w: wp.array(dtype=wp.vec3),
    umin: float,
    dx: float,
    fr: float,
):
    """Whether the knife's edge is at bin b's crack tip and moving into the material."""
    if knife.enabled == 0 or c != active_cut:
        return False
    denom = wp.dot(knife.e_u, cut_u[c])
    if wp.abs(denom) < 1.0e-6:
        return False
    o = knife.origin - cut_o[c]
    s = (umin + (float(b) + 0.5) * dx - wp.dot(o, cut_u[c])) / denom
    if wp.abs(s) > knife.half_length:
        return False
    arm = s * knife.e_u + _edge_w(knife, s) * knife.e_w
    if wp.dot(o + arm, cut_w[c]) - fr > knife.dilation + 2.0 * dx:
        return False  # the edge is up in the open slit, clear of the tip
    vel = knife.lin_vel + wp.cross(knife.ang_vel, arm)
    normal = wp.normalize(knife.e_w - _edge_slope(knife, s) * knife.e_u)
    return -wp.dot(vel, normal) > 0.0


@wp.kernel
def _tear(
    front: wp.array2d(dtype=float),
    front_prev: wp.array2d(dtype=float),
    lig_n: wp.array2d(dtype=float),
    lig_nw: wp.array2d(dtype=float),
    lig_top: wp.array2d(dtype=float),
    open_sum: wp.array3d(dtype=float),
    open_cnt: wp.array3d(dtype=float),
    cut_live: wp.array(dtype=int),
    cut_board: wp.array(dtype=float),
    nbins: int,
    dx: float,
    dt: float,
    strength: float,
    critical_opening: float,
    hold: float,
    clock: wp.array2d(dtype=float),
    front_seen: wp.array2d(dtype=float),
    open_ref: wp.array2d(dtype=float),
    knife: KnifeState,
    active_cut: int,
    cut_o: wp.array(dtype=wp.vec3),
    cut_u: wp.array(dtype=wp.vec3),
    cut_w: wp.array(dtype=wp.vec3),
    umin: float,
):
    """Tears the uncut ligament down by one cell where it is pulled open harder than the tofu holds.

    Two ways, as a cohesive crack with a linear softening law fails: the opening stress at the
    crack tip reaches the tensile strength, or the crack faces next to the tip have opened
    further than the opening at which that law carries nothing. The stress is the ligament's
    tension and bending (``_ligament_load``) by beam theory, over a ligament from the board to
    half a cell above its highest bonded node. The opening is ``_tip_opening``'s; it is needed
    because a multi-field crack tip gives too easily: the faces in the cell above it open with
    little strain in the bonded cell below, so a blade levering the faces apart right over the
    tip opens them by millimetres while the ligament's stress stays low. It counts from the
    narrowest the faces have been since the front reached its cell: a blade dilated to the grid
    leaves a kerf behind it (around a millimetre at 4 mm cells) that a real edge does not.

    A bin more than a cell behind every neighbour that has material is a bridge one cell wide,
    which takes its neighbours' load as well, and tears too. Without this, bins left behind by
    the scatter of a coarse grid's per-bin loads stood as columns between torn-through bins and
    held the pieces together.

    Where the knife's edge is pushing into the material at the crack tip, the knife's own
    criterion runs the crack (_advance_crack), and the ligament waits: the push of a blade
    dilated to the grid wedges the tip open far more than a real edge does, and with tougher
    tofu that tore the ligament a cell ahead of the knife and took the cutting force with it.

    A crack runs no faster than about the Rayleigh wave speed, so the front drops only after the
    tip has been overloaded for ``hold``, the time such a crack takes to cross a cell.
    """
    c, b = wp.tid()
    if cut_live[c] == 0:
        return
    top = lig_top[c, b]
    wb = cut_board[c]
    if front[c, b] >= 0.5 * NO_CRACK:
        # No crack here to extend: tearing runs a slit deeper, it does not start one.
        clock[c, b] = 0.0
        return
    if _knife_cutting_bin(knife, active_cut, c, b, cut_o, cut_u, cut_w, umin, dx, front[c, b]):
        clock[c, b] = 0.0
        return
    if top < -0.5 * NO_CRACK:
        clock[c, b] = 0.0
        if open_cnt[c, b, 0] + open_cnt[c, b, 1] > 0.0 and front[c, b] > wb - 1.5 * dx:
            # Material at the crack front but no node below it shared by both sides: the sides
            # have already come apart here. Record it, so that cut_complete agrees.
            front[c, b] = wb - 3.0 * dx
        return
    t = wp.max(top + 0.5 * dx - wb, 0.5 * dx)
    n = lig_n[c, b]
    m = lig_nw[c, b] - n * (wb + 0.5 * t)
    sigma = n / (dx * t) + 6.0 * m / (dx * t * t)
    if front[c, b] != front_seen[c, b]:
        # The front moved here (the knife, or a tear): the faces next to it are new.
        front_seen[c, b] = front[c, b]
        open_ref[c, b] = NO_CRACK
    opening = float(0.0)
    if open_cnt[c, b, 0] > 0.0 and open_cnt[c, b, 1] > 0.0:
        now = open_sum[c, b, 1] / open_cnt[c, b, 1] - open_sum[c, b, 0] / open_cnt[c, b, 0]
        open_ref[c, b] = wp.min(open_ref[c, b], now)
        opening = now - open_ref[c, b]
    # Neighbours with material: still bonded, or torn through (front under the board).
    neighbours = int(0)
    lower = int(0)
    for k in range(2):
        nb = b - 1 + 2 * k
        if nb >= 0 and nb < nbins:
            fn = front_prev[c, nb]
            if lig_top[c, nb] > -0.5 * NO_CRACK or fn < wb - 1.5 * dx:
                neighbours = neighbours + 1
                if fn < front_prev[c, b] - 1.5 * dx:
                    lower = lower + 1
    bridge = neighbours > 0 and lower == neighbours
    if sigma < strength and opening < critical_opening and not bridge:
        clock[c, b] = 0.0
        return
    clock[c, b] = clock[c, b] + dt
    if clock[c, b] < hold:
        return
    clock[c, b] = 0.0
    # Unbond the highest bonded row (bonding is strictly below the front).
    new_front = wp.min(front[c, b] - dx, top)
    if top - dx < wb - 0.25 * dx:
        # What was left is the row under the board, which holds the bottom layer: that parts as
        # well, with the front well below the board as the knife leaves it (_advance_crack).
        new_front = wb - 3.0 * dx
    front[c, b] = new_front


@wp.kernel
def _grid_update(
    grid_m: wp.array4d(dtype=float),
    grid_v: wp.array4d(dtype=wp.vec3),
    grid_g: wp.array4d(dtype=wp.vec3),
    n_fields: int,
    dx: float,
    dt: float,
    gravity: wp.vec3,
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
    pair_cut: wp.array(dtype=int),
    n_pairs: int,
    cut_o: wp.array(dtype=wp.vec3),
    cut_u: wp.array(dtype=wp.vec3),
    cut_w: wp.array(dtype=wp.vec3),
    front: wp.array2d(dtype=float),
    umin: float,
    nbins: int,
    knife: KnifeState,
    active_cut: int,
    field_side: wp.array(dtype=int),
    contact_mu: float,
    board_k: int,
    board_mu: float,
    nx: int,
    ny: int,
    nz: int,
    knife_impulse: wp.array(dtype=wp.vec3),
    knife_moment: wp.array(dtype=wp.vec3),
    crack_work: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    m_total = float(0.0)
    p_total = wp.vec3(0.0)
    present = int(0)
    for f in range(n_fields):
        m = grid_m[f, i, j, k]
        if m > MASS_EPS:
            vel = grid_v[f, i, j, k] / m + dt * gravity
            grid_v[f, i, j, k] = vel
            m_total = m_total + m
            p_total = p_total + m * vel
            present = present + 1
    if present == 0:
        return
    xi = wp.vec3(float(i), float(j), float(k)) * dx

    if present > 1:
        # Pieces split by a cut are still one material below that cut's crack front.
        bonded = int(0)
        for q in range(n_pairs):
            a = pair_a[q]
            b = pair_b[q]
            ma = grid_m[a, i, j, k]
            mb = grid_m[b, i, j, k]
            if ma > MASS_EPS and mb > MASS_EPS:
                c = pair_cut[q]
                rc = xi - cut_o[c]
                bb = int(wp.floor((wp.dot(rc, cut_u[c]) - umin) / dx))
                fr = NO_CRACK
                if bb >= 0 and bb < nbins:
                    fr = front[c, bb]
                if wp.dot(rc, cut_w[c]) < fr:
                    vcm = (ma * grid_v[a, i, j, k] + mb * grid_v[b, i, j, k]) / (ma + mb)
                    grid_v[a, i, j, k] = vcm
                    grid_v[b, i, j, k] = vcm
                    bonded = bonded | (1 << a) | (1 << b)
        # Everything else touching here is in frictional, separable contact.
        vcm_all = p_total / m_total
        for f in range(n_fields):
            m = grid_m[f, i, j, k]
            if m > MASS_EPS and (bonded & (1 << f)) == 0:
                g = grid_g[f, i, j, k]
                gl = wp.length(g)
                if gl > 1.0e-12:
                    nrm = g / gl
                    vel = grid_v[f, i, j, k]
                    dv = vel - vcm_all
                    vn = wp.dot(dv, nrm)
                    if vn > 0.0:
                        vt = dv - vn * nrm
                        vtl = wp.length(vt)
                        vel = vel - vn * nrm
                        if vtl > 1.0e-12:
                            vel = vel - wp.min(1.0, contact_mu * vn / vtl) * vt
                        grid_v[f, i, j, k] = vel

    if knife.enabled == 1:
        r = xi - knife.origin
        ku = wp.dot(r, knife.e_u)
        kv = wp.dot(r, knife.e_v)
        edge = _edge_w(knife, ku)
        kw = wp.dot(r, knife.e_w) - edge  # height above the local edge
        reach = knife.dilation + dx
        near = wp.abs(ku) <= knife.half_length + dx and kw <= _spine_w(knife, ku) - edge + dx and kw >= -reach
        if near and wp.abs(kv) <= knife.spine_thickness + reach:
            kvel = knife.lin_vel + wp.cross(knife.ang_vel, r)
            separated = int(0)
            if active_cut >= 0:
                rc = xi - cut_o[active_cut]
                bb = int(wp.floor((wp.dot(rc, cut_u[active_cut]) - umin) / dx))
                if bb >= 0 and bb < nbins:
                    if wp.dot(rc, cut_w[active_cut]) > front[active_cut, bb]:
                        separated = 1
            for f in range(n_fields):
                m = grid_m[f, i, j, k]
                if m > MASS_EPS:
                    s = int(0)
                    if active_cut >= 0:
                        s = field_side[f]
                    hit = int(0)
                    nrm = wp.vec3(0.0)
                    if separated == 1 and s != 0:
                        # Cracked here: each side meets the blade's face on its own side.
                        if kw >= 0.0:
                            if float(s) * kv < _half_thickness(knife, kw):
                                hit = 1
                                nrm = float(s) * knife.e_v
                    else:
                        # Intact here: the blade is a dilated wedge with a rounded edge, and it
                        # pushes every field at a node the same way - by the node's side, not the
                        # field's. Pushing the two fields of one node apart would split intact
                        # tofu without the crack criterion ever being met.
                        side = 1.0
                        if kv < 0.0:
                            side = -1.0
                        if kw >= 0.0:
                            if wp.abs(kv) < _half_thickness(knife, kw) + knife.dilation:
                                hit = 1
                                nrm = side * knife.e_v
                        else:
                            d = wp.sqrt(kv * kv + kw * kw)
                            if d < 0.5 * knife.edge_thickness + knife.dilation:
                                hit = 1
                                nrm = (kv * knife.e_v + kw * knife.e_w) / wp.max(d, 1.0e-9)
                    if hit == 1:
                        vel = grid_v[f, i, j, k]
                        rel = vel - kvel
                        vn = wp.dot(rel, nrm)
                        if vn < 0.0:
                            relt = rel - vn * nrm
                            rtl = wp.length(relt)
                            if rtl > 1.0e-12:
                                relt = relt * wp.max(0.0, 1.0 - knife.friction * (-vn) / rtl)
                            nv = kvel + relt
                            dp = m * (nv - vel)
                            grid_v[f, i, j, k] = nv
                            wp.atomic_add(knife_impulse, 0, dp)
                            wp.atomic_add(knife_moment, 0, wp.cross(r, dp))
                            if active_cut >= 0 and separated == 0:
                                # Work of the knife's push on material the crack has not reached:
                                # only along the edge's normal, as slicing is counted by xi.
                                en = _edge_normal(knife, ku)
                                wp.atomic_add(crack_work, 0, wp.dot(dp, en) * wp.dot(kvel, en))

    for f in range(n_fields):
        m = grid_m[f, i, j, k]
        if m > MASS_EPS:
            vel = grid_v[f, i, j, k]
            if k <= board_k and vel[2] < 0.0:
                vn = -vel[2]
                vt = wp.vec3(vel[0], vel[1], 0.0)
                vtl = wp.length(vt)
                if vtl > 1.0e-12:
                    vt = vt * wp.max(0.0, 1.0 - board_mu * vn / vtl)
                vel = vt
            # Slip walls on the domain boundary, which the food should never reach.
            vx = vel[0]
            vy = vel[1]
            vz = vel[2]
            if i < 3:
                vx = wp.max(vx, 0.0)
            if i > nx - 4:
                vx = wp.min(vx, 0.0)
            if j < 3:
                vy = wp.max(vy, 0.0)
            if j > ny - 4:
                vy = wp.min(vy, 0.0)
            if k > nz - 4:
                vz = wp.min(vz, 0.0)
            grid_v[f, i, j, k] = wp.vec3(vx, vy, vz)


@wp.kernel
def _g2p(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    F: wp.array(dtype=wp.mat33),
    piece: wp.array(dtype=int),
    grid_v: wp.array4d(dtype=wp.vec3),
    dx: float,
    dt: float,
    lo: float,
    hi: wp.vec3,
    Fn: wp.array(dtype=wp.mat33),
    relax: float,
    viscoelastic: int,
):
    p = wp.tid()
    inv_dx = 1.0 / dx
    f = piece[p]
    xp = x[p]
    gx = xp * inv_dx
    bx = int(wp.floor(gx[0] - 0.5))
    by = int(wp.floor(gx[1] - 0.5))
    bz = int(wp.floor(gx[2] - 0.5))
    fx = gx[0] - float(bx)
    fy = gx[1] - float(by)
    fz = gx[2] - float(bz)
    nv = wp.vec3(0.0)
    nC = wp.mat33()
    for i in range(3):
        wx = _bspline(fx, i)
        for j in range(3):
            wy = _bspline(fy, j)
            for kk in range(3):
                weight = wx * wy * _bspline(fz, kk)
                gv = grid_v[f, bx + i, by + j, bz + kk]
                dpos = wp.vec3(float(i) - fx, float(j) - fy, float(kk) - fz) * dx
                nv = nv + weight * gv
                nC = nC + (4.0 * inv_dx * inv_dx * weight) * wp.outer(gv, dpos)
    v[p] = nv
    C[p] = nC
    xn = xp + dt * nv
    x[p] = wp.vec3(wp.clamp(xn[0], lo, hi[0]), wp.clamp(xn[1], lo, hi[1]), wp.clamp(xn[2], lo, hi[2]))
    step = wp.identity(n=3, dtype=float) + dt * nC
    F[p] = step @ F[p]
    if viscoelastic == 1:
        # Maxwell branch: deform with the material, then let the deviatoric log strain relax by
        # exp(-dt / tau), the exact decay of a linear Maxwell element over the step (as in
        # Fang et al., "Silly Rubber", SIGGRAPH 2019).
        U = wp.mat33()
        sig = wp.vec3()
        V = wp.mat33()
        wp.svd3(step @ Fn[p], U, sig, V)
        e0 = wp.log(wp.max(wp.abs(sig[0]), 1.0e-4))
        e1 = wp.log(wp.max(wp.abs(sig[1]), 1.0e-4))
        e2 = wp.log(wp.max(wp.abs(sig[2]), 1.0e-4))
        m = (e0 + e1 + e2) / 3.0
        s = wp.vec3(wp.exp((e0 - m) * relax), wp.exp((e1 - m) * relax), wp.exp((e2 - m) * relax))
        Fn[p] = U @ wp.diag(s) @ wp.transpose(V)


@dataclass(frozen=True)
class SimConfig:
    """Discretisation settings.

    Attributes:
        dx: Grid spacing [m]. The cut face and the knife's dilated edge are about this coarse.
        particles_per_axis: Particles per cell along each axis.
        cfl: Fraction of the elastic wave crossing time used as the time step.
        side_margin: Empty space around the block in x and y [m].
        top_margin: Empty space above the block [m]; the knife hovers there.
        board_cells: Height of the cutting board above the grid floor, in cells.
        max_pieces: Most pieces the solver can track (at most 30).
        max_cuts: Most strokes the solver can track.
        max_pairs: Most bonded piece pairs.
        crack_lead: How far past the dilated edge the crack may run ahead of the knife, in cells.
        dilation: How far the blade is dilated where the material is intact, in cells.
        gravity: Gravitational acceleration [m/s^2].
        jitter: Random offset of the initial particles, as a fraction of their spacing.
        seed: Seed for that jitter.
        slice_push: Cut with less energy per area where the edge slides along itself (see the
            module docstring); False cuts every stroke with the full G_c.
        slice_averaging: Decay time of the edge's sliding-speed envelope [s]; longer than half
            a sawing stroke.
        slice_push_max: Largest slice-to-push ratio that counts (see the module docstring).
        tearing: Let the uncut ligament of a partly cut piece tear down where it is pulled open
            harder than ``SoftSolidMaterial.tensile_strength`` (see the module docstring); False
            grows cracks only under the knife.
    """

    dx: float = 2.5e-3
    particles_per_axis: int = 2
    cfl: float = 0.35
    side_margin: float = 0.02
    top_margin: float = 0.03
    board_cells: int = 4
    max_pieces: int = MAX_PIECES
    max_cuts: int = 16
    max_pairs: int = 512
    crack_lead: float = 0.25
    dilation: float = 0.75
    gravity: float = 9.81
    jitter: float = 0.1
    seed: int = 0
    slice_push: bool = True
    slice_averaging: float = 0.25
    slice_push_max: float = 2.0
    tearing: bool = True


class MPMCuttingSim:
    """A block of soft solid on a cutting board, cut by a rigid knife.

    The block is axis-aligned and rests on a board at ``board_z``; the domain spans
    [0, domain_size] in each axis. Drive the knife with ``set_knife`` and ``begin_cut``, or
    hand ``advance`` a ``KnifeTrajectory``.

    Args:
        material: The solid.
        block_size: Size of the block, or of the box around ``inside`` [m].
        knife: Blade geometry.
        config: Discretisation settings.
        device: Warp device.
        inside: Optional shape within that box, for food that is not a box (a reconstructed
            mesh, say): a function taking points relative to the box's lower corner, shape
            (N, 3), and returning which lie in the solid. Its lowest point should touch the
            box's floor, which rests on the board. None fills the box.
    """

    def __init__(
        self,
        material: SoftSolidMaterial,
        block_size,
        *,
        knife: KnifeGeometry = KnifeGeometry(),
        config: SimConfig = SimConfig(),
        device: str = "cpu",
        inside=None,
    ):
        if config.max_pieces > MAX_PIECES:
            raise ValueError(f"max_pieces is limited to {MAX_PIECES}")
        wp.init()
        self.material = material
        self.knife_geometry = knife
        self.config = config
        self.device = device
        dx = config.dx
        self.dx = dx
        block_size = np.asarray(block_size, dtype=float)
        self.board_z = config.board_cells * dx
        self.domain_size = np.array(
            [
                block_size[0] + 2.0 * config.side_margin,
                block_size[1] + 2.0 * config.side_margin,
                self.board_z + block_size[2] + config.top_margin,
            ]
        )
        self.grid_shape = tuple(int(math.ceil(d / dx)) + 1 for d in self.domain_size)
        self.block_min = np.array([config.side_margin, config.side_margin, self.board_z])
        self.block_max = self.block_min + block_size

        # Particles: a lattice that fills the block exactly, lightly jittered.
        h = dx / config.particles_per_axis
        counts = [max(1, int(round(s / h))) for s in block_size]
        axes = [(np.arange(c) + 0.5) * (s / c) for c, s in zip(counts, block_size)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        rng = np.random.default_rng(config.seed)
        spacing = block_size / np.array(counts)
        if inside is None:
            pts = pts + rng.uniform(-config.jitter, config.jitter, pts.shape) * spacing + self.block_min
            depth = np.minimum(pts - self.block_min, self.block_max - pts).min(axis=1)
            skin = depth < float(spacing.max())
            self.particle_volume = float(np.prod(block_size)) / len(pts)
        else:
            # A lattice point is skin when a neighbour along some axis lies outside the solid.
            keep = np.asarray(inside(pts), dtype=bool)
            skin = np.zeros(len(pts), dtype=bool)
            for a in range(3):
                for s in (-1.0, 1.0):
                    q = pts.copy()
                    q[:, a] += s * spacing[a]
                    skin |= (q[:, a] < 0.0) | (q[:, a] > block_size[a]) | ~np.asarray(inside(q), dtype=bool)
            pts, skin = pts[keep], skin[keep]
            if not len(pts):
                raise ValueError("the shape holds no lattice point; is it given in the box's frame?")
            pts = pts + rng.uniform(-config.jitter, config.jitter, pts.shape) * spacing + self.block_min
            self.particle_volume = float(np.prod(spacing))
        self.n_particles = len(pts)
        # Where each particle started, and whether it was on the solid's outer skin: a cut face
        # is whatever surface shows later that was not skin (used for drawing).
        self.x0 = pts.copy()
        self.skin = skin
        self.particle_mass = material.density * self.particle_volume

        # Time step: elastic waves (CFL) and explicit viscosity, both with margin.
        self.dt = config.cfl * dx / material.wave_speed
        if material.viscosity > 0.0:
            self.dt = min(self.dt, 0.5 * material.density * dx * dx / (6.0 * material.viscosity))

        dev = device
        n = self.n_particles
        self.x = wp.array(pts.astype(np.float32), dtype=wp.vec3, device=dev)
        self.v = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.C = wp.zeros(n, dtype=wp.mat33, device=dev)
        self.F = wp.array(np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)), dtype=wp.mat33, device=dev)
        # Elastic deformation of the Maxwell (non-equilibrium) branch; unused for an elastic material.
        self.Fn = wp.array(np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)), dtype=wp.mat33, device=dev)
        self.piece = wp.zeros(n, dtype=int, device=dev)
        self.psi = wp.zeros(n, dtype=float, device=dev)

        gshape = (config.max_pieces,) + self.grid_shape
        self.grid_m = wp.zeros(gshape, dtype=float, device=dev)
        self.grid_v = wp.zeros(gshape, dtype=wp.vec3, device=dev)
        self.grid_g = wp.zeros(gshape, dtype=wp.vec3, device=dev)

        # Crack fronts live on bins along each cut's edge direction, one cell wide.
        diag = float(np.linalg.norm(self.domain_size))
        self.n_bins = int(math.ceil(2.0 * diag / dx)) + 2
        self.bin_umin = -0.5 * self.n_bins * dx
        mc = config.max_cuts
        self.cut_o = wp.zeros(mc, dtype=wp.vec3, device=dev)
        self.cut_u = wp.zeros(mc, dtype=wp.vec3, device=dev)
        self.cut_v = wp.zeros(mc, dtype=wp.vec3, device=dev)
        self.cut_w = wp.zeros(mc, dtype=wp.vec3, device=dev)
        self.front = wp.array(np.full((mc, self.n_bins), 1.0e3, dtype=np.float32), dtype=float, device=dev)
        self.slide = wp.zeros(self.n_bins, dtype=float, device=dev)  # averaged sliding speed per bin
        self.first_crack = wp.array(np.full(mc, 1.0e3, dtype=np.float32), dtype=float, device=dev)
        # The active cut's material, per bin: the top of the material in the cut's plane and how
        # much of the bin's width it fills; each bin's energy per area as a fraction of G_c; and
        # the work done and the fracture energy needed this step.
        self.bin_top = wp.zeros(self.n_bins, dtype=float, device=dev)
        self.bin_width = wp.zeros(self.n_bins, dtype=float, device=dev)
        self.bin_factor = wp.zeros(self.n_bins, dtype=float, device=dev)
        self.crack_work = wp.zeros(1, dtype=float, device=dev)
        self.crack_need = wp.zeros(1, dtype=float, device=dev)
        self.crack_initiated = wp.zeros(mc, dtype=int, device=dev)
        self.pair_a = wp.zeros(config.max_pairs, dtype=int, device=dev)
        self.pair_b = wp.zeros(config.max_pairs, dtype=int, device=dev)
        self.pair_cut = wp.zeros(config.max_pairs, dtype=int, device=dev)
        self.field_side = wp.zeros(config.max_pieces, dtype=int, device=dev)
        self.knife_impulse = wp.zeros(1, dtype=wp.vec3, device=dev)
        self.knife_moment = wp.zeros(1, dtype=wp.vec3, device=dev)
        # Tearing: per bonded pair, the side of the cut's plane its second field lies on; per cut,
        # whether anything is still bonded across it and the board's height in its frame; per
        # bin, the ligament's tension, its moment and highest bonded node, and how long the
        # crack tip has been overloaded.
        self.pair_side = wp.zeros(config.max_pairs, dtype=int, device=dev)
        self.cut_live = wp.zeros(mc, dtype=int, device=dev)
        self.cut_board = wp.zeros(mc, dtype=float, device=dev)
        self.lig_n = wp.zeros((mc, self.n_bins), dtype=float, device=dev)
        self.lig_nw = wp.zeros((mc, self.n_bins), dtype=float, device=dev)
        self.lig_top = wp.zeros((mc, self.n_bins), dtype=float, device=dev)
        self.tear_clock = wp.zeros((mc, self.n_bins), dtype=float, device=dev)
        self.front_prev = wp.zeros((mc, self.n_bins), dtype=float, device=dev)
        # The front each bin's opening reference belongs to, and that reference: how far apart the
        # faces next to the front were at their narrowest since it got there.
        self.front_seen = wp.array(np.full((mc, self.n_bins), 1.0e3, dtype=np.float32), dtype=float, device=dev)
        self.open_ref = wp.array(np.full((mc, self.n_bins), 1.0e3, dtype=np.float32), dtype=float, device=dev)
        # Side (+1 or -1 along e_v) of each field relative to each cut, for fields split by it,
        # and the crack faces' displacement next to each front, summed per side.
        self._sides = np.zeros((config.max_pieces, mc), dtype=np.int32)
        self.field_cut_side = wp.zeros((config.max_pieces, mc), dtype=int, device=dev)
        self.x0_dev = wp.array(self.x0.astype(np.float32), dtype=wp.vec3, device=dev)
        self.open_sum = wp.zeros((mc, self.n_bins, 2), dtype=float, device=dev)
        self.open_cnt = wp.zeros((mc, self.n_bins, 2), dtype=float, device=dev)

        self.time = 0.0
        self.n_pieces = 1
        self.n_cuts = 0
        self.active_cut = -1
        self._pairs = []  # (piece_a, piece_b, cut)
        self._cut_frames = []  # (origin, e_u, e_v, e_w) per cut, numpy
        self._started_cuts = set()  # start times of trajectory strokes already cut
        self.knife_pose = None
        self._knife = KnifeState()
        self._knife.enabled = 0
        # A tearing crack runs no faster than the Rayleigh wave (Viktorov's approximation), so it
        # takes at least this long to cross a cell.
        nu = material.poisson_ratio
        c_s = math.sqrt(material.shear_modulus / material.density)
        self.tear_hold = dx / ((0.862 + 1.14 * nu) / (1.0 + nu) * c_s)

    # -- knife -------------------------------------------------------------------------

    def set_knife(self, pose: KnifePose, lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)):
        """Places the knife. Velocities are world-frame, about the edge midpoint."""
        e_u, e_v, e_w = pose.axes()
        g = self.knife_geometry
        k = KnifeState()
        k.origin = wp.vec3(*pose.origin)
        k.e_u = wp.vec3(*e_u)
        k.e_v = wp.vec3(*e_v)
        k.e_w = wp.vec3(*e_w)
        k.lin_vel = wp.vec3(*lin_vel)
        k.ang_vel = wp.vec3(*ang_vel)
        k.half_length = g.half_length
        k.height = g.height
        k.edge_thickness = g.edge_thickness
        k.spine_thickness = g.spine_thickness
        k.bevel_height = g.bevel_height
        k.friction = self.material.knife_friction
        k.dilation = self.config.dilation * self.dx
        k.enabled = 1
        k.tip_rise = g.tip_rise
        k.belly_length = g.belly_length
        k.spine_taper = g.taper
        self._knife = k
        self.knife_pose = pose

    def begin_cut(self) -> int:
        """Starts a stroke in the knife's current plane.

        Every piece the plane passes through is split by side into two pieces, bonded to
        each other below the new crack front (which starts above everything, i.e. intact).

        Returns:
            Index of the new cut.
        """
        if self.knife_pose is None:
            raise RuntimeError("set_knife before begin_cut")
        if self.n_cuts >= self.config.max_cuts:
            raise RuntimeError("too many cuts")
        c = self.n_cuts
        o = np.asarray(self.knife_pose.origin, dtype=float)
        # The cut's frame lies in the blade's plane but stays level: a rocking blade turns in
        # that plane, and the board check (cut_complete) wants e_w to be "up" within it.
        e_v = np.asarray(self.knife_pose.axes()[1], dtype=float)
        up = np.array([0.0, 0.0, 1.0])
        e_w = up - (up @ e_v) * e_v
        if np.linalg.norm(e_w) < 1e-6:
            raise ValueError("a horizontal blade cannot start a cut")
        e_w = e_w / np.linalg.norm(e_w)
        e_u = np.cross(e_v, e_w)
        k = int(np.argmax(np.abs(e_u)))
        if abs(e_u[k]) > 1.0 - 1e-6:
            # A cut along a grid axis: slide the frame's origin along the edge so that the grid's
            # node columns fall on the bins' centres. Nodes are bonded by their bin's front, and a
            # column on a bin's boundary lands in one bin or its neighbour by rounding, which
            # leaves bins with two columns and bins with none, whose ligament cannot be measured.
            q = (e_u @ o + self.bin_umin) / self.dx
            o = o + (((0.5 - q) + 0.5) % 1.0 - 0.5) * self.dx * e_u
        x = self.x.numpy().astype(float)
        piece = self.piece.numpy()

        # Drop pairs of earlier cuts that went all the way through: nothing bonds there.
        self._pairs = [p for p in self._pairs if not self.cut_complete(p[2], x=x, piece=piece)]

        side = np.where((x - o) @ e_v >= 0.0, 1, -1)
        split = {}
        new_pairs = []
        for p in range(self.n_pieces):
            mask = piece == p
            if not mask.any():
                continue
            sp = side[mask]
            if (sp > 0).any() and (sp < 0).any():
                if self.n_pieces + len(split) >= self.config.max_pieces:
                    raise RuntimeError("too many pieces")
                q = self.n_pieces + len(split)
                piece[mask & (side > 0)] = q
                split[p] = q
                new_pairs.append((p, q, c))
        # A piece bonded across an earlier, unfinished cut stays bonded on both sides of this one.
        derived = []
        for a, b, cc in self._pairs:
            if a in split and b in split:
                derived.append((split[a], split[b], cc))
            if a in split:
                derived.append((split[a], b, cc))
            if b in split:
                derived.append((a, split[b], cc))
        self._pairs = self._pairs + derived + new_pairs
        if len(self._pairs) > self.config.max_pairs:
            raise RuntimeError("too many bonded pairs")
        self.n_pieces += len(split)
        for p, q in split.items():
            # The new field lies where its parent did relative to the earlier cuts.
            self._sides[q] = self._sides[p]
            self._sides[p, c], self._sides[q, c] = -1, 1

        sides = np.zeros(self.config.max_pieces, dtype=np.int32)
        for p in range(self.n_pieces):
            mask = piece == p
            if mask.any():
                sides[p] = 1 if side[mask].mean() >= 0.0 else -1

        self.piece = wp.array(piece.astype(np.int32), dtype=int, device=self.device)
        self.field_side = wp.array(sides, dtype=int, device=self.device)
        self._upload_pairs()
        frames = [self.cut_o, self.cut_u, self.cut_v, self.cut_w]
        host = [f.numpy() for f in frames]
        for arr, val in zip(host, (o, e_u, e_v, e_w)):
            arr[c] = val
        for dev_arr, arr in zip(frames, host):
            dev_arr.assign(arr)
        front = self.front.numpy()
        front[c] = 1.0e3
        self.front.assign(front)
        self.slide.zero_()  # its bins belong to the previous cut's frame
        self._set_bin_material(x, o, e_u, e_v, e_w)
        self.crack_work.zero_()
        self.crack_need.zero_()
        self._cut_frames.append((o, e_u, e_v, e_w))
        board = self.cut_board.numpy()
        board[c] = self._board_w(c)
        self.cut_board.assign(board)
        self.n_cuts += 1
        self.active_cut = c
        return c

    def _set_bin_material(self, x, o, e_u, e_v, e_w):
        """Top and width of the material in each crack-front bin of a new cut's plane.

        The crack's fracture energy is G_c times the new area within the material, so a bin at the
        food's side face, which the material only partly fills, costs only that part.
        """
        h = self.dx / self.config.particles_per_axis
        rel = x - o
        near = np.abs(rel @ e_v) < h
        top = np.full(self.n_bins, -1.0e3)
        width = np.zeros(self.n_bins)
        if near.any():
            u = rel[near] @ e_u
            w = rel[near] @ e_w
            b = np.floor((u - self.bin_umin) / self.dx).astype(int)
            ok = (b >= 0) & (b < self.n_bins)
            b, u, w = b[ok], u[ok], w[ok]
            np.maximum.at(top, b, w + 0.5 * h)
            lo = np.full(self.n_bins, np.inf)
            hi = np.full(self.n_bins, -np.inf)
            np.minimum.at(lo, b, u)
            np.maximum.at(hi, b, u)
            has = np.isfinite(lo)
            width[has] = np.clip(hi[has] - lo[has] + h, 0.0, self.dx)
        self.bin_top.assign(top.astype(np.float32))
        self.bin_width.assign(width.astype(np.float32))

    def _board_w(self, c: int) -> float:
        """Height of the board in cut c's frame, along e_w (which tilts with a rolled blade)."""
        o, _, _, e_w = self._cut_frames[c]
        return float((self.board_z - o[2]) / e_w[2])

    def _drop_finished_pairs(self):
        """Stops bonding across a cut as soon as it has gone all the way through.

        Waiting for the next stroke would keep the two sides joined at the board until then,
        a hinge that holds a cut slab upright. Any unfinished cut may finish, not only the
        active one: a ligament can tear through long after its stroke.
        """
        cuts = {pc for _, _, pc in self._pairs}
        if not cuts:
            return
        x = self.positions()
        done = {c for c in cuts if self.cut_complete(c, x=x)}
        if done:
            self._pairs = [p for p in self._pairs if p[2] not in done]
            self._upload_pairs()

    def _upload_pairs(self):
        a = np.zeros(self.config.max_pairs, dtype=np.int32)
        b = np.zeros_like(a)
        cc = np.zeros_like(a)
        side = np.zeros_like(a)
        live = np.zeros(self.config.max_cuts, dtype=np.int32)
        for idx, (pa, pb, pc) in enumerate(self._pairs):
            a[idx], b[idx], cc[idx] = pa, pb, pc
            side[idx] = self._sides[pb, pc]
            live[pc] = 1
        self.pair_a.assign(a)
        self.pair_b.assign(b)
        self.pair_cut.assign(cc)
        self.pair_side.assign(side)
        self.cut_live.assign(live)
        self.field_cut_side.assign(self._sides)

    # -- stepping ----------------------------------------------------------------------

    def substep(self, dt: float):
        """Advances the simulation by one explicit step of length dt (at most ``self.dt``)."""
        cfg = self.config
        mat = self.material
        dev = self.device
        nx, ny, nz = self.grid_shape
        wp.launch(_clear_grid, dim=(self.n_pieces, nx, ny, nz), inputs=[self.grid_m, self.grid_v, self.grid_g], device=dev)
        wp.launch(
            _p2g,
            dim=self.n_particles,
            inputs=[
                self.x, self.v, self.C, self.F, self.piece, self.Fn,
                self.particle_mass, self.particle_volume,
                mat.equilibrium_shear_modulus, mat.lame_lambda, mat.viscosity, mat.maxwell_shear_modulus,
                self.dx, dt,
                self.grid_m, self.grid_v, self.grid_g, self.psi,
            ],
            device=dev,
        )
        if cfg.tearing and mat.tensile_strength > 0.0 and self._pairs:
            # Before _grid_update merges the bonded fields: what the merge transmits is the load.
            self.lig_n.zero_()
            self.lig_nw.zero_()
            self.lig_top.fill_(-NO_CRACK)
            wp.launch(
                _ligament_load,
                dim=(nx, ny, nz),
                inputs=[
                    self.grid_m, self.grid_v, self.pair_a, self.pair_b, self.pair_cut, self.pair_side,
                    len(self._pairs), self.cut_o, self.cut_u, self.cut_v, self.cut_w, self.front,
                    self.bin_umin, self.n_bins, self.dx, dt, self.lig_n, self.lig_nw, self.lig_top,
                ],
                device=dev,
            )
            self.open_sum.zero_()
            self.open_cnt.zero_()
            wp.launch(
                _tip_opening,
                dim=self.n_particles,
                inputs=[
                    self.x, self.x0_dev, self.piece, self.field_cut_side, self.cut_live, self.n_cuts,
                    self.cut_o, self.cut_u, self.cut_v, self.cut_w, self.front, self.bin_umin, self.n_bins,
                    self.dx, self.dx / cfg.particles_per_axis, self.open_sum, self.open_cnt,
                ],
                device=dev,
            )
            wp.copy(self.front_prev, self.front)  # neighbours' fronts, unaffected by launch order
            wp.launch(
                _tear,
                dim=(self.n_cuts, self.n_bins),
                inputs=[
                    self.front, self.front_prev, self.lig_n, self.lig_nw, self.lig_top, self.open_sum,
                    self.open_cnt, self.cut_live, self.cut_board, self.n_bins, self.dx, dt,
                    mat.tensile_strength, mat.critical_opening, self.tear_hold, self.tear_clock,
                    self.front_seen, self.open_ref, self._knife, self.active_cut,
                    self.cut_o, self.cut_u, self.cut_w, self.bin_umin,
                ],
                device=dev,
            )
        wp.launch(
            _grid_update,
            dim=(nx, ny, nz),
            inputs=[
                self.grid_m, self.grid_v, self.grid_g, self.n_pieces,
                self.dx, dt, wp.vec3(0.0, 0.0, -cfg.gravity),
                self.pair_a, self.pair_b, self.pair_cut, len(self._pairs),
                self.cut_o, self.cut_u, self.cut_w, self.front,
                self.bin_umin, self.n_bins,
                self._knife, self.active_cut, self.field_side,
                mat.contact_friction, cfg.board_cells, mat.board_friction,
                nx, ny, nz, self.knife_impulse, self.knife_moment, self.crack_work,
            ],
            device=dev,
        )
        if self.active_cut >= 0 and self._knife.enabled == 1:
            # The crack runs as far as this step's push on the intact material pays for.
            c = self.active_cut
            wp.launch(
                _slice_factors, dim=self.n_bins, device=dev,
                inputs=[
                    self._knife, c, self.cut_o, self.cut_u, self.bin_umin, self.dx,
                    1 if cfg.slice_push else 0, self.slide,
                    min(1.0, dt / cfg.slice_averaging) if cfg.slice_averaging > 0.0 else 1.0,
                    cfg.slice_push_max, self.bin_factor,
                ],
            )
            common = [
                self.front, self.bin_top, self.bin_width, self._knife, c, self.cut_o, self.cut_u, self.cut_w,
                self.bin_umin, self.dx, cfg.crack_lead * self.dx, self.board_z, self.bin_factor,
            ]
            wp.launch(
                _crack_need, dim=self.n_bins, device=dev,
                inputs=common + [
                    mat.fracture_energy, mat.initiation_length, self.crack_initiated, self.crack_need,
                ],
            )
            wp.launch(
                _advance_crack, dim=self.n_bins, device=dev,
                inputs=common + [
                    mat.initiation_length, self.crack_work, self.crack_need, self.crack_initiated,
                    self.first_crack,
                ],
            )
            wp.launch(
                _settle_crack_work, dim=1, device=dev,
                inputs=[self.crack_work, self.crack_need, self.crack_initiated, c],
            )
        hi = wp.vec3(*((np.array(self.grid_shape) - 3) * self.dx))
        wp.launch(
            _g2p,
            dim=self.n_particles,
            inputs=[
                self.x, self.v, self.C, self.F, self.piece, self.grid_v, self.dx, dt, 2.0 * self.dx, hi,
                self.Fn, math.exp(-dt / mat.relaxation_time) if mat.relaxation_time > 0.0 else 1.0,
                1 if mat.maxwell_shear_modulus > 0.0 else 0,
            ],
            device=dev,
        )
        self.time += dt

    def advance(self, t_end: float, trajectory=None):
        """Steps until ``t_end``, moving the knife along ``trajectory`` if given.

        Strokes in the trajectory start their cut (``begin_cut``) when they begin.

        Returns:
            (force, moment): mean force [N] and moment [N m] of the material on the knife
            over the interval, the moment about the knife's edge midpoint.
        """
        self.knife_impulse.zero_()
        self.knife_moment.zero_()
        t_start = self.time
        cut_times = trajectory.cut_times if trajectory is not None else []
        fired = 0
        while self.time < t_end - 1e-12:
            dt = min(self.dt, t_end - self.time)
            if trajectory is not None:
                # A trajectory's strokes start their cut once, when the stroke begins.
                while fired < len(cut_times) and cut_times[fired] <= self.time + 1e-12:
                    if cut_times[fired] not in self._started_cuts:
                        pose, lin, ang = trajectory.state(cut_times[fired])
                        self.set_knife(pose, lin, ang)
                        self.begin_cut()
                        self._started_cuts.add(cut_times[fired])
                    fired += 1
                pose, lin, ang = trajectory.state(self.time + 0.5 * dt)
                self.set_knife(pose, lin, ang)
            self.substep(dt)
        self._drop_finished_pairs()
        span = max(self.time - t_start, 1e-12)
        imp = self.knife_impulse.numpy()[0].astype(float)
        mom = self.knife_moment.numpy()[0].astype(float)
        return -imp / span, -mom / span

    # -- queries -----------------------------------------------------------------------

    def positions(self) -> np.ndarray:
        """Particle positions, shape (N, 3)."""
        return self.x.numpy().astype(float)

    def pieces(self) -> np.ndarray:
        """Piece index of each particle, shape (N,)."""
        return self.piece.numpy().astype(int)

    def strain_energy(self) -> np.ndarray:
        """Strain energy density of each particle at the last step [J/m^3]."""
        return self.psi.numpy().astype(float)

    def crack_onset_height(self, c: int) -> float:
        """World height of the edge's lowest point in the food when cut c first cracked (NaN if not)."""
        w = float(self.first_crack.numpy()[c])
        if w > 999.0:
            return float("nan")
        o, _, _, e_w = self._cut_frames[c]
        return float(o[2] + w * e_w[2])

    def crack_front(self, c: int) -> np.ndarray:
        """Crack-front height of cut c per bin, in the cut's frame (1000 = not cracked)."""
        return self.front.numpy()[c].astype(float)

    def piece_groups(self) -> np.ndarray:
        """Physical piece of each grid field, numbered from 0.

        ``n_pieces`` counts grid fields; two fields joined below an unfinished cut (a slit)
        are still one piece of tofu, so they share a group.
        """
        x, pc = self.positions(), self.pieces()
        parent = list(range(self.n_pieces))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        complete = [self.cut_complete(c, x=x, piece=pc) for c in range(self.n_cuts)]
        for a, b, c in self._pairs:
            if not complete[c]:
                parent[find(a)] = find(b)
        roots = [find(f) for f in range(self.n_pieces)]
        index = {r: i for i, r in enumerate(dict.fromkeys(roots))}
        return np.array([index[r] for r in roots], dtype=int)

    def connected_pieces(self) -> int:
        """Number of pieces that are physically apart (see ``piece_groups``)."""
        return len(np.unique(self.piece_groups()[np.unique(self.pieces())]))

    def crack_faces(self, x=None) -> np.ndarray:
        """Particles on the faces of an unfinished cut: near its plane and above its front.

        For drawing a slit: the split into fields runs across the whole plane from the start
        of a stroke, but the tofu is only open where the crack has run. A finished cut parts
        the tofu into separate pieces, which are drawn by their outlines instead.
        """
        if x is None:
            x = self.positions()
        mask = np.zeros(len(x), dtype=bool)
        fronts = self.front.numpy()
        for c in range(self.n_cuts):
            if self.cut_complete(c, x=x):
                continue
            o, e_u, e_v, e_w = self._cut_frames[c]
            rel = x - o
            near = np.abs(rel @ e_v) < 0.6 * self.dx
            if not near.any():
                continue
            bins = np.clip(np.floor((rel @ e_u - self.bin_umin) / self.dx).astype(int), 0, self.n_bins - 1)
            mask |= near & (rel @ e_w > fronts[c][bins])
        return mask

    def cut_complete(self, c: int, x=None, piece=None) -> bool:
        """Whether cut c's crack front reached the board everywhere the material crosses its plane.

        It counts as through once the front is inside the bottom layer of particles: what is
        left is thinner than one particle spacing, which the particles cannot tell from nothing,
        and the knife's crack often ends a fraction of a millimetre short of the board. A front
        that stops higher leaves a ligament, however thin, which holds until it tears; within 1.5
        cells, as it used to be, a 4.5 mm ligament left on purpose on a 4 mm grid parted by
        bookkeeping.
        """
        if c >= self.n_cuts:
            return False
        front = self._material_fronts(c, x)
        return bool(np.all(front < self._board_w(c) + self.dx / self.config.particles_per_axis))

    def ligament_height(self, c: int, x=None) -> float:
        """Height of the uncut material left under cut c's crack front, where most is left [m].

        0 once the cut counts as through (``cut_complete``); the material's whole height where
        the crack never started.
        """
        if c >= self.n_cuts or self.cut_complete(c, x):
            return 0.0
        front, top = self._material_fronts(c, x, with_top=True)
        return max(0.0, float(np.minimum(front, top).max()) - self._board_w(c))

    def _material_fronts(self, c: int, x=None, with_top=False):
        """Crack front of cut c in the bins the material crosses (within two cells of its plane).

        With ``with_top``, also the top of the material in each of those bins.
        """
        if x is None:
            x = self.positions()
        o, e_u, e_v, e_w = self._cut_frames[c]
        rel = x - o
        near = np.abs(rel @ e_v) < 2.0 * self.dx
        bins = np.floor((rel[near] @ e_u - self.bin_umin) / self.dx).astype(int)
        bins = np.clip(bins, 0, self.n_bins - 1)
        used, index = np.unique(bins, return_inverse=True)
        front = self.crack_front(c)[used]
        if not with_top:
            return front
        top = np.full(len(used), -np.inf)
        np.maximum.at(top, index, rel[near] @ e_w)
        return front, top
