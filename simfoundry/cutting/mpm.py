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
* The crack front advances only where the strain energy density in the intact material
  just under the edge reaches G_c / dx (a crack-band criterion). So the knife first presses
  the food down, then cuts, and the force it needs scales with the fracture energy. Along
  the edge the front is continuous, and the last layer pinched against the board parts.
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
    pmass: float,
    pvol: float,
    mu: float,
    lam: float,
    eta: float,
    dx: float,
    dt: float,
    grid_m: wp.array4d(dtype=float),
    grid_v: wp.array4d(dtype=wp.vec3),
    grid_g: wp.array4d(dtype=wp.vec3),
    psi_out: wp.array(dtype=float),
    knife: KnifeState,
    active_cut: int,
    cut_o: wp.array(dtype=wp.vec3),
    cut_u: wp.array(dtype=wp.vec3),
    umin: float,
    nbins: int,
    edge_psi: wp.array(dtype=float),
):
    p = wp.tid()
    inv_dx = 1.0 / dx
    xp = x[p]
    f = piece[p]
    tau, psi = _stress(F[p], C[p], mu, lam, eta)
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
    if active_cut >= 0 and knife.enabled == 1:
        # Energy of the material just under the edge, per crack-front bin, for the crack criterion.
        r = xp - knife.origin
        ku = wp.dot(r, knife.e_u)
        kv = wp.dot(r, knife.e_v)
        kw = wp.dot(r, knife.e_w)
        if wp.abs(kv) < 1.5 * dx and kw > -2.0 * dx and kw < 0.5 * dx and wp.abs(ku) <= knife.half_length:
            b = int(wp.floor((wp.dot(xp - cut_o[active_cut], cut_u[active_cut]) - umin) / dx))
            if b >= 0 and b < nbins:
                wp.atomic_max(edge_psi, b, psi)


@wp.kernel
def _advance_crack(
    front: wp.array2d(dtype=float),
    edge_psi: wp.array(dtype=float),
    knife: KnifeState,
    c: int,
    cut_o: wp.array(dtype=wp.vec3),
    cut_u: wp.array(dtype=wp.vec3),
    cut_w: wp.array(dtype=wp.vec3),
    umin: float,
    nbins: int,
    dx: float,
    psi_crit: float,
    lead: float,
    board_z: float,
):
    b = wp.tid()
    # Where along the edge is this bin, in the cut's frame?
    denom = wp.dot(knife.e_u, cut_u[c])
    if wp.abs(denom) < 1.0e-6:
        return
    o = knife.origin - cut_o[c]
    ub = umin + (float(b) + 0.5) * dx
    s = (ub - wp.dot(o, cut_u[c])) / denom
    if wp.abs(s) > knife.half_length:
        return
    w_edge = wp.dot(o + s * knife.e_u, cut_w[c])
    if w_edge >= front[c, b]:
        return
    crack = edge_psi[b] >= psi_crit
    # A crack front is continuous along the edge: where the neighbouring bin is already cut
    # below the edge, this one goes too. Without this, the less confined material at the
    # food's free side faces never reaches the threshold and stays joined.
    if b > 0:
        if front[c, b - 1] <= w_edge:
            crack = True
    if b < nbins - 1:
        if front[c, b + 1] <= w_edge:
            crack = True
    # The last layer pinched between the edge and the board parts as well.
    gap = knife.origin[2] + s * knife.e_u[2] - board_z
    if gap <= knife.dilation:
        crack = True
    if crack:
        new_front = w_edge - lead
        if gap <= knife.dilation:
            new_front = w_edge - gap - dx
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
        kw = wp.dot(r, knife.e_w)
        reach = knife.dilation + dx
        near = wp.abs(ku) <= knife.half_length + dx and kw <= knife.height + dx and kw >= -reach
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
    F[p] = (wp.identity(n=3, dtype=float) + dt * nC) @ F[p]


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
        crack_lead: How far ahead of the edge a crack opens, in cells.
        dilation: How far the blade is dilated where the material is intact, in cells.
        gravity: Gravitational acceleration [m/s^2].
        jitter: Random offset of the initial particles, as a fraction of their spacing.
        seed: Seed for that jitter.
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
    crack_lead: float = 0.5
    dilation: float = 0.75
    gravity: float = 9.81
    jitter: float = 0.1
    seed: int = 0


class MPMCuttingSim:
    """A block of soft solid on a cutting board, cut by a rigid knife.

    The block is axis-aligned and rests on a board at ``board_z``; the domain spans
    [0, domain_size] in each axis. Drive the knife with ``set_knife`` and ``begin_cut``, or
    hand ``advance`` a ``KnifeTrajectory``.
    """

    def __init__(
        self,
        material: SoftSolidMaterial,
        block_size,
        *,
        knife: KnifeGeometry = KnifeGeometry(),
        config: SimConfig = SimConfig(),
        device: str = "cpu",
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
        pts = pts + rng.uniform(-config.jitter, config.jitter, pts.shape) * spacing + self.block_min
        self.n_particles = len(pts)
        self.particle_volume = float(np.prod(block_size)) / self.n_particles
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
        self.edge_psi = wp.zeros(self.n_bins, dtype=float, device=dev)
        self.pair_a = wp.zeros(config.max_pairs, dtype=int, device=dev)
        self.pair_b = wp.zeros(config.max_pairs, dtype=int, device=dev)
        self.pair_cut = wp.zeros(config.max_pairs, dtype=int, device=dev)
        self.field_side = wp.zeros(config.max_pieces, dtype=int, device=dev)
        self.knife_impulse = wp.zeros(1, dtype=wp.vec3, device=dev)
        self.knife_moment = wp.zeros(1, dtype=wp.vec3, device=dev)

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
        # Strain energy density at which the crack under the edge advances (crack band one cell wide).
        self.psi_crit = material.fracture_energy / dx

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
        e_u, e_v, e_w = self.knife_pose.axes()
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
        self._cut_frames.append((o, e_u, e_v, e_w))
        self.n_cuts += 1
        self.active_cut = c
        return c

    def _upload_pairs(self):
        a = np.zeros(self.config.max_pairs, dtype=np.int32)
        b = np.zeros_like(a)
        cc = np.zeros_like(a)
        for idx, (pa, pb, pc) in enumerate(self._pairs):
            a[idx], b[idx], cc[idx] = pa, pb, pc
        self.pair_a.assign(a)
        self.pair_b.assign(b)
        self.pair_cut.assign(cc)

    # -- stepping ----------------------------------------------------------------------

    def substep(self, dt: float):
        """Advances the simulation by one explicit step of length dt (at most ``self.dt``)."""
        cfg = self.config
        mat = self.material
        dev = self.device
        nx, ny, nz = self.grid_shape
        wp.launch(_clear_grid, dim=(self.n_pieces, nx, ny, nz), inputs=[self.grid_m, self.grid_v, self.grid_g], device=dev)
        self.edge_psi.zero_()
        wp.launch(
            _p2g,
            dim=self.n_particles,
            inputs=[
                self.x, self.v, self.C, self.F, self.piece,
                self.particle_mass, self.particle_volume,
                mat.shear_modulus, mat.lame_lambda, mat.viscosity,
                self.dx, dt,
                self.grid_m, self.grid_v, self.grid_g, self.psi,
                self._knife, self.active_cut, self.cut_o, self.cut_u,
                self.bin_umin, self.n_bins, self.edge_psi,
            ],
            device=dev,
        )
        if self.active_cut >= 0 and self._knife.enabled == 1:
            wp.launch(
                _advance_crack,
                dim=self.n_bins,
                inputs=[
                    self.front, self.edge_psi, self._knife, self.active_cut,
                    self.cut_o, self.cut_u, self.cut_w, self.bin_umin, self.n_bins, self.dx,
                    self.psi_crit, cfg.crack_lead * self.dx, self.board_z,
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
                nx, ny, nz, self.knife_impulse, self.knife_moment,
            ],
            device=dev,
        )
        hi = wp.vec3(*((np.array(self.grid_shape) - 3) * self.dx))
        wp.launch(
            _g2p,
            dim=self.n_particles,
            inputs=[self.x, self.v, self.C, self.F, self.piece, self.grid_v, self.dx, dt, 2.0 * self.dx, hi],
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
        """Whether cut c's crack front reached the board everywhere the material crosses its plane."""
        if c >= self.n_cuts:
            return False
        if x is None:
            x = self.positions()
        o, e_u, e_v, e_w = self._cut_frames[c]
        rel = x - o
        near = np.abs(rel @ e_v) < 2.0 * self.dx
        if not near.any():
            return True
        bins = np.floor((rel[near] @ e_u - self.bin_umin) / self.dx).astype(int)
        bins = np.clip(bins, 0, self.n_bins - 1)
        front = self.crack_front(c)[np.unique(bins)]
        w_board = float((np.array([0.0, 0.0, self.board_z]) - o) @ e_w)
        return bool(np.all(front < w_board + 1.5 * self.dx))
