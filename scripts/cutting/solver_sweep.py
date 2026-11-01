# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One vertical cut through a block (the demo's `single` scenario) with any material, on CPU.

Logs what the design needs to judge whether the solver still cuts at a given G_c and E: knife force, crack front and
crack area, knife work, stored strain energy, the most compressed particle, the floating particles, the work ledger and
the crack area split by the path that made it, and wall time per substep. No rendering. Options and their defaults are
the ones the research's reference runs were made with (their JSON "args" replays through this script).

Usage (CUDA hidden so Warp cannot touch the shared GPU):
    CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree> python scripts/cutting/solver_sweep.py --gc 30 --out run.json
"""

import argparse
import json
import math
import sys
import time

import numpy as np

from simfoundry.cutting.knife import KnifeGeometry, KnifeTrajectory
from simfoundry.cutting.materials import FIRM_TOFU, SoftSolidMaterial
from simfoundry.cutting.mpm import LEDGER_NAMES, MPMCuttingSim, SimConfig

ACROSS = -math.pi / 2.0  # as in scripts/cutting/tofu_cutting_demo.py


def descent_rows(rows, H):
    """The rows of the descent: the depth grows from one row to the next, between 2 mm and H - 2 mm deep.

    Args:
        rows: The samples, in time order; each has ``depth`` [m].
        H: The block's height [m].

    Returns:
        The rows of the descent, in time order.
    """
    return [r for k, r in enumerate(rows) if k > 0 and r["depth"] > rows[k - 1]["depth"] + 1e-7
            and 0.002 < r["depth"] < H - 0.002]


def force_ratio(cut_rows, gc, width):
    """Mean vertical knife force over the descent, divided by G_c times the edge's width (1 = G_c sets the force).

    Returns:
        (the ratio, the mean force [N]); both are 0 for an empty descent.
    """
    if not cut_rows:
        return 0.0, 0.0
    mean_fz = float(np.mean([r["fz"] for r in cut_rows]))
    return mean_fz / (gc * width), mean_fz


def lag_mm(cut_rows, width):
    """The crack's lag, edge depth minus the mean crack depth (crack area over ``width``), in mm: its mean and max.

    A positive lag: the edge is below the crack, pressing on intact material.

    Returns:
        (mean, max) in mm, or (None, None) with no rows.
    """
    if not cut_rows:
        return None, None
    lag = [r["depth"] - r["crack_area"] / width for r in cut_rows]
    return 1e3 * float(np.mean(lag)), 1e3 * float(np.max(lag))


def closure_residual(W, spent, U_end, K_end):
    """What the knife's work did not put into the crack, the material's strain and its motion [J], and as a fraction of W.

    The criterion: this residual is at least -0.02 W. ``spent`` is the crack ledger's spent work (the work the crack
    took, which for a vertical press is G_c times the area cut by the knife).
    """
    residual = W - spent - U_end - K_end
    return residual, residual / max(W, 1e-300)


def ledger_statistics(L, gc, knife_area):
    """The ledger's checks and step statistics, from its entries by name (LEDGER_NAMES) and the knife's crack area.

    Args:
        L: dict of the ledger, name -> value (energies in J; ``need_steps`` etc. are counts).
        gc: G_c [J/m^2].
        knife_area: the energy-balance crack area, the first crack included [m^2].

    Returns:
        A dict: the first crack's area ``A_init_m2`` and the area after it ``A_prop_m2``; the first crack's charge
        over G_c times its area ``init_overcharge_J`` (the existing initiation rule's side effect, R9 of the solver
        request); ``prop_rel`` = |spent_prop - G_c A_prop| / spent_prop, the vertical-press check (0 with nothing
        spent); the share of steps whose work fell short of the need, ``short_step_fraction``; the mean of
        min(work / need, 1), ``mean_pay_ratio``; and the need at the pinch threshold, ``unpaid_at_pinch_J``.
    """
    A_init = L["initiation_area"]
    A_prop = knife_area - A_init
    spent_prop = L["spent_prop"]
    n = L["need_steps"]
    return dict(
        A_init_m2=A_init,
        A_prop_m2=A_prop,
        G_c_A_init_J=gc * A_init,
        init_overcharge_J=L["spent_init"] - gc * A_init,
        prop_rel=abs(spent_prop - gc * A_prop) / max(spent_prop, 1e-300) if spent_prop > 0.0 else 0.0,
        short_step_fraction=(L["short_steps"] / n) if n > 0 else None,
        mean_pay_ratio=(L["pay_ratio_sum"] / n) if n > 0 else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=float, default=FIRM_TOFU.youngs_modulus)
    ap.add_argument("--nu", type=float, default=FIRM_TOFU.poisson_ratio)
    ap.add_argument("--rho", type=float, default=FIRM_TOFU.density)
    ap.add_argument("--gc", type=float, default=FIRM_TOFU.fracture_energy)
    ap.add_argument("--visc", type=float, default=FIRM_TOFU.viscosity)
    ap.add_argument("--relaxed-ratio", type=float, default=0.25, help="relaxed E / E; 0 = elastic")
    ap.add_argument("--tau", type=float, default=FIRM_TOFU.relaxation_time)
    ap.add_argument("--sigma-ratio", type=float, default=0.1, help="tensile strength / E")
    ap.add_argument("--dx", type=float, default=2.5e-3)
    ap.add_argument("--block", type=float, nargs=3, default=(0.08, 0.05, 0.03))
    ap.add_argument("--speed", type=float, default=0.05)
    ap.add_argument("--sample-hz", type=float, default=240.0)
    ap.add_argument("--no-tearing", action="store_true")
    ap.add_argument("--max-wall", type=float, default=1700.0, help="stop after this many seconds")
    ap.add_argument("--clearance", type=float, default=0.012, help="edge height above the food at the start")
    ap.add_argument("--stop-at-board", action="store_true", help="stop 0.05 s after the edge reaches the board")
    ap.add_argument("--start-time", type=float, default=0.15, help="settling time before the stroke [s]")
    ap.add_argument("--cfl", type=float, default=0.35, help="fraction of the wave crossing time per substep")
    ap.add_argument("--crack-lead", type=float, default=0.25)
    ap.add_argument("--dilation", type=float, default=0.75)
    ap.add_argument("--ppa", type=int, default=2, help="particles per cell along each axis")
    ap.add_argument("--board-mu", type=float, default=0.5)
    ap.add_argument("--knife-mu", type=float, default=0.3)
    ap.add_argument("--gravity-scale", type=float, default=1.0, help="extra factor on gravity (weight x this)")
    ap.add_argument("--mass-scale", type=float, default=1.0,
                    help="density x s and gravity / s: the same weight, a sqrt(s) longer time step")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rr = a.relaxed_ratio
    mat = SoftSolidMaterial(
        name="sweep", youngs_modulus=a.E, poisson_ratio=a.nu, density=a.rho * a.mass_scale, viscosity=a.visc,
        fracture_energy=a.gc, relaxed_modulus=a.E * rr if rr > 0 else 0.0,
        relaxation_time=a.tau if rr > 0 else 0.0, tensile_strength=a.E * a.sigma_ratio,
        board_friction=a.board_mu, knife_friction=a.knife_mu,
    )
    sim = MPMCuttingSim(mat, list(a.block), knife=KnifeGeometry(),
                        config=SimConfig(dx=a.dx, tearing=not a.no_tearing, gravity=9.81 * a.gravity_scale / a.mass_scale, cfl=a.cfl,
                                         crack_lead=a.crack_lead, dilation=a.dilation, particles_per_axis=a.ppa),
                        device="cpu")
    top_z = float(sim.block_max[2])
    c = 0.5 * (sim.block_min + sim.block_max)
    traj = KnifeTrajectory.strokes([(c[0], c[1], ACROSS)], top_z=top_z, board_z=sim.board_z,
                                   descend_speed=a.speed, retreat=0.08, clearance=a.clearance,
                                   start_time=a.start_time)
    vol_p = sim.particle_mass / mat.density
    width = float(a.block[1])  # the edge spans the block's y extent in this cut

    rows, pos_prev, W = [], None, 0.0
    t_wall0 = time.time()
    solver_wall, t_board = 0.0, None
    n = int(math.ceil(traj.duration * a.sample_hz))
    aborted = False
    for i in range(1, n + 1):
        t = min(i / a.sample_hz, traj.duration)
        ts = time.time()
        f, _ = sim.advance(t, traj)
        solver_wall += time.time() - ts
        pose = sim.knife_pose
        pos = np.array(pose.origin, dtype=float)
        if pos_prev is not None:
            W += -float(np.dot(f, pos - pos_prev))
        pos_prev = pos
        front_h, area = float("nan"), 0.0
        if sim.active_cut >= 0:
            o = sim._cut_frames[sim.active_cut][0]
            fr = sim.crack_front(sim.active_cut)
            top = sim.bin_top.numpy().astype(float)
            wid = sim.bin_width.numpy().astype(float)
            board_w = sim.board_z - o[2]
            eff = np.clip(np.minimum(fr, top), board_w, None)
            area = float(np.sum(wid * np.maximum(top - eff, 0.0)))
            if fr.min() < 999.0:
                front_h = float(np.min(eff) + o[2] - sim.board_z)
        Fm = sim.F.numpy()
        J = np.linalg.det(Fm)
        psi = sim.psi.numpy().astype(float)
        vel = sim.v.numpy().astype(float)
        rows.append(dict(
            t=round(t, 5), edge_h=float(pos[2] - sim.board_z), depth=float(top_z - pos[2]), fz=float(f[2]),
            fy=float(f[1]), front_h=front_h, crack_area=area, W_knife=W,
            U_strain=float(psi.sum() * vol_p), K=float(0.5 * sim.particle_mass * (vel ** 2).sum()),
            J_min=float(J.min()), pieces=sim.connected_pieces(), nan=bool(np.isnan(Fm).any()),
            floating=sim.floating_count(),
        ))
        if rows[-1]["nan"] or time.time() - t_wall0 > a.max_wall:
            aborted = True
            break
        if t_board is None and rows[-1]["edge_h"] < 0.5e-3:
            t_board = t
        if a.stop_at_board and t_board is not None and t > t_board + 0.05:
            break
    wall = time.time() - t_wall0
    H = float(a.block[2])
    # The descent only (depth growing), away from the top and the board.
    cut = descent_rows(rows, H)
    fr_value, mean_fz = force_ratio(cut, a.gc, width)
    lag_mean, lag_max = lag_mm(cut, width)
    pre_board = [r for r in rows if r["edge_h"] >= 1.0e-3]
    area_pre_board = pre_board[-1]["crack_area"] if pre_board else 0.0

    # The design's measures (R1), next to the legacy ones above, which the reference report already used.
    has_cut = sim.n_cuts > 0
    ledger = sim.ledger(0) if has_cut else np.zeros(len(LEDGER_NAMES))
    paths = sim.crack_area_paths(0) if has_cut else np.zeros(3)
    L = dict(zip(LEDGER_NAMES, ledger.tolist()))
    stats = ledger_statistics(L, a.gc, float(paths[0]))
    U_end, K_end = rows[-1]["U_strain"], rows[-1]["K"]
    W_end = rows[-1]["W_knife"]
    pb = sim.pre_board(0) if has_cut else None
    kind_work = sim.knife_work_by_kind()
    W_kind = float(kind_work.sum())
    binw_sum = float(sim.bin_width.numpy().astype(float).sum()) if has_cut else width
    lag_binw_mean = lag_mm(cut, binw_sum)[0] if has_cut else None
    closure_J, closure_frac = closure_residual(W_end, float(ledger[0]), U_end, K_end)
    res = dict(
        args=vars(a), particles=int(sim.n_particles), dt=float(sim.dt), substeps=int(round(sim.time / sim.dt)),
        sim_time=float(sim.time), wall_s=round(wall, 2), solver_wall_s=round(solver_wall, 2),
        solver_ms_per_substep=round(1e3 * solver_wall / max(1, sim.time / sim.dt), 3),
        wall_per_substep_ms=round(1e3 * wall / max(1, sim.time / sim.dt), 3),
        # Legacy field under its old name: the crack area at the last sample with the edge at least 1 mm above the
        # board, over the block's full width times H. Its threshold (1 mm) is not the design's
        # (dilation * dx + 0.5 mm) and its denominator is not the sum of bin widths times (H - height), so it
        # differs from pre_board_ratio: 1.02 where the design gives 0.61 on the B run. The 1 mm threshold lies inside
        # the pinch zone (the edge within dilation * dx + 0.5 mm of the board), where this number reads about 1 even
        # when the cut was not made through before the board, so it is no evidence either way. Reported beside the
        # design's ratio, not instead of it.
        cut_by_crack_before_board=float(area_pre_board / (width * H)),
        pre_board_ratio=(None if pb is None else float(sim.pre_board_ratio(0))),
        pre_board_area_m2=(None if pb is None else float(pb["area"])),
        pre_board_height_mm=(None if pb is None else round(1e3 * pb["height"], 3)),
        pre_board_W_J=(None if pb is None else float(pb["W"])),
        pre_board_gca_over_w=(None if pb is None or pb["W"] <= 0.0 else float(a.gc * pb["area"] / pb["W"])),
        aborted=aborted, nan=any(r["nan"] for r in rows),
        mean_fz_cutting=mean_fz, max_fz=float(max(r["fz"] for r in rows)),
        gc_width_N=a.gc * width, force_ratio=fr_value,
        crack_onset_depth_mm=(None if not np.isfinite(sim.crack_onset_height(0)) else
                              round(1e3 * (top_z - sim.crack_onset_height(0)), 2)) if sim.n_cuts else None,
        cut_complete=bool(sim.cut_complete(0)) if sim.n_cuts else False,
        ligament_mm=round(1e3 * sim.ligament_height(0), 2) if sim.n_cuts else None,
        final_crack_area_cm2=round(1e4 * rows[-1]["crack_area"], 3), full_area_cm2=round(1e4 * width * H, 3),
        W_knife_J=W_end, W_kind_J=kind_work.tolist(), W_kind_total_J=W_kind,
        W_kind_vs_knife_rel=float(abs(W_kind - W_end) / max(abs(W_end), 1e-300)),
        fracture_energy_J=a.gc * rows[-1]["crack_area"],
        energy_ratio=float(a.gc * rows[-1]["crack_area"] / max(W_end, 1e-12)),
        mean_lag_mm=lag_mean, max_lag_mm=lag_max,
        mean_lag_binw_mm=lag_binw_mean,
        # The ledger by name. Energies in J except the counts need_steps and short_steps and the sums pay_ratio_sum
        # and need_last (see the notes): need_shortfall_stepsum is a DIAGNOSTIC, the whole need left over summed over
        # steps, so one unfinished crack is counted again at every step; it is not an energy.
        ledger=L,
        crack_area_paths_m2=dict(zip(["knife", "pinch", "tear"], paths.tolist())),
        crack_area_m2=float(paths.sum()),
        # The first crack's over-charge (the need it was given, against the area it opened) and the vertical-press
        # check on the part after the first crack. A_init_m2 and A_prop_m2 are the knife's area before and after it.
        initiation_area_m2=stats["A_init_m2"], knife_area_after_first_crack_m2=stats["A_prop_m2"],
        G_c_A_init_J=stats["G_c_A_init_J"], init_overcharge_J=stats["init_overcharge_J"],
        spent_prop_vs_G_c_A_knife_prop_rel=stats["prop_rel"],
        short_step_fraction=stats["short_step_fraction"], mean_pay_ratio=stats["mean_pay_ratio"],
        unpaid_at_pinch_J=(None if pb is None else pb["need_at_pinch"]),
        notes=dict(
            spent="the need charged (spent_init at the first crack, spent_prop after it); the closure uses it",
            spent_init="the first crack's charge: an existing side effect of the initiation rule (R9 to fix)",
            need_shortfall_stepsum="DIAGNOSTIC, not an energy: the whole need left over, summed over steps",
            unpaid_at_pinch_J="the need of the step at which the edge first reaches the pinch threshold",
            mean_lag_mm="delay with the block's width as the denominator (the reference report's value)",
            mean_lag_binw_mm="delay with the sum of the bin widths as the denominator (the design's definition)",
        ),
        closure_J=float(closure_J), closure_frac_of_W=float(closure_frac),
        U_end_J=U_end, K_end_J=K_end, K_max_J=float(max(r["K"] for r in rows)),
        K_max_over_W=float(max(r["K"] for r in rows) / max(W_end, 1e-300)),
        floating_max=int(max(r["floating"] for r in rows)),
        J_min=float(min(r["J_min"] for r in rows)),
        Pi=a.gc / (a.E * a.dx),
        rows=rows,
    )
    with open(a.out, "w") as fh:
        json.dump(res, fh)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
