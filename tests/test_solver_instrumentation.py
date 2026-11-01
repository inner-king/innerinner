# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The instruments of the knife-cut solver: knife work by node kind, the crack ledger, the crack area by path, the
pre-board ratio, the energies, and the metric formulas of scripts/cutting/solver_sweep.py.

The physics itself is checked in tests/test_tofu_cutting.py; these check that the instruments read it right. Most tests
share one short stroke through a small block (a module-scoped fixture), so the whole file runs in about a minute.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from simfoundry.cutting.knife import KnifeTrajectory  # noqa: E402
from simfoundry.cutting.materials import FIRM_TOFU  # noqa: E402
from simfoundry.cutting.knife import KnifePose  # noqa: E402
from simfoundry.cutting.mpm import (  # noqa: E402
    LEDGER_NAMES,
    MPMCuttingSim,
    SimConfig,
    _settle_crack_work,
    crack_area_from_bins,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cutting"))
import solver_sweep as S  # noqa: E402

DX = 4e-3
BLOCK = (0.032, 0.024, 0.016)
CONFIG = SimConfig(dx=DX, side_margin=0.012, top_margin=0.016)
ACROSS = -math.pi / 2.0
SAMPLE_DT = 0.01


def _sim():
    return MPMCuttingSim(FIRM_TOFU, BLOCK, config=CONFIG, device="cpu")


@pytest.fixture(scope="module")
def stroke():
    """One vertical cut through the small block, sampled every 10 ms; the instruments are read at each sample."""
    sim = _sim()
    c0 = 0.5 * (sim.block_min + sim.block_max)
    traj = KnifeTrajectory.strokes([(c0[0] + 1.0e-3, c0[1], ACROSS)], top_z=sim.block_max[2], board_z=sim.board_z,
                                   start_time=0.05, clearance=0.006, descend_speed=0.08, settle=0.1)
    rec = dict(sim=sim, t=[], forces=[], origins=[], kind=[], low=[], pre_first=None, imp=[], side=[],
               cracked=[], active=[])
    t = 0.0
    while t < traj.duration:
        t = min(t + SAMPLE_DT, traj.duration)
        f, _ = sim.advance(t, traj)
        rec["t"].append(t)
        rec["forces"].append(np.asarray(f, float).copy())
        rec["origins"].append(np.asarray(sim.knife_pose.origin, float).copy())
        rec["kind"].append(sim.knife_work_by_kind().copy())
        rec["imp"].append(sim.knife_impulse_by_kind().copy())
        rec["side"].append(sim.knife_side_impulse_by_kind().copy())
        rec["active"].append(sim.active_cut >= 0)
        rec["cracked"].append(bool(sim.active_cut >= 0 and sim.crack_front(sim.active_cut).min() < 999.0))
        rec["low"].append(float(sim.knife_geometry.lowest_edge_point(sim.knife_pose)[2] - sim.board_z))
        if rec["pre_first"] is None and sim.pre_board(0) is not None:
            rec["pre_first"] = dict(sim.pre_board(0))
            rec["pre_first_index"] = len(rec["t"]) - 1
    return rec


# ---- the metric formulas, on their own ------------------------------------------------------------------------------

def test_crack_area_is_each_bins_width_times_its_height_above_the_board():
    top = np.array([0.01, 0.01, 0.01, -1.0e3, 0.01])
    width = np.array([0.004, 0.004, 0.0, 0.004, 0.004])
    front = np.array([0.005, 1.0e3, 0.002, 0.0, -0.5])
    # bin 0: the crack runs 5 mm down from the top (0.010 - 0.005), all above the board at 1 mm: 4 mm x 5 mm.
    # bin 1 is not cracked, bin 2 has zero width, bin 3 has no material, and bin 4 is cracked below the board, so its
    # whole height above the board counts: 4 mm x (10 mm - 1 mm).
    assert crack_area_from_bins(front, top, width, 0.001) == pytest.approx(0.004 * 0.005 + 0.004 * 0.009)


def test_descent_rows_are_the_rows_where_the_depth_grows_between_2_mm_and_H_minus_2_mm():
    depth = [0.0, 0.001, 0.003, 0.0035, 0.0034, 0.0045, 0.0150, 0.0170]
    rows = [{"depth": d, "fz": 1.0} for d in depth]
    kept = S.descent_rows(rows, 0.016)
    assert [r["depth"] for r in kept] == [0.003, 0.0035, 0.0045]  # 0.001 is above 2 mm; 0.0034 is a rise; 0.0150 is
    # above H - 2 mm = 0.014; the first row (no predecessor) never counts.


def test_force_ratio_is_the_mean_descent_force_over_G_c_times_the_width():
    rows = [{"fz": 0.2}, {"fz": 0.4}]
    ratio, mean = S.force_ratio(rows, gc=3.0, width=0.024)
    assert mean == pytest.approx(0.3)
    assert ratio == pytest.approx(0.3 / (3.0 * 0.024))
    assert S.force_ratio([], 3.0, 0.024) == (0.0, 0.0)


def test_lag_is_the_edge_depth_minus_the_crack_depth_in_mm():
    rows = [{"depth": 0.010, "crack_area": 0.0001}, {"depth": 0.012, "crack_area": 0.0003}]
    mean, worst = S.lag_mm(rows, 0.01)  # crack depths 0.01 and 0.03: lags 0 and -0.018
    assert mean == pytest.approx(-9.0)
    assert worst == pytest.approx(0.0)
    assert S.lag_mm([], 0.01) == (None, None)


def test_closure_is_W_minus_spent_minus_strain_and_kinetic_energy():
    residual, frac = S.closure_residual(W=1.0, spent=0.5, U_end=0.2, K_end=0.1)
    assert residual == pytest.approx(0.2)
    assert frac == pytest.approx(0.2)


# The research's reference run A (setA, the JSON replayed by solver_sweep.py): the knife's work, the ledger's spent
# work, and the energies at the end of the stroke, as solver_sweep.py wrote them.
A_W, A_SPENT, A_U_END, A_K_END = 4.356232534288483e-3, 4.111786372959614e-3, 1.0744231522042957e-4, 9.103591707098524e-8
A_FULL_CRACK_AREA = 1.509016816271469e-3  # the whole crack at the end of run A


def test_closure_with_the_spent_work_holds_on_reference_run_A():
    """Criterion 3 on run A: W - spent - U_end - K_end >= -0.02 W, with the spent work the crack was charged."""
    residual, frac = S.closure_residual(A_W, A_SPENT, A_U_END, A_K_END)
    assert frac >= -0.02  # about +3.1 % on run A
    # The same closure with the whole crack's area in place of spent would fail (-6.4 %): the board rule's part of the
    # crack costs no knife work, so the closure does not count G_c times it.
    full_residual, full_frac = S.closure_residual(A_W, 3.0 * A_FULL_CRACK_AREA, A_U_END, A_K_END)  # G_c = 3 in run A
    assert full_frac == pytest.approx(-0.064, abs=0.002)


def test_the_ledger_statistics_of_a_small_example_by_hand():
    L = dict.fromkeys(LEDGER_NAMES, 0.0)
    L.update(initiation_area=0.25, spent_init=2.0, spent_prop=1.5, need_steps=4.0, short_steps=2.0, pay_ratio_sum=3.25)
    stats = S.ledger_statistics(L, gc=3.0, knife_area=0.45)
    assert stats["A_init_m2"] == pytest.approx(0.25)
    assert stats["A_prop_m2"] == pytest.approx(0.20)
    assert stats["init_overcharge_J"] == pytest.approx(2.0 - 3.0 * 0.25)  # the need beyond G_c times the area opened
    assert stats["prop_rel"] == pytest.approx(abs(1.5 - 3.0 * 0.20) / 1.5)  # 0.6
    assert stats["short_step_fraction"] == pytest.approx(0.5)  # 2 of 4 steps fell short
    assert stats["mean_pay_ratio"] == pytest.approx(3.25 / 4.0)  # 0.8125


def test_the_propagation_check_is_zero_when_the_spent_work_is_G_c_times_the_area_after_the_first_crack():
    L = dict.fromkeys(LEDGER_NAMES, 0.0)
    L.update(initiation_area=0.1, spent_prop=3.0 * 0.2)
    assert S.ledger_statistics(L, gc=3.0, knife_area=0.3)["prop_rel"] == pytest.approx(0.0)


# ---- the ledger's rules, on the settle kernel with small arrays ------------------------------------------------------

def _settle_sequence(steps, initiated0=0):
    """Runs the settle kernel once per (work, need, knife area) step on cut 0; returns the ledger row (LEDGER_NAMES)."""
    work = wp.array(np.zeros(1, np.float32), dtype=float, device="cpu")
    need = wp.array(np.zeros(1, np.float32), dtype=float, device="cpu")
    initiated = wp.array(np.array([initiated0], np.int32), dtype=int, device="cpu")
    ledger = wp.zeros((2, len(LEDGER_NAMES)), dtype=float, device="cpu")
    area = wp.zeros((2, 3), dtype=float, device="cpu")
    for w, n, a in steps:
        work.assign(np.array([w], np.float32))
        need.assign(np.array([n], np.float32))
        area.assign(np.array([[a, 0.0, 0.0], [0.0, 0.0, 0.0]], np.float32))
        wp.launch(_settle_crack_work, dim=1, inputs=[work, need, initiated, 0, ledger, area], device="cpu")
    return dict(zip(LEDGER_NAMES, ledger.numpy()[0].astype(float).tolist()))


def _expect(**values):
    """The full ledger with the given entries, the rest zero."""
    row = dict.fromkeys(LEDGER_NAMES, 0.0)
    row.update(values)
    return row


def _assert_ledger(got, expect):
    for name in LEDGER_NAMES:
        assert got[name] == pytest.approx(expect[name], abs=1e-6), name  # float32 storage


def test_before_the_first_crack_work_only_accumulates_and_the_step_counts_as_short():
    got = _settle_sequence([(1.0, 2.0, 0.0)])
    _assert_ledger(got, _expect(need_steps=1.0, short_steps=1.0, pay_ratio_sum=0.5, need_last=2.0))


def test_the_first_crack_pays_its_need_and_banks_the_rest_as_discarded_and_initiation_work():
    got = _settle_sequence([(3.0, 2.0, 0.25)])
    _assert_ledger(got, _expect(spent=2.0, discarded=1.0, initiation_work=3.0, spent_init=2.0, need_steps=1.0,
                                pay_ratio_sum=1.0, need_last=2.0, initiation_area=0.25))


def test_after_the_first_crack_work_short_of_the_need_is_spent_and_the_rest_is_shortfall():
    got = _settle_sequence([(3.0, 2.0, 0.25), (1.5, 2.0, 0.4)])
    _assert_ledger(got, _expect(spent=3.5, discarded=1.0, need_shortfall_stepsum=0.5, initiation_work=3.0, spent_init=2.0, spent_prop=1.5,
                                need_steps=2.0, short_steps=1.0, pay_ratio_sum=1.75, need_last=2.0,
                                initiation_area=0.25))


def test_after_the_first_crack_work_beyond_the_need_is_discarded():
    got = _settle_sequence([(3.0, 2.0, 0.25), (3.0, 2.0, 0.4)])
    _assert_ledger(got, _expect(spent=4.0, discarded=2.0, initiation_work=3.0, spent_init=2.0, spent_prop=2.0, need_steps=2.0,
                                pay_ratio_sum=2.0, need_last=2.0, initiation_area=0.25))


def test_work_done_while_nothing_is_to_be_cut_is_discarded_as_idle_and_not_counted_as_a_need_step():
    got = _settle_sequence([(3.0, 2.0, 0.25), (0.7, 0.0, 0.4)])
    _assert_ledger(got, _expect(spent=2.0, discarded=1.0, discarded_idle=0.7, initiation_work=3.0, spent_init=2.0,
                                need_steps=1.0, pay_ratio_sum=1.0, need_last=0.0, initiation_area=0.25))


def test_the_short_step_fraction_and_the_mean_pay_ratio_of_five_steps_by_hand():
    steps = [(1.0, 2.0, 0.0), (3.0, 2.0, 0.25), (1.5, 2.0, 0.4), (3.0, 2.0, 0.5), (0.7, 0.0, 0.5)]
    L = _settle_sequence(steps)
    stats = S.ledger_statistics(L, gc=3.0, knife_area=0.5)
    # Steps with a need: 1 to 4 (4 steps). Short of the need: steps 1 and 3 (2 of 4). Pay ratios: 0.5, 1, 0.75, 1.
    assert L["need_steps"] == pytest.approx(4.0)
    assert L["short_steps"] == pytest.approx(2.0)
    assert stats["short_step_fraction"] == pytest.approx(0.5)
    assert stats["mean_pay_ratio"] == pytest.approx((0.5 + 1.0 + 0.75 + 1.0) / 4.0)  # 0.8125


def test_the_need_of_the_latest_step_is_kept_for_the_pinch_threshold_read():
    assert _settle_sequence([(1.0, 2.5, 0.0)])["need_last"] == pytest.approx(2.5)
    assert _settle_sequence([(1.0, 2.5, 0.0), (0.0, 0.0, 0.0)])["need_last"] == pytest.approx(0.0)  # overwritten


def test_the_area_at_the_first_crack_is_kept_and_later_area_does_not_overwrite_it():
    assert _settle_sequence([(1.0, 2.0, 0.0), (3.0, 2.0, 0.7)])["initiation_area"] == pytest.approx(0.7)
    assert _settle_sequence([(3.0, 2.0, 0.25), (1.5, 2.0, 0.9)])["initiation_area"] == pytest.approx(0.25)


# ---- one short stroke through the small block -----------------------------------------------------------------------

def test_work_by_kind_sums_to_the_knife_work_within_one_percent(stroke):
    origins = np.array(stroke["origins"])
    forces = np.array(stroke["forces"])
    W_knife = -float(np.sum(np.einsum("ij,ij->i", forces[1:], origins[1:] - origins[:-1])))
    W_kind = float(stroke["kind"][-1].sum())
    assert W_knife > 0.0
    assert W_kind == pytest.approx(W_knife, rel=0.01)


def test_the_impulse_by_kind_sums_to_the_impulse_the_forces_report(stroke):
    """The knife's impulse, summed over the kinds, is minus the forces times their intervals (advance's convention)."""
    spans = np.diff(np.concatenate([[0.0], stroke["t"]]))
    expected = -np.sum(np.array(stroke["forces"]) * spans[:, None], axis=0)
    total = stroke["imp"][-1].sum(axis=0)
    assert np.allclose(total, expected, rtol=1e-3, atol=1e-9)
    assert total[2] < 0.0  # the blade is pushed down, so the material's impulse on it is upward


def test_the_side_impulse_is_the_absolute_push_along_e_v_not_the_signed_one(stroke):
    sim = stroke["sim"]
    e_v = np.asarray(sim.knife_pose.axes()[1], float)
    side, signed = stroke["side"][-1], stroke["imp"][-1] @ e_v
    # The crack faces push both ways along e_v, so their side impulse exceeds the net push along it.
    assert side[2] > 1.1 * abs(signed[2])
    assert np.all(side >= np.abs(signed) - 1e-12)


def test_both_the_edge_nose_and_its_belly_take_work(stroke):
    assert stroke["kind"][-1][0] > 0.0
    assert stroke["kind"][-1][1] > 0.0


def test_crack_faces_take_no_work_while_the_knife_presses_intact_material(stroke):
    """While the cut is active but no crack has formed, every contact is intact material: the crack-face kind stays 0."""
    kind = stroke["kind"]
    checked = 0
    for i in range(1, len(kind)):
        pressed = stroke["active"][i] and not stroke["cracked"][i]
        contact = (kind[i][0] + kind[i][1]) > (kind[i - 1][0] + kind[i - 1][1]) + 1e-15
        if pressed and contact:
            checked += 1
            assert kind[i][2] == kind[i - 1][2]
    assert checked >= 1


def test_crack_faces_take_no_knife_work_before_the_cut_begins(stroke):
    before = [k for t, k in zip(stroke["t"], stroke["kind"]) if t < 0.05 - 1e-9]
    assert before, "the stroke should have samples before it begins"
    assert all(k[2] == 0.0 for k in before)
    assert stroke["kind"][-1][2] > 0.0  # and once the crack exists, its faces take work


def test_a_block_cut_through_has_a_crack_area_of_width_times_height_within_five_percent(stroke):
    sim = stroke["sim"]
    assert sim.cut_complete(0)
    assert sim.crack_area(0) == pytest.approx(BLOCK[1] * BLOCK[2], rel=0.05)


def test_the_crack_area_paths_add_up_to_the_crack_area(stroke):
    sim = stroke["sim"]
    paths = sim.crack_area_paths(0)
    assert paths[0] > 0.0
    assert paths.sum() == pytest.approx(sim.crack_area(0), rel=1e-4)


def test_after_the_first_crack_the_spent_work_is_G_c_times_the_knife_area_within_two_percent(stroke):
    sim = stroke["sim"]
    L = dict(zip(LEDGER_NAMES, sim.ledger(0).tolist()))
    stats = S.ledger_statistics(L, FIRM_TOFU.fracture_energy, float(sim.crack_area_paths(0)[0]))
    assert stats["prop_rel"] <= 0.02
    assert L["spent"] == pytest.approx(L["spent_init"] + L["spent_prop"], rel=1e-6)


def test_the_first_crack_over_charge_is_the_need_beyond_what_its_area_paid_for(stroke):
    sim = stroke["sim"]
    L = dict(zip(LEDGER_NAMES, sim.ledger(0).tolist()))
    stats = S.ledger_statistics(L, FIRM_TOFU.fracture_energy, float(sim.crack_area_paths(0)[0]))
    assert L["initiation_area"] > 0.0
    assert stats["init_overcharge_J"] == pytest.approx(L["spent_init"] - FIRM_TOFU.fracture_energy * L["initiation_area"])
    assert L["spent_init"] >= FIRM_TOFU.fracture_energy * L["initiation_area"]  # a charge, not a credit


def test_the_pre_board_state_is_stored_once_at_the_first_approach(stroke):
    sim = stroke["sim"]
    first = stroke["pre_first"]
    assert first is not None
    k = stroke["pre_first_index"]
    threshold = sim.config.dilation * sim.dx + 0.5e-3
    assert first["height"] < threshold
    assert stroke["low"][k] < threshold
    assert all(h >= threshold for h in stroke["low"][:k])
    assert sim.pre_board(0) == first  # not overwritten while the crack grew
    assert sim.crack_area(0) > first["area"]
    assert sim.pre_board_ratio(0) == pytest.approx(
        first["area"] / (first["width_sum"] * (first["H"] - first["height"])))
    assert first["need_at_pinch"] > 0.0  # the need of the step at which the edge reached the pinch threshold


def test_the_pinch_read_takes_the_need_of_the_latest_step_not_the_initiation_area():
    """The pre-board state's need is the latest step's need (the ledger's need_last), read at the pinch threshold."""
    sim = _sim()
    sim.set_knife(KnifePose(np.array([0.016, 0.012, sim.board_z + 0.5e-3]), ACROSS))  # the edge below the threshold
    sim.begin_cut()
    ledger = np.zeros((sim.config.max_cuts, len(LEDGER_NAMES)), np.float32)
    ledger[0, LEDGER_NAMES.index("need_last")] = 7.5
    ledger[0, LEDGER_NAMES.index("initiation_area")] = 0.25
    sim.crack_ledger.assign(ledger)
    sim._track_pre_board()
    assert sim.pre_board(0) is not None
    assert sim.pre_board(0)["need_at_pinch"] == pytest.approx(7.5)


def test_each_cut_keeps_its_own_bins_for_the_tear(stroke):
    sim = stroke["sim"]
    assert np.array_equal(sim.cut_bin_width.numpy()[0], sim.bin_width.numpy())
    assert np.array_equal(sim.cut_bin_top.numpy()[0], sim.bin_top.numpy())


# ---- the energies and the floating particles, on a block at rest --------------------------------------------------

def test_strain_energy_is_psi_times_particle_volume_summed_and_kinetic_energy_is_half_m_v_squared():
    sim = _sim()
    n = sim.n_particles
    sim.psi.assign(np.full(n, 2.0, np.float32))
    sim.v.assign(np.tile(np.array([1.0, 0.0, 0.0], np.float32), (n, 1)))
    assert sim.strain_energy_total() == pytest.approx(2.0 * n * sim.particle_volume, rel=1e-6)
    assert sim.kinetic_energy() == pytest.approx(0.5 * n * sim.particle_mass, rel=1e-6)


def test_floating_particles_are_bottom_layer_ones_lifted_more_than_half_a_spacing():
    sim = _sim()
    h = sim.dx / sim.config.particles_per_axis
    bottom = np.flatnonzero(sim.x0[:, 2] - sim.block_min[2] < h)
    upper = np.flatnonzero(sim.x0[:, 2] - sim.block_min[2] >= h)
    assert len(bottom) > 3 and len(upper) > 0
    x = sim.x.numpy().astype(np.float32)
    x[bottom[:3], 2] += 0.6 * h  # three bottom-layer particles risen 0.6 spacing: counted
    x[bottom[3], 2] += 0.4 * h  # one risen 0.4 spacing: not
    x[upper[0], 2] += 0.6 * h  # one from above the bottom layer: not
    sim.x.assign(x)
    assert sim.floating_count() == 3


def test_a_step_whose_work_exactly_equals_the_need_is_not_short(stroke):
    """Work equal to the need pays it in full: such a step is no shortfall, and its pay ratio is one."""
    L = _settle_sequence([(2.0, 2.0, 0.0)])
    assert L["need_steps"] == pytest.approx(1.0)
    assert L["short_steps"] == pytest.approx(0.0)
    assert L["pay_ratio_sum"] == pytest.approx(1.0)


def test_the_pre_board_work_counts_only_the_work_since_the_cut_began():
    sim = _sim()
    sim.kind_work.assign(np.array([5.0, 0.0, 0.0], np.float32))  # work done before the cut begins
    sim.set_knife(KnifePose(np.array([0.016, 0.012, sim.board_z + 0.5e-3]), ACROSS))
    sim.begin_cut()
    sim._track_pre_board()
    assert sim.pre_board(0) is not None
    assert sim.pre_board(0)["W"] == pytest.approx(0.0, abs=1e-6)  # the 5 J before the cut is not its work


def test_the_descent_window_is_strictly_above_2_mm_and_below_H_minus_2_mm():
    H = 0.016
    depth = [0.0, 0.0015, 0.00201, 0.0139, 0.0141]  # growing; 1.5 mm and 14.1 mm lie outside the window
    rows = [{"depth": d, "fz": 1.0} for d in depth]
    kept = [r["depth"] for r in S.descent_rows(rows, H)]
    assert kept == [0.00201, 0.0139]


def test_the_descent_drops_rows_where_the_depth_does_not_grow():
    depth = [0.0, 0.003, 0.0035, 0.0035, 0.0045, 0.0040]  # the repeated 3.5 mm and the drop to 4.0 mm are not descent
    rows = [{"depth": d, "fz": 1.0} for d in depth]
    kept = [r["depth"] for r in S.descent_rows(rows, 0.016)]
    assert kept == [0.003, 0.0035, 0.0045]


def _pried_half_cut(sim):
    """A half cut levered open with the blade: the scene of tests/test_tofu_cutting.py's half-cut test (the ligament tears)."""
    c = 0.5 * (sim.block_min + sim.block_max)
    top = sim.block_max[2]
    depth = 0.5 * BLOCK[2]
    hover = KnifePose(np.array([c[0], c[1], top + 0.004]), 0.0)  # blade along x
    down = KnifePose(np.array([c[0], c[1], top - depth]), 0.0)
    pried = KnifePose(down.origin - np.array([0.0, 0.003, 0.0]), 0.0, roll=math.radians(8.0))
    moves = [(hover, down, (0.004 + depth) / 0.08, True), (down, pried, 0.3, False), (pried, down, 0.3, False),
             (down, hover, 0.1, False), (hover, hover, 0.3, False)]
    segs, t = [], 0.05
    for a, b, dt, cut in moves:
        segs += KnifeTrajectory.linear(t, t + dt, a, b, begins_cut=cut).segments
        t += dt
    traj = KnifeTrajectory(segments=segs)
    t = sim.time
    while t < traj.duration - 1e-9:
        t = min(t + 0.01, traj.duration)
        sim.advance(t, traj)


def test_a_tear_of_a_pried_half_cut_is_counted_on_the_tear_path():
    sim = _sim()
    _pried_half_cut(sim)
    paths = sim.crack_area_paths(0)
    assert paths[2] > 0.0  # the ligament tore, and the tear took area
    assert paths.sum() == pytest.approx(sim.crack_area(0), rel=1e-4)
