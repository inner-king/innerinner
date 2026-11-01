# SPDX-License-Identifier: Apache-2.0
"""The support-height module (simfoundry/cutting/support_height.py): values, inverses, the fit and its wiring. CPU only."""

import math

import numpy as np
import pytest
import scipy.optimize
from scipy.spatial.transform import Rotation

import simfoundry.cutting.support_height as sh
from simfoundry.cutting.cosim import pose_from_axes
from simfoundry.cutting.knife import CHEF_KNIFE, KnifeGeometry
from simfoundry.cutting.support_height import (
    EDGE_SAMPLES,
    EPS_M,
    edge_points,
    fit_knife_in_eef,
    height_from_log,
    hand_z_for_support_height,
    hand_z_from_log_height,
    log_height,
    lowest_edge_z,
    state_with_support,
    support_height,
)

TILTED_KNIFE = KnifeGeometry(half_length=0.12, height=0.05, tip_rise=0.03, belly_length=0.05, spine_taper=0.06)
GEOMETRIES = [KnifeGeometry(), CHEF_KNIFE, TILTED_KNIFE]


def _transforms(rng, n, shift=0.5):
    """Random rigid transforms: arbitrary rotations (not only tilts), translations within +-shift."""
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(rng.normal(size=(n, 3))).as_matrix()
    T[:, :3, 3] = rng.uniform(-shift, shift, size=(n, 3))
    return T


def _rot_y(angle):
    return Rotation.from_rotvec([0.0, angle, 0.0]).as_matrix()


def _dataset_like_hand_poses(rng, n):
    """Hand poses like the demonstrations: the blade tilted 20-40 deg, yaw +-15, roll +-5, z 0.012-0.08 m."""
    E = np.tile(np.eye(4), (n, 1, 1))
    tilt = np.radians(rng.uniform(20.0, 40.0, n))
    yaw = np.radians(rng.uniform(-15.0, 15.0, n))
    roll = np.radians(rng.uniform(-5.0, 5.0, n))
    for i in range(n):
        E[i, :3, :3] = (
            Rotation.from_rotvec([0, 0, yaw[i]]).as_matrix()
            @ _rot_y(tilt[i])
            @ Rotation.from_rotvec([roll[i], 0, 0]).as_matrix()
        )
    E[:, :3, 3] = np.column_stack([rng.uniform(-0.1, 0.1, n), rng.uniform(-0.1, 0.1, n), rng.uniform(0.012, 0.08, n)])
    return E


def _knife(rotvec, translation):
    K = np.eye(4)
    K[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    K[:3, 3] = translation
    return K


# ---- 1. the edge's shape: the same lowest point as KnifeGeometry, for arbitrary poses ----------------------------

@pytest.mark.parametrize("geometry, samples", [(g, 100) for g in GEOMETRIES] + [(g, 400) for g in GEOMETRIES])
def test_lowest_edge_z_matches_knife_geometry(geometry, samples):
    rng = np.random.default_rng(11)
    eefs = _transforms(rng, 400)
    knives = _transforms(rng, 400)
    K = eefs @ knives
    keep = np.flatnonzero(np.abs(K[:, 2, 0]) < 0.9)[:200]  # pose_from_axes has no yaw for a vertical edge
    assert len(keep) == 200
    batch = lowest_edge_z(eefs[keep], knives[keep], geometry, samples)
    for row, i in enumerate(keep):
        Ki = K[i]
        pose = pose_from_axes(Ki[:3, 3], Ki[:3, 0], Ki[:3, 1], Ki[:3, 2])
        reference = geometry.lowest_edge_point(pose, samples)[2]
        assert abs(lowest_edge_z(eefs[i], knives[i], geometry, samples) - reference) < 1e-12
        assert abs(batch[row] - reference) < 1e-12


def test_edge_points_are_knife_frame_edge():
    pts = edge_points(CHEF_KNIFE, 100)
    assert pts.shape == (3, 100)
    np.testing.assert_array_equal(pts[1], 0.0)
    np.testing.assert_allclose(pts[2], CHEF_KNIFE.edge_height(pts[0]), rtol=0, atol=0)
    assert pts[0, 0] == pytest.approx(-CHEF_KNIFE.half_length)
    assert pts[0, -1] == pytest.approx(CHEF_KNIFE.half_length)


# ---- 2. a hand-computed oracle (a plain loop over the edge; edge_points is not used) -----------------------------

def _oracle_support_height(z0, theta, s, samples=100):
    """Lowest edge point of a knife tilted by theta about its own normal, found by looping over the edge.

    The hand is rotated about y by theta, so the edge point at u sits at z0 - u sin(theta) + edge_height(u) cos(theta).
    Returns (h, index of the lowest point).
    """
    lowest, where = math.inf, -1
    for i, u in enumerate(np.linspace(-CHEF_KNIFE.half_length, CHEF_KNIFE.half_length, samples)):
        z = z0 - u * math.sin(theta) + float(CHEF_KNIFE.edge_height(u)) * math.cos(theta)
        if z < lowest:
            lowest, where = z, i
    return lowest - s, where


def test_identity_hand_gives_hand_computed_height():
    eef = np.eye(4)
    eef[2, 3] = 0.05
    assert support_height(eef, np.eye(4), 0.011, CHEF_KNIFE) == pytest.approx(0.039, abs=1e-12)


@pytest.mark.parametrize("theta_deg, lowest_at", [(30.0, "tip"), (-30.0, "heel")])
def test_tilted_hand_matches_loop_oracle(theta_deg, lowest_at):
    theta = math.radians(theta_deg)
    eef = np.eye(4)
    eef[:3, :3] = _rot_y(theta)
    eef[2, 3] = 0.05
    expected, where = _oracle_support_height(0.05, theta, 0.011)
    assert where == (EDGE_SAMPLES - 1 if lowest_at == "tip" else 0)
    assert support_height(eef, np.eye(4), 0.011, CHEF_KNIFE) == pytest.approx(expected, abs=1e-12)


# ---- 3. slope one in the hand's z, and independence from x, y and the support height -----------------------------

def test_moving_the_hand_up_moves_the_height_one_for_one():
    rng = np.random.default_rng(3)
    eefs = _transforms(rng, 50, shift=0.3)
    knife = _transforms(rng, 1)[0]
    base = support_height(eefs, knife, 0.02)
    for d in (0.0123, -0.04):
        shifted = eefs.copy()
        shifted[:, 2, 3] += d
        np.testing.assert_allclose(support_height(shifted, knife, 0.02), base + d, rtol=0, atol=1e-12)


def test_height_ignores_x_and_y_and_falls_with_support_z():
    rng = np.random.default_rng(4)
    eefs = _transforms(rng, 50, shift=0.3)
    knife = _transforms(rng, 1)[0]
    base = support_height(eefs, knife, 0.02)
    moved = eefs.copy()
    moved[:, 0, 3] += 0.21
    moved[:, 1, 3] -= 0.37
    np.testing.assert_allclose(support_height(moved, knife, 0.02), base, rtol=0, atol=1e-12)
    np.testing.assert_allclose(support_height(eefs, knife, 0.02 + 0.007), base - 0.007, rtol=0, atol=1e-12)


# ---- 4. the inverse --------------------------------------------------------------------------------------------

def test_hand_z_for_support_height_gives_the_target():
    rng = np.random.default_rng(5)
    eefs = _transforms(rng, 400, shift=0.4)
    knife = _transforms(rng, 1)[0]
    targets = rng.uniform(-0.5e-3, 0.3, size=400)
    z = hand_z_for_support_height(eefs, knife, 0.013, targets)
    moved = eefs.copy()
    moved[..., 2, 3] = z
    np.testing.assert_allclose(support_height(moved, knife, 0.013), targets, rtol=0, atol=1e-12)


def test_batch_matches_single_poses():
    rng = np.random.default_rng(6)
    eefs = _transforms(rng, 30, shift=0.4)
    knife = _transforms(rng, 1)[0]
    targets = rng.uniform(0.0, 0.2, size=30)
    batch = hand_z_for_support_height(eefs, knife, 0.011, targets)
    singles = np.array([hand_z_for_support_height(eefs[i], knife, 0.011, targets[i]) for i in range(30)])
    np.testing.assert_allclose(batch, singles, rtol=0, atol=1e-12)


def test_log_round_trip_restores_the_hand_z():
    rng = np.random.default_rng(7)
    eefs = _transforms(rng, 1000, shift=0.5)
    knife = _transforms(rng, 1)[0]
    # Give each hand a height in (-EPS_M, 0.3 m) by moving it, then check that the log path returns the same z.
    targets = rng.uniform(-EPS_M + 1e-6, 0.3, size=1000)
    eefs[:, 2, 3] += targets - support_height(eefs, knife, 0.009)
    u = log_height(support_height(eefs, knife, 0.009))
    z = hand_z_from_log_height(eefs, knife, 0.009, u)
    np.testing.assert_allclose(z, eefs[:, 2, 3], rtol=0, atol=1e-12)


def test_float32_log_value_restores_the_hand_z_within_a_micrometre():
    rng = np.random.default_rng(8)
    eefs = _transforms(rng, 1000, shift=0.5)
    knife = _transforms(rng, 1)[0]
    targets = rng.uniform(-EPS_M + 1e-6, 0.3, size=1000)
    eefs[:, 2, 3] += targets - support_height(eefs, knife, 0.009)
    u32 = log_height(support_height(eefs, knife, 0.009)).astype(np.float32)
    z = hand_z_from_log_height(eefs, knife, 0.009, u32)
    assert np.max(np.abs(z - eefs[:, 2, 3])) < 1e-6


# ---- 5. log_height -------------------------------------------------------------------------------------------

def test_log_height_rejects_heights_at_or_below_minus_eps():
    with pytest.raises(ValueError, match=r"^1 height\(s\)"):
        log_height(-EPS_M)
    with pytest.raises(ValueError, match=r"2 height\(s\).*-0\.02 m"):
        log_height(np.array([0.1, -0.0042, -0.02]))


def test_log_height_accepts_just_above_minus_eps():
    assert np.isfinite(log_height(-EPS_M + 1e-9))


def test_log_height_is_increasing_and_inverts():
    hs = np.linspace(-0.9 * EPS_M, 0.3, 200)
    u = log_height(hs)
    assert np.all(np.diff(u) > 0)
    np.testing.assert_allclose(height_from_log(u), hs, rtol=0, atol=1e-12)
    other = 2e-3
    hs2 = np.linspace(-0.9 * other, 0.3, 200)
    np.testing.assert_allclose(height_from_log(log_height(hs2, other), other), hs2, rtol=0, atol=1e-12)


# ---- 6. float32 input -------------------------------------------------------------------------------------

def test_float32_input_gives_float64_results_equal_to_float64_input():
    rng = np.random.default_rng(9)
    eefs = _transforms(rng, 40, shift=0.4).astype(np.float32)
    knife = _transforms(rng, 1)[0].astype(np.float32)
    z32 = lowest_edge_z(eefs, knife)
    assert np.asarray(z32).dtype == np.float64
    np.testing.assert_array_equal(z32, lowest_edge_z(eefs.astype(np.float64), knife.astype(np.float64)))
    h32 = hand_z_for_support_height(eefs, knife, 0.01, 0.02)
    assert np.asarray(h32).dtype == np.float64
    np.testing.assert_array_equal(h32, hand_z_for_support_height(eefs.astype(np.float64), knife.astype(np.float64), 0.01, 0.02))


# ---- 7. state_with_support ---------------------------------------------------------------------------------

def test_state_with_support_appends_the_height_in_float32():
    rng = np.random.default_rng(10)
    state = rng.normal(size=16)
    before = state.copy()
    out = state_with_support(state, 0.0137)
    assert out.dtype == np.float32
    assert out.shape == (17,)
    assert out[-1] == np.float32(0.0137)
    np.testing.assert_array_equal(out[:-1], state.astype(np.float32))
    np.testing.assert_array_equal(state, before)


def test_state_with_support_rejects_a_2d_state():
    with pytest.raises(ValueError):
        state_with_support(np.zeros((2, 16)), 0.01)


# ---- 8. the fit ---------------------------------------------------------------------------------------------

TRUE_KNIFE = _knife(np.radians(30.0) * np.array([1.0, 2.0, 2.0]) / 3.0, [0.17, -0.014, 0.015])


def _synthetic_problem(seed=0, n=300):
    eefs = _dataset_like_hand_poses(np.random.default_rng(seed), n)
    edge_z = lowest_edge_z(eefs, TRUE_KNIFE)
    return eefs, edge_z


def _start_25_degrees_off():
    K = np.eye(4)
    K[:3, :3] = Rotation.from_rotvec(np.radians(25.0) * np.array([0.3, -0.5, 0.8])).as_matrix() @ TRUE_KNIFE[:3, :3]
    K[:3, 3] = TRUE_KNIFE[:3, 3] + [0.03, 0.0, 0.0]
    return K


def test_fit_recovers_a_synthetic_knife_and_predicts_new_poses():
    eefs, edge_z = _synthetic_problem()
    K, rms = fit_knife_in_eef(eefs, edge_z, _start_25_degrees_off())
    assert rms < 5e-6
    new = _dataset_like_hand_poses(np.random.default_rng(1), 300)
    err = lowest_edge_z(new, K) - lowest_edge_z(new, TRUE_KNIFE)
    assert np.max(np.abs(err)) < 20e-6


def test_fit_is_repeatable_for_the_same_seed():
    eefs, edge_z = _synthetic_problem()
    rng = np.random.default_rng(12)
    noisy = edge_z + rng.normal(0.0, 0.15e-3, size=edge_z.shape)
    K1, _ = fit_knife_in_eef(eefs, noisy, _start_25_degrees_off(), seed=4)
    K2, _ = fit_knife_in_eef(eefs, noisy, _start_25_degrees_off(), seed=4)
    assert np.array_equal(K1, K2)


def test_fit_with_edge_noise_predicts_within_0_3_mm():
    eefs, edge_z = _synthetic_problem()
    noisy = edge_z + np.random.default_rng(13).normal(0.0, 0.15e-3, size=edge_z.shape)
    K, _ = fit_knife_in_eef(eefs, noisy, _start_25_degrees_off())
    new = _dataset_like_hand_poses(np.random.default_rng(2), 300)
    err = lowest_edge_z(new, K) - lowest_edge_z(new, TRUE_KNIFE)
    assert np.std(err) < 0.3e-3


def _start_that_needs_restarts():
    """A start 81 deg off that one local fit cannot leave (seed 125 of the explorations; restarts=1 misses)."""
    r = np.random.default_rng(125)
    axis = r.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(r.uniform(60.0, 179.0))
    K = np.eye(4)
    K[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix() @ TRUE_KNIFE[:3, :3]
    K[:3, 3] = TRUE_KNIFE[:3, 3] + r.normal(size=3) * 0.03
    return K


def test_restarts_find_the_minimum_a_single_start_misses():
    eefs, edge_z = _synthetic_problem()
    start = _start_that_needs_restarts()
    _, one = fit_knife_in_eef(eefs, edge_z, start, restarts=1)
    _, twelve = fit_knife_in_eef(eefs, edge_z, start, restarts=12)
    assert one > 1e-5
    assert twelve < 1e-6


def test_fit_starts_options_and_rms_follow_the_documented_rules(monkeypatch):
    eefs, edge_z = _synthetic_problem(n=120)
    k0 = _start_that_needs_restarts()
    real = scipy.optimize.least_squares
    starts, options, results = [], [], []

    def spy(fun, x0, **kwargs):
        starts.append(np.array(x0, copy=True))
        options.append(kwargs)
        result = real(fun, x0, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(scipy.optimize, "least_squares", spy)
    # With seed 2 the last of the six starts is stuck (rms 1e-4) while another reaches ~1e-17, so
    # taking the last start instead of the cheapest one fails here.
    seed, restarts = 2, 6
    K, rms = fit_knife_in_eef(eefs, edge_z, k0, restarts=restarts, seed=seed)

    assert len(starts) == restarts
    # The first start is exactly k0: its rotation vector, then its translation.
    np.testing.assert_allclose(starts[0][:3], Rotation.from_matrix(k0[:3, :3]).as_rotvec(), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(starts[0][3:], k0[:3, 3])
    # The others come from default_rng(seed): a rotation perturbation, then a translation perturbation.
    rng = np.random.default_rng(seed)
    x_start = starts[0]
    for k in range(1, restarts):
        expected_rot = (Rotation.from_rotvec(rng.normal(size=3) * 0.5) * Rotation.from_rotvec(x_start[:3])).as_rotvec()
        expected_t = k0[:3, 3] + rng.normal(size=3) * 0.03
        np.testing.assert_allclose(starts[k][:3], expected_rot, rtol=0, atol=1e-15)
        np.testing.assert_allclose(starts[k][3:], expected_t, rtol=0, atol=1e-15)
    for kw in options:
        np.testing.assert_allclose(np.asarray(kw["x_scale"]), [0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        assert kw["xtol"] == 1e-12 and kw["ftol"] == 1e-12
    # The start with the smallest cost wins, and rms is the root-mean-square of its residuals.
    best = min(results, key=lambda r: r.cost)
    assert rms == pytest.approx(float(np.sqrt(np.mean(best.fun**2))), rel=1e-12, abs=1e-18)
    np.testing.assert_array_equal(K[:3, 3], best.x[3:])
    independent = lowest_edge_z(eefs, K) - edge_z
    assert rms == pytest.approx(float(np.sqrt(np.mean(independent**2))), rel=1e-9, abs=1e-18)


def test_fit_rejects_bad_shapes_and_no_starts():
    eefs, edge_z = _synthetic_problem(n=10)
    with pytest.raises(ValueError):
        fit_knife_in_eef(eefs, edge_z[:, None], np.eye(4))
    with pytest.raises(ValueError):
        fit_knife_in_eef(eefs[:, :3], edge_z, np.eye(4))
    with pytest.raises(ValueError):
        fit_knife_in_eef(eefs, edge_z, np.eye(4), restarts=0)


# ---- 9. the wiring: which function calls which -------------------------------------------------------------

def test_hand_z_for_support_height_goes_through_support_height(monkeypatch):
    calls = []

    def stub(*args, **kwargs):
        calls.append(args)
        return 0.02

    monkeypatch.setattr(sh, "support_height", stub)
    eef = np.eye(4)
    eef[2, 3] = 0.3
    z = hand_z_for_support_height(eef, np.eye(4), 0.01, 0.05)
    assert z == pytest.approx(0.3 + 0.05 - 0.02)
    assert len(calls) == 1


def test_support_height_goes_through_lowest_edge_z(monkeypatch):
    monkeypatch.setattr(sh, "lowest_edge_z", lambda *args, **kwargs: 0.5)
    assert support_height(np.eye(4), np.eye(4), 0.2) == pytest.approx(0.3)


def test_hand_z_from_log_height_uses_height_from_log(monkeypatch):
    seen = []

    def spy(eef, knife, support_z, height, geometry, samples):
        seen.append(height)
        return 0.0

    monkeypatch.setattr(sh, "hand_z_for_support_height", spy)
    hand_z_from_log_height(np.eye(4), np.eye(4), 0.01, np.log(0.25 + 2e-3), eps=2e-3)
    assert seen[0] == pytest.approx(0.25, abs=1e-15)


def test_samples_and_geometry_reach_the_edge():
    # Tilted by 0.5 rad, the lowest point sits on the belly, not at an end, so 100 and 400 samples differ.
    eef = np.eye(4)
    eef[:3, :3] = _rot_y(0.5)
    eef[2, 3] = 0.05
    fine = lowest_edge_z(eef, np.eye(4), TILTED_KNIFE, 400)
    coarse = lowest_edge_z(eef, np.eye(4), TILTED_KNIFE, 100)
    assert fine != coarse
    assert support_height(eef, np.eye(4), 0.01, TILTED_KNIFE, 400) == fine - 0.01
    assert support_height(eef, np.eye(4), 0.01, TILTED_KNIFE, 100) == coarse - 0.01


# ---- 10. constants ----------------------------------------------------------------------------------------

def test_constants_are_fixed():
    assert EPS_M == 1.0e-3
    assert EDGE_SAMPLES == 100


# ---- 11. samples and geometry reach every use of the edge -------------------------------------------------

def _tilted_pose(z=0.1):
    """Tilted 0.5 rad: 400, 100 and 50 samples read different heights here, and CHEF_KNIFE a different one again."""
    eef = np.eye(4)
    eef[:3, :3] = _rot_y(0.5)
    eef[:3, 3] = [0.1, -0.05, z]
    return eef


def test_the_inverse_lands_on_its_target_at_the_samples_it_was_given():
    eef, K, support_z, target = _tilted_pose(), _knife([0.2, -0.1, 0.3], [0.1, 0.0, 0.02]), 0.01, 0.02
    for samples in (400, 50):
        # precondition: at this sampling the height differs from the default 100, so a wrong sampling would show
        assert abs(support_height(eef, K, support_z, TILTED_KNIFE, samples)
                   - support_height(eef, K, support_z, TILTED_KNIFE)) > 1e-7
        z = hand_z_for_support_height(eef, K, support_z, target, TILTED_KNIFE, samples)
        moved = eef.copy()
        moved[2, 3] = z
        assert support_height(moved, K, support_z, TILTED_KNIFE, samples) == pytest.approx(target, abs=1e-12)


def test_the_inverse_uses_the_blade_it_was_given():
    eef, K, support_z, target = _tilted_pose(), _knife([0.2, -0.1, 0.3], [0.1, 0.0, 0.02]), 0.01, 0.02
    z = hand_z_for_support_height(eef, K, support_z, target, TILTED_KNIFE, 400)
    moved = eef.copy()
    moved[2, 3] = z
    assert support_height(moved, K, support_z, TILTED_KNIFE, 400) == pytest.approx(target, abs=1e-12)
    # precondition: the blades differ here by about 9 mm, so reading the wrong one would miss by that much
    assert abs(support_height(eef, K, support_z, TILTED_KNIFE, 400)
               - support_height(eef, K, support_z, CHEF_KNIFE, 400)) > 1e-3


def test_the_log_round_trip_uses_the_blade_and_samples_it_was_given():
    eef, K, support_z = _tilted_pose(), _knife([0.2, -0.1, 0.3], [0.1, 0.0, 0.02]), 0.01
    for samples in (400, 50):
        u = log_height(support_height(eef, K, support_z, TILTED_KNIFE, samples))
        lowered = eef.copy()
        lowered[2, 3] = 0.0  # the solve starts from height zero, so the answer cannot be the input's own
        z = hand_z_from_log_height(lowered, K, support_z, u, geometry=TILTED_KNIFE, samples=samples)
        assert z == pytest.approx(eef[2, 3], abs=1e-12)


def test_the_fit_on_the_tilted_knife_at_400_samples_is_exact_and_the_defaults_are_not():
    eefs = _dataset_like_hand_poses(np.random.default_rng(0), 300)
    edge_z = lowest_edge_z(eefs, TRUE_KNIFE, TILTED_KNIFE, 400)
    k0 = TRUE_KNIFE @ _knife([0.15, -0.1, 0.05], [0.01, -0.005, 0.004])
    K, rms = fit_knife_in_eef(eefs, edge_z, k0, geometry=TILTED_KNIFE, samples=400)
    # measured: 1.5e-17 m here. The same data fitted with 100 samples leaves 2.4e-6 m, so a 5e-6 limit would miss it
    assert rms < 1e-9
    assert np.abs(lowest_edge_z(eefs, K, TILTED_KNIFE, 400) - edge_z).max() < 1e-9
    # measured: 8.7e-5 m with the defaults (CHEF_KNIFE and 100 samples), so the wrong blade is far off too
    _, default_rms = fit_knife_in_eef(eefs, edge_z, k0)
    assert default_rms > 1e-5
