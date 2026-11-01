# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared reader of recorded demonstrations and its knife choice (simfoundry/cutting/demonstrations.py).

The tests call the module directly. They run on synthetic folders (tests/demo_folder.py), and the environment checks run
in a fresh interpreter, so the module's own imports are what is measured.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import simfoundry.cutting.demonstrations as D
from demo_folder import BASE_POS, BASE_YAW_DEG, K1, K2, SPT, edit_npz, robot_base, write_folder

ROOT = Path(__file__).resolve().parents[1]


def _generator():
    """The generator module, or a skip: it needs the simulator's packages in some environments."""
    sys.path.insert(0, str(ROOT / "scripts/cutting"))
    try:
        import tofu_policy_episodes as G
    except ImportError as error:
        pytest.skip(f"the generator cannot be imported here: {error}")
    return G


def _cross_check(demo):
    return np.abs(demo["E_cmd"][:, :3, 3] - demo["cmd_steps"][demo["j"], :3, 3]).max()


# ---- the reading and the arithmetic it shares with the generator and scipy ---------------------------------------

def test_pose_from_rot6_is_the_generators_pose9_bit_for_bit():
    G = _generator()
    rng = np.random.default_rng(1)
    for a9 in rng.normal(size=(200, 9)):  # non-orthogonal columns on purpose: Gram-Schmidt has work to do
        assert np.array_equal(D.pose_from_rot6(a9), G.pose9(a9))


def test_hand_poses_are_scipys_matrix_from_quaternion():
    # to rounding, not bit for bit: bit-level equality with scipy holds only where this was checked
    rng = np.random.default_rng(2)
    xyzw = rng.normal(size=(500, 4))  # unnormalised on purpose: both sides normalise
    position = rng.uniform(-0.5, 0.5, (500, 3))
    P = D.hand_poses(position, xyzw)
    ref = Rotation.from_quat(xyzw).as_matrix()
    np.testing.assert_allclose(P[:, :3, :3], ref, rtol=0, atol=1e-14)
    assert np.array_equal(P[:, :3, 3], position)
    assert np.array_equal(P[:, 3], np.tile([0.0, 0.0, 0.0, 1.0], (500, 1)))


@pytest.mark.parametrize("yaw_deg", [-12.06, 0.0, 20.0, 143.0])
def test_base_pose_is_scipys_level_rotation_about_z_at_the_recorded_position(yaw_deg):
    setup = {"scene_changes": {"robot_pos": [-0.7835336654815893, -0.06622971215165507, 0.0],
                               "robot_yaw_deg": yaw_deg}}
    B = D.base_pose(setup)
    ref = np.eye(4)
    ref[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    np.testing.assert_allclose(B[:3, :3], ref[:3, :3], rtol=0, atol=1e-14)
    assert B[2, 3] == 0.0
    assert np.array_equal(B[:2, 3], setup["scene_changes"]["robot_pos"][:2])
    assert np.array_equal(B[2, :3], [0.0, 0.0, 1.0])


def test_base_pose_of_the_synthetic_folders_is_the_one_they_are_written_in():
    B = robot_base()
    assert np.array_equal(B[:2, 3], BASE_POS[:2])
    np.testing.assert_allclose(B[:3, :3], Rotation.from_euler("z", BASE_YAW_DEG, degrees=True).as_matrix(), atol=1e-14)


# ---- what the module imports, checked in a fresh interpreter -----------------------------------------------------

def _fresh(code, **env):
    full_env = {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", **env}
    return subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=full_env, cwd=ROOT)


def test_importing_the_module_loads_no_scipy_torch_warp_omnigibson_or_generator():
    code = ("import sys; import simfoundry.cutting.demonstrations; "
            "print([m for m in ('scipy', 'torch', 'warp', 'omnigibson', 'tofu_policy_episodes') if m in sys.modules])")
    result = _fresh(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_setup_and_episode_knives_work_without_scipy_and_only_the_fit_needs_it(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=3, json_knife={2})
    code = f"""
import sys
sys.modules["scipy"] = None  # any import of scipy now fails
import simfoundry.cutting.demonstrations as D
setup_demo = D.load_demo("{folder}/ep_00000.npz")
print(D.choose_knife(setup_demo, 9, {{}})[1])
episode_demo = D.load_demo("{folder}/ep_00002.npz")
print(D.choose_knife(episode_demo, 9, {{}})[1])
fitted_demo = D.load_demo("{folder}/ep_00001.npz")
try:
    D.choose_knife(fitted_demo, 9, {{}})
    print("no error")
except ImportError:
    print("ImportError")
"""
    result = _fresh(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["setup", "episode", "ImportError"]


# ---- the reading and the knife choice, called directly (moved from the round-trip script's tests) ----------------

def test_the_setup_grasp_explains_its_own_episode(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=3)
    demo = D.load_demo(folder / "ep_00000.npz")
    knife, source, fit = D.choose_knife(demo, 9, {})
    assert source == "setup"
    assert fit is None
    assert np.abs(D.edge_residual_mm(demo, knife)).max() < 1e-3


def test_a_residual_between_the_setup_tolerance_and_five_mm_is_not_the_setup(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=2, regrasp_from=2, bias_m={1: 0.002})
    cache = {}
    sources, residuals = [], []
    for ep in (0, 1):
        demo = D.load_demo(folder / f"ep_{ep:05d}.npz")
        knife, source, _ = D.choose_knife(demo, 9, cache)
        sources.append(source)
        residuals.append(D.rms(D.edge_residual_mm(demo, knife)))
    assert sources == ["setup", "fitted"]
    assert residuals[1] < 0.5


def test_later_episodes_share_one_knife_fitted_to_the_first_nine_others(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=12)
    cache = {}
    sources = []
    for ep in range(12):
        demo = D.load_demo(folder / f"ep_{ep:05d}.npz")
        knife, source, fit = D.choose_knife(demo, 9, cache)
        sources.append(source)
        if source == "fitted":
            assert D.rms(D.edge_residual_mm(demo, knife)) < 0.5
    assert sources == ["setup"] + ["fitted"] * 11
    assert cache[folder]["episodes"] == [f"ep_{e:05d}.npz" for e in range(1, 10)]


def test_an_episode_json_knife_takes_priority_over_the_fit(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=5, json_knife={3})
    demo = D.load_demo(folder / "ep_00003.npz")
    knife, source, fit = D.choose_knife(demo, 9, {})
    assert source == "episode"
    assert fit is None
    np.testing.assert_allclose(knife, K2, rtol=0, atol=1e-12)
    assert D.rms(D.edge_residual_mm(demo, knife)) < 0.5


def test_an_episode_json_knife_wins_even_when_the_setup_explains_the_episode_exactly(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=1)  # episode 0 holds the setup's knife exactly
    (folder / "ep_00000.json").write_text(json.dumps({"knife_in_eef": K1.tolist()}))
    demo = D.load_demo(folder / "ep_00000.npz")
    knife, source, fit = D.choose_knife(demo, 9, {})
    assert source == "episode"
    np.testing.assert_allclose(knife, K1, rtol=0, atol=1e-12)


def test_s_is_the_block_top_from_its_centre_and_height(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=2)
    block = np.load(folder / "ep_00000.npz")["block"].astype(float)
    assert D.load_demo(folder / "ep_00000.npz")["s"] == pytest.approx(block[2] - block[6] / 2, abs=1e-7)


def test_base_cross_check_matches_each_tick_to_its_own_physics_step(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=2)
    assert _cross_check(D.load_demo(folder / "ep_00001.npz")) < 1e-6


def test_float32_step_times_at_367_seconds_match_their_ticks(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=2, t0=367.6)
    path = folder / "ep_00001.npz"
    with np.load(path) as archive:
        t, steps = archive["t"], archive["steps"][:, 0].astype(float)
    # the premise: at this time the float32 step times are off their float64 ticks by more than the old 1e-5 s
    assert np.abs(steps[::SPT] - t).max() > 1e-5
    demo = D.load_demo(path)
    assert _cross_check(demo) < 1e-6


def test_a_tick_within_the_tolerance_of_its_step_is_accepted(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=2, t0=367.6)
    path = folder / "ep_00001.npz"
    with np.load(path) as archive:
        t = archive["t"].copy()
    ulp = float(np.spacing(np.float32(t.max())))
    t[5] += 3 * ulp  # three float32 steps: inside the tolerance of four
    edit_npz(path, t=t)
    assert D.load_demo(path)["j"][5] == 5 * SPT


def test_a_tick_half_a_physics_step_off_is_refused(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=2, t0=367.6)
    path = folder / "ep_00001.npz"
    with np.load(path) as archive:
        t = archive["t"].copy()
    t[5] += 0.5 * (0.1 / SPT)  # half of a physics step: 6.25 ms, far outside the tolerance
    edit_npz(path, t=t)
    with pytest.raises(ValueError, match="no physics step"):
        D.load_demo(path)


def test_matched_steps_must_increase_with_the_ticks(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=2, t0=367.6)
    path = folder / "ep_00001.npz"
    with np.load(path) as archive:
        steps, cmd_steps = archive["steps"][::-1].copy(), archive["cmd_steps"][::-1].copy()
    edit_npz(path, steps=steps, cmd_steps=cmd_steps)  # every tick still matches within the tolerance, in reverse
    with pytest.raises(ValueError, match="increase"):
        D.load_demo(path)


# ---- new checks ------------------------------------------------------------------------------------------------

def test_a_residual_of_a_hundredth_of_a_millimetre_is_still_the_setup(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=2, bias_m={0: 1e-5})  # 0.01 mm: under the 0.05 mm setup tolerance
    demo = D.load_demo(folder / "ep_00000.npz")
    assert D.choose_knife(demo, 9, {})[1] == "setup"


def test_fitted_knife_refuses_a_folder_the_setup_grasp_explains_entirely(tmp_path):
    # one exact episode: the synthetic folders add noise to episodes after the first, which the setup would not explain
    folder = write_folder(tmp_path / "gen", n_eps=1)
    with pytest.raises(ValueError, match="no knife to fit"):
        D.fitted_knife(folder, K1, 9)


def test_fitted_knife_uses_the_first_fit_episodes_it_is_asked_for(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=12)
    fit = D.fitted_knife(folder, K1, 3)
    assert fit["episodes"] == [f"ep_{e:05d}.npz" for e in range(1, 4)]
    assert D.rms(D.edge_residual_mm(D.load_demo(folder / "ep_00011.npz"), fit["knife"])) < 0.5
