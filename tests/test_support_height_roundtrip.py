# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The round-trip script on a synthetic folder laid out like a recorded demonstration set (no real data is read)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from simfoundry.cutting.support_height import (  # noqa: E402 - imported before the generator, so a missing module fails here
    EPS_M,
    lowest_edge_z,
    support_height,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cutting"))
try:
    import tofu_policy_episodes as G  # noqa: E402 - the generator's own imports need the simulator's packages in some envs
except ImportError as error:
    pytest.skip(f"the generator cannot be imported here: {error}", allow_module_level=True)

import support_height_roundtrip as R  # noqa: E402

TICKS = 40
SPT = 8  # physics steps per tick
BASE_YAW_DEG = 20.0
BASE_POS = [-0.7, -0.06, 0.0]


def _knife(rotvec, translation):
    K = np.eye(4)
    K[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    K[:3, 3] = translation
    return K


K1 = _knife([0.1, 2.1, 0.2], [0.14, 0.0, 0.10])  # the setup's grasp
K2 = K1 @ _knife([0.245, 0.0, 0.0], [0.01, -0.01, 0.025])  # a regrasp: 14 deg about its own x, 2.5 cm along its z


def _base():
    B = np.eye(4)
    B[:3, :3] = Rotation.from_euler("z", BASE_YAW_DEG, degrees=True).as_matrix()
    B[:3, 3] = BASE_POS
    return B


def _base_frame_poses(rng, n):
    """Hand poses in the base frame: knife tilted 20-40 deg, yaw +-15 deg, roll +-5 deg, height 0.012-0.08 m."""
    P = np.tile(np.eye(4), (n, 1, 1))
    angles = np.stack([rng.uniform(-15, 15, n), rng.uniform(20, 40, n), rng.uniform(-5, 5, n)], axis=1)
    P[:, :3, :3] = Rotation.from_euler("ZYX", angles, degrees=True).as_matrix()
    P[:, 0, 3] = 0.3 + rng.uniform(-0.05, 0.05, n)
    P[:, 1, 3] = rng.uniform(-0.05, 0.05, n)
    P[:, 2, 3] = rng.uniform(0.012, 0.08, n)
    return P


def _edit_npz(path, **changes):
    """Rewrites one episode archive with some of its arrays replaced."""
    with np.load(path) as archive:
        data = dict(archive)
    data.update(changes)
    np.savez(path, **data)


def _write_folder(folder, *, n_eps=12, regrasp_from=1, bias_m=None, below=(), json_knife=(), t0=5.0):
    """A folder laid out like a recorded demonstration set: setup.json and ep_XXXXX.npz, some with an ep_XXXXX.json.

    Episodes from ``regrasp_from`` on hold K2; the others hold K1, the setup's. ``bias_m`` adds a constant error to an
    episode's recorded edge heights. Episodes in ``below`` get a block top at the median edge height, so half their poses
    have the edge under the support. Episodes in ``json_knife`` write their knife into their JSON.
    """
    folder.mkdir(parents=True, exist_ok=True)
    base = _base()
    (folder / "setup.json").write_text(json.dumps({
        "scene_changes": {"robot_pos": BASE_POS, "robot_yaw_deg": BASE_YAW_DEG},
        "grasp": {"knife_in_eef": K1.tolist()},
    }))
    rng = np.random.default_rng(0)
    bias_m = bias_m or {}
    for ep in range(n_eps):
        knife = K2 if ep >= regrasp_from else K1
        P = _base_frame_poses(rng, TICKS)
        noise = rng.normal(0.0, 1e-4, TICKS) if ep >= 1 else np.zeros(TICKS)  # episode 0 is exact: the setup fits it
        edge = lowest_edge_z(base @ P, knife) + bias_m.get(ep, 0.0) + noise
        cmd_pos = P[:, :3, 3] + rng.uniform(-0.002, 0.002, (TICKS, 3))  # the command misses the hand by mm
        cmd = np.concatenate([cmd_pos, P[:, :3, 0], P[:, :3, 1]], axis=1)
        E_cmd = base @ np.stack([G.pose9(c) for c in cmd])
        s = float(np.median(edge)) if ep in below else float(edge.min()) - 0.005
        block = np.array([0.5, 0.0, s + 0.015, 0.1, 0.1, 0.1, 0.03], dtype=np.float32)
        state = np.zeros((TICKS, 16), np.float32)
        state[:, 9:12] = P[:, :3, 3]
        state[:, 12:16] = Rotation.from_matrix(P[:, :3, :3]).as_quat()
        steps = np.zeros((TICKS * SPT, 8))
        steps[:, 0] = t0 + 0.1 * np.arange(TICKS * SPT) / SPT
        steps[:, 6] = np.repeat(edge, SPT)
        steps[:, 7] = 0.011
        np.savez(folder / f"ep_{ep:05d}.npz", state=state, cmd=cmd.astype(np.float32),
                 t=t0 + 0.1 * np.arange(TICKS), block=block, cmd_steps=np.repeat(E_cmd, SPT, axis=0).astype(np.float32),
                 steps=steps.astype(np.float32))
        if ep in json_knife:
            (folder / f"ep_{ep:05d}.json").write_text(json.dumps({"knife_in_eef": knife.tolist()}))
    return folder


def _run(folder, episodes, out_name):
    """Runs the script's main on the episodes, returns (exit code, the JSON it wrote)."""
    out = folder / out_name
    code = R.main(["--npz", *[str(folder / f"ep_{e:05d}.npz") for e in episodes], "--out", str(out)])
    return code, json.loads(out.read_text())


def test_the_setup_grasp_explains_its_own_episode(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=3)
    code, out = _run(folder, [0], "one.json")
    demo = out["demos"][0]
    assert demo["knife"] == "setup"
    assert demo["fit_episodes"] is None
    assert demo["resid_max_mm"] < 1e-3
    assert code == 0


def test_a_residual_between_the_setup_tolerance_and_five_mm_is_not_the_setup(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=2, regrasp_from=2, bias_m={1: 0.002})
    _, out = _run(folder, [0, 1], "bias.json")
    assert [d["knife"] for d in out["demos"]] == ["setup", "fitted"]
    assert out["demos"][1]["resid_rms_mm"] < 0.5


def test_later_episodes_share_one_knife_fitted_to_the_first_nine_others(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=12)
    code, out = _run(folder, list(range(12)), "all.json")
    assert [d["knife"] for d in out["demos"]] == ["setup"] + ["fitted"] * 11
    fitted = out["demos"][1:]
    assert len({tuple(d["fit_episodes"]) for d in fitted}) == 1
    assert fitted[0]["fit_episodes"] == [f"ep_{e:05d}.npz" for e in range(1, 10)]
    assert all(d["resid_rms_mm"] < 0.5 for d in fitted)
    assert code == 0


def test_an_episode_json_knife_takes_priority_over_the_fit(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=5, json_knife={3})
    _, out = _run(folder, [0, 3], "json.json")
    assert out["demos"][1]["knife"] == "episode"
    assert out["demos"][1]["resid_rms_mm"] < 0.5


def test_s_is_the_block_top_from_its_centre_and_height(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=2)
    _, out = _run(folder, [0], "s.json")
    block = np.load(folder / "ep_00000.npz")["block"].astype(float)
    assert out["demos"][0]["s_m"] == pytest.approx(block[2] - block[6] / 2, abs=1e-7)


def test_round_trip_is_below_one_micrometre_after_float32_rounding_and_the_rounding_is_applied(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=12)
    code, out = _run(folder, list(range(12)), "rt.json")
    for demo in out["demos"]:
        for part in ("tick_roundtrip", "step_roundtrip"):
            assert demo[part]["err32_um"] < 1.0
            assert demo[part]["err32_um"] > demo[part]["err64_um"]  # rounding u moved it, so the test sees the dtype
    assert code == 0


def test_the_solve_starts_from_a_pose_at_height_zero(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=2)
    _, out = _run(folder, [1], "start.json")
    assert out["demos"][0]["tick_roundtrip"]["start_z_max_m"] == 0.0
    assert out["demos"][0]["step_roundtrip"]["start_z_max_m"] == 0.0


def test_poses_with_the_edge_under_the_support_are_counted_and_fail_the_run(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=3, below={1})
    code_ok, _ = _run(folder, [0, 2], "ok.json")
    code, out = _run(folder, [1], "below.json")
    demo = R.load_demo(folder / "ep_00001.npz")
    expected = int(np.count_nonzero(support_height(demo["E_cmd"], K2, demo["s"]) + EPS_M <= 0))
    assert code_ok == 0
    assert code == 1
    assert 0 < expected < TICKS
    assert out["demos"][0]["tick_roundtrip"]["left_out"] == expected


def test_base_cross_check_matches_each_tick_to_its_own_physics_step(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=2)
    _, out = _run(folder, [1], "cross.json")
    assert out["demos"][0]["cross_check_m"] < 1e-6


def test_accepted_list_gives_evenly_spaced_entries(tmp_path):
    lines = [f"/x/ep_{i:05d}.npz" for i in range(10)]
    listing = tmp_path / "accepted.txt"
    listing.write_text("\n".join(lines) + "\n")
    assert R.select_accepted(listing, 3) == [lines[0], lines[4], lines[9]]


def test_float32_step_times_at_367_seconds_match_their_ticks(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=2, t0=367.6)
    path = folder / "ep_00001.npz"
    with np.load(path) as archive:
        t, steps = archive["t"], archive["steps"][:, 0].astype(float)
    # the premise: at this time the float32 step times are off their float64 ticks by more than the old 1e-5 s
    assert np.abs(steps[::SPT] - t).max() > 1e-5
    code, out = _run(folder, [1], "t367.json")
    assert code == 0
    assert out["demos"][0]["cross_check_m"] < 1e-6


def test_a_tick_within_the_tolerance_of_its_step_is_accepted(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=2, t0=367.6)
    path = folder / "ep_00001.npz"
    with np.load(path) as archive:
        t = archive["t"].copy()
    ulp = float(np.spacing(np.float32(t.max())))
    t[5] += 3 * ulp  # three float32 steps: inside the tolerance of four
    _edit_npz(path, t=t)
    demo = R.load_demo(path)
    assert demo["j"][5] == 5 * SPT


def test_a_tick_half_a_physics_step_off_is_refused(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=2, t0=367.6)
    path = folder / "ep_00001.npz"
    with np.load(path) as archive:
        t = archive["t"].copy()
    t[5] += 0.5 * (0.1 / SPT)  # half of a physics step: 6.25 ms, far outside the tolerance
    _edit_npz(path, t=t)
    with pytest.raises(ValueError, match="no physics step"):
        R.load_demo(path)


def test_matched_steps_must_increase_with_the_ticks(tmp_path):
    folder = _write_folder(tmp_path / "gen", n_eps=2, t0=367.6)
    path = folder / "ep_00001.npz"
    with np.load(path) as archive:
        steps, cmd_steps = archive["steps"][::-1].copy(), archive["cmd_steps"][::-1].copy()
    _edit_npz(path, steps=steps, cmd_steps=cmd_steps)  # every tick still matches within the tolerance, in reverse
    with pytest.raises(ValueError, match="increase"):
        R.load_demo(path)


def test_the_pass_level_is_one_micrometre():
    assert R.ROUNDTRIP_TOL_M == 1e-6


def test_the_exit_code_follows_the_pass_level(tmp_path, monkeypatch):
    folder = _write_folder(tmp_path / "gen", n_eps=2)
    code, out = _run(folder, [0], "level.json")  # the module's own level
    assert code == 0
    # the synthetic errors sit between 1 nm and 1 um, so a level of 1e-9 or 1e-3 would not pass the same way
    assert 1e-9 < out["max_float32_roundtrip_um"] * 1e-6 < 1e-6
    monkeypatch.setattr(R, "ROUNDTRIP_TOL_M", 1e-12)  # a level below the errors
    code_tight, _ = _run(folder, [0], "tight.json")
    assert code_tight == 1
