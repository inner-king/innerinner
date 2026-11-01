# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The round-trip script on synthetic folders (no real data is read).

The reading of a demonstration and the knife choice are tested in tests/test_demonstrations.py, on the module they live
in; this file tests what the script does with them: its exit code, its errors, its counts and its file of results.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from demo_folder import K2, TICKS, write_folder
from simfoundry.cutting.demonstrations import load_demo
from simfoundry.cutting.support_height import EPS_M, support_height

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cutting"))
import support_height_roundtrip as R  # noqa: E402


def _run(folder, episodes, out_name):
    """Runs the script's main on the episodes, returns (exit code, the JSON it wrote)."""
    out = folder / out_name
    code = R.main(["--npz", *[str(folder / f"ep_{e:05d}.npz") for e in episodes], "--out", str(out)])
    return code, json.loads(out.read_text())


def test_round_trip_is_below_one_micrometre_after_float32_rounding_and_the_rounding_is_applied(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=12)
    code, out = _run(folder, list(range(12)), "rt.json")
    for demo in out["demos"]:
        for part in ("tick_roundtrip", "step_roundtrip"):
            assert demo[part]["err32_um"] < 1.0
            assert demo[part]["err32_um"] > demo[part]["err64_um"]  # rounding u moved it, so the test sees the dtype
    assert code == 0


def test_the_solve_starts_from_a_pose_at_height_zero(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=2)
    _, out = _run(folder, [1], "start.json")
    assert out["demos"][0]["tick_roundtrip"]["start_z_max_m"] == 0.0
    assert out["demos"][0]["step_roundtrip"]["start_z_max_m"] == 0.0


def test_poses_with_the_edge_under_the_support_are_counted_and_fail_the_run(tmp_path):
    folder = write_folder(tmp_path / "gen", n_eps=3, below={1})
    code_ok, _ = _run(folder, [0, 2], "ok.json")
    code, out = _run(folder, [1], "below.json")
    demo = load_demo(folder / "ep_00001.npz")
    expected = int(np.count_nonzero(support_height(demo["E_cmd"], K2, demo["s"]) + EPS_M <= 0))
    assert code_ok == 0
    assert code == 1
    assert 0 < expected < TICKS
    assert out["demos"][0]["tick_roundtrip"]["left_out"] == expected


def test_accepted_list_gives_evenly_spaced_entries(tmp_path):
    lines = [f"/x/ep_{i:05d}.npz" for i in range(10)]
    listing = tmp_path / "accepted.txt"
    listing.write_text("\n".join(lines) + "\n")
    assert R.select_accepted(listing, 3) == [lines[0], lines[4], lines[9]]


def test_the_pass_level_is_one_micrometre():
    assert R.ROUNDTRIP_TOL_M == 1e-6


def test_the_exit_code_follows_the_pass_level(tmp_path, monkeypatch):
    folder = write_folder(tmp_path / "gen", n_eps=2)
    code, out = _run(folder, [0], "level.json")  # the module's own level
    assert code == 0
    # the synthetic errors sit between 1 nm and 1 um, so a level of 1e-9 or 1e-3 would not pass the same way
    assert 1e-9 < out["max_float32_roundtrip_um"] * 1e-6 < 1e-6
    monkeypatch.setattr(R, "ROUNDTRIP_TOL_M", 1e-12)  # a level below the errors
    code_tight, _ = _run(folder, [0], "tight.json")
    assert code_tight == 1


def test_the_script_takes_its_reading_from_the_shared_module(tmp_path):
    # the script must not keep its own copy of the reader: its load_demo is the module's
    assert R.choose_knife.__module__ == "simfoundry.cutting.demonstrations"
    assert R.load_demo.__module__ == "simfoundry.cutting.demonstrations"
    assert not hasattr(R, "G") and not hasattr(R, "Rotation")


def test_the_script_does_not_import_the_generator_or_scipy():
    text = (ROOT / "scripts/cutting/support_height_roundtrip.py").read_text()
    assert re.search(r"^\s*(import|from)\s+(tofu_policy_episodes|scipy)\b", text, re.MULTILINE) is None
