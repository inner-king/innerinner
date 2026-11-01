# SPDX-License-Identifier: Apache-2.0
"""The log-support target of the LeRobot converter (scripts/cutting/policy/npz_to_lerobot.py), without LeRobot.

LeRobot is replaced by a recorder, so the converter's frames, features, options and metadata are checked in the
simfoundry environment. The default (absolute) path must keep its features exactly as they were.
"""

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "scripts" / "cutting" / "policy"
sys.path.insert(0, str(POLICY))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import npz_to_lerobot as L  # noqa: E402
from demo_folder import SPT, TICKS, write_folder  # noqa: E402

from simfoundry.cutting.knife import CHEF_KNIFE  # noqa: E402
from simfoundry.cutting.support_height import (  # noqa: E402
    EPS_M,
    hand_z_from_log_height,
    lowest_edge_z,
    support_drop_near_constant,
    support_state_mode,
)

EPS = EPS_M
FAKE_LEROBOT = '''
import sys
import types
from pathlib import Path
from types import SimpleNamespace


class FakeDataset:
    """Records what the converter writes instead of writing a LeRobot dataset."""

    made = []

    def __init__(self, root, features):
        self.root = Path(root)
        self.features = features
        self.frames = []
        self.saved = 0
        self.finalized = False
        self.meta = SimpleNamespace(total_episodes=0, total_frames=0)

    @classmethod
    def create(cls, **kwargs):
        ds = cls(kwargs["root"], kwargs["features"])
        cls.made.append(ds)
        return ds

    def add_frame(self, frame):
        self.frames.append(frame)
        self.meta.total_frames += 1

    def save_episode(self):
        self.saved += 1
        self.meta.total_episodes += 1

    def finalize(self):
        self.finalized = True


def modules(dataset_cls):
    base = types.ModuleType("lerobot")
    datasets = types.ModuleType("lerobot.datasets")
    lerobot_dataset = types.ModuleType("lerobot.datasets.lerobot_dataset")
    lerobot_dataset.LeRobotDataset = dataset_cls
    return {"lerobot": base, "lerobot.datasets": datasets, "lerobot.datasets.lerobot_dataset": lerobot_dataset}
'''
_fake = {}
exec(FAKE_LEROBOT, _fake)
FakeDataset = _fake["FakeDataset"]


@pytest.fixture
def fake_lerobot(monkeypatch):
    FakeDataset.made.clear()
    for name, module in _fake["modules"](FakeDataset).items():
        monkeypatch.setitem(sys.modules, name, module)
    return FakeDataset


def _pose9(a):
    """The commanded hand pose, written out here (the generator's and demonstrations' pose9, arithmetic only)."""
    a = np.asarray(a, dtype=float)
    b1 = a[3:6] / np.linalg.norm(a[3:6])
    b2 = a[6:9] - (b1 @ a[6:9]) * b1
    b2 = b2 / np.linalg.norm(b2)
    P = np.eye(4)
    P[:3, :3] = np.column_stack([b1, b2, np.cross(b1, b2)])
    P[:3, 3] = a[:3]
    return P


def _block_top(npz):
    """s as the converter reads it: the block's centre z less half its height, in float64."""
    b = np.load(npz)["block"].astype(float)
    return float(b[2] - b[6] / 2.0)


def _base(folder):
    setup = json.loads((folder / "setup.json").read_text())
    yaw = np.radians(setup["scene_changes"]["robot_yaw_deg"])
    B = np.eye(4)
    B[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    B[:2, 3] = setup["scene_changes"]["robot_pos"][:2]
    return B, np.asarray(setup["grasp"]["knife_in_eef"], dtype=float)


def _make_folder(folder, **kwargs):
    """A synthetic demonstration folder with the image arrays and an outcome JSON per episode, as the converter reads."""
    write_folder(folder, **kwargs)
    for npz in sorted(folder.glob("ep_*.npz")):
        z = dict(np.load(npz))
        z["images_front"] = np.zeros((TICKS, 8, 8, 3), np.uint8)
        z["images_wrist"] = np.zeros((TICKS, 8, 8, 3), np.uint8)
        np.savez(npz, **z)
        j = npz.with_suffix(".json")
        info = json.loads(j.read_text()) if j.exists() else {}
        info["outcome"] = {"success": True}
        j.write_text(json.dumps(info))
    return folder


def _list_file(folder, names, tmp_path):
    p = tmp_path / "accepted.txt"
    p.write_text("\n".join(str(folder / n) for n in names) + "\n")
    return p


def _run_main(argv, root, fake, tmp_path):
    L.main([*argv, "--root", str(root)])
    return {ds.root.name: ds for ds in fake.made}


# ---- 1. the action and its values --------------------------------------------------------------------------

def test_u_is_the_log_height_of_the_next_command_and_matches_by_hand(tmp_path):
    folder = _make_folder(tmp_path / "demos", n_eps=3, regrasp_from=1)
    cache = {}
    res = L.log_support_episode(folder / "ep_00001.npz", EPS, 9, cache)
    z = np.load(folder / "ep_00001.npz")
    base, _ = _base(folder)
    knife = cache[folder]["knife"]
    assert res["knife_source"] == "fitted"
    s = _block_top(folder / "ep_00001.npz")
    assert res["s"] == pytest.approx(s, abs=1e-12)
    cmd = z["cmd"].astype(float)
    nxt = [base @ _pose9(cmd[min(i + 1, len(cmd) - 1)]) for i in range(len(cmd))]
    expected = np.array([np.log(lowest_edge_z(E, knife) - s + EPS) for E in nxt])
    np.testing.assert_allclose(res["u"], expected, rtol=0, atol=1e-12)
    assert res["u"].dtype == np.float64
    assert len(res["u"]) == len(cmd)


def test_the_last_tick_uses_its_own_command(tmp_path):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    res = L.log_support_episode(folder / "ep_00000.npz", EPS, 9, {})
    z = np.load(folder / "ep_00000.npz")
    base, knife = _base(folder)
    s = _block_top(folder / "ep_00000.npz")
    last = base @ _pose9(z["cmd"][-1].astype(float))
    assert res["u"][-1] == pytest.approx(np.log(lowest_edge_z(last, knife) - s + EPS), abs=1e-12)


def test_u_rounded_to_float32_returns_the_commanded_height_within_a_micrometre(tmp_path):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    res = L.log_support_episode(folder / "ep_00000.npz", EPS, 9, {})
    z = np.load(folder / "ep_00000.npz")
    base, knife = _base(folder)
    cmd = z["cmd"].astype(float)
    targets = np.stack([base @ _pose9(cmd[min(i + 1, len(cmd) - 1)]) for i in range(len(cmd))])
    start = targets.copy()
    start[:, 2, 3] = 0.0
    z_back = hand_z_from_log_height(start, knife, res["s"], res["u"].astype(np.float32), EPS)
    assert np.max(np.abs(z_back - targets[:, 2, 3])) < 1e-6


def test_the_knife_is_the_setup_for_the_first_episode_and_fitted_after_it(tmp_path):
    folder = _make_folder(tmp_path / "demos", n_eps=4, regrasp_from=1)
    cache = {}
    first = L.log_support_episode(folder / "ep_00000.npz", EPS, 9, cache)
    second = L.log_support_episode(folder / "ep_00001.npz", EPS, 9, cache)
    assert first["knife_source"] == "setup"
    assert second["knife_source"] == "fitted"
    assert first["fit"] is None
    assert second["fit"] is cache[folder]


def test_an_episode_json_knife_is_taken_before_any_fit(tmp_path):
    folder = _make_folder(tmp_path / "demos", n_eps=3, regrasp_from=1, json_knife=(2,))
    res = L.log_support_episode(folder / "ep_00002.npz", EPS, 9, {})
    assert res["knife_source"] == "episode"


def test_an_edge_below_the_support_by_eps_stops_with_the_episode_name(tmp_path):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99, below=(0,))
    with pytest.raises(SystemExit, match="ep_00000.npz"):
        L.log_support_episode(folder / "ep_00000.npz", EPS, 9, {})


# ---- 2. the state and the features ----------------------------------------------------------------------

def test_features_default_is_the_absolute_layout_it_was():
    f = L.features("state", 224)
    assert f["observation.state"] == {"dtype": "float32", "shape": (9,), "names": L.STATE}
    assert f["action"] == {"dtype": "float32", "shape": (9,), "names": L.ACTION}
    assert f["observation.environment_state"]["shape"] == (7,)
    assert L.ACTION[2] == "z"
    fi = L.features("image", 224)
    assert fi["observation.images.front"]["shape"] == (224, 224, 3)
    assert fi["observation.state"]["shape"] == (9,)


def test_features_for_log_support_name_u_and_the_appended_support_under_append():
    f = L.features("state", 224, support=True, state_support="append")
    assert f["observation.state"]["shape"] == (10,)
    assert f["observation.state"]["names"][-1] == "support"
    assert f["action"]["names"][2] == "u"
    assert "z" not in f["action"]["names"]
    assert f["action"]["shape"] == (9,)


def test_features_for_log_support_by_default_keep_the_absolute_state_names():
    f = L.features("state", 224, support=True)
    absolute = L.features("state", 224)
    assert f["observation.state"] == absolute["observation.state"]
    assert f["observation.state"]["shape"] == (9,)
    assert "support" not in f["observation.state"]["names"]
    assert f["action"]["names"][2] == "u"


def test_features_for_drop_and_support_under_append_have_eight_state_entries_and_eight_actions():
    f = L.features("image", 224, drop=True, support=True, state_support="append")
    assert f["observation.state"]["shape"] == (8,)
    assert f["observation.state"]["names"] == L.STATE[2:] + ["support"]
    assert f["action"]["names"] == L.ACTION_LOG[:8]
    assert f["action"]["names"][2] == "u"


def test_features_for_drop_and_support_by_default_have_the_seven_hand_pose_entries():
    f = L.features("image", 224, drop=True, support=True)
    assert f["observation.state"]["shape"] == (7,)
    assert f["observation.state"]["names"] == L.STATE[2:]


def test_under_append_the_state_gets_the_support_plane_as_its_last_entry_in_float32(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    lst = _list_file(folder, ["ep_00000.npz"], tmp_path)
    ds = _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support",
                    "--state-support", "append"], tmp_path / "out", fake_lerobot, tmp_path)["tofu_state"]
    s = _block_top(folder / "ep_00000.npz")
    default = _run_main(["--list", str(lst), "--kinds", "state"], tmp_path / "out_default",
                        fake_lerobot, tmp_path)["tofu_state"]
    for fr, fd in zip(ds.frames, default.frames):
        assert fr["observation.state"].dtype == np.float32
        assert fr["observation.state"].shape == (10,)
        assert fr["observation.state"][-1] == np.float32(s)
        np.testing.assert_array_equal(fr["observation.state"][:9], fd["observation.state"])
        np.testing.assert_array_equal(fr["observation.environment_state"], fd["observation.environment_state"])


def test_by_default_the_state_is_the_hand_and_the_same_as_the_absolute_dataset(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    lst = _list_file(folder, ["ep_00000.npz"], tmp_path)
    log = _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support"],
                    tmp_path / "log_default", fake_lerobot, tmp_path)["tofu_state"]
    absolute = _run_main(["--list", str(lst), "--kinds", "state"], tmp_path / "abs_default",
                         fake_lerobot, tmp_path)["tofu_state"]
    assert len(log.frames) == len(absolute.frames) > 0
    for fl, fa in zip(log.frames, absolute.frames):
        assert fl["observation.state"].dtype == np.float32
        assert fl["observation.state"].shape == (9,)
        np.testing.assert_array_equal(fl["observation.state"], fa["observation.state"])
        np.testing.assert_array_equal(fl["observation.environment_state"], fa["observation.environment_state"])


def test_append_changes_only_the_state_and_not_the_action(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    lst = _list_file(folder, ["ep_00000.npz"], tmp_path)
    none = _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support"],
                     tmp_path / "none", fake_lerobot, tmp_path)["tofu_state"]
    appended = _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support",
                          "--state-support", "append"], tmp_path / "append", fake_lerobot, tmp_path)["tofu_state"]
    for fn, fa in zip(none.frames, appended.frames):
        np.testing.assert_array_equal(fa["observation.state"][:9], fn["observation.state"])
        np.testing.assert_array_equal(fa["action"], fn["action"])


def test_drop_near_constant_puts_the_support_on_the_eight_entries(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    lst = _list_file(folder, ["ep_00000.npz"], tmp_path)
    argv = ["--list", str(lst), "--kinds", "state", "--action-height", "log-support", "--state-support", "append",
            "--drop-near-constant"]
    ds = _run_main(argv, tmp_path / "out", fake_lerobot, tmp_path)["tofu_state"]
    fr = ds.frames[0]
    assert fr["observation.state"].shape == (8,)
    assert fr["action"].shape == (8,)
    assert fr["observation.state"][-1] == np.float32(_block_top(folder / "ep_00000.npz"))


def test_the_action_keeps_the_other_eight_values_and_puts_u_in_the_z_slot(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    lst = _list_file(folder, ["ep_00000.npz"], tmp_path)
    log = _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support"], tmp_path / "log",
                    fake_lerobot, tmp_path)["tofu_state"]
    absolute = _run_main(["--list", str(lst), "--kinds", "state"], tmp_path / "abs", fake_lerobot, tmp_path)["tofu_state"]
    res = L.log_support_episode(folder / "ep_00000.npz", EPS, 9, {})
    for i, (fl, fa) in enumerate(zip(log.frames, absolute.frames)):
        a_log, a_abs = fl["action"], fa["action"]
        np.testing.assert_array_equal(np.delete(a_log, 2), np.delete(a_abs, 2))
        assert a_log[2] == np.float32(res["u"][i])


# ---- 3. the metadata ----------------------------------------------------------------------------------

def test_the_metadata_names_each_episode_its_knife_source_and_the_fit(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=3, regrasp_from=1)
    lst = _list_file(folder, ["ep_00000.npz", "ep_00001.npz"], tmp_path)
    out = tmp_path / "meta_out"
    _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support"], out, fake_lerobot, tmp_path)
    meta = json.loads((out / "tofu_state" / "meta" / "support_height.json").read_text())
    assert meta["eps_m"] == pytest.approx(EPS)
    assert meta["action_height"] == "log-support"
    assert meta["state_support"] == "none"
    assert meta["state_dim"] == 9
    assert meta["edge_samples"] == 100
    assert [e["knife_source"] for e in meta["episodes"]] == ["setup", "fitted"]
    assert meta["episodes"][0]["s_m"] == pytest.approx(_block_top(folder / "ep_00000.npz"), abs=1e-12)
    assert meta["episodes"][1]["ticks"] == TICKS
    assert set(meta["episodes"][1]["h_mm"]) == {"min", "median", "max"}
    assert len(meta["fits"]) == 1
    fit = meta["fits"][0]
    assert fit["folder"] == str(folder)
    assert fit["episodes"][0] == "ep_00001.npz"
    assert fit["fit_rms_mm"] < 0.5


def test_the_metadata_records_the_fit_that_was_used(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=3, regrasp_from=1)
    lst = _list_file(folder, ["ep_00000.npz", "ep_00001.npz"], tmp_path)
    out = tmp_path / "fit_out"
    _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support"], out, fake_lerobot, tmp_path)
    meta = json.loads((out / "tofu_state" / "meta" / "support_height.json").read_text())
    used = {}
    L.log_support_episode(folder / "ep_00001.npz", EPS, 9, used)
    assert meta["fits"][0]["episodes"] == used[folder]["episodes"]
    assert meta["fits"][0]["fit_rms_mm"] == pytest.approx(used[folder]["fit_rms_mm"], rel=1e-12)


def test_the_metadata_is_not_written_for_the_default_path(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    lst = _list_file(folder, ["ep_00000.npz"], tmp_path)
    out = tmp_path / "default_out"
    _run_main(["--list", str(lst), "--kinds", "state"], out, fake_lerobot, tmp_path)
    assert not (out / "tofu_state" / "meta" / "support_height.json").exists()


# ---- 4. the option checks ----------------------------------------------------------------------------------

def test_an_eps_other_than_the_clients_stops_the_run():
    args = L.parse_args(["--root", "x", "--action-height", "log-support", "--eps-mm", "2.0"])
    with pytest.raises(SystemExit, match="EPS_M"):
        L.check_args(args)


def test_log_support_with_eps_equal_to_eps_m_passes():
    args = L.parse_args(["--root", "x", "--action-height", "log-support", "--eps-mm", "1.0"])
    L.check_args(args)


def test_log_support_with_delta_stops_the_run():
    args = L.parse_args(["--root", "x", "--action-height", "log-support", "--action", "delta"])
    with pytest.raises(SystemExit, match="delta"):
        L.check_args(args)


def test_absolute_ignores_eps_mm_and_delta_is_still_allowed():
    L.check_args(L.parse_args(["--root", "x", "--eps-mm", "2.0"]))
    L.check_args(L.parse_args(["--root", "x", "--action", "delta"]))


def test_absolute_takes_state_support_none_or_nothing_and_refuses_append():
    L.check_args(L.parse_args(["--root", "x", "--state-support", "none"]))
    L.check_args(L.parse_args(["--root", "x"]))
    with pytest.raises(SystemExit, match="log-support"):
        L.check_args(L.parse_args(["--root", "x", "--state-support", "append"]))


def test_the_state_support_default_is_none_for_log_support_and_append_is_explicit():
    assert L.parse_args(["--root", "x", "--action-height", "log-support"]).state_support is None
    args = L.parse_args(["--root", "x", "--action-height", "log-support", "--state-support", "append"])
    assert args.state_support == "append"


def test_absolute_with_append_stops_the_run_before_any_dataset_is_made(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    lst = _list_file(folder, ["ep_00000.npz"], tmp_path)
    with pytest.raises(SystemExit, match="log-support"):
        L.main(["--list", str(lst), "--kinds", "state", "--state-support", "append", "--root", str(tmp_path / "x")])
    assert fake_lerobot.made == []


def test_a_meta_without_the_state_support_key_reads_as_append():
    assert support_state_mode({"action_height": "log-support", "eps_m": 1e-3}) == "append"
    assert support_state_mode({"state_support": "none", "state_dim": 9}) == "none"
    assert support_state_mode({"state_support": "append", "state_dim": 10}) == "append"


def test_a_state_support_value_that_is_neither_mode_stops_with_the_value_named():
    with pytest.raises(ValueError, match="'both'"):
        support_state_mode({"state_support": "both"})


def test_the_append_meta_records_append_and_the_ten_entry_state(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=3, regrasp_from=1)
    lst = _list_file(folder, ["ep_00000.npz", "ep_00001.npz"], tmp_path)
    out = tmp_path / "append_meta"
    _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support", "--state-support", "append"],
              out, fake_lerobot, tmp_path)
    meta = json.loads((out / "tofu_state" / "meta" / "support_height.json").read_text())
    assert meta["state_support"] == "append"
    assert meta["state_dim"] == 10
    assert support_state_mode(meta) == "append"


def test_the_absolute_path_never_imports_simfoundry(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    lst = _list_file(folder, ["ep_00000.npz"], tmp_path)
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(POLICY)!r})\n"
        "ns = {}\n"
        f"exec({FAKE_LEROBOT!r}, ns)\n"
        "for name, mod in ns['modules'](ns['FakeDataset']).items():\n"
        "    sys.modules[name] = mod\n"
        "import npz_to_lerobot as L\n"
        f"L.main(['--list', {str(lst)!r}, '--kinds', 'state', '--root', {str(tmp_path / 'sub')!r}])\n"
        "print('simfoundry-loaded', any(m == 'simfoundry' or m.startswith('simfoundry.') for m in sys.modules))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "simfoundry-loaded False" in proc.stdout


def test_a_fitted_knife_without_scipy_stops_with_the_clear_message(tmp_path):
    folder = _make_folder(tmp_path / "demos", n_eps=3, regrasp_from=1)
    code = (
        "import sys\n"
        "sys.modules['scipy'] = None\n"
        f"sys.path.insert(0, {str(POLICY)!r})\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "import npz_to_lerobot as L\n"
        f"L.log_support_episode({str(folder / 'ep_00001.npz')!r}, {EPS!r}, 9, {{}})\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path))
    assert proc.returncode != 0
    assert "this episode's knife transform has to be fitted (needs scipy)" in proc.stderr
    assert "knife_in_eef" in proc.stderr


def test_setup_and_episode_knives_work_without_scipy(tmp_path):
    folder = _make_folder(tmp_path / "demos", n_eps=3, regrasp_from=99, json_knife=(2,))
    code = (
        "import sys\n"
        "sys.modules['scipy'] = None\n"
        f"sys.path.insert(0, {str(POLICY)!r})\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "import npz_to_lerobot as L\n"
        f"a = L.log_support_episode({str(folder / 'ep_00000.npz')!r}, {EPS!r}, 9, {{}})\n"
        f"b = L.log_support_episode({str(folder / 'ep_00002.npz')!r}, {EPS!r}, 9, {{}})\n"
        "print(a['knife_source'], b['knife_source'])\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["setup", "episode"]


# ---- 5. the default path is unchanged ------------------------------------------------------------------------

def test_default_frames_are_the_absolute_values_of_the_last_tick_rule(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    lst = _list_file(folder, ["ep_00000.npz"], tmp_path)
    ds = _run_main(["--list", str(lst), "--kinds", "state"], tmp_path / "abs", fake_lerobot, tmp_path)["tofu_state"]
    cmd = np.load(folder / "ep_00000.npz")["cmd"].astype(np.float32)
    act = np.concatenate([cmd[1:], cmd[-1:]], axis=0)
    for i, fr in enumerate(ds.frames):
        np.testing.assert_array_equal(fr["action"], act[i])
        assert fr["observation.state"].shape == (9,)


# ---- 6. the fit window and the sign of the residual ---------------------------------------------------------

def test_the_fit_uses_the_first_nine_unexplained_episodes(tmp_path, fake_lerobot):
    assert L.FIT_EPISODES == 9
    folder = _make_folder(tmp_path / "demos", n_eps=12, regrasp_from=1)
    lst = _list_file(folder, ["ep_00000.npz", "ep_00001.npz"], tmp_path)
    out = tmp_path / "window_out"
    _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support"], out, fake_lerobot, tmp_path)
    meta = json.loads((out / "tofu_state" / "meta" / "support_height.json").read_text())
    assert meta["fits"][0]["episodes"] == [f"ep_{i:05d}.npz" for i in range(1, 10)]


def test_the_residual_maximum_is_the_largest_magnitude_when_every_residual_is_negative(tmp_path):
    # A bias of +2 mm makes the recorded edge higher than the knife predicts, so every residual (predicted minus
    # recorded) is about -2 mm and the largest of them is the largest in magnitude only after the absolute value.
    folder = _make_folder(tmp_path / "demos", n_eps=4, regrasp_from=99, bias_m={3: 2e-3}, json_knife=(3,))
    res = L.log_support_episode(folder / "ep_00003.npz", EPS, 9, {})
    assert res["knife_source"] == "episode"
    z = np.load(folder / "ep_00003.npz")
    state = z["state"].astype(float)
    edge = z["steps"].astype(float)[::SPT, 6]  # the physics step matched to each tick, as the converter takes it
    base, knife = _base(folder)
    u = np.linspace(-CHEF_KNIFE.half_length, CHEF_KNIFE.half_length, 100)
    eh = CHEF_KNIFE.edge_height(u)
    residuals = []
    for t in range(TICKS):
        hand = np.eye(4)
        hand[:3, :3] = Rotation.from_quat(state[t, 12:16]).as_matrix()
        hand[:3, 3] = state[t, 9:12]
        K = base @ hand @ knife
        predicted = K[2, 3] + np.min(K[2, 0] * u + K[2, 2] * eh)
        residuals.append(1e3 * (predicted - edge[t]))
    residuals = np.array(residuals)
    assert residuals.max() < 0  # every residual is negative, so the signed maximum is not the largest magnitude
    assert np.all(np.abs(residuals + 2.0) < 0.5)  # about the bias, with the 0.1 mm noise of write_folder
    assert res["resid_max_mm"] > 0
    assert res["resid_max_mm"] == pytest.approx(float(np.max(np.abs(residuals))), abs=1e-9)


# ---- 7. the finger entries left out (--drop-near-constant) ------------------------------------------------------

def test_drop_by_default_the_state_is_the_pose_and_the_same_as_the_absolute_drop_dataset(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    lst = _list_file(folder, ["ep_00000.npz"], tmp_path)
    log = _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support", "--drop-near-constant"],
                    tmp_path / "log_drop", fake_lerobot, tmp_path)["tofu_state"]
    absolute = _run_main(["--list", str(lst), "--kinds", "state", "--drop-near-constant"], tmp_path / "abs_drop",
                         fake_lerobot, tmp_path)["tofu_state"]
    assert log.features["observation.state"]["names"] == L.STATE[2:]
    assert len(log.frames) == len(absolute.frames) > 0
    for fl, fa in zip(log.frames, absolute.frames):
        assert fl["observation.state"].dtype == np.float32
        assert fl["observation.state"].shape == (7,)
        np.testing.assert_array_equal(fl["observation.state"], fa["observation.state"])


def test_drop_under_append_is_the_pose_then_the_support_plane(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=2, regrasp_from=99)
    lst = _list_file(folder, ["ep_00000.npz"], tmp_path)
    none = _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support", "--drop-near-constant"],
                     tmp_path / "none", fake_lerobot, tmp_path)["tofu_state"]
    appended = _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support",
                          "--drop-near-constant", "--state-support", "append"], tmp_path / "append",
                         fake_lerobot, tmp_path)["tofu_state"]
    s = _block_top(folder / "ep_00000.npz")
    assert appended.features["observation.state"]["names"] == L.STATE[2:] + ["support"]
    for fa, fn in zip(appended.frames, none.frames):
        assert fa["observation.state"].dtype == np.float32
        assert fa["observation.state"].shape == (8,)
        np.testing.assert_array_equal(fa["observation.state"][:7], fn["observation.state"])
        assert fa["observation.state"][-1] == np.float32(s)
        np.testing.assert_array_equal(fa["action"], fn["action"])


def test_the_meta_records_the_drop_and_the_state_dimension_of_each_layout(tmp_path, fake_lerobot):
    folder = _make_folder(tmp_path / "demos", n_eps=3, regrasp_from=1)
    lst = _list_file(folder, ["ep_00000.npz", "ep_00001.npz"], tmp_path)
    cases = [
        (["--drop-near-constant"], "none", True, 7),
        (["--drop-near-constant", "--state-support", "append"], "append", True, 8),
        ([], "none", False, 9),
    ]
    for k, (extra, mode, drop, dim) in enumerate(cases):
        out = tmp_path / f"meta_{k}"
        _run_main(["--list", str(lst), "--kinds", "state", "--action-height", "log-support", *extra], out,
                  fake_lerobot, tmp_path)
        meta = json.loads((out / "tofu_state" / "meta" / "support_height.json").read_text())
        assert meta["state_support"] == mode
        assert meta["drop_near_constant"] is drop
        assert meta["state_dim"] == dim
        assert support_drop_near_constant(meta) is drop


def test_a_meta_without_the_drop_key_keeps_the_fingers_and_a_non_boolean_stops_with_the_value_named():
    assert support_drop_near_constant({"state_support": "none"}) is False
    assert support_drop_near_constant({"drop_near_constant": True}) is True
    with pytest.raises(ValueError, match="'yes'"):
        support_drop_near_constant({"drop_near_constant": "yes"})
