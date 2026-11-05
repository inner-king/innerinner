# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--probe-out`` of the policy runner: a test-only probe of the segmentation channels and camera poses, off by default.

The functions in ``simfoundry/utils/camera_probe.py`` are tested on their own, on fake observations. ``setup_args`` and
``EpisodeRun.capture`` run on the runner, which imports in the simfoundry env. ``capture`` is driven by fake cameras
that count their ``get_obs`` reads, so the one read per camera is counted, not assumed. The structure tests read the
runner's syntax tree against the commit the probe was cut from, and allow exactly the probe's additions.
"""

import ast
import copy
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simfoundry.utils import camera_probe as cp

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts/cutting/tofu_policy_episodes.py"
BASE_COMMIT = "b76779e"  # the integration tip before the probe: the probe's own additions are the only difference
sys.path.insert(0, str(ROOT / "scripts/cutting"))
try:
    import tofu_policy_episodes as G  # noqa: E402 - the runner's own imports need the simulator's packages in some envs
    RUNNER_ERROR = None
except ImportError as error:  # only setup_args and capture need the runner; the structure tests always run
    G, RUNNER_ERROR = None, error


def _runner():
    if G is None:
        pytest.skip(f"the runner cannot be imported here: {RUNNER_ERROR}")
    return G


try:
    BASE = ast.parse(subprocess.run(["git", "show", f"{BASE_COMMIT}:scripts/cutting/tofu_policy_episodes.py"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout)
except (OSError, subprocess.CalledProcessError) as error:
    pytest.skip(f"the base commit {BASE_COMMIT} is not in this clone: {error}", allow_module_level=True)
CURRENT = ast.parse(GEN.read_text())
PROBE_FLAGS = ("--probe-out", "--probe-ticks")
PROBE_TEST = "args.probe_out is not None"
CAPTURE_TEST = "self.probe is not None and self.probe.wants(len(self.rec))"
PROBE_ASSIGNS = ("run.probe", "run.probe_seed")
DEFAULT_TICKS = [0, 25, 50, 75, 99]


# ------------------------------------------------------------------ syntax-tree helpers


def _function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _method(tree, cls, name):
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _text(node):
    return ast.unparse(node)


def _expr(text):
    return ast.parse(text, mode="eval").body


def _is_test(node, text):
    return ast.dump(node) == ast.dump(_expr(text))


def _parents(tree):
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _enclosing(node, parents, kind):
    out, cur = [], node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, kind):
            out.append(cur)
    return out


def _add_argument_named(tree, flag):
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "add_argument" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == flag]
    assert len(calls) == 1, f"{flag} is added {len(calls)} times"
    return {k.arg: k.value for k in calls[0].keywords}


MAIN = _function(CURRENT, "main")
MAIN_BASE = _function(BASE, "main")
CAPTURE = _method(CURRENT, "EpisodeRun", "capture")
CAPTURE_BASE = _method(BASE, "EpisodeRun", "capture")


def _is_probe_arg(stmt):
    call = stmt.value if isinstance(stmt, ast.Expr) else None
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument"
            and call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value in PROBE_FLAGS)


def _is_probe_assign(stmt):
    return isinstance(stmt, ast.Assign) and any(_text(t) in PROBE_ASSIGNS for t in stmt.targets)


def _drop_probe(stmts):
    """``stmts`` without the probe's statements, at any depth (its seed is set inside the episode loop)."""
    out = []
    for s in stmts:
        if _is_probe_arg(s) or _is_probe_assign(s) or (isinstance(s, ast.If) and _is_test(s.test, PROBE_TEST)):
            continue
        for field in ("body", "orelse", "finalbody"):
            if isinstance(getattr(s, field, None), list):
                setattr(s, field, _drop_probe(getattr(s, field)))
        out.append(s)
    return out


def _main_without_probe(main):
    tree = copy.deepcopy(main)
    tree.body = _drop_probe(tree.body)
    # The colour-matrix hook and the name it adds to the splat import belong to the visual background, and they are
    # additions over the base too. Taken out here, so that this test still checks the probe's additions alone.
    tree.body = [s for s in tree.body
                 if not (isinstance(s, ast.If) and _is_test(s.test, "visual_bg is not None and visual_bg.ccm is not None"))]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "simfoundry.utils.splat_background":
            node.names = [alias for alias in node.names if alias.name != "apply_splat_ccm"]
    return tree


def _capture_without_read_and_probe(fn):
    """The statements of ``capture`` except the one read of the cameras and the probe block (both of which the probe changes)."""
    out = []
    for s in fn.body:
        text = _text(s)
        if isinstance(s, ast.If) and _is_test(s.test, CAPTURE_TEST):
            continue
        if "frames" in text or text.startswith("imgs = {n: T.np_(c.get_obs()"):
            continue
        out.append(s)
    return out


def _base_function(name):
    node = _function(BASE, name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<base>", "exec"), namespace)  # noqa: S102 - the base's own source
    return namespace[name]


# ------------------------------------------------------- behaviour: the pure functions


def _obs(height=4, width=5, channels=4, seed=0):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
    inst = rng.integers(0, 40, size=(height, width)).astype(np.int64)
    sem = rng.integers(0, 9, size=(height, width)).astype(np.int64)
    return {"rgb": rgb, "seg_instance": inst, "seg_semantic": sem}


def test_probe_arrays_keep_the_rgb_channels_and_the_segmentation_values():
    obs = _obs()
    arrays, _ = cp.probe_arrays(obs, {})
    assert arrays["rgb"].dtype == np.uint8 and arrays["rgb"].shape == (4, 5, 3)
    assert np.array_equal(arrays["rgb"], obs["rgb"][:, :, :3])
    for key in ("seg_instance", "seg_semantic"):
        assert arrays[key].dtype == np.uint32 and arrays[key].shape == (4, 5)
        assert np.array_equal(arrays[key], obs[key])
    assert set(arrays) == {"rgb", "seg_instance", "seg_semantic"}


def test_a_single_channel_segmentation_is_squeezed_to_height_by_width():
    obs = _obs()
    obs["seg_instance"] = obs["seg_instance"][:, :, None]
    arrays, _ = cp.probe_arrays(obs, {})
    assert arrays["seg_instance"].shape == (4, 5)


def test_a_segmentation_of_another_shape_than_the_rgb_is_refused():
    obs = _obs()
    obs["seg_semantic"] = np.zeros((3, 5), np.int64)
    with pytest.raises(ValueError, match="seg_semantic"):
        cp.probe_arrays(obs, {})


def test_a_negative_segmentation_id_is_refused():
    obs = _obs()
    obs["seg_instance"][0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        cp.probe_arrays(obs, {})


def test_tensors_are_read_without_changing_their_dtype():
    torch = pytest.importorskip("torch")
    obs = {key: torch.from_numpy(value) for key, value in _obs().items()}
    arrays, _ = cp.probe_arrays(obs, {})
    assert arrays["rgb"].dtype == np.uint8 and arrays["seg_instance"].dtype == np.uint32
    assert np.array_equal(arrays["seg_semantic"], obs["seg_semantic"].numpy())


def test_segmentation_ids_become_string_keys_and_the_names_are_kept():
    info = {"seg_instance": {3: "tofu", 0: "floor"}, "seg_semantic": {np.int64(1): "food"}}
    _, seg_info = cp.probe_arrays(_obs(), info)
    assert seg_info == {"instance": {"3": "tofu", "0": "floor"}, "semantic": {"1": "food"}}
    assert all(isinstance(key, str) for table in seg_info.values() for key in table)


@pytest.mark.parametrize("info", [None, {}])
def test_missing_name_tables_are_empty(info):
    _, seg_info = cp.probe_arrays(_obs(), info)
    assert seg_info == {"instance": {}, "semantic": {}}


def _obs_without_semantic(seed=0):
    obs = _obs(seed=seed)
    del obs["seg_semantic"]
    return obs


def test_without_seg_semantic_the_probe_reads_rgb_and_instance_only():
    obs = _obs_without_semantic()
    arrays, _ = cp.probe_arrays(obs, {})
    assert set(arrays) == {"rgb", "seg_instance"}
    assert np.array_equal(arrays["rgb"], obs["rgb"][:, :, :3])
    assert arrays["seg_instance"].dtype == np.uint32 and np.array_equal(arrays["seg_instance"], obs["seg_instance"])


def test_without_seg_semantic_the_semantic_names_stay_empty_even_when_a_table_is_given():
    info = {"seg_instance": {3: "tofu"}, "seg_semantic": {np.int64(1): "food"}}
    _, seg_info = cp.probe_arrays(_obs_without_semantic(), info)
    assert seg_info == {"instance": {"3": "tofu"}, "semantic": {}}


@pytest.mark.parametrize("missing", ["rgb", "seg_instance"])
def test_the_probe_still_needs_rgb_and_instance(missing):
    obs = _obs()
    del obs[missing]
    with pytest.raises(ValueError, match=missing):
        cp.probe_arrays(obs, {})


def test_pose_record_gives_plain_lists_and_refuses_the_wrong_lengths():
    pose = cp.pose_record(np.array([0.5, -1.0, 2.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    assert pose == {"position": [0.5, -1.0, 2.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    assert all(type(v) is float for v in pose["position"] + pose["orientation_xyzw"])
    with pytest.raises(ValueError):
        cp.pose_record([1.0, 2.0], [0.0, 0.0, 0.0, 1.0])


class _FakeAttr:
    def __init__(self, value, valid=True):
        self._value, self._valid = value, valid

    def IsValid(self):
        return self._valid

    def Get(self):
        return self._value


class _FakePrim:
    def __init__(self, attrs):
        self._attrs = attrs

    def GetAttribute(self, name):
        return self._attrs.get(name, _FakeAttr(None, valid=False))


def test_camera_attributes_read_the_lens_values_and_null_the_missing_ones():
    prim = _FakePrim({"focalLength": _FakeAttr(17.0), "horizontalAperture": _FakeAttr(20.955),
                      "verticalAperture": _FakeAttr(15.2), "horizontalApertureOffset": _FakeAttr(0.0, valid=False)})
    assert cp.camera_attributes(prim) == {
        "focal_length": 17.0, "horizontal_aperture": 20.955, "vertical_aperture": 15.2,
        "horizontal_aperture_offset": None, "vertical_aperture_offset": None,
    }
    assert cp.camera_attributes(None) == {key: None for key in cp.CAMERA_ATTRIBUTES}


def _probe_payload(height=4, width=5):
    rng = np.random.default_rng(1)
    arrays = {"rgb": rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
              "seg_instance": rng.integers(0, 40, (height, width), dtype=np.uint32),
              "seg_semantic": rng.integers(0, 9, (height, width), dtype=np.uint32)}
    pose = {"position": [0.1, 0.2, 0.3], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    camera_info = {"focal_length": 17.0, "horizontal_aperture": 20.955, "vertical_aperture": None,
                   "horizontal_aperture_offset": 0.0, "vertical_aperture_offset": 0.0}
    seg_info = {"instance": {"1": "tofu", "0": "floor"}, "semantic": {"2": "food"}}
    return arrays, pose, camera_info, seg_info


def test_write_probe_names_the_files_by_seed_tick_and_camera(tmp_path):
    arrays, pose, camera_info, seg_info = _probe_payload()
    npz_path, json_path = cp.write_probe(tmp_path, 7, 25, "front", arrays, pose, camera_info, seg_info)
    assert npz_path == tmp_path / "ep00007" / "t0025_front.npz"
    assert json_path == tmp_path / "ep00007" / "t0025_front.json"


def test_write_probe_keeps_the_dtypes_and_the_values_of_the_arrays(tmp_path):
    arrays, pose, camera_info, seg_info = _probe_payload()
    npz_path, _ = cp.write_probe(tmp_path, 7, 0, "wrist", arrays, pose, camera_info, seg_info)
    with np.load(npz_path) as stored:
        assert set(stored.files) == {"rgb", "seg_instance", "seg_semantic"}
        assert stored["rgb"].dtype == np.uint8 and stored["seg_instance"].dtype == np.uint32
        assert stored["seg_semantic"].dtype == np.uint32
        for key in arrays:
            assert np.array_equal(stored[key], arrays[key])


def test_write_probe_json_has_the_pose_size_lens_and_names(tmp_path):
    arrays, pose, camera_info, seg_info = _probe_payload(height=4, width=5)
    _, json_path = cp.write_probe(tmp_path, 7, 0, "front", arrays, pose, camera_info, seg_info)
    record = json.loads(json_path.read_text())
    assert list(record) == ["position", "orientation_xyzw", "image_width", "image_height", "focal_length",
                            "horizontal_aperture", "vertical_aperture", "horizontal_aperture_offset",
                            "vertical_aperture_offset", "seg_info"]
    assert record["position"] == [0.1, 0.2, 0.3] and record["orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert (record["image_width"], record["image_height"]) == (5, 4)
    assert record["focal_length"] == 17.0 and record["vertical_aperture"] is None
    assert record["seg_info"] == {"instance": {"1": "tofu", "0": "floor"}, "semantic": {"2": "food"}}


def test_write_probe_without_seg_semantic_writes_two_arrays_and_an_empty_semantic_table(tmp_path):
    arrays, pose, camera_info, seg_info = _probe_payload()
    del arrays["seg_semantic"]  # the read had no semantic channel, although the table is given
    npz_path, json_path = cp.write_probe(tmp_path, 7, 0, "front", arrays, pose, camera_info, seg_info)
    with np.load(npz_path) as stored:
        assert set(stored.files) == {"rgb", "seg_instance"}
    record = json.loads(json_path.read_text())
    assert record["seg_info"] == {"instance": {"1": "tofu", "0": "floor"}, "semantic": {}}


def test_the_default_ticks_are_the_documented_five():
    assert cp.CameraProbe("unused").ticks == frozenset({0, 25, 50, 75, 99})
    assert cp.DEFAULT_TICKS == (0, 25, 50, 75, 99)


def test_only_the_probed_ticks_are_written(tmp_path):
    arrays, pose, camera_info, seg_info = _probe_payload()
    probe = cp.CameraProbe(tmp_path, ticks=[0, 25])
    results = [probe.save(3, tick, "front", arrays, pose, camera_info, seg_info) for tick in range(30)]
    assert [tick for tick, paths in enumerate(results) if paths is not None] == [0, 25]
    assert sorted(p.name for p in (tmp_path / "ep00003").glob("*.npz")) == ["t0000_front.npz", "t0025_front.npz"]


def test_an_unprobed_tick_creates_no_folder(tmp_path):
    arrays, pose, camera_info, seg_info = _probe_payload()
    assert cp.CameraProbe(tmp_path, ticks=[0]).save(5, 1, "wrist", arrays, pose, camera_info, seg_info) is None
    assert not (tmp_path / "ep00005").exists()


# ----------------------------------------------------- behaviour: setup_args on the runner


def _args(**over):
    values = {"scene": Path("Data/x"), "visual_background": None, "probe_out": None, "probe_ticks": DEFAULT_TICKS,
              "mode": "expert", "seed": 3, "dart_sigma_mm": 0.0, "dart_tau_s": 0.1, "dart_clip_mm": 2.0}
    values.update(over)
    return Namespace(**values)


@pytest.mark.parametrize("visual", [None, Path("spec.json")])
@pytest.mark.parametrize("probe", [None, Path("probe_dir")])
def test_setup_args_has_the_probe_options_only_with_probe_out(visual, probe):
    args = _args(visual_background=visual, probe_out=probe)
    out = _runner().setup_args(args)
    assert ("probe_out" in out) is (probe is not None)
    assert ("probe_ticks" in out) is (probe is not None)
    assert ("visual_background" in out) is (visual is not None)
    assert list(out) == [k for k in vars(args) if k in out], "the keys keep the order of the namespace"
    if probe is not None:
        assert out["probe_out"] is args.probe_out and out["probe_ticks"] is args.probe_ticks


def test_setup_args_with_the_probe_off_matches_the_base_runner():
    base_args = Namespace(**{k: v for k, v in vars(_args()).items() if k not in ("probe_out", "probe_ticks", "visual_background")})
    assert _runner().setup_args(_args()) == _base_function("setup_args")(base_args)


# -------------------------------------------- behaviour: capture on fake cameras (one read each)


class _FakeCam:
    """A camera whose get_obs reads count, whose lens is its own, and whose pose the capture can set."""

    def __init__(self, rgb, inst, sem, position, lens, semantic=True):
        self.reads = 0
        self._obs = {"rgb": rgb, "seg_instance": inst, "seg_semantic": sem}
        self._info = {"seg_instance": {1: "tofu"}, "seg_semantic": {2: "food"}}
        if not semantic:  # a read on a subset install: no semantic channel at all
            del self._obs["seg_semantic"], self._info["seg_semantic"]
        self.position, self.orientation = list(position), [0.0, 0.0, 0.0, 1.0]
        self.prim = _FakePrim({name: _FakeAttr(value) for name, value in lens.items()})

    def get_obs(self):
        self.reads += 1
        return dict(self._obs), dict(self._info)

    def set_position_orientation(self, position, orientation):
        self.position, self.orientation = position.tolist(), orientation.tolist()

    def get_position_orientation(self):
        import torch
        return torch.tensor(self.position), torch.tensor(self.orientation)


FRONT_LENS = {"focalLength": 17.0, "horizontalAperture": 20.955, "verticalAperture": 15.2}
WRIST_LENS = {"focalLength": 12.0, "horizontalAperture": 14.4, "verticalAperture": 10.8, "horizontalApertureOffset": 0.5}


def _cams(semantic=True):
    """Two cameras with different images and different lenses, so that a mix-up between them shows."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)

    def cam(position, lens):
        return _FakeCam(torch.from_numpy(rng.integers(0, 256, (6, 8, 4), dtype=np.uint8)),
                        torch.from_numpy(rng.integers(0, 5, (6, 8), dtype=np.int64)),
                        torch.from_numpy(rng.integers(0, 3, (6, 8), dtype=np.int64)), position, lens, semantic)
    return {"front": cam([0.5, 0.0, 0.5], FRONT_LENS), "wrist": cam([0.0, 0.0, 0.0], WRIST_LENS)}


def _episode_run(cams, probe=None, seed=7):
    run = _runner().EpisodeRun.__new__(_runner().EpisodeRun)  # skips the simulator-bound constructor
    run.cams = cams
    run.arm = SimpleNamespace(eef=lambda: np.eye(4), base_inv=np.eye(4), joints=lambda: np.zeros(7),
                              fingers=lambda: np.zeros(2))
    run.wrist_offset, run.cmd, run.points = np.eye(4), np.eye(4), None
    run.og = SimpleNamespace(sim=SimpleNamespace(render=lambda: None))
    run.rec, run.frame_times, run.overview = [], [], []
    run.k, run.dt, run.phase, run.dart = 0, 0.1, "policy", None
    run.cuts_done = lambda: 0
    run.probe, run.probe_seed = probe, seed
    return run


def test_capture_reads_each_camera_once_with_the_probe_off():
    cams = _cams()
    run = _episode_run(cams)
    run.capture()
    assert [c.reads for c in cams.values()] == [1, 1]
    assert run.rec[-1]["front"].dtype == np.uint8 and run.rec[-1]["front"].shape == (6, 8, 3)


def test_capture_reads_each_camera_once_with_the_probe_on_and_saves_only_the_probed_tick(tmp_path):
    cams = _cams()
    run = _episode_run(cams, probe=cp.CameraProbe(tmp_path, ticks=[0]), seed=7)
    run.capture()  # tick 0, probed
    run.capture()  # tick 1, not probed
    assert [c.reads for c in cams.values()] == [2, 2]
    assert sorted(p.name for p in (tmp_path / "ep00007").iterdir()) == [
        "t0000_front.json", "t0000_front.npz", "t0000_wrist.json", "t0000_wrist.npz"]
    assert not np.array_equal(cams["front"]._obs["seg_instance"].numpy(), cams["wrist"]._obs["seg_instance"].numpy())
    for name in ("front", "wrist"):  # each file holds its own camera's arrays, not the other camera's
        with np.load(tmp_path / "ep00007" / f"t0000_{name}.npz") as stored:
            obs = cams[name]._obs
            assert stored["rgb"].dtype == np.uint8 and stored["seg_instance"].dtype == np.uint32
            assert stored["seg_semantic"].dtype == np.uint32
            assert np.array_equal(stored["rgb"], obs["rgb"].numpy()[:, :, :3]), name
            assert np.array_equal(stored["seg_instance"], obs["seg_instance"].numpy()), name
            assert np.array_equal(stored["seg_semantic"], obs["seg_semantic"].numpy()), name


def test_capture_with_the_probe_on_and_no_seg_semantic_saves_two_arrays_per_camera(tmp_path):
    cams = _cams(semantic=False)
    run = _episode_run(cams, probe=cp.CameraProbe(tmp_path, ticks=[0]), seed=7)
    run.capture()
    assert [c.reads for c in cams.values()] == [1, 1]
    for name in ("front", "wrist"):
        with np.load(tmp_path / "ep00007" / f"t0000_{name}.npz") as stored:
            assert set(stored.files) == {"rgb", "seg_instance"}
            assert np.array_equal(stored["seg_instance"], cams[name]._obs["seg_instance"].numpy()), name
        record = json.loads((tmp_path / "ep00007" / f"t0000_{name}.json").read_text())
        assert record["seg_info"] == {"instance": {"1": "tofu"}, "semantic": {}}


def test_the_probe_leaves_the_recorded_ticks_unchanged(tmp_path):
    off, on = _cams(), _cams()
    run_off, run_on = _episode_run(off), _episode_run(on, probe=cp.CameraProbe(tmp_path, ticks=[0]))
    run_off.capture()
    run_on.capture()
    for key in ("front", "wrist", "state", "cmd", "t"):
        assert np.array_equal(np.asarray(run_off.rec[-1][key]), np.asarray(run_on.rec[-1][key])), key


def test_each_probe_json_holds_its_own_camera_pose_and_lens(tmp_path):
    cams = _cams()
    run = _episode_run(cams, probe=cp.CameraProbe(tmp_path, ticks=[0]), seed=2)
    run.capture()
    front = json.loads((tmp_path / "ep00002" / "t0000_front.json").read_text())
    wrist = json.loads((tmp_path / "ep00002" / "t0000_wrist.json").read_text())
    assert front["position"] == pytest.approx(cams["front"].position)
    assert wrist["position"] == pytest.approx(cams["wrist"].position)
    assert front["orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0] and wrist["orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert front["focal_length"] == 17.0 and front["horizontal_aperture"] == 20.955
    assert front["vertical_aperture"] == 15.2 and front["horizontal_aperture_offset"] is None
    assert wrist["focal_length"] == 12.0 and wrist["horizontal_aperture"] == 14.4
    assert wrist["vertical_aperture"] == 10.8 and wrist["horizontal_aperture_offset"] == 0.5
    assert wrist["vertical_aperture_offset"] is None


def test_probe_arrays_keep_segmentation_ids_above_float32_precision():
    big = 16777217  # 2**24 + 1, the first integer a float32 cannot hold
    obs = _obs()
    obs["seg_instance"][0, 0] = big
    arrays, _ = cp.probe_arrays(obs, {})
    assert arrays["seg_instance"].dtype == np.uint32 and arrays["seg_instance"][0, 0] == big


def test_numpy_keeps_integer_dtypes_and_values_above_float32_precision():
    values = np.array([16777217, 4294967295], dtype=np.int64)
    out = cp._numpy(values)
    assert out.dtype == np.int64 and np.array_equal(out, values)
    torch = pytest.importorskip("torch")
    out_torch = cp._numpy(torch.tensor([16777217], dtype=torch.int64))
    assert out_torch.dtype == np.int64 and out_torch[0] == 16777217


# ----------------------------------------------------- structure: the syntax tree


def test_the_runner_hands_the_probe_its_two_options():
    creates = [n for n in ast.walk(MAIN) if isinstance(n, ast.Assign) and _text(n.targets[0]) == "run.probe"]
    call = creates[0].value.body
    assert isinstance(call, ast.Call) and _text(call.func) == "CameraProbe"
    assert [_text(a) for a in call.args] == ["args.probe_out", "args.probe_ticks"]


def test_the_probe_import_follows_the_other_simfoundry_imports_in_order():
    names = [n.module for n in CURRENT.body if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("simfoundry")]
    assert "simfoundry.utils.camera_probe" in names
    assert names == sorted(names), "the simfoundry imports are in alphabetical order"


def test_probe_options_are_optional_with_the_stated_defaults():
    probe_out = _add_argument_named(MAIN, "--probe-out")
    assert _text(probe_out["default"]) == "None" and _text(probe_out["type"]) == "Path"
    assert "required" not in probe_out
    probe_ticks = _add_argument_named(MAIN, "--probe-ticks")
    assert _text(probe_ticks["type"]) == "int" and _text(probe_ticks["nargs"]) == "'*'"
    assert ast.literal_eval(probe_ticks["default"]) == DEFAULT_TICKS


def test_the_segmentation_channels_are_added_only_under_the_option():
    parents = _parents(CURRENT)
    blocks = [s for s in MAIN.body if isinstance(s, ast.If) and _is_test(s.test, PROBE_TEST)]
    assert len(blocks) == 1
    assigns = [n for n in ast.walk(blocks[0]) if isinstance(n, ast.Assign)]
    assert len(assigns) == 1 and _text(assigns[0].targets[0]) == "sensor['modalities']"
    loops = [s for s in blocks[0].body if isinstance(s, ast.For)]
    assert len(loops) == 1 and _text(loops[0].target) == "sensor" and _text(loops[0].iter) == "sensors", \
        "the probe's modalities are set on every sensor, not on a slice of them"
    assert [s for s in loops[0].body if isinstance(s, ast.Assign)] == assigns, "the assignment is the loop's body"
    subscript = assigns[0].targets[0]
    assert isinstance(subscript, ast.Subscript) and _text(subscript.value) == "sensor"
    assert ast.literal_eval(assigns[0].value) == ["rgb", "seg_instance"]
    for node in assigns:
        assert any(s is blocks[0] for s in _enclosing(node, parents, ast.If))
    literals = [ast.literal_eval(v) for n in ast.walk(MAIN) if isinstance(n, ast.Dict)
                for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and k.value == "modalities"]
    assert literals and all(m == ["rgb"] for m in literals), "the sensor literals keep the rgb-only modalities"


def test_the_probe_is_created_only_under_the_option_and_the_seed_is_set_per_episode():
    creates = [n for n in ast.walk(MAIN) if isinstance(n, ast.Assign) and _text(n.targets[0]) == "run.probe"]
    assert len(creates) == 1 and isinstance(creates[0].value, ast.IfExp)
    assert _is_test(creates[0].value.test, PROBE_TEST)
    loops = [n for n in ast.walk(MAIN) if isinstance(n, ast.For) and _text(n.target) == "seed"]
    assert len(loops) == 1
    seeds = [n for n in ast.walk(loops[0]) if isinstance(n, ast.Assign) and _text(n.targets[0]) == "run.probe_seed"]
    assert len(seeds) == 1 and _text(seeds[0].value) == "seed"


def test_capture_calls_get_obs_once_per_camera_in_the_loop_over_the_cameras():
    calls = [n for n in ast.walk(CAPTURE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "get_obs"]
    assert len(calls) == 1
    parents = _parents(CAPTURE)
    loops = [n for n in _enclosing(calls[0], parents, ast.For)]
    assert len(loops) == 1 and _text(loops[0].iter) == "self.cams.items()"


def test_the_probe_writes_only_under_its_check():
    parents = _parents(CAPTURE)
    saves = [n for n in ast.walk(CAPTURE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "save"]
    assert len(saves) == 1
    guards = _enclosing(saves[0], parents, ast.If)
    assert any(_is_test(g.test, CAPTURE_TEST) for g in guards)


def test_capture_matches_the_base_outside_the_read_and_the_probe_block():
    now = [ast.dump(s) for s in _capture_without_read_and_probe(CAPTURE)]
    base = [ast.dump(s) for s in _capture_without_read_and_probe(CAPTURE_BASE)]
    assert now == base


def test_main_matches_the_base_outside_the_probe_additions():
    assert ast.dump(_main_without_probe(MAIN)) == ast.dump(MAIN_BASE)
