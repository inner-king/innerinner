# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual background swap: the spec reader, the pose, the scene swap, and the splat's USDZ structure.

The module under test is ``simfoundry/utils/splat_background.py``. Most tests are synthetic and run in any env with
numpy and scipy. The tests that read the real files (the P3 scene JSON, the align matrix, the 58 MB USDZ) skip when a
file is missing. The USDZ structure test needs ``pxr``, which the simfoundry env does not have, so it skips there and
runs in simfoundry-editor. The ``edit_scene`` test imports ``scripts/cutting/tofu_og_cut.py``, which does not import
OmniGibson at module level, so it runs in the simfoundry env.
"""

import ast
import copy
import dataclasses
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from simfoundry.utils import splat_background as sb

ROOT = Path(__file__).resolve().parents[1]
# The tests that need the real files read their paths from these variables and skip when one is unset, so no machine path
# is written here. The variables are read only: a test opens the file and never writes it.
SCENE_ENV, ALIGN_ENV = "SIMFOUNDRY_TEST_SCENE_JSON", "SIMFOUNDRY_TEST_ALIGN_JSON"
PD_USDZ_ENV, PD_PLY_ENV = "SIMFOUNDRY_TEST_PD_USDZ", "SIMFOUNDRY_TEST_PD_PLY"


def _path_from_env(name):
    value = os.environ.get(name, "")
    return Path(value) if value else None


SCENE_JSON = _path_from_env(SCENE_ENV)
ALIGN_JSON = _path_from_env(ALIGN_ENV)
PD_USDZ = _path_from_env(PD_USDZ_ENV)
PD_PLY = _path_from_env(PD_PLY_ENV)
GS = "gs_background"
RECORD_KEYS = {"name", "usdz", "md5", "removed", "pos", "ori_xyzw", "scale", "dz_m", "floor_plane_visible", "kit_settings"}


def _present(path):
    return path is not None and path.exists()


def _skip_reason(env):
    return f"set {env} to an existing file to run this test (it is read only)"


def _need(path, env):
    if not _present(path):
        pytest.skip(_skip_reason(env))


def _rigid(rotvec=(0.3, -0.2, 0.5), t=(0.1, 0.2, 0.3), scale=1.0):
    """A similarity transform with a known rotation, translation and scale."""
    m = np.eye(4)
    m[:3, :3] = scale * Rotation.from_rotvec(rotvec).as_matrix()
    m[:3, 3] = t
    return m


def _sheared():
    m = _rigid()
    m[:3, :3] = m[:3, :3] + 0.05 * np.triu(np.ones((3, 3)), 1)
    return m


def _synthetic_scene():
    """A small scene with the names the swap touches: robot0, the board, a bystander, mesh_background_0 and iter_7."""

    def dataset_object(name, category):
        return {
            "class_module": "omnigibson.objects.usd_object",
            "class_name": "DatasetObject",
            "args": {"name": name, "category": category, "model": "m", "dataset_name": "real2sim-assets", "scale": [1.0, 1.0, 1.0]},
        }

    init = {
        "robot0": {"class_module": "omnigibson.robots.franka", "class_name": "FrankaPanda", "args": {"name": "robot0"}},
        "iter_1": dataset_object("iter_1", "cutting_board"),
        "iter_3": dataset_object("iter_3", "onion"),
        "mesh_background_0": {
            "class_module": "omnigibson.objects.usd_object",
            "class_name": "USDObject",
            "args": {
                "name": "mesh_background_0",
                "usd_path": "background/mesh_background_0.usd",
                "category": "mesh_background",
                "fixed_base": True,
                "visual_only": True,
                "expected_file_hash": "98ea90c59155fa834fd340d64ebc19b2",
            },
        },
        "iter_7": dataset_object("iter_7", "blue_sink_stopper"),
    }
    registry = {
        name: {
            "is_asleep": False,
            "non_kin": {},
            "root_link": {"pos": [0.1 * index, 0.0, 0.0], "ori": [0.0, 0.0, 0.0, 1.0], "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]},
        }
        for index, name in enumerate(init)
    }
    return {
        "versions": {"omnigibson": "test"},
        "metadata": {"name": "synthetic"},
        "objects_info": {"init_info": init},
        "state": {"registry": {"object_registry": registry}},
    }


def _spec_file(folder, body, name="spec.json"):
    path = Path(folder) / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _body(**changes):
    body = {"schema": 1, "name": "PD", "usdz": "PD_bg.usdz", "matrix": _rigid().tolist()}
    body.update(changes)
    return body


@pytest.fixture
def usdz_file(tmp_path):
    path = tmp_path / "PD_bg.usdz"
    path.write_bytes(b"a file of known bytes, standing in for the splat")
    return path


@functools.lru_cache(maxsize=None)
def _pd_md5():
    """The digest of the real 58 MB USDZ, computed once for all the tests that need it."""
    return hashlib.md5(PD_USDZ.read_bytes()).hexdigest()


# ---------------------------------------------------------------- pose (CPU-2)


@pytest.mark.skipif(not _present(ALIGN_JSON), reason=_skip_reason(ALIGN_ENV))
def test_pose_on_align_v4b_matches_scipy_including_sign():
    m = np.array(json.loads(ALIGN_JSON.read_text())["M_da3_to_scene"], float)
    pos, quat, scale = sb.pose_from_similarity(m)
    ref = Rotation.from_matrix(m[:3, :3]).as_quat()
    assert scale == pytest.approx(1.0, abs=1e-9)
    assert np.array_equal(pos, m[:3, 3])  # the translation column, copied exactly
    assert abs(np.dot(quat, ref)) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(quat, ref, atol=1e-12)


def test_dz_moves_only_the_z_of_the_position():
    m = _rigid()
    pos0, quat0, scale0 = sb.pose_from_similarity(m)
    pos1, quat1, scale1 = sb.pose_from_similarity(m, dz_m=0.004)
    assert pos1[2] == pytest.approx(pos0[2] + 0.004, abs=1e-15)
    assert pos1[0] == pos0[0] and pos1[1] == pos0[1]
    assert np.array_equal(quat0, quat1) and scale0 == scale1
    assert np.array_equal(m[:3, 3], _rigid()[:3, 3])  # the matrix itself is not changed


def test_similarity_with_scale_1_3_gives_the_scale_and_the_unit_rotation():
    rot = Rotation.from_rotvec([0.1, 0.7, -0.4]).as_matrix()
    m = np.eye(4)
    m[:3, :3] = 1.3 * rot
    m[:3, 3] = [0.5, -0.25, 0.75]
    pos, quat, scale = sb.pose_from_similarity(m)
    assert scale == pytest.approx(1.3, abs=1e-12)
    assert np.allclose(pos, [0.5, -0.25, 0.75], atol=0, rtol=0)
    assert np.allclose(Rotation.from_quat(quat).as_matrix(), rot, atol=1e-12)
    assert np.allclose(quat, Rotation.from_matrix(rot).as_quat(), atol=1e-12)


def test_quaternion_equals_scipy_on_random_rotations():
    for seed in range(500):
        rot = Rotation.random(random_state=seed).as_matrix()
        m = np.eye(4)
        m[:3, :3] = rot
        _, quat, _ = sb.pose_from_similarity(m)
        ref = Rotation.from_matrix(rot).as_quat()
        assert np.allclose(quat, ref, atol=1e-12), seed


@pytest.mark.parametrize(
    "rotvec",
    [
        np.pi * 0.999 * np.array([1.0, 0.0, 0.0]),  # near 180 degrees: each diagonal branch of the algorithm
        np.pi * 0.999 * np.array([0.0, 1.0, 0.0]),
        np.pi * 0.999 * np.array([0.0, 0.0, 1.0]),
        np.zeros(3),  # the identity: the trace branch
        np.array([0.2, 0.1, 0.05]),
    ],
)
def test_quaternion_on_every_branch_matches_scipy(rotvec):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    _, quat, _ = sb.pose_from_similarity(m)
    ref = Rotation.from_matrix(m[:3, :3]).as_quat()
    assert np.allclose(quat, ref, atol=1e-12)


def test_pose_refuses_a_matrix_that_is_not_4x4():
    with pytest.raises(sb.SplatBackgroundError):
        sb.pose_from_similarity(np.eye(3, 4))


def test_pose_outputs_are_float64_and_the_scale_is_a_python_float():
    pos, quat, scale = sb.pose_from_similarity(_rigid())
    assert pos.dtype == np.float64 and pos.shape == (3,)
    assert quat.dtype == np.float64 and quat.shape == (4,)
    assert type(scale) is float


# ---------------------------------------------------- spec refusals (CPU-3)


def _set_nan(body):
    body["matrix"][0][0] = float("nan")


def _bad_bottom_row(body):
    body["matrix"][3] = [0.0, 0.0, 0.0, 2.0]


def _ragged(body):
    body["matrix"] = [[1, 0, 0, 0], [0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _bool_entry(body):
    body["matrix"][0][3] = True  # a translation entry: a bool in the rotation would fail the orthogonality test instead


REFUSALS = {
    "unknown_key": lambda b: b.update(dz=0.001),
    "schema_2": lambda b: b.update(schema=2),
    "schema_true": lambda b: b.update(schema=True),
    "schema_string": lambda b: b.update(schema="1"),
    "no_schema": lambda b: b.pop("schema"),
    "no_name": lambda b: b.pop("name"),
    "empty_name": lambda b: b.update(name=""),
    "no_usdz": lambda b: b.pop("usdz"),
    "usdz_not_usdz": lambda b: b.update(usdz="PD_bg.ply"),
    "usdz_missing_file": lambda b: b.update(usdz="missing.usdz"),
    "neither_matrix_nor_pose": lambda b: b.pop("matrix"),
    "both_matrix_and_pose": lambda b: b.update(pose_json="align.json"),
    "matrix_3x4": lambda b: b.update(matrix=_rigid().tolist()[:3]),
    "matrix_nan": _set_nan,
    "matrix_nan_translation": lambda b: b["matrix"][0].__setitem__(3, float("nan")),
    "matrix_bottom_row": _bad_bottom_row,
    "matrix_reflection": lambda b: b.update(matrix=(np.diag([-1.0, 1.0, 1.0, 1.0]) @ _rigid()).tolist()),
    "matrix_shear": lambda b: b.update(matrix=_sheared().tolist()),
    "scale_too_big": lambda b: b.update(matrix=_rigid(scale=20.0).tolist()),
    "scale_too_small": lambda b: b.update(matrix=_rigid(scale=0.05).tolist()),
    "matrix_bool_entry": _bool_entry,
    "matrix_ragged": _ragged,
    "dz_too_big": lambda b: b.update(dz_m=0.021),
    "dz_too_small": lambda b: b.update(dz_m=-0.021),
    "dz_bool": lambda b: b.update(dz_m=True),
    "dz_string": lambda b: b.update(dz_m="0.001"),
    "remove_repeats": lambda b: b.update(remove=["mesh_background_0", "mesh_background_0"]),
    "remove_empty_name": lambda b: b.update(remove=["mesh_background_0", ""]),
    "remove_gs": lambda b: b.update(remove=["gs_background"]),
    "remove_not_a_list": lambda b: b.update(remove="mesh_background_0"),
    "remove_non_string": lambda b: b.update(remove=[3]),
    "kit_key_without_slash": lambda b: b.update(kit_settings={"rtx/post": True}),
    "kit_null_value": lambda b: b.update(kit_settings={"/rtx/a": None}),
    "kit_list_value": lambda b: b.update(kit_settings={"/rtx/a": [1, 2]}),
    "kit_nan_value": lambda b: b.update(kit_settings={"/rtx/a": float("nan")}),
    "kit_not_an_object": lambda b: b.update(kit_settings=["/rtx/a"]),
    "color_not_null": lambda b: b.update(color={"mode": "x"}),
    "floor_visible_string": lambda b: b.update(floor_plane_visible="false"),
    "floor_visible_int": lambda b: b.update(floor_plane_visible=0),
    "pose_json_missing_file": lambda b: (b.pop("matrix"), b.update(pose_json="nope.json")),
    "pose_json_without_M": lambda b: (b.pop("matrix"), b.update(pose_json="no_matrix.json")),
    "pose_json_bad_matrix": lambda b: (b.pop("matrix"), b.update(pose_json="bad_matrix.json")),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_spec_refusals(tmp_path, usdz_file, name):
    (tmp_path / "PD_bg.ply").write_bytes(b"exists, so only the suffix can refuse it")
    (tmp_path / "no_matrix.json").write_text(json.dumps({"other": 1}))
    (tmp_path / "bad_matrix.json").write_text(json.dumps({"M_da3_to_scene": [[1, 0], [0, 1]]}))
    body = _body()
    REFUSALS[name](body)
    with pytest.raises(sb.SplatBackgroundError):
        sb.load_visual_background(_spec_file(tmp_path, body))


def test_a_scaled_spec_loads_and_the_splat_takes_the_scale(tmp_path, usdz_file):
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(matrix=_rigid(scale=1.3).tolist(), remove=["iter_7"])))
    assert spec.matrix[0, 0] == pytest.approx(1.3 * Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()[0, 0])
    scene = _synthetic_scene()
    record = sb.swap_visual_background(scene, spec)
    assert record["scale"] == pytest.approx(1.3, abs=1e-12)
    assert scene["objects_info"]["init_info"][GS]["args"]["scale"] == pytest.approx([1.3, 1.3, 1.3], abs=1e-12)


def test_a_reflection_is_refused_for_its_negative_determinant(tmp_path, usdz_file):
    # Without the determinant test the scale test would refuse it too, with a message about the scale that hides the cause.
    reflected = (np.diag([-1.0, 1.0, 1.0, 1.0]) @ _rigid()).tolist()
    with pytest.raises(sb.SplatBackgroundError, match="determinant"):
        sb.load_visual_background(_spec_file(tmp_path, _body(matrix=reflected)))


def test_unknown_key_is_named_in_the_message(tmp_path, usdz_file):
    with pytest.raises(sb.SplatBackgroundError, match="'dz'"):
        sb.load_visual_background(_spec_file(tmp_path, _body(dz=0.001)))


def test_a_spec_that_is_not_json_is_refused(tmp_path, usdz_file):
    path = tmp_path / "spec.json"
    path.write_text("{ not json")
    with pytest.raises(sb.SplatBackgroundError, match="cannot read"):
        sb.load_visual_background(path)


def test_a_spec_that_is_not_an_object_is_refused(tmp_path, usdz_file):
    with pytest.raises(sb.SplatBackgroundError, match="JSON object"):
        sb.load_visual_background(_spec_file(tmp_path, [1, 2, 3]))


def test_refusal_is_a_value_error():
    assert issubclass(sb.SplatBackgroundError, ValueError)


# ------------------------------------------------- the root, the guard and the missing-file message


GUARD = {"mesh_background_0": "98ea90c59155fa834fd340d64ebc19b2", "iter_7": "db7a0f67450bd93abbc1c04f1a9b4306"}


def _root_with_files(root, usdz_name="PD_slab.usdz", pose_name="align.json"):
    root.mkdir(parents=True, exist_ok=True)
    (root / usdz_name).write_bytes(b"a splat file, read for its md5 only")
    (root / pose_name).write_text(json.dumps({"M_da3_to_scene": _rigid().tolist()}))


def test_a_root_gives_the_relative_names_and_the_spec_folder_is_not_searched(tmp_path):
    spec_folder, root = tmp_path / "spec_here", tmp_path / "root_there"
    spec_folder.mkdir()
    _root_with_files(root)
    path = _spec_file(spec_folder, {"schema": 1, "name": "P1", "usdz": "PD_slab.usdz", "pose_json": "align.json"})
    spec = sb.load_visual_background(path, root=root)
    assert spec.usdz == root / "PD_slab.usdz"
    assert spec.pose_json == root / "align.json"


def test_without_a_root_the_names_are_read_from_the_spec_folder(tmp_path):
    _root_with_files(tmp_path)
    path = _spec_file(tmp_path, {"schema": 1, "name": "P1", "usdz": "PD_slab.usdz", "pose_json": "align.json"})
    assert sb.load_visual_background(path).usdz == tmp_path / "PD_slab.usdz"


def test_missing_files_are_named_together_with_the_folder_searched(tmp_path):
    root = tmp_path / "empty_root"
    root.mkdir()
    path = _spec_file(tmp_path, {"schema": 1, "name": "P1", "usdz": "PD_slab.usdz", "pose_json": "align.json"})
    with pytest.raises(sb.MissingAssetError) as caught:
        sb.load_visual_background(path, root=root)
    assert caught.value.names == ("PD_slab.usdz", "align.json")
    assert caught.value.folder == root
    assert "PD_slab.usdz, align.json" in str(caught.value)


def test_the_missing_message_gives_both_ways_out_and_names_the_variable_only_for_the_packaged_spec(tmp_path):
    error = sb.MissingAssetError(["PD_slab.usdz", "align.json"], tmp_path)
    packaged = sb.missing_assets_message(error, packaged=True)
    for text in ("PD_slab.usdz", "align.json", str(tmp_path), "--visual-background-root DIR",
                 "SIMFOUNDRY_VISUAL_BACKGROUND_ROOT", "--visual-background none", "docs/BACKGROUND_SWAP.md"):
        assert text in packaged
    own = sb.missing_assets_message(error, packaged=False)
    assert "--visual-background-root DIR" in own and "--visual-background none" in own
    assert "SIMFOUNDRY_VISUAL_BACKGROUND_ROOT" not in own


def test_the_packaged_root_is_the_option_then_the_environment_then_the_checkout_default(tmp_path):
    env = {sb.VISUAL_BACKGROUND_ROOT_ENV: str(tmp_path / "from_env")}
    default = sb._CHECKOUT_DIR / "Data" / "visual_background"
    assert sb.packaged_visual_background_root(str(tmp_path / "from_option"), environ=env) == tmp_path / "from_option"
    assert sb.packaged_visual_background_root(None, environ=env) == tmp_path / "from_env"
    assert sb.packaged_visual_background_root(None, environ={}) == default


def test_an_empty_environment_variable_is_no_root():
    default = sb._CHECKOUT_DIR / "Data" / "visual_background"
    assert sb.packaged_visual_background_root(None, environ={sb.VISUAL_BACKGROUND_ROOT_ENV: ""}) == default


def test_the_checkout_default_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert sb._CHECKOUT_DIR == Path(sb.__file__).resolve().parents[2]
    assert sb.packaged_visual_background_root(None, environ={}) == sb._CHECKOUT_DIR / "Data" / "visual_background"


def test_the_environment_variable_names_for_the_real_file_tests_are_fixed():
    # the names are the interface the run is made with, so a rename is a change to the tests' documentation too
    assert (SCENE_ENV, ALIGN_ENV, PD_USDZ_ENV, PD_PLY_ENV) == (
        "SIMFOUNDRY_TEST_SCENE_JSON", "SIMFOUNDRY_TEST_ALIGN_JSON", "SIMFOUNDRY_TEST_PD_USDZ", "SIMFOUNDRY_TEST_PD_PLY")
    assert "SIMFOUNDRY_TEST_SCENE_JSON" in _skip_reason(SCENE_ENV)


def _guarded_scene():
    scene = _synthetic_scene()
    scene["objects_info"]["init_info"]["iter_7"]["args"]["expected_file_hash"] = GUARD["iter_7"]
    return scene


def _guarded_spec(tmp_path, usdz_file, requires=GUARD):
    return sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"], scene_requires=requires)))


def test_the_guard_passes_when_every_recorded_hash_matches(tmp_path, usdz_file):
    spec = _guarded_spec(tmp_path, usdz_file)
    assert spec.scene_requires == GUARD
    sb.swap_visual_background(_guarded_scene(), spec)


def test_the_guard_refuses_a_mismatch_and_names_the_scene_the_object_and_both_hashes(tmp_path, usdz_file):
    spec = _guarded_spec(tmp_path, usdz_file)
    scene = _guarded_scene()
    scene["objects_info"]["init_info"]["iter_7"]["args"]["expected_file_hash"] = "0" * 32
    with pytest.raises(sb.SplatBackgroundError) as caught:
        sb.swap_visual_background(scene, spec, label="scene Data/tofu_chefd/x.json")
    message = str(caught.value)
    for text in ("scene Data/tofu_chefd/x.json", "'iter_7'", "0" * 32, GUARD["iter_7"], "--visual-background none"):
        assert text in message


def test_the_guard_refuses_a_missing_object(tmp_path, usdz_file):
    spec = _guarded_spec(tmp_path, usdz_file)
    scene = _guarded_scene()
    del scene["objects_info"]["init_info"]["iter_7"]
    with pytest.raises(sb.SplatBackgroundError, match="no object 'iter_7'"):
        sb.swap_visual_background(scene, spec)


def test_a_scene_that_records_no_hash_is_cannot_verify_not_a_pass(tmp_path, usdz_file):
    spec = _guarded_spec(tmp_path, usdz_file)
    scene = _guarded_scene()
    del scene["objects_info"]["init_info"]["iter_7"]["args"]["expected_file_hash"]
    with pytest.raises(sb.SplatBackgroundError, match="cannot verify"):
        sb.swap_visual_background(scene, spec)


def test_the_guard_refuses_before_anything_is_changed(tmp_path, usdz_file):
    spec = _guarded_spec(tmp_path, usdz_file)
    scene = _guarded_scene()
    scene["objects_info"]["init_info"]["iter_7"]["args"]["expected_file_hash"] = "0" * 32
    before = copy.deepcopy(scene)
    with pytest.raises(sb.SplatBackgroundError):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_a_spec_without_the_key_is_not_guarded_and_loads_as_before(tmp_path, usdz_file):
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"])))
    assert spec.scene_requires is None
    scene = _guarded_scene()
    scene["objects_info"]["init_info"]["iter_7"]["args"]["expected_file_hash"] = "0" * 32
    sb.swap_visual_background(scene, spec)  # no guard, so the swap runs


@pytest.mark.parametrize("requires", [{}, {"iter_7": "DB7A0F67450BD93ABBC1C04F1A9B4306"}, {"iter_7": "db7a"},
                                      {"iter_7": 5}, {"": GUARD["iter_7"]}, None, GUARD["iter_7"]])
def test_a_bad_scene_requires_is_refused_at_load(tmp_path, usdz_file, requires):
    body = _body(remove=["iter_7"])
    body["scene_requires"] = requires
    with pytest.raises(sb.SplatBackgroundError, match="scene_requires"):
        sb.load_visual_background(_spec_file(tmp_path, body))


def test_the_guard_adds_nothing_to_the_record(tmp_path, usdz_file):
    record = sb.swap_visual_background(_guarded_scene(), _guarded_spec(tmp_path, usdz_file))
    assert set(record) == RECORD_KEYS


# ------------------------------------------------- the packaged spec


def test_the_packaged_spec_is_in_the_package_and_is_found_from_the_module():
    expected = Path(sb.__file__).resolve().parents[1] / "configs" / "visual_background" / "p1.json"
    assert sb.packaged_spec_path() == expected
    assert expected.is_file()


def test_the_packaged_spec_has_relative_names_and_no_machine_path():
    text = sb.packaged_spec_path().read_text(encoding="utf-8")
    spec = json.loads(text)
    assert spec["usdz"] == "PD_slab4_bg.usdz"
    assert spec["pose_json"] == "align_v4b.json"
    assert f"{os.sep}home{os.sep}" not in text  # no home-directory component, however the spec spells it

    def strings(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    assert not [s for s in strings(spec) if s.startswith("/")]


def test_the_packaged_spec_pins_its_reflectance_its_guard_and_its_removals():
    spec = json.loads(sb.packaged_spec_path().read_text(encoding="utf-8"))
    assert spec["schema"] == 1 and spec["name"] == "P1"
    assert spec["counter_patch"]["rgb"] == [0.1875, 0.1825, 0.185]
    assert spec["scene_requires"] == GUARD
    assert spec["remove"] == ["mesh_background_0", "iter_7"]


def test_the_packaged_spec_loads_with_stand_in_files_from_a_root(tmp_path):
    root = tmp_path / "root"
    _root_with_files(root, usdz_name="PD_slab4_bg.usdz", pose_name="align_v4b.json")
    spec = sb.load_visual_background(sb.packaged_spec_path(), root=root)
    assert spec.name == "P1"
    assert spec.usdz == root / "PD_slab4_bg.usdz"
    assert spec.pose_json == root / "align_v4b.json"
    assert spec.scene_requires == GUARD
    assert spec.remove == ("mesh_background_0", "iter_7")
    assert spec.counter_patch.rgb == (0.1875, 0.1825, 0.185)
    assert spec.ccm is not None


def test_without_its_files_the_packaged_spec_names_both_and_the_folder(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(sb.MissingAssetError) as caught:
        sb.load_visual_background(sb.packaged_spec_path(), root=root)
    assert caught.value.names == ("PD_slab4_bg.usdz", "align_v4b.json")
    assert caught.value.folder == root


def test_the_packaged_spec_path_is_the_same_from_another_working_directory(tmp_path):
    # the checkout is put on the path, as an editable install would, and the child starts in an empty folder
    code = "from simfoundry.utils import splat_background as sb; print(sb.packaged_spec_path())"
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert Path(done.stdout.strip()) == sb.packaged_spec_path()


def test_a_build_ships_the_packaged_spec(tmp_path):
    # built from a copy of setup.py and the package's own files, so the checkout is not written
    shutil.copy(ROOT / "setup.py", tmp_path / "setup.py")
    config = tmp_path / "simfoundry" / "configs" / "visual_background"
    config.mkdir(parents=True)
    shutil.copy(ROOT / "simfoundry" / "__init__.py", tmp_path / "simfoundry" / "__init__.py")
    shutil.copy(sb.packaged_spec_path(), config / "p1.json")
    done = subprocess.run([sys.executable, "setup.py", "-q", "build_py", "--build-lib", "build"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    shipped = tmp_path / "build" / "simfoundry" / "configs" / "visual_background" / "p1.json"
    assert shipped.read_bytes() == sb.packaged_spec_path().read_bytes()


def test_the_packaged_guard_matches_the_real_scene_json():
    _need(SCENE_JSON, SCENE_ENV)
    requires = json.loads(sb.packaged_spec_path().read_text(encoding="utf-8"))["scene_requires"]
    scene = json.loads(SCENE_JSON.read_text(encoding="utf-8"))
    sb.check_scene_requires(scene, requires, label="the scene JSON")


# ------------------------------------------------- spec reading and defaults


def test_defaults_are_the_documented_ones(tmp_path, usdz_file):
    spec = sb.load_visual_background(_spec_file(tmp_path, _body()))
    assert spec.dz_m == 0.0
    assert spec.floor_plane_visible is False
    assert spec.remove == ("mesh_background_0",)
    assert spec.kit_settings == {}
    assert spec.color is None
    assert spec.pose_json is None
    assert spec.usdz == usdz_file
    assert np.array_equal(spec.matrix, _rigid())
    assert spec.matrix.dtype == np.float64


def test_relative_paths_are_taken_from_the_spec_folder(tmp_path):
    folder, other = tmp_path / "a", tmp_path / "b"
    folder.mkdir()
    other.mkdir()
    (other / "PD_bg.usdz").write_bytes(b"x")
    (other / "align.json").write_text(json.dumps({"M_da3_to_scene": _rigid().tolist()}))
    spec = sb.load_visual_background(
        _spec_file(folder, {"schema": 1, "name": "PD", "usdz": "../b/PD_bg.usdz", "pose_json": "../b/align.json"})
    )
    assert spec.usdz == folder / "../b/PD_bg.usdz"  # as spelled: the .. is left for the OS
    assert spec.pose_json == folder / "../b/align.json"
    assert spec.usdz.resolve() == (other / "PD_bg.usdz").resolve()
    assert np.array_equal(spec.matrix, _rigid())


def test_absolute_paths_are_kept(tmp_path, usdz_file):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    spec = sb.load_visual_background(
        _spec_file(elsewhere, _body(usdz=str(usdz_file)))
    )
    assert spec.usdz == usdz_file
    assert spec.usdz.is_absolute()


def test_symlinks_keep_the_name_the_spec_gives(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    real_usdz = store / "PD_bg.usdz"
    real_usdz.write_bytes(b"splat bytes")
    real_align = store / "align.json"
    real_align.write_text(json.dumps({"M_da3_to_scene": _rigid().tolist()}))
    links = tmp_path / "links"
    links.mkdir()
    link_usdz = links / "PD_link.usdz"
    link_usdz.symlink_to(real_usdz)
    link_align = links / "align_link.json"
    link_align.symlink_to(real_align)
    specs = tmp_path / "specs"
    specs.mkdir()
    body = {"schema": 1, "name": "PD", "usdz": str(link_usdz), "pose_json": str(link_align), "remove": ["iter_7"]}
    spec = sb.load_visual_background(_spec_file(specs, body))
    assert spec.usdz == link_usdz  # the link's name, not the file it points to
    assert spec.pose_json == link_align
    scene = _synthetic_scene()
    record = sb.swap_visual_background(scene, spec)
    assert record["usdz"] == str(link_usdz)
    assert scene["objects_info"]["init_info"][GS]["args"]["usd_path"] == str(link_usdz)
    assert record["md5"] == hashlib.md5(real_usdz.read_bytes()).hexdigest()  # the content behind the link


def test_a_relative_symlink_keeps_its_spelling(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "PD_bg.usdz").write_bytes(b"splat bytes")
    links = tmp_path / "links"
    links.mkdir()
    (links / "PD_link.usdz").symlink_to(store / "PD_bg.usdz")
    specs = tmp_path / "specs"
    specs.mkdir()
    spec = sb.load_visual_background(_spec_file(specs, _body(usdz="../links/PD_link.usdz")))
    assert spec.usdz == specs / "../links/PD_link.usdz"
    assert spec.usdz.is_file()
    assert spec.usdz.resolve() == (store / "PD_bg.usdz").resolve()


def test_dotdot_after_a_symlinked_folder_is_left_for_the_os(tmp_path):
    # folder_link -> store/sub, so from inside it .. is store. A text collapse would give tmp_path/PD_bg.usdz instead.
    store = tmp_path / "store"
    (store / "sub").mkdir(parents=True)
    (store / "PD_bg.usdz").write_bytes(b"splat bytes")
    folder_link = tmp_path / "folder_link"
    folder_link.symlink_to(store / "sub", target_is_directory=True)
    spec = sb.load_visual_background(_spec_file(folder_link, _body(usdz="../PD_bg.usdz")))
    assert spec.usdz.is_file()
    assert spec.usdz.resolve() == (store / "PD_bg.usdz").resolve()


def test_tilde_is_not_expanded(tmp_path, usdz_file):
    with pytest.raises(sb.SplatBackgroundError, match="not found"):
        sb.load_visual_background(_spec_file(tmp_path, _body(usdz="~/PD_bg.usdz")))


def test_pose_json_is_read_and_recorded_as_an_absolute_path(tmp_path, usdz_file):
    (tmp_path / "align.json").write_text(json.dumps({"M_da3_to_scene": _rigid().tolist(), "scale": 1.0}))
    body = _body()
    body.pop("matrix")
    body["pose_json"] = "align.json"
    spec = sb.load_visual_background(_spec_file(tmp_path, body))
    assert spec.pose_json == tmp_path / "align.json"
    assert np.array_equal(spec.matrix, _rigid())


def test_inline_matrix_leaves_pose_json_none(tmp_path, usdz_file):
    assert sb.load_visual_background(_spec_file(tmp_path, _body())).pose_json is None


def test_kit_settings_and_remove_are_kept_as_given(tmp_path, usdz_file):
    kit = {"/rtx/post/registeredCompositing/invertToneMap": True, "/rtx/x": 2.5, "/rtx/y": "s", "/rtx/z": 3}
    spec = sb.load_visual_background(
        _spec_file(tmp_path, _body(kit_settings=kit, remove=["mesh_background_0", "iter_7"], floor_plane_visible=True, dz_m=-0.01))
    )
    assert spec.kit_settings == kit
    assert spec.remove == ("mesh_background_0", "iter_7")
    assert spec.floor_plane_visible is True
    assert spec.dz_m == pytest.approx(-0.01)


def test_pose_json_refusal_names_the_file(tmp_path, usdz_file):
    (tmp_path / "no_matrix.json").write_text(json.dumps({"other": 1}))
    body = _body()
    body.pop("matrix")
    body["pose_json"] = "no_matrix.json"
    with pytest.raises(sb.SplatBackgroundError, match="no_matrix.json"):
        sb.load_visual_background(_spec_file(tmp_path, body))


# ------------------------------------------------------------- scene swap


def test_swap_refuses_sections_that_are_not_objects_and_keeps_the_scene(tmp_path, usdz_file):
    scene = _synthetic_scene()
    scene["objects_info"]["init_info"] = ["robot0", "iter_7"]
    before = copy.deepcopy(scene)
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"])))
    with pytest.raises(sb.SplatBackgroundError, match="must be objects"):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_swap_removes_the_named_objects_from_both_places_and_adds_the_splat(tmp_path, usdz_file):
    scene = _synthetic_scene()
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["mesh_background_0", "iter_7"])))
    sb.swap_visual_background(scene, spec)
    for section in (scene["objects_info"]["init_info"], scene["state"]["registry"]["object_registry"]):
        assert "mesh_background_0" not in section and "iter_7" not in section
        assert GS in section


def test_swap_leaves_every_other_entry_as_it_was(tmp_path, usdz_file):
    scene = _synthetic_scene()
    before = copy.deepcopy(scene)
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["mesh_background_0", "iter_7"])))
    sb.swap_visual_background(scene, spec)
    for key in ("versions", "metadata"):
        assert scene[key] == before[key]
    for name in ("robot0", "iter_1", "iter_3"):
        assert scene["objects_info"]["init_info"][name] == before["objects_info"]["init_info"][name]
        assert scene["state"]["registry"]["object_registry"][name] == before["state"]["registry"]["object_registry"][name]


def test_swap_on_the_p3_scene_changes_only_the_background_entries(tmp_path, usdz_file):
    _need(SCENE_JSON, SCENE_ENV)
    scene = json.loads(SCENE_JSON.read_text())
    before = copy.deepcopy(scene)
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["mesh_background_0", "iter_7"])))
    sb.swap_visual_background(scene, spec)
    for section in ("init_info", "object_registry"):
        keep = (lambda d: d["objects_info"]["init_info"]) if section == "init_info" else (lambda d: d["state"]["registry"]["object_registry"])
        after_part = {k: v for k, v in keep(scene).items() if k != GS}
        before_part = {k: v for k, v in keep(before).items() if k not in ("mesh_background_0", "iter_7")}
        assert after_part == before_part
    assert {k: v for k, v in scene.items() if k not in ("objects_info", "state")} == {
        k: v for k, v in before.items() if k not in ("objects_info", "state")
    }


def test_golden_on_the_p3_scene_keeps_the_key_order_and_writes_the_expected_values(tmp_path, usdz_file):
    _need(SCENE_JSON, SCENE_ENV)
    _need(ALIGN_JSON, ALIGN_ENV)
    _need(PD_USDZ, PD_USDZ_ENV)
    scene = json.loads(SCENE_JSON.read_text())
    before = copy.deepcopy(scene)
    remove = ["mesh_background_0", "iter_7"]
    spec = sb.load_visual_background(
        _spec_file(tmp_path, {"schema": 1, "name": "PD", "usdz": str(PD_USDZ), "pose_json": str(ALIGN_JSON), "remove": remove})
    )
    sb.swap_visual_background(scene, spec)
    # The expected values are written out here from the design, not taken from the module's own helpers.
    m = np.array(json.loads(ALIGN_JSON.read_text())["M_da3_to_scene"], float)
    init_before = before["objects_info"]["init_info"]
    reg_before = before["state"]["registry"]["object_registry"]
    init_keys = [k for k in init_before if k not in remove] + [GS]
    reg_keys = [k for k in reg_before if k not in remove] + [GS]
    assert list(scene["objects_info"]["init_info"]) == init_keys
    assert list(scene["state"]["registry"]["object_registry"]) == reg_keys
    entry = scene["objects_info"]["init_info"][GS]
    assert entry["class_module"] == "omnigibson.objects.usd_object" and entry["class_name"] == "USDObject"
    assert list(entry["args"]) == ["name", "usd_path", "scale", "fixed_base", "visual_only", "expected_file_hash"]
    assert entry["args"]["usd_path"] == str(PD_USDZ)
    assert entry["args"]["expected_file_hash"] == _pd_md5()
    assert entry["args"]["scale"] == pytest.approx([float(np.cbrt(np.linalg.det(m[:3, :3])))] * 3, abs=1e-12)
    state = scene["state"]["registry"]["object_registry"][GS]
    assert state["root_link"]["pos"] == pytest.approx(m[:3, 3].tolist(), abs=1e-12)
    assert np.allclose(state["root_link"]["ori"], Rotation.from_matrix(m[:3, :3]).as_quat(), atol=1e-12)
    assert state["root_link"]["lin_vel"] == [0.0, 0.0, 0.0] and state["root_link"]["ang_vel"] == [0.0, 0.0, 0.0]
    for key in init_keys[:-1]:
        assert scene["objects_info"]["init_info"][key] == init_before[key]
    for key in reg_keys[:-1]:
        assert scene["state"]["registry"]["object_registry"][key] == reg_before[key]


def test_swap_refuses_a_name_that_is_not_in_the_scene_and_leaves_it_unchanged(tmp_path, usdz_file):
    scene = _synthetic_scene()
    before = copy.deepcopy(scene)
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["mesh_background_0", "not_here"])))
    with pytest.raises(sb.SplatBackgroundError, match="not_here"):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_swap_refuses_a_name_present_in_only_one_of_the_two_sections(tmp_path, usdz_file):
    scene = _synthetic_scene()
    del scene["state"]["registry"]["object_registry"]["iter_7"]
    before = copy.deepcopy(scene)
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"])))
    with pytest.raises(sb.SplatBackgroundError, match="iter_7"):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_swap_refuses_a_scene_that_already_has_the_splat(tmp_path, usdz_file):
    scene = _synthetic_scene()
    scene["objects_info"]["init_info"][GS] = {"args": {}}
    before = copy.deepcopy(scene)
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"])))
    with pytest.raises(sb.SplatBackgroundError, match="already"):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_a_second_swap_is_refused(tmp_path, usdz_file):
    scene = _synthetic_scene()
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"])))
    sb.swap_visual_background(scene, spec)
    once = copy.deepcopy(scene)
    with pytest.raises(sb.SplatBackgroundError):
        sb.swap_visual_background(scene, spec)
    assert scene == once


def test_a_scene_without_the_sections_is_refused(tmp_path, usdz_file):
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"])))
    scene = {"state": {}}
    with pytest.raises(sb.SplatBackgroundError, match="init_info"):
        sb.swap_visual_background(scene, spec)
    assert scene == {"state": {}}


def test_a_spec_built_in_code_with_a_bad_matrix_is_refused_before_any_change(tmp_path, usdz_file):
    scene = _synthetic_scene()
    before = copy.deepcopy(scene)
    spec = sb.SplatBackgroundSpec(
        name="PD",
        usdz=usdz_file,
        matrix=np.diag([20.0, 20.0, 20.0, 1.0]),
        pose_json=None,
        dz_m=0.0,
        floor_plane_visible=False,
        remove=("iter_7",),
        kit_settings={},
        color=None,
    )
    with pytest.raises(sb.SplatBackgroundError):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_the_record_is_complete_and_json_serialisable(tmp_path, usdz_file):
    scene = _synthetic_scene()
    spec = sb.load_visual_background(
        _spec_file(tmp_path, _body(remove=["mesh_background_0", "iter_7"], kit_settings={"/rtx/x": True}, dz_m=0.002))
    )
    record = sb.swap_visual_background(scene, spec)
    assert set(record) == RECORD_KEYS
    json.dumps(record)  # no numpy scalars or arrays
    assert record["name"] == "PD"
    assert record["usdz"] == str(usdz_file)
    assert record["md5"] == hashlib.md5(usdz_file.read_bytes()).hexdigest()
    assert record["removed"] == ["mesh_background_0", "iter_7"]
    assert record["dz_m"] == pytest.approx(0.002)
    assert record["floor_plane_visible"] is False
    assert record["kit_settings"] == {"/rtx/x": True}
    assert record["scale"] == pytest.approx(1.0, abs=1e-12)  # _rigid() has no scale


def test_the_scene_stays_json_serialisable_after_the_swap(tmp_path, usdz_file):
    scene = _synthetic_scene()
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"])))
    sb.swap_visual_background(scene, spec)
    json.dumps(scene)


def test_the_record_pose_has_the_dz_in_z_only(tmp_path, usdz_file):
    scene = _synthetic_scene()
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"], dz_m=0.004)))
    record = sb.swap_visual_background(scene, spec)
    m = _rigid()
    assert record["pos"][:2] == pytest.approx(m[:2, 3].tolist(), abs=0)
    assert record["pos"][2] == pytest.approx(m[2, 3] + 0.004, abs=1e-15)
    assert scene["state"]["registry"]["object_registry"][GS]["root_link"]["pos"] == record["pos"]


def test_the_splat_entry_has_no_category_and_the_stage7_flags(tmp_path, usdz_file):
    scene = _synthetic_scene()
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"])))
    sb.swap_visual_background(scene, spec)
    entry = scene["objects_info"]["init_info"][GS]
    assert "category" not in entry["args"]
    assert entry["args"]["visual_only"] is True and entry["args"]["fixed_base"] is True
    assert entry["args"]["name"] == GS
    assert all(type(v) is float for v in entry["args"]["scale"]) and len(entry["args"]["scale"]) == 3


def test_the_state_values_are_plain_python_numbers(tmp_path, usdz_file):
    scene = _synthetic_scene()
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"])))
    sb.swap_visual_background(scene, spec)
    state = scene["state"]["registry"]["object_registry"][GS]
    assert state["is_asleep"] is True and state["non_kin"] == {}
    assert all(type(v) is float for v in state["root_link"]["pos"] + state["root_link"]["ori"])


def test_entry_and_state_helpers_convert_numpy_values():
    entry = sb.gs_background_entry("/x/PD_bg.usdz", np.float64(1.25), "abc")
    assert entry["args"]["scale"] == [1.25, 1.25, 1.25]
    assert all(type(v) is float for v in entry["args"]["scale"])
    assert entry["args"]["usd_path"] == "/x/PD_bg.usdz"
    assert "category" not in entry["args"]
    state = sb.gs_background_state(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    assert all(type(v) is float for v in state["root_link"]["pos"] + state["root_link"]["ori"])
    json.dumps(state)


def test_md5_matches_hashlib_across_several_read_chunks(tmp_path):
    path = tmp_path / "big.usdz"
    path.write_bytes(bytes(range(256)) * (2 * 4096 + 17))  # more than two 1 MiB reads
    assert sb.md5_of_file(path) == hashlib.md5(path.read_bytes()).hexdigest()


def test_expected_file_hash_is_the_md5_of_the_file_on_disk(tmp_path, usdz_file):
    entry = sb.gs_background_entry(usdz_file, 1.0, sb.md5_of_file(usdz_file))
    assert entry["args"]["expected_file_hash"] == hashlib.md5(usdz_file.read_bytes()).hexdigest()


@pytest.mark.skipif(not _present(PD_USDZ), reason=_skip_reason(PD_USDZ_ENV))
def test_md5_of_the_real_usdz_matches_hashlib():
    assert sb.md5_of_file(PD_USDZ) == _pd_md5()


# --------------------------------------------- edit_scene does not depend on it (CPU-5)


def _import_tofu_og_cut():
    scripts = str(ROOT / "scripts" / "cutting")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        import tofu_og_cut
    except ImportError as error:
        pytest.skip(f"tofu_og_cut cannot be imported here: {error}")
    return tofu_og_cut


def _fake_fit_and_rec(scene):
    """What edit_scene reads from the reconstruction and the fit: names, categories, and the board's height."""
    objects = {}
    for name, entry in scene["objects_info"]["init_info"].items():
        category = entry.get("args", {}).get("category")
        if category:
            objects[name] = SimpleNamespace(name=name, category=category)

    class Rec:
        def __init__(self):
            self.objects = objects

        def find(self, query):
            hits = [o for o in objects.values() if query in o.category]
            if len(hits) != 1:
                raise KeyError(query)
            return hits[0]

    rec = Rec()
    fitted = SimpleNamespace(food=objects["iter_4"], board=objects["iter_1"], board_z=0.011043721188946797)
    return rec, fitted


def _run_edit_scene(tofu, scene):
    rec, fitted = _fake_fit_and_rec(scene)
    hx, hy = 0.239 / 2, 0.284 / 2
    box = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (0.0, 0.011)])
    cut_xy = np.column_stack([np.linspace(-0.24, -0.05, 25), np.full(25, 0.0453)])
    return tofu.edit_scene(
        scene,
        rec=rec,
        fitted=fitted,
        robot_pos=np.array([-0.78, -0.07, 0.0]),
        robot_yaw=-0.2,
        arm_controller={"name": "InverseKinematicsController"},
        knife_usd="knife/chef_knife.usd",
        board_hulls=[box],
        cut_xy=cut_xy,
        clear=["paper_towel"],
    )


def _without_background(scene):
    out = copy.deepcopy(scene)
    for section in (out["objects_info"]["init_info"], out["state"]["registry"]["object_registry"]):
        for name in ("mesh_background_0", "iter_7", GS):
            section.pop(name, None)
    return out


def test_edit_scene_gives_the_same_result_and_scene_with_and_without_the_swap(tmp_path):
    _need(SCENE_JSON, SCENE_ENV)
    tofu = _import_tofu_og_cut()
    usdz = tmp_path / "PD_bg.usdz"
    usdz.write_bytes(b"stand-in: the edit does not read it, the swap only hashes it")
    spec = sb.load_visual_background(
        _spec_file(tmp_path, {"schema": 1, "name": "PD", "usdz": str(usdz), "matrix": _rigid().tolist(),
                              "remove": ["mesh_background_0", "iter_7"]})
    )

    scene_a = json.loads(SCENE_JSON.read_text())  # edit only
    result_a = _run_edit_scene(tofu, scene_a)

    scene_b = json.loads(SCENE_JSON.read_text())  # swap, then edit
    sb.swap_visual_background(scene_b, spec)
    result_b = _run_edit_scene(tofu, scene_b)

    scene_c = json.loads(SCENE_JSON.read_text())  # edit, then swap
    result_c = _run_edit_scene(tofu, scene_c)
    sb.swap_visual_background(scene_c, spec)

    assert result_a == result_b == result_c
    assert _without_background(scene_a) == _without_background(scene_b)  # only the background entries differ
    assert scene_b == scene_c  # the order of the two edits does not matter


# ------------------------------------------------------- the USDZ (CPU-7)


_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _ply_vertex_aabb(path):
    """Min and max of x, y, z over the vertices of a binary little-endian PLY, read through a memory map."""
    with open(path, "rb") as handle:
        header = []
        while True:
            line = handle.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        offset = handle.tell()
    assert any(line == "format binary_little_endian 1.0" for line in header), "the PLY is not binary little-endian"
    count, fields, in_vertex = 0, [], False
    for line in header:
        parts = line.split()
        if parts[:1] == ["element"]:
            in_vertex = parts[1] == "vertex"
            count = int(parts[2]) if in_vertex else count
        elif parts[:1] == ["property"] and in_vertex:
            code = _PLY_TYPES[parts[1]]
            fields.append((parts[-1], code if code in ("u1", "i1") else "<" + code))
    verts = np.memmap(path, dtype=np.dtype(fields), mode="r", offset=offset, shape=(count,))
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    return xyz.min(axis=0), xyz.max(axis=0)


def test_usdz_is_a_nurec_volume_whose_extent_is_the_ply_aabb():
    pytest.importorskip("pxr")  # usd-core: the simfoundry-editor env has it, the simfoundry env does not
    _need(PD_USDZ, PD_USDZ_ENV)
    _need(PD_PLY, PD_PLY_ENV)
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(PD_USDZ))
    assert stage is not None
    prim = stage.GetPrimAtPath("/World/gauss/gauss")
    assert prim.IsValid(), "no /World/gauss/gauss prim in the USDZ"
    assert prim.GetTypeName() == "Volume"
    assert prim.GetAttribute("omni:nurec:isNuRecVolume").Get() is True
    extent = UsdGeom.Boundable(prim).GetExtentAttr().Get()
    assert extent is not None
    lo, hi = _ply_vertex_aabb(PD_PLY)
    assert np.max(np.abs(np.array(extent[0]) - lo)) <= 1e-4
    assert np.max(np.abs(np.array(extent[1]) - hi)) <= 1e-4


# ---------------------------------------------------------------- colour matrix (ccm): run-time colour of the splat only

CCM_ROWS = [[1.1, 0.0, 0.0, 0.02], [0.0, 1.0, 0.0, -0.01], [0.0, 0.0, 0.9, 0.0]]
BAD_CCMS = [
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],  # two rows
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # a short row
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]],  # a long row
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], "0 0 1 0"],  # a row that is not a list
    "identity",  # not a list
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, True]],  # a bool
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, None]],  # null
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, float("nan")]],  # not a number
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, float("inf")]],  # not finite
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 4.5]],  # over the limit
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -4.5]],  # under the limit
]


class _Attr:
    """Stands in for a USD attribute: the three calls ``apply_splat_ccm`` makes (``IsValid``, ``Get``, ``Set``)."""

    def __init__(self, name, valid=True):
        self.name, self.valid, self.value, self.sets = name, valid, None, []

    def IsValid(self):
        return self.valid

    def Get(self):
        return self.value

    def Set(self, value):
        self.sets.append(value)
        self.value = value


class _Prim:
    """Stands in for a USD prim: a path, child prims, and attributes looked up by name (a missing one is None, as in USD)."""

    def __init__(self, path, attrs=(), children=()):
        self.path, self.children = path, list(children)
        self.attrs = {attr.name: attr for attr in attrs}

    def GetPath(self):
        return self.path

    def GetChildren(self):
        return list(self.children)

    def GetAttribute(self, name):
        return self.attrs.get(name)


def _carriers():
    return [_Attr(name) for name in sb.CCM_ATTRIBUTES]


def _tuple_vec(*values):
    return ("vec", values)


def test_ccm_is_none_when_the_key_is_absent_or_null(tmp_path, usdz_file):
    assert sb.load_visual_background(_spec_file(tmp_path, _body())).ccm is None
    assert sb.load_visual_background(_spec_file(tmp_path, _body(ccm=None), "null.json")).ccm is None


def test_a_ccm_loads_as_three_tuples_of_four_floats(tmp_path, usdz_file):
    ccm = sb.load_visual_background(_spec_file(tmp_path, _body(ccm=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))).ccm
    assert ccm == ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
    assert all(type(v) is float for row in ccm for v in row)


@pytest.mark.parametrize("ccm", BAD_CCMS)
def test_a_bad_ccm_is_refused_and_named(tmp_path, usdz_file, ccm):
    with pytest.raises(sb.SplatBackgroundError, match="ccm"):
        sb.load_visual_background(_spec_file(tmp_path, _body(ccm=ccm)))


def test_a_hand_built_spec_with_a_bad_ccm_is_refused_before_any_change(tmp_path, usdz_file):
    scene = _synthetic_scene()
    before = copy.deepcopy(scene)
    spec = dataclasses.replace(sb.load_visual_background(_spec_file(tmp_path, _body())),
                               ccm=((1.0, 0.0, 0.0, 9.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)))
    with pytest.raises(sb.SplatBackgroundError, match="ccm"):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_the_record_has_a_ccm_key_only_when_the_spec_has_one(tmp_path, usdz_file):
    plain = sb.swap_visual_background(_synthetic_scene(), sb.load_visual_background(_spec_file(tmp_path, _body())))
    assert set(plain) == RECORD_KEYS
    with_ccm = sb.swap_visual_background(
        _synthetic_scene(), sb.load_visual_background(_spec_file(tmp_path, _body(ccm=CCM_ROWS), "ccm.json")))
    assert set(with_ccm) == RECORD_KEYS | {"ccm"}
    assert with_ccm["ccm"] == CCM_ROWS
    json.dumps(with_ccm)


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_apply_writes_row_i_to_attribute_i_of_the_prim_that_carries_them(depth):
    attrs = _carriers()
    carrier = _Prim("/World/gauss/gauss/emissive_color_field", attrs)
    root = carrier
    for _ in range(depth):  # the splat root can be the carrier, or sit above it
        root = _Prim("/World/wrapper", children=[root])
    written = sb.apply_splat_ccm(root, CCM_ROWS, make_vec=_tuple_vec)
    assert [a.sets for a in attrs] == [[_tuple_vec(*row)] for row in CCM_ROWS]
    assert written == [{"prim": carrier.path, "attr": name, "read_back": str(_tuple_vec(*row))}
                       for name, row in zip(sb.CCM_ATTRIBUTES, CCM_ROWS)]


def test_every_prim_that_carries_the_matrix_is_written_not_only_the_first():
    # Two carriers, as a splat with several volumes would have: each must get all three rows, in order.
    first, second = _carriers(), _carriers()
    a = _Prim("/World/gauss/a/emissive_color_field", first)
    b = _Prim("/World/gauss/b/emissive_color_field", second)
    written = sb.apply_splat_ccm(_Prim("/World/gauss", children=[a, b]), CCM_ROWS, make_vec=_tuple_vec)
    assert [x.sets for x in first] == [[_tuple_vec(*row)] for row in CCM_ROWS]
    assert [x.sets for x in second] == [[_tuple_vec(*row)] for row in CCM_ROWS]
    assert [w["prim"] for w in written] == [a.path] * 3 + [b.path] * 3


def test_a_splat_with_no_colour_matrix_is_refused_and_nothing_is_written():
    root = _Prim("/World/gauss", children=[_Prim("/World/gauss/gauss")])
    with pytest.raises(sb.SplatBackgroundError, match="ccm"):
        sb.apply_splat_ccm(root, CCM_ROWS, make_vec=_tuple_vec)


def test_an_attribute_that_is_not_valid_counts_as_absent():
    attrs = [_Attr(sb.CCM_ATTRIBUTES[0]), _Attr(sb.CCM_ATTRIBUTES[1]), _Attr(sb.CCM_ATTRIBUTES[2], valid=False)]
    with pytest.raises(sb.SplatBackgroundError, match="ccmB"):
        sb.apply_splat_ccm(_Prim("/World/gauss", attrs), CCM_ROWS, make_vec=_tuple_vec)
    assert all(a.sets == [] for a in attrs)


def test_a_partial_prim_is_refused_before_a_complete_one_is_written():
    full, partial = _carriers(), [_Attr(sb.CCM_ATTRIBUTES[0]), _Attr(sb.CCM_ATTRIBUTES[1])]
    root = _Prim("/World/gauss", children=[_Prim("/World/gauss/a", full), _Prim("/World/gauss/b", partial)])
    with pytest.raises(sb.SplatBackgroundError):
        sb.apply_splat_ccm(root, CCM_ROWS, make_vec=_tuple_vec)
    assert all(a.sets == [] for a in full + partial)


@pytest.mark.parametrize("rows", [CCM_ROWS[:2], [row[:3] for row in CCM_ROWS]])
def test_apply_refuses_rows_that_are_not_three_by_four(rows):
    attrs = _carriers()
    with pytest.raises(sb.SplatBackgroundError):
        sb.apply_splat_ccm(_Prim("/World/gauss", attrs), rows, make_vec=_tuple_vec)
    assert all(a.sets == [] for a in attrs)


def test_the_default_vector_is_a_pxr_gf_vec4f_of_plain_floats(monkeypatch):
    made = []

    def vec4f(*values):
        made.append(values)
        return ("Gf.Vec4f", values)

    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(Gf=SimpleNamespace(Vec4f=vec4f)))
    attrs = _carriers()
    written = sb.apply_splat_ccm(_Prim("/World/gauss", attrs), [[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0]])
    assert made == [(1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)]
    assert all(type(v) is float for values in made for v in values)
    assert [a.value for a in attrs] == [("Gf.Vec4f", values) for values in made]
    assert written[0]["read_back"] == str(("Gf.Vec4f", (1.0, 0.0, 0.0, 0.5)))


def test_loading_the_module_does_not_load_pxr():
    code = "import sys, simfoundry.utils.splat_background; print('pxr' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_apply_on_a_real_usd_stage_writes_the_matrix_and_reads_it_back():
    pytest.importorskip("pxr")  # usd-core: the simfoundry-editor env has it, the simfoundry env does not
    from pxr import Gf, Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    splat = stage.DefinePrim("/World/gauss", "Xform")
    emissive = stage.DefinePrim("/World/gauss/gauss/emissive_color_field", "Scope")
    for name in sb.CCM_ATTRIBUTES:
        emissive.CreateAttribute(name, Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0, 0, 0, 0))
    written = sb.apply_splat_ccm(splat, CCM_ROWS)  # the default make_vec: real Gf.Vec4f values
    assert written == [{"prim": "/World/gauss/gauss/emissive_color_field", "attr": name, "read_back": str(Gf.Vec4f(*row))}
                       for name, row in zip(sb.CCM_ATTRIBUTES, CCM_ROWS)]
    for name, row in zip(sb.CCM_ATTRIBUTES, CCM_ROWS):
        assert emissive.GetAttribute(name).Get() == Gf.Vec4f(*row)


# ---------------------------------------------- counter patch (I12): opaque boxes under the countertop

PATCH_RECTS = [[-0.62, 0.05, -0.62, -0.10], [-0.34, 0.005, -0.08, 0.28]]  # two rectangles, as a spec gives them


def _patch(**changes):
    """A valid counter_patch object with one rectangle, with ``changes`` applied."""
    body = {"rects": [[-0.62, 0.05, -0.62, -0.10]], "z_top_m": -0.0005, "thickness_m": 0.004, "rgb": [0.8, 0.8, 0.8]}
    body.update(changes)
    return body


def _patched_spec(tmp_path, patch=None, remove=("mesh_background_0", "iter_7")):
    """A spec with two boxes, unless ``patch`` is given. The spec file is written and loaded at once."""
    body = _body(remove=list(remove), counter_patch=_patch(rects=PATCH_RECTS) if patch is None else patch)
    return sb.load_visual_background(_spec_file(tmp_path, body, name="patched.json"))


PATCH_REFUSALS = {
    "missing_rects": lambda p: p.pop("rects"),
    "missing_z_top": lambda p: p.pop("z_top_m"),
    "missing_thickness": lambda p: p.pop("thickness_m"),
    "missing_rgb": lambda p: p.pop("rgb"),
    "unknown_key": lambda p: p.update(alpha=1.0),
    "no_rects": lambda p: p.update(rects=[]),
    "nine_rects": lambda p: p.update(rects=[[-0.5, -0.4, -0.5, -0.4] for _ in range(9)]),
    "rect_three_numbers": lambda p: p.update(rects=[[-0.62, 0.05, -0.62]]),
    "rect_five_numbers": lambda p: p.update(rects=[[-0.62, 0.05, -0.62, -0.10, 0.0]]),
    "rects_flat_numbers": lambda p: p.update(rects=[-0.62, 0.05, -0.62, -0.10]),
    "rect_is_a_number": lambda p: p.update(rects=[1.0]),
    "rect_xmin_equal_xmax": lambda p: p.update(rects=[[0.05, 0.05, -0.62, -0.10]]),
    "rect_xmin_above_xmax": lambda p: p.update(rects=[[0.10, 0.05, -0.62, -0.10]]),
    "rect_ymin_above_ymax": lambda p: p.update(rects=[[-0.62, 0.05, -0.10, -0.62]]),
    "rect_ymin_equal_ymax": lambda p: p.update(rects=[[-0.62, 0.05, -0.10, -0.10]]),
    "rect_corner_over_5_m": lambda p: p.update(rects=[[-5.5, 0.05, -0.62, -0.10]]),
    # a bool must be refused where its value would make the rectangle valid, or the order test refuses it instead
    "rect_bool_as_xmax": lambda p: p.update(rects=[[-0.62, True, -0.62, -0.10]]),
    "rect_bool_as_ymin": lambda p: p.update(rects=[[-0.62, 0.05, False, 0.5]]),
    "rect_nan": lambda p: p.update(rects=[[float("nan"), 0.05, -0.62, -0.10]]),
    "rect_inf": lambda p: p.update(rects=[[-0.62, float("inf"), -0.62, -0.10]]),
    "rect_string": lambda p: p.update(rects=[["-0.62", 0.05, -0.62, -0.10]]),
    "z_top_too_high": lambda p: p.update(z_top_m=0.021),
    "z_top_too_low": lambda p: p.update(z_top_m=-0.021),
    "z_top_nan": lambda p: p.update(z_top_m=float("nan")),
    "z_top_bool": lambda p: p.update(z_top_m=False),
    "thickness_zero": lambda p: p.update(thickness_m=0.0),
    "thickness_negative": lambda p: p.update(thickness_m=-0.004),
    "thickness_over_5_cm": lambda p: p.update(thickness_m=0.051),
    "thickness_inf": lambda p: p.update(thickness_m=float("inf")),
    "rgb_above_1": lambda p: p.update(rgb=[0.8, 0.8, 1.01]),
    "rgb_below_0": lambda p: p.update(rgb=[-0.01, 0.8, 0.8]),
    "rgb_two_numbers": lambda p: p.update(rgb=[0.8, 0.8]),
    "rgb_bool": lambda p: p.update(rgb=[True, 0.8, 0.8]),
    "rgb_nan": lambda p: p.update(rgb=[0.8, float("nan"), 0.8]),
    "rgb_is_a_string": lambda p: p.update(rgb="0.8 0.8 0.8"),
}

ACCEPTED_PATCHES = {
    "one_box": {},
    "eight_boxes": {"rects": [[-1.0 + 0.1 * i, -0.9 + 0.1 * i, -0.5, 0.5] for i in range(8)]},
    "z_top_at_plus_limit": {"z_top_m": 0.02},
    "z_top_at_minus_limit": {"z_top_m": -0.02},
    "thickness_at_limit": {"thickness_m": 0.05},
    "rgb_at_0_and_1": {"rgb": [0.0, 1.0, 0.0]},
    "corner_at_5_m": {"rects": [[-5.0, 5.0, -5.0, 5.0]]},
    "overlapping_boxes": {"rects": [[-0.5, 0.0, -0.5, 0.0], [-0.2, 0.3, -0.2, 0.3]]},
    "whole_numbers": {"rects": [[-1, 1, -1, 1]], "z_top_m": 0, "thickness_m": 0.05, "rgb": [0, 1, 0]},
}


def test_an_absent_or_null_counter_patch_is_none(tmp_path, usdz_file):
    assert sb.load_visual_background(_spec_file(tmp_path, _body())).counter_patch is None
    assert sb.load_visual_background(_spec_file(tmp_path, _body(counter_patch=None), "null.json")).counter_patch is None


@pytest.mark.parametrize("name", list(PATCH_REFUSALS))
def test_a_bad_counter_patch_is_refused_and_named(tmp_path, usdz_file, name):
    patch = _patch()
    PATCH_REFUSALS[name](patch)
    with pytest.raises(sb.SplatBackgroundError, match="counter_patch"):
        sb.load_visual_background(_spec_file(tmp_path, _body(counter_patch=patch)))


@pytest.mark.parametrize("value", [[], "on", 1, True, {}], ids=["list", "string", "number", "bool", "empty_object"])
def test_a_counter_patch_that_is_not_an_object_or_is_empty_is_refused(tmp_path, usdz_file, value):
    with pytest.raises(sb.SplatBackgroundError, match="counter_patch"):
        sb.load_visual_background(_spec_file(tmp_path, _body(counter_patch=value)))


def test_an_unknown_key_inside_the_patch_is_named(tmp_path, usdz_file):
    with pytest.raises(sb.SplatBackgroundError, match="'alpha'"):
        sb.load_visual_background(_spec_file(tmp_path, _body(counter_patch=_patch(alpha=1.0))))


def test_a_missing_key_inside_the_patch_is_named(tmp_path, usdz_file):
    patch = _patch()
    patch.pop("rgb")
    with pytest.raises(sb.SplatBackgroundError, match="'rgb'"):
        sb.load_visual_background(_spec_file(tmp_path, _body(counter_patch=patch)))


@pytest.mark.parametrize("name", list(ACCEPTED_PATCHES))
def test_a_counter_patch_at_a_limit_or_with_whole_numbers_loads_as_plain_floats(tmp_path, usdz_file, name):
    body = _patch(**ACCEPTED_PATCHES[name])
    patch = sb.load_visual_background(_spec_file(tmp_path, _body(counter_patch=body))).counter_patch
    assert patch.rects == tuple(tuple(float(v) for v in rect) for rect in body["rects"])
    assert patch.z_top_m == float(body["z_top_m"]) and patch.thickness_m == float(body["thickness_m"])
    assert patch.rgb == tuple(float(v) for v in body["rgb"])
    assert all(type(v) is float for rect in patch.rects for v in rect)
    assert all(type(v) is float for v in (patch.z_top_m, patch.thickness_m, *patch.rgb))


def test_a_counter_patch_survives_a_spec_file_round_trip(tmp_path, usdz_file):
    path = _spec_file(tmp_path, _body(counter_patch=_patch(rects=PATCH_RECTS)))
    assert json.loads(path.read_text(encoding="utf-8"))["counter_patch"] == _patch(rects=PATCH_RECTS)
    assert sb.load_visual_background(path).counter_patch == sb.CounterPatch(
        rects=((-0.62, 0.05, -0.62, -0.10), (-0.34, 0.005, -0.08, 0.28)), z_top_m=-0.0005, thickness_m=0.004,
        rgb=(0.8, 0.8, 0.8))


def test_the_counter_patch_object_is_frozen(tmp_path, usdz_file):
    patch = sb.load_visual_background(_spec_file(tmp_path, _body(counter_patch=_patch()))).counter_patch
    with pytest.raises(dataclasses.FrozenInstanceError):
        patch.z_top_m = 0.0


def test_each_box_is_a_unit_cube_scaled_to_its_rectangle_and_thickness(tmp_path, usdz_file):
    scene = _synthetic_scene()
    sb.swap_visual_background(scene, _patched_spec(tmp_path))
    entry = scene["objects_info"]["init_info"]["counter_patch_0"]
    assert entry["class_module"] == "omnigibson.objects.primitive_object" and entry["class_name"] == "PrimitiveObject"
    args = entry["args"]
    # no size: OmniGibson ignores it when a scale is given (primitive_object.py 150-152)
    assert list(args) == ["name", "primitive_type", "scale", "rgba", "visual_only", "fixed_base"]
    assert args["name"] == "counter_patch_0" and args["primitive_type"] == "Cube"
    assert args["scale"] == pytest.approx([0.67, 0.52, 0.004], abs=1e-12)  # xmax - xmin, ymax - ymin, thickness
    assert args["rgba"] == [0.8, 0.8, 0.8, 1.0]
    assert args["visual_only"] is True and args["fixed_base"] is True
    assert all(type(v) is float for v in args["scale"] + args["rgba"])


def test_each_box_state_is_centred_on_its_rectangle_with_its_top_at_z_top(tmp_path, usdz_file):
    scene = _synthetic_scene()
    sb.swap_visual_background(scene, _patched_spec(tmp_path))
    state = scene["state"]["registry"]["object_registry"]["counter_patch_0"]
    assert state["is_asleep"] is True and state["non_kin"] == {}
    root = state["root_link"]
    # the centre of the rectangle in x and y; in z the centre is half a thickness below the top, z_top_m
    assert root["pos"] == pytest.approx([-0.285, -0.36, -0.0025], abs=1e-12)
    assert root["ori"] == [0.0, 0.0, 0.0, 1.0]
    assert root["lin_vel"] == [0.0, 0.0, 0.0] and root["ang_vel"] == [0.0, 0.0, 0.0]
    assert all(type(v) is float for v in root["pos"] + root["ori"])


def test_each_box_state_has_the_keys_that_loading_a_cube_reads(tmp_path, usdz_file):
    # Restoring the initial state calls PrimitiveObject._load_state, which reads state["size"] for a Cube, and serialize
    # reads radius and height too (primitive_object.py 369-396). Every box has the keys _dump_state writes, for any rectangles.
    scene = _synthetic_scene()
    sb.swap_visual_background(scene, _patched_spec(tmp_path))
    for name in ("counter_patch_0", "counter_patch_1"):
        state = scene["state"]["registry"]["object_registry"][name]
        assert set(state) == {"is_asleep", "non_kin", "root_link", "radius", "height", "size"}
        assert (state["radius"], state["height"], state["size"]) == (-1, -1, 1.0)


def test_an_asymmetric_colour_reaches_the_box_in_the_order_it_was_given(tmp_path, usdz_file):
    # The three channels differ, so a swapped channel shows in the box and in the record.
    body = _body(counter_patch=_patch(rgb=[0.1, 0.5, 0.9]), remove=["iter_7"])
    path = _spec_file(tmp_path, body)
    assert json.loads(path.read_text(encoding="utf-8"))["counter_patch"]["rgb"] == [0.1, 0.5, 0.9]
    spec = sb.load_visual_background(path)
    assert spec.counter_patch.rgb == (0.1, 0.5, 0.9)
    scene = _synthetic_scene()
    record = sb.swap_visual_background(scene, spec)
    assert scene["objects_info"]["init_info"]["counter_patch_0"]["args"]["rgba"] == [0.1, 0.5, 0.9, 1.0]
    assert record["counter_patch"]["rgb"] == [0.1, 0.5, 0.9]


def test_the_boxes_follow_the_splat_in_rectangle_order_in_both_sections(tmp_path, usdz_file):
    scene = _synthetic_scene()
    sb.swap_visual_background(scene, _patched_spec(tmp_path))
    order = ["robot0", "iter_1", "iter_3", GS, "counter_patch_0", "counter_patch_1"]
    assert list(scene["objects_info"]["init_info"]) == order
    assert list(scene["state"]["registry"]["object_registry"]) == order
    assert scene["objects_info"]["init_info"]["counter_patch_1"]["args"]["scale"] == pytest.approx(
        [0.345, 0.36, 0.004], abs=1e-12)


def test_the_box_entries_do_not_depend_on_which_objects_are_removed(tmp_path, usdz_file):
    kept, dropped = _synthetic_scene(), _synthetic_scene()
    sb.swap_visual_background(kept, _patched_spec(tmp_path, remove=("iter_7",)))
    sb.swap_visual_background(dropped, _patched_spec(tmp_path, remove=("mesh_background_0", "iter_7")))
    for name in ("counter_patch_0", "counter_patch_1"):
        assert kept["objects_info"]["init_info"][name] == dropped["objects_info"]["init_info"][name]
        assert kept["state"]["registry"]["object_registry"][name] == dropped["state"]["registry"]["object_registry"][name]


@pytest.mark.parametrize("section", ["init_info", "registry"])
def test_a_scene_that_already_has_a_box_name_is_refused_and_unchanged(tmp_path, usdz_file, section):
    scene = _synthetic_scene()
    target = scene["objects_info"]["init_info"] if section == "init_info" else scene["state"]["registry"]["object_registry"]
    target["counter_patch_1"] = {"args": {}}
    before = copy.deepcopy(scene)
    with pytest.raises(sb.SplatBackgroundError, match="already has a 'counter_patch_1' object"):
        sb.swap_visual_background(scene, _patched_spec(tmp_path))
    assert scene == before


HAND_BUILT_GOOD = {"rects": ((-0.62, 0.05, -0.62, -0.10),), "z_top_m": -0.0005, "thickness_m": 0.004, "rgb": (0.8, 0.8, 0.8)}
HAND_BUILT_BAD = [
    {"rects": ((0.1, 0.1, -0.6, -0.1),)},  # xmin == xmax
    {"rects": ((0.0, 1.0, 0.0, 1.0),) * 9},  # nine boxes
    {"z_top_m": 0.5},
    {"thickness_m": 0.0},
    {"rgb": (0.8, 0.8, 1.5)},
]


@pytest.mark.parametrize("changes", HAND_BUILT_BAD)
def test_a_hand_built_patch_is_checked_before_any_change(tmp_path, usdz_file, changes):
    scene = _synthetic_scene()
    before = copy.deepcopy(scene)
    base = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"])))
    spec = dataclasses.replace(base, counter_patch=sb.CounterPatch(**{**HAND_BUILT_GOOD, **changes}))
    with pytest.raises(sb.SplatBackgroundError, match="counter_patch"):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_a_hand_built_patch_gives_the_same_scene_and_record_as_a_loaded_one(tmp_path, usdz_file):
    loaded = _patched_spec(tmp_path)
    built = dataclasses.replace(loaded, counter_patch=sb.CounterPatch(
        rects=((-0.62, 0.05, -0.62, -0.10), (-0.34, 0.005, -0.08, 0.28)), z_top_m=-0.0005, thickness_m=0.004,
        rgb=(0.8, 0.8, 0.8)))
    a, b = _synthetic_scene(), _synthetic_scene()
    record_a, record_b = sb.swap_visual_background(a, loaded), sb.swap_visual_background(b, built)
    assert a == b and record_a == record_b


def test_the_record_has_a_counter_patch_key_only_when_the_spec_has_boxes(tmp_path, usdz_file):
    plain = sb.swap_visual_background(_synthetic_scene(), sb.load_visual_background(_spec_file(tmp_path, _body())))
    assert set(plain) == RECORD_KEYS and "counter_patch" not in plain
    patched = sb.swap_visual_background(_synthetic_scene(), _patched_spec(tmp_path))
    assert set(patched) == RECORD_KEYS | {"counter_patch"}
    assert patched["counter_patch"] == {
        "names": ["counter_patch_0", "counter_patch_1"],
        "rects": [[-0.62, 0.05, -0.62, -0.10], [-0.34, 0.005, -0.08, 0.28]],
        "z_top_m": -0.0005,
        "thickness_m": 0.004,
        "rgb": [0.8, 0.8, 0.8],
    }
    json.dumps(patched)


def test_the_record_keeps_the_ccm_and_the_patch_keys_side_by_side(tmp_path, usdz_file):
    body = _body(ccm=CCM_ROWS, counter_patch=_patch(), remove=["iter_7"])
    spec = sb.load_visual_background(_spec_file(tmp_path, body, "both.json"))
    record = sb.swap_visual_background(_synthetic_scene(), spec)
    assert set(record) == RECORD_KEYS | {"ccm", "counter_patch"}
    assert record["ccm"] == CCM_ROWS and record["counter_patch"]["names"] == ["counter_patch_0"]


def test_golden_two_boxes_on_the_p3_scene_match_the_hand_computed_values(tmp_path, usdz_file):
    _need(SCENE_JSON, SCENE_ENV)
    scene = json.loads(SCENE_JSON.read_text())
    before = copy.deepcopy(scene)
    spec = sb.load_visual_background(_spec_file(
        tmp_path, _body(remove=["mesh_background_0", "iter_7"], counter_patch=_patch(rects=PATCH_RECTS))))
    sb.swap_visual_background(scene, spec)
    # Worked out by hand from the rectangles, not with the module's helpers:
    # centre = ((xmin + xmax) / 2, (ymin + ymax) / 2, z_top - thickness / 2); scale = (xmax - xmin, ymax - ymin, thickness).
    boxes = {
        "counter_patch_0": ([-0.285, -0.36, -0.0025], [0.67, 0.52, 0.004]),  # rectangle [-0.62, 0.05, -0.62, -0.10]
        "counter_patch_1": ([-0.1675, 0.1, -0.0025], [0.345, 0.36, 0.004]),  # rectangle [-0.34, 0.005, -0.08, 0.28]
    }
    init, registry = scene["objects_info"]["init_info"], scene["state"]["registry"]["object_registry"]
    kept = [k for k in before["objects_info"]["init_info"] if k not in ("mesh_background_0", "iter_7")]
    order = [*kept, GS, *boxes]
    assert list(init) == order and list(registry) == order
    for name, (centre, scale) in boxes.items():
        assert init[name]["args"]["scale"] == pytest.approx(scale, abs=1e-12)
        assert init[name]["args"]["rgba"] == [0.8, 0.8, 0.8, 1.0]
        assert registry[name]["root_link"]["pos"] == pytest.approx(centre, abs=1e-12)
        assert set(registry[name]) == {"is_asleep", "non_kin", "root_link", "radius", "height", "size"}
        assert (registry[name]["radius"], registry[name]["height"], registry[name]["size"]) == (-1, -1, 1.0)
    for key in kept:  # the other objects are as they were
        assert init[key] == before["objects_info"]["init_info"][key]
        assert registry[key] == before["state"]["registry"]["object_registry"][key]


def test_a_plain_swap_record_has_the_keys_in_the_order_it_always_had(tmp_path, usdz_file):
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["mesh_background_0", "iter_7"])))
    record = sb.swap_visual_background(_synthetic_scene(), spec)
    assert list(record) == ["name", "usdz", "md5", "removed", "pos", "ori_xyzw", "scale", "dz_m", "floor_plane_visible",
                            "kit_settings"]
    assert record["removed"] == ["mesh_background_0", "iter_7"]
    assert record["md5"] == hashlib.md5(usdz_file.read_bytes()).hexdigest()
    assert record["pos"] == pytest.approx([0.1, 0.2, 0.3], abs=1e-12)
    assert record["ori_xyzw"] == pytest.approx(Rotation.from_rotvec([0.3, -0.2, 0.5]).as_quat().tolist(), abs=1e-12)
    assert record["scale"] == pytest.approx(1.0, abs=1e-12) and record["dz_m"] == 0.0


def test_without_boxes_the_p3_swap_adds_no_box_entries_and_no_record_key(tmp_path, usdz_file):
    _need(SCENE_JSON, SCENE_ENV)
    scene = json.loads(SCENE_JSON.read_text())
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["mesh_background_0", "iter_7"])))
    record = sb.swap_visual_background(scene, spec)
    for section in (scene["objects_info"]["init_info"], scene["state"]["registry"]["object_registry"]):
        assert not [name for name in section if name.startswith("counter_patch_")]
    assert "counter_patch" not in record


def test_counter_patch_entry_keeps_the_colour_channels_in_order():
    # 0.1, 0.5 and 0.9 are all different, so reversing or permuting the channels changes the colour.
    entry = sb.counter_patch_entry(0, (-0.62, 0.05, -0.62, -0.10), 0.004, (0.1, 0.5, 0.9))
    assert entry["args"]["rgba"] == [0.1, 0.5, 0.9, 1.0]


def test_the_box_helpers_convert_numpy_values_and_give_plain_floats():
    rect = np.array([-1.0, 0.5, -0.25, 0.25])
    entry = sb.counter_patch_entry(3, rect, np.float64(0.004), np.array([0.5, 0.5, 0.5]))
    assert entry["args"]["name"] == "counter_patch_3"
    assert entry["args"]["scale"] == pytest.approx([1.5, 0.5, 0.004], abs=1e-12)
    assert all(type(v) is float for v in entry["args"]["scale"] + entry["args"]["rgba"])
    json.dumps(entry)
    state = sb.counter_patch_state(rect, np.float64(-0.0005), np.float64(0.004))
    assert state["root_link"]["pos"] == pytest.approx([-0.25, 0.0, -0.0025], abs=1e-12)
    assert all(type(v) is float for v in state["root_link"]["pos"] + state["root_link"]["ori"])
    json.dumps(state)


def test_the_module_imports_only_numpy_and_the_standard_library_at_load_time():
    # A check on the code as written: the module must load where pxr and omnigibson are absent. The one pxr import,
    # in _vec4f, is inside a function, so it is not in the module body; omnigibson is not imported anywhere.
    tree = ast.parse((ROOT / "simfoundry" / "utils" / "splat_background.py").read_text(encoding="utf-8"))
    loaded = set()
    for node in tree.body:  # module level only
        if isinstance(node, ast.Import):
            loaded.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            loaded.add(node.module.split(".")[0])
    outside = loaded - set(sys.stdlib_module_names) - {"numpy"}
    assert not outside, f"module-level import(s) beyond numpy and the standard library: {sorted(outside)}"
    everywhere = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    everywhere |= {alias.name.split(".")[0]
                   for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert "omnigibson" not in everywhere


# ------------------------------------------------------- I20: one set of checks, the reasons, and the minimum sizes


def _built_spec(tmp_path, usdz_file, **changes):
    """A spec built in code: the spec a file of ``_body()`` gives, with ``changes`` applied (``dataclasses.replace``)."""
    return dataclasses.replace(sb.load_visual_background(_spec_file(tmp_path, _body())), **changes)


@pytest.mark.parametrize("dz, reason", [
    (float("nan"), "NaN or infinite"),
    (float("inf"), "NaN or infinite"),
    (1.0, "is out of range"),
    (-0.5, "is out of range"),
    (True, "is a bool"),
    ("0.001", "is not a number"),
], ids=["nan", "inf", "too_big", "too_negative", "bool", "string"])
def test_a_hand_built_dz_that_is_not_allowed_is_refused_before_any_change(tmp_path, usdz_file, dz, reason):
    scene = _synthetic_scene()
    before = copy.deepcopy(scene)
    with pytest.raises(sb.SplatBackgroundError, match=reason):
        sb.swap_visual_background(scene, _built_spec(tmp_path, usdz_file, dz_m=dz, remove=("iter_7",)))
    assert scene == before


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_a_hand_built_kit_setting_that_is_not_finite_is_refused_before_any_change(tmp_path, usdz_file, value):
    scene = _synthetic_scene()
    before = copy.deepcopy(scene)
    spec = _built_spec(tmp_path, usdz_file, kit_settings={"/rtx/x": value}, remove=("iter_7",))
    with pytest.raises(sb.SplatBackgroundError, match="NaN or infinite"):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_a_hand_built_kit_key_that_is_not_a_string_is_a_refusal_not_an_attribute_error(tmp_path, usdz_file):
    scene = _synthetic_scene()
    before = copy.deepcopy(scene)
    spec = _built_spec(tmp_path, usdz_file, kit_settings={3: True}, remove=("iter_7",))
    with pytest.raises(sb.SplatBackgroundError, match="settings path"):
        sb.swap_visual_background(scene, spec)
    assert scene == before


@pytest.mark.parametrize("changes, reason", [
    ({"name": ""}, "non-empty string"),
    ({"name": None}, "non-empty string"),
    ({"color": {"mode": "x"}}, "not implemented yet"),
    ({"floor_plane_visible": "false"}, "true or false"),
], ids=["empty_name", "null_name", "colour", "floor_flag"])
def test_other_hand_built_fields_are_checked_before_any_change(tmp_path, usdz_file, changes, reason):
    scene = _synthetic_scene()
    before = copy.deepcopy(scene)
    spec = _built_spec(tmp_path, usdz_file, remove=("iter_7",), **changes)
    with pytest.raises(sb.SplatBackgroundError, match=reason):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_a_hand_built_spec_with_a_missing_splat_or_pose_file_is_refused_before_any_change(tmp_path, usdz_file):
    scene = _synthetic_scene()
    before = copy.deepcopy(scene)
    spec = _built_spec(tmp_path, usdz_file, usdz=tmp_path / "missing.usdz", remove=("iter_7",))
    with pytest.raises(sb.SplatBackgroundError, match="usdz not found"):
        sb.swap_visual_background(scene, spec)
    assert scene == before
    spec = _built_spec(tmp_path, usdz_file, pose_json=tmp_path / "missing.json", remove=("iter_7",))
    with pytest.raises(sb.SplatBackgroundError, match="pose_json not found"):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_a_valid_hand_built_spec_gives_the_same_scene_and_record_as_the_loaded_one(tmp_path, usdz_file):
    body = _body(remove=["mesh_background_0", "iter_7"], kit_settings={"/rtx/x": True, "/rtx/y": 2.5},
                 floor_plane_visible=True, dz_m=0.002)
    loaded = sb.load_visual_background(_spec_file(tmp_path, body))
    built = dataclasses.replace(loaded, remove=["mesh_background_0", "iter_7"], kit_settings=dict(loaded.kit_settings))
    a, b = _synthetic_scene(), _synthetic_scene()
    record_a, record_b = sb.swap_visual_background(a, loaded), sb.swap_visual_background(b, built)
    assert a == b and record_a == record_b


def test_a_hand_built_remove_given_as_a_list_is_accepted_and_recorded_as_a_list(tmp_path, usdz_file):
    scene = _synthetic_scene()
    record = sb.swap_visual_background(scene, _built_spec(tmp_path, usdz_file, remove=["iter_7"]))
    assert record["removed"] == ["iter_7"] and "iter_7" not in scene["objects_info"]["init_info"]


PATCH_REASONS = [
    ("rect_is_a_number", lambda p: p.update(rects=[1.0]), "must be a list of 4 numbers"),
    ("rect_has_three_numbers", lambda p: p.update(rects=[[-0.62, 0.05, -0.62]]), "has 3 numbers; it needs 4"),
    ("rect_bool", lambda p: p.update(rects=[[-0.62, True, -0.62, -0.10]]), "is a bool"),
    ("rect_nan", lambda p: p.update(rects=[[float("nan"), 0.05, -0.62, -0.10]]), "NaN or infinite"),
    ("rect_string", lambda p: p.update(rects=[["-0.62", 0.05, -0.62, -0.10]]), "is not a number"),
    ("rect_out_of_range", lambda p: p.update(rects=[[-5.5, 0.05, -0.62, -0.10]]), "is out of range"),
    ("rect_huge_int", lambda p: p.update(rects=[[10 ** 400, 10 ** 400 + 1, -0.62, -0.10]]), "is out of range"),
    ("nine_rects", lambda p: p.update(rects=[[-0.5, -0.4, -0.5, -0.4] for _ in range(9)]),
     "has 9 rectangles; it needs 1 to 8"),
    ("rgb_has_two_numbers", lambda p: p.update(rgb=[0.8, 0.8]), "has 2 numbers; it needs 3"),
    ("rgb_is_a_string", lambda p: p.update(rgb="0.8 0.8 0.8"), "must be a list of 3 numbers, got str"),
    ("rgb_nan", lambda p: p.update(rgb=[0.8, float("nan"), 0.8]), "NaN or infinite"),
    ("rgb_out_of_range", lambda p: p.update(rgb=[0.8, 0.8, 1.01]), "is out of range"),
    ("z_top_bool", lambda p: p.update(z_top_m=False), "is a bool"),
    ("z_top_nan", lambda p: p.update(z_top_m=float("nan")), "NaN or infinite"),
    ("z_top_out_of_range", lambda p: p.update(z_top_m=0.021), "is out of range"),
    ("thickness_nan", lambda p: p.update(thickness_m=float("nan")), "NaN or infinite"),
    ("thickness_zero", lambda p: p.update(thickness_m=0.0), "is out of range"),
]


@pytest.mark.parametrize("name, change, reason", PATCH_REASONS, ids=[case[0] for case in PATCH_REASONS])
def test_a_counter_patch_refusal_gives_its_reason_and_names_the_patch(tmp_path, usdz_file, name, change, reason):
    patch = _patch()
    change(patch)
    with pytest.raises(sb.SplatBackgroundError, match="counter_patch") as refused:
        sb.load_visual_background(_spec_file(tmp_path, _body(counter_patch=patch)))
    assert reason in str(refused.value)


MATRIX_REASONS = [
    ("entry_is_a_bool", lambda b: b["matrix"][0].__setitem__(3, True), "is a bool"),
    ("entry_is_a_string", lambda b: b["matrix"][0].__setitem__(3, "1.0"), "is not a number"),
    ("entry_is_nan", lambda b: b["matrix"][0].__setitem__(3, float("nan")), "NaN or infinite"),
    ("entry_too_large_for_a_float", lambda b: b["matrix"][0].__setitem__(3, 10 ** 400), "is out of range"),
]


@pytest.mark.parametrize("name, change, reason", MATRIX_REASONS, ids=[case[0] for case in MATRIX_REASONS])
def test_a_matrix_refusal_gives_its_reason(tmp_path, usdz_file, name, change, reason):
    body = _body()
    change(body)
    with pytest.raises(sb.SplatBackgroundError, match="matrix") as refused:
        sb.load_visual_background(_spec_file(tmp_path, body))
    assert reason in str(refused.value)


DZ_REASONS = [
    ("dz_bool", lambda b: b.update(dz_m=True), "is a bool"),
    ("dz_string", lambda b: b.update(dz_m="0.001"), "is not a number"),
    ("dz_nan", lambda b: b.update(dz_m=float("nan")), "NaN or infinite"),
    ("dz_too_big", lambda b: b.update(dz_m=0.021), "is out of range"),
]


@pytest.mark.parametrize("name, change, reason", DZ_REASONS, ids=[case[0] for case in DZ_REASONS])
def test_a_dz_refusal_gives_its_reason(tmp_path, usdz_file, name, change, reason):
    body = _body()
    change(body)
    with pytest.raises(sb.SplatBackgroundError, match="dz_m") as refused:
        sb.load_visual_background(_spec_file(tmp_path, body))
    assert reason in str(refused.value)


@pytest.mark.parametrize("rect", [[0.0, 0.001, 0.0, 0.1], [0.4, 0.401, -0.5, -0.499], [-0.5, 0.5, 0.2, 0.201],
                                  [0.008, 0.009, 0.0, 0.1], [0.0, 0.1, 0.017, 0.018]],
                         ids=["at_zero", "at_an_offset", "at_a_negative_offset", "x_float_difference_below_1mm",
                              "y_float_difference_below_1mm"])
def test_a_side_of_exactly_1_mm_is_accepted_at_any_offset(tmp_path, usdz_file, rect):
    # 0.009 - 0.008 is 0.0009999999999999992 in binary floats, so without the picometre rounding an exact millimetre is refused
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(counter_patch=_patch(rects=[rect]))))
    assert spec.counter_patch.rects == (tuple(rect),)


@pytest.mark.parametrize("rect, axis", [([0.0, 0.00099999, 0.0, 0.1], "x"), ([0.0, 0.1, 0.0, 0.00099999], "y"),
                                        ([0.4, 0.400999, -0.5, -0.4], "x")],
                         ids=["x_side", "y_side", "x_side_at_an_offset"])
def test_a_side_under_1_mm_is_refused(tmp_path, usdz_file, rect, axis):
    with pytest.raises(sb.SplatBackgroundError, match=f"too narrow in {axis}"):
        sb.load_visual_background(_spec_file(tmp_path, _body(counter_patch=_patch(rects=[rect]))))


def test_a_thickness_of_exactly_0_1_mm_is_accepted_and_just_under_it_is_refused(tmp_path, usdz_file):
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(counter_patch=_patch(thickness_m=1e-4))))
    assert spec.counter_patch.thickness_m == 1e-4
    with pytest.raises(sb.SplatBackgroundError, match="thickness_m is out of range"):
        sb.load_visual_background(_spec_file(tmp_path, _body(counter_patch=_patch(thickness_m=9.9999e-5)), "thin.json"))


def test_a_box_name_held_in_both_sections_is_named_with_both_sections(tmp_path, usdz_file):
    scene = _synthetic_scene()
    scene["objects_info"]["init_info"]["counter_patch_1"] = {"args": {}}
    scene["state"]["registry"]["object_registry"]["counter_patch_1"] = {"root_link": {}}
    before = copy.deepcopy(scene)
    with pytest.raises(sb.SplatBackgroundError, match="'counter_patch_1' object in init_info and object_registry"):
        sb.swap_visual_background(scene, _patched_spec(tmp_path))
    assert scene == before


def test_the_splat_name_held_only_in_the_registry_names_that_section(tmp_path, usdz_file):
    scene = _synthetic_scene()
    scene["state"]["registry"]["object_registry"][GS] = {"root_link": {}}
    before = copy.deepcopy(scene)
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"])))
    with pytest.raises(sb.SplatBackgroundError, match="'gs_background' object in object_registry"):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_a_removed_name_held_by_one_section_says_which_section_lacks_it(tmp_path, usdz_file):
    scene = _synthetic_scene()
    del scene["state"]["registry"]["object_registry"]["iter_7"]
    before = copy.deepcopy(scene)
    spec = sb.load_visual_background(_spec_file(tmp_path, _body(remove=["iter_7"])))
    with pytest.raises(sb.SplatBackgroundError, match="'iter_7' is in init_info but not in object_registry"):
        sb.swap_visual_background(scene, spec)
    assert scene == before


def test_the_minimum_sizes_are_named_constants():
    assert sb.MIN_RECT_SIDE_M == 1e-3 and sb.MIN_THICKNESS_M == 1e-4


def test_the_loader_and_the_swap_both_run_the_same_field_checks():
    tree = ast.parse((ROOT / "simfoundry" / "utils" / "splat_background.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("load_visual_background", "swap_visual_background"):
        called = {node.func.id for node in ast.walk(functions[name])
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert "_checked_spec" in called, f"{name} does not run _checked_spec"


def test_the_swap_checks_the_spec_before_it_reads_the_scene():
    tree = ast.parse((ROOT / "simfoundry" / "utils" / "splat_background.py").read_text(encoding="utf-8"))
    swap = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "swap_visual_background")
    has_docstring = isinstance(swap.body[0], ast.Expr) and isinstance(swap.body[0].value, ast.Constant)
    first = swap.body[1] if has_docstring else swap.body[0]
    assert isinstance(first, ast.Assign) and isinstance(first.value, ast.Call) and first.value.func.id == "_checked_spec"


def test_the_module_docstring_says_the_boxes_are_appended_too():
    assert "together with the counter-patch boxes when the spec has any" in " ".join(sb.__doc__.split())
