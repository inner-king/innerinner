# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual background swap: the spec reader, the pose, the scene swap, and the splat's USDZ structure.

The module under test is ``simfoundry/utils/splat_background.py``. Most tests are synthetic and run in any env with
numpy and scipy. The tests that read the real files (the P3 scene JSON, the align matrix, the 58 MB USDZ) skip when a
file is missing. The USDZ structure test needs ``pxr``, which the simfoundry env does not have, so it skips there and
runs in simfoundry-editor. The ``edit_scene`` test imports ``scripts/cutting/tofu_og_cut.py``, which does not import
OmniGibson at module level, so it runs in the simfoundry env.
"""

import copy
import functools
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from simfoundry.utils import splat_background as sb

ROOT = Path(__file__).resolve().parents[1]
SCENE_JSON = Path(
    "/path/to/reconstructed_og_scene.json"
)
ALIGN_JSON = Path("/path/to/align_v4b.json")
PD_USDZ = Path("/path/to/PD_bg.usdz")
PD_PLY = Path("/path/to/splat.ply")
GS = "gs_background"
RECORD_KEYS = {"name", "usdz", "md5", "removed", "pos", "ori_xyzw", "scale", "dz_m", "floor_plane_visible", "kit_settings"}


def _need(path):
    if not Path(path).exists():
        pytest.skip(f"{path} is not on this machine")


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


@pytest.mark.skipif(not ALIGN_JSON.exists(), reason="align_v4b.json is not on this machine")
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


# ------------------------------------------------- spec reading and defaults


def test_defaults_are_the_documented_ones(tmp_path, usdz_file):
    spec = sb.load_visual_background(_spec_file(tmp_path, _body()))
    assert spec.dz_m == 0.0
    assert spec.floor_plane_visible is False
    assert spec.remove == ("mesh_background_0",)
    assert spec.kit_settings == {}
    assert spec.color is None
    assert spec.pose_json is None
    assert spec.usdz == usdz_file.resolve()
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
    assert spec.usdz == (other / "PD_bg.usdz").resolve()
    assert spec.pose_json == (other / "align.json").resolve()
    assert np.array_equal(spec.matrix, _rigid())


def test_absolute_paths_are_kept(tmp_path, usdz_file):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    spec = sb.load_visual_background(
        _spec_file(elsewhere, _body(usdz=str(usdz_file)))
    )
    assert spec.usdz == usdz_file.resolve()
    assert spec.usdz.is_absolute()


def test_tilde_is_not_expanded(tmp_path, usdz_file):
    with pytest.raises(sb.SplatBackgroundError, match="not found"):
        sb.load_visual_background(_spec_file(tmp_path, _body(usdz="~/PD_bg.usdz")))


def test_pose_json_is_read_and_recorded_as_an_absolute_path(tmp_path, usdz_file):
    (tmp_path / "align.json").write_text(json.dumps({"M_da3_to_scene": _rigid().tolist(), "scale": 1.0}))
    body = _body()
    body.pop("matrix")
    body["pose_json"] = "align.json"
    spec = sb.load_visual_background(_spec_file(tmp_path, body))
    assert spec.pose_json == (tmp_path / "align.json").resolve()
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
    _need(SCENE_JSON)
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
    _need(SCENE_JSON)
    _need(ALIGN_JSON)
    _need(PD_USDZ)
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
    assert entry["args"]["usd_path"] == str(PD_USDZ.resolve())
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
    assert record["usdz"] == str(usdz_file.resolve())
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


@pytest.mark.skipif(not PD_USDZ.exists(), reason="PD_bg.usdz is not on this machine")
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
    _need(SCENE_JSON)
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
    _need(PD_USDZ)
    _need(PD_PLY)
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
