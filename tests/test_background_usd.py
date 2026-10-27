# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 14 attaches stage 5c's background mesh as `mesh_background_0`, the way the light editor would.

The USD tests need `pxr`, which the simfoundry env does not have; they run in simfoundry-editor, where
usd-core is installed (`PYTHONPATH=$PWD /home/inner/miniforge3/envs/simfoundry-editor/bin/python -B -m
pytest -p no:cacheprovider tests/test_background_usd.py -q -rs`). The rest runs in either env. Stage 14 itself needs
Isaac Sim, so its wiring is read from the source with ast.
"""

import ast
import json
import os
import pathlib
import struct
import sys
import types

import numpy as np
import pytest

from simfoundry.pipeline import background_usd as bu

REPO = pathlib.Path(__file__).resolve().parents[1]
STAGE14 = REPO / "scripts/pipeline/A_reconstruction/stages/14_create_og_scene.py"
CFG = REPO / "scripts/cfg/real2sim_cfg.yaml"
LIGHT_EDITOR = REPO / "scripts/interactive/light_editor"


# --------------------------------------------------------------------------------------
# Scene config and the scene JSON rewrite (no pxr)
# --------------------------------------------------------------------------------------

def test_the_background_names_are_the_editors():
    assert bu.BACKGROUND_NAME == "mesh_background_0"
    assert bu.BACKGROUND_CATEGORY == "mesh_background"


def test_the_default_scene_config_is_the_one_stage_14_always_built():
    assert bu.make_scene_cfg(False, False) == {
        "type": "Scene", "use_floor_plane": True, "floor_plane_visible": True, "use_skybox": True,
    }


def test_an_attached_mesh_hides_the_floor_plane_but_keeps_it_as_support():
    cfg = bu.make_scene_cfg(False, True)
    assert cfg["use_floor_plane"] is True
    assert cfg["floor_plane_visible"] is False
    assert cfg["use_skybox"] is True


def test_the_3dgs_background_still_hides_the_floor_and_the_sky():
    assert bu.make_scene_cfg(True, False) == {
        "type": "Scene", "use_floor_plane": True, "floor_plane_visible": False, "use_skybox": False,
    }


def _write_scene(path, usd_abs, name=bu.BACKGROUND_NAME):
    """A scene JSON in the shape OmniGibson saves: indent 4, no trailing newline, one absolute background path."""
    scene = {
        "versions": {"omnigibson": "test"},
        "objects_info": {"init_info": {
            "table_0": {
                "class_module": "omnigibson.objects.dataset_object", "class_name": "DatasetObject",
                "args": {"name": "table_0", "category": "table", "scale": [1.0, 1.0, 1.0],
                         "usd_path": "/data/table/model.usd"},
            },
            name: {
                "class_module": "omnigibson.objects.usd_object", "class_name": "USDObject",
                "args": {"name": name, "usd_path": usd_abs, "category": "mesh_background",
                         "fixed_base": True, "visual_only": True, "expected_file_hash": "0" * 32},
            },
        }},
        "state": {"registry": {"object_registry": {name: {
            "root_link": {"pos": [0.0, 0.0, 0.0], "ori": [0.0, 0.0, 0.0, 1.0]}, "is_asleep": False,
        }}}},
    }
    text = json.dumps(scene, indent=4)
    path.write_text(text)
    return text


def test_relativize_rewrites_the_background_path_and_nothing_else(tmp_path):
    json_path = tmp_path / "scene" / "reconstructed_og_scene.json"
    json_path.parent.mkdir()
    usd_abs = str(tmp_path / "scene" / "background" / "mesh_background_0.usd")
    before = _write_scene(json_path, usd_abs)

    assert bu.relativize_background_path(json_path) is True

    rel = os.path.relpath(usd_abs, json_path.parent)
    assert json_path.read_text() == before.replace(json.dumps(usd_abs), json.dumps(rel))
    assert rel == os.path.join("background", "mesh_background_0.usd")
    resolved = os.path.join(json_path.parent, json.loads(json_path.read_text())["objects_info"]["init_info"][
        bu.BACKGROUND_NAME]["args"]["usd_path"])
    assert os.path.normpath(resolved) == os.path.normpath(usd_abs)


def test_a_path_that_is_already_relative_is_left_alone(tmp_path):
    json_path = tmp_path / "reconstructed_og_scene.json"
    _write_scene(json_path, str(tmp_path / "background" / "mesh_background_0.usd"))
    bu.relativize_background_path(json_path)
    once = json_path.read_bytes()

    assert bu.relativize_background_path(json_path) is True
    assert json_path.read_bytes() == once


def test_a_missing_entry_writes_nothing_and_says_so(tmp_path):
    json_path = tmp_path / "reconstructed_og_scene.json"
    before = _write_scene(json_path, str(tmp_path / "background" / "mesh_background_0.usd"), name="mesh_background_7")

    assert bu.relativize_background_path(json_path) is False
    assert json_path.read_text() == before


def test_the_rewritten_scene_still_parses(tmp_path):
    json_path = tmp_path / "reconstructed_og_scene.json"
    _write_scene(json_path, str(tmp_path / "background" / "mesh_background_0.usd"))
    bu.relativize_background_path(json_path)
    scene = json.loads(json_path.read_text())
    assert scene["objects_info"]["init_info"]["table_0"]["args"]["usd_path"] == "/data/table/model.usd"


# --------------------------------------------------------------------------------------
# Configuration and the dry-run command line (the orchestrator needs omegaconf)
# --------------------------------------------------------------------------------------

def test_the_flag_is_off_by_default_in_the_config():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CFG.read_text())
    assert cfg["s14_og"]["include_bg_mesh"] is False


def test_the_override_is_added_only_when_the_flag_is_on_and_the_user_has_not_set_it():
    pytest.importorskip("omegaconf")
    from simfoundry.pipeline import orchestrator

    assert orchestrator.bg_mesh_overrides(["a=1"], False) == ["a=1"]
    assert orchestrator.bg_mesh_overrides(["a=1"], True) == ["a=1", "s14_og.include_bg_mesh=true"]
    assert orchestrator.bg_mesh_overrides(["s14_og.include_bg_mesh=false"], True) == ["s14_og.include_bg_mesh=false"]


def _dry_run_commands(capsys, bg_mesh, extra=()):
    pytest.importorskip("omegaconf")
    from simfoundry.pipeline import orchestrator

    orchestrator.run_pipeline(
        cwd=str(REPO), pipeline_name="reconstruction", input_mode="video", include_ids_csv=None,
        exclude_ids_csv=None, exec_mode="direct", python_bin="python", env_map={}, dry_run=True,
        stream_subseq_enabled=False, stream_start_stage=5, stream_end_stage=8,
        extra_overrides=list(extra), bg_mesh=bg_mesh,
    )
    return [line for line in capsys.readouterr().out.splitlines() if " cmd: " in line]


def test_the_dry_run_passes_the_flag_to_every_stage_when_bg_mesh_is_on(capsys):
    commands = _dry_run_commands(capsys, bg_mesh=True)
    assert len(commands) > 0
    assert all("s14_og.include_bg_mesh=true" in line for line in commands)
    assert any("[Stage 5c]" in line for line in commands) and any("[Stage 14]" in line for line in commands)


def test_the_dry_run_passes_nothing_when_bg_mesh_is_off(capsys):
    commands = _dry_run_commands(capsys, bg_mesh=False)
    assert len(commands) > 0
    assert not any("include_bg_mesh" in line for line in commands)


def test_a_value_the_user_gave_is_not_overwritten(capsys):
    commands = _dry_run_commands(capsys, bg_mesh=True, extra=["s14_og.include_bg_mesh=false"])
    assert all("s14_og.include_bg_mesh=false" in line for line in commands)
    assert not any("include_bg_mesh=true" in line for line in commands)


# --------------------------------------------------------------------------------------
# Stage 14 wiring, read from the source (it needs Isaac Sim to run)
# --------------------------------------------------------------------------------------

def _stage14():
    tree = ast.parse(STAGE14.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return tree, main


def _call_name(call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and _call_name(n) == name]


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _ifs_testing(main, name):
    return [n for n in ast.walk(main) if isinstance(n, ast.If) and name in _names(n.test)]


def test_the_background_module_is_imported_at_module_level():
    tree, _ = _stage14()
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "simfoundry.pipeline.background_usd"]
    assert len(imports) == 1
    names = {alias.name for alias in imports[0].names}
    assert {"write_background_usd", "add_background_object", "relativize_background_path", "make_scene_cfg",
            "resolve_background_exposure"} <= names


def test_the_scene_config_takes_the_attached_mesh_not_the_request():
    _, main = _stage14()
    calls = _calls(main, "make_scene_cfg")
    assert len(calls) == 1
    assert isinstance(calls[0].args[1], ast.Name) and calls[0].args[1].id == "attach_bg_mesh"
    assert calls[0].args[0].id == "include_gs"


def test_the_mesh_is_attached_only_when_it_is_requested_and_not_combined_with_3dgs_or_missing():
    _, main = _stage14()
    # Requested -> attached, then cleared by the 3DGS check and by a missing GLB, each in its own if.
    assert any(isinstance(n, ast.Assign) and [t.id for t in n.targets if isinstance(t, ast.Name)] == ["attach_bg_mesh"]
               and isinstance(n.value, ast.Name) and n.value.id == "include_bg_mesh"
               for n in ast.walk(main))
    cleared_by_gs = [i for i in _ifs_testing(main, "include_gs") if _clears_attach(i)]
    assert cleared_by_gs, "attach_bg_mesh must be cleared when include_gs is on"
    cleared_by_file = [i for i in ast.walk(main) if isinstance(i, ast.If) and _clears_attach(i)
                       and "exists" in ast.dump(i.test)]
    assert cleared_by_file, "attach_bg_mesh must be cleared when the GLB is missing"


def _clears_attach(if_node):
    return any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and n.value.value is False
               and any(isinstance(t, ast.Name) and t.id == "attach_bg_mesh" for t in n.targets)
               for n in if_node.body)


def _attach_block(main):
    """The `if attach_bg_mesh:` block that attaches the mesh (not the ones that clear it or check the save)."""
    blocks = [i for i in ast.walk(main) if isinstance(i, ast.If) and isinstance(i.test, ast.Name)
              and i.test.id == "attach_bg_mesh"]
    assert len(blocks) == 1
    return blocks[0]


def test_the_attach_block_writes_and_adds_the_mesh_under_the_flag():
    _, main = _stage14()
    block = _attach_block(main)
    assert _calls(block, "write_background_usd") and _calls(block, "add_background_object")


def _og_save(main):
    """The `og.sim.save(...)` call."""
    saves = [c for c in _calls(main, "save") if isinstance(c.func, ast.Attribute)
             and isinstance(c.func.value, ast.Attribute) and c.func.value.attr == "sim"]
    assert len(saves) == 1
    return saves[0]


def test_the_attach_block_comes_after_the_3dgs_block_and_before_the_save():
    _, main = _stage14()
    # `if include_gs:` by itself is the 3DGS block; `if not include_gs:` is a different one.
    gs_block = [i for i in ast.walk(main) if isinstance(i, ast.If) and isinstance(i.test, ast.Name)
                and i.test.id == "include_gs"]
    assert len(gs_block) == 1
    assert gs_block[0].lineno < _attach_block(main).lineno < _og_save(main).lineno


def test_the_saved_path_is_made_relative_after_the_save_and_only_when_attached():
    _, main = _stage14()
    save = _og_save(main)
    relativize = _calls(main, "relativize_background_path")
    assert len(relativize) == 1
    assert relativize[0].lineno > save.lineno
    guards = [i for i in ast.walk(main) if isinstance(i, ast.If) and _calls(i.test, "relativize_background_path")]
    assert len(guards) == 1 and "attach_bg_mesh" in _names(guards[0].test)


def test_the_bg_mesh_record_is_written_only_when_the_flag_is_on():
    _, main = _stage14()
    records = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                       and t.slice.value == "bg_mesh" for t in n.targets)]
    assert len(records) == 1
    guards = [i for i in ast.walk(main) if isinstance(i, ast.If) and records[0] in i.body]
    assert len(guards) == 1 and _names(guards[0].test) == {"include_bg_mesh"}


def test_the_aabb_calls_all_run_before_the_background_is_added():
    # Only visual-only objects have no aabb, so the background must come after every aabb read in stage 14.
    _, main = _stage14()
    added = min(c.lineno for c in _calls(main, "add_background_object"))
    aabb_lines = [n.lineno for n in ast.walk(main) if isinstance(n, ast.Attribute) and n.attr in {"aabb", "get_aabb"}]
    assert aabb_lines and max(aabb_lines) < added


# --------------------------------------------------------------------------------------
# USD writer (needs pxr)
# --------------------------------------------------------------------------------------

def test_the_object_is_a_fixed_visual_only_mesh_background():
    # OmniGibson cannot run here, so the constructor call is read from the source.
    tree = ast.parse(pathlib.Path(bu.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "USDObject"]
    assert len(calls) == 1
    keywords = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(keywords["category"], ast.Name) and keywords["category"].id == "BACKGROUND_CATEGORY"
    assert isinstance(keywords["fixed_base"], ast.Constant) and keywords["fixed_base"].value is True
    assert isinstance(keywords["visual_only"], ast.Constant) and keywords["visual_only"].value is True
    assert isinstance(keywords["name"], ast.Name) and keywords["name"].id == "name"


class _Recording:
    """Stands in for an OmniGibson object or module: every attribute path and call made on it is recorded."""

    def __init__(self, calls, path):
        self._calls = calls
        self._path = path

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Recording(self._calls, f"{self._path}.{name}")

    def __call__(self, *args, **kwargs):
        self._calls.append((self._path, args, kwargs))
        return _Recording(self._calls, self._path)


def test_adding_the_background_gives_the_object_no_pose(tmp_path, monkeypatch):
    # The mesh is already in the world frame, so the object keeps OmniGibson's identity pose. Any call on the object
    # (set_position_orientation, set_position, ...) would show up here.
    calls = []
    constructed = []

    class FakeUSDObject:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self._recorder = _Recording(calls, "obj")

        def __getattr__(self, name):
            return getattr(self._recorder, name)

    fake_og = types.ModuleType("omnigibson")
    fake_og.sim = _Recording(calls, "og.sim")
    fake_objects = types.ModuleType("omnigibson.objects")
    fake_objects.USDObject = FakeUSDObject
    monkeypatch.setitem(sys.modules, "omnigibson", fake_og)
    monkeypatch.setitem(sys.modules, "omnigibson.objects", fake_objects)

    usd = tmp_path / "background" / "mesh_background_0.usd"
    env = _Recording(calls, "env")
    obj = bu.add_background_object(env, usd)

    assert len(constructed) == 1
    assert constructed[0] == {
        "name": "mesh_background_0",
        "usd_path": str(usd.resolve()),
        "category": "mesh_background",
        "fixed_base": True,
        "visual_only": True,
    }
    assert [c[0] for c in calls] == ["env.scene.add_object", "og.sim.step"]
    assert calls[0][1][0] is obj
    assert not [c for c in calls if c[0].startswith("obj.")]


def test_the_usd_is_written_under_the_stage_output_dir_beside_the_scene_json():
    _, main = _stage14()

    def pieces(node):
        """An f-string as a list: ("name", id) for a plain name in braces, ("text", s) for literal text."""
        assert isinstance(node, ast.JoinedStr)
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(("text", value.value))
            else:
                assert isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name)
                parts.append(("name", value.value.id))
        return parts

    writes = _calls(main, "write_background_usd")
    assert len(writes) == 1
    assert pieces(writes[0].args[1]) == [("name", "out_dir"), ("text", "/background/"),
                                         ("name", "BACKGROUND_NAME"), ("text", ".usd")]
    json_paths = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "json_path" for t in n.targets)]
    assert len(json_paths) == 1
    assert pieces(json_paths[0].value) == [("name", "out_dir"), ("text", "/reconstructed_og_scene.json")]
    out_dirs = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "out_dir" for t in n.targets)]
    assert len(out_dirs) == 1 and ast.unparse(out_dirs[0].value) == "cfg.s14_og.out_dir"


def _world_mesh():
    """A 2 m by 1 m slab whose top is the floor (z = 0), with a 1 m block standing on it, one texture on both."""
    import trimesh
    from PIL import Image
    from trimesh.visual.material import SimpleMaterial
    from trimesh.visual.texture import TextureVisuals

    slab = trimesh.creation.box(extents=(2.0, 1.0, 0.1))
    slab.apply_translation([0.0, 0.0, -0.05])
    block = trimesh.creation.box(extents=(0.4, 0.4, 1.0))
    block.apply_translation([0.3, 0.0, 0.5])
    vertices = np.vstack([slab.vertices, block.vertices])
    faces = np.vstack([slab.faces, block.faces + len(slab.vertices)])
    uv = np.stack([(vertices[:, 0] + 1.0) / 2.0, vertices[:, 1] + 0.5], axis=1)
    pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    material = SimpleMaterial(image=Image.fromarray(pixels))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False,
                           visual=TextureVisuals(uv=uv, material=material))
    return mesh, pixels


def _glb(tmp_path):
    from simfoundry.pipeline.depth_mesh import export_glb

    mesh, pixels = _world_mesh()
    glb = tmp_path / "background_mesh.glb"
    export_glb(mesh, glb)
    return glb, mesh, pixels


def _write(tmp_path):
    glb, world, pixels = _glb(tmp_path)
    info = bu.write_background_usd(glb, tmp_path / "background" / "mesh_background_0.usd")
    return glb, world, pixels, info


def _glb_primitive(path):
    """The first primitive's POSITION and TEXCOORD_0 as stored in the GLB, read without trimesh."""
    data = pathlib.Path(path).read_bytes()
    json_len = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20 + json_len])
    binary = data[20 + json_len + 8:]
    attributes = document["meshes"][0]["primitives"][0]["attributes"]

    def accessor(index):
        spec = document["accessors"][index]
        view = document["bufferViews"][spec["bufferView"]]
        width = {"VEC2": 2, "VEC3": 3}[spec["type"]]
        offset = view.get("byteOffset", 0) + spec.get("byteOffset", 0)
        return np.frombuffer(binary, dtype="<f4", count=spec["count"] * width, offset=offset).reshape(spec["count"], width)

    return {"position": accessor(attributes["POSITION"]), "uv": accessor(attributes["TEXCOORD_0"])}


def _stage(info):
    from pxr import Usd

    return Usd.Stage.Open(info["usd_path"])


def test_the_usd_is_z_up_in_metres_with_the_background_as_its_default_prim(tmp_path):
    pytest.importorskip("pxr")
    from pxr import UsdGeom

    _, _, _, info = _write(tmp_path)
    stage = _stage(info)
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)
    assert stage.GetDefaultPrim().GetPath() == "/mesh_background"


def test_points_faces_and_bounds_match_the_world_mesh(tmp_path):
    pytest.importorskip("pxr")
    from pxr import UsdGeom

    _, world, _, info = _write(tmp_path)
    stage = _stage(info)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/mesh_background/base_link/mesh"))
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    assert len(points) == len(world.vertices)
    assert len(mesh.GetFaceVertexCountsAttr().Get()) == len(world.faces)
    assert np.allclose(points.min(axis=0), world.bounds[0], atol=1e-5)
    assert np.allclose(points.max(axis=0), world.bounds[1], atol=1e-5)
    nearest = np.linalg.norm(points[:, None, :] - world.vertices[None], axis=2).min(axis=1)
    assert np.all(nearest < 1e-5)


def test_the_floor_is_at_z_zero_and_the_block_top_at_its_height(tmp_path):
    pytest.importorskip("pxr")
    from pxr import UsdGeom

    glb, _, _, info = _write(tmp_path)
    stage = _stage(info)
    points = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath("/mesh_background/base_link/mesh")).GetPointsAttr().Get())
    assert points[:, 2].max() == pytest.approx(1.0, abs=1e-5)  # the block's top, 1 m above the floor
    assert points[:, 2].min() == pytest.approx(-0.1, abs=1e-5)  # the slab's underside, below it
    # The GLB is Y-up as glTF requires, so the same top is at y = 1 there: the writer rotated, not copied.
    assert _glb_primitive(glb)["position"][:, 1].max() == pytest.approx(1.0, abs=1e-5)


def test_uv_is_per_vertex_in_the_usd_convention_and_matches_the_glb(tmp_path):
    pytest.importorskip("pxr")
    from pxr import UsdGeom

    glb, _, _, info = _write(tmp_path)
    stage = _stage(info)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/mesh_background/base_link/mesh"))
    primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    assert primvar.IsDefined() and primvar.GetInterpolation() == UsdGeom.Tokens.vertex
    st = np.asarray(primvar.Get(), dtype=np.float64)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)

    raw = _glb_primitive(glb)
    raw_world = raw["position"].astype(np.float64) @ bu.GLTF_TO_WORLD[:3, :3].T
    # glTF's V counts down from the top of the image; USD's st counts up from the bottom.
    expected = np.stack([raw["uv"][:, 0], 1.0 - raw["uv"][:, 1]], axis=1).astype(np.float64)
    for i, point in enumerate(points):
        j = int(np.argmin(np.linalg.norm(raw_world - point, axis=1)))
        assert np.allclose(st[i], expected[j], atol=1e-5)


def test_the_texture_is_a_relative_reference_to_a_file_beside_the_usd(tmp_path):
    pytest.importorskip("pxr")
    from pxr import UsdShade
    from PIL import Image

    _, _, pixels, info = _write(tmp_path)
    assert info["texture"] == "./textures/mesh_background_0_basecolor.png"
    texture_file = tmp_path / "background" / "textures" / "mesh_background_0_basecolor.png"
    assert texture_file.is_file()
    assert np.array_equal(np.asarray(Image.open(texture_file).convert("RGB")), pixels)
    stage = _stage(info)
    texture = UsdShade.Shader(stage.GetPrimAtPath("/mesh_background/material/basecolor"))
    assert texture.GetInput("file").Get().path == "./textures/mesh_background_0_basecolor.png"


def test_the_mesh_is_bound_to_a_preview_surface_whose_colour_comes_from_the_texture(tmp_path):
    pytest.importorskip("pxr")
    from pxr import UsdGeom, UsdShade

    _, _, _, info = _write(tmp_path)
    stage = _stage(info)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/mesh_background/base_link/mesh"))
    bound = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()[0]
    assert bound and bound.GetPath() == "/mesh_background/material"
    surface = UsdShade.Shader(stage.GetPrimAtPath("/mesh_background/material/surface"))
    assert surface.GetIdAttr().Get() == "UsdPreviewSurface"
    source = surface.GetInput("diffuseColor").GetConnectedSource()
    assert source and source[0].GetPrim().GetPath().name == "basecolor"


def test_one_rigid_body_at_the_base_link_and_no_collision_or_mass(tmp_path):
    # OmniGibson needs exactly one rigid-body root link; nothing may make the background collide or weigh anything.
    pytest.importorskip("pxr")
    from pxr import UsdPhysics

    _, _, _, info = _write(tmp_path)
    stage = _stage(info)
    bodies = [p.GetPath().pathString for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert bodies == ["/mesh_background/base_link"]
    assert [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)] == []
    assert [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.MeshCollisionAPI)] == []
    assert [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.MassAPI)] == []


def test_the_prim_hierarchy_is_the_one_omnigibson_loads(tmp_path):
    pytest.importorskip("pxr")
    from pxr import UsdGeom

    _, _, _, info = _write(tmp_path)
    stage = _stage(info)
    paths = {p.GetPath().pathString for p in stage.Traverse()}
    assert paths == {
        "/mesh_background",
        "/mesh_background/base_link",
        "/mesh_background/base_link/mesh",
        "/mesh_background/material",
        "/mesh_background/material/surface",
        "/mesh_background/material/st",
        "/mesh_background/material/basecolor",
    }
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/mesh_background/base_link/mesh"))
    assert mesh.GetPrim().GetParent().GetPath().pathString == "/mesh_background/base_link"


def _scene_utils_loader():
    """The module's own loader, compiled from its source: importing scene_utils needs torch, which the editor env lacks.

    The module defines the name twice; the later definition is the one an import binds, so that is the one used.
    """
    tree = ast.parse((REPO / "simfoundry/utils/scene_utils.py").read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_json_with_absolute_usd_paths"]
    namespace = {"REPO_DIR": str(REPO)}
    exec(compile(ast.Module(body=[definitions[-1]], type_ignores=[]), "scene_utils.py", "exec"), namespace)
    return namespace["load_json_with_absolute_usd_paths"]


def test_a_loaded_scene_resolves_the_relative_path_whatever_the_working_directory(tmp_path, monkeypatch):
    # The consumer resolves the saved relative path against the JSON's own folder (load_json_with_absolute_usd_paths),
    # so the result is the USD the writer made, and it does not depend on where the process runs.
    pytest.importorskip("pxr")
    load_json_with_absolute_usd_paths = _scene_utils_loader()

    _, _, _, info = _write(tmp_path)
    json_path = tmp_path / "scene" / "reconstructed_og_scene.json"
    json_path.parent.mkdir()
    usd_abs = info["usd_path"]
    text = _write_scene(json_path, usd_abs)
    bu.relativize_background_path(json_path)

    monkeypatch.chdir(tmp_path / "scene")
    scene = load_json_with_absolute_usd_paths(str(json_path))
    resolved = scene["objects_info"]["init_info"][bu.BACKGROUND_NAME]["args"]["usd_path"]
    assert os.path.isabs(resolved)
    assert os.path.normpath(resolved) == os.path.normpath(usd_abs)
    assert pathlib.Path(resolved).is_file()
    assert scene["objects_info"]["init_info"]["table_0"]["args"]["usd_path"] == "/data/table/model.usd"

    monkeypatch.chdir(tmp_path)
    assert load_json_with_absolute_usd_paths(str(json_path)) == scene
    assert text != json_path.read_text()  # the file itself keeps the relative path


def test_the_editor_entry_has_the_shape_the_editor_writes_for_a_room(tmp_path, monkeypatch):
    pytest.importorskip("pxr")
    monkeypatch.syspath_prepend(str(LIGHT_EDITOR))
    import background_io
    import scene_io

    _, _, _, info = _write(tmp_path)
    json_path = tmp_path / "reconstructed_og_scene.json"
    _write_scene(json_path, info["usd_path"])
    bu.relativize_background_path(json_path)
    scene = json.loads(json_path.read_text())

    entry = scene["objects_info"]["init_info"][bu.BACKGROUND_NAME]
    spec = background_io.background_spec({"usd": info["usd_path"], "position": [0, 0, 0],
                                          "orientation": [0, 0, 0, 1]}, json_path)
    assert entry["class_module"] == spec["init_info"]["class_module"]
    assert entry["class_name"] == spec["init_info"]["class_name"]
    assert set(spec["init_info"]["args"]) <= set(entry["args"])
    assert entry["args"]["category"] == "mesh_background" and entry["args"]["fixed_base"] is True

    records = {r["name"]: r for r in scene_io.iter_objects(scene, json_path, usd_facts=False)}
    assert records[bu.BACKGROUND_NAME]["kind"] == "background"
    assert records[bu.BACKGROUND_NAME]["usd"] == pathlib.Path(info["usd_path"]).resolve()


# --------------------------------------------------------------------------------------
# Shading options: lit with a multiplier, unlit as an OmniPBR emission (B19)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("args, expected", [
    (("lit", None, None), ("lit", None, None)),
    (("lit", 0.5, None), ("lit", (0.5, 0.5, 0.5), None)),
    (("lit", (0.4, 0.5, 0.6), None), ("lit", (0.4, 0.5, 0.6), None)),
    (("unlit", None, None), ("unlit", None, 2700.0)),
    (("unlit", None, 1500), ("unlit", None, 1500.0)),
])
def test_the_accepted_options_resolve_to_their_values(args, expected):
    assert bu.resolve_background_exposure(*args) == expected


@pytest.mark.parametrize("args", [
    ("Lit", None, None), (" lit", None, None), ("", None, None), (None, None, None), (3, None, None),
    ("lit", None, 2700.0),
    ("unlit", 0.4, None),
    ("lit", 0, None), ("lit", -1, None), ("lit", float("nan"), None), ("lit", float("inf"), None),
    ("lit", True, None), ("lit", "0.4", None),
    ("lit", [0.4, 0.4], None), ("lit", [0.4] * 4, None), ("lit", [[0.4] * 3], None), ("lit", [], None),
    ("unlit", None, 0), ("unlit", None, -1), ("unlit", None, float("nan")), ("unlit", None, True),
    ("unlit", None, "2700"), ("unlit", None, [2700.0] * 3),
    # Each channel of a sequence is checked on its own, not only the length.
    ("lit", [0.4, 0, 0.4], None), ("lit", [0.4, -1, 0.4], None), ("lit", [0.4, float("nan"), 0.4], None),
    ("lit", [0.4, True, 0.4], None), ("lit", [0.4, "0.5", 0.4], None),
])
def test_the_refused_options_are_refused(args):
    with pytest.raises(ValueError):
        bu.resolve_background_exposure(*args)


@pytest.mark.parametrize("args", [
    ("lit", None, None), ("lit", 0.5, None), ("lit", (0.4, 0.5, 0.6), None),
    ("unlit", None, None), ("unlit", None, 1500),
])
def test_resolving_a_result_again_changes_nothing(args):
    once = bu.resolve_background_exposure(*args)
    assert bu.resolve_background_exposure(*once) == once


def test_an_omegaconf_list_counts_as_three_channels():
    omegaconf = pytest.importorskip("omegaconf")
    assert bu.resolve_background_exposure("lit", omegaconf.ListConfig([0.4, 0.5, 0.6]), None) == (
        "lit", (0.4, 0.5, 0.6), None)


def test_an_omegaconf_list_with_a_bad_channel_is_refused():
    omegaconf = pytest.importorskip("omegaconf")
    with pytest.raises(ValueError):
        bu.resolve_background_exposure("lit", omegaconf.ListConfig([0.4, 0, 0.4]), None)


def test_the_unlit_default_and_the_shading_names_are_pinned():
    assert bu.UNLIT_DEFAULT_EMISSIVE_INTENSITY == 2700.0
    assert bu.SHADINGS == ("lit", "unlit")


def _shader_inputs(stage):
    """Every input of every shader in the file as (shader path, input name, type, value, connection targets), sorted.

    The value is the authored one in plain Python: an asset is its path and a vector is a tuple. It is None for an input
    that is connected rather than given a value.
    """
    from pxr import Sdf, UsdShade

    rows = []
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            for item in UsdShade.Shader(prim).GetInputs():
                value = item.Get()
                if isinstance(value, Sdf.AssetPath):
                    value = value.path
                elif hasattr(value, "__len__") and not isinstance(value, str):
                    value = tuple(value)
                targets = [str(path) for path in item.GetAttr().GetConnections()]
                rows.append((str(prim.GetPath()), item.GetBaseName(), str(item.GetTypeName()), value, targets))
    return sorted(rows, key=lambda row: row[:2])


# The default material as it was before the options: UsdPreviewSurface, the st reader and the texture, nothing else.
# The values are pinned with the names: a different roughness or colour space changes every background written so far.
DEFAULT_LIT_INPUTS = [
    ("/mesh_background/material/basecolor", "file", "asset", "./textures/mesh_background_0_basecolor.png", []),
    ("/mesh_background/material/basecolor", "sourceColorSpace", "token", "sRGB", []),
    ("/mesh_background/material/basecolor", "st", "float2", None, ["/mesh_background/material/st.outputs:result"]),
    ("/mesh_background/material/st", "varname", "token", "st", []),
    ("/mesh_background/material/surface", "diffuseColor", "color3f", None,
     ["/mesh_background/material/basecolor.outputs:rgb"]),
    ("/mesh_background/material/surface", "metallic", "float", 0.0, []),
    ("/mesh_background/material/surface", "roughness", "float", 1.0, []),
]


def test_the_default_writes_the_shaders_and_inputs_it_always_wrote(tmp_path):
    pytest.importorskip("pxr")
    _, _, _, info = _write(tmp_path)
    assert _shader_inputs(_stage(info)) == DEFAULT_LIT_INPUTS
    assert (info["shading"], info["exposure_scale"], info["emissive_intensity"]) == ("lit", None, None)


def test_the_same_input_writes_the_same_bytes(tmp_path):
    pytest.importorskip("pxr")
    glb, _, _ = _glb(tmp_path)
    first = bu.write_background_usd(glb, tmp_path / "first" / "mesh_background_0.usd")
    second = bu.write_background_usd(glb, tmp_path / "second" / "mesh_background_0.usd")
    assert pathlib.Path(first["usd_path"]).read_bytes() == pathlib.Path(second["usd_path"]).read_bytes()


@pytest.mark.parametrize("scale, expected", [
    (0.5, (0.5, 0.5, 0.5)),
    ((0.4, 0.5, 0.6), (0.4, 0.5, 0.6)),
])
def test_lit_with_an_exposure_scale_multiplies_the_texture_and_keeps_the_diffuse_connection(tmp_path, scale, expected):
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, UsdShade

    glb, _, _ = _glb(tmp_path)
    info = bu.write_background_usd(glb, tmp_path / "background" / "mesh_background_0.usd", exposure_scale=scale)
    stage = _stage(info)
    scale_input = UsdShade.Shader(stage.GetPrimAtPath("/mesh_background/material/basecolor")).GetInput("scale")
    assert scale_input.GetTypeName() == Sdf.ValueTypeNames.Float4
    assert scale_input.Get() == Gf.Vec4f(*expected, 1.0)
    rows = _shader_inputs(stage)
    # The value of `scale` is checked above; here only its name, type and place matter.
    assert [row[:3] + row[4:] for row in rows if row[1] == "scale"] == [
        ("/mesh_background/material/basecolor", "scale", "float4", [])]
    assert ("/mesh_background/material/surface", "diffuseColor", "color3f", None,
            ["/mesh_background/material/basecolor.outputs:rgb"]) in rows
    assert not any("omnipbr" in row[0] for row in rows)
    assert (info["shading"], info["exposure_scale"]) == ("lit", list(expected))


def test_unlit_writes_one_omnipbr_material_with_the_documented_inputs(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, UsdShade

    glb, _, _ = _glb(tmp_path)
    info = bu.write_background_usd(glb, tmp_path / "background" / "mesh_background_0.usd", shading="unlit")
    stage = _stage(info)
    shader = UsdShade.Shader(stage.GetPrimAtPath("/mesh_background/material/omnipbr"))
    assert shader.GetImplementationSource() == "sourceAsset"
    assert shader.GetSourceAsset("mdl") == Sdf.AssetPath("OmniPBR.mdl")
    assert shader.GetSourceAssetSubIdentifier("mdl") == "OmniPBR"
    values = {}
    for item in shader.GetInputs():
        value = item.Get()
        values[item.GetBaseName()] = (str(item.GetTypeName()), value.path if isinstance(value, Sdf.AssetPath) else value)
    assert values == {
        "diffuse_color_constant": ("color3f", Gf.Vec3f(0, 0, 0)),
        "enable_emission": ("bool", True),
        "emissive_color": ("color3f", Gf.Vec3f(1, 1, 1)),
        "emissive_color_texture": ("asset", "./textures/mesh_background_0_basecolor.png"),
        "emissive_intensity": ("float", 2700.0),
        "specular_level": ("float", 0.0),
        "reflection_roughness_constant": ("float", 1.0),
        "metallic_constant": ("float", 0.0),
    }
    assert (pathlib.Path(info["usd_path"]).parent / info["texture"]).is_file()

    material = UsdShade.Material(stage.GetPrimAtPath("/mesh_background/material"))
    assert [str(path) for path in material.GetSurfaceOutput("mdl").GetAttr().GetConnections()] == [
        "/mesh_background/material/omnipbr.outputs:out"]
    universal = material.GetSurfaceOutput()
    assert not universal or not universal.HasConnectedSource()
    shader_ids = {UsdShade.Shader(prim).GetShaderId() for prim in stage.Traverse() if prim.IsA(UsdShade.Shader)}
    assert shader_ids == {None}
    bound = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/mesh_background/base_link/mesh"))
    assert bound.GetDirectBinding().GetMaterialPath().pathString == "/mesh_background/material"


def test_unlit_takes_the_provisional_intensity_unless_one_is_given(tmp_path):
    pytest.importorskip("pxr")
    from pxr import UsdShade

    def intensity(info):
        stage = _stage(info)  # kept in a name: a prim is invalid once the stage it came from is gone
        shader = UsdShade.Shader(stage.GetPrimAtPath("/mesh_background/material/omnipbr"))
        return shader.GetInput("emissive_intensity").Get()

    glb, _, _ = _glb(tmp_path)
    default = bu.write_background_usd(glb, tmp_path / "default" / "mesh_background_0.usd", shading="unlit")
    given = bu.write_background_usd(glb, tmp_path / "given" / "mesh_background_0.usd", shading="unlit",
                                    emissive_intensity=1500)
    assert (default["emissive_intensity"], intensity(default)) == (2700.0, 2700.0)
    assert (given["emissive_intensity"], intensity(given)) == (1500.0, 1500.0)


def test_a_glb_without_a_texture_keeps_its_colour_by_default_and_refuses_unlit_and_exposure(tmp_path):
    pytest.importorskip("pxr")
    import trimesh

    from simfoundry.pipeline.depth_mesh import export_glb

    glb = tmp_path / "plain.glb"
    export_glb(trimesh.creation.box(extents=(1.0, 1.0, 1.0)), glb)
    info = bu.write_background_usd(glb, tmp_path / "default" / "mesh_background_0.usd")
    assert info["texture"] is None
    refused = tmp_path / "refused" / "mesh_background_0.usd"
    with pytest.raises(ValueError, match="needs a base-colour texture"):
        bu.write_background_usd(glb, refused, shading="unlit")
    with pytest.raises(ValueError, match="needs a base-colour texture"):
        bu.write_background_usd(glb, refused, exposure_scale=0.5)
    assert not refused.exists()


def test_a_glb_without_a_texture_takes_its_colour_from_the_base_colour_factor(tmp_path):
    pytest.importorskip("pxr")
    import trimesh
    from pxr import UsdShade
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    from simfoundry.pipeline.depth_mesh import export_glb

    def diffuse(info):
        stage = _stage(info)
        shader = UsdShade.Shader(stage.GetPrimAtPath("/mesh_background/material/surface"))
        return tuple(shader.GetInput("diffuseColor").Get())

    coloured = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    coloured.visual = TextureVisuals(material=PBRMaterial(baseColorFactor=[200, 100, 50, 255]))
    coloured_glb = tmp_path / "coloured.glb"
    export_glb(coloured, coloured_glb)
    info = bu.write_background_usd(coloured_glb, tmp_path / "coloured" / "mesh_background_0.usd")
    assert info["texture"] is None
    assert diffuse(info) == pytest.approx((200 / 255, 100 / 255, 50 / 255), abs=1e-6)

    # A GLB with no material at all keeps the writer's grey fallback.
    plain_glb = tmp_path / "plain.glb"
    export_glb(trimesh.creation.box(extents=(1.0, 1.0, 1.0)), plain_glb)
    plain = bu.write_background_usd(plain_glb, tmp_path / "plain" / "mesh_background_0.usd")
    assert diffuse(plain) == pytest.approx((0.8, 0.8, 0.8), abs=1e-6)


def test_the_result_keeps_its_keys_and_adds_the_options(tmp_path):
    pytest.importorskip("pxr")
    glb, _, _ = _glb(tmp_path)
    lit = bu.write_background_usd(glb, tmp_path / "lit" / "mesh_background_0.usd", exposure_scale=0.5)
    unlit = bu.write_background_usd(glb, tmp_path / "unlit" / "mesh_background_0.usd", shading="unlit")
    for info in (lit, unlit):
        assert {"usd_path", "texture", "vertices", "faces", "bounds"} <= set(info)
        assert {"shading", "exposure_scale", "emissive_intensity"} <= set(info)
        json.dumps(info)
    assert (lit["shading"], lit["exposure_scale"], lit["emissive_intensity"]) == ("lit", [0.5, 0.5, 0.5], None)
    assert (unlit["shading"], unlit["exposure_scale"], unlit["emissive_intensity"]) == ("unlit", None, 2700.0)


def _resolve_assignment_targets(main):
    """The names the stage's resolve call assigns: the checked options that the write call must take."""
    assigns = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
               and _call_name(n.value) == "resolve_background_exposure"]
    assert len(assigns) == 1 and isinstance(assigns[0].targets[0], ast.Tuple)
    return [t.id for t in assigns[0].targets[0].elts]


def test_the_options_are_checked_once_before_isaac_sim_and_outside_any_loop_or_flag():
    _, main = _stage14()
    calls = _calls(main, "resolve_background_exposure")
    assert len(calls) == 1
    call = calls[0]
    environments = [c for c in _calls(main, "Environment") if ast.unparse(c.func.value) == "og"]
    assert len(environments) == 1 and call.lineno < environments[0].lineno
    loops = [n for n in ast.walk(main) if isinstance(n, (ast.For, ast.While))]
    assert not [n for n in loops if n.lineno <= call.lineno <= n.end_lineno]
    flagged = [n for n in ast.walk(main) if isinstance(n, ast.If) and {"include_bg_mesh", "attach_bg_mesh"} & _names(n.test)]
    assert not [n for n in flagged if n.lineno <= call.lineno <= n.end_lineno]


def test_the_write_call_takes_the_checked_options_and_nothing_else_changes_under_the_flag():
    _, main = _stage14()
    assert _resolve_assignment_targets(main) == ["bg_mesh_shading", "bg_mesh_exposure_scale", "bg_mesh_emissive_intensity"]
    writes = _calls(main, "write_background_usd")
    assert len(writes) == 1
    assert writes[0] in list(ast.walk(_attach_block(main)))
    options = {keyword.arg: ast.unparse(keyword.value) for keyword in writes[0].keywords}
    assert options == {"shading": "bg_mesh_shading", "exposure_scale": "bg_mesh_exposure_scale",
                       "emissive_intensity": "bg_mesh_emissive_intensity"}
    assert len(writes[0].args) == 2
    assert ast.unparse(writes[0].args[0]) == "bg_mesh_glb"
    assert isinstance(writes[0].args[1], ast.JoinedStr)


def test_the_config_is_read_with_the_defaults_the_yaml_states():
    _, main = _stage14()
    reads = {call.args[0].value: call.args[1] for call in _calls(main, "get")
             if ast.unparse(call.func.value) == "cfg.s14_og" and len(call.args) == 2
             and isinstance(call.args[0], ast.Constant) and str(call.args[0].value).startswith("bg_mesh_")}
    assert set(reads) == {"bg_mesh_shading", "bg_mesh_exposure_scale", "bg_mesh_emissive_intensity"}
    assert (reads["bg_mesh_shading"].value, reads["bg_mesh_exposure_scale"].value,
            reads["bg_mesh_emissive_intensity"].value) == ("lit", None, None)


def test_the_yaml_defaults_are_the_code_defaults():
    yaml = pytest.importorskip("yaml")
    s14 = yaml.safe_load(CFG.read_text())["s14_og"]
    assert (s14["bg_mesh_shading"], s14["bg_mesh_exposure_scale"], s14["bg_mesh_emissive_intensity"]) == (
        "lit", None, None)


def test_the_readme_section_states_the_defaults_and_recommends_no_option():
    import re

    text = (REPO / "scripts/pipeline/README.md").read_text()
    start = text.index("#### Background exposure (off by default)")
    heading = re.search(r"\n#{1,3} ", text[start + 1:])
    section = text[start:start + 1 + heading.start()] if heading else text[start:]
    intensity = int(bu.UNLIT_DEFAULT_EMISSIVE_INTENSITY)
    for fragment in (
        "The keys are `s14_og.bg_mesh_shading`, `s14_og.bg_mesh_exposure_scale` and "
        "`s14_og.bg_mesh_emissive_intensity`",
        f"`bg_mesh_emissive_intensity` sets the emission; null means {intensity}.",
        "one OmniPBR (MDL) shader whose emission is the texture",
    ):
        assert fragment in section
    assert "recommend" not in section.lower()


def test_the_background_is_added_from_the_usd_the_writer_returned():
    _, main = _stage14()
    calls = _calls(main, "add_background_object")
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[1]) == "bg_usd['usd_path']"  # ast.unparse writes string subscripts in single quotes


def test_the_record_is_made_under_the_flag_with_the_attached_and_usd_keys():
    _, main = _stage14()
    records = [n for n in ast.walk(_attach_block(main)) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "bg_mesh_record" for t in n.targets)]
    assert len(records) == 1 and isinstance(records[0].value, ast.Dict)
    assert [key.value for key in records[0].value.keys if key is not None] == ["attached", "usd"]
    assert any(key is None for key in records[0].value.keys)  # **bg_usd: the writer's keys come along
