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
    assert {"write_background_usd", "add_background_object", "relativize_background_path", "make_scene_cfg"} <= names


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
