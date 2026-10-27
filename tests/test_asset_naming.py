# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model names of stage 11's assets: the old rule kept bit for bit, the content hash, and the stage-11 wiring.

CPU only and without OmniGibson: the naming rules are pure functions, and stage 11 is read as source.
"""

import ast
import pathlib
import random
import string

import pytest
import yaml

from simfoundry.utils import asset_naming as naming

REPO = pathlib.Path(__file__).resolve().parents[1]
STAGE11 = REPO / "scripts/pipeline/A_reconstruction/stages/11_make_objects_sim_ready.py"
CFG = REPO / "scripts/cfg/real2sim_cfg.yaml"


def _old_generate_seeded_random_letters(seed_value, length=6):
    """Stage 11's function before asset_naming existed: it reseeds the global random state."""
    random.seed(seed_value)
    return "".join(random.choices(string.ascii_lowercase, k=length))


# 1. the old rule, bit for bit ------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("seed", "letters"), [
    ("lidar_kitchen_full_lidar_chef_knife_0", "pvsrwx"),  # the model the dataset holds for the chef knife
    ("lidar_kitchen_full_lidar_chef_knife_1", "doaajk"),
    ("lidar_kitchen_full_lidar_chef_knife_2", "gsttdx"),
    ("lidar_kitchen_full_lidar_cutting_board_0", "zmjqla"),
    ("lidar_kitchen_full_lidar_onion_0", "abnura"),
    ("lidar_kitchen_full_lidar_beige_tray_0", "nashzq"),
])
def test_the_letters_are_the_old_ones(seed, letters):
    assert naming.generate_seeded_random_letters(seed, length=6) == letters
    assert _old_generate_seeded_random_letters(seed, length=6) == letters


def test_the_old_seed_string_gives_the_old_name():
    assert naming.model_name("lidar_kitchen_full_lidar", "chef_knife", 0) == "pvsrwx"
    assert naming.model_name("scene", "chef_knife", 3) == _old_generate_seeded_random_letters("scene_chef_knife_3")


def test_the_global_random_state_is_not_touched():
    random.seed(12345)
    before = random.getstate()
    naming.generate_seeded_random_letters("some_seed", length=6)
    naming.model_name("scene", "chef_knife", 0, content_hash="abcdef12")
    assert random.getstate() == before


# 2. the content hash -----------------------------------------------------------------------------------------------------

def _file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_the_same_content_gives_the_same_hash_and_name(tmp_path):
    a = _file(tmp_path / "a" / "mesh.glb", b"abc")
    b = _file(tmp_path / "b" / "mesh.glb", b"abc")
    hash_a = naming.content_hash8([("mesh.glb", a)], 1.0, False)
    assert hash_a == naming.content_hash8([("mesh.glb", b)], 1.0, False)
    assert naming.model_name("scene", "cat", 0, hash_a) == naming.model_name("scene", "cat", 0, hash_a)


def test_one_changed_byte_gives_a_different_name(tmp_path):
    a = _file(tmp_path / "a.glb", b"abcd")
    b = _file(tmp_path / "b.glb", b"abce")
    hash_a = naming.content_hash8([("mesh.glb", a)], 1.0, False)
    hash_b = naming.content_hash8([("mesh.glb", b)], 1.0, False)
    assert hash_a != hash_b
    assert naming.model_name("scene", "cat", 0, hash_a) != naming.model_name("scene", "cat", 0, hash_b)


def test_scales_that_differ_in_the_last_bits_give_different_names(tmp_path):
    mesh = _file(tmp_path / "mesh.glb", b"abc")
    exact = naming.content_hash8([("mesh.glb", mesh)], 1.0, False)
    nudged = naming.content_hash8([("mesh.glb", mesh)], 1.0000001, False)
    assert exact != nudged
    assert naming.model_name("scene", "cat", 0, exact) != naming.model_name("scene", "cat", 0, nudged)


def test_a_rigid_and_an_articulated_asset_with_the_same_files_give_different_names(tmp_path):
    mesh = _file(tmp_path / "mesh.glb", b"abc")
    rigid = naming.content_hash8([("mesh.glb", mesh)], 1.0, False)
    articulated = naming.content_hash8([("mesh.glb", mesh)], 1.0, True)
    assert rigid != articulated


def test_the_same_content_under_another_scene_name_gives_another_name(tmp_path):
    mesh = _file(tmp_path / "mesh.glb", b"abc")
    content = naming.content_hash8([("mesh.glb", mesh)], 1.0, False)
    assert naming.model_name("scene_a", "cat", 0, content) != naming.model_name("scene_b", "cat", 0, content)


def test_the_order_of_the_files_does_not_change_the_hash(tmp_path):
    first = _file(tmp_path / "x" / "first.bin", b"1")
    second = _file(tmp_path / "x" / "second.bin", b"22")
    forward = naming.content_hash8([("a.bin", first), ("b.bin", second)], 1.0, True)
    backward = naming.content_hash8([("b.bin", second), ("a.bin", first)], 1.0, True)
    assert forward == backward


def test_a_renamed_file_with_the_same_bytes_gives_a_different_hash(tmp_path):
    mesh = _file(tmp_path / "mesh.glb", b"abc")
    assert naming.content_hash8([("mesh.glb", mesh)], 1.0, False) != naming.content_hash8([("other.glb", mesh)], 1.0, False)


def test_the_name_and_the_content_are_told_apart_so_different_splits_hash_differently(tmp_path):
    # "ab" + "c" and "a" + "bc" are the same bytes once the name and the content are simply joined; the NUL between
    # them keeps the two inputs apart.
    ab_c = _file(tmp_path / "one.bin", b"c")
    a_bc = _file(tmp_path / "two.bin", b"bc")
    assert naming.content_hash8([("ab", ab_c)], 1.0, False) != naming.content_hash8([("a", a_bc)], 1.0, False)


def test_articulated_inputs_include_the_parts_in_subfolders_of_the_parts_folder(tmp_path):
    results = tmp_path / "results"
    urdf = _file(results / "mobility.urdf", b"<robot/>")
    parts = results / "meshes"
    _file(parts / "top.obj", b"v 0 0 0")
    _file(parts / "sub" / "deep.obj", b"v 1 1 1")
    names = {name for name, _ in naming.articulated_input_files(results, urdf, parts)}
    assert names == {"mobility.urdf", "meshes/top.obj", "meshes/sub/deep.obj"}


def test_relative_names_must_be_unique(tmp_path):
    mesh = _file(tmp_path / "mesh.glb", b"abc")
    with pytest.raises(ValueError):
        naming.content_hash8([("same.glb", mesh), ("same.glb", mesh)], 1.0, False)


def test_the_content_hash_is_eight_hex_digits(tmp_path):
    mesh = _file(tmp_path / "mesh.glb", b"abc")
    digest = naming.content_hash8([("mesh.glb", mesh)], 1.0, False)
    assert len(digest) == 8 and all(ch in "0123456789abcdef" for ch in digest)


def test_articulated_inputs_are_named_relative_to_the_results_folder(tmp_path):
    results = tmp_path / "obj"
    urdf = _file(results / "mobility.urdf", b"<robot/>")
    part = _file(results / "meshes" / "lid.obj", b"v 0 0 0")
    pairs = dict(naming.articulated_input_files(str(results), str(urdf), str(results / "meshes")))
    assert set(pairs) == {"mobility.urdf", "meshes/lid.obj"}
    assert pairs["meshes/lid.obj"] == str(part)


# 3. the stage-11 wiring, read from its source ---------------------------------------------------------------------

def _stage11():
    return ast.parse(STAGE11.read_text())


def _main(tree):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def test_stage_11_imports_the_naming_functions_and_no_longer_defines_the_letters_function():
    tree = _stage11()
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "generate_seeded_random_letters" for n in tree.body)
    imported = {alias.name for n in tree.body if isinstance(n, ast.ImportFrom)
                and n.module == "simfoundry.utils.asset_naming" for alias in n.names}
    assert {"model_name", "content_hash8", "articulated_input_files"} <= imported


def test_the_name_is_computed_after_the_scale_and_before_its_first_use():
    main = _main(_stage11())
    scale_lines = [n.lineno for n in ast.walk(main) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in ("rigid_scale", "total_scale") for t in n.targets)]
    name_lines = [n.lineno for n in ast.walk(main) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "obj_model" for t in n.targets)]
    use_lines = [n.lineno for n in ast.walk(main) if isinstance(n, ast.Name) and n.id == "obj_model"
                 and isinstance(n.ctx, ast.Load)]
    assert scale_lines and name_lines and use_lines
    assert max(scale_lines) < min(name_lines) < min(use_lines)


def test_the_hash_is_taken_at_the_scale_the_files_are_written_at():
    # A rigid object is written at rigid_scale and an articulated one at tf_scale (its parts already carry
    # pre_scale_factor). Hashing any other factor would give two assets that differ by it the same name.
    main = _main(_stage11())
    parents = {child: node for node in ast.walk(main) for child in ast.iter_child_nodes(node)}
    written_at = {}
    for node in ast.walk(main):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "hash_scale" for t in node.targets):
            branch = parents[node]
            key = "articulated" if any(child is node for child in branch.body) else "rigid"
            written_at[key] = ast.unparse(node.value)
    assert written_at == {"articulated": "tf_scale", "rigid": "rigid_scale"}


def test_the_rigid_import_receives_the_name_by_keyword():
    main = _main(_stage11())
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "import_rigid_scene_object"]
    assert len(calls) == 1
    keywords = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert keywords["obj_model"] == "obj_model" and keywords["mesh_path"] == "mesh_path"


def test_the_hash_switch_is_read_from_the_config_with_the_yaml_default():
    main = _main(_stage11())
    reads = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "get" and ast.unparse(n.func.value).endswith("s11_sim")
             and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "model_name_content_hash"]
    assert len(reads) == 1
    assert ast.literal_eval(reads[0].args[1]) is True
    assert yaml.safe_load(CFG.read_text())["s11_sim"]["model_name_content_hash"] is True


def test_stage_11_hashes_the_exact_inputs_and_names_the_object_by_the_hash_switch():
    main = _main(_stage11())

    def values_of(name):
        return {ast.unparse(node.value) for node in ast.walk(main) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)}

    assert values_of("hash_files") == {"[(os.path.basename(mesh_path), mesh_path)]",
                                       "articulated_input_files(obj_results_dir, urdf_path, mesh_parts_dir)"}
    assert values_of("asset_content_hash") == {"content_hash8(hash_files, hash_scale, is_articulated)", "None"}
    assert values_of("obj_model") == {"model_name(cfg.scene_name, obj_category, idx, content_hash=asset_content_hash)",
                                      "model_name(cfg.scene_name, obj_category, idx)"}


def test_the_content_hash_is_recorded_only_when_the_name_uses_it():
    main = _main(_stage11())
    writes = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                      and t.slice.value == "asset_content_hash" for t in n.targets)]
    assert len(writes) == 2  # the rigid record and the articulated record
    parents = {child: node for node in ast.walk(main) for child in ast.iter_child_nodes(node)}
    for write in writes:
        node, guarded = write, False
        while node is not main:
            node = parents[node]
            if isinstance(node, ast.If) and "asset_content_hash" in ast.unparse(node.test):
                guarded = True
                break
        assert guarded, f"line {write.lineno} records the hash without a guard"
