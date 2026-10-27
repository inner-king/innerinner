# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The asset manifest (B15): folder hashes, the import decision, the conflict policy, the manifest file, and the stage-13
wiring. CPU only and without OmniGibson: the manifest module is pure, and stage 13 is read as source."""

import ast
import logging
import os
import pathlib

import pytest
import yaml

from simfoundry.pipeline import asset_manifest as am

REPO = pathlib.Path(__file__).resolve().parents[1]
STAGE13 = REPO / "scripts/pipeline/A_reconstruction/stages/13_import_usd.py"
CFG = REPO / "scripts/cfg/real2sim_cfg.yaml"
LOGGER = "simfoundry.pipeline.asset_manifest"
BASE = {
    "urdf/mesh.urdf": b"<robot><link><inertial><mass value='0.1'/></inertial></link></robot>",
    "meshes/mesh.obj": b"v 0 0 0",
    "textures/map.png": b"png",
}


def _tree(root, files):
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def _manifest(**overrides):
    record = {"geometry_hash": "g" * 64, "physics_hash": "p" * 64, "import_settings_hash": "s" * 16,
              "import_code_hash": "c" * 16, "scene_name": "old", "mass_kg": 0.2}
    record.update(overrides)
    return record


def _decision(conflict):
    return {"action": "import", "conflict": conflict, "reason": "r"}


def _describe(decision):
    return f"text for {decision['conflict']}"


# 1. the folder hashes -----------------------------------------------------------------------------------------------------

def test_a_urdf_change_moves_the_physics_hash_only(tmp_path):
    before = am.directory_hashes(_tree(tmp_path / "a", BASE))
    changed = dict(BASE, **{"urdf/mesh.urdf": b"<robot><link><inertial><mass value='0.2'/></inertial></link></robot>"})
    after = am.directory_hashes(_tree(tmp_path / "b", changed))
    assert before["geometry_hash"] == after["geometry_hash"]
    assert before["physics_hash"] != after["physics_hash"]


def test_a_mesh_change_moves_the_geometry_hash_only(tmp_path):
    before = am.directory_hashes(_tree(tmp_path / "a", BASE))
    after = am.directory_hashes(_tree(tmp_path / "b", dict(BASE, **{"meshes/mesh.obj": b"v 1 0 0"})))
    assert before["geometry_hash"] != after["geometry_hash"]
    assert before["physics_hash"] == after["physics_hash"]


def test_file_order_and_timestamps_do_not_count(tmp_path):
    first = am.directory_hashes(_tree(tmp_path / "a", BASE))
    root = _tree(tmp_path / "b", BASE)
    os.utime(root / "meshes" / "mesh.obj", (0, 0))
    assert am.directory_hashes(root) == first


def test_the_manifest_file_is_not_hashed(tmp_path):
    first = am.directory_hashes(_tree(tmp_path / "a", BASE))
    root = _tree(tmp_path / "b", BASE)
    (root / am.MANIFEST_NAME).write_text("{}")
    assert am.directory_hashes(root) == first


def test_subfolders_count_by_their_relative_path(tmp_path):
    first = am.directory_hashes(_tree(tmp_path / "a", dict(BASE, **{"x/y.bin": b"1"})))
    second = am.directory_hashes(_tree(tmp_path / "b", dict(BASE, **{"z/y.bin": b"1"})))
    assert first["geometry_hash"] != second["geometry_hash"]


def test_a_missing_folder_is_a_value_error(tmp_path):
    with pytest.raises(ValueError):
        am.directory_hashes(tmp_path / "missing")


def test_a_symbolic_link_counts_by_its_name_and_its_target_is_not_followed(tmp_path):
    root = _tree(tmp_path / "a", BASE)
    target = tmp_path / "outside.bin"
    target.write_bytes(b"one")
    os.symlink(target, root / "link.bin")
    first = am.directory_hashes(root)
    target.write_bytes(b"two")
    assert am.directory_hashes(root) == first
    os.replace(root / "link.bin", root / "renamed.bin")
    assert am.directory_hashes(root)["geometry_hash"] != first["geometry_hash"]


# 2. the settings, the code and the mass -----------------------------------------------------------------------------

def test_the_settings_hash_is_16_digits_and_does_not_depend_on_key_order():
    one = am.import_settings_hash({"opacity_threshold": 0.5, "asset_pipeline_materials": True})
    two = am.import_settings_hash({"asset_pipeline_materials": True, "opacity_threshold": 0.5})
    assert one == two and len(one) == 16
    assert one != am.import_settings_hash({"opacity_threshold": 0.6, "asset_pipeline_materials": True})


def test_the_code_hash_changes_with_a_file_or_a_version(tmp_path):
    code = tmp_path / "import.py"
    code.write_text("a = 1\n")
    base = am.import_code_hash([("import.py", code)], {"omnigibson": "3.8.0"})
    code.write_text("a = 2\n")
    assert am.import_code_hash([("import.py", code)], {"omnigibson": "3.8.0"}) != base
    code.write_text("a = 1\n")
    assert am.import_code_hash([("import.py", code)], {"omnigibson": "3.9.0"}) != base


def test_the_urdf_mass_is_the_sum_of_the_link_masses(tmp_path):
    urdf = tmp_path / "a.urdf"
    urdf.write_text("<robot><link><inertial><mass value='0.25'/></inertial></link>"
                    "<link><inertial><mass value='0.5'/></inertial></link></robot>")
    assert am.urdf_mass_kg(urdf) == pytest.approx(0.75)
    urdf.write_text("<robot/>")
    assert am.urdf_mass_kg(urdf) is None
    assert am.urdf_mass_kg(tmp_path / "missing.urdf") is None


# 3. the import decision ------------------------------------------------------------------------------------------------

def test_no_folder_is_an_import_without_a_conflict():
    decision = am.decide_import(False, None, _manifest())
    assert decision["action"] == "import" and decision["conflict"] is None


def test_a_folder_without_a_manifest_is_an_import_with_unknown_provenance():
    decision = am.decide_import(True, None, _manifest())
    assert decision["action"] == "import" and decision["conflict"] == "unknown_provenance"


def test_identical_records_are_a_skip_without_a_conflict():
    decision = am.decide_import(True, _manifest(), _manifest())
    assert decision["action"] == "skip" and decision["conflict"] is None and decision["reason"]


def test_a_different_geometry_is_the_geometry_conflict_before_any_other():
    new = _manifest(geometry_hash="h" * 64, import_settings_hash="t" * 16, physics_hash="q" * 64)
    decision = am.decide_import(True, _manifest(), new)
    assert decision["action"] == "import" and decision["conflict"] == "geometry"


def test_a_different_settings_or_code_hash_is_a_settings_conflict():
    assert am.decide_import(True, _manifest(), _manifest(import_settings_hash="x" * 16))["conflict"] == "settings"
    assert am.decide_import(True, _manifest(), _manifest(import_code_hash="x" * 16))["conflict"] == "settings"


def test_only_the_physics_differing_is_a_physics_conflict():
    decision = am.decide_import(True, _manifest(), _manifest(physics_hash="x" * 64))
    assert decision["action"] == "import" and decision["conflict"] == "physics"


def test_a_settings_change_is_reported_before_a_physics_change_when_both_differ():
    decision = am.decide_import(True, _manifest(), _manifest(import_settings_hash="x" * 16, physics_hash="x" * 64))
    assert decision["conflict"] == "settings"


# 4. the conflict policy ---------------------------------------------------------------------------------------------------

def test_fail_raises_for_a_geometry_conflict_and_only_that(caplog):
    with pytest.raises(am.AssetConflictError, match="text for geometry"):
        am.apply_conflict_policy(_decision("geometry"), "fail", _describe)
    for conflict in ("settings", "physics", "unknown_provenance"):
        am.apply_conflict_policy(_decision(conflict), "fail", _describe)


def test_warn_and_overwrite_log_a_geometry_conflict_at_different_levels(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        am.apply_conflict_policy(_decision("geometry"), "warn", _describe)
        am.apply_conflict_policy(_decision("geometry"), "overwrite", _describe)
    levels = [record.levelno for record in caplog.records if record.getMessage() == "text for geometry"]
    assert levels == [logging.WARNING, logging.INFO]


def test_the_other_conflicts_log_at_fixed_levels_whatever_the_policy(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        for conflict in ("settings", "unknown_provenance", "physics"):
            am.apply_conflict_policy(_decision(conflict), "overwrite", _describe)
    levels = {record.getMessage(): record.levelno for record in caplog.records}
    assert levels["text for settings"] == logging.WARNING
    assert levels["text for unknown_provenance"] == logging.WARNING
    assert levels["text for physics"] == logging.INFO


def test_no_conflict_logs_nothing_and_an_unknown_policy_is_a_value_error(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        am.apply_conflict_policy(_decision(None), "warn", _describe)
    assert not caplog.records
    with pytest.raises(ValueError):
        am.apply_conflict_policy(_decision("geometry"), "ignore", _describe)
    with pytest.raises(ValueError):
        am.apply_conflict_policy(_decision(None), "ignore", _describe)


def test_the_geometry_text_names_the_earlier_scene_the_hashes_and_what_changes():
    text = am.describe_decision(_decision("geometry"), "/data/chef_knife/pvsrwx",
                                _manifest(scene_name="lidar_a", geometry_hash="ab" * 32),
                                _manifest(geometry_hash="cd" * 32))
    for part in ("lidar_a", "abababab", "cdcdcdcd", "/data/chef_knife/pvsrwx", "point at this asset change"):
        assert part in text


# 5. the manifest file -----------------------------------------------------------------------------------------------

def test_a_manifest_round_trips_and_gets_a_utc_timestamp(tmp_path):
    am.write_manifest(tmp_path, {"geometry_hash": "g", "scene_name": "s"})
    back = am.read_manifest(tmp_path)
    assert back["geometry_hash"] == "g" and back["scene_name"] == "s"
    assert back["written_at"].endswith("+00:00")


def test_a_missing_manifest_is_none(tmp_path):
    assert am.read_manifest(tmp_path) is None


def test_a_corrupt_manifest_is_none_with_a_warning(tmp_path, caplog):
    (tmp_path / am.MANIFEST_NAME).write_text("{not json")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert am.read_manifest(tmp_path) is None
    assert any("cannot be read" in record.getMessage() for record in caplog.records)


def test_a_manifest_that_is_not_an_object_is_none(tmp_path):
    (tmp_path / am.MANIFEST_NAME).write_text("[1, 2]")
    assert am.read_manifest(tmp_path) is None


def test_a_failed_write_keeps_the_previous_manifest_and_leaves_no_partial_file(tmp_path, monkeypatch):
    am.write_manifest(tmp_path, {"geometry_hash": "old"})

    def broken(self, *args, **kwargs):
        # The write gets partway and then fails: a write straight to the manifest would leave this partial text behind.
        with open(self, "w") as handle:
            handle.write('{"geometry_hash": "new", "partial')
        raise RuntimeError("disk full")

    monkeypatch.setattr(pathlib.Path, "write_text", broken)
    with pytest.raises(RuntimeError):
        am.write_manifest(tmp_path, {"geometry_hash": "new"})
    monkeypatch.undo()
    assert am.read_manifest(tmp_path)["geometry_hash"] == "old"
    assert sorted(path.name for path in tmp_path.iterdir()) == [am.MANIFEST_NAME]


def test_write_manifest_keeps_a_timestamp_it_was_given(tmp_path):
    am.write_manifest(tmp_path, {"geometry_hash": "g", "written_at": "2026-01-02T03:04:05+00:00"})
    assert am.read_manifest(tmp_path)["written_at"] == "2026-01-02T03:04:05+00:00"


def test_remove_manifest_deletes_only_the_manifest(tmp_path):
    _tree(tmp_path, {"meshes/mesh.obj": b"v 0 0 0"})
    am.write_manifest(tmp_path, {"geometry_hash": "g"})
    assert am.remove_manifest(tmp_path) is True
    assert am.read_manifest(tmp_path) is None
    assert (tmp_path / "meshes" / "mesh.obj").read_bytes() == b"v 0 0 0"


def test_remove_manifest_without_a_manifest_is_false_and_not_an_error(tmp_path):
    _tree(tmp_path, {"meshes/mesh.obj": b"v 0 0 0"})
    assert am.remove_manifest(tmp_path) is False
    assert am.remove_manifest(tmp_path / "missing") is False
    assert (tmp_path / "meshes" / "mesh.obj").read_bytes() == b"v 0 0 0"


def test_a_symbolic_link_to_a_folder_is_named_and_not_walked_into(tmp_path):
    root = _tree(tmp_path / "a", BASE)
    outside = _tree(tmp_path / "outside", {"sub/x.obj": b"v 1 1 1"})
    os.symlink(outside, root / "linked")
    first = am.directory_hashes(root)
    (outside / "sub" / "x.obj").write_bytes(b"v 2 2 2")
    assert am.directory_hashes(root) == first


def test_a_urdf_in_a_subfolder_of_urdf_is_geometry_not_physics(tmp_path):
    before = am.directory_hashes(_tree(tmp_path / "a", dict(BASE, **{"urdf/sub/part.urdf": b"<robot/>"})))
    after = am.directory_hashes(_tree(tmp_path / "b", dict(BASE, **{"urdf/sub/part.urdf": b"<robot name='x'/>"})))
    assert before["physics_hash"] == after["physics_hash"]
    assert before["geometry_hash"] != after["geometry_hash"]


def test_the_physics_text_names_the_masses_before_then_after():
    text = am.describe_decision({"action": "import", "conflict": "physics", "reason": "r"}, "/d/a",
                                _manifest(mass_kg=0.2), _manifest(mass_kg=0.3))
    assert text == "physics changed for /d/a (mass 0.2 kg -> 0.3 kg); the asset is imported again"


# 6. the stage-13 wiring, read from its source -------------------------------------------------------------------------

def _stage13_main():
    tree = ast.parse(STAGE13.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def _lines_of_calls(main, predicate):
    return sorted(n.lineno for n in ast.walk(main) if isinstance(n, ast.Call) and predicate(n))


def _is_named(node, name):
    return isinstance(node.func, ast.Name) and node.func.id == name


def test_stage_13_decides_and_applies_the_policy_before_the_import_and_writes_the_manifest_before_the_copy():
    main = _stage13_main()
    decide = _lines_of_calls(main, lambda n: _is_named(n, "decide_import"))
    policy = _lines_of_calls(main, lambda n: _is_named(n, "apply_conflict_policy"))
    removes = _lines_of_calls(main, lambda n: _is_named(n, "remove_manifest"))
    imports = _lines_of_calls(main, lambda n: ast.unparse(n.func) == "subprocess.run" and n.args
                              and isinstance(n.args[0], ast.Name) and n.args[0].id == "import_cmd")
    post = _lines_of_calls(main, lambda n: ast.unparse(n.func) == "subprocess.run"
                           and "set_usd_opacity_threshold" in ast.unparse(n))
    writes = _lines_of_calls(main, lambda n: _is_named(n, "write_manifest"))
    copies = _lines_of_calls(main, lambda n: ast.unparse(n.func) == "shutil.copytree")
    assert len(decide) == 1 and len(policy) == 1 and len(removes) == 1 and len(imports) == 1 and len(writes) == 1
    assert post and len(copies) == 2  # the skip branch's copy and the copy after the import
    assert decide[0] < policy[0] < removes[0] < imports[0] < post[0] < writes[0] < copies[-1]


def test_stage_13_skips_the_import_only_in_the_skip_branch_of_skip_unchanged():
    main = _stage13_main()
    branches = [n for n in ast.walk(main) if isinstance(n, ast.If) and "skip_unchanged" in ast.unparse(n.test)
                and any(isinstance(c, ast.Constant) and c.value == "skip" for c in ast.walk(n.test))]
    assert len(branches) == 1
    body = branches[0].body
    assert any(isinstance(statement, ast.Continue) for statement in body)
    assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) == "subprocess.run"
                   for statement in body for n in ast.walk(statement))


def test_stage_13_defaults_are_the_yaml_defaults():
    main = _stage13_main()
    reads = {n.args[0].value: ast.literal_eval(n.args[1]) for n in ast.walk(main) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args
             and isinstance(n.args[0], ast.Constant) and n.args[0].value in ("on_conflict", "skip_unchanged")}
    assert reads == {"on_conflict": "warn", "skip_unchanged": False}
    section = yaml.safe_load(CFG.read_text())["s13_usd"]
    assert section["on_conflict"] == "warn" and section["skip_unchanged"] is False


def test_stage_13_records_the_decisions_in_stage_info_beside_the_existing_keys():
    main = _stage13_main()
    results = [n for n in ast.walk(main) if isinstance(n, ast.Call) and _is_named(n, "StageResult")]
    assert len(results) == 1
    keywords = {k.arg: k.value for k in results[0].keywords}
    assert isinstance(keywords["additional_info"], ast.Dict)
    assert [key.value for key in keywords["additional_info"].keys] == ["asset_imports"]


def test_stage_13_removes_the_old_manifest_in_the_loop_body_after_the_skip_branch():
    main = _stage13_main()
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For) and ast.unparse(n.iter).startswith("reversed("))
    skip = next(s for s in loop.body if isinstance(s, ast.If) and "skip_unchanged" in ast.unparse(s.test))
    removes = [s for s in loop.body if isinstance(s, ast.Expr) and ast.unparse(s.value) == "remove_manifest(dataset_obj_dir)"]
    assert len(removes) == 1
    assert loop.body.index(removes[0]) > loop.body.index(skip)


def test_stage_13_passes_the_manifests_and_the_policy_their_exact_arguments():
    main = _stage13_main()
    (read,) = [n for n in ast.walk(main) if isinstance(n, ast.Call) and _is_named(n, "read_manifest")]
    (decide,) = [n for n in ast.walk(main) if isinstance(n, ast.Call) and _is_named(n, "decide_import")]
    (policy,) = [n for n in ast.walk(main) if isinstance(n, ast.Call) and _is_named(n, "apply_conflict_policy")]
    (write,) = [n for n in ast.walk(main) if isinstance(n, ast.Call) and _is_named(n, "write_manifest")]
    assert [ast.unparse(a) for a in read.args] == ["dataset_obj_dir"]
    assert [ast.unparse(a) for a in decide.args] == ["os.path.isdir(dataset_obj_dir)", "existing", "new_manifest"]
    assert [ast.unparse(a) for a in policy.args] == ["decision", "on_conflict"]
    assert {k.arg: ast.unparse(k.value) for k in policy.keywords} == {
        "describe": "partial(describe_decision, asset_dir=dataset_obj_dir, existing=existing, new=new_manifest)"}
    assert [ast.unparse(a) for a in write.args] == ["dataset_obj_dir", "new_manifest"]


def test_stage_13_copies_the_asset_to_the_scene_in_the_skip_branch_and_after_the_import():
    main = _stage13_main()
    skip = next(n for n in ast.walk(main) if isinstance(n, ast.If) and "skip_unchanged" in ast.unparse(n.test))
    copies = [n for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func) == "shutil.copytree"]
    in_skip = [c for c in copies if any(c is n for n in ast.walk(skip))]
    assert len(copies) == 2 and len(in_skip) == 1
    for copy in copies:
        assert [ast.unparse(a) for a in copy.args] == ["dataset_obj_dir", "stage_obj_dir"]
        assert {k.arg: ast.unparse(k.value) for k in copy.keywords} == {"dirs_exist_ok": "True"}


def test_stage_13_writes_the_manifest_keys_its_decision_reads():
    main = _stage13_main()
    (new,) = [n.value for n in ast.walk(main) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "new_manifest" for t in n.targets)]
    assert isinstance(new, ast.Dict)
    assert {(ast.unparse(k) if k is not None else "**"): ast.unparse(v) for k, v in zip(new.keys, new.values)} == {
        "**": "directory_hashes(os.path.join(sim_dir, 'objects', obj_category, obj_model))",
        "'import_settings_hash'": "settings_hash",
        "'import_code_hash'": "code_hash",
        "'scene_name'": "cfg.scene_name",
        "'category'": "obj_category",
        "'model'": "obj_model",
        "'mass_kg'": "urdf_mass_kg(obj_urdf_fpath)",
    }


def test_stage_13_hashes_the_settings_it_imports_with_and_the_code_it_runs_under():
    main = _stage13_main()
    (settings,) = [n for n in ast.walk(main) if isinstance(n, ast.Call) and _is_named(n, "import_settings_hash")]
    (code,) = [n for n in ast.walk(main) if isinstance(n, ast.Call) and _is_named(n, "import_code_hash")]
    (record,) = settings.args
    assert {k.value: ast.unparse(v) for k, v in zip(record.keys, record.values)} == {
        "opacity_threshold": "opacity_threshold", "asset_pipeline_materials": "asset_pipeline_materials",
        "collision_method": "'none'"}
    files, versions = code.args
    assert {ast.unparse(item.elts[0]): ast.unparse(item.elts[1]) for item in files.elts} == {
        "'13_import_usd.py'": "__file__",
        "'og_asset_import.py'": "find_spec('simfoundry.utils.og_asset_import').origin",
        "'set_usd_opacity_threshold.py'": "resolve_pipeline_script('set_usd_opacity_threshold.py')",
        "'reparent_usd_joints.py'": "resolve_reparent_script()",
    }
    assert ast.unparse(versions) == "import_code_versions()"


def test_stage_13_records_the_action_conflict_and_reason_of_each_import():
    main = _stage13_main()
    (record,) = [n.value for n in ast.walk(main) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript)
                 and ast.unparse(n.targets[0].value) == "asset_imports"]
    assert {k.value: ast.unparse(v) for k, v in zip(record.keys, record.values)} == {
        "action": "decision['action']", "conflict": "decision['conflict']", "reason": "decision['reason']"}
