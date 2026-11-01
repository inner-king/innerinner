# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which OmniGibson dataset root a run writes into: SIMFOUNDRY_DATASET_ROOT, the unchanged default, stage 13's check
and run.sh's --dataset-root. No GPU and no OmniGibson: stage 13 and the orchestrator are read as source."""

import ast
import os
import subprocess
from pathlib import Path

import pytest
import yaml

import simfoundry
from simfoundry import DATASET_ROOT_ENV, configure_omnigibson_data_path, dataset_root_source, get_omnigibson_data_path
from simfoundry.pipeline import dataset_root as dr
from simfoundry.pipeline import orchestrator as orch

REPO = Path(__file__).resolve().parents[1]
STAGE13 = REPO / "scripts/pipeline/A_reconstruction/stages/13_import_usd.py"
ORCHESTRATOR = REPO / "simfoundry/pipeline/orchestrator.py"
RUN_SH = REPO / "scripts/pipeline/A_reconstruction/run.sh"
CFG = REPO / "scripts/cfg/real2sim_cfg.yaml"
README = REPO / "scripts/pipeline/README.md"


def _old_configure(*, force: bool = True) -> str:
    """configure_omnigibson_data_path before SIMFOUNDRY_DATASET_ROOT: its body, with the default root inlined."""
    dataset_root = os.path.join(simfoundry.REPO_DIR, "deps", "BEHAVIOR-1K", "datasets")
    if os.path.isdir(dataset_root) and (force or not os.environ.get("OMNIGIBSON_DATA_PATH")):
        os.environ["OMNIGIBSON_DATA_PATH"] = dataset_root
    return os.environ.get("OMNIGIBSON_DATA_PATH", dataset_root)


@pytest.fixture
def fake_checkout(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    repo.mkdir()
    monkeypatch.setattr(simfoundry, "REPO_DIR", str(repo))
    monkeypatch.delenv(DATASET_ROOT_ENV, raising=False)
    monkeypatch.delenv("OMNIGIBSON_DATA_PATH", raising=False)
    return repo


# --- the default is unchanged ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("deps_present", [False, True])
@pytest.mark.parametrize("ambient", [None, "/shell/omnigibson/datasets"])
@pytest.mark.parametrize("force", [True, False])
def test_the_default_root_matches_the_old_body_in_all_eight_combinations(fake_checkout, monkeypatch, deps_present,
                                                                         ambient, force):
    if deps_present:
        (fake_checkout / "deps/BEHAVIOR-1K/datasets").mkdir(parents=True)
    results = []
    for configure in (_old_configure, lambda **kwargs: configure_omnigibson_data_path(**kwargs)):
        if ambient is None:
            monkeypatch.delenv("OMNIGIBSON_DATA_PATH", raising=False)
        else:
            monkeypatch.setenv("OMNIGIBSON_DATA_PATH", ambient)
        returned = configure(force=force)
        results.append((returned, os.environ.get("OMNIGIBSON_DATA_PATH")))
    assert results[0] == results[1]


def test_the_default_root_is_this_checkouts_deps_datasets(fake_checkout):
    assert get_omnigibson_data_path() == str(fake_checkout / "deps" / "BEHAVIOR-1K" / "datasets")
    assert dataset_root_source() == "checkout"


def test_the_shell_variable_is_not_read_when_no_root_is_named(fake_checkout, monkeypatch):
    monkeypatch.setenv("OMNIGIBSON_DATA_PATH", "/shell/other")
    assert get_omnigibson_data_path() == str(fake_checkout / "deps" / "BEHAVIOR-1K" / "datasets")


def test_an_empty_explicit_root_is_the_same_as_none(fake_checkout, monkeypatch):
    monkeypatch.setenv(DATASET_ROOT_ENV, "")
    assert get_omnigibson_data_path() == str(fake_checkout / "deps" / "BEHAVIOR-1K" / "datasets")
    assert dataset_root_source() == "checkout"


# --- an explicit root --------------------------------------------------------------------------------------------------

def test_an_explicit_root_is_absolute_and_expands_the_user_and_relative_paths(fake_checkout, monkeypatch, tmp_path):
    target = tmp_path / "shared" / "datasets"
    target.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(DATASET_ROOT_ENV, "~/shared/datasets")
    assert get_omnigibson_data_path() == str(target)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(DATASET_ROOT_ENV, "shared/datasets")
    assert get_omnigibson_data_path() == str(target)
    assert dataset_root_source() == "explicit"


@pytest.mark.parametrize("force", [True, False])
def test_an_explicit_root_goes_into_the_environment_over_an_ambient_value(fake_checkout, monkeypatch, tmp_path, force):
    target = tmp_path / "shared"
    target.mkdir()
    monkeypatch.setenv("OMNIGIBSON_DATA_PATH", "/shell/other")
    monkeypatch.setenv(DATASET_ROOT_ENV, str(target))
    assert configure_omnigibson_data_path(force=force) == str(target)
    assert os.environ["OMNIGIBSON_DATA_PATH"] == str(target)


def test_an_explicit_root_wins_over_the_checkouts_datasets(fake_checkout, monkeypatch, tmp_path):
    (fake_checkout / "deps/BEHAVIOR-1K/datasets").mkdir(parents=True)
    target = tmp_path / "shared"
    target.mkdir()
    monkeypatch.setenv(DATASET_ROOT_ENV, str(target))
    assert configure_omnigibson_data_path(force=True) == str(target)


def test_an_explicit_root_that_is_not_a_directory_stops_instead_of_falling_back(fake_checkout, monkeypatch, tmp_path):
    (fake_checkout / "deps/BEHAVIOR-1K/datasets").mkdir(parents=True)
    monkeypatch.setenv(DATASET_ROOT_ENV, str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match=r"SIMFOUNDRY_DATASET_ROOT=.*is not a directory"):
        get_omnigibson_data_path()


def test_the_source_is_explicit_only_when_the_variable_names_a_root(fake_checkout, monkeypatch, tmp_path):
    assert dataset_root_source() == "checkout"
    monkeypatch.setenv(DATASET_ROOT_ENV, str(tmp_path))
    assert dataset_root_source() == "explicit"


@pytest.mark.parametrize("explicit, env_value, expected", [
    (True, "/anything", "explicit"),
    (True, None, "explicit"),
    (False, "/checkout", "checkout"),
    (False, "/shell/other", "ambient"),
    (False, None, "ambient"),
])
def test_stage_13_names_the_source_of_its_root(explicit, env_value, expected):
    assert dr.source_of_dataset_root(explicit=explicit, env_value=env_value, checkout_value="/checkout") == expected


# --- the pure check ----------------------------------------------------------------------------------------------------

def _shared_link_checkout(tmp_path):
    repo = tmp_path / "repo"
    shared = tmp_path / "shared" / "datasets"
    (repo / "deps/BEHAVIOR-1K").mkdir(parents=True)
    shared.mkdir(parents=True)
    (repo / "deps/BEHAVIOR-1K/datasets").symlink_to(shared, target_is_directory=True)
    return repo, shared


def test_a_folder_inside_the_checkout_is_not_outside_and_gets_no_warning(tmp_path):
    repo = tmp_path / "repo"
    folder = repo / "deps/BEHAVIOR-1K/datasets/real2sim-assets"
    folder.mkdir(parents=True)
    description = dr.describe_dataset_root(folder, repo, "checkout")
    assert description["outside_checkout"] is False
    assert dr.check_dataset_root(description, allow_external=False) == (None, None)


def test_a_sibling_whose_name_starts_like_the_checkout_is_outside(tmp_path):
    repo = tmp_path / "repo"
    sibling = tmp_path / "repo2" / "datasets"
    repo.mkdir()
    sibling.mkdir(parents=True)
    assert dr.describe_dataset_root(sibling, repo, "checkout")["outside_checkout"] is True


def test_a_deps_link_out_of_the_checkout_is_outside_once_followed(tmp_path):
    repo, shared = _shared_link_checkout(tmp_path)
    description = dr.describe_dataset_root(repo / "deps/BEHAVIOR-1K/datasets/real2sim-assets", repo, "checkout")
    assert description["outside_checkout"] is True
    assert description["dataset_path"] == os.path.realpath(shared / "real2sim-assets")
    warning, stop = dr.check_dataset_root(description, allow_external=True)
    assert warning is not None and stop is None


def test_a_checkout_default_outside_the_checkout_warns_with_both_paths_and_both_fixes(tmp_path):
    repo, shared = _shared_link_checkout(tmp_path)
    description = dr.describe_dataset_root(repo / "deps/BEHAVIOR-1K/datasets/real2sim-assets", repo, "checkout")
    warning, _ = dr.check_dataset_root(description, allow_external=True)
    assert os.path.realpath(shared / "real2sim-assets") in warning
    assert os.path.realpath(repo) in warning
    assert "SIMFOUNDRY_DATASET_ROOT" in warning
    assert "--dataset-root" in warning
    assert "allow_external_dataset=false" in warning


def test_the_same_folder_stops_when_external_datasets_are_not_allowed(tmp_path):
    repo, _ = _shared_link_checkout(tmp_path)
    description = dr.describe_dataset_root(repo / "deps/BEHAVIOR-1K/datasets/real2sim-assets", repo, "checkout")
    warning, stop = dr.check_dataset_root(description, allow_external=False)
    assert stop.startswith(warning)
    assert "Refusing: s13_usd.allow_external_dataset is false" in stop


@pytest.mark.parametrize("source", ["explicit", "ambient"])
@pytest.mark.parametrize("allow_external", [True, False])
def test_an_explicit_or_ambient_root_outside_the_checkout_is_not_warned_about(tmp_path, source, allow_external):
    repo, _ = _shared_link_checkout(tmp_path)
    description = dr.describe_dataset_root(repo / "deps/BEHAVIOR-1K/datasets/real2sim-assets", repo, source)
    assert dr.check_dataset_root(description, allow_external=allow_external) == (None, None)


# --- the orchestrator hands the same root to every stage ---------------------------------------------------------------

def _stage_spec(stage_id="1"):
    return orch.StageSpec(stage_id, "scripts/pipeline/A_reconstruction/stages/1.py", "s1", "simfoundry", "one")


def _stage_env_root(monkeypatch, tmp_path, *, streaming):
    captured = {}

    def fake_run(cmd, cwd, check, env):
        captured.setdefault("env", env)

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    monkeypatch.setattr(orch, "query_gpu_memory_used_gb", lambda *args, **kwargs: None)
    if streaming:
        orch.run_stage_subsequence_streaming(
            stream_start_stage=5, stream_end_stage=6,
            env_map={"simfoundry": "simfoundry", "da3": "da3", "hunyuan": "hunyuan", "b1k": "b1k"},
            exec_mode="direct", python_bin="python", cwd=str(tmp_path), dry_run=False,
            extra_overrides=["scene_name=dining_1", f"root_dir={tmp_path}"], timing_log_path=None,
        )
    else:
        orch.run_stage(
            _stage_spec(), env_map={"simfoundry": "simfoundry"}, exec_mode="direct", python_bin="python",
            cwd=str(tmp_path), dry_run=False, extra_overrides=[], timing_log_path=None,
        )
    return captured["env"]["OMNIGIBSON_DATA_PATH"]


@pytest.mark.parametrize("streaming", [False, True])
def test_the_stages_get_the_default_root_when_nothing_is_named(fake_checkout, monkeypatch, tmp_path, streaming):
    assert _stage_env_root(monkeypatch, tmp_path, streaming=streaming) == str(
        fake_checkout / "deps" / "BEHAVIOR-1K" / "datasets")


@pytest.mark.parametrize("streaming", [False, True])
def test_the_stages_get_the_explicit_root(fake_checkout, monkeypatch, tmp_path, streaming):
    target = tmp_path / "shared"
    target.mkdir()
    monkeypatch.setenv(DATASET_ROOT_ENV, str(target))
    assert _stage_env_root(monkeypatch, tmp_path, streaming=streaming) == str(target)


def test_the_run_names_its_root_and_source_once_before_the_first_stage(fake_checkout, monkeypatch, capsys):
    monkeypatch.setattr(orch, "get_stage_plan", lambda **_kwargs: [_stage_spec("1"), _stage_spec("2")])
    orch.run_pipeline(
        cwd=".", input_mode="video", include_ids_csv=None, exclude_ids_csv=None, exec_mode="direct",
        python_bin="python", env_map={"simfoundry": "simfoundry"}, dry_run=True, stream_subseq_enabled=False,
        stream_start_stage=5, stream_end_stage=8, extra_overrides=["root_dir=/tmp/unused", "scene_name=s"],
    )
    lines = capsys.readouterr().out.splitlines()
    root_lines = [i for i, line in enumerate(lines) if line.startswith("[Pipeline] dataset root")]
    assert len(root_lines) == 1
    assert lines[root_lines[0]] == (
        f"[Pipeline] dataset root = {os.path.realpath(fake_checkout / 'deps/BEHAVIOR-1K/datasets')} (source: checkout)")
    assert root_lines[0] < min(i for i, line in enumerate(lines) if line.startswith("[Stage "))


def test_the_run_names_an_explicit_root_with_its_source(fake_checkout, monkeypatch, tmp_path, capsys):
    target = tmp_path / "shared"
    target.mkdir()
    monkeypatch.setenv(DATASET_ROOT_ENV, str(target))
    monkeypatch.setattr(orch, "get_stage_plan", lambda **_kwargs: [])
    orch.run_pipeline(
        cwd=".", input_mode="video", include_ids_csv=None, exclude_ids_csv=None, exec_mode="direct",
        python_bin="python", env_map={}, dry_run=True, stream_subseq_enabled=False, stream_start_stage=5,
        stream_end_stage=8, extra_overrides=["root_dir=/tmp/unused", "scene_name=s"],
    )
    assert f"[Pipeline] dataset root = {os.path.realpath(target)} (source: explicit)" in capsys.readouterr().out


def _function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_the_run_log_line_is_printed_once_and_before_the_stage_loop():
    tree = ast.parse(ORCHESTRATOR.read_text(encoding="utf-8"))
    run_pipeline = _function(tree, "run_pipeline")
    prints = [c for c in ast.walk(run_pipeline) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "print"
              and c.args and isinstance(c.args[0], ast.JoinedStr)
              and "[Pipeline] dataset root" in ast.unparse(c.args[0])]
    assert len(prints) == 1
    loops = [n for n in ast.walk(run_pipeline) if isinstance(n, ast.While)]
    assert prints[0].lineno < min(loop.lineno for loop in loops)


def test_the_stage_environments_still_take_the_configured_root_with_force():
    tree = ast.parse(ORCHESTRATOR.read_text(encoding="utf-8"))
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == "OMNIGIBSON_DATA_PATH"
        for t in n.targets)]
    assert [ast.unparse(a.value) for a in assigns] == ["configure_omnigibson_data_path(force=True)"] * 2


# --- stage 13 -----------------------------------------------------------------------------------------------------------

def _stage13():
    tree = ast.parse(STAGE13.read_text(encoding="utf-8"), str(STAGE13))
    imports = {alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "simfoundry.pipeline.dataset_root" for alias in node.names}
    return imports, _function(tree, "main")


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name]


def test_stage_13_imports_the_dataset_root_helpers_at_module_level():
    imports, _ = _stage13()
    assert {"check_dataset_root", "describe_dataset_root", "source_of_dataset_root"} <= imports


def test_the_dataset_check_runs_before_the_first_import_loop():
    _, main = _stage13()
    first_loop = min(n.lineno for n in ast.walk(main) if isinstance(n, ast.For))
    checks = _calls(main, "check_dataset_root") + _calls(main, "describe_dataset_root")
    assert checks and all(c.lineno < first_loop for c in checks)


def test_a_stop_reason_exits_before_any_import():
    _, main = _stage13()
    first_loop = min(n.lineno for n in ast.walk(main) if isinstance(n, ast.For))
    stops = [n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == "stop_reason"]
    assert len(stops) == 1 and stops[0].lineno < first_loop
    exits = [c for c in ast.walk(stops[0]) if isinstance(c, ast.Call) and ast.unparse(c.func) == "sys.exit"]
    assert len(exits) == 1 and ast.unparse(exits[0].args[0]) == "stop_reason"


def test_the_allow_external_switch_is_read_with_a_default_of_true():
    _, main = _stage13()
    reads = [c for c in ast.walk(main) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
             and c.func.attr == "get" and ast.unparse(c.func.value) == "cfg.s13_usd"
             and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "allow_external_dataset"]
    assert len(reads) == 1
    assert ast.unparse(reads[0].args[1]) == "True"


def test_stage_info_records_the_dataset_folder_next_to_the_imports():
    _, main = _stage13()
    dicts = [n for n in ast.walk(main) if isinstance(n, ast.Dict)]
    record = next(d for d in dicts if "asset_imports" in [k.value for k in d.keys if isinstance(k, ast.Constant)])
    keys = {k.value for k in record.keys if isinstance(k, ast.Constant)}
    assert {"asset_imports", "dataset_path", "dataset_root_source", "dataset_outside_checkout"} <= keys


# --- run.sh, the config and the README -------------------------------------------------------------------------------

def _run_sh(*args, cwd=REPO, env=None):
    base_env = {k: v for k, v in os.environ.items() if k not in (DATASET_ROOT_ENV, "OMNIGIBSON_DATA_PATH")}
    return subprocess.run([str(RUN_SH), *args], cwd=cwd, capture_output=True, text=True, env=env or base_env)


def test_run_sh_documents_the_dataset_root_option():
    result = _run_sh("--help")
    assert result.returncode == 0
    assert "--dataset-root DIR" in result.stdout


def test_run_sh_refuses_a_dataset_root_that_is_not_a_directory(tmp_path):
    result = _run_sh("--dataset-root", str(tmp_path / "missing"), "--dry-run")
    assert result.returncode == 2
    assert "--dataset-root: not a directory" in result.stderr


def _probe(tmp_path):
    probe = tmp_path / "probe_python"
    probe.write_text(
        "#!/usr/bin/env bash\necho \"SIMFOUNDRY_DATASET_ROOT=${SIMFOUNDRY_DATASET_ROOT:-}\"\n", encoding="utf-8")
    probe.chmod(0o755)
    return probe


def test_run_sh_exports_the_absolute_dataset_root_to_the_stages(tmp_path):
    (tmp_path / "sub").mkdir()
    base_env = {k: v for k, v in os.environ.items() if k not in (DATASET_ROOT_ENV, "OMNIGIBSON_DATA_PATH")}
    result = _run_sh("--python-bin", str(_probe(tmp_path)), "--exec-mode", "direct", "--include", "1b",
                     "--dataset-root", "sub", cwd=tmp_path, env=base_env)
    assert result.returncode == 0, result.stderr
    assert f"SIMFOUNDRY_DATASET_ROOT={os.path.realpath(tmp_path / 'sub')}" in result.stdout


def test_run_sh_exports_nothing_without_the_option(tmp_path):
    result = _run_sh("--python-bin", str(_probe(tmp_path)), "--exec-mode", "direct", "--include", "1b", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "SIMFOUNDRY_DATASET_ROOT=\n" in result.stdout


def test_the_config_default_is_true():
    assert yaml.safe_load(CFG.read_text(encoding="utf-8"))["s13_usd"]["allow_external_dataset"] is True


def test_the_readme_names_the_variable_the_option_and_the_switch():
    text = README.read_text(encoding="utf-8")
    for name in ("SIMFOUNDRY_DATASET_ROOT", "--dataset-root", "allow_external_dataset", "dataset_outside_checkout"):
        assert name in text


# --- the explicit root guard (stage 13 stops when it would write outside the explicit root) -----------------------------

def test_a_stage_writing_under_the_explicit_root_has_no_mismatch(tmp_path):
    root = tmp_path / "root"
    (root / "real2sim-assets").mkdir(parents=True)
    assert dr.explicit_root_mismatch(root / "real2sim-assets", root) is None
    assert dr.explicit_root_mismatch(root, root) is None


def test_a_stage_writing_elsewhere_names_both_paths_and_both_ways_out(tmp_path):
    root = tmp_path / "root"
    other = tmp_path / "default" / "real2sim-assets"
    root.mkdir()
    other.mkdir(parents=True)
    message = dr.explicit_root_mismatch(other, root)
    assert message is not None
    assert message.startswith("SIMFOUNDRY_DATASET_ROOT is set to ")
    assert os.path.realpath(root) in message
    assert os.path.realpath(other) in message
    assert "OMNIGIBSON_DATA_PATH" in message
    assert "run.sh" in message and "orchestrator" in message


def test_a_sibling_whose_name_starts_with_the_root_is_a_mismatch(tmp_path):
    root = tmp_path / "root"
    sibling = tmp_path / "root2" / "real2sim-assets"
    root.mkdir()
    sibling.mkdir(parents=True)
    assert dr.explicit_root_mismatch(sibling, root) is not None


def test_links_on_either_side_are_followed_before_the_containment_check(tmp_path):
    real_root = tmp_path / "real_root"
    (real_root / "real2sim-assets").mkdir(parents=True)
    root_link = tmp_path / "root_link"
    root_link.symlink_to(real_root, target_is_directory=True)
    assert dr.explicit_root_mismatch(root_link / "real2sim-assets", real_root) is None
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = real_root / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    message = dr.explicit_root_mismatch(escape, root_link)
    assert message is not None and os.path.realpath(outside) in message


def test_a_checkout_given_through_a_link_is_resolved_before_the_containment_check(tmp_path):
    real_repo = tmp_path / "real_repo"
    folder = real_repo / "deps/BEHAVIOR-1K/datasets/real2sim-assets"
    folder.mkdir(parents=True)
    link_repo = tmp_path / "link_repo"
    link_repo.symlink_to(real_repo, target_is_directory=True)
    description = dr.describe_dataset_root(folder, link_repo, "checkout")
    assert description["outside_checkout"] is False
    assert description["repo_realpath"] == os.path.realpath(real_repo)


def test_the_run_line_names_a_symlinked_explicit_root_by_its_target(fake_checkout, monkeypatch, tmp_path, capsys):
    target = tmp_path / "shared"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv(DATASET_ROOT_ENV, str(link))
    monkeypatch.setattr(orch, "get_stage_plan", lambda **_kwargs: [])
    orch.run_pipeline(
        cwd=".", input_mode="video", include_ids_csv=None, exclude_ids_csv=None, exec_mode="direct",
        python_bin="python", env_map={}, dry_run=True, stream_subseq_enabled=False, stream_start_stage=5,
        stream_end_stage=8, extra_overrides=["root_dir=/tmp/unused", "scene_name=s"],
    )
    assert f"[Pipeline] dataset root = {os.path.realpath(target)} (source: explicit)" in capsys.readouterr().out


def _same_expression(text, node):
    return ast.unparse(node) == ast.unparse(ast.parse(text, mode="eval").body)


def _keyword(call, name):
    return next(k.value for k in call.keywords if k.arg == name)


def test_stage_13_sends_the_real_explicit_switch_to_its_source_decision():
    _, main = _stage13()
    (decision,) = _calls(main, "source_of_dataset_root")
    assert _same_expression('dataset_root_source() == "explicit"', _keyword(decision, "explicit"))


def test_stage_13_records_the_outside_flag_itself_not_its_negation():
    _, main = _stage13()
    record = next(n for n in ast.walk(main) if isinstance(n, ast.Dict)
                  and "asset_imports" in [k.value for k in n.keys if isinstance(k, ast.Constant)])
    value = next(v for k, v in zip(record.keys, record.values) if isinstance(k, ast.Constant)
                 and k.value == "dataset_outside_checkout")
    assert ast.unparse(value) == "dataset_info['outside_checkout']"


def test_stage_13_passes_the_allow_external_switch_unchanged():
    _, main = _stage13()
    (check,) = _calls(main, "check_dataset_root")
    assert _same_expression('bool(cfg.s13_usd.get("allow_external_dataset", True))', _keyword(check, "allow_external"))


def test_the_explicit_root_guard_sits_between_describe_and_check_and_before_the_loop():
    _, main = _stage13()
    (guard_call,) = _calls(main, "explicit_root_mismatch")
    (describe,) = _calls(main, "describe_dataset_root")
    (check,) = _calls(main, "check_dataset_root")
    first_loop = min(n.lineno for n in ast.walk(main) if isinstance(n, ast.For))
    assert describe.lineno < guard_call.lineno < check.lineno < first_loop
    assert [ast.unparse(a) for a in guard_call.args] == ["dataset_info['dataset_path']", "get_omnigibson_data_path()"]
    guards = [n for n in ast.walk(main) if isinstance(n, ast.If)
              and _same_expression('dataset_root_source() == "explicit"', n.test)]
    assert len(guards) == 1
    inside = list(ast.walk(guards[0]))
    assert guard_call in inside
    exits = [c for c in inside if isinstance(c, ast.Call) and ast.unparse(c.func) == "sys.exit"]
    assert len(exits) == 1 and ast.unparse(exits[0].args[0]) == "mismatch"


def test_stage_13_imports_the_explicit_root_mismatch_at_module_level():
    imports, _ = _stage13()
    assert "explicit_root_mismatch" in imports


def test_run_sh_takes_the_dataset_root_as_one_argument_and_keeps_the_options_after_it(tmp_path):
    dataset_dir = tmp_path / "root"
    dataset_dir.mkdir()
    probe = tmp_path / "probe_args"
    probe.write_text(
        "#!/usr/bin/env bash\n"
        "for a in \"$@\"; do printf 'ARG:%s\\n' \"$a\"; done\n"
        "echo \"ROOT=${SIMFOUNDRY_DATASET_ROOT:-}\"\n",
        encoding="utf-8",
    )
    probe.chmod(0o755)
    result = _run_sh("--python-bin", str(probe), "--exec-mode", "direct", "--dataset-root", str(dataset_dir),
                     "--scene-name", "sceneX", "--include", "5", "--dry-run", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "ARG:scene_name=sceneX" in lines
    assert f"ROOT={os.path.realpath(dataset_dir)}" in lines
    assert f"ARG:{dataset_dir}" not in lines
    assert "ARG:--dataset-root" not in lines


def test_an_explicit_root_given_through_a_link_is_resolved_before_the_containment_check(tmp_path):
    real_root = tmp_path / "real_root"
    (real_root / "real2sim-assets").mkdir(parents=True)
    root_link = tmp_path / "root_link"
    root_link.symlink_to(real_root, target_is_directory=True)
    # The dataset folder is named through the root's link or by its real path: both are inside the root.
    assert dr.explicit_root_mismatch(root_link / "real2sim-assets", root_link) is None
    assert dr.explicit_root_mismatch(real_root / "real2sim-assets", root_link) is None


def test_a_dataset_folder_linking_out_of_an_explicit_root_given_through_a_link_is_a_mismatch(tmp_path):
    real_root = tmp_path / "real_root"
    real_root.mkdir()
    root_link = tmp_path / "root_link"
    root_link.symlink_to(real_root, target_is_directory=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (real_root / "real2sim-assets").symlink_to(outside, target_is_directory=True)
    message = dr.explicit_root_mismatch(root_link / "real2sim-assets", root_link)
    assert message is not None
    assert os.path.realpath(outside) in message
    assert os.path.realpath(real_root) in message


def test_the_readme_says_a_direct_stage_13_run_needs_the_same_root_in_omnigibson_data_path():
    text = README.read_text(encoding="utf-8")
    assert ("If you run stage 13 directly with the variable set, `OMNIGIBSON_DATA_PATH` must name the same root too; "
            "otherwise stage 13 stops before any import.") in text
