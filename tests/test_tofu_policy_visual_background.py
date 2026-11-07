# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--visual-background`` of the policy runner: off by default, and with it off the runner does what it did before.

The runner needs Isaac Sim, so ``main`` is checked on its syntax tree against the base commit. The two plain pieces
(``setup_args`` and the floor-plane expression) are run on their own. The structure tests name the statements that may
differ from the base, and their exact content and place; anything else that differs fails.
"""

import ast
import copy
import json
import subprocess
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts/cutting/tofu_policy_episodes.py"
# integration's runner before this option (the port's base): no DART block, and setup.json writes vars(args)
BASE_COMMIT = "6cb6ad5"
sys.path.insert(0, str(ROOT / "scripts/cutting"))
try:
    import tofu_policy_episodes as G  # noqa: E402 - the runner's own imports need the simulator's packages in some envs
    RUNNER_ERROR = None
except ImportError as error:  # only setup_args needs the runner; the structure tests read its source and always run
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

SPLAT = "simfoundry.utils.splat_background"
VISUAL_TEST = "visual_bg is not None"
KIT_TEST = "visual_bg is not None and visual_bg.kit_settings"
# read and checked once, right after the arguments; the splat module is not touched for the spelling "none"
VISUAL_INIT = "visual_bg = load_run_visual_background(args)"
# the named record line: setup.json names the background of every run, "none" included
RECORD_LINE = 'changes["visual_background"] = {"name": "none"}'
VISUAL_BODY = """\
from simfoundry.utils.splat_background import apply_splat_ccm, swap_visual_background
changes["visual_background"] = swap_visual_background(scene, visual_bg, label=f"scene {args.scene / T.SCENE_JSON}")
print("[eps] visual background:", visual_bg.name, visual_bg.usdz, "dz_m", visual_bg.dz_m,
      "remove", visual_bg.remove, flush=True)
"""
KIT_BODY = """\
applied = {}
for key, value in visual_bg.kit_settings.items():
    try:
        og.app.set_setting(key, value)
        applied[key] = {"value": value, "ok": True, "error": None}
    except Exception as error:
        applied[key] = {"value": value, "ok": False, "error": str(error)}
        print(f"[eps] visual background: set_setting {key!r} failed: {error}", flush=True)
changes["visual_background"]["kit_settings_applied"] = applied
"""
CCM_TEST = "visual_bg is not None and visual_bg.ccm is not None"
CCM_BODY = """\
gs = env.scene.object_registry("name", "gs_background")
changes["visual_background"]["ccm_applied"] = apply_splat_ccm(gs.prim, visual_bg.ccm)
for _ in range(3):
    og.sim.render()
"""


def _function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _expr(text):
    return ast.parse(text, mode="eval").body


def _stmts(text):
    return ast.parse(text).body


def _text(node):
    return ast.unparse(node)


def _parents(tree):
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _enclosing_ifs(node, parents):
    out, cur = [], node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, ast.If):
            out.append(cur)
    return out


def _is_test(node, text):
    return ast.dump(node) == ast.dump(_expr(text))


def _is_visual_if(stmt):
    return isinstance(stmt, ast.If) and _is_test(stmt.test, VISUAL_TEST)


def _is_kit_if(stmt):
    return isinstance(stmt, ast.If) and _is_test(stmt.test, KIT_TEST)


def _is_ccm_if(stmt):
    return isinstance(stmt, ast.If) and _is_test(stmt.test, CCM_TEST)


def _is_visual_init(stmt):
    return isinstance(stmt, ast.Assign) and ast.dump(stmt) == ast.dump(_stmts(VISUAL_INIT)[0])


def _is_visual_arg(stmt):
    call = stmt.value if isinstance(stmt, ast.Expr) else None
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument"
            and call.args and isinstance(call.args[0], ast.Constant)
            and call.args[0].value in ("--visual-background", "--visual-background-root"))


def _is_record_line(stmt):
    return isinstance(stmt, ast.Assign) and ast.dump(stmt) == ast.dump(_stmts(RECORD_LINE)[0])


def _ancestors(node, parents):
    out, cur = [], node
    while cur in parents:
        cur = parents[cur]
        out.append(cur)
    return out


def _env_call(main):
    return next(n for n in ast.walk(main) if isinstance(n, ast.Call) and _text(n.func) == "og.Environment")


def _env_scene(call):
    configs = next(k.value for k in call.keywords if k.arg == "configs")
    return next(k.value for k in configs.keywords if k.arg == "scene")


def _assign_index(body, target, call_attr=None):
    """Indices of the top-level assignments to ``target`` (optionally whose value calls ``.call_attr``)."""
    found = []
    for i, s in enumerate(body):
        if isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == target for t in s.targets):
            value = s.value
            if call_attr is None or (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                                     and value.func.attr == call_attr):
                found.append(i)
    return found


def _if_index(body, test_text):
    return [i for i, s in enumerate(body) if isinstance(s, ast.If) and _is_test(s.test, test_text)]


MAIN = _function(CURRENT, "main")
MAIN_BASE = _function(BASE, "main")


def _is_probe_arg(stmt):
    call = stmt.value if isinstance(stmt, ast.Expr) else None
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument"
            and call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value in ("--probe-out", "--probe-ticks"))


def _is_probe_if(stmt):
    return isinstance(stmt, ast.If) and _is_test(stmt.test, "args.probe_out is not None")


def _is_probe_assign(stmt):
    return isinstance(stmt, ast.Assign) and any(_text(t) in ("run.probe", "run.probe_seed") for t in stmt.targets)


def _drop_probe(stmts):
    """``stmts`` without the --probe-out statements, at any depth (the probe's seed is set inside the episode loop)."""
    out = []
    for s in stmts:
        if _is_probe_arg(s) or _is_probe_if(s) or _is_probe_assign(s):
            continue
        for field in ("body", "orelse", "finalbody"):
            if isinstance(getattr(s, field, None), list):
                setattr(s, field, _drop_probe(getattr(s, field)))
        out.append(s)
    return out


class _WriterAsBase(ast.NodeTransformer):
    """setup.json's ``setup_args(args)`` read as the base's ``vars(args)``: with the option off the two are the same dict."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if _text(node.func) == "setup_args" and [_text(a) for a in node.args] == ["args"]:
            return _expr("vars(args)")
        return node


def _without_additions(main):
    """``main`` with the statements the change may add taken out, and the floor-plane keyword read as ``True``.

    The additions are the visual ones, and the test-only --probe-out ones (its two options, its segmentation block, and
    the probe's attributes on ``run``).
    """
    tree = _WriterAsBase().visit(copy.deepcopy(main))
    tree.body = [s for s in tree.body
                 if not (_is_visual_arg(s) or _is_visual_init(s) or _is_visual_if(s) or _is_kit_if(s) or _is_ccm_if(s)
                         or _is_record_line(s))]
    tree.body = _drop_probe(tree.body)
    scene = _env_scene(_env_call(tree))
    for kw in scene.keywords:
        if kw.arg == "floor_plane_visible":
            kw.value = ast.Constant(value=True)
    return tree


def _args(**over):
    values = {"scene": Path("Data/x"), "visual_background": None, "mode": "expert", "seed": 3}
    values.update(over)
    return Namespace(**values)


def _eval(node, visual):
    code = compile(ast.Expression(body=node), "<scene dict>", "eval")
    return eval(code, {"__builtins__": {}}, {"visual_bg": visual})


# ---------------------------------------------------------------- behaviour: the plain pieces, run on their own

@pytest.mark.parametrize("visual", [None, Path("spec.json"), "none"])
def test_setup_args_keeps_keys_order_and_values(visual):
    args = _args(visual_background=visual)
    out = _runner().setup_args(args)
    assert list(out) == list(vars(args)), "the keys keep the order of the namespace"
    assert out["visual_background"] == ("none" if visual is None else visual)
    assert all(out[k] is vars(args)[k] for k in out if k != "visual_background")


def test_setup_args_records_the_spec_root_only_when_it_is_given():
    args = _args()
    assert "visual_background_root" not in _runner().setup_args(args)
    args.visual_background_root = Path("roots/here")
    assert _runner().setup_args(args)["visual_background_root"] is args.visual_background_root


def test_setup_args_with_option_none_matches_the_base_runner_except_the_spelling():
    # the base runner writes vars(args) and has no such option: a run with "none" writes the same dict but for the one
    # key that names the background
    base_args = _args()
    del base_args.visual_background
    assert _runner().setup_args(_args()) == dict(vars(base_args), visual_background="none")


@pytest.mark.parametrize(("visual", "expect"), [
    (None, True),
    (types.SimpleNamespace(floor_plane_visible=False), False),
    (types.SimpleNamespace(floor_plane_visible=True), True),
])
def test_floor_plane_visible_follows_the_spec(visual, expect):
    keywords = {kw.arg: kw.value for kw in _env_scene(_env_call(MAIN)).keywords}
    assert _eval(keywords["floor_plane_visible"], visual) is expect


@pytest.mark.parametrize("visual", [None, types.SimpleNamespace(floor_plane_visible=False)])
def test_other_scene_keys_stay_true_and_skybox_ignores_the_spec(visual):
    keywords = {kw.arg: kw.value for kw in _env_scene(_env_call(MAIN)).keywords}
    for key in ("use_floor_plane", "use_skybox", "include_robots"):
        assert _eval(keywords[key], visual) is True


# ---------------------------------------------------------------- behaviour: the early read, on stand-in files

PACKAGED = ROOT / "simfoundry/configs/visual_background/p1.json"
GUARD_INIT = {
    "mesh_background_0": {"args": {"expected_file_hash": "98ea90c59155fa834fd340d64ebc19b2"}},
    "iter_7": {"args": {"expected_file_hash": "db7a0f67450bd93abbc1c04f1a9b4306"}},
}


def _stand_in_root(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "PD_slab4_bg.usdz").write_bytes(b"a splat file, read for its md5 only")
    (root / "align_v4b.json").write_text(json.dumps({"M_da3_to_scene": np.eye(4).tolist()}))
    return root


def _scene_folder(folder, init_info):
    path = folder / _runner().T.SCENE_JSON  # the scene's JSON sits under its own sub-folder
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"objects_info": {"init_info": init_info}}))
    return folder


def _early(**over):
    values = {"visual_background": str(PACKAGED), "visual_background_root": None}
    values.update(over)
    return _runner().load_run_visual_background(_args(**values))


def test_none_reads_no_file_and_gives_no_background(tmp_path):
    assert _early(visual_background="none", visual_background_root=tmp_path / "missing", scene=tmp_path / "no_scene") is None


def test_the_packaged_spec_without_its_files_stops_and_names_both_ways_out(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as caught:
        _early(visual_background_root=empty, scene=tmp_path)
    text = str(caught.value.code)
    for part in ("PD_slab4_bg.usdz", "align_v4b.json", str(empty), "--visual-background-root",
                 "SIMFOUNDRY_VISUAL_BACKGROUND_ROOT", "--visual-background none", "docs/BACKGROUND_SWAP.md"):
        assert part in text


def test_a_spec_of_your_own_without_its_files_names_its_folder_and_not_the_variable(tmp_path):
    folder = tmp_path / "mine"
    folder.mkdir()
    spec = folder / "my_spec.json"
    spec.write_text(json.dumps({"schema": 1, "name": "mine", "usdz": "a.usdz", "pose_json": "b.json"}))
    with pytest.raises(SystemExit) as caught:
        _early(visual_background=str(spec), scene=tmp_path)
    text = str(caught.value.code)
    assert "a.usdz" in text and str(folder) in text and "--visual-background none" in text
    assert "SIMFOUNDRY_VISUAL_BACKGROUND_ROOT" not in text


def test_the_environment_variable_is_the_packaged_root_when_no_option_is_given(tmp_path, monkeypatch):
    root = _stand_in_root(tmp_path / "from_env")
    monkeypatch.setenv("SIMFOUNDRY_VISUAL_BACKGROUND_ROOT", str(root))
    spec = _early(scene=_scene_folder(tmp_path / "scene", GUARD_INIT))
    assert spec.usdz == root / "PD_slab4_bg.usdz"


def test_the_option_wins_over_the_environment_variable_for_the_packaged_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMFOUNDRY_VISUAL_BACKGROUND_ROOT", str(tmp_path / "unused"))
    root = _stand_in_root(tmp_path / "option")
    spec = _early(visual_background_root=root, scene=_scene_folder(tmp_path / "scene", GUARD_INIT))
    assert spec.usdz == root / "PD_slab4_bg.usdz"


def test_a_matching_scene_gives_the_packaged_spec(tmp_path):
    root = _stand_in_root(tmp_path / "root")
    spec = _early(visual_background_root=root, scene=_scene_folder(tmp_path / "scene", GUARD_INIT))
    assert spec.name == "P1" and spec.scene_requires is not None


def test_a_scene_that_is_not_the_specs_stops_the_run_before_isaac_sim(tmp_path):
    root = _stand_in_root(tmp_path / "root")
    init = dict(GUARD_INIT, iter_7={"args": {"expected_file_hash": "0" * 32}})
    scene = _scene_folder(tmp_path / "scene", init)
    with pytest.raises(SystemExit) as caught:
        _early(visual_background_root=root, scene=scene)
    text = str(caught.value.code)
    for part in ("iter_7", "0" * 32, "db7a0f67450bd93abbc1c04f1a9b4306", str(scene), "--visual-background none"):
        assert part in text


def test_a_scene_that_records_no_hash_stops_the_run_as_cannot_verify(tmp_path):
    root = _stand_in_root(tmp_path / "root")
    scene = _scene_folder(tmp_path / "scene", {"mesh_background_0": GUARD_INIT["mesh_background_0"], "iter_7": {"args": {}}})
    with pytest.raises(SystemExit) as caught:
        _early(visual_background_root=root, scene=scene)
    assert "cannot verify" in str(caught.value.code)


def test_the_repo_reader_of_setup_json_ignores_the_background_keys():
    # demonstrations.base_pose reads scene_changes by key; a record with the background keys must still load the same
    from simfoundry.cutting.demonstrations import base_pose
    plain = {"scene_changes": {"robot_yaw_deg": 0.0, "robot_pos": [0.1, 0.2, 0.0]}}
    named = {"args": {"visual_background": "none"},
             "scene_changes": {"robot_yaw_deg": 0.0, "robot_pos": [0.1, 0.2, 0.0], "visual_background": {"name": "none"}}}
    assert np.array_equal(base_pose(named), base_pose(plain))


def test_a_scene_file_that_is_missing_stops_the_run(tmp_path):
    root = _stand_in_root(tmp_path / "root")
    with pytest.raises(SystemExit) as caught:
        _early(visual_background_root=root, scene=tmp_path / "no_scene_here")
    assert "cannot read scene" in str(caught.value.code)


# ---------------------------------------------------------------- structure: the syntax tree of main

def _add_argument_keywords(flag):
    calls = [n for n in ast.walk(MAIN) if isinstance(n, ast.Call) and n.args
             and isinstance(n.args[0], ast.Constant) and n.args[0].value == flag]
    assert len(calls) == 1, f"{flag} is registered once"
    return {k.arg: k.value for k in calls[0].keywords}


def test_visual_background_argument_defaults_to_the_packaged_spec_and_takes_a_spelling():
    kw = _add_argument_keywords("--visual-background")
    assert _text(kw["default"]) == "str(packaged_spec_path())"
    assert "type" not in kw, "the spelling is a string: a path or 'none'"
    assert isinstance(kw["metavar"], ast.Constant) and kw["metavar"].value == "SPEC|none"
    assert isinstance(kw.get("help"), ast.Constant) and "'none'" in kw["help"].value


def test_visual_background_root_is_an_optional_path_defaulting_to_none():
    kw = _add_argument_keywords("--visual-background-root")
    assert isinstance(kw["default"], ast.Constant) and kw["default"].value is None
    assert _text(kw["type"]) == "Path"
    assert not set(kw) - {"default", "type", "help"}, "the option takes no keyword beyond these"


def test_existing_arguments_are_the_base_ones():
    def adds(tree):
        return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"]
    assert [d for d in adds(MAIN) if "--visual-background" not in d and "--probe-" not in d] == adds(MAIN_BASE)


def test_visual_block_content_and_place():
    body = MAIN.body
    branch = [i for i, s in enumerate(body) if _is_visual_if(s)]
    record = [i for i, s in enumerate(body) if _is_record_line(s)]
    assert len(branch) == 1 and len(record) == 1
    stmt = body[branch[0]]
    assert not stmt.orelse
    assert [ast.dump(s) for s in stmt.body] == [ast.dump(s) for s in _stmts(VISUAL_BODY)]
    edit, board, sensors = (_assign_index(body, "changes", "edit_scene"), _if_index(body, "board_pose is not None"),
                            _assign_index(body, "sensors"))
    assert len(edit) == 1 and len(board) == 1 and len(sensors) == 1
    assert edit[0] < board[0] < record[0] < branch[0] < sensors[0]


def test_the_background_is_read_after_the_arguments_and_before_isaac_sim_starts():
    body = MAIN.body
    early = [i for i, s in enumerate(body) if _is_visual_init(s)]
    parse = _assign_index(body, "args", "parse_args")
    launch = [i for i, s in enumerate(body)
              if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and _text(s.value.func) == "og.launch"]
    assert len(early) == 1 and len(parse) == 1 and len(launch) == 1
    assert parse[0] < early[0] < launch[0]


def test_none_returns_before_the_splat_module_is_read():
    loader = _function(CURRENT, "load_run_visual_background")
    statements = [s for s in loader.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]  # no docstring
    first = statements[0]
    assert isinstance(first, ast.If) and _is_test(first.test, "args.visual_background == 'none'")
    assert len(first.body) == 1 and isinstance(first.body[0], ast.Return) and _text(first.body[0].value) == "None"
    imports = [i for i, s in enumerate(loader.body) if isinstance(s, ast.ImportFrom)]
    assert imports and min(imports) > 0, "the module is imported after the none return"


def test_kit_settings_block_content_and_place():
    body = MAIN.body
    kit = [i for i, s in enumerate(body) if _is_kit_if(s)]
    assert len(kit) == 1
    stmt = body[kit[0]]
    assert not stmt.orelse
    assert [ast.dump(s) for s in stmt.body] == [ast.dump(s) for s in _stmts(KIT_BODY)]
    sensors, hz, env = _assign_index(body, "sensors"), _assign_index(body, "hz"), _assign_index(body, "env", "Environment")
    assert len(sensors) == 1 and len(hz) == 1 and len(env) == 1
    assert sensors[0] < kit[0] < hz[0] < env[0]


def test_kit_settings_are_set_only_inside_the_kit_branch_and_before_the_environment():
    parents = _parents(CURRENT)
    body = MAIN.body
    kit_stmt = body[next(i for i, s in enumerate(body) if _is_kit_if(s))]
    calls = [n for n in ast.walk(MAIN) if isinstance(n, ast.Call) and _text(n.func) == "og.app.set_setting"]
    assert len(calls) == 1
    assert any(s is kit_stmt for s in _enclosing_ifs(calls[0], parents))
    assert calls[0].lineno < _env_call(MAIN).lineno


def test_the_splat_module_is_used_only_by_the_loader_or_under_the_option():
    parents = _parents(CURRENT)
    loader = _function(CURRENT, "load_run_visual_background")
    names = ("load_visual_background", "swap_visual_background", "apply_splat_ccm", "check_scene_requires")
    uses = [n for n in ast.walk(CURRENT) if isinstance(n, ast.Name) and n.id in names]
    assert uses
    for node in uses:
        in_loader = any(p is loader for p in _ancestors(node, parents))
        under_option = any(_is_visual_if(s) or _is_kit_if(s) or _is_ccm_if(s) for s in _enclosing_ifs(node, parents))
        assert in_loader or under_option, f"line {node.lineno} is outside the option"


def test_sensor_modalities_are_unchanged():
    def modalities(tree):
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == "modalities":
                        out.append(ast.literal_eval(value))
        return out
    assert modalities(MAIN) == modalities(MAIN_BASE)
    assert modalities(MAIN) and all(m == ["rgb"] for m in modalities(MAIN))


def test_visual_bg_is_defined_on_every_path_before_its_use():
    body = MAIN.body
    init = [i for i, s in enumerate(body) if _is_visual_init(s)]
    assert len(init) == 1 and init[0] < next(i for i, s in enumerate(body) if _is_visual_if(s))
    assert init[0] < next(i for i, s in enumerate(body) if _is_kit_if(s))


def test_main_matches_the_base_outside_the_allowed_additions():
    assert ast.dump(_without_additions(MAIN)) == ast.dump(MAIN_BASE), "main differs from the base beyond the allowed additions"


def test_changes_are_written_only_inside_the_visual_branches():
    def writes(main):
        parents = _parents(main)
        out = []
        for node in ast.walk(main):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    root = target
                    while isinstance(root, ast.Subscript):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id == "changes":
                        inside = any(_is_visual_if(s) or _is_kit_if(s) or _is_ccm_if(s) for s in _enclosing_ifs(node, parents))
                        out.append((_text(target), inside))
        return out
    now, base = writes(MAIN), writes(MAIN_BASE)
    record = _text(_expr('changes["visual_background"]'))
    assert [t for t, inside in now if not inside and t != record] == [t for t, _ in base]
    assert sorted(t for t, inside in now if inside) == sorted([
        _text(_expr('changes["visual_background"]')),
        _text(_expr('changes["visual_background"]["kit_settings_applied"]')),
        _text(_expr('changes["visual_background"]["ccm_applied"]')),
    ])


# ---------------------------------------------------------------- structure: the colour-matrix hook (off unless the spec has one)


def _ccm_index(body):
    found = [i for i, s in enumerate(body) if _is_ccm_if(s)]
    assert len(found) == 1, "the colour-matrix hook must be one top-level if of main, with exactly the condition in CCM_TEST"
    return found[0]


def test_ccm_hook_content_and_place():
    body = MAIN.body
    i = _ccm_index(body)
    assert not body[i].orelse
    assert [ast.dump(s) for s in body[i].body] == [ast.dump(s) for s in _stmts(CCM_BODY)]
    reset = [k for k, s in enumerate(body) if isinstance(s, ast.Expr) and _text(s.value) == "env.reset()"]
    # the runner has two physx blocks: one before the environment is built, and the one after env.reset() that counts here
    physx = [k for k in _if_index(body, "args.physx_enhanced_determinism") if k > reset[0]]
    cams = _assign_index(body, "cams")
    assert len(reset) == 1 and len(physx) == 1 and len(cams) == 1
    assert reset[0] < physx[0] < i < cams[0], "the hook comes after env.reset() and the physx block, and before cams"


def test_ccm_is_applied_only_inside_its_branch_and_then_renders_three_times():
    parents = _parents(CURRENT)
    branch = MAIN.body[_ccm_index(MAIN.body)]
    calls = [n for n in ast.walk(CURRENT) if isinstance(n, ast.Call) and _text(n.func) == "apply_splat_ccm"]
    assert len(calls) == 1
    assert any(s is branch for s in _enclosing_ifs(calls[0], parents))
    loops = [s for s in branch.body if isinstance(s, ast.For)]
    assert len(loops) == 1 and _text(loops[0].iter) == "range(3)"
    assert [_text(s.value.func) for s in loops[0].body if isinstance(s, ast.Expr)] == ["og.sim.render"]
