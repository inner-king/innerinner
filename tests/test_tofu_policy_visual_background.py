# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--visual-background`` of the policy runner: off by default, and with it off the runner does what it did before.

The runner needs Isaac Sim, so ``main`` is checked on its syntax tree against the base commit. The two plain pieces
(``setup_args`` and the floor-plane expression) are run on their own. The structure tests name the statements that may
differ from the base, and their exact content and place; anything else that differs fails.
"""

import ast
import copy
import subprocess
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts/cutting/tofu_policy_episodes.py"
BASE_COMMIT = "5b05fe3"
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
VISUAL_TEST = "args.visual_background is not None"
KIT_TEST = "visual_bg is not None and visual_bg.kit_settings"
VISUAL_INIT = "visual_bg = None"
VISUAL_BODY = """\
from simfoundry.utils.splat_background import load_visual_background, swap_visual_background, apply_splat_ccm
visual_bg = load_visual_background(args.visual_background)
changes["visual_background"] = swap_visual_background(scene, visual_bg)
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
            and call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value == "--visual-background")


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


def _without_additions(main):
    """``main`` with the statements the change may add taken out, and the floor-plane keyword read as ``True``.

    The additions are the visual ones, and the test-only --probe-out ones (its two options, its segmentation block, and
    the probe's attributes on ``run``).
    """
    tree = copy.deepcopy(main)
    tree.body = [s for s in tree.body
                 if not (_is_visual_arg(s) or _is_visual_init(s) or _is_visual_if(s) or _is_kit_if(s) or _is_ccm_if(s))]
    tree.body = _drop_probe(tree.body)
    scene = _env_scene(_env_call(tree))
    for kw in scene.keywords:
        if kw.arg == "floor_plane_visible":
            kw.value = ast.Constant(value=True)
    return tree


def _base_function(name):
    node = _function(BASE, name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<base>", "exec"), namespace)  # noqa: S102 - compiles the base's own source
    return namespace[name]


def _args(**over):
    values = {"scene": Path("Data/x"), "visual_background": None, "mode": "expert", "seed": 3,
              "dart_sigma_mm": 0.0, "dart_tau_s": 0.1, "dart_clip_mm": 2.0}
    values.update(over)
    return Namespace(**values)


def _eval(node, visual):
    code = compile(ast.Expression(body=node), "<scene dict>", "eval")
    return eval(code, {"__builtins__": {}}, {"visual_bg": visual})


# ---------------------------------------------------------------- behaviour: the plain pieces, run on their own

@pytest.mark.parametrize("dart_sigma_mm", [0.0, 2.5])
@pytest.mark.parametrize("visual", [None, Path("spec.json")])
def test_setup_args_keeps_keys_order_and_values(dart_sigma_mm, visual):
    args = _args(dart_sigma_mm=dart_sigma_mm, visual_background=visual)
    out = _runner().setup_args(args)
    expected = [k for k in vars(args)
                if (dart_sigma_mm > 0 or not k.startswith("dart_")) and (k != "visual_background" or visual is not None)]
    assert list(out) == expected
    assert all(out[k] is vars(args)[k] for k in out)


@pytest.mark.parametrize("dart_sigma_mm", [0.0, 2.5])
def test_setup_args_with_option_off_matches_the_base_runner(dart_sigma_mm):
    base_args = _args(dart_sigma_mm=dart_sigma_mm)
    del base_args.visual_background  # the base runner has no such option
    assert _runner().setup_args(_args(dart_sigma_mm=dart_sigma_mm)) == _base_function("setup_args")(base_args)


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


# ---------------------------------------------------------------- structure: the syntax tree of main

def test_visual_background_argument_is_optional_path_defaulting_to_none():
    calls = [n for n in ast.walk(MAIN) if isinstance(n, ast.Call) and n.args
             and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--visual-background"]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kw["default"], ast.Constant) and kw["default"].value is None
    assert _text(kw["type"]) == "Path"
    assert isinstance(kw.get("help"), ast.Constant) and isinstance(kw["help"].value, str)
    assert not set(kw) - {"default", "type", "help", "required"}, "the option takes no keyword beyond these"
    assert "required" not in kw or _text(kw["required"]) == "False", "the option must stay optional"


def test_existing_arguments_are_the_base_ones():
    def adds(tree):
        return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"]
    assert [d for d in adds(MAIN) if "--visual-background" not in d and "--probe-" not in d] == adds(MAIN_BASE)


def test_visual_block_content_and_place():
    body = MAIN.body
    init, branch = [i for i, s in enumerate(body) if _is_visual_init(s)], [i for i, s in enumerate(body) if _is_visual_if(s)]
    assert len(init) == 1 and len(branch) == 1
    stmt = body[branch[0]]
    assert not stmt.orelse
    assert [ast.dump(s) for s in stmt.body] == [ast.dump(s) for s in _stmts(VISUAL_BODY)]
    edit, board, sensors = (_assign_index(body, "changes", "edit_scene"), _if_index(body, "board_pose is not None"),
                            _assign_index(body, "sensors"))
    assert len(edit) == 1 and len(board) == 1 and len(sensors) == 1
    assert edit[0] < board[0] < init[0] < branch[0] < sensors[0]


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


def test_splat_module_is_imported_and_called_only_under_the_option():
    parents = _parents(CURRENT)
    imports = [n for n in ast.walk(CURRENT) if isinstance(n, ast.ImportFrom) and n.module == SPLAT]
    uses = [n for n in ast.walk(CURRENT) if isinstance(n, ast.Name)
            and n.id in ("load_visual_background", "swap_visual_background")]
    assert imports and uses
    for node in imports + uses:
        assert any(_is_visual_if(s) for s in _enclosing_ifs(node, parents)), f"line {node.lineno} is outside the option"


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
    assert [t for t, inside in now if not inside] == [t for t, _ in base]
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
