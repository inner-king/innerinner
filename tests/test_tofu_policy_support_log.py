# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--action-mode support-log`` of the evaluation client: the hand target it commands, and the wiring of ``main``.

The generator runs only with Isaac Sim, so ``main`` is checked on its syntax tree, and the target function and the two
unchanged modes are run on their own. The unchanged modes are compared with the base commit the branch was cut from.
"""

import ast
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from simfoundry.cutting.support_height import (  # noqa: E402 - imported before the generator, so a missing module fails here
    EPS_M,
    hand_z_from_log_height,
    log_height,
    lowest_edge_z,
    state_with_support,
    support_drop_near_constant,
    support_state_mode,
)

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts/cutting/tofu_policy_episodes.py"
BASE_COMMIT = "e05ceb3"
sys.path.insert(0, str(ROOT / "scripts/cutting"))
try:
    import tofu_policy_episodes as G  # noqa: E402 - the generator's own imports need the simulator's packages in some envs
except ImportError as error:
    pytest.skip(f"the generator cannot be imported here: {error}", allow_module_level=True)

CURRENT = ast.parse(GEN.read_text())


def _main(tree):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def _text(node):
    return ast.unparse(node)


def _block_of(tree, stmt):
    """The statement list that holds ``stmt`` directly."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and any(item is stmt for item in block):
                return block
    raise AssertionError(f"no statement list holds {_text(stmt)!r}")


def _holds(stmt, node):
    return any(child is node for child in ast.walk(stmt))


def _dict_value(node, key):
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    return None


def _level_base(rng, z):
    """A level base (a yaw, and a position at height ``z``)."""
    B = np.eye(4)
    B[:3, :3] = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)).as_matrix()
    B[:3, 3] = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), z]
    return B


def _random_rotation(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _case(rng, base, eps=EPS_M):
    """A hand pose E, a knife K, a support height s, and the policy's 9-vector that asks for E with its z slot as u."""
    E = np.eye(4)
    E[:3, :3] = _random_rotation(rng)
    E[:3, 3] = rng.uniform(-0.5, 0.5, 3)
    K = np.eye(4)
    K[:3, :3] = _random_rotation(rng)
    K[:3, 3] = rng.uniform(-0.2, 0.2, 3)
    h = rng.uniform(-eps + 1e-4, 0.2)
    s = lowest_edge_z(E, K) - h
    local = np.linalg.inv(base) @ E
    a9 = np.concatenate([local[:3, 3], G.rot6(local[:3, :3])])
    a9[2] = log_height(h, eps)
    return E, K, s, a9


@pytest.mark.parametrize("z", [0.0, 0.3])
def test_the_target_is_the_hand_pose_the_policy_was_asked_for(z):
    rng = np.random.default_rng(1)
    for _ in range(50):
        base = _level_base(rng, z)
        E, K, s, a9 = _case(rng, base)
        out = G.support_log_target(base, a9, K, s)
        assert np.abs(out[:3, 3] - E[:3, 3]).max() < 1e-9
        assert np.abs(out[:3, :3] - E[:3, :3]).max() < 1e-9


def test_the_policy_sends_float32_and_the_height_still_comes_back_within_a_micrometre():
    rng = np.random.default_rng(2)
    base = _level_base(rng, 0.3)
    E, K, s, a9 = _case(rng, base)
    a32 = a9.astype(np.float32).astype(float)
    out = G.support_log_target(base, a32, K, s)
    ref = base @ G.pose9(np.concatenate([a32[:2], [0.0], a32[3:]]))
    ref[2, 3] = hand_z_from_log_height(ref, K, s, a32[2])
    assert abs(out[2, 3] - ref[2, 3]) < 1e-9
    assert abs(out[2, 3] - E[2, 3]) < 1e-6


def test_eps_is_the_one_passed_in_and_the_default_would_miss():
    rng = np.random.default_rng(3)
    base = _level_base(rng, 0.0)
    E, K, s, a9 = _case(rng, base, eps=2e-3)
    assert abs(G.support_log_target(base, a9, K, s, eps=2e-3)[2, 3] - E[2, 3]) < 1e-9
    assert abs(G.support_log_target(base, a9, K, s)[2, 3] - E[2, 3]) > 1e-4


def test_a_log_height_above_one_metre_is_capped_so_the_target_stays_finite():
    rng = np.random.default_rng(5)
    base = _level_base(rng, 0.3)
    _, K, s, a9 = _case(rng, base)
    a_high, a_cap, a_low = a9.copy(), a9.copy(), a9.copy()
    a_high[2] = 800.0  # exp(800) overflows to inf
    a_cap[2] = math.log(1.0 + EPS_M)  # h = 1 m exactly
    a_low[2] = -800.0  # h -> -eps, finite without a cap
    out = G.support_log_target(base, a_high, K, s)
    assert np.isfinite(out).all()
    assert np.array_equal(out, G.support_log_target(base, a_cap, K, s))
    assert np.isfinite(G.support_log_target(base, a_low, K, s)).all()


def test_a_log_height_just_below_the_cap_is_not_cut():
    rng = np.random.default_rng(6)
    base = _level_base(rng, 0.0)
    _, K, s, a9 = _case(rng, base)
    a = a9.copy()
    a[2] = math.log(1.0 + EPS_M) - 1e-3
    out = G.support_log_target(base, a, K, s)
    local = np.array(a, float)
    local[2] = 0.0
    ref = base @ G.pose9(local)
    ref[2, 3] = hand_z_from_log_height(ref, K, s, a[2], EPS_M)
    assert np.array_equal(out, ref)


def test_a_tilted_base_is_refused_because_the_height_is_solved_in_world_z():
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("x", 5, degrees=True).as_matrix()
    with pytest.raises(ValueError):
        G.support_log_target(base, np.zeros(9), np.eye(4), 0.0)


def test_the_policy_server_reads_the_last_ten_entries_so_the_support_goes_last():
    out = state_with_support(np.arange(16.0), 0.011)
    assert out.dtype == np.float32
    # fingers (7, 8), tip position (9-11), quaternion (12-15), then the support: the ten entries a 10-entry policy reads
    np.testing.assert_array_equal(out[-10:], np.array([*range(7, 16), 0.011], dtype=np.float32))


def test_action_mode_offers_three_modes_and_absolute_stays_the_default():
    calls = [n for n in ast.walk(_main(CURRENT))
             if isinstance(n, ast.Call) and _text(n.func) == "ap.add_argument" and n.args
             and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--action-mode"]
    assert len(calls) == 1
    keywords = {k.arg: k.value for k in calls[0].keywords}
    assert ast.literal_eval(keywords["choices"]) == ["absolute", "delta", "support-log"]
    assert ast.literal_eval(keywords["default"]) == "absolute"


def test_support_log_needs_policy_mode_and_is_refused_right_after_parsing():
    main = _main(CURRENT)
    checks = [n for n in main.body
              if isinstance(n, ast.If) and _text(n.test) == "args.action_mode == 'support-log' and args.mode != 'policy'"]
    assert len(checks) == 1
    parse = [i for i, n in enumerate(main.body) if _text(n) == "args = ap.parse_args()"]
    assert len(parse) == 1 and main.body.index(checks[0]) == parse[0] + 1
    assert len(checks[0].body) == 1 and _text(checks[0].body[0]).startswith("ap.error(")


def test_support_s_is_the_solver_board_plane_set_after_the_inverse_and_before_the_mode_split():
    main = _main(CURRENT)
    inverse = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and _text(n) == "S = np.linalg.inv(W)"]
    support = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and _text(n.targets[0]) == "support_s"]
    assert len(inverse) == 1 and len(support) == 1
    assert _text(support[0].value) == "float(W[2, 3] + sim.board_z) if args.action_mode == 'support-log' else None"
    block = _block_of(main, inverse[0])
    assert block is _block_of(main, support[0])
    split = next(n for n in block if isinstance(n, ast.If) and _text(n.test) == "args.mode == 'expert'")
    assert block.index(inverse[0]) < block.index(support[0]) < block.index(split)


def test_the_act_message_carries_the_policy_state_built_for_the_dataset_layout():
    main = _main(CURRENT)
    want = "policy_state(tick['state'], state_dim, support_s if state_mode == 'append' else None, drop_near_constant)"
    assigns = [n for n in ast.walk(main)
               if isinstance(n, ast.Assign) and _text(n.targets[0]) == "state" and _text(n.value) == want]
    assert len(assigns) == 1
    kept = [n for n in ast.walk(main)
            if isinstance(n, ast.Assign) and _text(n.targets[0]) == "state" and _text(n.value) == "tick['state']"]
    assert len(kept) == 1
    sends = [n for n in ast.walk(main) if isinstance(n, ast.Call) and _text(n.func) == "conn.send"
             and isinstance(_dict_value(n.args[0], "cmd"), ast.Constant) and _dict_value(n.args[0], "cmd").value == "act"]
    assert len(sends) == 1
    assert _text(_dict_value(sends[0].args[0], "state")) == "pack(state)"
    (gate,) = [n for n in ast.walk(main) if isinstance(n, ast.If) and _text(n.test) == "support_s is None"]
    assert any(child is assigns[0] for stmt in gate.orelse for child in ast.walk(stmt))
    block = _block_of(main, gate)
    holder = next(stmt for stmt in block if _holds(stmt, sends[0]))
    assert block.index(gate) < block.index(holder)


def test_the_mode_split_puts_support_log_between_delta_and_absolute():
    main = _main(CURRENT)
    (chain,) = [n for n in ast.walk(main) if isinstance(n, ast.If) and _text(n.test) == "args.action_mode == 'delta'"]
    opened_by = [n for n in ast.walk(main) if isinstance(n, ast.If) and any(child is chain for child in n.orelse)]
    assert opened_by == [], "delta must open the chain, not sit in an elif of another branch"
    assert len(chain.orelse) == 1 and isinstance(chain.orelse[0], ast.If)
    branch = chain.orelse[0]
    assert _text(branch.test) == "args.action_mode == 'support-log'"
    assert [_text(s) for s in branch.body] == ["E_tgt = support_log_target(arm.base, a9, K_in_E, support_s)"]
    assert [_text(s) for s in branch.orelse] == ["E_tgt = arm.base @ pose9(a9)"]


def test_the_episode_info_records_the_support_only_when_it_is_in_use():
    main = _main(CURRENT)
    gates = [n for n in ast.walk(main) if isinstance(n, ast.If) and _text(n.test) == "support_s is not None"]
    assert len(gates) == 1
    info = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and _text(n.targets[0]) == "info"]
    assert len(info) == 1
    block = _block_of(main, info[0])
    assert block is _block_of(main, gates[0]) and block.index(info[0]) < block.index(gates[0])
    (assign,) = gates[0].body
    assert _text(assign.targets[0]) == "info['support_log']"
    value = assign.value
    assert {ast.literal_eval(k): _text(v) for k, v in zip(value.keys, value.values)} == {
        "eps_m": "EPS_M", "support_z_m": "support_s"}


def _base_tree():
    proc = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:scripts/cutting/tofu_policy_episodes.py"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"the base commit {BASE_COMMIT} is not in this checkout")
    return ast.parse(proc.stdout)


def _mode_split(tree):
    (node,) = [n for n in ast.walk(_main(tree)) if isinstance(n, ast.If) and _text(n.test) == "args.action_mode == 'delta'"]
    return node


def _run_split(node, mode, a9, K, s, arm):
    env = {"np": np, "pose9": G.pose9, "support_log_target": G.support_log_target,
           "args": types.SimpleNamespace(action_mode=mode), "arm": arm, "a9": a9, "K_in_E": K, "support_s": s}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<mode split>", "exec"), env)
    return env["E_tgt"]


def test_absolute_and_delta_targets_are_identical_to_the_base_commit_and_support_log_is_the_new_target():
    base_split, current_split = _mode_split(_base_tree()), _mode_split(CURRENT)
    rng = np.random.default_rng(4)
    for _ in range(50):
        B = _level_base(rng, rng.choice([0.0, 0.3]))
        hand = _case(rng, B)[0]
        arm = types.SimpleNamespace(base=B, base_inv=np.linalg.inv(B), eef=lambda hand=hand: hand)
        a9 = rng.normal(size=9)
        K = _case(rng, B)[1]
        s = float(rng.uniform(-0.1, 0.1))
        for mode in ("absolute", "delta"):
            assert np.array_equal(_run_split(current_split, mode, a9, K, s, arm),
                                  _run_split(base_split, mode, a9, K, s, arm))
        assert np.array_equal(_run_split(current_split, "support-log", a9, K, s, arm),
                              G.support_log_target(B, a9, K, s))


def test_the_state_sent_is_the_tick_state_outside_support_log_and_the_policy_state_under_it():
    main = _main(CURRENT)
    (gate,) = [n for n in ast.walk(main) if isinstance(n, ast.If) and _text(n.test) == "support_s is None"]
    module = ast.Module(body=[gate], type_ignores=[])
    arr = np.arange(16, dtype=np.float32)
    env = {"tick": {"state": arr}, "support_s": None, "state_mode": "append", "state_dim": 10,
           "drop_near_constant": False, "policy_state": G.policy_state}
    exec(compile(module, "<state>", "exec"), env)
    assert env["state"] is arr
    env.update(support_s=0.011, state_mode="none", state_dim=9)
    exec(compile(module, "<state>", "exec"), env)
    np.testing.assert_array_equal(env["state"], arr[7:16])
    env.update(state_mode="append", state_dim=10)
    exec(compile(module, "<state>", "exec"), env)
    assert env["state"].shape == (10,) and env["state"][-1] == np.float32(0.011)
    np.testing.assert_array_equal(env["state"][:9], arr[7:16])


def test_the_hand_alone_is_the_policy_state_of_the_default_layout():
    arr = np.arange(16.0)
    out = G.policy_state(arr, 9)
    assert out.dtype == np.float32 and out.shape == (9,)
    np.testing.assert_array_equal(out, np.array(range(7, 16), dtype=np.float32))


def test_the_policy_state_appends_the_support_plane_under_append_and_is_checked_against_state_dim():
    arr = np.arange(16.0)
    out = G.policy_state(arr, 10, 0.011)
    assert out.dtype == np.float32 and out.shape == (10,) and out[-1] == np.float32(0.011)
    np.testing.assert_array_equal(out[:9], np.array(range(7, 16), dtype=np.float32))
    with pytest.raises(SystemExit, match="10 entries"):
        G.policy_state(arr, 10)  # the policy wants s, the client has none
    with pytest.raises(SystemExit, match="9 entries"):
        G.policy_state(arr, 9, 0.011)  # the policy wants the hand alone, the client would add s


def test_the_server_reads_the_same_values_the_whole_state_it_is_sent_gives_for_each_layout():
    arr = np.arange(16.0)
    s = 0.011
    # policy_server.py: the last n entries of what it is sent, n being the checkpoint's state size
    before_none = np.asarray(arr, np.float32)  # the tick state, sent whole until this change
    np.testing.assert_array_equal(before_none[len(before_none) - 9:], G.policy_state(arr, 9))
    before_append = state_with_support(arr, s)
    np.testing.assert_array_equal(before_append[len(before_append) - 10:], G.policy_state(arr, 10, s))
    for n, st in ((9, G.policy_state(arr, 9)), (10, G.policy_state(arr, 10, s))):
        np.testing.assert_array_equal(st[len(st) - n:], st)  # what is sent is exactly what the server reads


def test_the_client_has_the_support_meta_option_and_refuses_it_outside_support_log():
    main = _main(CURRENT)
    calls = [n for n in ast.walk(main)
             if isinstance(n, ast.Call) and _text(n.func) == "ap.add_argument" and n.args
             and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--support-meta"]
    assert len(calls) == 1
    keywords = {k.arg: k.value for k in calls[0].keywords}
    assert ast.literal_eval(keywords["default"]) is None
    refusals = [n for n in ast.walk(main) if isinstance(n, ast.If)
                and _text(n.test) == "args.support_meta is not None and args.action_mode != 'support-log'"]
    assert len(refusals) == 1 and len(refusals[0].body) == 1 and _text(refusals[0].body[0]).startswith("ap.error(")


def test_the_client_reads_the_layout_from_the_meta_and_takes_the_hand_alone_without_one():
    main = _main(CURRENT)
    layout = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and _text(n.targets[0]) == "state_mode"]
    assert len(layout) == 1
    assert _text(layout[0].value) == ("support_state_mode(support_meta) if args.support_meta is not None else 'none'")
    dims = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and _text(n.targets[0]) == "state_dim"]
    assert len(dims) == 1
    assert G.support_state_mode is support_state_mode


def test_the_layout_statements_read_the_meta_they_are_given_and_fall_back_to_the_hand_alone(tmp_path):
    main = _main(CURRENT)
    stmts = [next(n for n in main.body if isinstance(n, ast.Assign) and _text(n.targets[0]) == name)
             for name in ("support_meta", "state_mode", "drop_near_constant", "state_dim")]
    code = compile(ast.Module(body=stmts, type_ignores=[]), "<layout>", "exec")
    env = {"json": json, "support_state_mode": support_state_mode,
           "support_drop_near_constant": support_drop_near_constant}

    env["args"] = types.SimpleNamespace(support_meta=None)
    exec(code, env)
    assert (env["state_mode"], env["state_dim"]) == ("none", 9)

    meta = tmp_path / "support_height.json"
    meta.write_text(json.dumps({"state_support": "append", "state_dim": 10}))
    env["args"] = types.SimpleNamespace(support_meta=meta)
    exec(code, env)
    assert (env["state_mode"], env["state_dim"]) == ("append", 10)

    meta.write_text(json.dumps({"state_support": "none", "state_dim": 9}))
    exec(code, env)
    assert (env["state_mode"], env["state_dim"]) == ("none", 9)

    meta.write_text(json.dumps({"eps_m": 0.001}))  # a dataset from before the key: s appended, so ten entries
    exec(code, env)
    assert (env["state_mode"], env["state_dim"]) == ("append", 10)

    meta.write_text(json.dumps({"state_support": "none", "drop_near_constant": True}))  # the pose alone: seven entries
    exec(code, env)
    assert (env["state_mode"], env["drop_near_constant"], env["state_dim"]) == ("none", True, 7)

    meta.write_text(json.dumps({"state_support": "append", "drop_near_constant": True, "state_dim": 8}))
    exec(code, env)
    assert (env["state_mode"], env["drop_near_constant"], env["state_dim"]) == ("append", True, 8)

    meta.write_text(json.dumps({"drop_near_constant": True}))  # no state_dim, no state_support: append, pose plus s
    exec(code, env)
    assert (env["state_mode"], env["drop_near_constant"], env["state_dim"]) == ("append", True, 8)

    env["args"] = types.SimpleNamespace(support_meta=None)
    exec(code, env)
    assert (env["drop_near_constant"], env["state_dim"]) == (False, 9)


def test_under_drop_the_policy_state_is_the_pose_and_the_support_plane_is_appended_under_append():
    arr = np.arange(16.0)
    pose = G.policy_state(arr, 7, None, True)
    assert pose.dtype == np.float32 and pose.shape == (7,)
    np.testing.assert_array_equal(pose, np.array(range(9, 16), dtype=np.float32))  # the pose, the fingers left out
    out = G.policy_state(arr, 8, 0.011, True)
    assert out.shape == (8,) and out[-1] == np.float32(0.011)
    np.testing.assert_array_equal(out[:7], pose)
    with pytest.raises(SystemExit, match="of 9 entries but this client builds 7"):
        G.policy_state(arr, 9, None, True)  # the policy wants the hand, the client builds the pose
    with pytest.raises(SystemExit, match="of 7 entries but this client builds 9"):
        G.policy_state(arr, 7)  # the policy wants the pose, the client builds the hand


def test_each_layout_the_converter_writes_is_sent_with_the_entries_it_has():
    arr = np.arange(16.0)
    s = 0.011
    hand = np.array(range(7, 16), dtype=np.float32)
    pose = np.array(range(9, 16), dtype=np.float32)
    cases = [  # (drop_near_constant, support plane, what the client must send)
        (False, None, hand),
        (False, s, np.append(hand, np.float32(s))),
        (True, None, pose),
        (True, s, np.append(pose, np.float32(s))),
    ]
    for drop, support_z, want in cases:
        st = G.policy_state(arr, len(want), support_z, drop)
        np.testing.assert_array_equal(st, want)
