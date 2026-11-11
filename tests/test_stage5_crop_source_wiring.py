# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The crop_source wiring of stage 5 (SPEC_crop_source v2 and the v3 memo; I58).

Stage 5's main() is about 2500 lines and needs the models, so it cannot run here. These tests read its syntax tree
instead. They check that every new call sits under a guard that is false in the default mode ("edited"), that a trace
skips no shared bookkeeping, that the resume refusal comes before anything is deleted, that the edited-mode guard
on original output is on, and that the rescue step writes what stage 6 and stage 8 read. The helper functions that
are module-level are extracted from the tree and run on synthetic folders.

S5_STAGE_FILE replaces the stage file, so that coder2_i58/mutate_wiring.py can run the same tests on mutated copies.
"""

import ast
import glob
import json
import os
import pathlib
import types

import numpy as np
import pytest
import yaml
from PIL import Image

from simfoundry.pipeline import crop_source as cs
from simfoundry.utils.python_utils import atomic_output_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGE = pathlib.Path(os.environ.get("S5_STAGE_FILE", ROOT / "scripts/pipeline/A_reconstruction/stages/5_decompose_scene.py"))
CFG = pathlib.Path(os.environ.get("REAL2SIM_CFG_FILE", ROOT / "scripts/cfg/real2sim_cfg.yaml"))

# Calls that exist only for crop_source: original (the module's API, the pool's methods, and the stage's own helpers).
NEW_CALLS = {
    "OriginalPool", "from_cfg", "match_iteration_mask", "composite_cutout", "crop_like_5a", "occluded_by",
    "object_record", "original_instance_record", "rescue_plan", "rescued_source_objects", "trace_record", "source_prompt_of",
    "_holds_original_output", "add_prompt", "claim", "claimed_union", "claimed_masks", "claimed_order", "entries",
    "check_pixel_source", "check_start_picture", "cutout_pixels",
}
# A guard counts when its test names one of these. Each is False in the default mode: crop_source is "edited", or the
# variable it names is None unless the original mode made it.
GUARD_TOKENS = ("crop_source ==", "crop_decision is not None", "is_trace", "matched_cutout is not None")
# The keys of the manifest that stage 6 and 8 read today. The original mode only adds keys next to them.
MANIFEST_KEYS = [
    "vlm_categories", "pruned_phrases", "removed_obj_phrase", "vlm_categories_full", "pruned_phrases_full",
    "removed_obj_phrase_full", "is_valid_removed_obj", "valid_pixel_proportion", "valid_pixel_proportion_threshold",
    "skipped_objects", "base_iter", "removal_prompt_type", "removal_prompt_types_tried",
]


def _tree(path=STAGE):
    return ast.parse(pathlib.Path(path).read_text(), filename=str(path))


def _main(tree):
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    assert len(found) == 1, f"expected one main() in {STAGE.name}, found {len(found)}"
    return found[0]


def _parents(root):
    parents = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _call_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _guarded(node, parents, tokens=GUARD_TOKENS):
    """True when some If above ``node`` names one of ``tokens`` and ``node`` is in its body (not its else).

    A call written inside the test of such an If (``if crop_source == "edited" and _holds_original_output(...)``) is
    guarded by that test, since the test is what decides it.
    """
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.If) and (child in parent.body or child is parent.test) and any(
                t in ast.unparse(parent.test) for t in tokens):
            return True
        child = parent
    return False


def _under_test(node, parents, tokens):
    """True when ``node`` is in the body of an If whose test names one of ``tokens`` (the else does not count)."""
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.If) and child in parent.body and any(t in ast.unparse(parent.test) for t in tokens):
            return True
        child = parent
    return False


def _extract_function(tree, name, namespace):
    """Compile the module-level function ``name`` of the stage's tree into ``namespace`` and return it."""
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1, f"expected one {name}() in {STAGE.name}, found {len(found)}"
    module = ast.Module(body=[found[0]], type_ignores=[])
    exec(compile(module, str(STAGE), "exec"), namespace)
    return namespace[name]


def _stage_namespace():
    return {
        "os": os, "json": json, "glob": glob, "np": np, "Image": Image, "atomic_output_path": atomic_output_path,
        "MARGIN_FRACTION": cs.MARGIN_FRACTION, "MARGIN_FRACTION_FOCUS": cs.MARGIN_FRACTION_FOCUS,
        "crop_like_5a": cs.crop_like_5a, "occluded_by": cs.occluded_by, "originals_record": cs.originals_record,
        "cutout_pixels": cs.cutout_pixels,
        "pack_originals": cs.pack_originals, "rescued_object_record": cs.rescued_object_record,
        "original_instance_record": cs.original_instance_record, "set_rescue_outcome": cs.set_rescue_outcome,
    }


# ---------------------------------------------------------------- static: main()'s syntax tree


def test_new_calls_sit_under_a_guard_that_is_false_by_default():
    main = _main(_tree())
    parents = _parents(main)
    checked = []
    for node in ast.walk(main):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        source_sam3 = name == "predict_segmentation" and any(
            isinstance(kw.value, ast.Name) and kw.value.id == "source_pil" for kw in node.keywords)
        if name in NEW_CALLS or source_sam3:
            checked.append((name, node.lineno))
            assert _guarded(node, parents), f"{name}() at line {node.lineno} is not under a crop_source guard"
    # The stage makes each of these calls at least once; a mutant that drops one of them shows here.
    names = {name for name, _ in checked}
    assert {"OriginalPool", "from_cfg", "match_iteration_mask", "composite_cutout", "crop_like_5a", "object_record",
            "rescue_plan", "rescued_source_objects", "trace_record", "_holds_original_output", "add_prompt"} <= names


def test_object_mask_is_written_only_for_a_matched_cutout_or_by_the_original_setup():
    main = _main(_tree())
    parents = _parents(main)
    hits = []
    for node in ast.walk(main):
        is_string = isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str))
        text = ast.unparse(node)
        # Only path strings count (a dict key or a subscript named object_mask is not a file), and only the outermost
        # string of an f-string, not its literal parts.
        is_path = "/object_mask" in text or "object_mask/" in text
        if is_string and is_path and not isinstance(parents.get(node), ast.JoinedStr):
            hits.append(node)
            assert _guarded(node, parents), f"object_mask at line {node.lineno} is written outside the original mode"
    assert len(hits) == 2, "expected the object_mask folder setup and the matched save, both under the original mode"


def test_edited_guard_on_original_output_is_on_only_in_the_edited_mode():
    main = _main(_tree())
    found = 0
    for node in ast.walk(main):
        if isinstance(node, ast.If) and any(
                isinstance(c, ast.Call) and _call_name(c) == "_holds_original_output" for c in ast.walk(node.test)):
            found += 1
            assert "crop_source == 'edited'" in ast.unparse(node.test), (
                f"the original-output guard at line {node.lineno} must run only when crop_source is edited")
    assert found == 1


def test_original_refusals_come_before_any_deletion_or_write():
    main = _main(_tree())
    refusals = [node for node in main.body if isinstance(node, ast.If) and "last_recorded_iteration" in ast.unparse(node.test)]
    assert len(refusals) == 1, "the resume refusal must be a single top-level If in main()"
    deletes = [node.lineno for node in ast.walk(main)
               if isinstance(node, ast.Call) and _call_name(node) == "delete_iteration_data"]
    writes = [node.lineno for node in ast.walk(main)
              if isinstance(node, ast.Call) and _call_name(node) == "write_decomposition_frame"]
    assert deletes and writes
    assert refusals[0].lineno < min(deletes), "the refusal must come before the stage deletes anything"
    assert refusals[0].lineno < min(writes), "the refusal must come before the stage writes the frame record"


def test_shared_bookkeeping_is_not_skipped_for_a_trace():
    main = _main(_tree())
    parents = _parents(main)
    targets = []
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and _call_name(node) == "remove" and any(
                isinstance(a, ast.Name) and a.id == "obj_phrase_to_remove" for a in node.args):
            targets.append(("category removal", node))
        if isinstance(node, ast.Call) and _call_name(node) == "append" and isinstance(node.func, ast.Attribute) and (
                isinstance(node.func.value, ast.Name) and node.func.value.id == "removed_regions"):
            targets.append(("removed_regions append", node))
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "last_kept_iteration" for t in node.targets) \
                and isinstance(node.value, ast.Name) and node.value.id == "current_iteration":
            targets.append(("last_kept_iteration", node))
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "current_iteration":
            targets.append(("current_iteration += 1", node))
    labels = {label for label, _ in targets}
    assert {"category removal", "removed_regions append", "last_kept_iteration", "current_iteration += 1"} <= labels
    for label, node in targets:
        assert not _under_test(node, parents, ("is_trace", "crop_decision")), (
            f"{label} at line {node.lineno} must run for a trace as for an object")


def test_trace_branch_writes_a_skip_record_and_no_object_record():
    main = _main(_tree())
    trace_ifs = [node for node in ast.walk(main) if isinstance(node, ast.If) and ast.unparse(node.test) == "is_trace"]
    assert len(trace_ifs) == 1
    body_text = "\n".join(ast.unparse(stmt) for stmt in trace_ifs[0].body)
    assert "write_skipped_iteration" in body_text
    # No object file of the trace: no manifest, no cut-out (path strings; a record value may name the list obj_cat_list).
    assert "obj_cat_list/" not in body_text and "masked_object" not in body_text


def test_manifest_keeps_its_keys_and_original_mode_only_adds():
    main = _main(_tree())
    manifests = [node for node in ast.walk(main) if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "manifest" for t in node.targets) and isinstance(node.value, ast.Dict)]
    assert len(manifests) == 1
    keys = [key.value for key in manifests[0].value.keys]
    assert keys == MANIFEST_KEYS, "the manifest keys (and their order) of the edited mode must not change"


def test_crop_source_defaults_in_the_config():
    config = yaml.safe_load(CFG.read_text())
    s5 = config["s5_scene"]
    assert s5["crop_source"] == "edited"
    assert s5["original_match_iou"] == 0.5
    assert s5["revealed_inside_fraction"] == 0.5
    assert s5["rescue_scope"] == "dropped_only"
    assert s5["rescue_min_score"] == 0.5
    assert s5["matched_mask"] == "union"
    assert s5["matched_pixel_source"] == "frame"


# ---------------------------------------------------------------- runtime: helpers extracted from the stage


def test_source_prompt_of_is_the_lowercased_first_name():
    tree = _tree()
    ns = {}
    get_primary = _extract_function(tree, "get_primary_object_name", ns)
    ns["get_primary_object_name"] = get_primary
    source_prompt = _extract_function(tree, "source_prompt_of", ns)
    assert source_prompt("Cutting Board / white cutting board / plastic board") == "cutting board"
    assert source_prompt("wooden cutting board") == "wooden cutting board"
    assert source_prompt(" / Knife") == "/ knife", "an empty first name queries the whole category, lowercased"
    assert source_prompt("Red Onion") == "red onion"


def test_holds_original_output_reads_only_files_and_record_keys(tmp_path):
    tree = _tree()
    ns = _stage_namespace()
    holds = _extract_function(tree, "_holds_original_output", ns)
    plain = tmp_path / "plain"
    (plain / "obj_cat_list").mkdir(parents=True)
    (plain / "obj_cat_list" / "iter_0.json").write_text(json.dumps({"removed_obj_phrase": "pot"}))
    assert holds(str(plain)) is False
    (plain / "obj_cat_list" / "iter_1.json").write_text(json.dumps({"crop_source": "original"}))
    assert holds(str(plain)) is True
    skipped = tmp_path / "skipped"
    (skipped / "skipped_iterations").mkdir(parents=True)
    (skipped / "skipped_iterations" / "iter_3.json").write_text(json.dumps({"crop_source": "original"}))
    assert holds(str(skipped)) is True
    marked = tmp_path / "marked"
    marked.mkdir()
    (marked / "original_instances.json").write_text("{}")
    assert holds(str(marked)) is True
    assert holds(str(tmp_path / "missing")) is False


def _synthetic_scene(h=48, w=48):
    rng = np.random.default_rng(0)
    source = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    start = source.copy()
    start[:4] = 0  # a few pixels the start image disagrees with, so the validity share is not trivially 1
    def rect(y0, y1, x0, x1):
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask
    return source, start, rect


def _pool_for(rect, tomato_score=0.2):
    pool = cs.OriginalPool()
    pool.add_prompt("cutting board", [0.9], [rect(10, 30, 10, 30)])            # index 0, free, score 0.9
    pool.add_prompt("knife", [0.8], [rect(30, 44, 2, 16)])                     # index 1, claimed below
    pool.add_prompt("tomato", [tomato_score], [rect(2, 12, 36, 46)])           # index 2, free
    pool.claim(1, 2)
    return pool


def _trace_record(iteration, area):
    return {"iteration": iteration, "crop_source": "original", "reason": "trace_no_original_match",
            "best_original_iou": 0.0, "best_original_index": None, "mask_area": area,
            "rescued_as": None, "lost_object": None, "removed_obj_phrase": "x", "base_iter": None}


def test_rescued_source_objects_writes_the_records_stage_6_and_8_read(tmp_path):
    tree = _tree()
    ns = _stage_namespace()
    write_skipped = _extract_function(tree, "write_skipped_iteration", ns)
    ns["write_skipped_iteration"] = write_skipped
    rescue_objects = _extract_function(tree, "rescued_source_objects", ns)
    source, start, rect = _synthetic_scene()
    pool = _pool_for(rect)
    params = cs.CropParams()
    traces = [rect(10, 30, 18, 38),   # its own original is 0 (free, score 0.9): rescued
              rect(30, 44, 4, 18),    # its own original is 1, claimed by iteration 2: lost
              rect(0, 4, 0, 4)]       # no original overlaps: lost
    plan = cs.rescue_plan(pool, traces, params)
    assert [o.rescued for o in plan.per_dropped] == [True, False, False]
    out = tmp_path / "s5"
    (out / "skipped_iterations").mkdir(parents=True)
    (out / "obj_cat_list").mkdir()
    for d in ("masked_object", "masked_object_focus", "object_mask"):
        (out / d).mkdir()
    records = [_trace_record(4, 400), _trace_record(6, 196), _trace_record(7, 16)]
    for record in records:
        write_skipped(str(out), record)
    summary = rescue_objects(str(out), pool, plan, records, 8, source, start, 1.0, (48, 48), 0.1,
                             asked_names={"cutting board": ["Cutting Board / white cutting board"]})
    assert summary == {"rescued_own": 1, "rescued_extra": 0, "lost_object": 2, "originals": 3}

    rescued = json.loads((out / "obj_cat_list" / "iter_8.json").read_text())
    assert rescued["crop_source"] == "original" and rescued["crop_reason"] == "rescued"
    assert rescued["base_iter"] is None, "a rescued object must say base_iter null, or resume falls back to the chain"
    assert rescued["removed_obj_phrase"] == "cutting board"
    assert isinstance(rescued["is_valid_removed_obj"], bool)
    assert rescued["rescued_from_iteration"] == 4
    assert rescued["original_instance"]["index"] == 0

    trace4 = json.loads((out / "skipped_iterations" / "iter_4.json").read_text())
    assert trace4["rescued_as"] == 8 and trace4["lost_object"] is False
    for iteration in (6, 7):
        trace = json.loads((out / "skipped_iterations" / f"iter_{iteration}.json").read_text())
        assert trace["rescued_as"] is None and trace["lost_object"] is True

    cut = np.array(Image.open(out / "masked_object" / "iter_8.png"))
    assert cut.shape == (48, 48, 4)
    outside = cut[..., 3] == 0
    assert outside.any() and (cut[outside][:, :3] == 255).all(), "outside the mask the cut-out is white"
    focus = np.array(Image.open(out / "masked_object_focus" / "iter_8.png"))
    assert focus.shape == (48, 48, 4)
    mask_png = np.array(Image.open(out / "object_mask" / "iter_8.png"))
    assert set(np.unique(mask_png).tolist()) == {0, 255}
    assert (mask_png > 0).sum() == 400

    listing = json.loads((out / "original_instances.json").read_text())
    by_index = {item["index"]: item for item in listing["originals"]}
    assert by_index[0]["reason"] == "rescued as object 8"
    assert by_index[1]["reason"] == "taken by iteration 2"
    assert by_index[2]["reason"] == "left free"
    assert listing["asked_prompts"] == {"cutting board": ["Cutting Board / white cutting board"]}
    packed = np.load(out / "original_instances.npz")
    assert {"shape", "o0", "o1", "o2"} <= set(packed.files)


def test_scope_all_numbers_the_extra_originals_after_the_own_rescues(tmp_path):
    tree = _tree()
    ns = _stage_namespace()
    write_skipped = _extract_function(tree, "write_skipped_iteration", ns)
    ns["write_skipped_iteration"] = write_skipped
    rescue_objects = _extract_function(tree, "rescued_source_objects", ns)
    source, start, rect = _synthetic_scene()
    pool = _pool_for(rect, tomato_score=0.7)  # index 2 is free and scores over the bar: an extra under scope "all"
    plan = cs.rescue_plan(pool, [rect(10, 30, 18, 38)], cs.CropParams(rescue_scope="all"))
    assert plan.extra == [2]
    out = tmp_path / "s5"
    (out / "skipped_iterations").mkdir(parents=True)
    (out / "obj_cat_list").mkdir()
    for d in ("masked_object", "masked_object_focus", "object_mask"):
        (out / d).mkdir()
    records = [_trace_record(4, 400)]
    write_skipped(str(out), records[0])
    rescue_objects(str(out), pool, plan, records, 8, source, start, 1.0, (48, 48), 0.1)
    own = json.loads((out / "obj_cat_list" / "iter_8.json").read_text())
    assert own["rescued_from_iteration"] == 4
    extra = json.loads((out / "obj_cat_list" / "iter_9.json").read_text())
    assert extra["rescued_from_iteration"] is None and extra["crop_reason"] == "rescued"
    assert extra["base_iter"] is None and extra["original_instance"]["index"] == 2
    listing = json.loads((out / "original_instances.json").read_text())
    assert {item["index"]: item["rescued_as"] for item in listing["originals"]} == {0: 8, 1: None, 2: 9}


# ---------------------------------------------------------------- matched_pixel_source: where the matched pixels come from

PIXEL_CHECKS = ("check_pixel_source", "check_start_picture", "cutout_pixels")
TRY_TYPES = tuple(t for t in (getattr(ast, "Try", None), getattr(ast, "TryStar", None)) if t is not None)


def _module_function(tree, name):
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1, f"expected one {name}() in {STAGE.name}, found {len(found)}"
    return found[0]


def _calls_named(root, name):
    return [node for node in ast.walk(root) if isinstance(node, ast.Call) and _call_name(node) == name]


def _inside_try(node, parents, stop):
    """True when a try statement lies between ``node`` and ``stop``, the function that holds it."""
    child = node
    while child is not stop and child in parents:
        if isinstance(parents[child], TRY_TYPES):
            return True
        child = parents[child]
    return False


def _enclosing_ifs(node, parents):
    """The If statements that hold ``node`` in their body (not their else), innermost first."""
    found, child = [], node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.If) and child in parent.body:
            found.append(parent)
        child = parent
    return found


def test_pixel_checks_are_called_once_each_under_the_original_guard():
    main = _main(_tree())
    parents = _parents(main)
    for name in PIXEL_CHECKS:
        calls = _calls_named(main, name)
        assert len(calls) == 1, f"{name}() must be called once in main(), found {len(calls)}"
        assert _guarded(calls[0], parents, tokens=("crop_source == 'original'",)), (
            f"{name}() at line {calls[0].lineno} is not under if crop_source == 'original'")


def test_pixel_source_refusal_comes_before_the_first_write():
    main = _main(_tree())
    [call] = _calls_named(main, "check_pixel_source")
    top_ifs = [stmt for stmt in main.body if isinstance(stmt, ast.If) and ast.unparse(stmt.test) == "crop_source == 'original'"]
    assert any(call in ast.walk(stmt) for stmt in top_ifs), (
        "check_pixel_source() must sit in a top-level if crop_source == 'original' of main()")
    writes = [node.lineno for node in ast.walk(main) if isinstance(node, ast.Call)
              and _call_name(node) in ("mkdir", "delete_iteration_data", "write_decomposition_frame")]
    assert writes and call.lineno < min(writes), "the refusal must come before the first mkdir, delete or frame write"


def test_start_picture_check_and_cutout_come_after_the_upsample_step():
    main = _main(_tree())
    parents = _parents(main)
    [check] = _calls_named(main, "check_start_picture")
    [cut] = _calls_named(main, "cutout_pixels")
    saves = [node for node in _calls_named(main, "save") if "source_resized_upsampled_image_fpath" in ast.unparse(node)]
    assert len(saves) == 1, "expected the one save of the upsampled picture"
    assert saves[0].lineno < check.lineno and saves[0].lineno < cut.lineno, "both must come after the upsample step"
    check_tests = {ast.unparse(stmt.test) for stmt in _enclosing_ifs(check, parents)}
    assert "crop_source == 'original'" in check_tests and any("start_picture" in test for test in check_tests), (
        "check_start_picture() must run only under the original guard and for start_picture")
    cut_tests = {ast.unparse(stmt.test) for stmt in _enclosing_ifs(cut, parents)}
    assert "crop_source == 'original'" in cut_tests
    assigned = [node for node in ast.walk(main) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "cutout_rgb" for target in node.targets)]
    assert len(assigned) == 1 and assigned[0].value is cut, "cutout_rgb must be the value of cutout_pixels()"


def test_the_pixel_checks_are_not_inside_a_try_block():
    """A refusal that is caught, reworded or dropped would let a start picture equal to the frame run as a frame run."""
    main = _main(_tree())
    parents = _parents(main)
    for name in PIXEL_CHECKS:
        for call in _calls_named(main, name):
            assert not _inside_try(call, parents, main), f"{name}() at line {call.lineno} is inside a try block"
    function = _module_function(_tree(), "rescued_source_objects")
    function_parents = _parents(function)
    for call in _calls_named(function, "cutout_pixels"):
        assert not _inside_try(call, function_parents, function), f"cutout_pixels() at line {call.lineno} is inside a try block"


def test_composite_takes_the_cutout_picture_as_its_source():
    main = _main(_tree())
    calls = _calls_named(main, "composite_cutout")
    assert len(calls) == 1
    source = calls[0].args[1]
    assert isinstance(source, ast.Name) and source.id == "cutout_rgb", "the second argument must be cutout_rgb, not source_rgb"


def test_rescue_call_passes_the_pixel_source_and_the_summary_records_it():
    main = _main(_tree())
    [call] = _calls_named(main, "rescued_source_objects")
    keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    assert keywords.get("matched_pixel_source") == "crop_params.matched_pixel_source"
    summaries = [node for node in ast.walk(main) if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
                 and isinstance(node.targets[0].slice, ast.Constant) and node.targets[0].slice.value == "crop_source_summary"]
    assert len(summaries) == 1 and isinstance(summaries[0].value, ast.Dict)
    keys = {key.value for key in summaries[0].value.keys if isinstance(key, ast.Constant)}
    assert {"kind_counts", "matched_pixel_source"} <= keys


def test_matched_record_gets_the_pixel_source_and_the_other_records_do_not():
    main = _main(_tree())
    parents = _parents(main)
    calls = _calls_named(main, "object_record")
    matched = [c for c in calls if "crop_decision.kind == 'matched'" in {ast.unparse(s.test) for s in _enclosing_ifs(c, parents)}]
    others = [c for c in calls if c not in matched]
    assert len(matched) == 1 and len(others) == 1, "expected one matched and one other object_record() call"
    assert "matched_pixel_source" in {kw.arg for kw in matched[0].keywords}
    assert "matched_pixel_source" not in {kw.arg for kw in others[0].keywords}


def test_the_new_key_is_read_or_written_only_under_the_original_guard():
    main = _main(_tree())
    parents = _parents(main)
    hits = [node for node in ast.walk(main)
            if (isinstance(node, ast.Constant) and node.value == "matched_pixel_source")
            or (isinstance(node, ast.keyword) and node.arg == "matched_pixel_source")]
    assert len(hits) == 4, "the config read, the matched record, the rescue call and the summary: four places in main()"
    for node in hits:
        assert _guarded(node, parents), f"matched_pixel_source at line {node.lineno} is outside the original guard"


def test_the_stage_default_of_the_pixel_source_is_frame():
    defaults = []
    for node in _calls_named(_tree(), "get"):
        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "matched_pixel_source":
            defaults.append(ast.unparse(node.args[1]) if len(node.args) > 1 else None)
    assert defaults == ["'frame'"], f"the stage's default must be frame, found {defaults}"


def _rescue_once(out, source, start, rect, **keywords):
    """The end-of-run rescue of the trace setup of test_rescued_source_objects_writes_the_records_stage_6_and_8_read, into
    ``out`` (not made yet). The trace records are written first, as stage 5 writes them. Returns the summary."""
    tree = _tree()
    ns = _stage_namespace()
    ns["write_skipped_iteration"] = _extract_function(tree, "write_skipped_iteration", ns)
    rescue_objects = _extract_function(tree, "rescued_source_objects", ns)
    pool = _pool_for(rect)
    traces = [rect(10, 30, 18, 38), rect(30, 44, 4, 18), rect(0, 4, 0, 4)]
    plan = cs.rescue_plan(pool, traces, cs.CropParams())
    out.mkdir(parents=True)
    (out / "skipped_iterations").mkdir()
    (out / "obj_cat_list").mkdir()
    for d in ("masked_object", "masked_object_focus", "object_mask"):
        (out / d).mkdir()
    records = [_trace_record(4, 400), _trace_record(6, 196), _trace_record(7, 16)]
    for record in records:
        ns["write_skipped_iteration"](str(out), record)
    return rescue_objects(str(out), pool, plan, records, 8, source, start, 1.0, (48, 48), 0.1, **keywords)


def test_rescued_cutout_takes_its_pixels_from_the_start_picture_only_when_asked(tmp_path):
    source, _, rect = _synthetic_scene()
    start = source.copy()
    start[::2] = 255 - start[::2]    # every other row differs from the frame, so the two cut-outs differ inside the mask
    default, frame, picked, bad = (tmp_path / name for name in ("default", "frame", "start", "bad"))
    summary = _rescue_once(default, source, start, rect)
    assert _rescue_once(frame, source, start, rect, matched_pixel_source="frame") == summary
    unchanged = ("masked_object/iter_8.png", "masked_object_focus/iter_8.png", "object_mask/iter_8.png",
                 "obj_cat_list/iter_8.json", "skipped_iterations/iter_4.json", "skipped_iterations/iter_6.json",
                 "skipped_iterations/iter_7.json", "original_instances.json")
    for rel in unchanged:
        assert (default / rel).read_bytes() == (frame / rel).read_bytes(), f"{rel} changed with an explicit frame"
    assert _rescue_once(picked, source, start, rect, matched_pixel_source="start_picture") == summary
    mask = rect(10, 30, 10, 30)      # the original that the trace rescues (object 0)
    cut = np.array(Image.open(picked / "masked_object" / "iter_8.png"))
    assert np.array_equal(cut, cs.crop_like_5a(start, mask, 1.0, (48, 48), margin_fraction=cs.MARGIN_FRACTION)), (
        "the cut-out must be cut from the start picture")
    focus = np.array(Image.open(picked / "masked_object_focus" / "iter_8.png"))
    assert np.array_equal(focus, cs.crop_like_5a(start, mask, 1.0, (48, 48), margin_fraction=cs.MARGIN_FRACTION_FOCUS))
    assert not np.array_equal(cut, np.array(Image.open(default / "masked_object" / "iter_8.png"))), (
        "the two cut-outs must differ, or this test proves nothing")
    recorded = json.loads((picked / "obj_cat_list" / "iter_8.json").read_text())
    assert recorded["matched_pixel_source"] == "start_picture"
    default_record = json.loads((default / "obj_cat_list" / "iter_8.json").read_text())
    assert default_record["matched_pixel_source"] == "frame"
    assert recorded["valid_pixel_proportion"] == default_record["valid_pixel_proportion"], (
        "the validity share uses the same two pictures in both modes")
    with pytest.raises(ValueError):
        _rescue_once(bad, source, start, rect, matched_pixel_source="Frame")
    assert not (bad / "masked_object" / "iter_8.png").exists() and not (bad / "obj_cat_list" / "iter_8.json").exists()
    assert json.loads((bad / "skipped_iterations" / "iter_4.json").read_text())["rescued_as"] is None, (
        "a refused value must stop the rescue before any record is rewritten")


# ---------------------------------------------------------------- the two pixel blocks of main(), run with real values


class _Section:
    """What the early check reads from cfg.s5_scene: get(key, default), as the config does."""

    def __init__(self, values):
        self._values = dict(values)

    def get(self, key, default=None):
        return self._values.get(key, default)


def _top_block(call_name):
    """The top-level ``if crop_source == 'original':`` of main() that holds a call to ``call_name``."""
    main = _main(_tree())
    blocks = [stmt for stmt in main.body if isinstance(stmt, ast.If) and ast.unparse(stmt.test) == "crop_source == 'original'"
              and call_name in {_call_name(call) for call in ast.walk(stmt) if isinstance(call, ast.Call)}]
    assert len(blocks) == 1, f"expected one top-level if crop_source == 'original' holding {call_name}(), found {len(blocks)}"
    return blocks[0]


def _run_statement(statement, names):
    """Run one statement of the stage with ``names`` as its globals, the way the stage would run it."""
    exec(compile(ast.Module(body=[statement], type_ignores=[]), str(STAGE), "exec"), names)
    return names


def test_early_pixel_check_refuses_only_the_start_picture_without_the_upsample():
    block = _top_block("check_pixel_source")

    def run(crop_source, **values):
        names = {"crop_source": crop_source, "cfg": types.SimpleNamespace(s5_scene=_Section(values)),
                 "check_pixel_source": cs.check_pixel_source}
        return _run_statement(block, names)

    with pytest.raises(ValueError, match="start_picture_needs_upsampled_source"):
        run("original", matched_pixel_source="start_picture", use_upsampled_source_image=False)
    with pytest.raises(ValueError, match="must be true or false"):
        run("original", matched_pixel_source="start_picture", use_upsampled_source_image="false")
    run("original", matched_pixel_source="start_picture")                            # no key: the default True passes
    run("original", matched_pixel_source="frame", use_upsampled_source_image=False)  # frame does not read the flag
    run("original")                                                                  # no keys: frame, and True
    with pytest.raises(ValueError, match="must be one of"):
        run("original", matched_pixel_source="Frame")                                # a wrong value is refused
    run("edited", matched_pixel_source="Frame", use_upsampled_source_image="maybe")   # not run in the edited mode


def test_cutout_picture_follows_the_choice_and_the_start_check_refuses_an_equal_picture():
    block = _top_block("cutout_pixels")

    def run(crop_source, frame, start, mode):
        names = {"crop_source": crop_source, "crop_params": cs.CropParams(matched_pixel_source=mode), "source_rgb": frame,
                 "upsampled_obj_img_raw": start, "check_start_picture": cs.check_start_picture,
                 "cutout_pixels": cs.cutout_pixels}
        return _run_statement(block, names)

    frame = np.random.default_rng(3).integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    start = frame.copy()
    start[::2] = 255 - start[::2]    # every other row differs from the frame
    assert run("original", frame, start, "start_picture")["cutout_rgb"] is start
    assert run("original", frame, start, "frame")["cutout_rgb"] is frame
    with pytest.raises(ValueError, match="start_picture_equals_frame"):
        run("original", frame, frame.copy(), "start_picture")
    assert run("original", frame, frame.copy(), "frame")["cutout_rgb"] is frame
    assert "cutout_rgb" not in run("edited", frame, start, "start_picture")


def test_matched_record_says_the_configured_pixel_source():
    main = _main(_tree())
    parents = _parents(main)
    matched = [call for call in _calls_named(main, "object_record")
               if "crop_decision.kind == 'matched'" in {ast.unparse(stmt.test) for stmt in _enclosing_ifs(call, parents)}]
    assert len(matched) == 1, "expected one object_record() call under if crop_decision.kind == 'matched'"
    values = {kw.arg: ast.unparse(kw.value) for kw in matched[0].keywords}
    assert values.get("matched_pixel_source") == "crop_params.matched_pixel_source", (
        "the matched record must say the configured source, not a constant")


def test_summary_says_the_configured_pixel_source():
    main = _main(_tree())
    summaries = [node for node in ast.walk(main) if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
                 and isinstance(node.targets[0].slice, ast.Constant) and node.targets[0].slice.value == "crop_source_summary"]
    assert len(summaries) == 1 and isinstance(summaries[0].value, ast.Dict)
    summary = summaries[0].value
    values = {key.value: ast.unparse(value) for key, value in zip(summary.keys, summary.values) if isinstance(key, ast.Constant)}
    assert values.get("matched_pixel_source") == "crop_params.matched_pixel_source", (
        "the summary must say the configured source, not a constant")
