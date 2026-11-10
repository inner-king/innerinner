# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stages 8 and 8b read an object's own mask and the source frame only when its record says so (SPEC_crop_source 3; I58).

An object that stage 5 cut out of the source frame (crop_source: original) has object_mask/iter_k.png, the source
frame's depth (original_depth.npy) and the source picture (source_padded_resized.png). Stage 8 and 8b read those for
exactly the objects whose obj_cat_list record says "crop_source": "original". A file that exists is not enough to say so,
since an edited run writes no such file. The mesh loops are too long to run here, so each test reads the stage's
syntax tree, and the one helper that is module-level is extracted and run on synthetic folders.

S8_STAGE_FILE and S8B_STAGE_FILE replace the stage files, for coder2_i58/mutate_wiring.py.
"""

import ast
import json
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGE8 = pathlib.Path(os.environ.get(
    "S8_STAGE_FILE", ROOT / "scripts/pipeline/A_reconstruction/stages/8_match_object_poses.py"))
STAGE8B = pathlib.Path(os.environ.get(
    "S8B_STAGE_FILE", ROOT / "scripts/pipeline/A_reconstruction/stages/8b_match_object_poses_interactive.py"))
SELECTOR = "idx in from_source_frame"
LOOP = "sorted(from_source_frame)"


def _tree(path):
    return ast.parse(pathlib.Path(path).read_text(), filename=str(path))


def _main(tree, path):
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    assert len(found) == 1, f"expected one main() in {path.name}"
    return found[0]


def _parents(root):
    parents = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosed_by_selector(node, parents):
    """True when ``node`` is in the body of ``if idx in from_source_frame`` or of ``for ... in sorted(from_source_frame)``."""
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.If) and child in parent.body and ast.unparse(parent.test) == SELECTOR:
            return True
        if isinstance(parent, ast.For) and child in parent.body and ast.unparse(parent.iter) == LOOP:
            return True
        child = parent
    return False


def _extract_helper(path, name):
    tree = _tree(path)
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1, f"expected one {name}() in {path.name}"
    namespace = {"Path": pathlib.Path, "json": json}
    exec(compile(ast.Module(body=[found[0]], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("path", [STAGE8, STAGE8B], ids=["stage8", "stage8b"])
def test_original_crop_indices_follows_the_record_not_the_file(path, tmp_path):
    helper = _extract_helper(path, "original_crop_indices")
    scene = tmp_path / "scene"
    (scene / "obj_cat_list").mkdir(parents=True)
    (scene / "object_mask").mkdir()
    records = {
        0: {"removed_obj_phrase": "pot"},                  # an edited object
        1: {"crop_source": "original"},                    # a matched cut-out, file present
        2: {"crop_source": "edited", "crop_reason": "no_original"},
        3: {"crop_source": "original", "crop_reason": "rescued"},  # a rescued object, no file yet
    }
    for index, record in records.items():
        (scene / "obj_cat_list" / f"iter_{index}.json").write_text(json.dumps(record))
    for index in (0, 1, 2):
        (scene / "object_mask" / f"iter_{index}.png").write_bytes(b"not read here")
    assert helper(str(scene)) == {1, 3}


@pytest.mark.parametrize("path", [STAGE8, STAGE8B], ids=["stage8", "stage8b"])
def test_no_record_means_no_source_frame_read(path, tmp_path):
    helper = _extract_helper(path, "original_crop_indices")
    scene = tmp_path / "scene"
    (scene / "obj_cat_list").mkdir(parents=True)
    (scene / "object_mask").mkdir()
    (scene / "object_mask" / "iter_0.png").write_bytes(b"x")
    (scene / "obj_cat_list" / "iter_0.json").write_text(json.dumps({"base_iter": 2}))
    assert helper(str(scene)) == set()


@pytest.mark.parametrize("path", [STAGE8, STAGE8B], ids=["stage8", "stage8b"])
def test_the_source_frame_reads_sit_under_the_record_selector(path):
    main = _main(_tree(path), path)
    parents = _parents(main)
    assigned = [node for node in ast.walk(main) if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "from_source_frame" for t in node.targets)]
    assert len(assigned) == 1 and ast.unparse(assigned[0].value) == "original_crop_indices(scene_dir)"
    selectors = [node for node in ast.walk(main) if isinstance(node, ast.If) and ast.unparse(node.test) == SELECTOR]
    assert len(selectors) == 1, f"{path.name} must select on the record exactly once"
    selector = selectors[0]
    # The original-mode branch reads the source frame; the else keeps the edited reads, through resolve_base_iteration.
    body_text = "\n".join(ast.unparse(stmt) for stmt in selector.body)
    else_text = "\n".join(ast.unparse(stmt) for stmt in selector.orelse)
    assert "object_mask/iter_" in body_text and "original_depth.npy" in body_text and "source_padded_resized.png" in body_text
    assert "removal_mask" not in body_text, "the original-mode branch must not read the removal mask"
    assert "resolve_base_iteration(" in else_text and "removal_mask/iter_" in else_text
    for node in ast.walk(main):
        is_string = isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str))
        if is_string and "object_mask" in ast.unparse(node) and not isinstance(parents.get(node), ast.JoinedStr):
            assert _enclosed_by_selector(node, parents), (
                f"{path.name} line {node.lineno} reads object_mask outside the record selector")


def test_stage8_size_ring_keeps_off_the_source_frame_masks_by_record():
    main = _main(_tree(STAGE8), STAGE8)
    parents = _parents(main)
    loops = [node for node in ast.walk(main) if isinstance(node, ast.For) and ast.unparse(node.iter) == LOOP]
    assert len(loops) == 1, "the size ring must add the object masks of the record-selected objects, once"
    loop_text = "\n".join(ast.unparse(stmt) for stmt in loops[0].body)
    assert "object_mask/iter_" in loop_text and "removal_mask" not in loop_text
    removal_loops = [node for node in ast.walk(main) if isinstance(node, ast.For) and "removal_mask" in ast.unparse(node.iter)]
    assert len(removal_loops) == 1, "the removal-mask loop of the size ring must still be there"
    assert not _enclosed_by_selector(loops[0], parents), "the ring loop is itself on the record list, not inside the mesh loop"
