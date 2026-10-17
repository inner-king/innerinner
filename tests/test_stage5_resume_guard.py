# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 5 refuses to resume a decomposition that was taken from a different frame."""

import json
import logging
import os
import pathlib
import re

import pytest


# --------------------------------------------------------------------------------------
# Stage 5 resume guard
# --------------------------------------------------------------------------------------

def _load_stage5_module():
    """Import 5_decompose_scene's helpers without pulling in its heavy stage imports."""
    import ast
    import textwrap
    import types

    src = pathlib.Path("scripts/pipeline/A_reconstruction/stages/5_decompose_scene.py").read_text()
    tree = ast.parse(src)
    wanted = {
        "DECOMPOSITION_FRAME_FILENAME", "read_decomposition_frame", "write_decomposition_frame",
        "check_resume_frame_matches", "infer_resume_state", "load_iteration_record",
    }
    kept = [
        node for node in tree.body
        if (isinstance(node, (ast.FunctionDef,)) and node.name in wanted)
        or (isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) in wanted for t in node.targets))
    ]
    from simfoundry.pipeline.removal_ghosts import GHOST_SKIP_REASON

    module = types.ModuleType("stage5_helpers")
    module.__dict__.update(
        os=os, json=json, re=re, logger=logging.getLogger("stage5"), GHOST_SKIP_REASON=GHOST_SKIP_REASON,
    )
    exec(compile(ast.Module(body=kept, type_ignores=[]), "5_decompose_scene.py", "exec"), module.__dict__)
    return module


def _complete_iteration(out_dir, i):
    for sub, ext in [("post_object_removal", ".png"), ("metric_depth", ".npy"), ("obj_cat_list", ".json")]:
        d = out_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / f"iter_{i}{ext}").write_text("x")


def test_stage5_records_and_reads_back_its_frame(tmp_path):
    m = _load_stage5_module()
    assert m.read_decomposition_frame(str(tmp_path)) is None
    m.write_decomposition_frame(str(tmp_path), 5)
    assert m.read_decomposition_frame(str(tmp_path)) == 5


def test_stage5_refuses_to_resume_across_a_frame_change(tmp_path):
    m = _load_stage5_module()
    m.write_decomposition_frame(str(tmp_path), 0)
    _complete_iteration(tmp_path, 0)
    with pytest.raises(RuntimeError, match="mix object crops from two viewpoints"):
        m.check_resume_frame_matches(str(tmp_path), 5)


def test_stage5_resumes_normally_on_the_same_frame(tmp_path):
    m = _load_stage5_module()
    m.write_decomposition_frame(str(tmp_path), 5)
    _complete_iteration(tmp_path, 0)
    m.check_resume_frame_matches(str(tmp_path), 5)  # no raise


def test_stage5_warns_on_a_legacy_output_dir_with_no_marker(tmp_path, caplog):
    m = _load_stage5_module()
    _complete_iteration(tmp_path, 0)
    with caplog.at_level(logging.WARNING, logger="stage5"):
        m.check_resume_frame_matches(str(tmp_path), 5)  # no raise, but flagged
    assert "cannot be checked" in caplog.text


def test_stage5_is_quiet_on_an_empty_output_dir(tmp_path, caplog):
    m = _load_stage5_module()
    with caplog.at_level(logging.WARNING, logger="stage5"):
        m.check_resume_frame_matches(str(tmp_path), 5)
    assert caplog.text == ""


# --------------------------------------------------------------------------------------
# Resume across an iteration that removed a removal ghost
# --------------------------------------------------------------------------------------

def _write(out_dir, sub, name, text="x"):
    d = out_dir / sub
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text)


def test_stage5_resumes_after_an_iteration_that_removed_a_ghost(tmp_path):
    # Such an iteration edits the image like a removal but has a skip record, not a manifest.
    m = _load_stage5_module()
    _complete_iteration(tmp_path, 0)
    ghost_record = {"iteration": 1, "reason": "removal_ghost", "ghost_of_iter": 0, "removed": True,
                    "vlm_categories_full": ["teal tray", "green banana"]}
    _write(tmp_path, "post_object_removal", "iter_1.png")
    _write(tmp_path, "metric_depth", "iter_1.npy")
    _write(tmp_path, "skipped_iterations", "iter_1.json", json.dumps(ghost_record))

    resume_iteration, stale, metadata = m.infer_resume_state(str(tmp_path))

    assert (resume_iteration, stale, metadata) == (2, [], ghost_record)


def test_stage5_keeps_a_ghost_no_prompt_type_could_remove(tmp_path):
    # Nothing was written but the record, so the record alone completes the iteration:
    # otherwise a resume would delete it and try the same prompt types again.
    m = _load_stage5_module()
    _complete_iteration(tmp_path, 0)
    record = {"iteration": 1, "reason": "removal_ghost", "ghost_of_iter": 0, "removed": False,
              "removal_prompt_types_tried": ["outline", "bbox"]}
    _write(tmp_path, "removal_mask", "iter_1.png")
    _write(tmp_path, "pre_object_removal", "iter_1.png")
    _write(tmp_path, "skipped_iterations", "iter_1.json", json.dumps(record))

    resume_iteration, stale, metadata = m.infer_resume_state(str(tmp_path))

    assert (resume_iteration, stale, metadata) == (2, [], record)


def test_stage5_does_not_complete_an_iteration_with_a_stale_skip_record(tmp_path):
    # A rerun detected an object where an earlier run had found nothing, and died after the
    # image and depth but before the manifest: the old skip record must not complete it.
    m = _load_stage5_module()
    _complete_iteration(tmp_path, 0)
    _write(tmp_path, "skipped_iterations", "iter_1.json", json.dumps({"reason": "no_masks_detected"}))
    _write(tmp_path, "post_object_removal", "iter_1.png")
    _write(tmp_path, "metric_depth", "iter_1.npy")

    resume_iteration, stale, _ = m.infer_resume_state(str(tmp_path))

    assert (resume_iteration, stale) == (1, [1])


def test_stage5_needs_the_depth_of_an_erased_ghost(tmp_path):
    m = _load_stage5_module()
    _complete_iteration(tmp_path, 0)
    _write(tmp_path, "skipped_iterations", "iter_1.json",
           json.dumps({"reason": "removal_ghost", "ghost_of_iter": 0, "removed": True}))
    _write(tmp_path, "post_object_removal", "iter_1.png")  # interrupted before the depth

    resume_iteration, stale, _ = m.infer_resume_state(str(tmp_path))

    assert (resume_iteration, stale) == (1, [1])
