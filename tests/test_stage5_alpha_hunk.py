# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stage 5 alpha rule, run on the code of the stage rather than on a copy of it.

The stage is one `main()` of about 2000 lines, so it cannot be imported and its hunk cannot be moved out into a function
(this repository's tests may not change the stage). Instead the `if` that decides on the alpha of a removed-object frame
is taken out of the stage's syntax tree and compiled into a small function. The hunk is the only `if` in the stage with
that condition; if there is none or two, the test fails rather than picking one.
"""

import ast
import logging
import pathlib

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGE = ROOT / "scripts/pipeline/A_reconstruction/stages/5_decompose_scene.py"
CONDITION = "removed_obj_img_raw.ndim == 3 and removed_obj_img_raw.shape[-1] == 4"
SHAPE = (200, 200)  # 40000 pixels, so 1 % is exactly 400 pixels and 0.5 % is exactly 200

logger = logging.getLogger("test_stage5_alpha_hunk")


def _hunk():
    tree = ast.parse(STAGE.read_text())
    found = [node for node in ast.walk(tree) if isinstance(node, ast.If) and ast.unparse(node.test) == CONDITION]
    assert len(found) == 1, f"expected one alpha hunk in {STAGE.name}, found {len(found)}"
    return found[0]


def _run():
    """The hunk as a function `run(removed_obj_img_raw, out_dir, current_iteration, obj_removal_prompt_type, removal_seed)`.

    It returns ("continued", frame) when the hunk rejects the frame, and ("fell through", frame) otherwise. The hunk sits in
    a one-pass loop, so the stage's `continue` means the same here: it leaves the loop, and the return after the loop
    reports it.
    """
    module = ast.parse(
        "def run(removed_obj_img_raw, out_dir, current_iteration, obj_removal_prompt_type, removal_seed):\n"
        "    for _ in [0]:\n"
        "        return ('fell through', removed_obj_img_raw)\n"
        "    return ('continued', removed_obj_img_raw)\n"
    )
    loop = module.body[0].body[0]
    loop.body.insert(0, _hunk())
    ast.fix_missing_locations(module)
    namespace = {"logger": logger, "Image": Image, "np": np}
    exec(compile(module, STAGE.name, "exec"), namespace)
    return namespace["run"]


def _frame(transparent, alpha=0):
    """A 4-channel frame of one colour with `transparent` pixels at `alpha` and the rest opaque."""
    frame = np.zeros((*SHAPE, 4), dtype=np.uint8)
    frame[..., :3] = (200, 120, 40)
    frame[..., 3] = 255
    frame.reshape(-1, 4)[:transparent, 3] = alpha
    return frame


def _call(run, frame, tmp_path, iteration=3):
    (tmp_path / "pre_object_removal").mkdir(exist_ok=True)
    return run(frame, str(tmp_path), iteration, "object", 7)


def test_the_stage_has_exactly_one_alpha_hunk_and_its_thresholds_are_1_percent_and_128():
    hunk = _hunk()
    nested = {ast.unparse(node.test) for node in ast.walk(hunk) if isinstance(node, ast.If) and node is not hunk}
    assert "(removed_obj_img_raw[..., 3] < 128).mean() > 0.01" in nested


def test_more_than_one_percent_below_alpha_128_is_rejected_and_saved_with_a_warning(tmp_path, caplog):
    run = _run()
    frame = _frame(404)  # 1.01 %
    with caplog.at_level(logging.WARNING, logger=logger.name):
        status, _ = _call(run, frame, tmp_path)
    assert status == "continued"
    saved = tmp_path / "pre_object_removal" / "iter_3_rejected_object_seed7.png"
    assert saved.exists()
    assert np.array_equal(np.asarray(Image.open(saved)), frame)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and "transparent" in warnings[0].getMessage()


def test_exactly_one_percent_below_alpha_128_is_not_rejected_and_drops_alpha(tmp_path):
    run = _run()
    frame = _frame(400)  # exactly 1.00 %: the rule is "more than" 1 %
    status, out = _call(run, frame, tmp_path)
    assert status == "fell through"
    assert out.ndim == 3 and out.shape[-1] == 3
    assert out.flags["C_CONTIGUOUS"]
    assert np.array_equal(out, frame[..., :3])
    assert not list((tmp_path / "pre_object_removal").iterdir())


def test_a_fully_opaque_frame_is_cut_to_three_channels(tmp_path):
    run = _run()
    frame = _frame(0)
    status, out = _call(run, frame, tmp_path)
    assert status == "fell through"
    assert out.shape == (*SHAPE, 3)
    assert out.flags["C_CONTIGUOUS"]
    assert np.array_equal(out, frame[..., :3])


def test_a_three_channel_frame_is_left_alone(tmp_path):
    run = _run()
    frame = np.full((*SHAPE, 3), 90, dtype=np.uint8)
    status, out = _call(run, frame, tmp_path)
    assert status == "fell through"
    assert out is frame
    assert np.array_equal(out, np.full((*SHAPE, 3), 90, dtype=np.uint8))
    assert not list((tmp_path / "pre_object_removal").iterdir())


def test_a_greyscale_frame_is_left_alone(tmp_path):
    run = _run()
    frame = np.full(SHAPE, 90, dtype=np.uint8)
    status, out = _call(run, frame, tmp_path)
    assert status == "fell through"
    assert out is frame
    assert not list((tmp_path / "pre_object_removal").iterdir())
