# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 6's reference check: the metrics, the flags, the file-level revert, the config and the stage wiring.

Needs no GPU and no model: the metrics and the revert are pure functions on synthetic pictures written to
tmp_path, and the stage is read with ast.
"""

import ast
import json
import pathlib

import numpy as np
import pytest
from PIL import Image

from simfoundry.pipeline import upsample_check as uc

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGE = REPO_ROOT / "scripts/pipeline/A_reconstruction/stages/6_upsample_object_images.py"
CONFIG = REPO_ROOT / "scripts/cfg/real2sim_cfg.yaml"
DEFAULT = {"action": "warn", "min_iou": 0.9, "max_hole_change": 0.05, "area_ratio_range": [0.8, 1.25]}


def square(size, top, left, frame=100):
    alpha = np.zeros((frame, frame), dtype=np.uint8)
    alpha[top:top + size, left:left + size] = 255
    return alpha


def rgba_of(alpha, colour=(200, 90, 40)):
    rgba = np.zeros(alpha.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = colour
    rgba[..., 3] = alpha
    return rgba


# --------------------------------------------------------------------------------------
# The metrics
# --------------------------------------------------------------------------------------

def test_the_same_silhouette_matches_at_any_resolution():
    ref = square(60, 20, 20)
    metrics = uc.silhouette_metrics(ref, ref)
    doubled = uc.silhouette_metrics(ref, np.kron(ref, np.ones((2, 2), dtype=np.uint8)))

    assert metrics["iou"] == 1.0 and metrics["area_ratio"] == 1.0 and metrics["hole_change"] == 0.0
    assert uc.evaluate_flags(metrics, DEFAULT) == []
    assert doubled["iou"] == 1.0 and doubled["area_ratio"] == 1.0 and doubled["hole_change"] == 0.0
    assert doubled["aspect_mismatch"] is False


def test_a_non_square_picture_matches_at_double_size():
    # 100 x 60 doubled: a swapped width and height would resize the reference to the wrong shape.
    ref = np.zeros((60, 100), dtype=np.uint8)
    ref[10:50, 20:80] = 255
    metrics = uc.silhouette_metrics(ref, np.kron(ref, np.ones((2, 2), dtype=np.uint8)))

    assert metrics["iou"] == 1.0 and metrics["area_ratio"] == 1.0


def test_a_realistic_non_square_picture_is_resized_to_the_upsampled_shape():
    # The crop is 512 x 701 and the model returned 864 x 1184 for it.
    ref = np.zeros((512, 701), dtype=np.uint8)
    ref[80:440, 150:560] = 255
    up = np.asarray(Image.fromarray(ref).resize((1184, 864), Image.NEAREST))

    metrics = uc.silhouette_metrics(ref, up)

    assert metrics["iou"] == 1.0
    assert metrics["aspect_mismatch"] is False


def test_a_five_percent_aspect_difference_is_recorded_and_one_percent_is_not():
    ref = square(60, 20, 20)
    five = np.zeros((100, 105), dtype=np.uint8)
    five[20:80, 20:80] = 255
    one = np.zeros((100, 101), dtype=np.uint8)
    one[20:80, 20:80] = 255

    assert uc.silhouette_metrics(ref, five)["aspect_mismatch"] is True
    assert uc.silhouette_metrics(ref, one)["aspect_mismatch"] is False


def test_an_alpha_of_127_is_not_the_object_and_128_is():
    below = uc.silhouette_metrics(np.full((10, 10), 127, dtype=np.uint8), np.full((10, 10), 128, dtype=np.uint8))
    above = uc.silhouette_metrics(np.full((10, 10), 128, dtype=np.uint8), np.full((10, 10), 127, dtype=np.uint8))

    assert below["ref_area_frac"] == 0.0 and below["up_area_frac"] == 1.0
    assert above["ref_area_frac"] == 1.0 and above["up_area_frac"] == 0.0


def test_each_area_fraction_is_over_its_own_pictures_frame():
    ref = square(60, 20, 20, frame=100)  # 3600 of 10000
    up = np.zeros((60, 200), dtype=np.uint8)  # a frame of 12000, so the two denominators differ
    up[10:40, 20:60] = 255  # 1200 of 12000

    metrics = uc.silhouette_metrics(ref, up)

    assert metrics["ref_area_frac"] == pytest.approx(0.36)
    assert metrics["up_area_frac"] == pytest.approx(0.10)


def test_cut_corners_give_the_iou_by_hand_and_fail_the_minimum():
    ref = square(60, 20, 20)
    up = ref.copy()
    for row, col in [(20, 20), (20, 65), (65, 20), (65, 65)]:
        up[row:row + 15, col:col + 15] = 0  # four 15 x 15 corners: 900 of the 3600 pixels

    metrics = uc.silhouette_metrics(ref, up)

    assert metrics["iou"] == pytest.approx(2700 / 3600)  # the cut silhouette is a subset of the crop
    assert "low_iou" in uc.evaluate_flags(metrics, DEFAULT)


def test_a_filled_hole_is_a_negative_hole_change():
    ref = square(60, 20, 20)
    ref[38:62, 38:62] = 0  # a 24 x 24 hole in the middle of the crop
    up = square(60, 20, 20)  # the model filled it

    metrics = uc.silhouette_metrics(ref, up)

    assert metrics["ref_hole_frac"] == pytest.approx(576 / 3600)
    assert metrics["hole_change"] == pytest.approx(-576 / 3600)
    assert "hole_changed" in uc.evaluate_flags(metrics, DEFAULT)


def test_a_new_hole_is_a_positive_hole_change():
    ref = square(60, 20, 20)
    up = square(60, 20, 20)
    up[38:62, 38:62] = 0

    metrics = uc.silhouette_metrics(ref, up)

    assert metrics["hole_change"] == pytest.approx(576 / 3600)
    assert "hole_changed" in uc.evaluate_flags(metrics, DEFAULT)


def test_a_smaller_object_fails_the_area_range_and_the_thresholds_are_read():
    ref = square(60, 20, 20)
    up = square(40, 30, 30)  # the same centre, 40 x 40: 1600 of 3600 pixels

    metrics = uc.silhouette_metrics(ref, up)
    flags = uc.evaluate_flags(metrics, DEFAULT)

    assert metrics["area_ratio"] == pytest.approx(1600 / 3600)
    assert "area_changed" in flags and "low_iou" in flags
    assert "area_changed" not in uc.evaluate_flags(metrics, {**DEFAULT, "area_ratio_range": (0.4, 1.3)})
    assert "low_iou" not in uc.evaluate_flags(metrics, {**DEFAULT, "min_iou": 0.3})


def test_a_threshold_equal_to_the_measured_value_does_not_fail():
    ref = square(60, 20, 20)
    up = ref.copy()
    up[20:35, 20:35] = 0  # a cut corner: iou 0.9722...
    metrics = uc.silhouette_metrics(ref, up)
    assert "low_iou" not in uc.evaluate_flags(metrics, {**DEFAULT, "min_iou": metrics["iou"]})

    ring = square(60, 20, 20)
    ring[38:62, 38:62] = 0
    holed = uc.silhouette_metrics(ring, square(60, 20, 20))
    assert "hole_changed" not in uc.evaluate_flags(holed, {**DEFAULT, "max_hole_change": abs(holed["hole_change"])})


def test_an_area_ratio_at_either_end_of_the_range_does_not_fail():
    ref = square(60, 20, 20)
    lower = ref.copy()
    lower[20:80, 68:80] = 0  # 2880 of 3600 pixels: a ratio of exactly 0.8
    upper = np.zeros((100, 100), dtype=np.uint8)
    upper[20:80, 20:95] = 255  # 4500 of 3600: a ratio of exactly 1.25

    assert uc.silhouette_metrics(ref, lower)["area_ratio"] == 0.8
    assert "area_changed" not in uc.evaluate_flags(uc.silhouette_metrics(ref, lower), DEFAULT)
    assert uc.silhouette_metrics(ref, upper)["area_ratio"] == 1.25
    assert "area_changed" not in uc.evaluate_flags(uc.silhouette_metrics(ref, upper), DEFAULT)


def test_a_threshold_that_is_none_skips_its_check():
    ref = square(60, 20, 20)
    metrics = uc.silhouette_metrics(ref, np.zeros_like(ref))
    skipped = {"action": "warn", "min_iou": None, "max_hole_change": None, "area_ratio_range": None}

    assert uc.evaluate_flags(metrics, skipped) == []
    # A hole filled in (hole change -0.16) and a silhouette only 0.84 alike: with every threshold None, nothing fails.
    ring = square(60, 20, 20)
    ring[38:62, 38:62] = 0
    holed = uc.silhouette_metrics(ring, square(60, 20, 20))

    assert holed["hole_change"] == pytest.approx(-576 / 3600)
    assert uc.evaluate_flags(holed, skipped) == []


def test_two_empty_silhouettes_match_and_a_one_sided_one_does_not():
    empty = np.zeros((100, 100), dtype=np.uint8)
    ref = square(60, 20, 20)

    both = uc.silhouette_metrics(empty, empty)
    lost = uc.silhouette_metrics(ref, empty)
    invented = uc.silhouette_metrics(empty, ref)

    assert both["iou"] == 1.0 and both["area_ratio"] is None
    assert uc.evaluate_flags(both, DEFAULT) == []
    assert lost["iou"] == 0.0 and lost["area_ratio"] == 0.0
    assert {"low_iou", "area_changed"} <= set(uc.evaluate_flags(lost, DEFAULT))
    assert invented["iou"] == 0.0 and invented["area_ratio"] is None
    assert uc.evaluate_flags(invented, DEFAULT) == ["low_iou"]


def test_soft_edges_are_binarised_at_the_alpha_threshold():
    ref = square(60, 20, 20)
    soft = np.zeros((100, 100), dtype=np.uint8)
    ring = np.zeros((100, 100), dtype=bool)
    ring[18:82, 18:82] = True
    soft[ring] = 60  # a soft rim below the threshold
    soft[ref > 0] = 200

    assert uc.silhouette_metrics(ref, soft, alpha_threshold=127)["iou"] == 1.0
    assert uc.silhouette_metrics(ref, soft, alpha_threshold=50)["iou"] < 1.0


def test_a_three_channel_picture_is_refused():
    with pytest.raises(ValueError):
        uc.check_pair(np.zeros((4, 4, 3), dtype=np.uint8), rgba_of(np.zeros((4, 4), dtype=np.uint8)),
                      uc.resolve_thresholds(DEFAULT))


# --------------------------------------------------------------------------------------
# The thresholds
# --------------------------------------------------------------------------------------

def test_the_config_section_has_the_provisional_defaults_and_bad_actions_are_refused():
    from omegaconf import OmegaConf

    section = OmegaConf.load(CONFIG).s6_upsample.reference_check
    assert OmegaConf.to_container(section) == {
        "enabled": True, "action": "warn", "min_iou": 0.9, "max_hole_change": 0.05, "area_ratio_range": [0.8, 1.25],
    }
    with pytest.raises(ValueError):
        uc.resolve_thresholds({"action": "block"})


def test_the_real_section_reads_its_numbers_and_the_range_as_a_tuple():
    from omegaconf import OmegaConf

    thresholds = uc.resolve_thresholds(OmegaConf.load(CONFIG).s6_upsample.reference_check)

    assert thresholds["min_iou"] == 0.9
    assert thresholds["max_hole_change"] == 0.05
    assert thresholds["action"] == "warn"
    assert thresholds["area_ratio_range"] == (0.8, 1.25)


def test_missing_keys_resolve_to_none_and_a_missing_action_is_warn():
    only_action = uc.resolve_thresholds({"action": "revert"})
    no_action = uc.resolve_thresholds({"min_iou": 0.5})

    assert only_action["min_iou"] is None and only_action["max_hole_change"] is None
    assert only_action["area_ratio_range"] is None and only_action["action"] == "revert"
    assert no_action["action"] == "warn" and no_action["min_iou"] == 0.5


# --------------------------------------------------------------------------------------
# The file-level check and revert
# --------------------------------------------------------------------------------------

def _pictures(tmp_path, transparent_alpha):
    """The upsampled RGB picture and its background-removed RGBA picture, as stage 6 writes them."""
    rng = np.random.default_rng(1)
    rgb_path = tmp_path / "up.png"
    rgba_path = tmp_path / "up_transparent.png"
    Image.fromarray(rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)).save(rgb_path)
    Image.fromarray(rgba_of(transparent_alpha, colour=(10, 200, 30))).save(rgba_path)
    return rgb_path, rgba_path


def _thresholds(action):
    return uc.resolve_thresholds({**DEFAULT, "action": action})


def test_a_warning_leaves_both_files_exactly_as_they_were(tmp_path):
    reference = rgba_of(square(60, 20, 20))
    rgb_path, rgba_path = _pictures(tmp_path, square(40, 30, 30))
    before = (rgb_path.read_bytes(), rgba_path.read_bytes())

    result = uc.run_reference_check(reference, rgb_path, rgba_path, _thresholds("warn"))

    assert result["flags"] and result["reverted"] is False
    assert (rgb_path.read_bytes(), rgba_path.read_bytes()) == before


def test_a_revert_writes_the_crop_back_in_both_files(tmp_path):
    reference = rgba_of(square(60, 20, 20))
    rgb_path, rgba_path = _pictures(tmp_path, square(40, 30, 30))

    result = uc.run_reference_check(reference, rgb_path, rgba_path, _thresholds("revert"))

    assert result["flags"] and result["reverted"] is True
    assert np.array_equal(np.asarray(Image.open(rgb_path).convert("RGB")), reference[..., :3])
    assert np.array_equal(np.asarray(Image.open(rgba_path).convert("RGBA")), reference)


def test_a_revert_without_flags_leaves_both_files_exactly_as_they_were(tmp_path):
    reference = rgba_of(square(60, 20, 20))
    rgb_path, rgba_path = _pictures(tmp_path, square(60, 20, 20))
    before = (rgb_path.read_bytes(), rgba_path.read_bytes())

    result = uc.run_reference_check(reference, rgb_path, rgba_path, _thresholds("revert"))

    assert result["flags"] == [] and result["reverted"] is False
    assert (rgb_path.read_bytes(), rgba_path.read_bytes()) == before


def test_the_record_serialises_to_json_and_carries_the_reverted_flag(tmp_path):
    rgb_path, rgba_path = _pictures(tmp_path, square(40, 30, 30))

    result = uc.run_reference_check(rgba_of(square(60, 20, 20)), rgb_path, rgba_path, _thresholds("warn"))

    assert json.loads(json.dumps(result)) == result
    assert result["reverted"] is False
    for key, value in result.items():
        assert type(value) in (float, int, bool, str, list, type(None)), (key, type(value))


def test_the_revert_writes_each_file_as_a_picture_of_its_own_kind(tmp_path):
    reference = rgba_of(square(60, 20, 20))
    rgb_path, rgba_path = _pictures(tmp_path, square(40, 30, 30))

    uc.run_reference_check(reference, rgb_path, rgba_path, _thresholds("revert"))

    assert Image.open(rgb_path).mode == "RGB"
    assert Image.open(rgba_path).mode == "RGBA"


# --------------------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------------------

def _stage_main():
    tree = ast.parse(STAGE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return tree, main


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name]


def test_the_stage_imports_the_check_at_module_level():
    tree, _ = _stage_main()
    imported = {alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
                and node.module == "simfoundry.pipeline.upsample_check" for alias in node.names}

    assert {"run_reference_check", "resolve_thresholds"} <= imported


def test_the_check_is_called_with_the_four_names_in_order_and_stored_under_the_image_name():
    _, main = _stage_main()
    calls = _calls(main, "run_reference_check")

    assert len(calls) == 1
    assert [arg.id if isinstance(arg, ast.Name) else None for arg in calls[0].args] == [
        "input_img_padded", "upsampled_img_fpath", "transparent_fpath", "reference_thresholds",
    ]
    stores = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and n.value is calls[0]]
    assert len(stores) == 1
    target = stores[0].targets[0]
    assert isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
    assert target.value.id == "reference_checks"
    assert isinstance(target.slice, ast.Name) and target.slice.id == "img_name"


def test_the_check_sits_under_the_enabled_guard_after_the_background_removal_and_before_validity():
    _, main = _stage_main()
    call = _calls(main, "run_reference_check")[0]
    removals = _calls(main, "remove_background")
    validity_ifs = [n for n in ast.walk(main) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Name) and n.test.id == "check_valid"]
    guards = [n for n in ast.walk(main) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Name) and n.test.id == "reference_check_enabled"]

    guard = [g for g in guards if any(c is call for stmt in g.body for c in ast.walk(stmt))]
    assert len(guard) == 1
    upsample_removal = max(r.lineno for r in removals)
    loop_validity = max(v.lineno for v in validity_ifs)
    assert upsample_removal < call.lineno < loop_validity


def test_the_record_is_defined_before_the_final_call_and_written_only_when_enabled():
    _, main = _stage_main()
    finals = _calls(main, "finalize_stage")
    assigns = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "reference_checks" for t in n.targets)]

    assert len(finals) == 1 and len(assigns) == 1
    assert assigns[0].lineno < finals[0].lineno
    keyword = next(k for k in finals[0].keywords if k.arg == "result")
    additional = next(k for k in _calls(keyword.value, "StageResult")[0].keywords if k.arg == "additional_info")
    assert any(isinstance(n, ast.Name) and n.id == "reference_checks" for n in ast.walk(additional.value))
    assert any(isinstance(n, ast.Name) and n.id == "reference_check_enabled" for n in ast.walk(additional.value))
