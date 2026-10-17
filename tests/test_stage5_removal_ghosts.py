# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 5 must not cut out the ghost an image model paints in place of a removed object.

Regression cover for repro/README.md chapter 22.1: Qwen-Image-2.1-Turbo turned a banana that
stage 5 had filled in green into a green banana, which the next detection reported as a new
object; stage 5 cut it out twice with the same failing prompt type, and a scene of six objects
came out with eight.
"""

import json
import os

import numpy as np
import pytest
from PIL import Image

from simfoundry.pipeline.removal_ghosts import (
    GHOST_SKIP_REASON,
    RemovedRegion,
    find_ghost_origin,
    find_ghosts,
    grow_region,
    load_removed_regions,
    mask_coverage,
    prompt_types_tried,
)

PROMPT_ORDER = ["mask", "outline", "bbox"]
THRESHOLD = 0.95  # s5_scene.duplicate_mask_intersect_area_threshold
H, W = 120, 160


def _box(y0, y1, x0, x1):
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _removed(iteration, mask, origin_iter=None, tried=("mask",)):
    return RemovedRegion(
        iteration=iteration,
        origin_iter=iteration if origin_iter is None else origin_iter,
        region=grow_region(mask),
        prompt_types_tried=list(tried),
    )


def test_a_repaint_inside_the_removed_region_is_a_ghost_of_that_removal():
    banana = _box(40, 80, 30, 110)
    repaint = _box(45, 75, 35, 95)  # drawn inside the marked region, and smaller
    regions = [_removed(0, _box(0, 20, 0, 20)), _removed(1, banana)]
    assert find_ghost_origin(repaint, regions, THRESHOLD) == (1, 1.0)


def test_an_object_a_removal_revealed_is_not_a_ghost():
    # A cup the marker stood in: the marker's region now shows cup, but most of the cup
    # was visible all along.
    marker = _box(30, 90, 70, 80)
    cup = _box(50, 110, 40, 120)
    assert mask_coverage(cup, grow_region(marker)) < 0.5
    assert find_ghost_origin(cup, [_removed(0, marker)], THRESHOLD) is None


def test_an_object_touching_a_removed_one_is_not_a_ghost():
    removed = _box(40, 80, 30, 80)
    neighbour = _box(40, 80, 80, 120)  # shares an edge, lies inside the grown margin only there
    assert find_ghost_origin(neighbour, [_removed(0, removed)], THRESHOLD) is None


def test_a_restored_object_segmented_slightly_larger_is_still_a_ghost():
    # The model put the object back as it was; SAM's boundary moved by a couple of pixels.
    removed = _box(40, 60, 40, 60)
    restored = _box(38, 62, 38, 62)
    assert mask_coverage(restored, removed) < THRESHOLD
    assert find_ghost_origin(restored, [_removed(0, removed)], THRESHOLD) == (0, 1.0)


def test_the_latest_edit_of_the_pixels_names_the_ghost():
    banana = _box(40, 80, 30, 110)
    repaint = _box(45, 75, 35, 95)
    regions = [
        _removed(1, banana),
        _removed(2, _box(0, 20, 0, 20)),
        _removed(3, repaint, origin_iter=1, tried=["outline"]),  # an earlier ghost removal
        _removed(4, _box(30, 90, 20, 120), tried=["mask"]),  # a later removal over it all
    ]
    assert find_ghost_origin(repaint, regions, THRESHOLD)[0] == 4
    assert find_ghost_origin(repaint, regions[:3], THRESHOLD)[0] == 1


def test_regions_of_another_resolution_are_ignored():
    region = RemovedRegion(iteration=0, origin_iter=0, region=np.ones((10, 10), dtype=bool))
    assert find_ghost_origin(_box(0, 5, 0, 5), [region], THRESHOLD) is None


def test_each_retry_uses_the_prompt_types_the_object_has_not_had():
    banana = _box(40, 80, 30, 110)
    repaint = _box(45, 75, 35, 95)
    apple = _box(5, 25, 120, 140)
    regions = [_removed(1, banana, tried=["mask"])]

    ghosts = find_ghosts([apple, repaint], regions, PROMPT_ORDER, THRESHOLD)
    assert [(g.index, g.origin_iter, g.untried_prompt_types) for g in ghosts] == [(1, 1, ["outline", "bbox"])]

    # The retry used outline and the ghost is still there: only bbox is left.
    regions.append(_removed(3, repaint, origin_iter=1, tried=["outline"]))
    assert prompt_types_tried(regions, 1) == ["mask", "outline"]
    assert find_ghosts([repaint], regions, PROMPT_ORDER, THRESHOLD)[0].untried_prompt_types == ["bbox"]

    # bbox failed too: the ghost has nothing left to try and stage 5 drops it.
    regions.append(_removed(4, repaint, origin_iter=1, tried=["bbox"]))
    assert find_ghosts([repaint], regions, PROMPT_ORDER, THRESHOLD)[0].untried_prompt_types == []


def test_prompt_types_rejected_before_the_accepted_one_count_as_tried():
    banana = _box(40, 80, 30, 110)
    regions = [_removed(1, banana, tried=["mask", "outline"])]  # mask rejected, outline accepted
    ghost = find_ghosts([_box(45, 75, 35, 95)], regions, PROMPT_ORDER, THRESHOLD)[0]
    assert ghost.untried_prompt_types == ["bbox"]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def test_a_resumed_run_rebuilds_the_removals_from_disk(tmp_path):
    banana = _box(40, 80, 30, 110)
    repaint = _box(45, 75, 35, 95)
    # 0: an object recorded before removal_prompt_types_tried existed.
    _write_json(tmp_path / "obj_cat_list" / "iter_0.json", {"removed_obj_phrase": "pear"})
    _write_mask(tmp_path / "removal_mask" / "iter_0.png", _box(0, 20, 0, 20))
    # 1: the banana, whose removal left a ghost.
    _write_json(tmp_path / "obj_cat_list" / "iter_1.json",
                {"removal_prompt_type": "mask", "removal_prompt_types_tried": ["mask"]})
    _write_mask(tmp_path / "removal_mask" / "iter_1.png", banana)
    # 2: a detection that found nothing.
    _write_json(tmp_path / "skipped_iterations" / "iter_2.json", {"reason": "no_masks_detected"})
    # 3: the ghost, removed again with outline.
    _write_json(tmp_path / "skipped_iterations" / "iter_3.json",
                {"reason": GHOST_SKIP_REASON, "ghost_of_iter": 1, "removal_prompt_types_tried": ["outline"]})
    _write_mask(tmp_path / "removal_mask" / "iter_3.png", repaint)
    # A half-published manifest and an iteration at the resume point are not removals yet.
    _write_json(tmp_path / "obj_cat_list" / "iter_4.partial.json", {})
    _write_json(tmp_path / "obj_cat_list" / "iter_5.json", {"removal_prompt_types_tried": ["bbox"]})

    regions = load_removed_regions(str(tmp_path), before_iter=5, default_prompt_type="mask")

    assert [(r.iteration, r.origin_iter, r.prompt_types_tried) for r in regions] == [
        (0, 0, ["mask"]),
        (1, 1, ["mask"]),
        (3, 1, ["outline"]),
    ]
    assert all(r.region.shape == (H, W) for r in regions)
    ghost = find_ghosts([repaint], regions, PROMPT_ORDER, THRESHOLD)[0]
    assert (ghost.origin_iter, ghost.untried_prompt_types) == (1, ["bbox"])


def test_an_unreadable_record_is_skipped_not_fatal(tmp_path, caplog):
    _write_json(tmp_path / "obj_cat_list" / "iter_1.json", {"removal_prompt_types_tried": ["mask"]})
    (tmp_path / "obj_cat_list" / "iter_0.json").write_text("")  # e.g. a truncated write
    with caplog.at_level("WARNING"):
        regions = load_removed_regions(str(tmp_path), before_iter=5, default_prompt_type="mask")
    assert [r.iteration for r in regions] == [1]
    assert "unreadable" in caplog.text


def test_a_removal_without_a_mask_on_disk_still_counts_its_prompt_types(tmp_path):
    _write_json(tmp_path / "skipped_iterations" / "iter_3.json",
                {"reason": GHOST_SKIP_REASON, "ghost_of_iter": 1, "removal_prompt_types_tried": ["outline", "bbox"]})
    regions = load_removed_regions(str(tmp_path), before_iter=10, default_prompt_type="mask")
    assert len(regions) == 1 and regions[0].region is None
    assert prompt_types_tried(regions, 1) == ["outline", "bbox"]


# --------------------------------------------------------------------------------------
# The reconstruction the bug was found on (optional fixture)
# --------------------------------------------------------------------------------------

DATA_ROOT = os.environ.get(
    "SIMFOUNDRY_TEST_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data"),
)


def test_the_serve_fruits_ghosts_are_recognised_and_its_real_objects_are_not():
    """Data/serve_fruits/s5_scene from before this fix: 8 removals, 2 of them ghosts.

    Iteration 1 removed the yellow banana with the mask prompt; iterations 3 and 6 then cut
    out the green banana the image model had painted in its place.
    """
    s5_dir = os.path.join(DATA_ROOT, "serve_fruits", "s5_scene")
    mask_fpaths = [os.path.join(s5_dir, "removal_mask", f"iter_{i}.png") for i in range(8)]
    if not all(os.path.isfile(p) for p in mask_fpaths):
        pytest.skip(f"Optional reference fixture serve_fruits is missing under {DATA_ROOT}")
    masks = [np.array(Image.open(p).convert("L")) > 127 for p in mask_fpaths]

    ghosts_of = {}
    regions = []
    for i, mask in enumerate(masks):
        match = find_ghost_origin(mask, regions, THRESHOLD)
        if match is not None:
            ghosts_of[i] = match[0]
        regions.append(RemovedRegion(iteration=i, origin_iter=i if match is None else match[0],
                                     region=grow_region(mask), prompt_types_tried=["mask"]))

    assert ghosts_of == {3: 1, 6: 1}
