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
    point_in_removed_regions,
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


# --------------------------------------------------------------------------------------
# VLM points that a removal has made stale
# --------------------------------------------------------------------------------------


def test_a_point_inside_a_removed_region_returns_that_region():
    removal = _removed(3, _box(40, 80, 30, 110))
    assert point_in_removed_regions((70, 60), [removal], (H, W)) is removal


def test_a_point_outside_every_removed_region_is_none():
    regions = [_removed(0, _box(0, 20, 0, 20)), _removed(1, _box(40, 80, 30, 110))]
    assert point_in_removed_regions((140, 10), regions, (H, W)) is None


def test_no_removed_regions_means_no_point_is_stale():
    assert point_in_removed_regions((70, 60), [], (H, W)) is None


def test_a_region_not_on_disk_is_ignored():
    banana = _removed(1, _box(40, 80, 30, 110))
    not_on_disk = RemovedRegion(iteration=2, origin_iter=2, region=None)
    assert point_in_removed_regions((70, 60), [banana, not_on_disk], (H, W)) is banana
    assert point_in_removed_regions((70, 60), [not_on_disk], (H, W)) is None


def test_a_region_of_another_resolution_is_ignored():
    banana = _removed(1, _box(40, 80, 30, 110))
    # Without the shape check the larger region would match, and the smaller one would index out of bounds.
    too_big = RemovedRegion(iteration=2, origin_iter=2, region=np.ones((2 * H, 2 * W), dtype=bool))
    too_small = RemovedRegion(iteration=3, origin_iter=3, region=np.ones((10, 10), dtype=bool))
    assert point_in_removed_regions((70, 60), [banana, too_big], (H, W)) is banana
    assert point_in_removed_regions((70, 60), [banana, too_small], (H, W)) is banana


def test_the_latest_removal_holding_the_point_is_returned():
    first = _removed(1, _box(40, 80, 30, 110))
    second = _removed(2, _box(50, 70, 40, 100))  # inside the first, and also holds (70, 60)
    assert point_in_removed_regions((70, 60), [first, second], (H, W)) is second
    assert point_in_removed_regions((70, 60), [second, first], (H, W)) is first


def test_removals_before_the_point_was_made_do_not_make_it_stale():
    pot = _box(40, 80, 30, 110)
    erased_earlier = _removed(4, pot)
    # A point given at iteration 5 saw the image after iteration 4's removal, so that removal is not news to it.
    assert point_in_removed_regions((70, 60), [erased_earlier], (H, W), since_iter=5) is None
    # A removal in the iteration that gave the point counts: the boundary is included.
    assert point_in_removed_regions((70, 60), [erased_earlier], (H, W), since_iter=4) is erased_earlier
    erased_later = _removed(6, pot)
    assert point_in_removed_regions((70, 60), [erased_earlier, erased_later], (H, W), since_iter=5) is erased_later


def test_a_point_a_few_pixels_outside_the_mask_is_inside_the_grown_region():
    mask = _box(40, 80, 30, 110)
    removal = _removed(0, mask)
    assert not mask[60, 112]  # 3 pixels past the mask's last column, 109
    assert point_in_removed_regions((112, 60), [removal], (H, W)) is removal
    assert point_in_removed_regions((115, 60), [removal], (H, W)) is None  # 6 pixels past: beyond the growth


def test_fractional_points_round_to_the_nearest_pixel():
    region = np.zeros((H, W), dtype=bool)
    region[21, 10] = True  # row 21, column 10
    removal = RemovedRegion(iteration=0, origin_iter=0, region=region)
    assert point_in_removed_regions((10.4, 20.6), [removal], (H, W)) is removal  # rounds to (10, 21)
    assert point_in_removed_regions((10.4, 20.4), [removal], (H, W)) is None  # rounds to (10, 20)


def test_x_is_the_column_and_y_the_row():
    wide = _removed(0, _box(50, 60, 10, 150))  # 10 rows by 140 columns
    assert point_in_removed_regions((100, 55), [wide], (H, W)) is wide
    assert point_in_removed_regions((55, 100), [wide], (H, W)) is None  # the same numbers, transposed


def test_points_outside_the_frame_are_none_and_do_not_wrap_around():
    # A region covering the whole frame: a negative index would wrap to the far edge and match.
    everything = RemovedRegion(iteration=0, origin_iter=0, region=np.ones((H, W), dtype=bool))
    for point in [(-1, 5), (5, -1), (W, 5), (5, H), (-1, -1)]:
        assert point_in_removed_regions(point, [everything], (H, W)) is None, point
    assert point_in_removed_regions((W - 1, H - 1), [everything], (H, W)) is everything


def test_the_cooktop_case_is_stale_only_inside_removals_from_the_point_on():
    # Iteration 2 removed the pot. Iteration 6 removed the ghost of the pot: a box inside its region.
    pot_region = _removed(2, _box(30, 90, 40, 120))
    ghost_region = _removed(6, _box(45, 75, 50, 110), origin_iter=2)
    regions = [pot_region, ghost_region]

    # The VLM gave this point on the iteration-6 image, and iteration 6 removed the region under it.
    assert point_in_removed_regions((80, 60), regions, (H, W), since_iter=6) is ghost_region

    # Same pot area, but outside the ghost's region. Iteration 2 is older than the point, so not stale.
    assert point_in_removed_regions((45, 35), regions, (H, W), since_iter=6) is None
    assert point_in_removed_regions((45, 35), regions, (H, W), since_iter=2) is pot_region

    # Known limit, not a bug to patch here: a point inside no removed region is never stale. The knife's
    # point in the real run lay outside every region, so this rule does not catch it.
    assert point_in_removed_regions((10, 100), regions, (H, W), since_iter=6) is None
