# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests of the stage-5 crop_source decision rules (spec SPEC_crop_source_ko.v2 sections 2, 4, 6, 8-1).

The byte-equality test of ``crop_like_5a`` reads the stage-5 code from its syntax tree: the inline formula of stage 5a
and the ``crop_image_with_mask`` it calls. Nothing is copied from the stage by hand, the same way
``test_stage5_alpha_hunk.py`` works.
"""

import ast
import copy
import dataclasses
import pathlib
import re
from typing import Optional, Tuple

import cv2
import numpy as np
import pytest

from simfoundry.pipeline import crop_source as cs

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGE = ROOT / "scripts/pipeline/A_reconstruction/stages/5_decompose_scene.py"
PROCESSING = ROOT / "simfoundry/utils/processing_utils.py"

H, W = 40, 60


def _block(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _pool_with(*masks, prompt="board", scores=None):
    pool = cs.OriginalPool()
    scores = scores or [0.9] * len(masks)
    pool.add_prompt(prompt, scores, list(masks))
    return pool


PARAMS = cs.CropParams()


# --- 8-1: the matching rules, one case each ---

def test_the_same_mask_is_paired_with_iou_one():
    m = _block(5, 15, 5, 25)
    pool = _pool_with(m)
    d = cs.match_iteration_mask(m, pool, PARAMS, "board")
    assert d.kind == "matched" and d.index == 0
    assert d.iou == pytest.approx(1.0) and d.containment == pytest.approx(1.0) and d.coverage == pytest.approx(1.0)


def test_a_mask_moved_by_a_third_of_the_frame_is_not_a_pair():
    m = _block(5, 15, 5, 25)
    pool = _pool_with(m)
    moved = np.roll(m, W // 3, axis=1)
    d = cs.match_iteration_mask(moved, pool, PARAMS, "board")
    # Not a pair. It overlaps no original, so under the rule of v3 notes item 5 it is no_original (cut from the iteration image).
    assert d.kind == "no_original" and d.index is None and d.best_original_iou == 0.0


def test_a_mask_inside_an_earlier_pair_is_revealed_later():
    big = _block(0, 30, 0, 50)
    other = _block(32, 38, 0, 10)
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.9, 0.9], [big, other])
    first = cs.match_iteration_mask(big, pool, PARAMS, "board")
    assert first.kind == "matched" and first.index == 0
    pool.claim(first.index, 0)
    hidden = _block(5, 15, 5, 25)                       # wholly inside the claimed big original
    d = cs.match_iteration_mask(hidden, pool, PARAMS, "board")
    assert d.kind == "revealed_later" and d.index is None
    assert d.inside_claimed_fraction == pytest.approx(1.0)


def test_a_category_with_no_original_is_no_original():
    m = _block(5, 15, 5, 25)
    pool = _pool_with(m, prompt="board")
    d = cs.match_iteration_mask(_block(20, 30, 30, 40), pool, PARAMS, "black knife handle")
    assert d.kind == "no_original"


def test_a_category_whose_originals_overlap_nothing_is_no_original():
    # The category has instances on the original frame, but this mask overlaps none of them (largest IoU below 0.1).
    pool = _pool_with(_block(0, 5, 0, 5), prompt="board")
    d = cs.match_iteration_mask(_block(20, 30, 30, 40), pool, PARAMS, "board")
    assert d.kind == "no_original" and d.best_original_iou == 0.0


def test_the_earlier_count_zero_rule_is_still_available_for_the_replay():
    pool = _pool_with(_block(0, 5, 0, 5), prompt="board")
    assert cs._is_no_original(pool, "board", 0.0, rule="count_zero") is False
    assert cs._is_no_original(pool, "board", 0.0, rule="count_zero_or_no_overlap") is True
    assert cs._is_no_original(pool, "knife", 0.6, rule="count_zero_or_no_overlap") is True
    with pytest.raises(ValueError):
        cs._is_no_original(pool, "board", 0.0, rule="other")


def test_a_thin_band_at_iou_044_is_a_trace_and_its_original_survives_as_dropped_only():
    original = _block(0, 4, 0, 18)                      # 72 px
    band = _block(0, 4, 7, 25)                          # 72 px, sharing 44 px with the original: IoU 0.44
    pool = _pool_with(original)
    d = cs.match_iteration_mask(band, pool, PARAMS, "board")
    assert d.kind == "trace" and d.iou == 0.0
    assert d.best_original_iou == pytest.approx(0.44)
    plan = cs.rescue_plan(pool, [band], PARAMS)
    outcome = plan.per_dropped[0]
    assert outcome.rescued and outcome.own_index == 0 and outcome.reason == "rescued"
    assert plan.extra == []


def test_when_two_masks_want_one_original_the_earlier_iteration_takes_it():
    original = _block(5, 15, 5, 25)
    pool = _pool_with(original)
    first = cs.match_iteration_mask(original, pool, PARAMS, "board")
    assert first.kind == "matched"
    pool.claim(first.index, 0)
    second = cs.match_iteration_mask(original.copy(), pool, PARAMS, "board")
    assert second.kind == "revealed_later" and second.index is None
    with pytest.raises(ValueError):
        pool.claim(0, 1)


# --- boundaries and refusals ---

def test_an_iou_of_exactly_half_is_a_pair():
    orig = _block(0, 6, 0, 5)                           # 30 px
    mask = _block(0, 4, 0, 5)                           # rows 0-3, 20 px shared
    mask[4:6, 5:10] = True                              # plus 10 px beside it: 30 px, IoU 20/40 = 0.5
    pool = _pool_with(orig)
    d = cs.match_iteration_mask(mask, pool, PARAMS, "board")
    assert d.iou == pytest.approx(0.5) and d.kind == "matched"


def test_a_tie_goes_to_the_lower_number():
    whole = _block(0, 10, 0, 10)
    left = _block(0, 10, 0, 5)
    right = _block(0, 10, 5, 10)
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.9, 0.9], [left, right])
    d = cs.match_iteration_mask(whole, pool, PARAMS, "board")
    assert d.kind == "matched" and d.index == 0 and d.iou == pytest.approx(0.5)


def test_an_empty_iteration_mask_is_refused():
    pool = _pool_with(_block(0, 5, 0, 5))
    with pytest.raises(ValueError):
        cs.match_iteration_mask(np.zeros((H, W), bool), pool, PARAMS, "board")


def test_a_mask_of_another_shape_is_refused():
    pool = _pool_with(_block(0, 5, 0, 5))
    with pytest.raises(ValueError):
        cs.match_iteration_mask(np.zeros((H + 1, W), bool), pool, PARAMS, "board")
    pool = _pool_with(_block(0, 5, 0, 5))
    with pytest.raises(ValueError):
        pool.add_prompt("board", [0.9], [np.zeros((4, 4), bool)])   # a shape mismatch is refused, empty or not


def test_an_empty_sam3_mask_is_skipped_and_counted_not_refused():
    # An instance above the threshold can still come back with an empty mask (the mask is made at another threshold).
    # It is not an object: not in the pool, not in n_found; the pool counts it so that the stage can log it.
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.9, 0.8], [np.zeros((H, W), bool), _block(0, 5, 0, 5)])
    assert [e.index for e in pool.entries()] == [0]
    assert pool.n_found("board") == 1
    assert pool.n_empty_skipped == 1 and pool.empty_skipped_by_prompt == {"board": 1}
    pool.add_prompt("knife", [0.9], [np.zeros((H, W), bool)])
    assert pool.n_found("knife") == 0 and pool.n_empty_skipped == 2
    assert cs.match_iteration_mask(_block(20, 25, 30, 35), pool, PARAMS, "knife").kind == "no_original"


def test_a_mask_of_the_wrong_shape_is_still_refused_even_when_empty():
    pool = _pool_with(_block(0, 5, 0, 5))
    with pytest.raises(ValueError):
        pool.add_prompt("board", [0.9], [np.zeros((H + 1, W), bool)])


def test_settings_default_and_bad_values_are_refused():
    assert cs.CropParams.from_cfg({}) == cs.CropParams()
    good = cs.CropParams.from_cfg({"rescue_scope": "all", "matched_mask": "original_plus_hidden", "original_match_iou": 0.6})
    assert good.rescue_scope == "all" and good.matched_mask == "original_plus_hidden" and good.original_match_iou == 0.6
    bad = [{"rescue_scope": "some"}, {"matched_mask": "both"}, {"original_match_iou": 0.0}, {"original_match_iou": 1.5},
           {"revealed_inside_fraction": float("nan")}, {"rescue_min_score": True}, {"rescue_min_score": "0.5"},
           {"rescue_min_score": -0.1}]
    for cfg in bad:
        with pytest.raises(ValueError):
            cs.CropParams.from_cfg(cfg)


def test_the_pool_refuses_a_bad_dedupe_threshold():
    with pytest.raises(ValueError):
        cs.OriginalPool(dedupe_iou=0.0)


# --- 2-1: the original pool ---

def test_duplicates_are_merged_across_categories_and_the_better_name_wins():
    m = _block(5, 25, 5, 35)
    pool = cs.OriginalPool()
    pool.add_prompt("wooden cutting board", [0.9], [m])
    pool.add_prompt("cutting board", [0.8], [m])
    pool.add_prompt("board", [0.7], [m])
    entries = pool.entries()
    assert len(entries) == 1
    assert entries[0].prompt == "wooden cutting board"
    assert entries[0].aliases == ["cutting board", "board"]
    assert pool.n_found("cutting board") == 1 and pool.n_found("board") == 1


def test_a_higher_scoring_newcomer_takes_the_name_and_the_old_name_becomes_an_alias():
    m = _block(5, 25, 5, 35)
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.6], [m])
    pool.add_prompt("wooden cutting board", [0.9], [m])
    e = pool.entries()[0]
    assert e.prompt == "wooden cutting board" and e.score == pytest.approx(0.9) and e.aliases == ["board"]


def test_a_claimed_original_keeps_its_mask_name_and_score():
    m = _block(5, 25, 5, 35)
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.6], [m])
    pool.claim(0, 3)
    pool.add_prompt("wooden cutting board", [0.9], [m.copy()])
    e = pool.entries()[0]
    assert e.prompt == "board" and e.score == pytest.approx(0.6) and np.array_equal(e.mask, m)
    assert e.aliases == ["wooden cutting board"] and e.claimed_by == 3


def test_the_dedupe_threshold_is_strict():
    a = _block(0, 6, 0, 5)
    b = _block(0, 4, 0, 5)
    b[4:6, 5:10] = True                                 # IoU exactly 0.5 with a
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.9, 0.8], [a, b])
    assert len(pool.entries()) == 2


def test_n_found_counts_instances_before_merging():
    m = _block(5, 25, 5, 35)
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.9, 0.8], [m, m.copy()])
    assert len(pool.entries()) == 1 and pool.n_found("board") == 2 and pool.n_found("knife") == 0


def test_numbers_follow_the_order_of_first_appearance():
    pool = cs.OriginalPool()
    masks = [_block(0, 5, 0, 5), _block(10, 15, 10, 15), _block(20, 25, 20, 25)]
    pool.add_prompt("carrot", [0.9], [masks[1]])
    pool.add_prompt("onion", [0.9, 0.9], [masks[0], masks[2]])
    assert [e.index for e in pool.entries()] == [0, 1, 2]
    assert [e.prompt for e in pool.entries()] == ["carrot", "onion", "onion"]
    assert pool.free_indices() == [0, 1, 2]


def test_claimed_union_and_claim_order_follow_the_removal_order():
    a, b = _block(0, 5, 0, 5), _block(10, 15, 10, 15)
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.9, 0.9], [a, b])
    pool.claim(1, 0)
    pool.claim(0, 4)
    assert pool.claimed_order() == [1, 0]
    assert pool.free_indices() == []
    assert np.array_equal(pool.claimed_union(), a | b)
    assert [int(np.argmax(m)) for m in pool.claimed_masks()] == [int(np.argmax(b)), int(np.argmax(a))]
    assert cs.OriginalPool().claimed_union((H, W)).sum() == 0


# --- 2-2 step 3: the composite cutout ---

def _images():
    rng = np.random.default_rng(0)
    current = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    source = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    return current, source


def test_the_three_modes_have_the_areas_the_spec_gives():
    original = _block(5, 15, 5, 25)
    iteration = _block(8, 20, 10, 35)                    # extends beyond the original
    claimed = _block(12, 30, 30, 50)                     # an earlier claimed object, overlapping the iteration mask
    current, source = _images()
    _, union, _ = cs.composite_cutout(current, source, original, iteration, claimed, "union")
    _, plus_hidden, _ = cs.composite_cutout(current, source, original, iteration, claimed, "original_plus_hidden")
    _, only_original, share0 = cs.composite_cutout(current, source, original, iteration, claimed, "original")
    assert union.sum() >= iteration.sum()
    assert np.array_equal(plus_hidden, original | (iteration & claimed))
    assert np.array_equal(only_original, original) and share0 == 0.0


def test_the_final_mask_contains_the_original_mask_in_the_two_composite_modes():
    original = _block(5, 15, 5, 25)
    iteration = _block(0, 3, 0, 3)                       # disjoint from the original
    current, source = _images()
    for mode in ("union", "original_plus_hidden"):
        _, mask, _ = cs.composite_cutout(current, source, original, iteration, None, mode)
        assert np.all(mask[original])


def test_pixels_inside_the_original_come_from_the_source_and_the_rest_from_the_current_image():
    original = _block(5, 15, 5, 25)
    iteration = _block(12, 20, 20, 35)
    current, source = _images()
    image, mask, share = cs.composite_cutout(current, source, original, iteration, None, "union")
    assert np.array_equal(image[original], source[original])
    outside = mask & ~original
    assert np.array_equal(image[outside], current[outside])
    assert share == pytest.approx(outside.sum() / mask.sum())


def test_the_edited_share_is_zero_for_an_original_only_cutout_and_refusals_hold():
    original = _block(5, 15, 5, 25)
    current, source = _images()
    _, _, share = cs.composite_cutout(current, source, original, original.copy(), None, "original")
    assert share == 0.0
    with pytest.raises(ValueError):
        cs.composite_cutout(current, source, original, original, None, "both")
    with pytest.raises(ValueError):
        cs.composite_cutout(current, source[:-1], original, original, None, "union")
    with pytest.raises(ValueError):
        cs.composite_cutout(current, source, np.zeros((H, W), bool), np.zeros((H, W), bool), None, "union")


# --- 2-3: rescue ---

def _dropped_setup(scores):
    own = _block(0, 4, 0, 18)
    band = _block(0, 4, 7, 25)                            # IoU 0.44 with own
    pool = cs.OriginalPool()
    pool.add_prompt("board", scores, [own])
    return pool, own, band


def test_dropped_only_rescues_the_own_original_and_nothing_else():
    pool, own, band = _dropped_setup([0.9])
    other = _block(30, 38, 40, 58)
    pool.add_prompt("knife", [0.9], [other])
    plan = cs.rescue_plan(pool, [band], cs.CropParams(rescue_scope="dropped_only"))
    assert plan.per_dropped[0].rescued and plan.extra == []


def test_all_scope_adds_the_other_free_originals():
    pool, own, band = _dropped_setup([0.9])
    other = _block(30, 38, 40, 58)
    pool.add_prompt("knife", [0.9], [other])
    plan = cs.rescue_plan(pool, [band], cs.CropParams(rescue_scope="all"))
    assert plan.per_dropped[0].rescued and plan.extra == [1]


def test_none_scope_rescues_nothing_but_reports_the_own_original():
    pool, own, band = _dropped_setup([0.9])
    plan = cs.rescue_plan(pool, [band], cs.CropParams(rescue_scope="none"))
    assert not plan.per_dropped[0].rescued and plan.per_dropped[0].reason == "scope_none"
    assert plan.per_dropped[0].own_index == 0 and plan.extra == []


def test_a_claimed_own_original_is_lost_with_its_reason():
    pool, own, band = _dropped_setup([0.9])
    pool.claim(0, 0)
    plan = cs.rescue_plan(pool, [band], PARAMS)
    assert not plan.per_dropped[0].rescued and plan.per_dropped[0].reason == "self_original_taken"


def test_a_mask_without_any_overlap_has_no_self_original():
    pool, own, band = _dropped_setup([0.9])
    far = _block(30, 38, 40, 58)
    plan = cs.rescue_plan(pool, [far], PARAMS)
    assert plan.per_dropped[0].own_index is None and plan.per_dropped[0].reason == "no_self_original"


def test_a_free_own_original_below_the_score_is_not_rescued():
    pool, own, band = _dropped_setup([0.3])
    plan = cs.rescue_plan(pool, [band], PARAMS)
    assert plan.per_dropped[0].reason == "self_original_below_score" and not plan.per_dropped[0].rescued


def test_two_dropped_masks_with_one_free_own_original_rescue_it_once():
    pool, own, band = _dropped_setup([0.9])
    plan = cs.rescue_plan(pool, [band, band.copy()], PARAMS)
    assert [o.rescued for o in plan.per_dropped] == [True, False]
    assert plan.per_dropped[1].reason == "self_original_taken"


def test_the_own_original_is_the_largest_iou_of_any_number_even_when_claimed():
    # The largest-IoU original is claimed; a smaller free overlap must not be rescued instead (research 2's definition).
    big = _block(0, 4, 0, 18)
    small = _block(0, 2, 20, 30)                          # overlaps the band a little, less than big (IoU 10/82)
    band = _block(0, 4, 7, 25)
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.9, 0.9], [big, small])
    pool.claim(0, 0)
    plan = cs.rescue_plan(pool, [band], PARAMS)
    assert plan.per_dropped[0].own_index == 0 and plan.per_dropped[0].reason == "self_original_taken"


def test_a_tie_for_the_own_original_goes_to_the_lower_number():
    whole = _block(0, 10, 0, 10)
    left, right = _block(0, 10, 0, 5), _block(0, 10, 5, 10)
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.9, 0.9], [left, right])
    plan = cs.rescue_plan(pool, [whole], PARAMS)
    assert plan.per_dropped[0].own_index == 0 and plan.per_dropped[0].own_iou == pytest.approx(0.5)


# --- 4: occlusion record ---

def test_occluded_by_uses_a_five_by_five_square():
    original = _block(10, 12, 10, 12)                    # 2x2 block
    touching = _block(10, 12, 13, 15)                    # reached by the 2 px grow (column 13 is 2 px from column 11)
    apart = _block(10, 12, 14, 16)                       # column 14 is 3 px away: not reached
    assert cs.occluded_by(original, [touching, apart], [7, 9]) == [7]


def test_occluded_by_returns_the_numbers_given_not_positions():
    original = _block(10, 12, 10, 12)
    assert cs.occluded_by(original, [_block(10, 12, 13, 15)], [41]) == [41]
    with pytest.raises(ValueError):
        cs.occluded_by(original, [original], [1, 2])


# --- 6: the records ---

def test_object_record_of_a_matched_object_has_the_original_and_the_share():
    decision = cs.Decision(kind="matched", index=2, iou=0.8, containment=0.9, coverage=0.85)
    inst = cs.original_instance_record(cs.OriginalEntry(index=2, prompt="board", score=0.9, mask=_block(0, 2, 0, 2)), decision)
    rec = cs.object_record(decision, matched_mask="union", edited_pixel_share=0.12, original_instance=inst)
    assert rec["crop_source"] == "original" and rec["crop_reason"] == "matched" and rec["matched_mask_mode"] == "union"
    assert rec["edited_pixel_share"] == pytest.approx(0.12) and rec["original_instance"]["iou_with_iteration_mask"] == 0.8
    with pytest.raises(ValueError):
        cs.object_record(decision, matched_mask="union", edited_pixel_share=0.1)


def test_no_original_and_revealed_objects_are_edited_with_share_one():
    no_orig = cs.Decision(kind="no_original", best_original_iou=0.1, best_original_index=None)
    rec = cs.object_record(no_orig, matched_mask="union", edited_pixel_share=0.3)
    assert rec["crop_source"] == "edited" and rec["crop_reason"] == "no_original" and rec["edited_pixel_share"] == 1.0
    revealed = cs.Decision(kind="revealed_later", inside_claimed_fraction=0.7)
    rec = cs.object_record(revealed, matched_mask="union", edited_pixel_share=0.3)
    assert rec["crop_reason"] == "revealed_later" and rec["inside_claimed_fraction"] == pytest.approx(0.7)
    with pytest.raises(ValueError):
        cs.object_record(cs.Decision(kind="trace"), matched_mask="union", edited_pixel_share=0.0)


def test_a_rescued_object_has_base_iter_none_and_no_edited_pixels():
    inst = {"index": 4, "category_prompt": "board", "aliases": [], "sam3_score": 0.9}
    rec = cs.rescued_object_record(inst, occluded_by_numbers=[1])
    assert rec["base_iter"] is None and rec["crop_reason"] == "rescued" and rec["crop_source"] == "original"
    assert rec["edited_pixel_share"] == 0.0 and rec["occluded_by"] == [1]


def test_a_trace_record_is_filled_with_the_rescue_outcome():
    decision = cs.Decision(kind="trace", best_original_iou=0.44, best_original_index=0)
    record = cs.trace_record(decision, mask_area=72)
    assert record["reason"] == "trace_no_original_match" and record["mask_area"] == 72
    lost = cs.set_rescue_outcome(record, None)
    assert lost["rescued_as"] is None and lost["lost_object"] is True and record["lost_object"] is None
    kept = cs.set_rescue_outcome(record, 5)
    assert kept["rescued_as"] == 5 and kept["lost_object"] is False
    with pytest.raises(ValueError):
        cs.trace_record(cs.Decision(kind="matched", index=0), mask_area=1)


def test_originals_record_says_what_happened_to_each_original():
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.9, 0.8, 0.7], [_block(0, 5, 0, 5), _block(10, 15, 10, 15), _block(20, 25, 20, 25)])
    pool.claim(0, 3)
    rec = cs.originals_record(pool, rescued_as={1: 9})
    reasons = [item["reason"] for item in rec["originals"]]
    assert reasons == ["taken by iteration 3", "rescued as object 9", "left free"]
    assert rec["originals"][1]["bbox_yx_min_max"] == [10, 10, 14, 14] and rec["originals"][0]["area"] == 25


def test_packed_originals_unpack_to_the_same_masks():
    masks = [_block(0, 5, 0, 7), _block(10, 13, 20, 33)]
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.9, 0.9], masks)
    arrays = cs.pack_originals(pool)
    shape = tuple(int(v) for v in arrays["shape"])
    assert shape == (H, W)
    for entry in pool.entries():
        assert np.array_equal(cs.unpack_original(arrays[f"o{entry.index}"], shape), entry.mask)


def test_the_original_instance_record_carries_the_pairing_numbers_only_for_a_match():
    entry = cs.OriginalEntry(index=1, prompt="knife", score=0.7, mask=_block(0, 2, 0, 2), aliases=["blade"])
    rec = cs.original_instance_record(entry, None)
    assert rec["iou_with_iteration_mask"] is None and rec["aliases"] == ["blade"] and rec["sam3_score"] == pytest.approx(0.7)
    matched = cs.original_instance_record(entry, cs.Decision(kind="matched", index=1, iou=0.6, containment=0.7, coverage=0.8))
    assert matched["iou_with_iteration_mask"] == pytest.approx(0.6) and matched["coverage"] == pytest.approx(0.8)


def test_originals_record_keeps_the_primary_name_and_the_raw_object_categories():
    pool = cs.OriginalPool()
    pool.add_prompt("cutting board", [0.9], [_block(0, 5, 0, 5)])
    rec = cs.originals_record(pool, raw_obj_cats={"cutting board": ["cutting board / wooden board"]})
    assert rec["originals"][0]["category_prompt"] == "cutting board"
    assert rec["originals"][0]["raw_obj_cats"] == ["cutting board / wooden board"]


def test_the_module_says_original_mode_does_not_resume():
    assert "original 모드는 이어 돌리기를 지원하지 않는다" in cs.__doc__


# --- 5a: crop_like_5a is the stage's inline formula, byte for byte ---

def _stage_5a_statements():
    """The statements of stage 5a, from the stage's syntax tree: from the `np.where(removal_mask)` line through the loop
    that puts the alpha channel on the crops, which is the last statement of 5a. The saves that follow it are not part
    of 5a. The end is the loop itself, not the `if` after it, because the stage's code after 5a changes (stage 5 skips
    trace iterations there) while 5a does not.
    """
    tree = ast.parse(STAGE.read_text())
    margin_formula = "int(np.ceil(max(max_h - min_h, max_w - min_w) * 0.5))"
    found = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and ast.unparse(n.value) == margin_formula]
    assert len(found) == 1, f"expected one 5a margin in {STAGE.name}, found {len(found)}"
    body = None
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and any(s is found[0] for s in stmts):
                body = stmts
    assert body is not None
    start = [i for i, s in enumerate(body) if isinstance(s, ast.Assign) and "mask_h_idxs" in ast.unparse(s.targets[0])]
    stop = [i for i, s in enumerate(body) if isinstance(s, ast.For) and ast.unparse(s.iter) == "{'', '_focus'}"]
    assert len(start) == 1 and len(stop) == 1 and start[0] < stop[0]
    stmts = body[start[0]:stop[0] + 1]                  # the loop itself is the last statement of 5a
    text = "\n".join(ast.unparse(s) for s in stmts)
    assert "crop_image_with_mask(" in text and "for suffix in {'', '_focus'}:" in text   # ast.unparse quotes with single quotes
    return stmts


def _crop_image_with_mask_from_utils():
    """`crop_image_with_mask` of processing_utils.py, taken from its syntax tree (that module imports torch)."""
    tree = ast.parse(PROCESSING.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "crop_image_with_mask"]
    assert len(fns) == 1
    ns = {"np": np, "Optional": Optional, "Tuple": Tuple}
    exec(compile(ast.Module(body=fns, type_ignores=[]), PROCESSING.name, "exec"), ns)
    return ns["crop_image_with_mask"]


def _stage_5a_outputs(removal_mask, current_rgb, target_ratio, resolution):
    module = ast.Module(body=_stage_5a_statements(), type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "cv2": cv2, "deepcopy": copy.deepcopy, "removal_mask": removal_mask,
                 "current_rgb": current_rgb, "target_ratio": target_ratio, "resolution": resolution,
                 "crop_image_with_mask": _crop_image_with_mask_from_utils()}
    exec(compile(module, STAGE.name, "exec"), namespace)
    return namespace["masked_images"]


def _random_mask(rng, h, w):
    kind = rng.integers(3)
    m = np.zeros((h, w), bool)
    if kind == 0:
        y0, x0 = rng.integers(0, h - 1), rng.integers(0, w - 1)
        y1, x1 = rng.integers(y0 + 1, h + 1), rng.integers(x0 + 1, w + 1)
        m[y0:y1, x0:x1] = True
    elif kind == 1:
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        ry, rx = rng.uniform(1, h / 2), rng.uniform(1, w / 2)
        yy, xx = np.mgrid[0:h, 0:w]
        m = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    else:
        for _ in range(int(rng.integers(2, 5))):
            y0, x0 = rng.integers(0, h - 1), rng.integers(0, w - 1)
            m[y0:y0 + int(rng.integers(1, h)), x0:x0 + int(rng.integers(1, w))] = True
    if not m.any():
        m[h // 2, w // 2] = True
    return m


@pytest.mark.parametrize("margin_fraction, stage_key", [(0.5, "no_background"), (0.1, "no_background_focus")])
def test_crop_like_5a_is_byte_equal_to_the_stage_inline_formula(margin_fraction, stage_key):
    rng = np.random.default_rng(12345)
    ratios = [1.0, 16 / 9, 4 / 3, 1408 / 736, 0.5, 2.0]
    for trial in range(120):
        h, w = int(rng.integers(40, 120)), int(rng.integers(60, 200))
        mask = _random_mask(rng, h, w)
        image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        ratio = float(ratios[trial % len(ratios)]) if trial % 3 else float(rng.uniform(0.5, 2.5))
        resolution = (int(rng.integers(32, 129)), int(rng.integers(32, 129)))
        staged = _stage_5a_outputs(mask, image, ratio, resolution)[stage_key]
        ours = cs.crop_like_5a(image, mask, ratio, resolution, margin_fraction=margin_fraction)
        assert ours.dtype == np.uint8 and ours.shape == staged.shape, f"trial {trial}"
        assert ours.tobytes() == staged.tobytes(), f"trial {trial}: {ours.shape} {ratio} {resolution}"


def test_crop_like_5a_is_rgba_white_outside_and_alpha_is_the_mask():
    mask = _block(5, 20, 8, 30)
    image = np.full((H, W, 3), 60, np.uint8)
    out = cs.crop_like_5a(image, mask, 1.0, (48, 48))
    assert out.shape == (48, 48, 4) and out.dtype == np.uint8
    alpha = out[..., 3]
    assert set(np.unique(alpha)) <= {0, 255} or alpha.max() == 255
    assert np.all(out[..., :3][alpha == 0] == 255)
    assert np.all(out[..., :3][alpha == 255] == 60)


def test_crop_like_5a_refuses_an_empty_mask_and_a_bad_resolution():
    with pytest.raises(ValueError):
        cs.crop_like_5a(np.zeros((H, W, 3), np.uint8), np.zeros((H, W), bool), 1.0, (8, 8))
    mask = _block(5, 20, 8, 30)
    for bad in [(0, 8), (8, 8.5), (8,), "8x8"]:
        with pytest.raises(ValueError):
            cs.crop_like_5a(np.zeros((H, W, 3), np.uint8), mask, 1.0, bad)


def test_the_dataclasses_are_frozen_where_the_spec_needs_it():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PARAMS.rescue_scope = "all"


# --- boundaries the mutation check relies on ---

def test_exactly_half_inside_a_claimed_original_is_not_revealed_later():
    big = _block(0, 2, 0, 10)                            # 20 px, claimed below
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.9], [big])
    pool.claim(0, 0)
    half = _block(0, 2, 5, 15)                           # 20 px, of which 10 px lie inside big: exactly 0.5
    d = cs.match_iteration_mask(half, pool, PARAMS, "board")
    assert d.inside_claimed_fraction == pytest.approx(0.5)
    assert d.kind == "trace"


def test_an_overlap_of_exactly_point_one_is_not_no_original_and_below_it_is():
    original = _block(0, 2, 0, 10)                       # 20 px
    pool = _pool_with(original)
    two_px_inside = _block(0, 1, 0, 2)                   # 2 px inside the original: IoU exactly 0.1
    assert cs.match_iteration_mask(two_px_inside, pool, PARAMS, "board").kind == "trace"
    one_px_outside = _block(0, 1, 9, 11)                 # 1 px shared, 2 px in all: IoU 1/21, below 0.1
    assert cs.match_iteration_mask(one_px_outside, pool, PARAMS, "board").kind == "no_original"


def test_a_self_original_with_iou_exactly_point_one_is_rescued():
    original = _block(0, 2, 0, 10)
    pool = _pool_with(original)
    two_px_inside = _block(0, 1, 0, 2)
    plan = cs.rescue_plan(pool, [two_px_inside], PARAMS)
    assert plan.per_dropped[0].own_index == 0 and plan.per_dropped[0].rescued


def test_no_original_is_decided_before_revealed_later():
    # The category has no instance on the original frame and the mask lies inside a claimed original: the first rule wins.
    big = _block(0, 2, 0, 10)
    pool = cs.OriginalPool()
    pool.add_prompt("board", [0.9], [big])
    pool.claim(0, 0)
    inside = _block(0, 1, 2, 6)
    d = cs.match_iteration_mask(inside, pool, PARAMS, "black knife handle")
    assert d.inside_claimed_fraction == pytest.approx(1.0)
    assert d.kind == "no_original"


# --- matched_pixel_source: the picture the matched cutout takes its pixels from (the start-picture experiment) ---

def _refuses(call, *fragments):
    """``call`` raises ValueError with every fragment in its message. One ``match=`` per fragment, so a message that blurs
    any one of them fails the test."""
    for fragment in fragments:
        with pytest.raises(ValueError, match=re.escape(fragment)):
            call()


def _without(record, key):
    return {k: v for k, v in record.items() if k != key}


def _start_picture(seed=1):
    """A picture that is not the frame: another random image of the frame's shape."""
    return np.random.default_rng(seed).integers(0, 256, size=(H, W, 3), dtype=np.uint8)


def test_matched_pixel_source_defaults_to_frame_and_both_values_are_read_from_the_config():
    assert cs.CropParams().matched_pixel_source == "frame"
    assert cs.CropParams.from_cfg({}).matched_pixel_source == "frame"
    assert cs.CropParams.from_cfg({"matched_pixel_source": "frame"}).matched_pixel_source == "frame"
    assert cs.CropParams.from_cfg({"matched_pixel_source": "start_picture"}).matched_pixel_source == "start_picture"


@pytest.mark.parametrize("value", ["Frame", "", None, 1])
def test_matched_pixel_source_refuses_any_other_value_in_the_from_cfg_wording(value):
    with pytest.raises(ValueError, match=r"s5_scene\.matched_pixel_source must be one of"):
        cs.CropParams.from_cfg({"matched_pixel_source": value})


@pytest.mark.parametrize("upsampled", [True, False, None, 0, "false"])
def test_check_pixel_source_frame_never_reads_the_upsample_flag(upsampled):
    cs.check_pixel_source("frame", upsampled)


def test_check_pixel_source_start_picture_passes_only_with_the_upsampled_source_on():
    cs.check_pixel_source("start_picture", True)
    _refuses(lambda: cs.check_pixel_source("start_picture", False),
             "start_picture_needs_upsampled_source", "s5_scene.matched_pixel_source: start_picture",
             "s5_scene.use_upsampled_source_image")


@pytest.mark.parametrize("upsampled", [None, 0, 1, "false", "true"])
def test_check_pixel_source_start_picture_refuses_a_flag_that_is_not_a_bool(upsampled):
    with pytest.raises(ValueError, match="must be true or false"):
        cs.check_pixel_source("start_picture", upsampled)


def test_check_pixel_source_takes_a_numpy_bool_as_the_bool_it_is():
    cs.check_pixel_source("start_picture", np.True_)
    _refuses(lambda: cs.check_pixel_source("start_picture", np.False_), "start_picture_needs_upsampled_source")


@pytest.mark.parametrize("value", ["Frame", "", None, 1])
@pytest.mark.parametrize("upsampled", [True, False])
def test_check_pixel_source_refuses_a_wrong_value_in_the_from_cfg_wording(value, upsampled):
    with pytest.raises(ValueError, match=r"s5_scene\.matched_pixel_source must be one of"):
        cs.check_pixel_source(value, upsampled)


def test_check_start_picture_refuses_a_start_picture_equal_to_the_frame():
    frame = _images()[1]
    _refuses(lambda: cs.check_start_picture(frame, frame.copy()),
             "start_picture_equals_frame", "the start picture equals the frame")
    # The same values as a float array are still the frame: the check is on values, not on the dtype.
    _refuses(lambda: cs.check_start_picture(frame, frame.astype(np.float32)), "the start picture equals the frame")


def test_check_start_picture_passes_when_one_channel_of_one_pixel_differs():
    frame = _images()[1]
    start = frame.copy()
    start[7, 9, 1] ^= 1
    cs.check_start_picture(frame, start)


def test_check_start_picture_refuses_another_shape_with_a_shape_message_that_is_not_the_equality_one():
    frame = _images()[1]
    with pytest.raises(ValueError, match="has shape") as info:
        cs.check_start_picture(frame, frame[:-1])
    assert "equals the frame" not in str(info.value)


def test_cutout_pixels_gives_the_frame_or_the_start_picture_itself_without_a_copy():
    frame, start = _images()[1], _start_picture()
    assert cs.cutout_pixels("frame", frame, start) is frame
    assert cs.cutout_pixels("start_picture", frame, start) is start


def test_cutout_pixels_refuses_a_wrong_value_and_a_start_picture_of_another_shape():
    frame, start = _images()[1], _start_picture()
    with pytest.raises(ValueError, match=r"s5_scene\.matched_pixel_source must be one of"):
        cs.cutout_pixels("Frame", frame, start)
    with pytest.raises(ValueError, match="has shape"):
        cs.cutout_pixels("start_picture", frame, start[:-1])
    with pytest.raises(ValueError, match="has shape"):
        cs.cutout_pixels("start_picture", frame, None)


@pytest.mark.parametrize(
    "start", [None, np.zeros((H - 1, W, 3), np.uint8), np.zeros((H, W, 4), np.uint8), [[0, 1], [2]]],
    ids=["none", "other_shape", "other_channels", "ragged"])
def test_cutout_pixels_frame_returns_the_frame_and_never_looks_at_the_start_picture(start):
    frame = _images()[1]
    assert cs.cutout_pixels("frame", frame, start) is frame


def test_the_frame_source_gives_the_909b823_pixels_on_a_hand_worked_grid():
    # Worked by hand on a 2x3 grid. The original is the two left pixels of row 0; the iteration mask is the two right pixels
    # of row 0 (the middle one shared). The union is all of row 0, and its column-2 pixel lies outside the original, so the
    # edited share is 1/3. Inside the original the frame's 200 shows; everywhere else the current picture's 10.
    current = np.full((2, 3, 3), 10, np.uint8)
    frame = np.full((2, 3, 3), 200, np.uint8)
    original = np.zeros((2, 3), bool)
    original[0, 0:2] = True
    iteration = np.zeros((2, 3), bool)
    iteration[0, 1:3] = True
    image, mask, share = cs.composite_cutout(current, frame, original, iteration, None, "union")
    row0 = np.zeros((2, 3), bool)
    row0[0, :] = True
    assert np.array_equal(mask, row0) and share == pytest.approx(1 / 3)
    assert image[0, 0].tolist() == [200, 200, 200] and image[0, 1].tolist() == [200, 200, 200]
    assert image[0, 2].tolist() == [10, 10, 10] and np.all(image[1] == 10)
    for mode in ("original_plus_hidden", "original"):
        _, only_original, share_zero = cs.composite_cutout(current, frame, original, iteration, None, mode)
        assert np.array_equal(only_original, original) and share_zero == 0.0


@pytest.mark.parametrize("mode", list(cs.MATCHED_MASK_MODES))
def test_a_start_picture_in_the_source_slot_changes_only_the_pixels_inside_the_original(mode):
    original = _block(5, 15, 5, 25)
    iteration = _block(8, 20, 10, 35)
    claimed = _block(12, 30, 30, 50)
    current, frame = _images()
    start = _start_picture()
    assert not np.array_equal(start[original], frame[original])     # otherwise the checks below would see nothing
    frame_img, frame_mask, frame_share = cs.composite_cutout(current, frame, original, iteration, claimed, mode)
    source = cs.cutout_pixels("start_picture", frame, start)
    start_img, start_mask, start_share = cs.composite_cutout(current, source, original, iteration, claimed, mode)
    assert np.array_equal(start_mask, frame_mask) and start_share == frame_share
    assert np.array_equal(start_img[original], start[original])
    outside = start_mask & ~original
    assert np.array_equal(start_img[outside], current[outside])
    assert not np.array_equal(start_img[original], frame_img[original])


def test_the_pixel_source_is_recorded_right_after_the_mask_mode_on_a_matched_object():
    decision = cs.Decision(kind="matched", index=0, iou=0.8, containment=0.9, coverage=0.85)
    for source in cs.MATCHED_PIXEL_SOURCES:
        rec = cs.object_record(decision, matched_mask="union", edited_pixel_share=0.1, original_instance={"index": 0},
                               matched_pixel_source=source)
        keys = list(rec)
        assert rec["matched_pixel_source"] == source
        assert keys.index("matched_pixel_source") == keys.index("matched_mask_mode") + 1
    default = cs.object_record(decision, matched_mask="union", edited_pixel_share=0.1, original_instance={"index": 0})
    assert default["matched_pixel_source"] == "frame"


def test_the_pixel_source_is_none_for_no_original_and_revealed_later_objects():
    no_orig = cs.Decision(kind="no_original", best_original_iou=0.1, best_original_index=None)
    revealed = cs.Decision(kind="revealed_later", inside_claimed_fraction=0.7)
    for decision in (no_orig, revealed):
        rec = cs.object_record(decision, matched_mask="union", edited_pixel_share=0.3, matched_pixel_source="start_picture")
        assert rec["matched_pixel_source"] is None
        keys = list(rec)
        assert keys.index("matched_pixel_source") == keys.index("matched_mask_mode") + 1


def test_the_rescued_record_carries_the_pixel_source_right_after_the_mask_mode():
    for source in cs.MATCHED_PIXEL_SOURCES:
        rec = cs.rescued_object_record({"index": 4}, matched_pixel_source=source)
        keys = list(rec)
        assert rec["matched_pixel_source"] == source
        assert keys.index("matched_pixel_source") == keys.index("matched_mask_mode") + 1
    assert cs.rescued_object_record({"index": 4})["matched_pixel_source"] == "frame"


def test_a_wrong_pixel_source_is_refused_by_both_records():
    decision = cs.Decision(kind="matched", index=0, iou=0.8, containment=0.9, coverage=0.85)
    with pytest.raises(ValueError, match=r"s5_scene\.matched_pixel_source must be one of"):
        cs.object_record(decision, matched_mask="union", edited_pixel_share=0.1, original_instance={"index": 0},
                         matched_pixel_source="Frame")
    with pytest.raises(ValueError, match=r"s5_scene\.matched_pixel_source must be one of"):
        cs.rescued_object_record({"index": 4}, matched_pixel_source=None)


def test_apart_from_the_new_key_the_records_are_the_909b823_records():
    inst = {"index": 2, "category_prompt": "board", "aliases": [], "sam3_score": 0.9}
    matched = cs.object_record(cs.Decision(kind="matched", index=2, iou=0.8, containment=0.9, coverage=0.85),
                               matched_mask="union", edited_pixel_share=0.12, original_instance=inst,
                               occluded_by_numbers=[1], matched_pixel_source="start_picture")
    assert _without(matched, "matched_pixel_source") == {
        "crop_source": "original", "crop_reason": "matched", "matched_mask_mode": "union", "edited_pixel_share": 0.12,
        "original_instance": inst, "occluded_by": [1]}
    no_orig = cs.object_record(cs.Decision(kind="no_original", best_original_iou=0.1, best_original_index=None),
                               matched_mask="union", edited_pixel_share=0.3)
    assert _without(no_orig, "matched_pixel_source") == {
        "crop_source": "edited", "crop_reason": "no_original", "matched_mask_mode": None, "edited_pixel_share": 1.0,
        "best_original_iou": 0.1, "best_original_index": None}
    rescued = cs.rescued_object_record(inst, occluded_by_numbers=[1], matched_pixel_source="start_picture")
    assert _without(rescued, "matched_pixel_source") == {
        "crop_source": "original", "crop_reason": "rescued", "matched_mask_mode": "original", "edited_pixel_share": 0.0,
        "base_iter": None, "original_instance": inst, "occluded_by": [1]}
