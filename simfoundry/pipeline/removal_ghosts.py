# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognise stage-5 removal ghosts: objects the image model painted back instead of erasing.

Stage 5 removes objects one at a time. It marks the chosen object in removal_outline_color,
asks the image model to erase it, and detects the next object on the edited image. A model
can render the marking instead: Qwen-Image-2.1-Turbo turned a banana filled in green into a
green banana. The next detection then reported that ghost as a new object, stage 5 cut it out
with the same prompt type, which failed the same way, and a scene of six objects came out
with eight (repro/README.md, chapter 22.1).

A ghost lies inside the region its removal marked. A real object that a removal revealed
extends past that region: it was visible before, so most of its mask lies outside the mask
of the object that hid part of it. A mask (nearly) contained in an earlier removal's region is
therefore that removal having failed. This is the one-way containment ``disentangle_masks``
tests within one detection, at the duplicate-mask threshold. It is one-way because a repaint
is drawn inside the marked region and comes out smaller: in the scene above the ghost lay
99.9% inside the banana's mask but filled only 75% of it.

Stage 5 removes a ghost again with a prompt type its object has not had yet, and drops it
once every type has been tried. It records the prompt types each iteration tried, so
:func:`load_removed_regions` rebuilds the same state when a run resumes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# skipped_iterations/iter_N.json reason for an iteration spent removing a ghost. That
# iteration edits the image like any removal but emits no object.
GHOST_SKIP_REASON = "removal_ghost"

# A removal mask is grown by this many pixels (3x3 dilation steps) before the containment
# test. That absorbs boundary jitter between two SAM segmentations of one object and covers
# the margin the mask prompt paints around it (5 dilation steps in the removal crop, so at
# most 5 px at this resolution). It stays well inside the band of roughly 35 px over which
# stage 5 blends the model's output back, so a real object next to a removed one keeps all
# but a sliver of its mask outside the region.
REGION_GROWTH_ITERATIONS = 5

_RECORD_NAME = re.compile(r"iter_(\d+)\.json")


@dataclass
class RemovedRegion:
    """What ghost matching needs to know about one stage-5 removal."""

    iteration: int
    # The object iteration this removal belongs to: the iteration itself, or, for an
    # iteration that removed a ghost, the iteration whose object left that ghost.
    origin_iter: int
    region: Optional[np.ndarray]  # grown removal mask (bool); None when not on disk
    prompt_types_tried: list = field(default_factory=list)


@dataclass
class Ghost:
    """A detected mask recognised as the leftover of an earlier removal."""

    index: int  # position among the masks passed to find_ghosts
    origin_iter: int
    overlap: float  # fraction of the mask inside the removed region
    untried_prompt_types: list


def grow_region(mask: np.ndarray) -> np.ndarray:
    """The region a removal marked: its mask grown by REGION_GROWTH_ITERATIONS pixels."""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=REGION_GROWTH_ITERATIONS).astype(bool)


def mask_coverage(mask: np.ndarray, region: np.ndarray) -> float:
    """Fraction of ``mask``'s pixels that lie inside ``region`` (0.0 for an empty mask)."""
    n_pixels = int(np.count_nonzero(mask))
    if n_pixels == 0:
        return 0.0
    return int(np.count_nonzero(np.logical_and(mask, region))) / n_pixels


def find_ghost_origin(mask, removed_regions: Sequence[RemovedRegion], threshold: float):
    """``(origin_iter, overlap)`` of the latest removal whose region holds ``mask``, or None.

    The latest one is the edit that last repainted those pixels. ``removed_regions`` must be
    in iteration order.
    """
    match = None
    for removed in removed_regions:
        if removed.region is None or removed.region.shape != mask.shape:
            continue
        overlap = mask_coverage(mask, removed.region)
        if overlap >= threshold:
            match = (removed.origin_iter, overlap)
    return match


def prompt_types_tried(removed_regions: Sequence[RemovedRegion], origin_iter: int) -> list:
    """Prompt types already tried on the object of ``origin_iter`` and on its ghosts."""
    tried = []
    for removed in removed_regions:
        if removed.origin_iter == origin_iter:
            tried += [t for t in removed.prompt_types_tried if t not in tried]
    return tried


def find_ghosts(masks, removed_regions: Sequence[RemovedRegion], prompt_order, threshold: float) -> list:
    """The masks that are ghosts of earlier removals, each with the prompt types left to try."""
    ghosts = []
    for index, mask in enumerate(masks):
        match = find_ghost_origin(mask, removed_regions, threshold)
        if match is None:
            continue
        origin_iter, overlap = match
        tried = prompt_types_tried(removed_regions, origin_iter)
        untried = [t for t in prompt_order if t not in tried]
        ghosts.append(Ghost(index=index, origin_iter=origin_iter, overlap=overlap, untried_prompt_types=untried))
    return ghosts


def load_removed_regions(out_dir: str, before_iter: int, default_prompt_type: str) -> list:
    """Rebuild the removals of the iterations below ``before_iter`` from a stage-5 output dir.

    An object iteration is read from its obj_cat_list manifest, an iteration that removed a
    ghost from its skip record; other skips removed nothing. A manifest written before
    ``removal_prompt_types_tried`` existed is taken to have used ``default_prompt_type``, the
    type the removal loop tries first. Unreadable files are skipped with a warning, as stage 5
    does when it infers where to resume.
    """
    records = {}
    # Manifests are read last so that one wins over a stale skip record of the same index.
    for dir_name in ("skipped_iterations", "obj_cat_list"):
        dir_path = os.path.join(out_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for fname in os.listdir(dir_path):
            match = _RECORD_NAME.fullmatch(fname)  # skips iter_N.partial.json
            if match is None or int(match.group(1)) >= before_iter:
                continue
            try:
                with open(os.path.join(dir_path, fname)) as f:
                    records[int(match.group(1))] = (dir_name, json.load(f))
            except (OSError, ValueError) as e:
                logger.warning(f"Ignoring unreadable stage-5 record {os.path.join(dir_path, fname)}: {e}")

    regions = []
    for iteration in sorted(records):
        dir_name, record = records[iteration]
        if dir_name == "obj_cat_list":
            origin_iter = iteration
            tried = record.get("removal_prompt_types_tried") or [
                record.get("removal_prompt_type") or default_prompt_type
            ]
        elif record.get("reason") == GHOST_SKIP_REASON:
            origin_iter = record["ghost_of_iter"]
            tried = record.get("removal_prompt_types_tried", [])
        else:
            continue
        region = None
        mask_fpath = os.path.join(out_dir, "removal_mask", f"iter_{iteration}.png")
        if os.path.isfile(mask_fpath):
            try:
                mask = np.array(Image.open(mask_fpath).convert("L"))
            except OSError as e:
                logger.warning(f"Ignoring unreadable removal mask {mask_fpath}: {e}")
            else:
                region = grow_region(mask > 127)
        regions.append(RemovedRegion(
            iteration=iteration, origin_iter=origin_iter, region=region, prompt_types_tried=list(tried),
        ))
    return regions
