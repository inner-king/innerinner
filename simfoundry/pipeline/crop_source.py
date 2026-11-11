# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide which stage-5 object cutouts come from the original frame (spec SPEC_crop_source_ko.v2, I58).

Stage 5 erases objects one at a time and cuts each erased object out of the image it was erased from. The rules here
decide, for each erased object, which SAM3 instance of the ORIGINAL frame it is, and build its cutout from that
instance wherever the original shows it. The module is pure: numpy and cv2 only, no SAM3, no torch, no file I/O. The
stage-5 code calls these functions and writes the files.

Limits the caller must respect:

- original 모드는 이어 돌리기를 지원하지 않는다(풀과 가져간 원본을 이어받지 못함). The pool and the originals it has
  claimed are not saved, so a run that resumes cannot continue them. The stage-5 caller refuses to resume in that mode
  rather than silently producing a different result.
- An original is claimed once. A later mask that wants a claimed original does not get it; the earlier iteration does.

Coordinates are the stage-5 work grid: the same pixel grid as the removal masks and ``source_padded_resized.png``.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

RESCUE_SCOPES = ("dropped_only", "all", "none")
MATCHED_MASK_MODES = ("union", "original_plus_hidden", "original")
# The picture a matched cutout takes its pixels from inside the original mask. "frame" is the original frame and the default;
# "start_picture" is the picture the image model drew again before the removal. The second is an experiment, so
# check_pixel_source refuses it unless the upsampled image is on.
MATCHED_PIXEL_SOURCES = ("frame", "start_picture")
DECISION_KINDS = ("matched", "no_original", "revealed_later", "trace")

# Constants of the spec, not settings: the duplicate rule across all categories (2-1), the least overlap that makes an
# original the own original of a dropped mask (2-3), and the two margins of stage 5a (masked_object, masked_object_focus).
DEDUPE_IOU = 0.5
SELF_ORIGINAL_MIN_IOU = 0.1
MARGIN_FRACTION = 0.5
MARGIN_FRACTION_FOCUS = 0.1
OCCLUSION_PX = 2

# The rule that makes a mask ``no_original`` (2-2, step 2 (a)), decided by research 2 (v3 notes, md5 ef5ba3ec85d1): SAM3
# returned no instance for the category's own prompt, counted before merging, OR the mask overlaps no original of the
# pool (its largest IoU is below NO_OVERLAP_IOU). "count_zero" is the earlier rule, kept for comparison in the replay.
NO_ORIGINAL_RULE = "count_zero_or_no_overlap"
NO_OVERLAP_IOU = 0.1


def _is_no_original(pool: "OriginalPool", category_prompt: str, best_any_iou: float, rule: str = NO_ORIGINAL_RULE) -> bool:
    """Whether a mask that paired with nothing counts as ``no_original`` under the given rule.

    ``best_any_iou`` is the largest IoU with any original of the pool, claimed or free.
    """
    if rule == "count_zero_or_no_overlap":
        return pool.n_found(category_prompt) == 0 or best_any_iou < NO_OVERLAP_IOU
    if rule == "count_zero":
        return pool.n_found(category_prompt) == 0
    raise ValueError(f"unknown no_original rule {rule!r}")


def _as_bool(mask, name: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    return arr.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / float(union)


@dataclass(frozen=True)
class CropParams:
    """The ``s5_scene`` keys of spec section 5, with the spec's defaults.

    ``matched_pixel_source`` is not a spec key. It is the switch of the start-picture experiment, and its default keeps the
    spec's pixels: the original frame.
    """

    original_match_iou: float = 0.5
    revealed_inside_fraction: float = 0.5
    rescue_scope: str = "dropped_only"
    rescue_min_score: float = 0.5
    matched_mask: str = "union"
    matched_pixel_source: str = "frame"

    @classmethod
    def from_cfg(cls, s5_scene) -> "CropParams":
        """Read the keys with ``s5_scene.get(key, default)``. A value out of range or of the wrong type raises ValueError.

        ``original_match_iou`` must be above 0: an IoU bar of 0 would pair every mask with some original.
        """

        def number(key, default, low_open=False):
            value = s5_scene.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"s5_scene.{key} must be a number, got {value!r}")
            value = float(value)
            low_ok = value > 0.0 if low_open else value >= 0.0
            if not (np.isfinite(value) and low_ok and value <= 1.0):
                bound = "(0, 1]" if low_open else "[0, 1]"
                raise ValueError(f"s5_scene.{key} must be in {bound}, got {value}")
            return value

        def choice(key, default, options):
            value = s5_scene.get(key, default)
            if value not in options:
                raise ValueError(f"s5_scene.{key} must be one of {options}, got {value!r}")
            return value

        return cls(
            original_match_iou=number("original_match_iou", 0.5, low_open=True),
            revealed_inside_fraction=number("revealed_inside_fraction", 0.5),
            rescue_scope=choice("rescue_scope", "dropped_only", RESCUE_SCOPES),
            rescue_min_score=number("rescue_min_score", 0.5),
            matched_mask=choice("matched_mask", "union", MATCHED_MASK_MODES),
            matched_pixel_source=choice("matched_pixel_source", "frame", MATCHED_PIXEL_SOURCES),
        )


@dataclass
class OriginalEntry:
    """One SAM3 object of the ORIGINAL frame, after merging across categories (2-1)."""

    index: int  # the number: 0-based, in the order the object first appeared
    prompt: str  # the category prompt that holds it now (the first one, or a higher-scoring newcomer)
    score: float
    mask: np.ndarray  # bool (H, W), the work grid
    aliases: List[str] = field(default_factory=list)  # other prompts that found the same object; merged, not objects
    claimed_by: Optional[int] = None  # the iteration that took this original (2-2); None while free


class OriginalPool:
    """The originals of the ORIGINAL frame, merged across categories, and which iteration took each one (2-1, 2-2).

    The caller adds the prompts in the order their categories first appear in the iterations, one ``add_prompt`` call per
    prompt, and asks SAM3 on the original frame for each new prompt only. A prompt is the object's primary name,
    ``get_primary_object_name(obj_cat).lower()`` (the first segment before " / "); if that is empty, the whole ``obj_cat``
    in lower case (spec 2-1 as amended by research 2, Q3). The same string is the pool key, the category of the matching,
    and the key of ``n_found``. The caller passes only the instances at or above the SAM3 threshold.
    """

    def __init__(self, dedupe_iou: float = DEDUPE_IOU):
        if not 0.0 < dedupe_iou <= 1.0:
            raise ValueError(f"dedupe_iou must be in (0, 1], got {dedupe_iou}")
        self._dedupe_iou = float(dedupe_iou)
        self._entries: List[OriginalEntry] = []
        self._counts: Dict[str, int] = {}
        self._claim_order: List[int] = []
        self.shape: Optional[Tuple[int, int]] = None
        # SAM3 can return an empty mask for an instance above the threshold (the mask is made at a different threshold).
        # Such a mask is not an object: it is skipped, counted here, and not counted by n_found.
        self.n_empty_skipped = 0
        self.empty_skipped_by_prompt: Dict[str, int] = {}

    @staticmethod
    def _key(prompt) -> str:
        # Case and surrounding spaces do not make another category; underscores are left as they are.
        return str(prompt).strip().lower()

    def add_prompt(self, prompt: str, scores: Sequence[float], masks: Sequence[np.ndarray]) -> None:
        """Add the SAM3 instances of one prompt, merging them with every original already in the pool.

        An instance whose IoU with an existing original is above ``dedupe_iou`` is the same object. If the original is
        still free and the newcomer scores higher, the newcomer's name, score and mask replace the old ones and the old
        name becomes an alias. A claimed original keeps its mask and name (its cutout may exist already); a new name for it
        is only recorded as an alias. Numbers follow the order of first appearance.
        """
        key = self._key(prompt)
        scores = [float(s) for s in scores]
        masks = list(masks)
        if len(scores) != len(masks):
            raise ValueError(f"{len(scores)} scores for {len(masks)} masks of prompt {key!r}")
        for score, raw in zip(scores, masks):
            mask = _as_bool(raw, f"SAM3 mask of {key!r}")
            if self.shape is None:
                self.shape = mask.shape
            elif mask.shape != self.shape:
                raise ValueError(f"mask of {key!r} has shape {mask.shape}, the pool has {self.shape}")
            if not mask.any():
                self.n_empty_skipped += 1
                self.empty_skipped_by_prompt[key] = self.empty_skipped_by_prompt.get(key, 0) + 1
                continue
            self._counts[key] = self._counts.get(key, 0) + 1
            dup = self._first_duplicate(mask)
            if dup is None:
                self._entries.append(OriginalEntry(index=len(self._entries), prompt=key, score=score, mask=mask))
                continue
            entry = self._entries[dup]
            if entry.claimed_by is None and score > entry.score:
                old = entry.prompt
                if old != key and old not in entry.aliases:
                    entry.aliases.append(old)
                if key in entry.aliases:
                    entry.aliases.remove(key)
                entry.prompt, entry.score, entry.mask = key, score, mask
            elif key != entry.prompt:
                _add_alias(entry, key)

    def _first_duplicate(self, mask: np.ndarray) -> Optional[int]:
        for entry in self._entries:
            if _iou(mask, entry.mask) > self._dedupe_iou:
                return entry.index
        return None

    def n_found(self, prompt: str) -> int:
        """How many instances SAM3 returned for this prompt, before any merging. Zero means ``no_original``."""
        return self._counts.get(self._key(prompt), 0)

    def entries(self) -> List[OriginalEntry]:
        return list(self._entries)

    def free_indices(self) -> List[int]:
        """The numbers of the originals nobody has claimed yet, in number order."""
        return [entry.index for entry in self._entries if entry.claimed_by is None]

    def claim(self, index: int, iteration: int) -> None:
        """Give the original to an iteration. A second claim of the same original raises ValueError."""
        if not 0 <= index < len(self._entries):
            raise ValueError(f"no original number {index}")
        entry = self._entries[index]
        if entry.claimed_by is not None:
            raise ValueError(f"original {index} is already taken by iteration {entry.claimed_by}")
        entry.claimed_by = int(iteration)
        self._claim_order.append(index)

    def claimed_union(self, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """The union of the masks of the claimed originals; all False when none is claimed.

        ``shape`` is needed only for an empty pool, which has no shape of its own.
        """
        if shape is None:
            if self.shape is None:
                raise ValueError("an empty pool has no shape; pass shape")
            shape = self.shape
        out = np.zeros(tuple(shape), dtype=bool)
        for entry in self._entries:
            if entry.claimed_by is not None:
                out |= entry.mask
        return out

    def claimed_order(self) -> List[int]:
        """The numbers of the claimed originals, in the order the iterations took them (the removal order)."""
        return list(self._claim_order)

    def claimed_masks(self) -> List[np.ndarray]:
        """The masks of the claimed originals, in the same order as ``claimed_order``."""
        return [self._entries[i].mask for i in self._claim_order]


def _add_alias(entry: OriginalEntry, name: str) -> None:
    if name != entry.prompt and name not in entry.aliases:
        entry.aliases.append(name)


@dataclass
class Decision:
    """What the rules decided for one erased object (2-2)."""

    kind: str  # one of DECISION_KINDS
    index: Optional[int] = None  # the original taken; None unless kind == "matched"
    iou: float = 0.0  # IoU with the taken original (matched only)
    containment: float = 0.0  # |mask & original| / |mask| with the taken original; recorded, not used to decide
    coverage: float = 0.0  # |mask & original| / |original| with the taken original; recorded, not used to decide
    best_original_iou: float = 0.0  # the nearest original of ANY number (the reference of 2-3)
    best_original_index: Optional[int] = None
    inside_claimed_fraction: float = 0.0  # the share of the mask inside the claimed originals (the revealed_later rule)


def match_iteration_mask(iter_mask: np.ndarray, pool: OriginalPool, params: CropParams, category_prompt: str) -> Decision:
    """Pair one erased object's mask with an original (2-2, steps 1 and 2). The pool is not changed.

    Step 1: among the free originals, the largest IoU with ``iter_mask``, ties going to the lower number. An IoU of at
    least ``params.original_match_iou`` makes the pair ("matched"); the caller then claims the original. The name of the
    category does not enter this choice: a name condition would lose matches, since the VLM names change per iteration.

    Step 2, when nothing pairs, in this order:

    - ``no_original``: SAM3 returned no instance for ``category_prompt`` on the original (``pool.n_found`` is 0), or the
      mask overlaps no original of the pool: the largest IoU with any original, claimed ones included, is below
      ``NO_OVERLAP_IOU`` (0.1). The rule is decided by research 2 (v3 notes item 5); see ``_is_no_original``.
    - ``revealed_later``: more than ``params.revealed_inside_fraction`` of the mask lies inside the claimed originals.
    - ``trace``: otherwise (the largest IoU is at least 0.1 and below 0.5, and the mask is not revealed later). The cutout
      is not written; the trace is recorded.

    ``category_prompt`` is the primary name of the object's category (see ``OriginalPool``): the same key the pool uses.
    Containment and coverage are recorded for the matched original and are not used for the decision.
    """
    mask = _as_bool(iter_mask, "iteration mask")
    if pool.shape is not None and mask.shape != pool.shape:
        raise ValueError(f"iteration mask has shape {mask.shape}, the pool has {pool.shape}")
    if not mask.any():
        raise ValueError("the iteration mask is empty")
    entries = pool.entries()
    best_any, best_any_iou = None, 0.0
    best_free, best_free_iou = None, 0.0
    for entry in entries:
        iou = _iou(mask, entry.mask)
        if iou > best_any_iou:
            best_any, best_any_iou = entry.index, iou
        if entry.claimed_by is None and iou > best_free_iou:
            best_free, best_free_iou = entry.index, iou
    inside = _inside_claimed_fraction(mask, pool)
    if best_free is not None and best_free_iou >= params.original_match_iou:
        original = entries[best_free].mask
        inter = np.count_nonzero(mask & original)
        return Decision(
            kind="matched",
            index=best_free,
            iou=best_free_iou,
            containment=float(inter) / float(np.count_nonzero(mask)),
            coverage=float(inter) / float(np.count_nonzero(original)),
            best_original_iou=best_any_iou,
            best_original_index=best_any,
            inside_claimed_fraction=inside,
        )
    if _is_no_original(pool, category_prompt, best_any_iou):
        kind = "no_original"
    elif inside > params.revealed_inside_fraction:
        kind = "revealed_later"
    else:
        kind = "trace"
    return Decision(
        kind=kind,
        best_original_iou=best_any_iou,
        best_original_index=best_any,
        inside_claimed_fraction=inside,
    )


def _inside_claimed_fraction(mask: np.ndarray, pool: OriginalPool) -> float:
    claimed = pool.claimed_union(mask.shape)
    return float(np.count_nonzero(mask & claimed)) / float(np.count_nonzero(mask))


def _check_source_value(matched_pixel_source) -> None:
    if matched_pixel_source not in MATCHED_PIXEL_SOURCES:
        raise ValueError(
            f"s5_scene.matched_pixel_source must be one of {MATCHED_PIXEL_SOURCES}, got {matched_pixel_source!r}")


def check_pixel_source(matched_pixel_source, use_upsampled_source_image) -> None:
    """Refuse a pixel-source setting the stage cannot honour.

    ``"frame"`` is accepted whatever ``use_upsampled_source_image`` holds, and that flag is not read then: a config that ran
    before this key existed keeps running. ``"start_picture"`` needs the upsampled image on. With it off, the start picture
    is the frame itself, so the experiment would silently run the frame arm. The flag must also be a bool, so that a string
    or a 0 or 1 is not read as true.
    """
    _check_source_value(matched_pixel_source)
    if matched_pixel_source == "frame":
        return
    if not isinstance(use_upsampled_source_image, (bool, np.bool_)):
        raise ValueError(
            "s5_scene.use_upsampled_source_image must be true or false when s5_scene.matched_pixel_source is "
            f"start_picture, got {use_upsampled_source_image!r}")
    if not use_upsampled_source_image:
        raise ValueError(
            "s5_scene.matched_pixel_source: start_picture needs s5_scene.use_upsampled_source_image: true "
            "(start_picture_needs_upsampled_source). With it false the start picture is the frame itself, so this run "
            "would silently be a frame run.")


def check_start_picture(frame_rgb, start_rgb) -> None:
    """Refuse a start picture that cannot stand in for the frame's pixels.

    The two pictures must have the same shape. They must also differ. A start picture equal to the frame (same shape and
    same values, which for two uint8 pictures is the same bytes) is the frame under another name, and a run with it would
    be a frame run. That happens when the upsample call gave no picture and the stage fell back to the frame, or when the
    picture is the frame itself.
    """
    frame = np.asarray(frame_rgb)
    start = np.asarray(start_rgb)
    if frame.shape != start.shape:
        raise ValueError(f"the start picture has shape {start.shape}, the frame has shape {frame.shape}")
    if np.array_equal(frame, start):
        raise ValueError(
            "the start picture equals the frame (start_picture_equals_frame): same shape and same values. Either the "
            "upsample call returned no picture and the stage fell back to the frame, or the start picture is the frame "
            "itself. This run would be the same as a frame run, so it is refused.")


def cutout_pixels(matched_pixel_source, frame_rgb, start_rgb) -> np.ndarray:
    """The picture a matched object's cutout takes its pixels inside the original mask from (2-2 step 3).

    ``"frame"`` returns ``frame_rgb`` itself and does not look at ``start_rgb``, whatever it holds: the default must not depend
    on the start picture, as it does not depend on ``use_upsampled_source_image``. ``"start_picture"`` returns ``start_rgb``
    itself, after checking that it has the frame's shape, so the choice cannot move the grid. Neither returns a copy. A wrong
    value is a ValueError. That the start picture is not the frame is ``check_start_picture``'s job.
    """
    _check_source_value(matched_pixel_source)
    if matched_pixel_source == "frame":
        return frame_rgb
    frame = np.asarray(frame_rgb)
    start = np.asarray(start_rgb)
    if frame.shape != start.shape:
        raise ValueError(f"the start picture has shape {start.shape}, the frame has shape {frame.shape}")
    return start_rgb


def composite_cutout(
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    original_mask: np.ndarray,
    iter_mask: np.ndarray,
    claimed_union: Optional[np.ndarray],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build the cutout of a matched object (2-2, step 3). Returns (image, mask, edited_pixel_share).

    Pixels: inside ``original_mask`` they come from ``source_rgb``, the picture the caller chose with ``cutout_pixels``;
    outside it they come from ``current_rgb`` (this iteration's image). Mask, by ``mode``:

    - ``union``: original | iter_mask. Its area is never smaller than the iteration mask's.
    - ``original_plus_hidden``: original | (iter_mask & claimed_union); adds only what an earlier claimed object hid.
    - ``original``: the original mask alone (v1, for comparison).

    ``edited_pixel_share`` is the share of final-mask pixels outside ``original_mask``, which come from ``current_rgb``.
    ``claimed_union`` is the union of the masks of the originals claimed before this iteration (None when there are none).

    ``source_rgb`` is, by default (``matched_pixel_source="frame"``), the stage's ``source_padded_resized.png``: the original
    frame on the work grid, whose pixels inside the original mask are the real ones (spec 2-2 step 3; research 2, Q1).
    With ``"start_picture"`` the caller passes the start picture instead: the upsampled image the image model drew again
    before the removal, so the pixels inside ``original_mask`` are that drawing's. The code below is the same for both, and
    the mask and the edited share do not depend on the choice.
    """
    if mode not in MATCHED_MASK_MODES:
        raise ValueError(f"mode must be one of {MATCHED_MASK_MODES}, got {mode!r}")
    current = np.asarray(current_rgb)
    source = np.asarray(source_rgb)
    if current.shape != source.shape:
        raise ValueError(f"current image {current.shape} and source image {source.shape} differ")
    original = _as_bool(original_mask, "original mask")
    iteration = _as_bool(iter_mask, "iteration mask")
    if original.shape != current.shape[:2] or iteration.shape != current.shape[:2]:
        raise ValueError("the masks must match the image's height and width")
    if mode == "union":
        final = original | iteration
    elif mode == "original_plus_hidden":
        hidden = np.zeros_like(original) if claimed_union is None else iteration & _as_bool(claimed_union, "claimed union")
        final = original | hidden
    else:
        final = original.copy()
    if not final.any():
        raise ValueError("the final mask is empty")
    image = current.copy()
    image[original] = source[original]
    share = float(np.count_nonzero(final & ~original)) / float(np.count_nonzero(final))
    return image, final, share


def _crop_with_mask(image: np.ndarray, mask: np.ndarray, buffer_pixels: int, target_aspect_ratio: float):
    """``crop_image_with_mask`` of simfoundry/utils/processing_utils.py, copied step for step in numpy.

    That module imports torch and open3d, which this one must not. The stage-5 test checks that this copy and the original
    give the same bytes, so a change on either side shows up. Returns (cropped, (x1, y1, x2, y2)).
    """
    img_height, img_width = image.shape[:2]
    active = np.where(mask > 0)
    if len(active[0]) == 0:
        raise ValueError("No active pixels found in mask")
    y_min, y_max = active[0].min(), active[0].max()
    x_min, x_max = active[1].min(), active[1].max()
    x1 = max(0, x_min - buffer_pixels)
    y1 = max(0, y_min - buffer_pixels)
    x2 = min(img_width - 1, x_max + buffer_pixels)
    y2 = min(img_height - 1, y_max + buffer_pixels)
    current_width = x2 - x1 + 1
    current_height = y2 - y1 + 1
    if target_aspect_ratio is not None:
        current_aspect = current_width / current_height
        if current_aspect < target_aspect_ratio:
            target_width_new = int(current_height * target_aspect_ratio)
            width_diff = target_width_new - current_width
            expand_left = width_diff // 2
            expand_right = width_diff - expand_left
            new_x1 = x1 - expand_left
            new_x2 = x2 + expand_right
            if new_x1 < 0:
                excess = -new_x1
                new_x1 = 0
                new_x2 = min(img_width - 1, new_x2 + excess)
            if new_x2 >= img_width:
                excess = new_x2 - (img_width - 1)
                new_x2 = img_width - 1
                new_x1 = max(0, new_x1 - excess)
            x1, x2 = new_x1, new_x2
        elif current_aspect > target_aspect_ratio:
            target_height_new = int(current_width / target_aspect_ratio)
            height_diff = target_height_new - current_height
            expand_top = height_diff // 2
            expand_bottom = height_diff - expand_top
            new_y1 = y1 - expand_top
            new_y2 = y2 + expand_bottom
            if new_y1 < 0:
                excess = -new_y1
                new_y1 = 0
                new_y2 = min(img_height - 1, new_y2 + excess)
            if new_y2 >= img_height:
                excess = new_y2 - (img_height - 1)
                new_y2 = img_height - 1
                new_y1 = max(0, new_y1 - excess)
            y1, y2 = new_y1, new_y2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return image[y1:y2 + 1, x1:x2 + 1], (x1, y1, x2, y2)


def crop_like_5a(
    image: np.ndarray,
    mask: np.ndarray,
    target_ratio: float,
    resolution: Tuple[int, int],
    margin_fraction: float = MARGIN_FRACTION,
) -> np.ndarray:
    """Cut the object out as stage 5a does: one function, returning RGBA. Returns uint8 (H, W, 4).

    The margin is ``ceil(max(height, width) of the mask * margin_fraction)`` pixels, 0.5 for ``masked_object`` and 0.1 for
    ``masked_object_focus``. The crop takes the work ratio ``target_ratio``, is resized to ``resolution`` (W, H) with
    cv2's default interpolation, and pixels outside the mask become white. The alpha channel is the resized mask itself.
    The default path keeps the inline formula of stage 5a. A test extracts that formula from the stage's syntax tree and
    checks byte equality with this function, so the two cannot drift apart unnoticed.
    """
    mask = _as_bool(mask, "mask")
    if not mask.any():
        raise ValueError("No active pixels found in mask")
    if (not isinstance(resolution, (tuple, list)) or len(resolution) != 2
            or not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v > 0 for v in resolution)):
        raise ValueError(f"resolution must be a (W, H) pair of positive ints, got {resolution!r}")
    resolution = (int(resolution[0]), int(resolution[1]))
    h_idx, w_idx = np.where(mask)
    min_h, max_h = h_idx.min(), h_idx.max()
    min_w, max_w = w_idx.min(), w_idx.max()
    margin = int(np.ceil(max(max_h - min_h, max_w - min_w) * margin_fraction))
    rgb, _ = _crop_with_mask(image, mask, margin, target_ratio)
    rgb = cv2.resize(rgb, resolution)
    mask_u8 = mask.astype(np.uint8) * 255
    mask_crop, _ = _crop_with_mask(mask_u8, mask, margin, target_ratio)
    mask_out = cv2.resize(mask_crop, resolution)
    rgb[~mask_out.astype(bool)] = 255
    return np.concatenate([rgb, mask_out[..., np.newaxis]], axis=-1)


@dataclass
class DroppedOutcome:
    """What happens to one dropped (trace) mask at the end of the loop (2-3)."""

    position: int  # the position of the dropped mask in the list the caller gave, which is the iteration order
    own_index: Optional[int]  # the own original: IoU largest of any number, at least SELF_ORIGINAL_MIN_IOU; None if none
    own_iou: float
    rescued: bool
    # "rescued" | "no_self_original" | "self_original_taken" | "self_original_below_score" | "scope_none"
    reason: str


@dataclass
class RescuePlan:
    """The rescue decisions of 2-3: one outcome per dropped mask, and the extra originals of the ``all`` scope."""

    per_dropped: List[DroppedOutcome]
    extra: List[int]  # ``all`` only: free originals that are no dropped mask's own, in number order


def _self_original(mask: np.ndarray, entries: List[OriginalEntry]) -> Tuple[Optional[int], float]:
    best, best_iou = None, 0.0
    for entry in entries:
        iou = _iou(mask, entry.mask)
        if iou > best_iou:
            best, best_iou = entry.index, iou
    if best is None or best_iou < SELF_ORIGINAL_MIN_IOU:
        return None, best_iou
    return best, best_iou


def rescue_plan(pool: OriginalPool, dropped: Sequence[np.ndarray], params: CropParams) -> RescuePlan:
    """The rescue decisions at the end of the loop (2-3). ``dropped`` holds the masks of the trace decisions.

    The own original of a dropped mask is the original of ANY number with the largest IoU, when that IoU is at least
    SELF_ORIGINAL_MIN_IOU. It is rescued when nobody took it and its score is at least ``params.rescue_min_score``. Two
    dropped masks with the same free own original rescue it once, the first in position order; the second is outcome
    "self_original_taken". A dropped mask whose own original an iteration took is lost, and so is one with no own original.

    Scopes: ``dropped_only`` (default) gives the own rescues only. ``all`` adds every other free original whose score
    passes. ``none`` rescues nothing, but still reports the own originals for the record.
    """
    entries = pool.entries()
    rescued_now = set()
    outcomes: List[DroppedOutcome] = []
    for position, raw in enumerate(dropped):
        mask = _as_bool(raw, f"dropped mask {position}")
        if pool.shape is not None and mask.shape != pool.shape:
            raise ValueError(f"dropped mask {position} has shape {mask.shape}, the pool has {pool.shape}")
        own, own_iou = _self_original(mask, entries)
        if own is None:
            outcomes.append(DroppedOutcome(position, None, own_iou, False, "no_self_original"))
            continue
        if params.rescue_scope == "none":
            outcomes.append(DroppedOutcome(position, own, own_iou, False, "scope_none"))
            continue
        entry = entries[own]
        if entry.claimed_by is not None or own in rescued_now:
            outcomes.append(DroppedOutcome(position, own, own_iou, False, "self_original_taken"))
        elif entry.score < params.rescue_min_score:
            outcomes.append(DroppedOutcome(position, own, own_iou, False, "self_original_below_score"))
        else:
            rescued_now.add(own)
            outcomes.append(DroppedOutcome(position, own, own_iou, True, "rescued"))
    extra: List[int] = []
    if params.rescue_scope == "all":
        extra = [
            entry.index
            for entry in entries
            if entry.claimed_by is None and entry.score >= params.rescue_min_score and entry.index not in rescued_now
        ]
    return RescuePlan(per_dropped=outcomes, extra=extra)


def occluded_by(original_mask: np.ndarray, earlier_masks: Sequence[np.ndarray], earlier_numbers: Sequence[int],
                px: int = OCCLUSION_PX) -> List[int]:
    """The numbers of the objects removed earlier whose originals touch this original grown by ``px`` pixels (4절).

    ``earlier_masks[i]`` is the original of the object numbered ``earlier_numbers[i]``; both come from the pool in
    removal order (``claimed_masks`` and ``claimed_order``). The grow uses a (2 px + 1) square, as cv2.MORPH_RECT does.
    The result is recorded for the later infill experiment and is never used for a decision.
    """
    if len(earlier_masks) != len(earlier_numbers):
        raise ValueError("earlier_masks and earlier_numbers differ in length")
    if px < 0:
        raise ValueError(f"px must be 0 or more, got {px}")
    original = _as_bool(original_mask, "original mask").astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px + 1, 2 * px + 1))
    grown = cv2.dilate(original, kernel) > 0
    return [int(number) for number, mask in zip(earlier_numbers, earlier_masks)
            if np.any(grown & _as_bool(mask, "earlier mask"))]


def original_instance_record(entry: OriginalEntry, decision: Optional[Decision] = None) -> dict:
    """The ``original_instance`` dict of spec 6절 for one original (a pool entry), with the pairing numbers when it is matched."""
    record = {
        "index": int(entry.index),
        "category_prompt": entry.prompt,
        "aliases": list(entry.aliases),
        "sam3_score": float(entry.score),
        "iou_with_iteration_mask": None,
        "containment": None,
        "coverage": None,
    }
    if decision is not None and decision.kind == "matched" and decision.index == entry.index:
        record["iou_with_iteration_mask"] = float(decision.iou)
        record["containment"] = float(decision.containment)
        record["coverage"] = float(decision.coverage)
    return record


def object_record(decision: Decision, *, matched_mask: str, edited_pixel_share: float,
                  original_instance: Optional[dict] = None, occluded_by_numbers: Optional[List[int]] = None,
                  matched_pixel_source: str = "frame") -> dict:
    """The keys this module adds to an object's ``obj_cat_list`` record (spec 6절) for a matched, no_original or
    revealed_later object. Only added keys; no file I/O. A trace has no object record: use ``trace_record``.

    A matched object is cut from the original, so its share of pixels from the iteration image is what the caller measured.
    A no_original or revealed_later object is cut from the iteration image, so its share is 1.0 by definition.

    ``matched_pixel_source`` is the picture the cutout's pixels inside the original mask came from (``cutout_pixels``). It
    is recorded for a matched object; the other kinds record None, as ``matched_mask_mode`` does.
    """
    if decision.kind == "trace":
        raise ValueError("a trace has no object record; use trace_record")
    _check_source_value(matched_pixel_source)
    if decision.kind == "matched":
        if matched_mask not in MATCHED_MASK_MODES:
            raise ValueError(f"matched_mask must be one of {MATCHED_MASK_MODES}, got {matched_mask!r}")
        if original_instance is None:
            raise ValueError("a matched object needs its original_instance record")
        record = {
            "crop_source": "original",
            "crop_reason": "matched",
            "matched_mask_mode": matched_mask,
            "matched_pixel_source": matched_pixel_source,
            "edited_pixel_share": float(edited_pixel_share),
            "original_instance": original_instance,
        }
    else:
        record = {
            "crop_source": "edited",
            "crop_reason": decision.kind,
            "matched_mask_mode": None,
            "matched_pixel_source": None,
            "edited_pixel_share": 1.0,
            "best_original_iou": float(decision.best_original_iou),
            "best_original_index": decision.best_original_index,
        }
        if decision.kind == "revealed_later":
            record["inside_claimed_fraction"] = float(decision.inside_claimed_fraction)
    if occluded_by_numbers is not None:
        record["occluded_by"] = [int(n) for n in occluded_by_numbers]
    return record


def rescued_object_record(original_instance: dict, occluded_by_numbers: Optional[List[int]] = None,
                          matched_pixel_source: str = "frame") -> dict:
    """The keys of a rescued object (2-3): it is a new object cut from the original alone, so its mask is the original mask.

    ``base_iter`` is written as None on purpose: a missing key would make the resume code fall back to the last chain
    iteration, and a rescued object has no iteration. ``matched_pixel_source`` is the picture its pixels came from, as for a
    matched object (``cutout_pixels``).
    """
    _check_source_value(matched_pixel_source)
    record = {
        "crop_source": "original",
        "crop_reason": "rescued",
        "matched_mask_mode": "original",
        "matched_pixel_source": matched_pixel_source,
        "edited_pixel_share": 0.0,
        "base_iter": None,
        "original_instance": original_instance,
    }
    if occluded_by_numbers is not None:
        record["occluded_by"] = [int(n) for n in occluded_by_numbers]
    return record


def trace_record(decision: Decision, mask_area: int) -> dict:
    """The ``skipped_iterations`` record of a trace (spec 6절). ``rescued_as`` and ``lost_object`` are filled later by
    ``set_rescue_outcome``, once the rescue plan is known."""
    if decision.kind != "trace":
        raise ValueError(f"trace_record needs a trace decision, got {decision.kind!r}")
    return {
        "reason": "trace_no_original_match",
        "best_original_iou": float(decision.best_original_iou),
        "best_original_index": decision.best_original_index,
        "mask_area": int(mask_area),
        "rescued_as": None,
        "lost_object": None,
    }


def set_rescue_outcome(record: dict, rescued_as: Optional[int]) -> dict:
    """A copy of a trace record with the object number it was rescued as, or None. ``lost_object`` is True when no rescue
    happened (no own original, or the own original was taken by another iteration, or the scope does not rescue it)."""
    out = dict(record)
    out["rescued_as"] = None if rescued_as is None else int(rescued_as)
    out["lost_object"] = rescued_as is None
    return out


def originals_record(pool: OriginalPool, rescued_as: Optional[Dict[int, int]] = None,
                     raw_obj_cats: Optional[Dict[str, Sequence[str]]] = None) -> dict:
    """The json of ``original_instances.json`` (spec 6절): every original, its name, aliases, score, box, area, and what
    happened to it. ``rescued_as`` maps an original's number to the object number it was rescued as. ``raw_obj_cats`` maps
    a primary name to the full ``obj_cat`` strings it came from, so the file keeps both (research 2, Q3). Pure: no file I/O."""
    rescued_as = dict(rescued_as or {})
    raw_obj_cats = dict(raw_obj_cats or {})
    items = []
    for entry in pool.entries():
        ys, xs = np.nonzero(entry.mask)
        if entry.claimed_by is not None:
            reason = f"taken by iteration {entry.claimed_by}"
        elif entry.index in rescued_as:
            reason = f"rescued as object {rescued_as[entry.index]}"
        else:
            reason = "left free"
        items.append({
            "index": int(entry.index),
            "category_prompt": entry.prompt,
            "aliases": list(entry.aliases),
            "sam3_score": float(entry.score),
            "bbox_yx_min_max": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())],
            "area": int(np.count_nonzero(entry.mask)),
            "claimed_by": entry.claimed_by,
            "rescued_as": rescued_as.get(entry.index),
            "raw_obj_cats": list(raw_obj_cats.get(entry.prompt, [])),
            "reason": reason,
        })
    return {"shape": None if pool.shape is None else [int(v) for v in pool.shape], "originals": items}


def pack_originals(pool: OriginalPool) -> Dict[str, np.ndarray]:
    """The arrays of ``original_instances.npz`` (spec 6절): one bit-packed mask per original, key ``o<number>``, and ``shape``.

    The caller writes them with ``np.savez_compressed``. ``unpack_original`` reads one back. Pure: no file I/O.
    """
    if pool.shape is None:
        raise ValueError("an empty pool has nothing to pack")
    arrays = {"shape": np.asarray(pool.shape, dtype=np.int64)}
    for entry in pool.entries():
        arrays[f"o{entry.index}"] = np.packbits(entry.mask.ravel())
    return arrays


def unpack_original(packed: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """A bool mask of the given shape from one array made by ``pack_originals``."""
    h, w = int(shape[0]), int(shape[1])
    return np.unpackbits(packed)[: h * w].reshape(h, w).astype(bool)
