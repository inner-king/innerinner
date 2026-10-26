# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from simfoundry.pipeline.frame_selection import (
    FrameBundle,
    FrameScore,
    FrameSelectionError,
    is_auto_img_idx,
    load_selection,
    rank_frames,
    refine_with_vlm,
    resolve_img_idx,
    select_canonical_frame,
    selection_cfg,
    write_selection,
)
from simfoundry.pipeline import frame_selection


def make_cfg(tmp_path, img_idx="auto", **s3_overrides):
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "visualize": False,
            "gcloud_project": "test-project",
            "s3_ground": {
                "out_dir": str(tmp_path),
                "img_idx": img_idx,
                "floor_categories": ["desk", "table"],
                "floor_threshold": 0.5,
                "floor_tilt_threshold": 0.125,
                "detection_model": "gemini-3.1-pro-preview",
                **s3_overrides,
            },
            "s4_frame": {"z_far": 5.0},
            "s5_scene": {"img_idx": "${s3_ground.img_idx}"},
        }
    )
    return cfg


# --------------------------------------------------------------------------------------
# Index resolution
# --------------------------------------------------------------------------------------

def test_is_auto_img_idx():
    assert is_auto_img_idx(None)
    assert is_auto_img_idx("auto")
    assert is_auto_img_idx("  AUTO  ")
    assert is_auto_img_idx("")
    assert not is_auto_img_idx(0)
    assert not is_auto_img_idx(7)


def test_resolve_img_idx_honours_a_pinned_frame(tmp_path):
    cfg = make_cfg(tmp_path, img_idx=3)
    assert resolve_img_idx(cfg) == 3
    # s5_scene interpolates s3_ground, so pinning one pins both.
    assert resolve_img_idx(cfg, stage_key="s5_scene") == 3


def test_resolve_img_idx_reads_the_persisted_selection(tmp_path):
    cfg = make_cfg(tmp_path)
    (tmp_path / "frame_selection.json").write_text(json.dumps({"selected_idx": 6}))
    assert resolve_img_idx(cfg) == 6
    assert resolve_img_idx(cfg, stage_key="s5_scene") == 6


def test_resolve_img_idx_without_a_selection_names_the_stage_to_run(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(FrameSelectionError, match="Run stage 3"):
        resolve_img_idx(cfg)


def test_a_pinned_frame_wins_over_a_stale_selection_file(tmp_path):
    cfg = make_cfg(tmp_path, img_idx=2)
    (tmp_path / "frame_selection.json").write_text(json.dumps({"selected_idx": 9}))
    assert resolve_img_idx(cfg) == 2


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

def test_selection_cfg_defaults_and_inheritance(tmp_path):
    cfg = make_cfg(tmp_path)
    sel = selection_cfg(cfg)
    assert sel["mode"] == "hybrid"
    # Unset gates inherit the stage-3 values they exist to protect.
    assert sel["max_tilt"] == pytest.approx(0.125)
    assert sel["vlm_model"] == "gemini-3.1-pro-preview"


def test_selection_cfg_merges_partial_overrides(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "max_tilt": 0.4, "weights": {"sharpness": 0.5}}
    sel = selection_cfg(cfg)
    assert sel["mode"] == "heuristic"
    assert sel["max_tilt"] == pytest.approx(0.4)
    assert sel["weights"]["sharpness"] == pytest.approx(0.5)
    # Unspecified keys keep their defaults rather than disappearing.
    assert sel["weights"]["object_coverage"] == pytest.approx(0.40)
    assert sel["vlm_top_k"] == 4


def test_selection_cfg_rejects_an_unknown_mode(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "vibes"}
    with pytest.raises(ValueError, match="Unknown frame_selection.mode"):
        selection_cfg(cfg)


# --------------------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------------------

def _score(idx, **kw):
    base = dict(
        object_coverage=0.01, n_objects=2, sharpness=100.0,
        support_coverage=0.5, plane_inlier_ratio=0.9, clipped_frac=0.0,
    )
    base.update(kw)
    return FrameScore(idx=idx, **base)


def test_rank_frames_prefers_larger_objects(tmp_path):
    sel = selection_cfg(make_cfg(tmp_path))
    ranked = rank_frames([_score(0, object_coverage=0.004), _score(1, object_coverage=0.02)], sel)
    assert [s.idx for s in ranked] == [1, 0]


def test_rank_frames_prefers_sharper_and_better_separated_frames(tmp_path):
    sel = selection_cfg(make_cfg(tmp_path))
    blurry = _score(0, sharpness=10.0)
    sharp = _score(1, sharpness=500.0)
    assert [s.idx for s in rank_frames([blurry, sharp], sel)] == [1, 0]

    merged = _score(0, n_objects=1)
    separated = _score(1, n_objects=4)
    assert [s.idx for s in rank_frames([merged, separated], sel)] == [1, 0]


def test_rank_frames_penalizes_objects_running_off_the_edge(tmp_path):
    sel = selection_cfg(make_cfg(tmp_path))
    # Same frame twice apart from the clipping, so the penalty is what decides it.
    clipped = _score(0, clipped_frac=0.4)
    contained = _score(1, clipped_frac=0.0)
    assert [s.idx for s in rank_frames([clipped, contained], sel)] == [1, 0]


def test_rank_frames_drops_ineligible_frames(tmp_path):
    sel = selection_cfg(make_cfg(tmp_path))
    rejected = FrameScore(idx=0, eligible=False, reject_reason="roll 0.9 > max_tilt 0.125", object_coverage=0.5)
    ranked = rank_frames([rejected, _score(1)], sel)
    assert [s.idx for s in ranked] == [1]


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def _fake_bundle(n=4):
    rgbs = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(n)]
    return FrameBundle(rgbs, rgbs, [np.eye(3)] * n, [None] * n)


def _patch_scores(monkeypatch, scores_by_idx):
    def fake_score_frame(idx, bundle, sam3, cfg, sel_cfg):
        return scores_by_idx[idx]

    monkeypatch.setattr(frame_selection, "score_frame", fake_score_frame)


def test_select_canonical_frame_picks_the_best_heuristic_frame(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": False}
    _patch_scores(monkeypatch, {
        0: _score(0, object_coverage=0.004),
        1: _score(1, object_coverage=0.006),
        2: _score(2, object_coverage=0.02),
        3: _score(3, object_coverage=0.005),
    })
    selection = select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle())
    assert selection.selected_idx == 2
    assert selection.decided_by == "heuristic"
    assert selection.vlm_shortlist == []


def test_select_canonical_frame_lets_the_vlm_override_the_shortlist(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "hybrid", "vlm_top_k": 3, "write_debug_sheet": False}
    _patch_scores(monkeypatch, {i: _score(i, object_coverage=0.004 * (i + 1)) for i in range(4)})

    seen = {}

    def fake_refine(cfg_, bundle_, shortlist, sel_cfg_):
        seen["shortlist"] = list(shortlist)
        return shortlist[-1], "vlm picked option 3"

    selection = select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle(), refine_fn=fake_refine)
    assert seen["shortlist"] == [3, 2, 1]
    assert selection.selected_idx == 1
    assert selection.decided_by == "vlm"


def test_select_canonical_frame_falls_back_when_the_vlm_declines(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "hybrid", "write_debug_sheet": False}
    _patch_scores(monkeypatch, {i: _score(i, object_coverage=0.004 * (i + 1)) for i in range(4)})

    selection = select_canonical_frame(
        cfg, sam3=None, bundle=_fake_bundle(),
        refine_fn=lambda *a: (None, "vlm call failed: no credentials"),
    )
    assert selection.selected_idx == 3
    assert selection.decided_by == "heuristic"
    assert "no credentials" in selection.vlm_note


def test_select_canonical_frame_reports_why_every_frame_was_rejected(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": False}
    _patch_scores(monkeypatch, {
        i: FrameScore(idx=i, eligible=False, reject_reason=f"roll 0.{i}9 > max_tilt 0.125")
        for i in range(4)
    })
    with pytest.raises(FrameSelectionError, match="max_tilt"):
        select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle())


def test_select_canonical_frame_strides_long_captures(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "max_candidates": 4, "write_debug_sheet": False}
    scored = []

    def fake_score_frame(idx, bundle, sam3, cfg_, sel_cfg):
        scored.append(idx)
        return _score(idx)

    monkeypatch.setattr(frame_selection, "score_frame", fake_score_frame)
    select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle(n=40))
    # Evenly spread across the whole capture rather than only its first frames.
    assert scored == [0, 10, 20, 30]


def test_select_canonical_frame_returns_frame_ids_not_positions(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("open3d")
    pytest.importorskip("scipy")

    # Per-file backends number frames by filename, which need not match bundle positions.
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": False}
    bundle = _fake_bundle(3)
    bundle.frame_ids = [4, 9, 11]

    def fake_score_frame(position, bundle_, sam3, cfg_, sel_cfg):
        return _score(bundle_.frame_ids[position], object_coverage=0.004 * (position + 1))

    monkeypatch.setattr(frame_selection, "score_frame", fake_score_frame)
    selection = select_canonical_frame(cfg, sam3=None, bundle=bundle)
    assert selection.selected_idx == 11


def test_selection_round_trips_through_disk(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": False}
    _patch_scores(monkeypatch, {i: _score(i, object_coverage=0.004 * (i + 1)) for i in range(4)})

    selection = select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle())
    write_selection(cfg, selection)

    payload = load_selection(cfg)
    assert payload["selected_idx"] == 3
    assert len(payload["scores"]) == 4
    assert resolve_img_idx(cfg) == 3


# --------------------------------------------------------------------------------------
# VLM refinement
# --------------------------------------------------------------------------------------

class _StubVLM:
    def __init__(self, text, **kwargs):
        self._text = text

    def __call__(self, **kwargs):
        return self._text

    def get_result_text(self, result):
        return result


def _stub_vlm(monkeypatch, text):
    import simfoundry.models.vlm as vlm_module

    monkeypatch.setattr(vlm_module, "Gemini", lambda **kwargs: _StubVLM(text))
    monkeypatch.setattr(frame_selection, "_write_vlm_candidate_images", lambda *a, **k: ["a.png", "b.png", "c.png"])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Option 2 shows the marker largest.\nANSWER: 2", 5),
        ("ANSWER: [3]", 1),
        ("blah blah ANSWER: 1 ", 7),
    ],
)
def test_refine_with_vlm_parses_the_answer(tmp_path, monkeypatch, text, expected):
    _stub_vlm(monkeypatch, text)
    cfg = make_cfg(tmp_path)
    idx, note = refine_with_vlm(cfg, _fake_bundle(), [7, 5, 1], selection_cfg(cfg))
    assert idx == expected


@pytest.mark.parametrize("text", ["I like option two", "ANSWER: 9", "ANSWER: 0"])
def test_refine_with_vlm_declines_on_an_unusable_answer(tmp_path, monkeypatch, text):
    _stub_vlm(monkeypatch, text)
    cfg = make_cfg(tmp_path)
    idx, note = refine_with_vlm(cfg, _fake_bundle(), [7, 5, 1], selection_cfg(cfg))
    assert idx is None
    assert note


def test_refine_with_vlm_survives_a_failed_call(tmp_path, monkeypatch):
    import simfoundry.models.vlm as vlm_module

    def boom(**kwargs):
        raise RuntimeError("no credentials")

    monkeypatch.setattr(vlm_module, "Gemini", boom)
    monkeypatch.setattr(frame_selection, "_write_vlm_candidate_images", lambda *a, **k: ["a.png"])
    cfg = make_cfg(tmp_path)
    idx, note = refine_with_vlm(cfg, _fake_bundle(), [7, 5], selection_cfg(cfg))
    assert idx is None
    assert "no credentials" in note


# --------------------------------------------------------------------------------------
# Geometry (needs the optional CV stack)
# --------------------------------------------------------------------------------------

def test_score_frame_measures_objects_standing_on_the_support_plane(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("open3d")
    pytest.importorskip("scipy")

    H, W = 120, 160
    K = np.array([[200.0, 0.0, W / 2], [0.0, 200.0, H / 2], [0.0, 0.0, 1.0]])
    # A support plane 2 m out, with a 20 cm-tall block sitting on it.
    depth = np.full((H, W), 2.0, dtype=np.float32)
    depth[40:70, 50:90] = 1.8
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[40:70, 50:90] = 255

    class _StubSAM3:
        def predict_segmentation(self, pil_img, text_prompt):
            mask = np.ones((1, 1, H, W), dtype=bool)
            return mask, np.array([[0, 0, W, H]]), np.array([0.99])

    cfg = make_cfg(tmp_path)
    bundle = FrameBundle([rgb], [depth], [K], [None])
    score = frame_selection.score_frame(0, bundle, _StubSAM3(), cfg, selection_cfg(cfg))

    assert score.eligible, score.reject_reason
    assert score.n_objects == 1
    assert score.object_coverage == pytest.approx((30 * 40) / (H * W), rel=0.1)
    assert score.tilt == pytest.approx(0.0, abs=1e-6)
    assert score.plane_inlier_ratio > 0.9
    assert score.clipped_frac == pytest.approx(0.0)


def test_score_frame_rejects_a_frame_whose_objects_run_off_the_edge(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("open3d")
    pytest.importorskip("scipy")

    H, W = 120, 160
    K = np.array([[200.0, 0.0, W / 2], [0.0, 200.0, H / 2], [0.0, 0.0, 1.0]])
    depth = np.full((H, W), 2.0, dtype=np.float32)
    depth[40:70, :40] = 1.8  # block pushed against the left edge of the frame
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    class _StubSAM3:
        def predict_segmentation(self, pil_img, text_prompt):
            return np.ones((1, 1, H, W), dtype=bool), np.array([[0, 0, W, H]]), np.array([0.99])

    cfg = make_cfg(tmp_path)
    score = frame_selection.score_frame(0, FrameBundle([rgb], [depth], [K], [None]), _StubSAM3(), cfg, selection_cfg(cfg))
    assert score.clipped_frac == pytest.approx(1.0)
    assert not score.eligible
    assert "runs off the frame" in score.reject_reason


def test_score_frame_rejects_a_frame_with_no_support_surface(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("open3d")

    class _EmptySAM3:
        def predict_segmentation(self, pil_img, text_prompt):
            return np.zeros((0, 1, 8, 8), dtype=bool), np.zeros((0, 4)), np.zeros((0,))

    cfg = make_cfg(tmp_path)
    score = frame_selection.score_frame(0, _fake_bundle(1), _EmptySAM3(), cfg, selection_cfg(cfg))
    assert not score.eligible
    assert "support surface" in score.reject_reason


# --------------------------------------------------------------------------------------
# Hard gates, and the lone-candidate waiver
# --------------------------------------------------------------------------------------

def _passing(idx, **kw):
    """A frame that passes every hard gate; `kw` overrides metrics."""
    metrics = dict(tilt=0.0)
    metrics.update(kw)
    return _score(idx, **metrics)


def _rejected(idx, reason, **kw):
    """A frame that failed a gate, carrying the reason `score_frame` would have recorded."""
    score = _passing(idx, **kw)
    score.eligible = False
    score.reject_reason = reason
    return score


def _clipped(idx):
    return _rejected(idx, "65% of object area runs off the frame", clipped_frac=0.65)


@pytest.mark.parametrize(
    "metrics, gate, reason",
    [
        ({"tilt": 0.5}, "tilt", "roll 0.500 > max_tilt 0.125"),
        ({"plane_inlier_ratio": 0.3}, "plane", "plane inliers 0.30 too low"),
        ({"support_coverage": 0.01}, "support", "support fills only 1.0% of frame"),
        ({"n_objects": 0}, "no_objects", "no objects on the support surface"),
        ({"clipped_frac": 0.6}, "clipped", "60% of object area runs off the frame"),
    ],
)
def test_failed_gate_names_the_gate_and_gives_its_reason(tmp_path, metrics, gate, reason):
    sel = selection_cfg(make_cfg(tmp_path))
    assert frame_selection._failed_gate(_passing(0, **metrics), sel) == (gate, reason)


def test_failed_gate_is_none_when_every_gate_passes(tmp_path):
    sel = selection_cfg(make_cfg(tmp_path))
    assert frame_selection._failed_gate(_passing(0), sel) is None


@pytest.mark.parametrize(
    "metrics, gate",
    [
        ({"tilt": 0.5, "plane_inlier_ratio": 0.3, "clipped_frac": 0.9}, "tilt"),
        ({"plane_inlier_ratio": 0.3, "support_coverage": 0.01}, "plane"),
        ({"support_coverage": 0.01, "n_objects": 0}, "support"),
        # score_frame measures clipping only over kept objects, so this pair cannot come out of it.
        # It pins the order anyway: the reason a frame is reported under is part of its meaning.
        ({"n_objects": 0, "clipped_frac": 0.9}, "no_objects"),
    ],
)
def test_failed_gate_reports_the_first_gate_in_its_fixed_order(tmp_path, metrics, gate):
    sel = selection_cfg(make_cfg(tmp_path))
    assert frame_selection._failed_gate(_passing(0, **metrics), sel)[0] == gate


def test_a_lone_frame_rejected_only_for_clipping_is_kept(tmp_path):
    sel = selection_cfg(make_cfg(tmp_path))
    lone = _clipped(3)
    assert frame_selection.sole_candidate_fallback([lone], sel) is lone


@pytest.mark.parametrize(
    "metrics, reason",
    [
        ({"tilt": 0.5}, "roll 0.500 > max_tilt 0.125"),
        ({"plane_inlier_ratio": 0.3}, "plane inliers 0.30 too low"),
        ({"support_coverage": 0.01}, "support fills only 1.0% of frame"),
        ({"n_objects": 0}, "no objects on the support surface"),
        ({"tilt": 0.5, "clipped_frac": 0.9}, "roll 0.500 > max_tilt 0.125"),
    ],
)
def test_a_lone_frame_failing_any_other_gate_is_not_waived(tmp_path, metrics, reason):
    sel = selection_cfg(make_cfg(tmp_path))
    assert frame_selection.sole_candidate_fallback([_rejected(0, reason, **metrics)], sel) is None


def test_a_lone_frame_that_was_never_measured_is_not_waived(tmp_path):
    sel = selection_cfg(make_cfg(tmp_path))
    unmeasured = FrameScore(idx=0, eligible=False, reject_reason="no support surface detected")
    failed = FrameScore(idx=0, eligible=False, reject_reason="scoring failed: boom")
    assert frame_selection.sole_candidate_fallback([unmeasured], sel) is None
    assert frame_selection.sole_candidate_fallback([failed], sel) is None


def test_the_waiver_needs_exactly_one_candidate(tmp_path):
    sel = selection_cfg(make_cfg(tmp_path))
    assert frame_selection.sole_candidate_fallback([_clipped(0), _clipped(1)], sel) is None
    assert frame_selection.sole_candidate_fallback([], sel) is None


def test_an_eligible_lone_frame_is_not_a_fallback(tmp_path):
    sel = selection_cfg(make_cfg(tmp_path))
    assert frame_selection.sole_candidate_fallback([_passing(0)], sel) is None
    # score_frame never marks such a frame eligible; this checks the guard holds even if it did.
    assert frame_selection.sole_candidate_fallback([_passing(0, clipped_frac=0.8)], sel) is None


def test_the_waiver_follows_the_configured_clipping_limit(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"max_clipped_frac": 0.7}
    sel = selection_cfg(cfg)
    assert frame_selection.sole_candidate_fallback([_clipped(0)], sel) is None


def test_a_lone_frame_rejected_only_for_clipping_is_kept_without_asking_the_vlm(tmp_path, monkeypatch, caplog):
    import logging

    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "hybrid", "write_debug_sheet": False}
    _patch_scores(monkeypatch, {0: _clipped(0)})

    def refuse(*args):
        raise AssertionError("a lone candidate must not reach the VLM")

    with caplog.at_level(logging.WARNING, logger=frame_selection.__name__):
        selection = select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle(1), refine_fn=refuse)

    assert selection.selected_idx == 0
    assert selection.decided_by == "sole_candidate"
    assert selection.vlm_shortlist == []
    assert selection.scores[0].eligible is False
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("candidate frame (idx 0)" in w for w in warnings)


def test_max_candidates_of_one_scores_one_frame_and_keeps_it_when_clipped(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "max_candidates": 1, "write_debug_sheet": False}
    scored = []

    def fake_score_frame(position, bundle, sam3, cfg_, sel_cfg):
        scored.append(position)
        return _clipped(position)

    monkeypatch.setattr(frame_selection, "score_frame", fake_score_frame)
    selection = select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle(4))
    assert scored == [0]
    assert selection.selected_idx == 0
    assert selection.decided_by == "sole_candidate"


def test_a_lone_frame_rejected_for_roll_still_fails_stage_3(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": False}
    _patch_scores(monkeypatch, {0: _rejected(0, "roll 0.500 > max_tilt 0.125", tilt=0.5)})
    with pytest.raises(FrameSelectionError, match="max_tilt"):
        select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle(1))


def test_two_frames_rejected_for_clipping_still_fail_stage_3(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": False}
    _patch_scores(monkeypatch, {0: _clipped(0), 1: _clipped(1)})
    with pytest.raises(FrameSelectionError, match="runs off the frame"):
        select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle(2))


def test_an_eligible_frame_beats_a_clipped_one(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": False}
    _patch_scores(monkeypatch, {0: _clipped(0), 1: _passing(1, object_coverage=0.01)})
    selection = select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle(2))
    assert selection.selected_idx == 1
    assert selection.decided_by == "heuristic"


def test_a_lone_frame_still_gets_the_debug_sheet(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": True}
    _patch_scores(monkeypatch, {0: _clipped(0)})
    select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle(1))
    assert (tmp_path / "frame_selection.png").is_file()


def test_a_lone_frame_round_trips_through_the_selection_file(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)  # img_idx stays auto
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": False}
    bundle = _fake_bundle(1)
    bundle.frame_ids = [5]
    _patch_scores(monkeypatch, {0: _clipped(5)})
    selection = select_canonical_frame(cfg, sam3=None, bundle=bundle)
    write_selection(cfg, selection)

    payload = load_selection(cfg)
    assert payload["decided_by"] == "sole_candidate"
    assert payload["n_candidates"] == 1
    assert payload["mode"] == "heuristic"
    assert payload["scores"][0]["eligible"] is False
    assert resolve_img_idx(cfg) == 5


def test_a_real_lone_frame_with_objects_at_the_edge_is_kept(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("open3d")
    pytest.importorskip("scipy")

    # The geometry of test_score_frame_rejects_a_frame_whose_objects_run_off_the_edge, which fails
    # the clipping gate and no other.
    H, W = 120, 160
    K = np.array([[200.0, 0.0, W / 2], [0.0, 200.0, H / 2], [0.0, 0.0, 1.0]])
    depth = np.full((H, W), 2.0, dtype=np.float32)
    depth[40:70, :40] = 1.8
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    class _StubSAM3:
        def predict_segmentation(self, pil_img, text_prompt):
            return np.ones((1, 1, H, W), dtype=bool), np.array([[0, 0, W, H]]), np.array([0.99])

    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": False}
    selection = select_canonical_frame(cfg, _StubSAM3(), FrameBundle([rgb], [depth], [K], [None]))
    assert selection.decided_by == "sole_candidate"
    assert selection.selected_idx == 0
    assert selection.scores[0].clipped_frac == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Pinned frames are recorded too
# --------------------------------------------------------------------------------------

def test_a_pinned_selection_has_the_same_payload_keys_as_an_automatic_one(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": False}
    _patch_scores(monkeypatch, {i: _passing(i, object_coverage=0.004 * (i + 1)) for i in range(4)})
    automatic = select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle()).to_payload()
    pinned = frame_selection.pinned_selection(2).to_payload()
    assert set(pinned) == set(automatic)
    assert pinned["selected_idx"] == 2


def test_a_pinned_frame_is_what_automatic_stages_resolve(tmp_path):
    cfg = make_cfg(tmp_path)  # img_idx stays auto
    write_selection(cfg, frame_selection.pinned_selection(4))
    payload = load_selection(cfg)
    assert payload["mode"] == "pinned"
    assert payload["decided_by"] == "pinned"
    assert resolve_img_idx(cfg) == 4


def test_pinning_replaces_an_earlier_automatic_selection_on_disk(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": False}
    _patch_scores(monkeypatch, {i: _passing(i, object_coverage=0.004 * (i + 1)) for i in range(4)})
    write_selection(cfg, select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle()))
    assert resolve_img_idx(cfg) == 3

    write_selection(cfg, frame_selection.pinned_selection(4))
    assert resolve_img_idx(cfg) == 4
    assert load_selection(cfg)["decided_by"] == "pinned"


def test_stage3_writes_the_selection_on_both_the_automatic_and_the_pinned_path():
    # The stage needs GPU models, so only its wiring is checked: which calls sit under the auto test.
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("scripts/pipeline/A_reconstruction/stages/3_segment_ground_plane.py").read_text())

    def called_names(nodes):
        return {
            node.func.id
            for root in nodes
            for node in ast.walk(root)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    branches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and "is_auto_img_idx" in called_names([node.test])
    ]
    assert len(branches) == 1
    branch = branches[0]
    assert {"select_canonical_frame", "write_selection"} <= called_names(branch.body)
    assert {"pinned_selection", "write_selection"} <= called_names(branch.orelse)
    # The frame written on the pinned path is the one the stage pinned, not a constant.
    writes = [
        node for root in branch.orelse for node in ast.walk(root)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "write_selection"
    ]
    assert len(writes) == 1
    frame = writes[0].args[1]
    assert isinstance(frame, ast.Call) and isinstance(frame.func, ast.Name) and frame.func.id == "pinned_selection"
    assert len(frame.args) == 1 and isinstance(frame.args[0], ast.Name) and frame.args[0].id == "img_idx"


# --------------------------------------------------------------------------------------
# The shared tail, as the several-candidate path uses it
# --------------------------------------------------------------------------------------

def test_the_multi_candidate_path_keeps_the_vlm_shortlist_note_and_counts(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "hybrid", "vlm_top_k": 3, "write_debug_sheet": False}
    _patch_scores(monkeypatch, {i: _score(i, object_coverage=0.004 * (i + 1)) for i in range(4)})

    def fake_refine(cfg_, bundle_, shortlist, sel_cfg_):
        return shortlist[-1], "vlm picked option 3"

    selection = select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle(), refine_fn=fake_refine)
    assert selection.mode == "hybrid"
    assert selection.n_candidates == 4
    assert selection.vlm_shortlist == [3, 2, 1]
    assert selection.vlm_note == "vlm picked option 3"
    assert len(selection.scores) == 4
    assert selection.decided_by == "vlm"
    assert selection.selected_idx == 1


def test_the_multi_candidate_path_still_writes_the_debug_sheet(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    cfg.s3_ground.frame_selection = {"mode": "heuristic", "write_debug_sheet": True}
    _patch_scores(monkeypatch, {i: _score(i, object_coverage=0.004 * (i + 1)) for i in range(4)})
    selection = select_canonical_frame(cfg, sam3=None, bundle=_fake_bundle())
    assert (tmp_path / "frame_selection.png").is_file()
    assert selection.n_candidates == 4
    assert selection.mode == "heuristic"
