# SPDX-License-Identifier: Apache-2.0
"""Replay the VLM calls repro/README.md found weak, on saved inputs, through ``Gemini(...)`` and the shim.

  s1   auto_bg quadmask object proposal on keyframe 0 of put_marker_in_cup_bg (and, with
       --s1-frames, the same call on more frames of that video, scored on names only)
  s3   canonical-frame selection over the saved 4-option candidate sheets (refine_with_vlm itself)
  s5   object detection on the decomposition frame and on every post-removal image; the last,
       empty-desk image is asked three times, as the loop does before it stops

Same prompts, images and call arguments as the stages; only the Ollama model differs, set in the
environment (SIMFOUNDRY_OLLAMA_MODEL, SIMFOUNDRY_OLLAMA_THINK). Reference boxes are the original
runs' SAM3 masks. Run with the response cache off. Results: <out_dir>/results.jsonl, graded by
grade_weak_calls.py.

Usage: python repro/vlm_compare/replay_weak_calls.py <out_dir> [--only s1,s3_put_marker_in_cup] [--s1-frames 10,20] [--data-root DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402  (puts the repository root on sys.path)

from omegaconf import OmegaConf  # noqa: E402

from simfoundry.models.vlm import Gemini  # noqa: E402
from simfoundry.pipeline import frame_selection as fsel  # noqa: E402
from simfoundry.utils.prompt_utils import parse_json_response, prompt_object_setlist_specific_floorplane_v3  # noqa: E402

# s5_scene.detection_model == s3_ground.detection_model; run_auto_bg.sh gives step 1 the same one.
DETECTION_MODEL = "gemini-3.1-pro-preview"


def mask_box(path):
    m = np.array(Image.open(path)) > 0
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def s5_refs(scene):
    """Object removed at each iteration with its SAM3 mask box; objects do not move between iterations."""
    s5 = bc.DATA / scene / "s5_scene"
    return [{"phrase": json.load(open(s5 / "obj_cat_list" / f"iter_{i}.json"))["removed_obj_phrase"],
             "box": mask_box(s5 / "removal_mask" / f"iter_{i}.png")} for i in range(3)]


def floor_category(scene):
    idx = json.load(open(bc.DATA / scene / "s5_scene" / "decomposition_frame.json"))["img_idx"]
    return json.load(open(bc.DATA / scene / "s3_ground" / f"image_{idx}_floor_info.json"))["floor_category"]


def build_cases(s1_frames):
    cases = []
    kf_dir = bc.DATA / "put_marker_in_cup_bg/s1_video/frames_subsampled_160"
    frames = sorted(kf_dir.glob("*.png"))
    masks = bc.DATA / "put_marker_in_cup_bg/auto_bg/void/input/keyframe_masks_debug"
    cases.append({"id": "s1_keyframe0", "stage": 1, "scene": "put_marker_in_cup_bg", "image": str(frames[0]),
                  "floor": "desk, table, or counter",  # run_auto_bg.sh's default floor
                  "gt": [{"phrase": "red marker", "box": mask_box(masks / "obj00_red_marker.png")},
                         {"phrase": "white cup", "box": mask_box(masks / "obj01_white_cup.png")},
                         {"phrase": "blue scissors", "box": mask_box(masks / "obj02_blue_scissors.png")}]})
    for i in s1_frames:
        cases.append({"id": f"s1x_f{i:03d}", "stage": 1, "scene": "put_marker_in_cup_bg", "image": str(frames[i]),
                      "floor": "desk, table, or counter",
                      "gt": [{"phrase": p, "box": None} for p in ("red marker", "white cup", "blue scissors")]})
    for scene in ("put_marker_in_cup", "put_marker_in_cup_v2", "put_marker_in_cup_bg"):
        cand = sorted((bc.DATA / scene / "s3_ground/frame_selection_candidates").glob("option_*_frame_*.png"),
                      key=lambda p: int(p.name.split("_")[1]))
        cases.append({"id": f"s3_{scene}", "stage": 3, "scene": scene, "images": [str(p) for p in cand],
                      "shortlist": [int(p.stem.split("_")[-1]) for p in cand]})
    for scene in ("put_marker_in_cup", "put_marker_in_cup_v2", "put_cup_in_bowl_v2"):
        refs, s5, floor = s5_refs(scene), bc.DATA / scene / "s5_scene", floor_category(scene)
        cases.append({"id": f"s5_{scene}_src", "stage": 5, "scene": scene, "floor": floor, "gt": refs,
                      "image": str(s5 / "source_padded_resized_upsampled.png")})
        for k in range(3):
            n_ask = 3 if k == 2 else 1
            for rep in range(n_ask):
                cases.append({"id": f"s5_{scene}_post{k}" + (f"_rep{rep}" if n_ask > 1 else ""), "stage": 5,
                              "scene": scene, "floor": floor, "gt": refs[k + 1:],
                              "image": str(s5 / "post_object_removal" / f"iter_{k}.png")})
    return cases


_LAST_TEXT = {}
_get_text = Gemini.get_result_text


def _recording_get_text(self, result):
    _LAST_TEXT["text"] = None
    _LAST_TEXT["text"] = _get_text(self, result)
    return _LAST_TEXT["text"]


Gemini.get_result_text = _recording_get_text


def run_detection(case):
    """The stage-1 / stage-5 call: same prompt and arguments, then the stages' own parsing."""
    vlm = Gemini(project="local", location="global", model=DETECTION_MODEL)
    result = vlm(prompt=prompt_object_setlist_specific_floorplane_v3(case["floor"]), image_paths=case["image"],
                 temperature=0, top_p=0, seed=0, print_results=False)
    try:
        out = {"text": "".join((p.text or "") for chunk in result for p in chunk.candidates[0].content.parts)}
    except Exception:
        out = {"text": None}
    text = vlm.get_result_text(result)  # stage 5 does not catch a refusal: the stage would stop here
    parsed = parse_json_response(text)
    if isinstance(parsed, list):
        parsed = {"detections": parsed}
    objects, detections = parsed.get("objects", []), parsed.get("detections", [])
    if not objects and detections:
        objects = [d.get("name", "") for d in detections if d.get("name")]
    return {**out, "objects": objects, "detections": detections}


def run_frame_select(case, out_dir):
    """refine_with_vlm itself, handed the saved candidate images instead of re-rendering them."""
    fsel._write_vlm_candidate_images = lambda bundle, shortlist, o, max_side: list(case["images"])
    cfg = OmegaConf.create({"gcloud_project": "local", "s3_ground": {"out_dir": str(out_dir / "s3_tmp")}, "visualize": False})
    _LAST_TEXT["text"] = None
    idx, note = fsel.refine_with_vlm(cfg, None, case["shortlist"], {"vlm_max_side": 1024, "vlm_model": DETECTION_MODEL})
    return {"selected_frame": idx, "note": note, "text": _LAST_TEXT.get("text")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--only", default="", help="comma-separated case id prefixes")
    ap.add_argument("--s1-frames", default="", help="extra put_marker_in_cup_bg frame indices for the stage-1 call")
    ap.add_argument("--data-root", default=str(bc.DATA), help="scene outputs (default: <repo>/Data)")
    args = ap.parse_args()
    bc.DATA = Path(args.data_root)
    bc.check_setup()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = build_cases([int(x) for x in args.s1_frames.split(",") if x.strip()])
    if args.only:
        cases = [c for c in cases if any(c["id"].startswith(p) for p in args.only.split(","))]
    print(f"== {bc.lb.OLLAMA_MODEL} think={bc.lb.OLLAMA_THINK!r} keep_alive={bc.lb.OLLAMA_KEEP_ALIVE} "
          f"max_tokens={bc.lb.OLLAMA_MAX_TOKENS} num_ctx={bc.lb.OLLAMA_NUM_CTX}", flush=True)
    bc.warmup()
    for case in cases:
        res, meta = bc.measured(lambda: run_frame_select(case, out) if case["stage"] == 3 else run_detection(case))
        bc.append_jsonl(out / "results.jsonl", {"case": case, **meta, **res})
        tail = (f"frame {res.get('selected_frame')} ({res.get('note')})" if case["stage"] == 3 else
                res.get("error") or [o.split(" / ")[0] for o in res.get("objects", [])])
        print(bc.summary_line(case["id"], meta, tail), flush=True)


if __name__ == "__main__":
    main()
