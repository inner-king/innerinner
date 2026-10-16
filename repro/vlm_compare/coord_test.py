# SPDX-License-Identifier: Apache-2.0
"""Which box / point order does an Ollama VLM write, and does it follow a prompt that asks for another?

The shim pins Qwen to [x1, y1, x2, y2] / [x, y] (0-1000) with SYSTEM_PROMPT and swaps the answer into
Gemini's [y, x] order, so a model that obeys a prompt's "[ymin, xmin, ymax, xmax]" instead gets its
boxes transposed. This asks for boxes and points on two synthetic images (landscape and portrait,
shapes at known places) and on a real stage-5 frame (SAM3 reference boxes), with and without the
shim's system message, sending the shim's own image preprocessing straight to Ollama (raw answers,
before any swap). Each answer is scored by geometry, not by its labels: under each reading the
boxes are matched to the reference they overlap most, and the reading that fits best is reported.

Usage: python repro/vlm_compare/coord_test.py <out_dir> [--configs qwen3-vl:8b-instruct,qwen3.5:9b:false,qwen3.5:9b]
       [--extra-options '{"presence_penalty": 0}'] [--data-root DIR]
A config is <model>[:<think>] with think true/false; without it no think field is sent.
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simfoundry.models import local_backends as lb  # noqa: E402
from simfoundry.utils.prompt_utils import parse_json_response, prompt_object_setlist_specific_floorplane_v3  # noqa: E402

SHAPES = "the red circle, the blue square and the green triangle"
PROMPTS = {
    "box_yx": f'Detect {SHAPES} in the image. Return a JSON list where each entry is {{"label": <name>, "box_2d": [ymin, xmin, ymax, xmax]}} with coordinates normalized to 0-1000.',
    "box_native": f"Locate {SHAPES} and output their bbox coordinates in JSON format.",
    "pt_yx": f'Point to the center of {SHAPES}. Return a JSON list where each entry is {{"label": <name>, "point": [y, x]}} with coordinates normalized to 0-1000.',
    "pt_native": f"Point to {SHAPES} and output their point coordinates in JSON format.",
}


def synth(w, h, shapes):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    gt = {}
    for name, kind, geom, color in shapes:
        if kind == "circle":
            cx, cy, r = geom
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
            gt[name] = [cx - r, cy - r, cx + r, cy + r]
        elif kind == "rect":
            d.rectangle(geom, fill=color)
            gt[name] = list(geom)
        else:
            d.polygon(geom, fill=color)
            gt[name] = [min(p[0] for p in geom), min(p[1] for p in geom), max(p[0] for p in geom), max(p[1] for p in geom)]
    return im, gt


def images(data_root):
    land = synth(1280, 720, [("red circle", "circle", (220, 520, 70), (220, 30, 30)),
                             ("blue square", "rect", (900, 120, 1060, 280), (30, 60, 220)),
                             ("green triangle", "poly", [(560, 560), (700, 560), (630, 420)], (30, 170, 60))])
    port = synth(720, 1280, [("red circle", "circle", (180, 1000, 70), (220, 30, 30)),
                             ("blue square", "rect", (450, 150, 610, 310), (30, 60, 220)),
                             ("green triangle", "poly", [(300, 700), (440, 700), (370, 560)], (30, 170, 60))])
    s5 = Path(data_root) / "put_marker_in_cup/s5_scene"
    real = Image.open(s5 / "source_padded_resized_upsampled.png").convert("RGB")
    gt_real = {}
    for i in range(3):
        m = np.array(Image.open(s5 / "removal_mask" / f"iter_{i}.png")) > 0
        ys, xs = np.nonzero(m)
        gt_real[json.load(open(s5 / "obj_cat_list" / f"iter_{i}.json"))["removed_obj_phrase"]] = \
            [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return {"landscape": land, "portrait": port, "real_s5": (real, gt_real)}


def chat(model, think, system, prompt, image, extra_options):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    msgs = ([{"role": "system", "content": lb.SYSTEM_PROMPT}] if system else []) + [
        {"role": "user", "content": prompt, "images": [lb._resize_for_vlm(buf.getvalue())]}]
    body = {"model": model, "stream": False, "keep_alive": "120s", "messages": msgs,
            "options": {"temperature": 0.0, "seed": 0, "num_predict": lb.OLLAMA_MAX_TOKENS, "num_ctx": lb.OLLAMA_NUM_CTX,
                        **extra_options}}
    if think is not None:
        body["think"] = think
    t0 = time.time()
    out = requests.post(f"{lb.OLLAMA_URL}/api/chat", json=body, timeout=3600).json()
    return out, time.time() - t0


def coordinates(text):
    """Numbers under any box / point key, in answer order."""
    try:
        obj = parse_json_response(text)
    except Exception:
        return []
    items = obj if isinstance(obj, list) else obj.get("detections", [obj]) if isinstance(obj, dict) else []
    found = []
    for it in items:
        if not isinstance(it, dict):
            continue
        for k in ("box_2d", "bbox_2d", "bbox", "box", "point", "point_2d"):
            v = it.get(k)
            if isinstance(v, (list, tuple)) and len(v) in (2, 4) and all(isinstance(x, (int, float)) for x in v):
                found.append([float(x) for x in v])
                break
    return found


def iou(a, b):
    a = [min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])]
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_reading(coords, gt, w, h):
    """(reading, per-answer fit): IoU for boxes, distance to the reference centre (px) for points."""
    if not coords:
        return None, []
    refs = list(gt.values())
    if len(coords[0]) == 4:
        readings = {"xyxy": lambda v: [v[0] / 1000 * w, v[1] / 1000 * h, v[2] / 1000 * w, v[3] / 1000 * h],
                    "yxyx": lambda v: [v[1] / 1000 * w, v[0] / 1000 * h, v[3] / 1000 * w, v[2] / 1000 * h]}
        fits = {k: [max(iou(f(v), g) for g in refs) for v in coords] for k, f in readings.items()}
        best = max(fits, key=lambda k: sum(fits[k]))
    else:
        readings = {"xy": lambda v: (v[0] / 1000 * w, v[1] / 1000 * h), "yx": lambda v: (v[1] / 1000 * w, v[0] / 1000 * h)}
        fits = {k: [min(float(np.hypot(f(v)[0] - (g[0] + g[2]) / 2, f(v)[1] - (g[1] + g[3]) / 2)) for g in refs) for v in coords]
                for k, f in readings.items()}
        best = min(fits, key=lambda k: sum(fits[k]))
    return best, [round(x, 3) for x in fits[best]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--configs", default="qwen3-vl:8b-instruct,qwen3.5:9b:false,qwen3.5:9b")
    ap.add_argument("--extra-options", default="{}", help="JSON merged into Ollama's options, e.g. a sampler override")
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2] / "Data"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extra = json.loads(args.extra_options)
    imgs = images(args.data_root)
    imgs["landscape"][0].save(out / "synth_landscape.png")
    imgs["portrait"][0].save(out / "synth_portrait.png")
    results = []
    for cfg in args.configs.split(","):
        model, _, think_s = cfg.rpartition(":") if cfg.endswith((":true", ":false")) else (cfg, "", "")
        think = {"true": True, "false": False}.get(think_s)
        cases = []
        for name in ("landscape", "portrait"):
            for pname, prompt in PROMPTS.items():
                for system in (True, False):
                    cases.append((name, pname, prompt, system))
        for system in (True, False):
            cases.append(("real_s5", "pipeline_floorplane_v3", prompt_object_setlist_specific_floorplane_v3("desk"), system))
        if think is None and "qwen3.5" in model:
            # Thinking is slow: only the cases the shim relies on (its system message, an order request).
            cases = [c for c in cases if c[3] and (c[1] in ("box_yx", "pt_yx") and c[0] == "landscape" or c[0] == "real_s5")]
        for name, pname, prompt, system in cases:
            img, gt = imgs[name]
            ans, dt = chat(model, think, system, prompt, img, extra)
            text = ans.get("message", {}).get("content", "")
            reading, fit = best_reading(coordinates(text), gt, *img.size)
            rec = {"model": model, "think": think, "image": name, "prompt": pname, "system_prompt": system,
                   "extra_options": extra, "seconds": round(dt, 1), "gen_tokens": ans.get("eval_count"),
                   "done": ans.get("done_reason"), "thinking_chars": len(ans.get("message", {}).get("thinking") or ""),
                   "text": text, "reading": reading, "fit": fit}
            results.append(rec)
            print(f"{model:22s} think={str(think):5s} {name:9s} {pname:22s} sys={int(system)} {dt:6.1f}s "
                  f"gen={ans.get('eval_count')} done={ans.get('done_reason')} reading={reading} fit={fit}", flush=True)
            json.dump({"gt": {k: v[1] for k, v in imgs.items()}, "results": results}, open(out / "coord_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
