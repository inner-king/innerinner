# SPDX-License-Identifier: Apache-2.0
"""Score replay_weak_calls.py results: recall, false positives and box IoU against the SAM3 reference
per detection call, the frame each selection call picked, and what every call cost (latency,
tokens, MAX_TOKENS retries, device memory it added).

Usage: python repro/vlm_compare/grade_weak_calls.py <out_dir> <label>=<results.jsonl> [...]
Writes <out_dir>/scores.json, <out_dir>/scores.md and box sheets <out_dir>/sheets/*.jpg
(green: SAM3 reference, magenta: VLM box on a reference object, orange: any other VLM box).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(sys.argv[1])
RUNS = [a.split("=", 1) for a in sys.argv[2:]]
(OUT / "sheets").mkdir(parents=True, exist_ok=True)

CATS = {  # checked in this order, so "pen" cannot steal "scissors"/"cup"
    "marker": [("scissors", ["scissor", "shear"]), ("cup", ["cup", "mug", "glass", "tumbler"]),
               ("marker", ["marker", "pen", "highlighter"])],
    "bowl": [("bowl", ["bowl"]), ("cup", ["cup", "mug", "glass"]), ("pen", ["pen", "marker"])],
}
# Per candidate sheet, the option a careful reader picks (every object fully visible, then the largest)
# and the runner-up, judged by eye (repro/README.md, chapter 16); the heuristic ranks them the same way.
S3_BEST = {"put_marker_in_cup": 5, "put_marker_in_cup_v2": 5, "put_marker_in_cup_bg": 77}
S3_OK = {"put_marker_in_cup": {4}, "put_marker_in_cup_v2": {4}, "put_marker_in_cup_bg": {70}}


def cat_of(name, scene):
    table = CATS["bowl" if "bowl" in scene else "marker"]
    n = (name or "").lower()
    for cat, keys in table:
        if any(k in n for k in keys):
            return cat
    return None


def gemini_box_to_xyxy(b, w, h):
    """convert_gemini_box2d_to_xyxy from 5_decompose_scene.py."""
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        y1n, x1n, y2n, x2n = [float(v) for v in b]
    except (TypeError, ValueError):
        return None
    y1, x1, y2, x2 = int(y1n / 1000 * h), int(x1n / 1000 * w), int(y2n / 1000 * h), int(x2n / 1000 * w)
    x0, x3 = sorted((x1, x2)); y0, y3 = sorted((y1, y2))
    x0, x3 = np.clip([x0, x3], 0, w - 1); y0, y3 = np.clip([y0, y3], 0, h - 1)
    if x3 <= x0 or y3 <= y0:
        return None
    return [float(x0), float(y0), float(x3), float(y3)]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def score_detection(rec):
    case = rec["case"]
    scene = case["scene"]
    img = Image.open(case["image"])
    w, h = img.size
    gt = {cat_of(g["phrase"], scene): g["box"] for g in case["gt"]}
    out = {"n_gt": len(gt), "error": rec.get("error")}
    if rec.get("error"):
        out.update(found=[], missed=sorted(gt), extras=[], ious={}, center_in={})
        return out, []
    objects = rec.get("objects") or []
    dets = rec.get("detections") or []
    obj_cats = [cat_of(o, scene) for o in objects]
    found = sorted({c for c in obj_cats if c in gt})
    extras = [o.split(" / ")[0] for o, c in zip(objects, obj_cats) if c not in gt]
    dup = len([c for c in obj_cats if c in gt]) - len(found)
    ious, center_in, drawn = {}, {}, []
    for dct in dets:
        c = cat_of(dct.get("name"), scene)
        bx = gemini_box_to_xyxy(dct.get("box_2d"), w, h)
        drawn.append((dct.get("name", "?").split(" / ")[0], bx, c in gt))
        if c in gt and bx is not None and gt[c] is not None:
            v = iou(bx, gt[c])
            if v >= ious.get(c, -1):
                ious[c] = round(v, 3)
                cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                g = gt[c]
                center_in[c] = bool(g[0] <= cx <= g[2] and g[1] <= cy <= g[3])
    out.update(found=found, missed=sorted(set(gt) - set(found)), extras=extras, duplicates=dup, ious=ious, center_in=center_in)
    return out, drawn


def cost(rec):
    a = rec.get("attempts") or []
    return {
        "wall_s": rec["wall_s"], "attempts": len(a), "max_tokens_hits": sum(1 for x in a if x.get("done_reason") == "length"),
        "prompt_tokens": a[0].get("prompt_tokens") if a else None, "gen_tokens": sum((x.get("gen_tokens") or 0) for x in a),
        "thinking_chars": sum(len(x.get("thinking") or "") for x in a), "load_s": round(sum(x.get("load_s", 0) for x in a), 1),
        "prompt_s": round(sum(x.get("prompt_s", 0) for x in a), 1), "gen_s": round(sum(x.get("gen_s", 0) for x in a), 1),
        "vram_peak_mib": rec["vram_peak_mib"], "vram_base_mib": rec["vram_base_mib"],
        # Device memory the call added: other processes (desktop ~76 MiB, another session's job) are in base.
        "vram_model_mib": rec["vram_peak_mib"] - rec["vram_base_mib"],
        "ollama_ps": rec.get("ollama_ps"),
    }


all_scores = {}
sheets = {}
for label, path in RUNS:
    recs = [json.loads(l) for l in open(path)]
    rows = []
    for rec in recs:
        case = rec["case"]
        row = {"id": case["id"], "stage": case["stage"], "scene": case["scene"], **cost(rec)}
        if case["stage"] == 3:
            sel = rec.get("selected_frame")
            row.update(selected=sel, note=rec.get("note"),
                       grade=("fail->heuristic" if sel is None else "best" if sel == S3_BEST[case["scene"]]
                              else "ok" if sel in S3_OK[case["scene"]] else "poor"),
                       error=rec.get("error"))
        else:
            sc, drawn = score_detection(rec)
            row.update(sc)
            sheets.setdefault(case["id"], {})[label] = (case["image"], case["gt"], drawn)
        rows.append(row)
    all_scores[label] = rows

json.dump(all_scores, open(OUT / "scores.json", "w"), indent=1)

# ------------------------------------------------------------------ box sheets (one per detection case)
for cid, per in sheets.items():
    tiles = []
    for label, (img_path, gt, drawn) in per.items():
        im = Image.open(img_path).convert("RGB")
        d = ImageDraw.Draw(im)
        for g in gt:
            if g["box"] is not None:
                d.rectangle(g["box"], outline=(0, 255, 0), width=3)
        for name, bx, ok in drawn:
            if bx is None:
                continue
            col = (255, 0, 255) if ok else (255, 140, 0)
            d.rectangle(bx, outline=col, width=3)
            d.text((bx[0] + 3, max(0, bx[1] - 12)), name, fill=col)
        d.rectangle([0, 0, 520, 22], fill=(0, 0, 0))
        d.text((6, 5), f"{label}  (green: SAM3 ref, magenta: VLM, orange: VLM extra)", fill=(255, 255, 255))
        scale = 900 / im.width
        tiles.append(im.resize((900, int(im.height * scale))))
    sheet = Image.new("RGB", (900 * len(tiles), max(t.height for t in tiles)), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, (900 * i, 0))
    sheet.save(OUT / "sheets" / f"{cid}.jpg", quality=85)

# ------------------------------------------------------------------ markdown summary
lines = []
for label, rows in all_scores.items():
    lines.append(f"\n### {label}\n")
    lines.append("| call | result | box IoU | wall s | load/prompt/gen s | prompt tok | gen tok | think chars | MAX_TOKENS | VRAM added MiB (base) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["stage"] == 3:
            res = f"frame {r['selected']} ({r['grade']}; {r['note']})"
            ious = ""
        else:
            res = "ERROR " + r["error"][:60] if r.get("error") else \
                f"found {len(r['found'])}/{r['n_gt']} {r['found']}" + (f" missed {r['missed']}" if r['missed'] else "") + \
                (f" extra {r['extras']}" if r['extras'] else "") + (f" dup {r['duplicates']}" if r.get('duplicates') else "")
            ious = ", ".join(f"{k} {v:.2f}" for k, v in sorted(r.get("ious", {}).items()))
        lines.append(f"| {r['id']} | {res} | {ious} | {r['wall_s']:.1f} | {r['load_s']}/{r['prompt_s']}/{r['gen_s']} | {r['prompt_tokens']} | "
                     f"{r['gen_tokens']} | {r['thinking_chars']} | {r['max_tokens_hits']} | {r['vram_model_mib']} ({r['vram_base_mib']}) |")
lines.append("\n### totals\n")
lines.append("| config | calls | wall s total | gen tok total | MAX_TOKENS hits | s1 recall | s5 recall | s5 extras | final-empty ok | s5 IoU mean (marker/cup/scissors) | max VRAM added MiB |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
for label, rows in all_scores.items():
    det = [r for r in rows if r["stage"] != 3]
    s1 = [r for r in det if r["stage"] == 1]
    s5 = [r for r in det if r["stage"] == 5]
    def recall(rs):
        n = sum(r["n_gt"] for r in rs); f = sum(len(r.get("found", [])) for r in rs)
        return f"{f}/{n}"
    empty = [r for r in s5 if r["n_gt"] == 0]
    empty_ok = sum(1 for r in empty if not r.get("error") and not r.get("extras"))
    ious = {}
    for r in s5:
        for k, v in (r.get("ious") or {}).items():
            ious.setdefault(k, []).append(v)
    iou_s = "/".join(f"{np.mean(ious[k]):.2f}" if k in ious else "-" for k in ("marker", "cup", "scissors"))
    lines.append(f"| {label} | {len(rows)} | {sum(r['wall_s'] for r in rows):.0f} | {sum(r['gen_tokens'] for r in rows)} | "
                 f"{sum(r['max_tokens_hits'] for r in rows)} | {recall(s1)} | {recall(s5)} | {sum(len(r.get('extras') or []) for r in s5)} | "
                 f"{empty_ok}/{len(empty)} | {iou_s} | {max(r['vram_model_mib'] for r in rows)} |")
open(OUT / "scores.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
