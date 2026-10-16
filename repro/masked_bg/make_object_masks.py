# SPDX-License-Identifier: Apache-2.0
"""Per-frame foreground-object masks for background splat training (object = 255).

Stands in for the VOID inpainting of the official auto-background flow: instead of erasing the
tabletop objects from the video, their pixels are excluded from the splat's photometric loss, its
seed point cloud and its depth loss. Masks come from SAM3 text prompts (the object phrases stage 5
detected), unioned and dilated.

Usage: python repro/masked_bg/make_object_masks.py <frames_dir> <out_dir> "phrase a" "phrase b" ... [--dilate 14]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]  # repro/<dir>/<this file> -> repository root
sys.path.insert(0, str(REPO))
from simfoundry.models.sam_v3 import SAM3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("frames_dir")
ap.add_argument("out_dir")
ap.add_argument("phrases", nargs="+")
ap.add_argument("--dilate", type=int, default=14)
ap.add_argument("--conf", type=float, default=0.3)
args = ap.parse_args()

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
sam = SAM3(confidence_threshold=args.conf, device="cuda", video=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.dilate + 1, 2 * args.dilate + 1))
frames = sorted(p for p in Path(args.frames_dir).iterdir() if p.suffix == ".png" and p.stem.startswith("frame_"))
coverage = []
for k, fp in enumerate(frames):
    img = Image.open(fp).convert("RGB")
    union = np.zeros((img.height, img.width), dtype=bool)
    for phrase in args.phrases:
        masks, _, scores = sam.predict_segmentation(img, phrase)
        for m in masks:
            union |= np.asarray(m).reshape(img.height, img.width) > 0.5
    m = cv2.dilate(union.astype(np.uint8) * 255, kernel)
    cv2.imwrite(str(out / fp.name), m)
    coverage.append((m > 127).mean())
    if k % 40 == 0:
        print(f"{k}/{len(frames)} {fp.name}: object coverage {coverage[-1] * 100:.1f}%", flush=True)
print(f"wrote {len(frames)} masks to {out}; mean coverage {np.mean(coverage) * 100:.1f}%, "
      f"frames with no object found: {sum(c == 0 for c in coverage)}")
