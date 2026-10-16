# SPDX-License-Identifier: Apache-2.0
"""Three-panel video: real input video | reconstructed objects turning 360 deg | simulation rollout.

Left: the real input video, then the anchor frame stage 3 picked (the camera the reconstruction is
anchored to). Middle: each reconstructed object on a turntable (turntable.py), looping, labelled with its
category and reconstructed size. Right: the reconstructed scene from that same anchor camera, first at rest
while the input video plays, then the robot rollout.

The input video is handheld, so the left and right views coincide only at the anchor frame; that is
why the left panel freezes on it during the rollout.

Usage: python repro/compose_real_vs_sim.py <scene> <input.mp4> <sim_realview.mp4> <turntable.mp4> <out.mp4> <title> [sim note]
"""
import glob
import json
import sys
from pathlib import Path

import imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

scene, video_in, sim_in, turn_in, out, title = sys.argv[1:7]
sim_note = sys.argv[7] if len(sys.argv) > 7 else None      # e.g. which background the sim panel shows
DATA = str(Path(__file__).resolve().parents[1] / "Data")  # repro/<this file> -> repository root
FPS = 15
W, H = 640, 360

idx = json.load(open(f"{DATA}/{scene}/s3_ground/frame_selection.json"))["selected_idx"]
# The subsampled frames DA3 ran on (15 for the default recipe, more for a splat-prep run).
n_da = len(np.load(f"{DATA}/{scene}/s2_da/da/exports/npz/results.npz")["extrinsics"])
anchor_path = sorted(glob.glob(f"{DATA}/{scene}/s1_video/frames_subsampled_{n_da}/*.png"))[idx]
anchor_no = int(anchor_path.split("_")[-1].split(".")[0])
turn_index = json.load(open(Path(turn_in).with_suffix(".json")))

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    small = ImageFont.truetype("DejaVuSans.ttf", 14)
except OSError:
    font = small = ImageFont.load_default()


def panel(frame, label, sub=None):
    im = Image.fromarray(np.asarray(frame)[:, :, :3]).convert("RGB").resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 26 if sub is None else 46], fill=(0, 0, 0))
    d.text((10, 4), label, font=font, fill=(255, 255, 255))
    if sub:
        d.text((10, 26), sub, font=small, fill=(205, 205, 205))
    return np.asarray(im)


meta = iio.immeta(video_in)
step = max(1, round(meta.get("fps", 30) / FPS))
real = [f for i, f in enumerate(iio.imiter(video_in)) if i % step == 0]
sim = list(iio.imiter(sim_in))
turn = list(iio.imiter(turn_in))
anchor = np.asarray(Image.open(anchor_path).convert("RGB"))


def turntable_panel(t):
    k = t % len(turn)
    entry = next(e for e in turn_index if e["start"] <= k < e["end"])
    name = entry["category"].replace("_", " ")
    dims = " x ".join(f"{v:g}" for v in entry["size_cm"])
    return panel(turn[k], f"Reconstructed object, 360 deg: {name}", f"{dims} cm  (Hunyuan3D-2.1 mesh, sim renderer)")


# A wider simulation lens than the real camera: outline the part the real frame covers.
cam_meta = Path(sim_in).with_suffix(".json")
frac = None
if cam_meta.exists():
    cm = json.load(open(cam_meta))
    if cm["hfov_deg"] > cm["anchor_hfov_deg"] + 0.5:
        frac = np.tan(np.radians(cm["anchor_hfov_deg"]) / 2) / np.tan(np.radians(cm["hfov_deg"]) / 2)


def outlined(frame):
    if frac is None:
        return frame
    im = Image.fromarray(np.asarray(frame)[:, :, :3]).convert("RGB")
    w, h = im.size
    x0, y0 = w * (1 - frac) / 2, h * (1 - frac) / 2
    d = ImageDraw.Draw(im)
    d.rectangle([x0, y0, w - x0, h - y0], outline=(255, 200, 0), width=max(2, w // 320))
    d.text((x0 + 8, h - y0 - 30), "real camera view", font=font, fill=(255, 200, 0))
    return np.asarray(im)


sim = [outlined(f) for f in sim]
frames = []
sub = f"camera of input frame {anchor_no}" + (f", wider lens ({cm['hfov_deg']:.0f} vs {cm['anchor_hfov_deg']:.0f} deg)" if frac else "")
if sim_note:
    sub = f"{sub}; {sim_note}"
sim_rest = panel(sim[0], "Simulation: reconstructed scene", sub)
for f in real:
    t = len(frames)
    frames.append(np.concatenate([panel(f, "Real: input video", "handheld phone video, no robot"),
                                  turntable_panel(t), sim_rest], axis=1))
real_anchor = panel(anchor, f"Real: input frame {anchor_no}", "the view the reconstruction is anchored to")
for _ in range(FPS):
    t = len(frames)
    frames.append(np.concatenate([real_anchor, turntable_panel(t), sim_rest], axis=1))
for f in sim:
    t = len(frames)
    frames.append(np.concatenate([real_anchor, turntable_panel(t),
                                  panel(f, f"Simulation: Franka, {title}", "scripted pick-and-place; task goal predicate met"
                                        + (f"; {sim_note}" if sim_note else ""))],
                                 axis=1))
imageio.mimwrite(out, frames, fps=FPS, macro_block_size=2, quality=7)
print(f"wrote {out}: {len(frames)} frames ({len(frames) / FPS:.0f} s), {frames[0].shape[1]}x{frames[0].shape[0]}")
