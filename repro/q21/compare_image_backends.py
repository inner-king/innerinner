# SPDX-License-Identifier: Apache-2.0
"""A/B the two local image backends on the exact inputs and prompts the SimFoundry pipeline sends.

Run once per backend (the backend is read from SIMFOUNDRY_IMAGE_BACKEND at import):
    SIMFOUNDRY_LOCAL_BACKEND=1 SIMFOUNDRY_IMAGE_BACKEND=edit2511 python repro/q21/compare_image_backends.py run edit2511
    SIMFOUNDRY_LOCAL_BACKEND=1 SIMFOUNDRY_IMAGE_BACKEND=qwen21   python repro/q21/compare_image_backends.py run qwen21
    SIMFOUNDRY_LOCAL_BACKEND=1 SIMFOUNDRY_IMAGE_BACKEND=qwen21 SIMFOUNDRY_Q21_UNET=qwen-image-2.1-Q8_0.gguf \
        SIMFOUNDRY_Q21_STEPS=25 python repro/q21/compare_image_backends.py run qwen21base
then score both:
    python repro/q21/compare_image_backends.py score
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = str(Path(__file__).resolve().parents[2])  # repro/<dir>/<this file> -> repository root
sys.path.insert(0, REPO)
OUT = f"{REPO}/results/diagnostics/q21/out"
SCENE = f"{REPO}/Data/put_marker_in_cup"
GREEN = np.array([0, 255, 0])
# Backends run so far: edit2511 (Qwen-Image-Edit-2511 Q3 + Lightning 4 steps), qwen21 (2.1-Turbo Q8, 8 steps),
# qwen21base (2.1 base Q8, 25 steps).
TAGS = [t for t in ("edit2511", "qwen21", "qwen21base") if os.path.exists(f"{OUT}/{t}/timings.json")]

CUP = "white paper cup / paper cup / disposable cup"
SCISSORS = "blue scissors / scissors / cutting scissors"


def cases():
    from simfoundry.utils.prompt_utils import (
        prompt_flux_object_removal_mask, prompt_flux_object_removal_outline, prompt_upsample_image_gemini)
    cousin_prompt = json.load(open(f"{SCENE}/prompt_cousin_structured/iter_0/generation_metadata.json"))[0]["prompt"]
    return {
        # Stage 5 removal: the default first prompt type, which Edit-2511 turned into a glass "ghost".
        "A1_remove_mask_cup": ([f"{SCENE}/s5_scene_maskprompt_failed/pre_object_removal/iter_0.png"],
                               prompt_flux_object_removal_mask(obj_phrase=CUP)),
        # Stage 5 removal with the outline prompt the reported runs switched to.
        "A2_remove_outline_cup": ([f"{SCENE}/s5_scene/pre_object_removal/iter_0.png"],
                                  prompt_flux_object_removal_outline(obj_phrase=CUP)),
        "A3_remove_outline_scissors": ([f"{SCENE}/s5_scene/pre_object_removal/iter_1.png"],
                                       prompt_flux_object_removal_outline(obj_phrase=SCISSORS)),
        # Stage 6: studio re-shoot of a masked RGBA cutout.
        "B1_upsample_marker": ([f"{SCENE}/s6_upsample/padded/iter_2.png"], prompt_upsample_image_gemini()),
        "B2_upsample_scissors": ([f"{SCENE}/s6_upsample/padded/iter_1.png"], prompt_upsample_image_gemini()),
        # Pipeline B step e: cousin image (object cutout + scene frame), asks for a transparent background.
        "C1_cousin_taller_cup": ([f"{SCENE}/s6_upsample/upsampled/iter_0_transparent.png",
                                  f"{SCENE}/s1_video/frames_all/frame_0001.png"], cousin_prompt),
    }


def run(tag):
    from simfoundry.models.vlm import Gemini
    os.makedirs(f"{OUT}/{tag}", exist_ok=True)
    g = Gemini(model="gemini-3-pro-image")
    timings = {}
    for name, (paths, prompt) in cases().items():
        t0 = time.time()
        res = g(prompt, image_paths=paths, temperature=0, top_p=0, seed=0)
        timings[name] = round(time.time() - t0, 1)
        img = g.get_result_images(res)[0]
        img.save(f"{OUT}/{tag}/{name}.png")
        print(f"{tag} {name}: {img.size} {img.mode} {timings[name]}s", flush=True)
    json.dump(timings, open(f"{OUT}/{tag}/timings.json", "w"), indent=1)


def green_mask(a):
    return np.all(np.abs(a.astype(int) - GREEN) <= 50, axis=-1)


def dilate(mask, r):
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, iterations=r)


def score():
    rows = []
    for name, (paths, _) in cases().items():
        if not name.startswith("A"):
            continue
        src = np.array(Image.open(paths[0]).convert("RGB"))
        annot = dilate(green_mask(src), 25)
        h, w = src.shape[:2]
        for tag in TAGS:
            out_img = Image.open(f"{OUT}/{tag}/{name}.png")
            alpha_min = np.array(out_img.convert("RGBA"))[..., 3].min()
            out = np.array(out_img.convert("RGB").resize((w, h), Image.LANCZOS))
            leftover = int((green_mask(out) & annot).sum())
            keep = ~dilate(annot, 15)
            diff = np.abs(out.astype(float) - src.astype(float)).mean(-1)[keep].mean()
            rows.append((name, tag, leftover, diff, alpha_min))
    print(f"{'case':28s} {'backend':9s} {'green px left':>13s} {'mean |Δ| outside (0-255)':>25s} {'min alpha':>9s}")
    for r in rows:
        print(f"{r[0]:28s} {r[1]:9s} {r[2]:13d} {r[3]:25.1f} {r[4]:9d}")
    for tag in TAGS:
        t = json.load(open(f"{OUT}/{tag}/timings.json"))
        print(tag, "timings (s):", t, "mean", round(sum(t.values()) / len(t), 1))


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "score": score}[sys.argv[1]]()
