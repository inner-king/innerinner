# SPDX-License-Identifier: Apache-2.0
"""How bright the images the policies see are, against the real DROID images they were trained on (README 30.18).

Reads the eval stage's videos (per frame: first exterior camera | second | wrist, each 320x180 as the policy's
images scaled down) and reports each camera's per-channel mean and std, and the gamma that brings our mean
luminance to DROID-100's. DROID-100's numbers are lerobot/droid_100's meta/stats.json (100 episodes, 10,650
frames sampled), which this file copies rather than downloads.

    python repro/policy_eval_ours/frame_stats.py "label=<results dir>/videos/*.mp4" [...]

Videos of runs with s15_eval.image_gamma show the corrected images; the earlier client-side gamma (30.13's first
table) does not appear in them.
"""
import glob
import sys

import imageio.v3 as iio
import numpy as np
from scipy.optimize import brentq

# lerobot/droid_100 meta/stats.json: per-channel mean and std of the RGB images in [0, 1]
DROID = {"ext": (np.array([0.3877, 0.3852, 0.3402]), np.array([0.2646, 0.2624, 0.2610])),
         "wrist": (np.array([0.3857, 0.3745, 0.3104]), np.array([0.2870, 0.2791, 0.2682]))}
LUMA = np.array([0.299, 0.587, 0.114])  # Rec. 601 luma weights
for spec in sys.argv[1:]:
    label, pattern = spec.split("=", 1)
    vids = sorted(glob.glob(pattern))[:12]
    px = {"ext": [], "wrist": []}
    for v in vids:
        frames = iio.imread(v, plugin="pyav")          # (T, 192, 960, 3)
        frames = frames[::4].astype(np.float32) / 255.0
        px["ext"].append(frames[:, :, 0:320].reshape(-1, 3)[::7])
        px["wrist"].append(frames[:, :, 640:960].reshape(-1, 3)[::7])
    out = []
    for cam in ("ext", "wrist"):
        x = np.concatenate(px[cam])
        m, s = x.mean(0), x.std(0)
        target = DROID[cam][0] @ LUMA
        g = brentq(lambda g: (x ** g).mean(0) @ LUMA - target, 0.2, 5.0)
        out.append(f"{cam}: mean {np.round(m, 3)} std {np.round(s, 3)} luma {m @ LUMA:.3f} (DROID {target:.3f}) -> matching gamma {g:.2f}")
    print(f"{label} [{len(vids)} videos]\n   " + "\n   ".join(out))
