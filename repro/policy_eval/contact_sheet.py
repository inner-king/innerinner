# SPDX-License-Identifier: Apache-2.0
"""Contact sheet of eval episode videos: a few frames per episode, one row per episode.

1_eval_policy_og_scene.py saves each episode as <results>/videos/episode_<i>_<success|fail>.mp4, the two
exterior cameras and the wrist camera side by side. This puts the first, middle and last frame of each
(or of the given) episodes in one JPEG, labelled, to check what the policy did and that a success is one.

    python repro/policy_eval/contact_sheet.py <results dir> out.jpg [--episodes 0 3 7] [--frames 3]
"""
import argparse
import glob
import os
import re

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("out")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--width", type=int, default=1440, help="width of each row")
    args = ap.parse_args()

    videos = sorted(glob.glob(os.path.join(args.results_dir, "videos", "episode_*.mp4")))
    rows = []
    for path in videos:
        m = re.match(r"episode_(\d+)_(\w+)\.mp4", os.path.basename(path))
        idx, outcome = int(m.group(1)), m.group(2)
        if args.episodes is not None and idx not in args.episodes:
            continue
        frames = imageio.mimread(path, memtest=False)
        picks = np.linspace(0, len(frames) - 1, args.frames).round().astype(int)
        strip = np.concatenate([frames[i] for i in picks], axis=1)
        img = Image.fromarray(strip)
        img = img.resize((args.width, int(img.height * args.width / img.width)))
        ImageDraw.Draw(img).text((6, 4), f"ep {idx} {outcome} ({len(frames)} frames)", fill=(255, 255, 0))
        rows.append(np.asarray(img))
    if not rows:
        raise SystemExit(f"no episode videos under {args.results_dir}/videos")
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out, quality=85)
    print(f"{len(rows)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
