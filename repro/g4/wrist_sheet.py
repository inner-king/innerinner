# SPDX-License-Identifier: Apache-2.0
"""One sheet comparing splat backgrounds from the robot's hand camera (repro/README.md, chapter 30).

Takes the videos `repro/scripted_pick_place.py --wrist` wrote for several backgrounds of one scene (the same scripted
motion, so a frame index is the same arm pose in each) and lays out, per moment of the motion, the first external
camera next to the hand camera, one pair of columns per background.

    python repro/g4/wrist_sheet.py --out sheet.jpg default=results/wrist_3dgs/cup_in_bowl_default.mp4 \\
        sreg+carve=results/wrist_3dgs/cup_in_bowl_carve.mp4
"""
import argparse

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

H = 240  # tile height


def tile(frame, label):
    im = Image.fromarray(frame)
    im = im.resize((int(im.width * H / im.height), H))
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    draw.text((5, 3), label, fill=(255, 255, 0))
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="label=<demo>.mp4 (its <demo>.wrist.mp4 is read too)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--moments", type=float, nargs="+", default=[0.08, 0.25, 0.42, 0.58, 0.75, 0.92],
                    help="fractions of the motion to show, one row each")
    args = ap.parse_args()
    cols = []
    for item in args.videos:
        label, path = item.split("=", 1)
        ext = iio.imread(path)
        wrist = iio.imread(path[: -len(".mp4")] + ".wrist.mp4")
        n = min(len(ext), len(wrist))
        # The external video holds the cameras side by side at 16:9 each; keep the first.
        ncams = max(1, round(ext.shape[2] / (ext.shape[1] * 16 / 9)))
        first = ext[:, :, : ext.shape[2] // ncams]
        cols.append([(tile(first[int(f * (n - 1))], f"{label}: external, step {int(f * (n - 1))}"),
                      tile(wrist[int(f * (n - 1))], f"{label}: hand camera")) for f in args.moments])
    rows = []
    for k in range(len(args.moments)):
        ims = [im for col in cols for im in col[k]]
        row = Image.new("RGB", (sum(im.width for im in ims), H))
        x = 0
        for im in ims:
            row.paste(im, (x, 0))
            x += im.width
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, H * len(rows)))
    for k, row in enumerate(rows):
        sheet.paste(row, (0, k * H))
    sheet.save(args.out, quality=88)
    print(f"saved {args.out} {sheet.size}")


if __name__ == "__main__":
    main()
