# SPDX-License-Identifier: Apache-2.0
"""Compare the desk's and the wall's mean colour in our render with other pictures of the same room (README 30.13).

The robot's real exterior-camera frames are not published, so the references are the paper's figures: a photo of
the real setup and the authors' own render of the same room. The cameras differ, so pixels cannot be paired; each
surface is picked by colour over the whole frame instead (desk: orange-brown in the lower part; wall: pale and
unsaturated in the upper left), and its mean compared. A boxed region gave answers that moved with the box.

For each reference and channel this prints gamma = log(reference) / log(ours), the power that would carry our
mean onto the reference's: above 1 the reference is darker than our render.

    python repro/policy_eval_ours/tone_compare.py --ours <render.png> \\
        --ref real_photo=<paper>/figs/imgs/DROID_tasks/MarkerInCup.jpg \\
        --ref authors_render=<paper>/figs/imgs/polaris/SF_stackdishware_frame.jpg --sheet masks.jpg
"""
import argparse

import numpy as np
from PIL import Image


def load(path):
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img).astype(np.float64) / 255.0
    hsv = np.asarray(img.convert("HSV")).astype(np.float64) / 255.0
    return rgb, hsv


def masks(rgb, hsv):
    """Desk and wall pixels, picked by colour and by where in the frame they can be."""
    h, s, v = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    rows, cols = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
    desk = (h > 15) & (h < 45) & (s > 0.25) & (v > 0.15) & (rows > rgb.shape[0] * 0.35)
    wall = (s < 0.12) & (v > 0.55) & (rows < rgb.shape[0] * 0.45) & (cols < rgb.shape[1] * 0.6)
    return desk, wall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, help="our render from the exterior camera")
    ap.add_argument("--ref", action="append", required=True, help="NAME=PATH of a picture to compare with")
    ap.add_argument("--sheet", help="write the frames with the desk (red) and wall (blue) masks overlaid")
    args = ap.parse_args()

    pictures = [("ours", args.ours)] + [tuple(r.split("=", 1)) for r in args.ref]
    means, overlays = {}, []
    for name, path in pictures:
        rgb, hsv = load(path)
        desk, wall = masks(rgb, hsv)
        means[name] = (rgb[desk].mean(0), rgb[wall].mean(0))
        print(f"{name:16s} desk {means[name][0].round(3)} ({desk.mean():.0%} of pixels)  "
              f"wall {means[name][1].round(3)} ({wall.mean():.0%})")
        tint = np.zeros_like(rgb)
        tint[desk], tint[wall] = (1, 0, 0), (0, 0, 1)
        overlays.append(Image.fromarray((np.clip(0.5 * rgb + 0.5 * tint, 0, 1) * 255).astype(np.uint8)).resize((640, 360)))
    for name, _ in pictures[1:]:
        gamma = [np.log(means[name][i]) / np.log(means["ours"][i]) for i in range(2)]
        print(f"{name} = ours ** gamma: desk {gamma[0].round(2)}  wall {gamma[1].round(2)}")
    if args.sheet:
        sheet = Image.new("RGB", (640 * 2, 360 * ((len(overlays) + 1) // 2)))
        for i, im in enumerate(overlays):
            sheet.paste(im, ((i % 2) * 640, (i // 2) * 360))
        sheet.save(args.sheet)


if __name__ == "__main__":
    main()
