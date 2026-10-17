# SPDX-License-Identifier: Apache-2.0
"""Compare two stage-5 (scene decomposition) output folders iteration by iteration.

Used to check that a change to stage 5 leaves a decomposition as it was (repro/README.md,
chapter 24): the per-iteration records must match, and every mask and image must be identical
pixel for pixel. Depth is reported, not required to match: Prior-Depth-Anything adds unseeded
noise when it fits scale and shift (`_calc_scale_shift_fp32` in 5_decompose_scene.py), so it
differs slightly from run to run even on identical inputs.

Keys that only one side's records have (fields added to the manifest after the older run) are
listed but not counted as differences.

Usage: python repro/compare_s5_runs.py <old>/s5_scene <new>/s5_scene    (exit 1 on a difference)
"""

import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image

IMAGE_DIRS = [
    "detected_phrases", "removal_mask", "removal_mask_resized", "masked_object", "masked_object_focus",
    "masked_object_background", "pre_object_removal", "post_object_removal",
]
RECORD_DIRS = ["obj_cat_list", "skipped_iterations"]
RECORD_NAME = re.compile(r"iter_(\d+)\.json")


def record_indices(s5_dir):
    indices = set()
    for sub in RECORD_DIRS:
        path = os.path.join(s5_dir, sub)
        if os.path.isdir(path):
            indices |= {int(m.group(1)) for m in map(RECORD_NAME.fullmatch, os.listdir(path)) if m}
    return indices


def load_image(path):
    return np.array(Image.open(path)) if os.path.isfile(path) else None


def compare(old, new):
    differences = []
    for name in ["source_padded_resized.png", "source_padded_resized_upsampled.png"]:
        a, b = load_image(os.path.join(old, name)), load_image(os.path.join(new, name))
        if a is None or b is None or a.shape != b.shape or not np.array_equal(a, b):
            differences.append(name)

    old_iters, new_iters = record_indices(old), record_indices(new)
    print(f"iterations: old {sorted(old_iters)}, new {sorted(new_iters)}")
    if old_iters != new_iters:
        differences.append("iteration indices")

    for i in sorted(old_iters | new_iters):
        line = [f"iter_{i}:"]
        for sub in RECORD_DIRS:
            pa, pb = (os.path.join(d, sub, f"iter_{i}.json") for d in (old, new))
            if not (os.path.isfile(pa) or os.path.isfile(pb)):
                continue
            if not (os.path.isfile(pa) and os.path.isfile(pb)):
                differences.append(f"iter_{i} {sub} only in {'new' if os.path.isfile(pb) else 'old'}")
                line.append(f"{sub} only in one run")
                continue
            ra, rb = (json.load(open(p)) for p in (pa, pb))
            shared = set(ra) & set(rb)
            changed = sorted(k for k in shared if ra[k] != rb[k])
            added = sorted(set(rb) - set(ra))
            if changed:
                differences.append(f"iter_{i} {sub} {changed}")
            line.append(f"{sub} {'same' if not changed else 'DIFFERENT ' + str(changed)}"
                        + (f" (new keys {added})" if added else ""))
            if sub == "obj_cat_list":
                line.append(f"'{rb.get('removed_obj_phrase')}'")
        for sub in IMAGE_DIRS:
            a, b = (load_image(os.path.join(d, sub, f"iter_{i}.png")) for d in (old, new))
            if a is None and b is None:
                continue
            if a is None or b is None or a.shape != b.shape or not np.array_equal(a, b):
                differences.append(f"iter_{i} {sub}")
                line.append(f"{sub} DIFFERENT")
        da, db = (os.path.join(d, "metric_depth", f"iter_{i}.npy") for d in (old, new))
        if os.path.isfile(da) and os.path.isfile(db):
            diff = np.abs(np.load(da) - np.load(db))
            line.append(f"depth: {np.mean(diff > 0):.1%} px differ, p99 {np.percentile(diff, 99) * 1000:.2f} mm, "
                        f"max {diff.max() * 1000:.1f} mm")
        print("  ".join(line))
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old")
    parser.add_argument("new")
    args = parser.parse_args()
    differences = compare(args.old, args.new)
    if differences:
        print("DIFFERENT:", "; ".join(differences))
        sys.exit(1)
    print("Records, masks and images identical (depth reported above).")


if __name__ == "__main__":
    main()
