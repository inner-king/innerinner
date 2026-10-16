# SPDX-License-Identifier: Apache-2.0
"""Compare a reconstructed OmniGibson scene against the authors' published scene for the same video.

Frames differ between runs (stage 4 picks the world origin from the chosen camera frame), so only
frame-invariant quantities are compared: per-object bounding-box size (sorted, so yaw does not
matter) and pairwise distances between object centres.

Usage: python repro/compare_scenes.py OURS.json THEIRS.json [--match ours_name=their_name ...]
"""
import argparse
import itertools
import json
import math
import os

SKIP_CLASSES = {"FrankaPanda", "FrankaMounted", "R1", "Tiago"}
# Pipeline A registers its assets as DatasetObjects (category/model) under OmniGibson's data path.
REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # repro/<this file> -> repository root
OG_DATASETS = os.path.join(REPO, "deps", "BEHAVIOR-1K", "datasets")


def metadata_path(args, scene_dir):
    usd = args.get("usd_path")
    if usd:
        usd_abs = usd if os.path.isabs(usd) else os.path.normpath(os.path.join(scene_dir, usd))
        return os.path.join(os.path.dirname(os.path.dirname(usd_abs)), "misc", "metadata.json")
    if args.get("model"):
        return os.path.join(OG_DATASETS, args.get("dataset_name", "real2sim-assets"), "objects",
                            args["category"], args["model"], "misc", "metadata.json")
    return None


def load_objects(scene_path):
    with open(scene_path) as f:
        scene = json.load(f)
    base = os.path.dirname(os.path.abspath(scene_path))
    init_info = scene["objects_info"]["init_info"]
    registry = scene["state"]["registry"]["object_registry"]
    objects = {}
    for name, info in init_info.items():
        args = info.get("args", {})
        meta_path = metadata_path(args, base)
        if info.get("class_name") in SKIP_CLASSES or meta_path is None or "background" in name:
            continue
        bbox = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                bbox = json.load(f).get("bbox_size")
        scale = args.get("scale")
        if bbox is not None and scale is not None:
            s = scale if isinstance(scale, (list, tuple)) else [scale] * 3
            bbox = [b * k for b, k in zip(bbox, s)]
        pos = registry.get(name, {}).get("root_link", {}).get("pos")
        objects[name] = {"category": args.get("category", name), "bbox": bbox, "pos": pos}
    return objects


def fmt_cm(v):
    return "x".join(f"{x * 100:.1f}" for x in v) if v else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("theirs")
    ap.add_argument("--match", nargs="*", default=[], help="ours_name=their_name pairs")
    a = ap.parse_args()

    ours, theirs = load_objects(a.ours), load_objects(a.theirs)
    print(f"objects: ours={len(ours)} theirs={len(theirs)}")
    for label, objs in (("ours", ours), ("theirs", theirs)):
        for n, o in objs.items():
            print(f"  [{label}] {n:40s} cat={o['category']:20s} bbox(cm)={fmt_cm(o['bbox'])}")

    pairs = [m.split("=", 1) for m in a.match]
    if not pairs:
        return
    print("\nper-object size (sorted bbox dims, cm):")
    rel_errs = []
    for o_name, t_name in pairs:
        ob, tb = sorted(ours[o_name]["bbox"]), sorted(theirs[t_name]["bbox"])
        err = [abs(x - y) / y for x, y in zip(ob, tb)]
        rel_errs += err
        print(f"  {o_name} vs {t_name}: ours {fmt_cm(ob)} | theirs {fmt_cm(tb)} | rel.err "
              + ", ".join(f"{e * 100:.0f}%" for e in err))
    print(f"  mean relative size error: {100 * sum(rel_errs) / len(rel_errs):.1f}%")

    if len(pairs) >= 2:
        print("\npairwise centre distances (m):")
        for (o1, t1), (o2, t2) in itertools.combinations(pairs, 2):
            d_o = math.dist(ours[o1]["pos"], ours[o2]["pos"])
            d_t = math.dist(theirs[t1]["pos"], theirs[t2]["pos"])
            print(f"  {o1}<->{o2}: ours {d_o:.3f} | theirs {d_t:.3f} | diff {abs(d_o - d_t) * 100:.1f} cm")


if __name__ == "__main__":
    main()
