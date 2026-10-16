# SPDX-License-Identifier: Apache-2.0
"""Side-by-side summary of A reconstructions of the same video, e.g. RTX 3060 v2 against G4.

For each scene directory: the A run's stage times (its block of pipeline_timing.log), and per
object the category, size (s11_sim metadata bbox_size, cm), mass (URDF, g), friction, and the
one-sided pose-fit residual of repro/eval_pose_fit.py.

Usage: python compare_runs.py <label>=<scene_dir> [<label>=<scene_dir> ...]
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval_pose_fit import pose_fit  # noqa: E402


def a_run_times(scene_dir):
    """Stage -> seconds for the last A run in the log: the last WALL-terminated block with stage 14."""
    blocks, cur = [], {}
    for line in open(os.path.join(scene_dir, "pipeline_timing.log")):
        m = re.search(r"stage=(\S+) \| status=(\S+) \| elapsed_s=([\d.]+)", line)
        if not m:
            continue
        cur[m.group(1)] = float(m.group(3))
        if m.group(1) == "WALL":
            blocks.append(cur)
            cur = {}
    a_blocks = [b for b in blocks if "14" in b]
    return a_blocks[-1] if a_blocks else {}


def objects(scene_dir):
    info = json.load(open(os.path.join(scene_dir, "s11_sim", "scene_objects_info.json")))
    fits = {r[0]: r for r in pose_fit(scene_dir)}
    rows = []
    for o in info.values():
        base = os.path.join(scene_dir, "s11_sim", "objects", o["category"], o["model"])
        bbox = json.load(open(os.path.join(base, "misc", "metadata.json")))["bbox_size"]
        urdf = open(os.path.join(base, "urdf", f"{o['model']}.urdf")).read()
        mass = sum(float(v) for v in re.findall(r'<mass value="([\d.eE+-]+)"', urdf))
        fit = fits.get(o["name"])
        rows.append((o["name"], o["category"], [100 * v for v in bbox], 1000 * mass, o.get("friction"), fit))
    return sorted(rows)


def main():
    runs = [a.split("=", 1) for a in sys.argv[1:]]
    times = {label: a_run_times(d) for label, d in runs}
    stages = []
    for t in times.values():
        stages += [s for s in t if s not in stages]
    print(f"{'stage':>6s} " + " ".join(f"{label:>12s}" for label, _ in runs))
    for s in stages:
        print(f"{s:>6s} " + " ".join(f"{times[label].get(s, float('nan')):12.1f}" for label, _ in runs))
    for label, d in runs:
        print(f"\n[{label}] {d}")
        print(f"  {'object':7s} {'category':34s} {'size (cm)':>20s} {'mass g':>7s} {'fric':>5s} "
              f"{'fit mean mm':>11s} {'<1cm':>6s}")
        for name, cat, size, mass, fric, fit in objects(d):
            size_s = " x ".join(f"{v:.1f}" for v in size)
            fit_s = f"{fit[2] * 1000:11.2f} {fit[4] * 100:5.1f}%" if fit else f"{'-':>11s} {'-':>6s}"
            print(f"  {name:7s} {cat:34s} {size_s:>20s} {mass:7.0f} {fric if fric is not None else '-':>5} {fit_s}")


if __name__ == "__main__":
    main()
