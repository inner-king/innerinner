# SPDX-License-Identifier: Apache-2.0
"""Dicing demonstrations to keep: every episode the expert dice-judged strict (dice_strict, which already requires a
clear knife plan), none of the seeds skipped for an unclear plan. Writes <out>/accepted.txt (npz paths) and
<out>/summary.json, and prints the counts and the wall time per episode.

Usage: dice_select.py <out dir> <run dir> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    keep, rows, skipped = [], [], []
    for d in map(Path, sys.argv[2:]):
        skipped += [json.loads(p.read_text())["seed"] for p in sorted(d.glob("skipped_*.json"))]
        for j in sorted(d.glob("ep_*.json")):
            o = json.loads(j.read_text())["outcome"]
            dz = o.get("dice", {})
            r = {"npz": str(j.with_suffix(".npz")), "seed": json.loads(j.read_text())["seed"], "basic": o.get("dice_basic"),
                 "strict": o.get("dice_strict"), "pieces": dz.get("pieces_over_tenth"), "cv": dz.get("mass_cv"),
                 "wall_s": o.get("wall_s"), "sim_s": o.get("sim_time_s"), "fmax": o.get("max_force_n"),
                 "others_max_mm": max((o.get("other_objects_moved_mm") or {"-": 0.0}).values())}
            rows.append(r)
            if r["strict"]:
                keep.append(r["npz"])
    (out / "accepted.txt").write_text("\n".join(keep) + "\n")
    (out / "summary.json").write_text(json.dumps({"episodes": rows, "skipped_unclear": skipped}, indent=1))
    w = np.array([r["wall_s"] for r in rows], float)
    print(f"episodes {len(rows)}, strict {len(keep)}, basic {sum(bool(r['basic']) for r in rows)}, skipped (unclear plan) "
          f"{len(skipped)}; wall per episode median {np.median(w) if len(w) else float('nan'):.0f} s "
          f"(min {w.min() if len(w) else float('nan'):.0f}, max {w.max() if len(w) else float('nan'):.0f})")
    for r in rows:
        if not r["strict"]:
            print("  not kept:", r)


if __name__ == "__main__":
    main()
