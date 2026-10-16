# SPDX-License-Identifier: Apache-2.0
"""Drop the Gaussians of an auto_bg background splat that lie below the support surface.

The video looks down at the desk, so what is under it (floor, cables, the robot cart) is seen only in
glimpses, and its Gaussians come out as shards; from the robot's cameras, which sit about 0.4 m above
the desk and look across it, they show up past the desk's front edge (repro/README.md, 17.7). The
reconstruction's support surface is the OG world's z = 0 (the objects rest on it), and step 6's
sidecar maps the splat (DA3 frame) into that world, so this keeps the Gaussians whose centre is above
z_min there and writes the rest unchanged, next to a copy of the sidecar.

Usage: python crop_bg_splat.py <in.ply> <out.ply> [z_min=-0.10]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
z_min = float(sys.argv[3]) if len(sys.argv) > 3 else -0.10
M = np.array(json.load(open(f"{src}.pose.json"))["M_src_to_og"])
ply = PlyData.read(str(src))
v = ply["vertex"].data
z = np.stack([v["x"], v["y"], v["z"]], 1) @ M[2, :3] + M[2, 3]
keep = z >= z_min
dst.parent.mkdir(parents=True, exist_ok=True)
PlyData([PlyElement.describe(v[keep], "vertex")], text=False).write(str(dst))
shutil.copy2(f"{src}.pose.json", f"{dst}.pose.json")
print(f"kept {keep.sum()} / {len(keep)} Gaussians with OG z >= {z_min} -> {dst}")
