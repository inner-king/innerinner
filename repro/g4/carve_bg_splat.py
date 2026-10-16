# SPDX-License-Identifier: Apache-2.0
"""Remove free-space floaters from an auto_bg background splat, using the depth it was trained on.

Seen from the video's own cameras, a Gaussian floating in front of the desk blends in or is too faint
to matter; from the robot's cameras, which look across the desk from about 0.4 m above it, the same
Gaussians stand between the camera and the scene as grey fog and shards (repro/README.md, 17.7). A
Gaussian whose centre projects in front of the observed surface, by more than the margin below, in a
good share of the frames that see it is in space the video saw through; one that no frame sees at
all was never constrained. Both are dropped; everything else is written unchanged.

The splat lives in the original DA3 frame (step 5 trains with the original run's poses and the
void-DA3 depth, here the scale-corrected void_sim3.npz from scale_void_da3.py), so the test projects
with the original run's cameras and compares against that depth where void-DA3 is confident.

Usage: python carve_bg_splat.py <scene_dir> <in.ply> <out.ply> [min_share=0.2]
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

scene, src, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
min_share = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
orig = np.load(scene / "s2_da" / "da" / "exports" / "npz" / "results.npz")
void = np.load(scene / "auto_bg" / "da3" / "void_sim3.npz")
ext, K = orig["extrinsics"], orig["intrinsics"]          # world-to-camera (OpenCV), at DA3 resolution
depth, conf = void["depth"], void["conf"]
n_frames, h, w = depth.shape

ply = PlyData.read(str(src))
v = ply["vertex"].data
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
seen = np.zeros(len(xyz), np.int32)
ahead = np.zeros(len(xyz), np.int32)
for i in range(n_frames):
    cam = xyz @ ext[i, :3, :3].T + ext[i, :3, 3]
    z = cam[:, 2]
    front = z > 0.05
    u = np.full(len(xyz), -1, np.int64)
    r = np.full(len(xyz), -1, np.int64)
    u[front] = np.floor(K[i, 0, 0] * cam[front, 0] / z[front] + K[i, 0, 2]).astype(np.int64)
    r[front] = np.floor(K[i, 1, 1] * cam[front, 1] / z[front] + K[i, 1, 2]).astype(np.int64)
    inside = front & (u >= 0) & (u < w) & (r >= 0) & (r < h)
    idx = np.nonzero(inside)[0]
    d = depth[i, r[idx], u[idx]]
    c = conf[i, r[idx], u[idx]]
    ok = (d > 0) & (c >= 2.0)          # the confidence floor step 4 and the depth loss use
    idx, d = idx[ok], d[ok]
    seen[idx] += 1
    ahead[idx[z[idx] < d - np.maximum(0.03, 0.05 * d)]] += 1

share = np.where(seen > 0, ahead / np.maximum(seen, 1), 1.0)
keep = (seen > 0) & ((share < min_share) | (ahead < 3))
dst.parent.mkdir(parents=True, exist_ok=True)
PlyData([PlyElement.describe(v[keep], "vertex")], text=False).write(str(dst))
pose = Path(f"{src}.pose.json")
if pose.exists():
    shutil.copy2(pose, f"{dst}.pose.json")
print(f"{len(xyz)} Gaussians: {int((seen == 0).sum())} never seen, "
      f"{int(((seen > 0) & ~keep).sum())} in free space in >= {min_share:.0%} of the frames that see them; "
      f"kept {int(keep.sum())} -> {dst}")
