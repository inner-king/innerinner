# SPDX-License-Identifier: Apache-2.0
"""Geometric fit of each reconstructed object against its observed (depth-derived) point cloud.

Stage 8 writes, per object, the observed partial cloud (s8_pose/pc/iter_N.ply), the canonical mesh
(s8_pose/canonical_mesh/iter_N.glb) and the pose mapping one onto the other (s8_pose/info/iter_N.json,
"z_up": scale, rot, trans). The observation only covers the visible side, so the metric is one-sided:
distance from every observed point to the posed mesh surface. Reported in the spirit of the paper's
Table L.2 (Chamfer, F1 @ 1 cm), but against monocular depth rather than a ground-truth scan.

Usage: python eval_pose_fit.py <scene_dir>
"""
import glob
import json
import os
import sys

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

scene_dir = sys.argv[1]
s8 = os.path.join(scene_dir, "s8_pose")
rows = []
for info_path in sorted(glob.glob(f"{s8}/info/iter_*.json")):
    name = os.path.splitext(os.path.basename(info_path))[0]
    tf = json.load(open(info_path))["z_up"]
    mesh = trimesh.load(f"{s8}/canonical_mesh/{name}.glb", force="mesh")
    scale = np.asarray(tf["scale"], dtype=float)
    rot, trans = np.asarray(tf["rot"]), np.asarray(tf["trans"])
    mesh.vertices = (scale * mesh.vertices) @ rot.T + trans
    surface, _ = trimesh.sample.sample_surface(mesh, 200_000, seed=0)
    obs = np.asarray(o3d.io.read_point_cloud(f"{s8}/pc/{name}.ply").points)
    d, _ = cKDTree(surface).query(obs)
    rows.append((name, len(obs), d.mean(), np.median(d), (d < 0.01).mean(), (d < 0.005).mean()))

print(f"{'object':8s} {'n_obs':>7s} {'mean(mm)':>9s} {'median(mm)':>11s} {'<1cm':>6s} {'<5mm':>6s}")
for name, n, mean, med, p10, p5 in rows:
    print(f"{name:8s} {n:7d} {mean * 1000:9.2f} {med * 1000:11.2f} {p10 * 100:5.1f}% {p5 * 100:5.1f}%")
means = np.array([r[2] for r in rows]); p10s = np.array([r[4] for r in rows])
print(f"{'avg':8s} {'':7s} {means.mean() * 1000:9.2f} {'':11s} {p10s.mean() * 100:5.1f}%")
