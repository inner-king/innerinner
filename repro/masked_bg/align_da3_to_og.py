# SPDX-License-Identifier: Apache-2.0
"""Sim(3) from a dense DA3 run (background splat frames) into a reconstruction's OmniGibson world.

The reconstruction's world comes from its own sparse DA3 run: stage 4 maps the anchor frame's camera
coordinates to the world with s4_frame/image_<i>_cam2world.npy, so
    T_og<-da_sparse = cam2world_anchor @ world2cam_sparse[anchor].
The dense run contains every sparse frame (same file names), so their camera centres in both runs give
an Umeyama Sim(3) dense -> OG world.

Checks: camera-centre residual, camera rotation agreement, and the height of the desk (non-object,
high-confidence DA3 points of the anchor frame) in the OG world, which should be ~0.

Usage: python repro/masked_bg/align_da3_to_og.py <scene> <bg_dir>     (writes <bg_dir>/sim3_da3_to_og.json)
"""
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

scene, bg = sys.argv[1:3]
D = str(Path(__file__).resolve().parents[2] / "Data")  # repro/<dir>/<this file> -> repository root
sp = np.load(f"{D}/{scene}/s2_da/da/exports/npz/results.npz")
dn = np.load(f"{bg}/s2_da/da/exports/npz/results.npz")
sp_names = [os.path.basename(p) for p in sorted(glob.glob(f"{D}/{scene}/s1_video/frames_subsampled_15/*.png"))]
dn_names = [os.path.basename(p) for p in sorted(glob.glob(f"{bg}/frames_960/*.png"))]
anchor = json.load(open(f"{D}/{scene}/s3_ground/frame_selection.json"))["selected_idx"]
cam2world = np.load(f"{D}/{scene}/s4_frame/image_{anchor}_cam2world.npy")


def h(E):
    T = np.eye(4)
    T[:3, :4] = E
    return T


T_og_sp = cam2world @ h(sp["extrinsics"][anchor])
src, dst, R_src, R_dst = [], [], [], []
for i, name in enumerate(sp_names):
    j = dn_names.index(name)
    C_og = T_og_sp @ np.linalg.inv(h(sp["extrinsics"][i]))
    C_dn = np.linalg.inv(h(dn["extrinsics"][j]))
    dst.append(C_og[:3, 3]); src.append(C_dn[:3, 3])
    R_dst.append(C_og[:3, :3]); R_src.append(C_dn[:3, :3])
src, dst = np.array(src), np.array(dst)

# Umeyama with scale
mu_s, mu_d = src.mean(0), dst.mean(0)
xs, xd = src - mu_s, dst - mu_d
U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
E = np.eye(3)
if np.linalg.det(U @ Vt) < 0:
    E[2, 2] = -1
R = U @ E @ Vt
s = float(np.trace(np.diag(S) @ E) / (xs ** 2).sum(1).mean())
t = mu_d - s * R @ mu_s
res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
rot_err = [np.degrees(np.arccos(np.clip((np.trace(Rd.T @ R @ Rs) - 1) / 2, -1, 1))) for Rs, Rd in zip(R_src, R_dst)]
print(f"{scene}: {len(src)} shared frames | scale {s:.4f} | centre residual mean {res.mean() * 1000:.1f} mm "
      f"max {res.max() * 1000:.1f} mm | camera rotation error mean {np.mean(rot_err):.2f} deg max {np.max(rot_err):.2f} deg")

# Desk height check on the anchor frame (dense run), objects masked out.
j = dn_names.index(sp_names[anchor])
depth, conf, K = dn["depth"][j], dn["conf"][j], dn["intrinsics"][j]
H, W = depth.shape
mask = cv2.imread(f"{bg}/masks_960/{dn_names[j]}", cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 127
v, u = np.mgrid[0:H, 0:W]
keep = (~mask) & (conf >= 2.0) & (depth > 0)
pts_cam = np.stack([(u[keep] - K[0, 2]) / K[0, 0] * depth[keep], (v[keep] - K[1, 2]) / K[1, 1] * depth[keep], depth[keep]], 1)
C_dn = np.linalg.inv(h(dn["extrinsics"][j]))
pts_dn = (C_dn[:3, :3] @ pts_cam.T).T + C_dn[:3, 3]
pts_og = (s * (R @ pts_dn.T)).T + t
near = np.linalg.norm(pts_og[:, :2] - pts_og[:, :2].mean(0), axis=1) < 0.4
z = pts_og[near, 2]
print(f"desk points near objects (anchor frame): z median {np.median(z) * 1000:.1f} mm, "
      f"10-90% [{np.percentile(z, 10) * 1000:.1f}, {np.percentile(z, 90) * 1000:.1f}] mm")
T = np.eye(4)
T[:3, :3] = s * R
T[:3, 3] = t
json.dump({"scale": s, "R": R.tolist(), "t": t.tolist(), "T": T.tolist(), "anchor": anchor,
           "residual_mm_mean": float(res.mean() * 1000), "rot_err_deg_mean": float(np.mean(rot_err))},
          open(f"{bg}/sim3_da3_to_og.json", "w"), indent=1)
