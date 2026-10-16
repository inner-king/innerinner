# SPDX-License-Identifier: Apache-2.0
"""Dense DA3 poses/depth for background-splat frames, expressed directly in the OmniGibson world.

Chaining short DA3 chunks (what fits in 12 GB) drifted by >20 deg over the video. Instead every DA3 call
here sees a fixed set of anchor frames from the reconstruction's own sparse DA3 run plus a few new
frames, and each call is aligned on its anchors (Umeyama Sim(3)) to the OG world, which is defined as
    T_og<-sparse = s4 cam2world_anchor @ world2cam_sparse[anchor]
so there is no accumulation. Depths are scaled by the same Sim(3) scale.

Writes <bg_dir>/da3_og/results.npz with DA3's layout (world2cam extrinsics N x 3 x 4, intrinsics,
depth, conf, image), ordered like the sorted frames in <bg_dir>/frames_960.

Usage: python repro/masked_bg/dense_da3_anchored.py <scene> <bg_dir> [--anchors 8] [--per-call 8]
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]  # repro/<dir>/<this file> -> repository root
sys.path.insert(0, str(REPO))
from simfoundry.models.depth_anything_v3 import DepthAnythingV3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("bg")
ap.add_argument("--anchors", type=int, default=8)
ap.add_argument("--per-call", type=int, default=8)
ap.add_argument("--res", type=int, default=448)
args = ap.parse_args()

D = str(REPO / "Data")
sp = np.load(f"{D}/{args.scene}/s2_da/da/exports/npz/results.npz")
sp_paths = sorted(glob.glob(f"{D}/{args.scene}/s1_video/frames_subsampled_15/*.png"))
sp_names = [os.path.basename(p) for p in sp_paths]
dn_paths = sorted(glob.glob(f"{args.bg}/frames_960/*.png"))
dn_names = [os.path.basename(p) for p in dn_paths]
anchor = json.load(open(f"{D}/{args.scene}/s3_ground/frame_selection.json"))["selected_idx"]
cam2world = np.load(f"{D}/{args.scene}/s4_frame/image_{anchor}_cam2world.npy")


def h(E):
    T = np.eye(4)
    T[:3, :E.shape[1]] = E[:3]
    return T


T_og_sp = cam2world @ h(sp["extrinsics"][anchor])
C_og_sp = [T_og_sp @ np.linalg.inv(h(e)) for e in sp["extrinsics"]]     # sparse cameras in OG world

# Anchors spread over the sweep; the sparse frames are evenly spaced in time.
a_idx = np.linspace(0, len(sp_names) - 1, args.anchors).round().astype(int)
a_paths = [os.path.join(args.bg, "frames_960", sp_names[i]) for i in a_idx]
targets = [i for i, n in enumerate(dn_names) if n not in {sp_names[k] for k in a_idx}]

model = DepthAnythingV3().model
N = len(dn_names)
out = {k: [None] * N for k in ("extrinsics", "intrinsics", "depth", "conf", "image")}


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = float(np.trace(np.diag(S) @ E) / (xs ** 2).sum(1).mean())
    return s, R, mu_d - s * R @ mu_s


def store(i, ext_run, K, depth, conf, img, s, R, t):
    C = np.linalg.inv(h(ext_run))
    C_og = np.eye(4)
    C_og[:3, :3] = R @ C[:3, :3]
    C_og[:3, 3] = s * R @ C[:3, 3] + t
    out["extrinsics"][i] = np.linalg.inv(C_og)[:3, :4]
    out["intrinsics"][i], out["depth"][i] = K, depth * s
    out["conf"][i], out["image"][i] = conf, img


stats = []
batches = [targets[k:k + args.per_call] for k in range(0, len(targets), args.per_call)]
for b, batch in enumerate(batches):
    paths = a_paths + [dn_paths[i] for i in batch]
    with torch.no_grad():
        pred = model.inference(image=paths, process_res=args.res, process_res_method="upper_bound_resize")
    ext = pred.extrinsics
    src = np.array([np.linalg.inv(h(ext[k]))[:3, 3] for k in range(len(a_paths))])
    dst = np.array([C_og_sp[i][:3, 3] for i in a_idx])
    s, R, t = umeyama(src, dst)
    res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    rot = [np.degrees(np.arccos(np.clip((np.trace(C_og_sp[i][:3, :3].T @ R @ np.linalg.inv(h(ext[k]))[:3, :3]) - 1) / 2, -1, 1)))
           for k, i in enumerate(a_idx)]
    stats.append((res.mean() * 1000, np.mean(rot), s))
    for k in range(len(paths)):
        if k < len(a_paths):
            i = dn_names.index(os.path.basename(paths[k]))
            if out["depth"][i] is not None:
                continue
        else:
            i = batch[k - len(a_paths)]
        store(i, ext[k], pred.intrinsics[k], pred.depth[k], pred.conf[k], pred.processed_images[k], s, R, t)
    print(f"call {b + 1}/{len(batches)}: anchor residual {res.mean() * 1000:.1f} mm, rot {np.mean(rot):.2f} deg, "
          f"scale {s:.3f}", flush=True)

os.makedirs(f"{args.bg}/da3_og", exist_ok=True)
np.savez_compressed(f"{args.bg}/da3_og/results.npz", **{k: np.stack(v) for k, v in out.items()})
st = np.array(stats)
print(f"done: {N} frames | anchor residual mean {st[:, 0].mean():.1f} mm (max {st[:, 0].max():.1f}) | "
      f"rot mean {st[:, 1].mean():.2f} deg | scale {st[:, 2].min():.3f}-{st[:, 2].max():.3f}")
