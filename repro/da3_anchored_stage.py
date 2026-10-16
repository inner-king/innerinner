# SPDX-License-Identifier: Apache-2.0
"""Drop-in replacement for the stage-2 DA3 run when the frame count exceeds what fits in 12 GB at once.

The repo chunks long sequences and chains chunks by Umeyama on their overlaps (sized for 220-frame chunks
on 24 GB). With the 16-frame chunks a 12 GB card can hold, that chain drifted >20 deg over a 15 s video.
Here the chain is replaced by a star:
  1. one joint DA3 call on --ref frames spread evenly over the sequence defines the reference world;
  2. every other call holds --anchors of those reference frames plus up to --per-call new frames, and is
     mapped into the reference world by Umeyama on the anchors' camera centres (no accumulation).
Depths are scaled by each call's Sim(3) scale. The output has the repo's results.npz layout and frame
order (sorted frame_*.png), at <out_dir>/da/exports/npz/results.npz.

Usage: python repro/da3_anchored_stage.py <frames_dir> <out_dir> [--ref 15] [--anchors 8] [--per-call 8] [--res 448]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]  # repro/<this file> -> repository root
sys.path.insert(0, str(REPO))
from simfoundry.models.depth_anything_v3 import DepthAnythingV3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("frames_dir")
ap.add_argument("out_dir")
ap.add_argument("--ref", type=int, default=15)
ap.add_argument("--anchors", type=int, default=8)
ap.add_argument("--per-call", type=int, default=8)
ap.add_argument("--res", type=int, default=448)
args = ap.parse_args()

paths = sorted(str(p) for p in Path(args.frames_dir).iterdir() if p.suffix == ".png" and p.stem.startswith("frame_"))
N = len(paths)
model = DepthAnythingV3().model


def h(E):
    T = np.eye(4)
    T[:3, :E.shape[1]] = E[:3]
    return T


def run(ps):
    with torch.no_grad():
        return model.inference(image=ps, process_res=args.res, process_res_method="upper_bound_resize")


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


out = {k: [None] * N for k in ("extrinsics", "intrinsics", "depth", "conf", "image")}


def store(i, pred, k, s=1.0, R=np.eye(3), t=np.zeros(3)):
    C = np.linalg.inv(h(pred.extrinsics[k]))
    C_ref = np.eye(4)
    C_ref[:3, :3] = R @ C[:3, :3]
    C_ref[:3, 3] = s * R @ C[:3, 3] + t
    out["extrinsics"][i] = np.linalg.inv(C_ref)[:3, :4]
    out["intrinsics"][i] = pred.intrinsics[k]
    out["depth"][i] = pred.depth[k] * s
    out["conf"][i] = pred.conf[k]
    out["image"][i] = pred.processed_images[k]


ref_idx = np.linspace(0, N - 1, min(args.ref, N)).round().astype(int)
ref = run([paths[i] for i in ref_idx])
ref_centres = {int(i): np.linalg.inv(h(ref.extrinsics[k]))[:3, 3] for k, i in enumerate(ref_idx)}
ref_rots = {int(i): np.linalg.inv(h(ref.extrinsics[k]))[:3, :3] for k, i in enumerate(ref_idx)}
for k, i in enumerate(ref_idx):
    store(int(i), ref, k)
print(f"reference: {len(ref_idx)} frames jointly", flush=True)

anc = ref_idx[np.linspace(0, len(ref_idx) - 1, min(args.anchors, len(ref_idx))).round().astype(int)]
todo = [i for i in range(N) if out["depth"][i] is None]
batches = [todo[k:k + args.per_call] for k in range(0, len(todo), args.per_call)]
stats = []
for b, batch in enumerate(batches):
    pred = run([paths[i] for i in anc] + [paths[i] for i in batch])
    src = np.array([np.linalg.inv(h(pred.extrinsics[k]))[:3, 3] for k in range(len(anc))])
    dst = np.array([ref_centres[int(i)] for i in anc])
    s, R, t = umeyama(src, dst)
    res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    rot = [np.degrees(np.arccos(np.clip((np.trace(ref_rots[int(i)].T @ R @ np.linalg.inv(h(pred.extrinsics[k]))[:3, :3]) - 1) / 2, -1, 1)))
           for k, i in enumerate(anc)]
    stats.append((res.mean(), np.mean(rot), s))
    for k, i in enumerate(batch):
        store(i, pred, len(anc) + k, s, R, t)
    print(f"call {b + 1}/{len(batches)}: anchor residual {res.mean() * 1000:.1f} mm, rot {np.mean(rot):.2f} deg, "
          f"scale {s:.3f}", flush=True)

npz_dir = Path(args.out_dir) / "da" / "exports" / "npz"
npz_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(npz_dir / "results.npz", **{k: np.stack(v) for k, v in out.items()})
st = np.array(stats) if stats else np.zeros((1, 3))
print(f"done: {N} frames -> {npz_dir / 'results.npz'} | anchor residual mean {st[:, 0].mean() * 1000:.1f} mm "
      f"(max {st[:, 0].max() * 1000:.1f}) | rot mean {st[:, 1].mean():.2f} deg | scale {st[:, 2].min():.3f}-{st[:, 2].max():.3f}")
