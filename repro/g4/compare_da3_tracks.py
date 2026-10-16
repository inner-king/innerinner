# SPDX-License-Identifier: Apache-2.0
"""How closely void-DA3's camera track (auto_bg) follows the original DA3's (canonical stage 2).

auto_bg step 4 fits a rigid transform from void-DA3's camera centres onto the original's and builds
the splat's seed cloud with it; the splat's depth loss then trusts void-DA3's per-frame depth in that
frame. A track that drifts (as 16-frame chunks did on the 12 GB GPU, repro/README.md chapter 13)
shows up here as a large residual. Reported for the rigid fit step 4 uses and for a similarity fit:
camera-centre residual (mm) and the per-frame orientation disagreement (deg) after alignment.

Usage: python compare_da3_tracks.py <scene_dir>
"""
import sys

import numpy as np


def cam2world(ext):
    """(N,3,4) world-to-camera -> camera rotations (N,3,3) and centres (N,3) in world."""
    rot = np.transpose(ext[:, :3, :3], (0, 2, 1))
    return rot, -np.einsum("nij,nj->ni", rot, ext[:, :3, 3])


def umeyama(src, dst, with_scale):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(xd.T @ xs / len(src))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = u @ d @ vt
    scale = (sig * np.diag(d)).sum() / xs.var(0).sum() if with_scale else 1.0
    return scale, rot, mu_d - scale * rot @ mu_s


def main(scene_dir):
    r_orig, c_orig = cam2world(np.load(f"{scene_dir}/s2_da/da/exports/npz/results.npz")["extrinsics"])
    r_void, c_void = cam2world(np.load(f"{scene_dir}/auto_bg/da3/void/da/exports/npz/results.npz")["extrinsics"])
    print(f"{scene_dir}: {len(c_orig)} frames, track length {np.linalg.norm(np.diff(c_orig, axis=0), axis=1).sum():.2f} m")
    for name, with_scale in (("rigid", False), ("similarity", True)):
        s, rot, t = umeyama(c_void, c_orig, with_scale)
        res = np.linalg.norm(s * c_void @ rot.T + t - c_orig, axis=1) * 1000
        cos = (np.einsum("nij,nij->n", np.einsum("ij,njk->nik", rot, r_void), r_orig) - 1) / 2
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        print(f"  {name:10s} scale {s:.4f} | centre residual mean {res.mean():.1f} mm, median {np.median(res):.1f}, "
              f"max {res.max():.1f} | orientation mean {ang.mean():.2f} deg, max {ang.max():.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
