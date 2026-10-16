# SPDX-License-Identifier: Apache-2.0
"""Put void-DA3 into the original DA3's metric scale before auto_bg seeds and trains the splat.

auto_bg runs DA3 twice: on the input frames (stage 2, the world the objects live in) and on the
object-free VOID frames (void-DA3, the splat's depth supervision). Each run fixes its own metric scale;
step 4 aligns the two only rigidly, and step 5 pairs the original run's camera poses with void-DA3's
depth, so a scale gap between the runs (6.2% on PutCupInBowl, repro/README.md 17.4) becomes a depth
error in the splat. This fits the similarity transform between the two camera tracks and writes a copy
of the void-DA3 npz with depth and camera centres multiplied by its scale, for step 4
(s4_seed_ply.void_da3_npz) and step 5 (s5_train_bg_splat.depth_da3_npz).

Usage: python scale_void_da3.py <scene_dir> <out.npz>
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from compare_da3_tracks import cam2world, umeyama  # noqa: E402

scene, out = sys.argv[1], sys.argv[2]
orig = np.load(f"{scene}/s2_da/da/exports/npz/results.npz")
void = dict(np.load(f"{scene}/auto_bg/da3/void/da/exports/npz/results.npz"))
_, c_orig = cam2world(orig["extrinsics"])
_, c_void = cam2world(void["extrinsics"])
s, _, _ = umeyama(c_void, c_orig, with_scale=True)
void["depth"] = (void["depth"] * s).astype(void["depth"].dtype)
ext = void["extrinsics"].copy()
ext[:, :3, 3] *= s  # world-to-camera translation scales with the camera centres
void["extrinsics"] = ext
np.savez(out, **void)
print(f"void-DA3 scale to original DA3: {s:.4f} -> {out}")
