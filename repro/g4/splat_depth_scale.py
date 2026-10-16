# SPDX-License-Identifier: Apache-2.0
"""Depth scale of an auto_bg background splat against the original DA3 (canonical stage 2).

auto_bg step 5 trains with the original DA3's camera poses but supervises depth with void-DA3, a
separate DA3 run on the object-free frames, and step 4 aligns the two rigidly (no scale). When the
two runs disagree in metric scale, the splat sits between them, and since step 6 places it in the
OmniGibson world through the original DA3, a scale error shows as a desk surface above or below the
objects. For every k-th frame this renders the splat's depth and reports the median ratio
rendered / original-DA3 and rendered / void-DA3 (pixels where both DA3 runs are confident; the
median ignores the few percent of pixels the removed objects covered).

Run in the nerfstudio_simfoundry env through repro/g4/ns_run.sh.
Usage: repro/g4/ns_run.sh python repro/g4/splat_depth_scale.py <Data/<scene>> [every_k=10]
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from nerfstudio.utils.eval_utils import eval_setup

scene = os.path.abspath(sys.argv[1])
every = int(sys.argv[2]) if len(sys.argv) > 2 else 10
splat = os.path.join(scene, "auto_bg", "splat")
orig = np.load(os.path.join(scene, "s2_da", "da", "exports", "npz", "results.npz"))
void = np.load(os.path.join(scene, "auto_bg", "da3", "void", "da", "exports", "npz", "results.npz"))

os.chdir(splat)  # config.yml paths are relative to the splat directory
cfg = sorted(glob.glob("outputs/ns_data/splatfacto/*/config.yml"))[-1]
_, pipeline, _, _ = eval_setup(__import__("pathlib").Path(cfg), test_mode="inference")
dm = pipeline.datamanager
sets = [(dm.train_dataset.cameras, dm.train_dataset._dataparser_outputs.image_filenames)]
if getattr(dm, "eval_dataset", None) is not None:
    sets.append((dm.eval_dataset.cameras, dm.eval_dataset._dataparser_outputs.image_filenames))

rows = []
for cameras, names in sets:
    for i, name in enumerate(names):
        idx = int(os.path.splitext(os.path.basename(str(name)))[0].split("_")[-1])
        if idx % every:
            continue
        cam = cameras[i:i + 1].to(pipeline.device)
        with torch.no_grad():
            out = pipeline.model.get_outputs_for_camera(cam)
        rend = out["depth"].squeeze(-1)[None, None]
        h, w = orig["depth"].shape[1:]
        rend = F.interpolate(rend, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        valid = (orig["conf"][idx] > np.quantile(orig["conf"][idx], 0.3)) & \
                (void["conf"][idx] > np.quantile(void["conf"][idx], 0.3)) & (rend > 0)
        rows.append((idx, np.median(rend[valid] / orig["depth"][idx][valid]),
                     np.median(rend[valid] / void["depth"][idx][valid])))

rows.sort()
r_o = np.array([r[1] for r in rows]); r_v = np.array([r[2] for r in rows])
print(json.dumps({"scene": os.path.basename(scene), "frames": len(rows),
                  "render_over_orig_da3": {"median": float(np.median(r_o)), "min": float(r_o.min()), "max": float(r_o.max())},
                  "render_over_void_da3": {"median": float(np.median(r_v)), "min": float(r_v.min()), "max": float(r_v.max())}},
                 indent=1))
