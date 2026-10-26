# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Optional stage (run with `--bg-mesh`): a textured mesh of everything around the objects.

Joins the canonical frame's depth map (stage 2), the world frame (stages 3-4) and the clean plate stage 5
leaves behind (the frame repainted with every detected object removed): each depth pixel becomes a vertex,
neighbouring pixels are joined into triangles, pixels under an object are moved onto the support plane, and
the clean plate is the texture. The method and its limits are in `simfoundry/pipeline/depth_mesh.py`.

Outputs under `Data/<scene>/s_bg_mesh/`:
  - background_mesh.glb      Y-up as glTF requires; the light editor's mesh import (up axis `auto`) turns it
                             back into the Z-up world, support plane at z = 0
  - background_texture.png   the texture, for inspection
  - stage_info.json          counts, settings and which stage-5 files were used

Should be run from the `simfoundry` environment (CPU only).
"""
import logging
from pathlib import Path

import hydra
import numpy as np
from PIL import Image

from simfoundry import CFG_DIR
from simfoundry.pipeline.depth_mesh import DEFAULT_PARAMS, build_background_mesh, export_glb, find_stage5_inputs
from simfoundry.pipeline.frame_selection import resolve_img_idx
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage

logger = logging.getLogger(__name__)

bootstrap_hydra_workdir(__file__)


def load_depth_and_intrinsics(cfg, img_idx):
    """The canonical frame's depth (m) and intrinsics, from the backend stage 2 used (as stage 4 does)."""
    if cfg.s3_ground.get("use_fs", False):
        fs_dir = cfg.s2_fs.out_dir
        return np.load(f"{fs_dir}/image_{img_idx}_depth_meter.npy"), np.load(f"{fs_dir}/image_{img_idx}_K.npy")
    results = np.load(f"{cfg.s2_da.out_dir}/da/exports/npz/results.npz")
    return results["depth"][img_idx], results["intrinsics"][img_idx]


def union_of_masks(mask_paths, shape_hw):
    """Boolean union of the removal masks, on a frame of `shape_hw`; None when there are none."""
    union = None
    for path in mask_paths:
        mask = np.asarray(Image.open(path))
        if mask.ndim == 3:
            mask = mask.max(axis=-1)
        if mask.shape != tuple(shape_hw):
            mask = np.asarray(Image.fromarray(mask).resize((shape_hw[1], shape_hw[0]), Image.NEAREST))
        union = mask > 0 if union is None else union | (mask > 0)
    return union


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    out_dir = Path(cfg.s_bg_mesh.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_idx = resolve_img_idx(cfg)

    depth, K = load_depth_and_intrinsics(cfg, img_idx)
    cam2world = np.load(f"{cfg.s4_frame.out_dir}/image_{img_idx}_cam2world.npy")
    inputs = find_stage5_inputs(cfg.s5_scene.out_dir)
    plate = np.asarray(Image.open(inputs["plate"]).convert("RGB"))
    object_mask = union_of_masks(inputs["masks"], plate.shape[:2])
    if inputs["iteration"] is None:
        logger.warning("Stage 5 removed no object; the source frame is the texture and no pixel is replaced by the plane.")
    logger.info(
        "Building the background mesh from depth %s, clean plate %s (iteration %s) and %d removal mask(s)",
        depth.shape, inputs["plate"], inputs["iteration"], len(inputs["masks"]),
    )

    params = {k: cfg.s_bg_mesh[k] for k in DEFAULT_PARAMS if k in cfg.s_bg_mesh}
    if params.get("z_far") is None:
        params["z_far"] = float(cfg.s4_frame.z_far)
    mesh, texture, stats = build_background_mesh(depth, K, cam2world, plate, object_mask, params)

    mesh_fpath = out_dir / "background_mesh.glb"
    export_glb(mesh, mesh_fpath)
    texture.save(out_dir / "background_texture.png")
    logger.info("Wrote %s: %d vertices, %d triangles", mesh_fpath, stats["vertices"], stats["triangles"])

    finalize_stage(
        stage_cfg=cfg.s_bg_mesh,
        out_dir=out_dir,
        result=StageResult(success=True, additional_info={
            **stats,
            "img_idx": int(img_idx),
            "clean_plate": inputs["plate"],
            "clean_plate_iteration": inputs["iteration"],
            "removal_masks_used": len(inputs["masks"]),
            "mesh": str(mesh_fpath),
            "file_up_axis": "Y",
        }),
    )


if __name__ == "__main__":
    main()
