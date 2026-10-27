# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from b1k

Requires installing:

- BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K
"""
import os
import sys
# We need to run this in headless mode for omnigibson
os.environ["OMNIGIBSON_HEADLESS"] = "1"

import omnigibson as og
from omnigibson.utils.asset_utils import get_dataset_path, get_omnigibson_git_hash
from functools import partial
from importlib.util import find_spec
from pathlib import Path
import json
import shutil
import subprocess
import hydra
from simfoundry import CFG_DIR
from simfoundry.pipeline.asset_manifest import (
    apply_conflict_policy,
    decide_import,
    describe_decision,
    directory_hashes,
    import_code_hash,
    import_settings_hash,
    read_manifest,
    remove_manifest,
    urdf_mass_kg,
    write_manifest,
)
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage
import logging

scripts_dir = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
bootstrap_hydra_workdir(__file__)


def resolve_pipeline_script(script_name: str) -> str:
    """Find a USD post-processing helper shared with augmentation."""
    pipeline_dir = os.path.dirname(os.path.dirname(scripts_dir))
    candidates = [
        os.path.join(scripts_dir, script_name),
        os.path.join(pipeline_dir, "B_augmentation", "stages", script_name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find {script_name} in any expected location: " + ", ".join(candidates)
    )


def resolve_reparent_script() -> str:
    """Find the USD joint reparenting helper shared with augmentation."""
    return resolve_pipeline_script("reparent_usd_joints.py")


def imported_usd_path(dataset_name: str, obj_category: str, obj_model: str) -> str:
    dataset_path = get_dataset_path(dataset_name)
    return os.path.join(dataset_path, "objects", obj_category, obj_model, "usd", f"{obj_model}.usd")


def import_code_versions() -> dict:
    """The OmniGibson version and checkout the import runs under; a missing git checkout records "unknown"."""
    try:
        git_hash = get_omnigibson_git_hash()
    except Exception:  # a checkout without git must not stop the stage; the version still counts
        git_hash = "unknown"
    return {"omnigibson": og.__version__, "omnigibson_git": git_hash}


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sim_dir = cfg.s11_sim.out_dir
    out_dir = cfg.s13_usd.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # null disables the pass, leaving whatever the importer authored.
    opacity_threshold = cfg.s13_usd.get("opacity_threshold", 0.5)
    asset_pipeline_materials = bool(cfg.s13_usd.get("asset_pipeline_materials", True))
    # on_conflict: what a re-import does when it would replace an asset that an earlier scene uses (warn | fail |
    # overwrite). skip_unchanged: skip the import when the asset's manifest says nothing changed. It is off by default:
    # the decision is still logged and recorded in the manifest, the import just runs as it did before.
    on_conflict = cfg.s13_usd.get("on_conflict", "warn")
    skip_unchanged = bool(cfg.s13_usd.get("skip_unchanged", False))
    settings_hash = import_settings_hash({
        "opacity_threshold": opacity_threshold,
        "asset_pipeline_materials": asset_pipeline_materials,
        "collision_method": "none",
    })
    # The import code: a change to any of these files, or to the OmniGibson it runs under, makes an asset imported before
    # it a settings conflict rather than a skip (asset_manifest.import_code_hash).
    code_hash = import_code_hash(
        [
            ("13_import_usd.py", __file__),
            ("og_asset_import.py", find_spec("simfoundry.utils.og_asset_import").origin),
            ("set_usd_opacity_threshold.py", resolve_pipeline_script("set_usd_opacity_threshold.py")),
            ("reparent_usd_joints.py", resolve_reparent_script()),
        ],
        import_code_versions(),
    )

    # Load scene objects info
    scene_objects_info_fpath = f"{sim_dir}/scene_objects_info.json"
    with open(scene_objects_info_fpath, "r") as f:
        scene_objects_info = json.load(f)

    logger.info("="*60)
    logger.info("Starting USD import pipeline...")
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*60)

    asset_imports = {}
    for idx, obj_info in reversed(scene_objects_info.items()):
        obj_category = obj_info["category"]
        obj_model = obj_info["model"]
        obj_name = obj_info["name"]

        # Make sure there are no "-" or " " in the string -- this causes downstream importing issues
        assert "-" not in obj_category
        assert " " not in obj_category

        obj_urdf_fpath = f"{sim_dir}/objects/{obj_category}/{obj_model}/urdf/{obj_model}.urdf"

        # Check if this is an articulated object
        is_articulated = obj_info.get("is_articulated", False)

        # The asset the import writes, and the manifest that says what it was made from. The decision is taken before the
        # import: the hashes come from the stage-11 folder, the settings and the import code this run uses.
        dataset_obj_dir = os.path.join(get_dataset_path(cfg.s13_usd.dataset_name), "objects", obj_category, obj_model)
        stage_obj_dir = os.path.join(out_dir, "objects", obj_category, obj_model)
        new_manifest = {
            **directory_hashes(os.path.join(sim_dir, "objects", obj_category, obj_model)),
            "import_settings_hash": settings_hash,
            "import_code_hash": code_hash,
            "scene_name": cfg.scene_name,
            "category": obj_category,
            "model": obj_model,
            "mass_kg": urdf_mass_kg(obj_urdf_fpath),
        }
        existing = read_manifest(dataset_obj_dir)
        decision = decide_import(os.path.isdir(dataset_obj_dir), existing, new_manifest)
        apply_conflict_policy(
            decision,
            on_conflict,
            describe=partial(describe_decision, asset_dir=dataset_obj_dir, existing=existing, new=new_manifest),
        )
        asset_imports[obj_name] = {
            "action": decision["action"], "conflict": decision["conflict"], "reason": decision["reason"],
        }
        if skip_unchanged and decision["action"] == "skip":
            shutil.copytree(dataset_obj_dir, stage_obj_dir, dirs_exist_ok=True)
            logger.info(f"Skipped the import of {obj_name}: {decision['reason']}; copied the asset to {stage_obj_dir}")
            continue

        # The old manifest goes before the import overwrites the asset, so a failed import leaves none. The next run then
        # reports unknown provenance rather than trusting a manifest that describes the asset it replaced.
        remove_manifest(dataset_obj_dir)

        # Import
        # Uses simfoundry.utils.og_asset_import rather than OmniGibson's own
        # `examples.objects.import_custom_object`: that example passes `keep_instanceable=` to
        # import_og_asset_from_urdf, which current OmniGibson main no longer declares, so it
        # fails with TypeError before importing anything. Our module reproduces the URDF path of
        # that example and filters arguments against the installed signature, so it works across
        # the OmniGibson revisions this repo can have checked out. See that module's docstring.
        import_cmd = [
            "python",
            "-m", "simfoundry.utils.og_asset_import",
            "--dataset-name", cfg.s13_usd.dataset_name,
            "--asset-path", obj_urdf_fpath,
            "--category", obj_category,
            "--model", obj_model,
            "--collision-method", "none",
            # keep_instanceable is intentionally omitted (equivalent to the old
            # --no_keep_instanceable), matching upstream's _ALLOW_INSTANCING = False.
            "--overwrite",
        ]
        if asset_pipeline_materials:
            import_cmd.append("--asset-pipeline-materials")
        subprocess.run(import_cmd, check=True)

        # For articulated objects, reparent joints in the USD file
        # OmniGibson expects joints to be children of their parent link prims,
        # but Isaac Sim's URDF importer places them in a separate /joints scope
        usd_path = imported_usd_path(cfg.s13_usd.dataset_name, obj_category, obj_model)
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"Import produced no USD for {obj_name} at: {usd_path}")

        if is_articulated:
            if os.path.exists(usd_path):
                logger.info(f"Reparenting joints for articulated object: {obj_name}")
                # Run reparenting as a separate subprocess (needs its own SimulationApp instance)
                reparent_script = resolve_reparent_script()
                subprocess.run(["python", reparent_script, usd_path], check=True)
            else:
                logger.warning(f"USD file not found for reparenting: {usd_path}")

        # TRELLIS.2 bakes transparency into its textures, so the importer wires them up as
        # OmniPBR_Opacity materials whose opacity_threshold defaults to 0.0 -- alpha blending,
        # which renders the object as a translucent cloud of particles. A non-zero threshold
        # switches the material to alpha cutout so it renders solid.
        if opacity_threshold is not None:
            if os.path.exists(usd_path):
                logger.info(f"Setting material opacity threshold for {obj_name}: {opacity_threshold}")
                subprocess.run([
                    "python", resolve_pipeline_script("set_usd_opacity_threshold.py"),
                    usd_path, "--threshold", str(opacity_threshold),
                ], check=True)
            else:
                logger.warning(f"USD file not found for opacity threshold: {usd_path}")

        # The manifest is written once the import and its post-processing are done, so a failed one leaves none
        # (remove_manifest cleared the old one before the import). It is written before the copy below, so the
        # scene's copy carries the same manifest as the dataset folder.
        write_manifest(dataset_obj_dir, new_manifest)

        # The importer writes into the shared BEHAVIOR dataset folder; keep a copy of the
        # finished asset (post reparent/opacity) with the scene's own outputs as well.
        shutil.copytree(dataset_obj_dir, stage_obj_dir, dirs_exist_ok=True)
        logger.info(f"Copied imported asset to stage output: {stage_obj_dir}")

    logger.info("="*60)
    logger.info("USD import complete!")
    logger.info("="*60)

    finalize_stage(
        stage_cfg=cfg.s13_usd,
        out_dir=cfg.s13_usd.out_dir,
        result=StageResult(success=True, additional_info={"asset_imports": asset_imports}),
    )

    # Bypass OmniGibson/Isaac Sim teardown — avoids SIGSEGV in libomni.syntheticdata.plugin.so
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
