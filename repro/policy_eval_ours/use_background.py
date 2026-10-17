# SPDX-License-Identifier: Apache-2.0
"""Write a copy of a DROID scene state that loads another mesh background, e.g. droid_v1 without its scanned robot.

The copy goes next to the scene as `<scene>_scene_state_<tag>.json` (the saved-scene naming), so every relative
path in it, the objects' and the background's, stays valid, and the scene's own files are not touched (AGENTS.md
rule 4). Only the mesh background's `usd_path` and `expected_file_hash` change; the background's pose is kept, so a
layer over the same scan (mask_scanned_robot.py) lands where the original did.

    python repro/policy_eval_ours/use_background.py --tag norobot \\
        --background assets/backgrounds/mesh_backgrounds/droid_v1_norobot.usd \\
        assets/scenes/DROID/droid_desk_put_marker_in_cup/droid_desk_put_marker_in_cup_scene_state_latest.json
"""
import argparse
import hashlib
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="+", help="scene state JSONs (…_scene_state_<tag>.json)")
    ap.add_argument("--background", required=True, help="the background .usd to load instead")
    ap.add_argument("--tag", required=True, help="the copy is <scene>_scene_state_<tag>.json")
    args = ap.parse_args()
    bg_hash = hashlib.md5(open(args.background, "rb").read()).hexdigest()
    for path in args.scenes:
        scene = json.load(open(path))
        names = [n for n in scene["objects_info"]["init_info"] if n.startswith("mesh_background")]
        if len(names) != 1:
            raise SystemExit(f"{path}: expected one mesh background, found {names}")
        bg_args = scene["objects_info"]["init_info"][names[0]]["args"]
        old = bg_args["usd_path"]
        bg_args["usd_path"] = os.path.relpath(os.path.abspath(args.background), os.path.dirname(os.path.abspath(path)))
        bg_args["expected_file_hash"] = bg_hash
        stem = os.path.basename(path).rsplit("_scene_state_", 1)[0]
        out = os.path.join(os.path.dirname(path), f"{stem}_scene_state_{args.tag}.json")
        with open(out, "w") as f:
            json.dump(scene, f, indent=2)
        print(f"[background] {out}: {old} -> {bg_args['usd_path']}")


if __name__ == "__main__":
    main()
