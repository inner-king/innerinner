# SPDX-License-Identifier: Apache-2.0
"""Starts for a scene copy in which one object was swapped for another (repro/README.md, chapter 30.7).

A scene built with one of our objects swapped for an authors' one (build_droid_scene.py) keeps the original scene's
25 starts, so the two scenes differ in that object alone. Each start keeps the swapped object's xy in that episode;
its height and orientation become the new object's resting pose in the new scene, since the two objects differ in
size and in how their meshes are oriented. The other objects' entries are copied unchanged.

    python repro/policy_eval_ours/swap_init_object.py --init <starts.json> --scene <new scene state JSON> \\
        --old orange_bowl_sfufgv_1 --new orange_bowl_ubzmrk_1 --task cup_in_bowl_ours_real_authbowl --out <new.json>
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="the original scene's init-state file")
    ap.add_argument("--scene", required=True, help="the new scene state JSON (absolute, or under assets/)")
    ap.add_argument("--old", required=True, help="the object's name in the original scene")
    ap.add_argument("--new", required=True, help="the name of the object that replaces it in the new scene")
    ap.add_argument("--task", required=True, help="the new scene's task key, recorded in the file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    starts = json.load(open(args.init))
    scene = json.load(open(args.scene))
    root = scene["state"]["registry"]["object_registry"][args.new]["root_link"]
    for st in starts["init_states"]:
        old = st["objects"].pop(args.old)
        st["objects"][args.new] = {"pos": [old["pos"][0], old["pos"][1], root["pos"][2]], "ori": list(root["ori"])}
        if args.old in st.get("cells", {}):
            st["cells"][args.new] = st["cells"].pop(args.old)
        st.pop("check", None)  # the physics check was of the old object
    for o in starts.get("config", {}).get("objects", []):
        if o.get("name") == args.old:
            o["name"] = args.new
    starts["task"] = args.task
    starts["scene"] = os.path.relpath(args.scene, os.path.join(os.path.dirname(__file__), "..", "..", "assets")) \
        if os.path.isabs(args.scene) else args.scene
    starts["derived_from"] = {"init": os.path.basename(args.init), "swapped": {args.old: args.new}}
    with open(args.out, "w") as f:
        json.dump(starts, f, indent=1)
    print(f"{args.out}: {len(starts['init_states'])} starts, {args.old} -> {args.new} at z {root['pos'][2]:.4f}")


if __name__ == "__main__":
    main()
