# SPDX-License-Identifier: Apache-2.0
"""Write a copy of the Marker in Cup scene whose mug weighs what a ceramic mug does (repro/README.md, chapter 18).

The published scene's cream_mug carries the 80 g its USD authors; a ceramic mug of its size (9.4 x 7.3 x 8.8
cm) weighs about 300 g. The copy records the mass at objects_info.init_info[<mug>].args.mass, which
1_eval_policy_og_scene.py applies after load (simfoundry.utils.og_utils.apply_scene_masses). It is written
next to the original so its relative USD paths still resolve; the original and the USD are not touched.

    python repro/policy_eval/ablations/make_mug_mass_scene.py assets/scenes/DROID/droid_desk_put_marker_in_cup 0.30
"""
import json
import os
import sys

scene_dir, mass = sys.argv[1], float(sys.argv[2])
name = os.path.basename(os.path.normpath(scene_dir))
src = os.path.join(scene_dir, f"{name}_scene_state_latest.json")
dst = os.path.join(scene_dir, f"{name}_scene_state_mug{int(round(mass * 1000))}g.json")
scene = json.load(open(src))
mugs = [n for n, i in scene["objects_info"]["init_info"].items() if i["args"].get("category") == "cream_mug"]
assert len(mugs) == 1, mugs
scene["objects_info"]["init_info"][mugs[0]]["args"]["mass"] = mass
with open(dst, "w") as f:
    json.dump(scene, f, indent=2)
print(f"{mugs[0]}: mass {mass} kg -> {dst}")
