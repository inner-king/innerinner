# SPDX-License-Identifier: Apache-2.0
"""Lower the gripper straight down in front of the objects and report where/why it stops."""
import sys, os
from pathlib import Path
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repro/, for scripted_pick_place
import numpy as np
import scripted_pick_place as spp
import omnigibson as og
from omnigibson.utils.config_utils import parse_config
os.chdir(spp.CFG_DIR)
scene_json = spp.load_json_with_absolute_usd_paths(str(spp.DATA / "put_marker_in_cup_v2/s14_og/reconstructed_og_scene.json"))
reg = scene_json["state"]["registry"]["object_registry"]
reg["robot0"]["root_link"]["pos"] = [-0.682, 0.058, 0.0]
env = og.Environment(configs={"env": {"action_frequency": 15, "rendering_frequency": 15, "physics_frequency": 120},
    "scene": {"type": "Scene", "scene_file": scene_json, "use_floor_plane": True, "floor_plane_visible": False, "use_skybox": True, "include_robots": True}})
env.reset()
d = spp.Demo(env); r = d.robot
print("floor/ground objects:", [o.name for o in env.scene.objects if "floor" in o.name or "ground" in o.name or "plane" in o.name])
for o in env.scene.objects:
    if o.name.startswith("iter") or o.name == "robot0":
        lo, hi = (spp.np_(v) for v in o.aabb); print(f"  {o.name:8s} aabb z [{lo[2]:.3f}, {hi[2]:.3f}] xy [{lo[:2].round(3)}, {hi[:2].round(3)}]")
for i in range(45):
    d.step(np.array([0, 0, -0.01, 0, 0, 0]), 1.0)
    if i % 5 == 4:
        p, _ = d.eef()
        contacts = sorted({c for c in (getattr(link, "contact_list", lambda: [])() and [] or []) })
        q = spp.np_(r.get_joint_positions())[:7]
        print(f"step {i+1:2d} eef {p.round(3)} q {np.degrees(q).round(0)}", flush=True)
# what is the hand touching?
from omnigibson.utils.sim_utils import prims_to_rigid_prim_set
try:
    touching = set()
    for link in r.links.values():
        for c in link.contact_list():
            touching.add((link.name, c.body1 if c.body0.endswith(link.name) else c.body0))
    print("contacts:", sorted(touching)[:12])
except Exception as e:
    print("contact query failed:", e)
og.shutdown()
