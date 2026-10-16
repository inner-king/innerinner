# SPDX-License-Identifier: Apache-2.0
import sys, os
from pathlib import Path
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repro/, for scripted_pick_place
import numpy as np
import scripted_pick_place as spp
import omnigibson as og
os.chdir(spp.CFG_DIR)
scene_json = spp.load_json_with_absolute_usd_paths(str(spp.DATA / "put_marker_in_cup_v2/s14_og/reconstructed_og_scene.json"))
scene_json["state"]["registry"]["object_registry"]["robot0"]["root_link"]["pos"] = [-0.682, 0.058, 0.0]
env = og.Environment(configs={"env": {"action_frequency": 15, "rendering_frequency": 15, "physics_frequency": 120},
    "scene": {"type": "Scene", "scene_file": scene_json, "use_floor_plane": True, "floor_plane_visible": False, "use_skybox": True, "include_robots": True}})
env.reset()
d = spp.Demo(env); r = d.robot
p, R = d.eef()
hand = spp.np_(r.links["panda_hand"].get_position_orientation()[0])
approach_w = (p - hand) / np.linalg.norm(p - hand)
approach_l = R.T @ approach_w
print("approach axis (hand->eef) local:", approach_l.round(3), " world:", approach_w.round(3))
closing_l = R.T @ d.finger_axis_world()
print("closing axis local:", closing_l.round(3))
# canonical: approach local -> world down, closing local -> world +y
a_l = approach_l / np.linalg.norm(approach_l); c_l = closing_l - closing_l.dot(a_l) * a_l; c_l /= np.linalg.norm(c_l)
Rd = np.stack([[0, 0, -1.], [0, 1., 0], np.cross([0, 0, -1.], [0, 1., 0])], 1) @ np.stack([a_l, c_l, np.cross(a_l, c_l)], 1).T
marker = np.array([-0.143, 0.17, 0.0105])
for tag, tgt, n in [("orient down", p, 80), ("above marker", marker + [0, 0, 0.15], 120), ("descend", marker + [0, 0, 0.0], 150)]:
    d.servo(np.asarray(tgt), Rd, 1.0, max_steps=n, label=tag)
og.shutdown()
