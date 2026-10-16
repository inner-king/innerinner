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
r = env.robots[0]
def report(tag):
    for _ in range(5): og.sim.step()
    print(tag, "| root", spp.np_(r.get_position_orientation()[0]).round(3),
          "| panda_link0", spp.np_(r.links["panda_link0"].get_position_orientation()[0]).round(3),
          "| panda_hand", spp.np_(r.links["panda_hand"].get_position_orientation()[0]).round(3),
          "| eef", spp.np_(r.get_eef_position()).round(3), flush=True)
report("after reset (JSON pose -0.682)")
import torch as th
r.set_position_orientation(position=th.tensor([-0.682, 0.058, 0.0]), orientation=th.tensor([0., 0., 0., 1.]))
report("after set_position_orientation")
og.sim.stop(); og.sim.play()
report("after sim stop/play")
og.shutdown()
