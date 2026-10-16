# SPDX-License-Identifier: Apache-2.0
"""Probe how the scene's IK arm controller responds to pose_delta_ori commands."""
import sys
from pathlib import Path
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repro/, for scripted_pick_place
import numpy as np, torch as th
import scripted_pick_place as spp
import omnigibson as og
from omnigibson.utils.config_utils import parse_config
import os
os.chdir(spp.CFG_DIR)
scene_json = spp.load_json_with_absolute_usd_paths(str(spp.DATA / "put_marker_in_cup_v2/s14_og/reconstructed_og_scene.json"))
cams = parse_config(str(Path(spp.CFG_DIR) / "external_sensors" / "sim_franka_teleop.yaml"))["external_sensors"]
env = og.Environment(configs={"env": {"external_sensors": cams, "action_frequency": 15, "rendering_frequency": 15, "physics_frequency": 120},
    "scene": {"type": "Scene", "scene_file": scene_json, "use_floor_plane": True, "floor_plane_visible": False, "use_skybox": True, "include_robots": True}})
env.reset()
d = spp.Demo(env); r = d.robot
print("fixed_base:", r.fixed_base, "base pose:", [np.round(spp.np_(x), 3) for x in r.get_position_orientation()])
print("arm idx", d.arm_idx, "grip idx", d.grip_idx, "action_dim", r.action_dim)
ctrl = r.controllers[f"arm_{d.arm}"]
print("ctrl:", type(ctrl).__name__, "mode", ctrl.mode, "input limits", getattr(ctrl, "_command_input_limits", None), "output limits", getattr(ctrl, "_command_output_limits", None))
for axis, vec in [("+x", [0.01, 0, 0]), ("+z", [0, 0, 0.01]), ("-z", [0, 0, -0.01])]:
    p0, _ = d.eef()
    for i in range(10):
        d.step(np.array(vec + [0, 0, 0]), 1.0)
    p1, _ = d.eef()
    print(f"cmd {axis} x10 -> eef moved {np.round((p1 - p0) * 100, 2)} cm", flush=True)
for i in range(30):
    d.step(np.zeros(6), 1.0)
p2, _ = d.eef()
print("after 30 zero steps drift (cm):", np.round((p2 - p1) * 100, 2), flush=True)
og.shutdown()
