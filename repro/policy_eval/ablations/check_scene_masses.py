# SPDX-License-Identifier: Apache-2.0
"""Check that a scene's recorded masses reach PhysX (repro/README.md, chapter 18).

Builds the scene the way 1_eval_policy_og_scene.py does, applies simfoundry.utils.og_utils.apply_scene_masses
while the simulator is stopped, plays, and prints each changed object's mass and inertia as the physics view
reports them.

    python repro/policy_eval/ablations/check_scene_masses.py scenes/DROID/<scene>/<scene>_scene_state_mug300g.json
"""
import json
import os
import sys

import omnigibson as og
import torch as th
from simfoundry import ASSET_DIR, CFG_DIR, import_og_dependencies
from simfoundry.utils.og_utils import apply_scene_masses
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths
from omnigibson.utils.config_utils import parse_config

import_og_dependencies()
scene_path = os.path.join(ASSET_DIR, sys.argv[1])
scene = load_json_with_absolute_usd_paths(scene_path)
resolved = os.path.join(os.getcwd(), "check_masses_scene.json")
with open(resolved, "w") as f:
    json.dump(scene, f)
task_cfg = parse_config(os.path.join(CFG_DIR, "task", "load_scene.yaml"))["og_task_config"]
env = og.Environment(configs=dict(
    env={"action_frequency": 15, "rendering_frequency": 15, "physics_frequency": 120},
    scene={"type": "Scene", "scene_file": resolved, "use_floor_plane": True, "floor_plane_visible": False,
           "use_skybox": False, "include_robots": True},
    task=task_cfg))
env.reset()
names = [n for n, i in scene["objects_info"]["init_info"].items() if i["args"].get("mass") is not None]
before = {n: float(env.scene.object_registry("name", n).root_link.mass) for n in names}
with og.sim.stopped():
    changed = apply_scene_masses(env, scene)
for n in names:
    link = env.scene.object_registry("name", n).root_link
    view = link._rigid_prim_view
    print(f"[masses] {n}: view mass {before[n]:.4f} -> {float(link.mass):.4f} kg (recorded {changed.get(n)}), "
          f"view inertia {view.get_inertias().reshape(-1)[:9].tolist()}", flush=True)
og.shutdown()
