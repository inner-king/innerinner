# SPDX-License-Identifier: Apache-2.0
"""Load a DROID-desk scene the way the eval stage does and report what the policy will see.

Builds the same environment as scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py (the
nv_franka_droid external cameras at 1280x720, floor plane hidden, skybox, the task YAML), resets it, lets
physics run for two seconds, and then:
  - prints each task object's mass, size, and how far it moved while settling;
  - prints the task's goal and milestone states (all should be false at the start);
  - saves every camera's RGB frame to --out, so the policy's view of our scene can be put next to the
    authors' scene (run it on both).

Runs in the simfoundry env and boots Isaac Sim (about a minute, 3-4 GB of VRAM):
    python repro/policy_eval_ours/check_scene.py --spec repro/policy_eval_ours/ours.yaml \\
        --task cup_in_bowl_ours --out results/diagnostics/policy_eval_ours/check
"""
import argparse
import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def lazy_get_tonemap():
    import omnigibson.lazy as lazy
    return lazy.carb.settings.get_settings().get("/rtx/post/tonemap/op")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=os.path.join(HERE, "ours.yaml"))
    ap.add_argument("--task", required=True, help="key under tasks: in the spec")
    ap.add_argument("--task-yaml", default=None, help="default: <spec dir>/tasks/<task>.yaml")
    ap.add_argument("--scene", default=None, help="scene JSON under the asset dir; default: the task's")
    ap.add_argument("--out", required=True)
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--tonemap", type=int, default=None,
                    help="RTX tone-mapping operator (/rtx/post/tonemap/op), e.g. 1 linear; default: Isaac Sim's (6, ACES)")
    ap.add_argument("--hide-robot", action="store_true",
                    help="render without the simulated robot, to see what the background itself shows where it stands")
    args = ap.parse_args()

    import omnigibson as og
    import torch as th
    from omnigibson.macros import gm
    from omnigibson.utils.config_utils import parse_config
    from PIL import Image
    from simfoundry import ASSET_DIR, CFG_DIR, import_og_dependencies
    from simfoundry.utils.og_utils import apply_scene_masses
    from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

    gm.ENABLE_CCD = True  # as in the eval stage
    import_og_dependencies()
    spec = yaml.safe_load(open(args.spec))["tasks"][args.task]
    task_yaml = args.task_yaml or os.path.join(os.path.dirname(os.path.abspath(args.spec)), "tasks",
                                               f"{spec['task']}.yaml")
    scene_rel = args.scene or spec["scene"]
    os.makedirs(args.out, exist_ok=True)
    resolved = os.path.join(os.path.abspath(args.out), "scene_resolved.json")
    with open(resolved, "w") as f:
        json.dump(load_json_with_absolute_usd_paths(os.path.join(ASSET_DIR, scene_rel)), f)

    sensors = parse_config(os.path.join(CFG_DIR, "external_sensors", "nv_franka_droid.yaml"))["external_sensors"]
    for s in sensors:
        if "sensor_kwargs" in s:
            s["sensor_kwargs"]["image_height"], s["sensor_kwargs"]["image_width"] = 720, 1280
    env = og.Environment(configs=dict(
        env={"external_sensors": sensors, "action_frequency": 15, "rendering_frequency": 15, "physics_frequency": 120},
        scene={"type": "Scene", "scene_file": resolved, "use_floor_plane": True, "floor_plane_visible": False,
               "use_skybox": True, "include_robots": True},
        task=parse_config(task_yaml)["og_task_config"]))
    # The masses the scene records, applied as the eval stage applies them (stopped, before the first reset).
    with og.sim.stopped():
        with open(resolved) as f:
            for n, (old, new) in apply_scene_masses(env, json.load(f)).items():
                print(f"[mass] {n}: {old:.3f} -> {new:.3f} kg (from the scene)")
    env.reset()
    if args.hide_robot:
        env.robots[0].visible = False
    if args.tonemap is not None:
        import omnigibson.lazy as lazy
        lazy.carb.settings.get_settings().set("/rtx/post/tonemap/op", args.tonemap)
    print(f"[check] tonemap op {lazy_get_tonemap()}")

    names = [o["name"] for o in spec["init_states"]["objects"]]
    objs = {n: env.scene.object_registry("name", n) for n in names}
    start = {n: o.get_position_orientation()[0].cpu().numpy() for n, o in objs.items()}
    for _ in range(int(args.settle_s * 120)):
        og.sim.step_physics()
    # Render settings such as the tone mapper reach the sensors' outputs only after several frames.
    for _ in range(20):
        og.sim.render()
    print(f"[check] {args.task}: {scene_rel}")
    for n, o in objs.items():
        pos = o.get_position_orientation()[0].cpu().numpy()
        ext = (o.aabb[1] - o.aabb[0]).cpu().numpy()
        print(f"  {n}: mass {float(o.mass) * 1000:.1f} g, aabb {np.round(ext * 100, 1).tolist()} cm, "
              f"z {pos[2]:.3f}, settled by {np.linalg.norm(pos - start[n]) * 1000:.1f} mm")

    obs, _ = env.get_obs()
    task = env.task
    for kind in ("goal", "milestone"):
        fn = getattr(task, f"_check_{kind}s", None) or getattr(task, f"check_{kind}s", None)
        if fn is not None:
            try:
                print(f"  {kind}s at start: {fn(env)}")
            except TypeError:
                pass

    def save_rgb(tree, prefix):
        for k, v in tree.items():
            if isinstance(v, dict):
                save_rgb(v, f"{prefix}{k}__")
            elif k == "rgb":
                img = v[..., :3].cpu().numpy().astype(np.uint8)
                Image.fromarray(img).save(os.path.join(args.out, f"{prefix.rstrip('_')}.png"))
                print(f"  saved {prefix.rstrip('_')}.png {img.shape[1]}x{img.shape[0]}")

    save_rgb(obs, "")
    og.shutdown()


if __name__ == "__main__":
    main()
