# SPDX-License-Identifier: Apache-2.0
"""What happened to the objects in each episode of a Table G.1 run (repro/README.md, chapter 18).

1_eval_policy_og_scene.py records the serialized simulator state of every step in rollouts.hdf5 (one
"demo" per env.reset(); the episodes are the long ones). This rebuilds the eval's environment -- same
scene, task, and the JointController / MultiFingerGripperController the eval stage swaps in, since the
controllers' state is part of the serialized vector -- loads each recorded state, and reports per episode
and per movable object: the largest tilt from upright, the largest lift above its start height, how far it
ended from where it started, and its height at the end; for articulated props (drawers, lids), each joint's
position as a fraction of its range at the start, its extremes, and at the end. That separates failures the policy made (never
grasped, dropped the object, timed out) from ones the scene makes easy (a light cup knocked over, an object
pushed off the table).

    python repro/policy_eval/analyze_rollouts.py --task marker_in_cup --results <results dir> --out analysis.json
"""
import argparse
import json
import math
import os

import h5py
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def tilt_deg(q):
    """Angle between the object's z axis and world z, for an xyzw quaternion."""
    x, y, z, w = q
    zz = 1 - 2 * (x * x + y * y)  # world-z component of the body z axis
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=os.path.join(HERE, "table_g1.yaml"))
    ap.add_argument("--task", required=True)
    ap.add_argument("--results", required=True, help="results_<timestamp> directory of one run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3, help="analyse every n-th step")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.spec))["tasks"][args.task]
    eval_results = json.load(open(os.path.join(args.results, "eval_results.json"))) \
        if os.path.exists(os.path.join(args.results, "eval_results.json")) else None

    import omnigibson as og
    import torch as th
    from omnigibson.macros import gm
    from omnigibson.utils.config_utils import parse_config
    from simfoundry import ASSET_DIR, import_og_dependencies
    from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

    gm.ENABLE_CCD = True
    import_og_dependencies()
    scene_json = load_json_with_absolute_usd_paths(os.path.join(ASSET_DIR, spec["scene"]))
    resolved = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"{args.task}_analysis_scene.json")
    with open(resolved, "w") as f:
        json.dump(scene_json, f)
    task_cfg = parse_config(os.path.join(HERE, "tasks", f"{spec['task']}.yaml"))["og_task_config"]
    env = og.Environment(configs=dict(
        env={"action_frequency": 15, "rendering_frequency": 15, "physics_frequency": 120},
        scene={"type": "Scene", "scene_file": resolved, "use_floor_plane": True, "floor_plane_visible": False,
               "use_skybox": False, "include_robots": True},
        task=task_cfg))
    env.reset()
    robot = env.robots[0]
    # The controllers 1_eval_policy_og_scene.py installs before recording (their state is serialized too).
    ctrl = {}
    for arm in robot.arm_names:
        ctrl[f"arm_{arm}"] = {"name": "JointController", "motor_type": "position", "command_input_limits": None,
                              "use_delta_commands": False, "smoothing_filter_size": 5}
        ctrl[f"gripper_{arm}"] = {"name": "MultiFingerGripperController", "command_input_limits": (0.0, 1.0),
                                  "mode": "smooth", "inverted": True}
    robot.reload_controllers(ctrl)

    movable = [o for o in env.scene.objects if not o.fixed_base and o is not robot
               and "background" not in o.name]
    # Articulated props (drawers, lids): their joints, as a fraction of each joint's range from its lower limit.
    articulated = [o for o in env.scene.objects if o is not robot and len(getattr(o, "joints", {})) > 0]
    f = h5py.File(os.path.join(args.results, "rollouts.hdf5"), "r", locking=False)
    data = f["data"]
    demos = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
    episodes = [k for k in demos if data[k]["state"].shape[0] > 1]

    out = []
    for e, key in enumerate(episodes):
        states = np.asarray(data[key]["state"])
        sizes = np.asarray(data[key]["state_size"])
        rec = {"episode": e, "demo": key, "steps": int(states.shape[0] - 1)}
        if eval_results is not None and e < len(eval_results["episodes"]):
            rec["success"] = bool(eval_results["episodes"][e]["success"])
            rec["milestones"] = eval_results["episodes"][e].get("milestones")
        track = {o.name: {"tilt": [], "z": [], "xy": []} for o in movable}
        joints = {(o.name, j): [] for o in articulated for j in o.joints}
        for t in list(range(0, states.shape[0], args.stride)) + [states.shape[0] - 1]:
            og.sim.load_state(th.tensor(states[t, : int(sizes[t])]), serialized=True)
            for o in movable:
                pos, ori = o.get_position_orientation()
                pos, ori = pos.cpu().numpy(), ori.cpu().numpy()
                track[o.name]["tilt"].append(tilt_deg(ori))
                track[o.name]["z"].append(float(pos[2]))
                track[o.name]["xy"].append(pos[:2].tolist())
            for o in articulated:
                for j, joint in o.joints.items():
                    lo, hi = float(joint.lower_limit), float(joint.upper_limit)
                    joints[(o.name, j)].append((float(joint.get_state()[0][0]) - lo) / max(hi - lo, 1e-9))
        rec["objects"] = {}
        for name, tr in track.items():
            z0, xy0 = tr["z"][0], np.array(tr["xy"][0])
            rec["objects"][name] = {
                "start_tilt_deg": round(tr["tilt"][0], 1),
                "max_tilt_change_deg": round(max(abs(t - tr["tilt"][0]) for t in tr["tilt"]), 1),
                "final_tilt_deg": round(tr["tilt"][-1], 1),
                "max_lift_m": round(max(tr["z"]) - z0, 3),
                "final_dz_m": round(tr["z"][-1] - z0, 3),
                "final_dxy_m": round(float(np.linalg.norm(np.array(tr["xy"][-1]) - xy0)), 3),
            }
        rec["joints"] = {f"{n}:{j}": {"start": round(v[0], 3), "max": round(max(v), 3), "min": round(min(v), 3),
                                      "final": round(v[-1], 3)} for (n, j), v in joints.items()}
        out.append(rec)
        objs = " | ".join(f"{n}: tilt {v['max_tilt_change_deg']:.0f} deg, lift {v['max_lift_m']:.2f}, "
                          f"dz {v['final_dz_m']:+.2f}, dxy {v['final_dxy_m']:.2f}" for n, v in rec["objects"].items())
        jts = " | ".join(f"{k.split(':')[1][-14:]}: start {v['start']:.2f} max {v['max']:.2f} final {v['final']:.2f}"
                         for k, v in rec["joints"].items())
        print(f"[analysis] ep {e} {'success' if rec.get('success') else 'fail'} {rec['steps']} steps: {objs}"
              + (f" || joints (fraction of range from lower limit) {jts}" if jts else ""), flush=True)

    with open(args.out, "w") as fo:
        json.dump(out, fo, indent=1)
    print(f"[analysis] {len(out)} episodes -> {args.out}", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
