# SPDX-License-Identifier: Apache-2.0
"""Check that a robot's finger frictions survive patches/BEHAVIOR-1K-link-physics-materials.patch (repro/README.md, 20.4).

ManipulationRobot writes finger_static_friction / finger_dynamic_friction into the object's link_physics_materials
after BaseObject has stored it, and the patch made that stored dict a copy. Builds two Frankas: one with finger
frictions only, as OmniGibson's own configs give them, and one that also passes link_physics_materials with an entry
for one finger (restitution only), so the robot has to merge into what the caller gave. Reports what PhysX holds for
every finger shape, what each robot's init info keeps, and whether the scene still saves.

Runs in the simfoundry env with this checkout first on PYTHONPATH, and boots Isaac Sim (about a minute):
    PYTHONPATH=$PWD python repro/friction/check_finger_friction.py --out results/diagnostics/friction/finger_friction.json
"""
import argparse
import json
import os
import tempfile
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="JSON report")
    ap.add_argument("--static", type=float, default=1.5, help="finger_static_friction")
    ap.add_argument("--dynamic", type=float, default=1.2, help="finger_dynamic_friction")
    ap.add_argument("--restitution", type=float, default=0.3, help="restitution the caller gives the left finger")
    args = ap.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    from simfoundry import configure_omnigibson_data_path

    configure_omnigibson_data_path(force=True)

    import omnigibson as og
    import torch as th
    from simfoundry.utils.og_utils import og_copies_link_physics_materials

    def physx_materials(link):
        """What PhysX holds for every collision shape of @link: (static, dynamic, restitution) and a shape count."""
        view = link._rigid_prim_view._physics_view
        props = th.as_tensor(view.get_material_properties()).reshape(view.count, view.max_shapes, 3)[0]
        rows = [tuple(round(float(v), 4) for v in row) for row in props]
        return [dict(zip(("static", "dynamic", "restitution"), key), shapes=n) for key, n in Counter(rows).items()]

    given = {"panda_leftfinger": {"restitution": args.restitution}}
    robot = dict(type="FrankaPanda", end_effector="gripper", obs_modalities=[], action_normalize=False,
                 finger_static_friction=args.static, finger_dynamic_friction=args.dynamic)
    env = og.Environment(configs=dict(
        scene={"type": "Scene", "use_floor_plane": True, "floor_plane_visible": True, "use_skybox": False},
        robots=[dict(robot, name="fingers_only", position=[0.0, -1.0, 0.0]),
                dict(robot, name="fingers_and_kwarg", position=[0.0, 1.0, 0.0], link_physics_materials=given)]))
    env.reset(get_obs=False)
    for _ in range(5):
        og.sim.step()

    report = {"og_copies_link_physics_materials": og_copies_link_physics_materials(),
              "finger_static_friction": args.static, "finger_dynamic_friction": args.dynamic,
              "given_link_physics_materials": {"panda_leftfinger": {"restitution": args.restitution}},
              "robots": {}}
    for rob in env.robots:
        entry = rob.get_init_info()["args"].get("link_physics_materials")
        report["robots"][rob.name] = {
            "fingers": {name: physx_materials(rob.links[name]) for names in rob.finger_link_names.values()
                        for name in names},
            # Should be what the config gave, untouched by the robot's finger frictions and by the materials made.
            "init_info": entry if entry is None or all(isinstance(v, dict) for v in entry.values())
            else {k: type(v).__name__ for k, v in entry.items()},
        }

    try:
        env.scene.save()
        report["scene_save"] = "ok"
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        report["scene_save"] = f"{type(e).__name__}: {e}"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            og.sim.save(json_paths=[os.path.join(tmp, "scene.json")])
            report["og_sim_save"] = "ok"
        except Exception as e:  # noqa: BLE001
            report["og_sim_save"] = f"{type(e).__name__}: {e}"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    os._exit(0)  # OmniGibson's atexit teardown is slow and sometimes hangs; the report is written


if __name__ == "__main__":
    main()
