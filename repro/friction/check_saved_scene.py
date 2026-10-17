# SPDX-License-Identifier: Apache-2.0
"""Load a saved scene the way teleop and policy evaluation do, then save it again (repro/README.md, section 20.4).

A scene records friction at objects_info.init_info[<name>].args.link_physics_materials, which OmniGibson takes as a
constructor kwarg and binds at load. This builds an og.Environment with the scene JSON as its scene_file (as
2_teleop_og_scene.py and 1_eval_policy_og_scene.py do) and reports, per object, what the scene records next to what
PhysX holds for every collision shape. Then it saves the scene the three ways a consumer does: scene.save(),
og.sim.save(), and the HDF5CollectionWrapper both stages record through, whose constructor stores env.scene.save()
and the environment config. Unpatched OmniGibson 3.8.0 fails all three on a scene that records a friction
(patches/BEHAVIOR-1K-link-physics-materials.patch).

Runs in the simfoundry env with this checkout first on PYTHONPATH, and boots Isaac Sim (about a minute):
    PYTHONPATH=$PWD python repro/friction/check_saved_scene.py \\
        --scene Data/put_cup_in_bowl_v2/s14_og_friction_record/reconstructed_og_scene.json \\
        --out results/diagnostics/friction/saved_scene_reload.json
"""
import argparse
import copy
import json
import os
import tempfile
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="saved scene JSON, e.g. a stage 14 reconstructed_og_scene.json")
    ap.add_argument("--out", required=True, help="JSON report")
    args = ap.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    from simfoundry import configure_omnigibson_data_path

    configure_omnigibson_data_path(force=True)

    import omnigibson as og
    import torch as th
    from omnigibson.envs import HDF5CollectionWrapper
    from omnigibson.robots import BaseRobot
    from simfoundry.utils.og_utils import og_copies_link_physics_materials
    from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

    def physx_materials(obj):
        """What PhysX holds for every collision shape of @obj: (static, dynamic, restitution) and a shape count."""
        rows = []
        for link in obj.links.values():
            view = link._rigid_prim_view._physics_view
            props = th.as_tensor(view.get_material_properties()).reshape(view.count, view.max_shapes, 3)[0]
            rows += [tuple(round(float(v), 4) for v in row) for row in props]
        return [dict(zip(("static", "dynamic", "restitution"), key), shapes=n) for key, n in Counter(rows).items()]

    scene_json = load_json_with_absolute_usd_paths(args.scene)
    init_info = scene_json["objects_info"]["init_info"]
    recorded = {name: info["args"].get("link_physics_materials") for name, info in init_info.items()}
    pristine = copy.deepcopy(recorded)
    report = {"scene": os.path.abspath(args.scene),
              "og_copies_link_physics_materials": og_copies_link_physics_materials(),
              "objects": {}}

    env = og.Environment(configs=dict(scene={
        "type": "Scene", "scene_file": scene_json, "use_floor_plane": True, "floor_plane_visible": True,
        "use_skybox": False, "include_robots": True}))
    env.reset(get_obs=False)
    for _ in range(5):
        og.sim.step()

    for obj in env.scene.objects:
        if isinstance(obj, BaseRobot) or not obj.links:
            continue
        entry = obj.get_init_info()["args"].get("link_physics_materials")
        report["objects"][obj.name] = {
            "recorded": pristine.get(obj.name),
            "physx": physx_materials(obj),
            # dict when OmniGibson left the kwarg alone; PhysicsMaterial where it wrote back into it
            "init_info_value_types": sorted({type(v).__name__ for v in entry.values()}) if entry else None,
        }
    # The scene_file dict itself: unpatched, its entries are the ones OmniGibson replaced.
    report["scene_file_dict_untouched"] = recorded == pristine and all(
        isinstance(v, dict) for m in recorded.values() if m for v in m.values())

    def attempt(key, fn):
        try:
            fn()
            report[key] = "ok"
        except Exception as e:  # noqa: BLE001 - recorded, not raised
            report[key] = f"{type(e).__name__}: {e}"

    attempt("scene_save", env.scene.save)
    with tempfile.TemporaryDirectory() as tmp:
        saved_path = os.path.join(tmp, "scene.json")
        attempt("og_sim_save", lambda: og.sim.save(json_paths=[saved_path]))
        if report["og_sim_save"] == "ok":
            with open(saved_path) as f:
                saved = json.load(f)["objects_info"]["init_info"]
            report["resaved_equals_recorded"] = all(
                saved[name]["args"].get("link_physics_materials") == pristine[name] for name in report["objects"])

        def wrap():
            wrapper = HDF5CollectionWrapper(env=env, output_path=os.path.join(tmp, "demos.hdf5"),
                                            only_successes=False, viewport_camera_path=None)
            wrapper.close_dataset()

        attempt("hdf5_collection_wrapper", wrap)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    os._exit(0)  # OmniGibson's atexit teardown is slow and sometimes hangs; the report is written


if __name__ == "__main__":
    main()
