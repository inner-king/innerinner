# SPDX-License-Identifier: Apache-2.0
"""Check that the friction stage 11 estimates reaches PhysX (repro/README.md, section 20.4).

Builds a reconstructed scene's objects the way stage 14 does (each DatasetObject at its stage-12 pose on the
ground plane) and reads the material of every collision shape back from PhysX (omni.physics.tensors), next to
the material bound to it in USD:
  before    nothing bound yet: what every object ran on until now;
  playing   after simfoundry.utils.og_utils.set_obj_materials, called while the simulator plays, as stage 14
            and B6 called it until now;
  replayed  after a stop and a play, when PhysX has parsed the stage afresh;
  stopped   a fresh copy of each object, added and bound inside one og.sim.stopped() block, as stage 14 and
            B6 now do it.
Then two experiments on probe objects placed away from the scene:
  slide     a copy of one object given 0.5 m/s along +x on the ground plane under several materials; it slides
            v^2 / (2 mu g), mu being the dynamic friction PhysX combines from the two surfaces (their mean);
  contract  an object built with OmniGibson's link_physics_materials kwarg, i.e. the saved-scene key
            objects_info.init_info[<name>].args.link_physics_materials, whose material PhysX should report;
            then the scene is serialized the way og.sim.save() and the DataCollectionWrapper do.

Runs in the simfoundry env with this checkout first on PYTHONPATH, and boots Isaac Sim (about a minute, ~3 GB
of VRAM):
    PYTHONPATH=$PWD python repro/friction/check_friction.py --scene-dir Data/put_cup_in_bowl_v2 \\
        --slide-object iter_0 --out results/diagnostics/friction/put_cup_in_bowl_v2.json
"""
import argparse
import json
import os
import tempfile
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", required=True, help="Data/<scene>, with s11_sim/ and s12_physics/")
    ap.add_argument("--out", required=True, help="JSON report")
    ap.add_argument("--dataset-name", default="real2sim-assets")
    ap.add_argument("--settle-steps", type=int, default=100, help="as stage 14")
    ap.add_argument("--slide-object", default=None, help="object name to copy for the slide test; default: the first")
    ap.add_argument("--slide-speed", type=float, default=0.5, help="m/s")
    args = ap.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    from simfoundry import configure_omnigibson_data_path

    configure_omnigibson_data_path(force=True)

    import omnigibson as og
    import omnigibson.lazy as lazy
    import torch as th
    from omnigibson.objects import DatasetObject
    from simfoundry.utils.og_utils import set_obj_materials

    with open(os.path.join(args.scene_dir, "s11_sim", "scene_objects_info.json")) as f:
        scene_objects_info = json.load(f)
    with open(os.path.join(args.scene_dir, "s12_physics", "pb_scene_poses.json")) as f:
        obj_poses = json.load(f)
    obj_frictions = {info["name"]: info["friction"] for info in scene_objects_info.values()}

    def grouped(rows):
        return [dict(zip(("static", "dynamic", "restitution"), key), shapes=n) for key, n in Counter(rows).items()]

    def physx_materials(obj):
        """What PhysX holds for every collision shape of @obj: (static, dynamic, restitution)."""
        rows = []
        for link in obj.links.values():
            view = link._rigid_prim_view._physics_view
            props = th.as_tensor(view.get_material_properties()).reshape(view.count, view.max_shapes, 3)[0]
            rows += [tuple(round(float(v), 4) for v in row) for row in props]
        return grouped(rows)

    def usd_materials(obj):
        """The physics material bound to every collision mesh of @obj in USD, None where nothing is."""
        rows = []
        for link in obj.links.values():
            for msh in link.collision_meshes.values():
                mat = msh.get_applied_physics_material()
                if mat is None:
                    rows.append((None, None, None))
                    continue
                api = lazy.pxr.UsdPhysics.MaterialAPI(mat.prim)
                rows.append(tuple(None if not a.HasAuthoredValue() else round(float(a.Get()), 4)
                                  for a in (api.GetStaticFrictionAttr(), api.GetDynamicFrictionAttr(),
                                            api.GetRestitutionAttr())))
        return grouped(rows)

    def ground_material():
        stage = og.sim.stage
        for prim in lazy.pxr.Usd.PrimRange(stage.GetPrimAtPath("/World/ground_plane")):
            if prim.HasAPI(lazy.pxr.UsdPhysics.CollisionAPI):
                binding = lazy.pxr.UsdShade.MaterialBindingAPI(prim).GetDirectBinding(materialPurpose="physics")
                api = lazy.pxr.UsdPhysics.MaterialAPI(stage.GetPrimAtPath(binding.GetMaterialPath()))
                return {"collider": str(prim.GetPath()), "material": str(binding.GetMaterialPath()),
                        "static": api.GetStaticFrictionAttr().Get(), "dynamic": api.GetDynamicFrictionAttr().Get(),
                        "restitution": api.GetRestitutionAttr().Get()}
        return None

    def snapshot(objs):
        return {name: {"physx": physx_materials(obj), "usd": usd_materials(obj)} for name, obj in objs.items()}

    report = {"scene_dir": os.path.abspath(args.scene_dir), "estimates": obj_frictions, "phases": {}}

    # The environment stage 14 builds, less the robot and the external cameras, which carry no object material.
    env = og.Environment(configs=dict(
        scene={"type": "Scene", "use_floor_plane": True, "floor_plane_visible": True, "use_skybox": False},
        robots=[]))
    env.reset(get_obs=False)
    scene = env.scene
    report["ground_plane"] = ground_material()

    # Stage 14's object loop, verbatim but for the material readback.
    objs = dict()
    with og.sim.stopped():
        for _, obj_info in reversed(list(scene_objects_info.items())):
            obj = DatasetObject(name=obj_info["name"], dataset_name=args.dataset_name,
                                category=obj_info["category"], model=obj_info["model"])
            scene.add_object(obj)
            pos = th.tensor(obj_poses[obj_info["name"]][0])
            pos[2] -= 0.005
            obj.set_position_orientation(position=pos, orientation=th.tensor(obj_poses[obj_info["name"]][1]))
            og.sim.step()
            objs[obj_info["name"]] = obj
            for _ in range(10):
                og.sim.step()
    report["link_names"] = {name: list(obj.links) for name, obj in objs.items()}

    report["phases"]["before"] = snapshot(objs)
    set_obj_materials(objs=objs, obj_frictions=obj_frictions)  # playing, where stage 14 and B6 called it until now
    for _ in range(2):
        og.sim.step()
    report["phases"]["playing"] = snapshot(objs)

    with og.sim.stopped():
        pass
    for _ in range(2):
        og.sim.step()
    report["phases"]["replayed"] = snapshot(objs)

    # Where stage 14 and B6 call it now: a fresh copy of each object, added and bound in one stopped block, 3 m to
    # the side of the scene.
    copies, copy_frictions = dict(), dict()
    with og.sim.stopped():
        for name in objs:
            src = next(info for info in scene_objects_info.values() if info["name"] == name)
            copy = DatasetObject(name=f"{name}_stopped", dataset_name=args.dataset_name, category=src["category"],
                                 model=src["model"])
            scene.add_object(copy)
            pos = th.tensor(obj_poses[name][0])
            pos[1] += 3.0
            copy.set_position_orientation(position=pos, orientation=th.tensor(obj_poses[name][1]))
            copies[copy.name], copy_frictions[copy.name] = copy, obj_frictions[name]
        set_obj_materials(objs=copies, obj_frictions=copy_frictions)
    for _ in range(2):
        og.sim.step()
    report["phases"]["stopped"] = snapshot(copies)

    # Stage 14's settle, now that PhysX has the estimates, with each object's pose and speed at its end.
    start = {name: obj.get_position_orientation()[0].clone() for name, obj in objs.items()}
    for _ in range(args.settle_steps):
        og.sim.step()
    report["settle"] = {}
    for name, obj in objs.items():
        pos, ori = obj.get_position_orientation()
        report["settle"][name] = {"pos": pos.tolist(), "ori": ori.tolist(),
                                  "moved_m": float(th.linalg.norm(pos - start[name])),
                                  "speed_m_s": float(th.linalg.norm(obj.get_linear_velocity()))}
    settled = {name: (obj.get_position_orientation()[0].clone(), obj.get_position_orientation()[1].clone())
               for name, obj in objs.items()}

    # Slide test on a copy of one object, 1.5 m to the side of the scene.
    slide_name = args.slide_object or next(iter(objs))
    src = next(info for info in scene_objects_info.values() if info["name"] == slide_name)
    probe = DatasetObject(name="slide_probe", dataset_name=args.dataset_name, category=src["category"],
                          model=src["model"])
    with og.sim.stopped():
        scene.add_object(probe)
    start_pos = settled[slide_name][0].clone()
    start_pos[1] -= 1.5
    start_pos[2] += 0.002
    g = 9.81
    report["slide"] = {"object": slide_name, "speed_m_s": args.slide_speed, "runs": []}
    # (tag, static, dynamic): nothing bound; the original material with its bind loop uncommented (static only);
    # what set_obj_materials binds now; and two other coefficients to show the response.
    for tag, static, dynamic in (("unbound", None, None), ("static_only_0.5", 0.5, None),
                                 ("both_0.5", 0.5, 0.5), ("both_0.2", 0.2, 0.2), ("both_1.0", 1.0, 1.0)):
        with og.sim.stopped():
            if static is not None:
                mat = lazy.isaacsim.core.api.materials.PhysicsMaterial(
                    prim_path=f"{probe.prim_path}/Looks/slide_{tag.replace('.', '_')}", name=f"slide_{tag}",
                    static_friction=static, dynamic_friction=dynamic, restitution=None)
                for link in probe.links.values():
                    for msh in link.collision_meshes.values():
                        msh.apply_physics_material(mat)
            probe.set_position_orientation(position=start_pos, orientation=settled[slide_name][1])
        for _ in range(30):
            og.sim.step()
        p0 = probe.get_position_orientation()[0].clone()
        mats = physx_materials(probe)
        probe.set_linear_velocity(th.tensor([args.slide_speed, 0.0, 0.0]))
        for _ in range(90):
            og.sim.step()
        p1 = probe.get_position_orientation()[0].clone()
        ground = (report["ground_plane"] or {}).get("dynamic")
        mu = None if ground is None else (mats[0]["dynamic"] + ground) / 2.0  # PhysX's "average" combine mode
        report["slide"]["runs"].append({
            "material": tag, "physx": mats, "slid_m": float(p1[0] - p0[0]),
            "drift_y_m": float(p1[1] - p0[1]), "combined_dynamic": mu,
            "expected_m": args.slide_speed ** 2 / (2.0 * mu * g) if mu else None,
            "speed_after_m_s": float(th.linalg.norm(probe.get_linear_velocity()))})
    with og.sim.stopped():
        scene.remove_object(probe)

    # The saved-scene key: link_physics_materials as a constructor kwarg, keyed by the link name read out of the
    # loaded asset, as a scene JSON would hand it to OmniGibson.
    contract_src = next(iter(objs))
    src = next(info for info in scene_objects_info.values() if info["name"] == contract_src)
    link_name = next(iter(objs[contract_src].links))
    materials = {link_name: {"static_friction": 0.7, "dynamic_friction": 0.7}}
    with og.sim.stopped():
        contract = DatasetObject(name="contract_probe", dataset_name=args.dataset_name, category=src["category"],
                                 model=src["model"], link_physics_materials=materials)
        scene.add_object(contract)
        pos = settled[contract_src][0].clone()
        pos[1] += 1.5
        contract.set_position_orientation(position=pos, orientation=settled[contract_src][1])
    for _ in range(5):
        og.sim.step()
    entry = contract.get_init_info()["args"]["link_physics_materials"]
    report["contract"] = {
        "link": link_name, "physx": physx_materials(contract), "usd": usd_materials(contract),
        "init_info_is_kwarg_dict": entry is materials,
        "init_info_value_type": type(entry[link_name]).__name__,
    }
    try:
        env.scene.save()  # the JSON string the DataCollectionWrapper stores
        report["contract"]["scene_save"] = "ok"
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        report["contract"]["scene_save"] = f"{type(e).__name__}: {e}"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            og.sim.save(json_paths=[os.path.join(tmp, "scene.json")])
            report["contract"]["og_sim_save"] = "ok"
        except Exception as e:  # noqa: BLE001
            report["contract"]["og_sim_save"] = f"{type(e).__name__}: {e}"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    os._exit(0)  # OmniGibson's atexit teardown is slow and sometimes hangs; the report is written


if __name__ == "__main__":
    main()
