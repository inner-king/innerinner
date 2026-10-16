# SPDX-License-Identifier: Apache-2.0
"""Put objects our pipeline reconstructed onto the authors' DROID desk, for Table G.1's zero-shot rows.

Table G.1 evaluates the public pi0/pi0.5 DROID checkpoints zero-shot on Cup in Bowl and Marker in Cup. The
authors released a DROID scene for Marker in Cup but none for Cup in Bowl, whose task config
(scripts/cfg/task/droid/droid_desk_put_cup_in_bowl.yaml) names the objects of the repository's example video
PutCupInBowl.mp4: a white paper cup, an orange bowl and a black pen. Our pipeline reconstructed that video
(Data/put_cup_in_bowl_v2), so this script builds the missing scene from it, and a Marker in Cup scene from
our PutMarkerInCup reconstruction the same way.

Everything except the task objects comes unchanged from the authors' scene droid_desk_put_marker_in_cup: the
Franka and its controller and sensor config, the droid_v1 mesh background at the pose they tuned, and the
ground plane. Its own task objects are removed and ours are added:

  - each object keeps the orientation it was settled in by our stage 12 (resting on z = 0) and is moved to a
    given xy; z is shifted by the template's ground-plane height, which is where its desk is;
  - `align_to` turns an object about z so that its long axis lies along that of a template object (our
    marker takes the authors' marker's heading, whichever way round is the smaller turn);
  - the object's folder is copied from the real2sim-assets dataset next to the scene JSON and loaded as a
    USDObject, as the authors' scenes carry theirs, so the scene directory is self-contained;
  - names follow the authors' scheme, <category>_<model>_1, and `category` is what the task YAML groups by.

Velocities are zeroed (AGENTS.md rule 3). The output is a new scene; nothing existing is modified.

    python repro/policy_eval_ours/build_droid_scene.py --task cup_in_bowl_ours \\
        --asset-dir <checkout>/assets --data-dir <checkout>/Data \\
        --dataset-dir <BEHAVIOR-1K datasets>/real2sim-assets
"""
import argparse
import hashlib
import json
import os
import shutil

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def long_axis_heading(ori_xyzw, bbox_size):
    """World-frame heading (rad) of an object's longest local axis, projected onto the xy plane."""
    axis = np.zeros(3)
    axis[int(np.argmax(bbox_size))] = 1.0
    v = Rotation.from_quat(ori_xyzw).apply(axis)
    if np.hypot(v[0], v[1]) < 0.5:
        raise ValueError(f"long axis {v} is not horizontal; align_to only makes sense for objects lying down")
    return float(np.arctan2(v[1], v[0]))


def wrap_half_turn(a):
    """Map an angle to (-pi/2, pi/2]: an axis has no direction, so a and a + pi are the same heading."""
    a = (a + np.pi / 2) % np.pi - np.pi / 2
    return a if a > -np.pi / 2 else a + np.pi


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=os.path.join(HERE, "ours.yaml"))
    ap.add_argument("--task", required=True, help="key under tasks: in the spec")
    ap.add_argument("--asset-dir", default=os.path.join(REPO, "assets"))
    ap.add_argument("--data-dir", default=os.path.join(REPO, "Data"))
    ap.add_argument("--dataset-dir", default=os.path.join(REPO, "deps", "BEHAVIOR-1K", "datasets", "real2sim-assets"),
                    help="the real2sim-assets dataset our stage 13 registered the objects in")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.spec))
    task = spec["tasks"][args.task]
    build = task["build"]
    template_path = os.path.join(args.asset_dir, spec["template"])
    template_dir = os.path.dirname(template_path)
    out_path = os.path.join(args.asset_dir, task["scene"])
    out_dir = os.path.dirname(out_path)
    source = json.load(open(os.path.join(args.data_dir, build["source"], "s14_og", "reconstructed_og_scene.json")))
    scene = json.load(open(template_path))

    init = scene["objects_info"]["init_info"]
    reg = scene["state"]["registry"]["object_registry"]
    template_objects = {n: (init[n], reg[n]) for n in list(init) if not n.startswith(("robot", "mesh_background"))}
    for n in template_objects:
        del init[n]
        reg.pop(n, None)
    ground_z = float((scene.get("ground_plane_info") or {}).get("position", [0.0, 0.0, 0.0])[2])

    # Keep the background's relative USD path valid from the new scene directory.
    for n, info in init.items():
        usd = info.get("args", {}).get("usd_path")
        if usd and not os.path.isabs(usd):
            info["args"]["usd_path"] = os.path.relpath(os.path.normpath(os.path.join(template_dir, usd)), out_dir)

    src_init = source["objects_info"]["init_info"]
    src_reg = source["state"]["registry"]["object_registry"]
    os.makedirs(out_dir, exist_ok=True)
    report = []
    for category, obj in build["objects"].items():
        a = src_init[obj["from"]]["args"]
        model = a["model"]
        src_dir = os.path.join(args.dataset_dir, "objects", a["category"], model)
        dst_rel = os.path.join("objects", category, model)
        shutil.copytree(src_dir, os.path.join(out_dir, dst_rel), dirs_exist_ok=True)
        usd_rel = os.path.join(dst_rel, "usd", f"{model}.usd")
        file_hash = md5(os.path.join(out_dir, usd_rel))
        if a.get("expected_file_hash") and a["expected_file_hash"] != file_hash:
            raise SystemExit(f"{obj['from']}: {usd_rel} hashes to {file_hash}, the source scene expects "
                             f"{a['expected_file_hash']}; the dataset copy is not the one stage 13 wrote")

        root = src_reg[obj["from"]]["root_link"]
        ori = np.asarray(root["ori"], dtype=float)
        turn = 0.0
        if obj.get("align_to"):
            ref_info, ref_reg = template_objects[obj["align_to"]]
            ref_meta = json.load(open(os.path.join(
                template_dir, os.path.dirname(os.path.dirname(ref_info["args"]["usd_path"])), "misc", "metadata.json")))
            ours_meta = json.load(open(os.path.join(src_dir, "misc", "metadata.json")))
            turn = wrap_half_turn(long_axis_heading(ref_reg["root_link"]["ori"], ref_meta["bbox_size"])
                                  - long_axis_heading(ori, ours_meta["bbox_size"]))
            ori = (Rotation.from_euler("z", turn) * Rotation.from_quat(ori)).as_quat()
        pos = [float(obj["xy"][0]), float(obj["xy"][1]), float(root["pos"][2]) + ground_z]

        name = f"{category}_{model}_1"
        init[name] = {"class_module": "omnigibson.objects.usd_object", "class_name": "USDObject",
                      "args": {"name": name, "usd_path": usd_rel, "category": category,
                               "scale": a.get("scale", [1.0, 1.0, 1.0]), "expected_file_hash": file_hash}}
        reg[name] = {"is_asleep": False, "non_kin": {},
                     "root_link": {"pos": pos, "ori": [float(x) for x in ori],
                                   "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]}}
        report.append(f"  {name} <- {build['source']}:{obj['from']} ({a['category']}/{model}) "
                      f"pos {np.round(pos, 3).tolist()} turned {np.degrees(turn):.1f} deg")

    for n in reg:  # saved states carry live velocities; nothing should move at t = 0
        for k in ("lin_vel", "ang_vel"):
            if k in reg[n].get("root_link", {}):
                reg[n]["root_link"][k] = [0.0, 0.0, 0.0]
    scene.setdefault("metadata", {})["built_by"] = {
        "script": "repro/policy_eval_ours/build_droid_scene.py", "task": args.task,
        "template": spec["template"], "source": build["source"]}
    with open(out_path, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"[build] {out_path}\n  template {spec['template']} (ground plane z {ground_z}), "
          f"dropped {sorted(template_objects)}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
