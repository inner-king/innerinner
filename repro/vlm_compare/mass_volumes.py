# SPDX-License-Identifier: Apache-2.0
"""The volumes stage 11 can state in its mass prompt, per object, and the masses the A11 replays got for them
(repro/README.md, chapter 20.5).

For every stage 8 mesh of the given scenes, writes <out_dir>/volumes.json: the volume and material volume
stage 11 states before (mass_prompt_volumes) and after (weld_seams=True) seam welding, the welded volume itself,
the convex hull, and the two alternatives considered: the summed convex parts stage 11 wrote (CoACD, read from
s11_sim) and a filled voxel grid of the mesh. If <out_dir> holds A11 replays (replay_other_calls.py with
MASS_PROMPT=hollow and welded, in <out_dir>/<variant>/), also writes <out_dir>/summary.json with the volume each
replay sent and the mass it got back, and prints that as a table. Reads only; nothing is written under Data/.

Usage: python repro/vlm_compare/mass_volumes.py <out_dir> [--data-root DIR] [--scenes S1,S2]
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from simfoundry.utils.prompt_utils import mass_prompt_volumes, welded_volume  # noqa: E402

SCENES = ("serve_fruits", "put_cup_in_bowl_v2", "put_marker_in_cup_v2")
VARIANTS = ("hollow", "welded", "welded_shell_material")  # the last: chapter 20.5's variant that was not kept


def cm3(v):
    return None if v is None else round(float(v) * 1e6, 1)


def coacd_sum(scene_dir, idx):
    """Summed volume of the convex collision parts stage 11 wrote for object idx, at their URDF scale."""
    info = json.load(open(scene_dir / "s11_sim/scene_objects_info.json"))[str(idx)]
    udir = scene_dir / "s11_sim/objects" / info["category"] / info["model"] / "urdf"
    total = 0.0
    for col in ET.parse(udir / f"{info['model']}.urdf").getroot().iter("collision"):
        mesh = col.find("geometry/mesh")
        hull = trimesh.load(udir / mesh.attrib["filename"], force="mesh", process=False)
        hull.apply_scale([float(x) for x in mesh.attrib["scale"].split()])
        total += hull.convex_hull.volume
    return total


def object_volumes(scene_dir):
    for f in sorted((scene_dir / "s8_pose/canonical_mesh").glob("iter_*.glb")):
        idx = int(f.stem.split("iter_")[-1])
        info = json.load(open(scene_dir / "s8_pose/info" / f"{f.stem}.json"))
        scale = info["z_up"]["scale"]
        scale = np.mean(scale) if isinstance(scale, list) else scale
        phrase = json.load(open(scene_dir / "s5_scene/obj_cat_list" / f"{f.stem}.json"))["removed_obj_phrase"]
        tm = trimesh.load(str(f), process=True)  # as stage 11 loads it
        tm = tm.to_geometry() if isinstance(tm, trimesh.Scene) else tm
        tm.apply_scale(scale)
        _, ext = trimesh.bounds.oriented_bounds(tm)
        old_v, old_m = mass_prompt_volumes(tm, ext)
        new_v, new_m = mass_prompt_volumes(tm, ext, weld_seams=True)
        welded = welded_volume(tm)
        hull = tm.convex_hull.volume
        yield dict(scene=scene_dir.name, object=f.stem, phrase=phrase, obb_cm=[round(float(x) * 100, 2) for x in ext],
                   obb_cm3=cm3(np.prod(ext)), open_as_loaded=not tm.is_volume, welded_cm3=cm3(welded),
                   convex_hull_cm3=cm3(hull), welded_of_hull=None if welded is None else round(welded / hull, 3),
                   old_volume_cm3=cm3(old_v), old_material_cm3=cm3(old_m),
                   new_volume_cm3=cm3(new_v), new_material_cm3=cm3(new_m),
                   coacd_sum_cm3=cm3(coacd_sum(scene_dir, idx)),
                   voxel_fill_cm3=cm3(tm.voxelized(max(ext) / 128).fill().volume))


def replay_results(out, variant):
    path = out / variant / "other_calls.jsonl"
    results = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            obj = r["case"]["id"].split(f"A11_{r['case']['scene']}_", 1)[1]
            results[(r["case"]["scene"], obj)] = dict(volume_cm3=r.get("volume_cm3"), material_cm3=r.get("material_cm3"),
                                                      mass_g=None if r.get("mass") is None else round(float(r["mass"]) * 1000, 1))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--data-root", default=str(REPO / "Data"))
    ap.add_argument("--scenes", default=",".join(SCENES))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for scene in args.scenes.split(",") for r in object_volumes(Path(args.data_root) / scene)]
    json.dump(rows, open(out / "volumes.json", "w"), indent=1)
    print(f"wrote {out / 'volumes.json'} ({len(rows)} objects)")

    replays = {v: replay_results(out, v) for v in VARIANTS}
    if not any(replays.values()):
        return
    summary = [dict(scene=r["scene"], object=r["object"], phrase=r["phrase"],
                    **{v: replays[v].get((r["scene"], r["object"])) for v in VARIANTS if replays[v]}) for r in rows]
    json.dump(summary, open(out / "summary.json", "w"), indent=1)

    def cell(res):
        if not res:
            return ""
        material = "" if res["material_cm3"] is None else f" (material {res['material_cm3']:,.1f})"
        return f"{res['volume_cm3']:,.1f}{material} -> {res['mass_g']:g} g"
    shown = [v for v in VARIANTS if replays[v]]
    print("| scene / object | " + " | ".join(f"{v}: volume (cm3) -> mass" for v in shown) + " |")
    print("|---|" + "---|" * len(shown))
    for s in summary:
        print(f"| {s['scene']} / {s['phrase']} | " + " | ".join(cell(s[v]) for v in shown) + " |")


if __name__ == "__main__":
    main()
