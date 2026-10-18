# SPDX-License-Identifier: Apache-2.0
"""Score stage 11's mass estimates against real masses (repro/README.md, chapter 29).

Stage 11 (11_make_objects_sim_ready.py) asks its VLM for an object's mass and friction from the object's photo,
its oriented bounding box and a volume, and every downstream stage takes the answer as the object's mass. This
asks the same VLM, through the same shim, for objects whose real mass is known:

  ycb     28 YCB objects (render_ycb.py: a studio render of the google_16k scan and its geometry); masses from
          Calli et al., arXiv:1502.03143, Table II. The plastic replica fruit are left out: a realistic render of a
          66 g plastic banana is a real banana to anyone looking at it.
  scenes  our reconstructed objects (stage 6 photo, stage 8 mesh at stage 11's scale), with typical masses of
          such objects or the authors' scene masses as approximate references (REFERENCE).

Methods, each one VLM call per object:
  original  the repository's stage 11 as published (its prompt and volume rule; the paper asks Gemini)
  current   stage 11's prompt as it is now (prompt_object_mass_friction with the volume note, mass_prompt_volumes
            with merged seams)
  typical   the mass a real object of that kind and size weighs, from the photo and the box alone
  material  no mass from the VLM: it names the material, its density, whether the object is solid or a thin
            shell, the wall thickness and any contents, and the mass follows from the mesh
The typical and material questions and the mass computation are simfoundry/utils/mass_estimation.py's, which stage
11 now uses (estimate_rigid_mass: the material mass held within a factor 2 of the typical one).
Offline combinations of those answers are scored too (summary).

    SIMFOUNDRY_LOCAL_BACKEND=1 SIMFOUNDRY_OLLAMA_MODEL=qwen3.5:9b SIMFOUNDRY_OLLAMA_THINK=0 \\
      python repro/mass_friction/bench_mass.py results/mass_friction/bench --methods current,typical,material
    python repro/mass_friction/bench_mass.py results/mass_friction/bench --summary-only
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vlm_compare"))
import bench_common as bc  # noqa: E402  (puts the repository root on sys.path)

from simfoundry.utils import mass_estimation as me  # noqa: E402
from simfoundry.utils.prompt_utils import (  # noqa: E402
    mass_prompt_volumes, parse_json_response, prompt_object_mass_friction)

RENDERS = bc.REPO / "results" / "mass_friction" / "ycb"
VLM_MODEL = __import__("omegaconf").OmegaConf.load(bc.REPO / "scripts/cfg/real2sim_cfg.yaml").s11_sim.vlm_model

# name: (phrase as stage 5 would give it, mass g). Calli et al., arXiv:1502.03143, Table II.
YCB = {
    "003_cracker_box": ("cracker box", 453), "004_sugar_box": ("sugar box", 514),
    "005_tomato_soup_can": ("tomato soup can", 349), "006_mustard_bottle": ("yellow mustard bottle", 431),
    "007_tuna_fish_can": ("tuna fish can", 171), "008_pudding_box": ("pudding box", 187),
    "009_gelatin_box": ("gelatin box", 97), "010_potted_meat_can": ("potted meat can", 370),
    "019_pitcher_base": ("blue pitcher", 178), "024_bowl": ("red bowl", 147), "025_mug": ("red mug", 118),
    "026_sponge": ("sponge", 6.2), "029_plate": ("red plate", 279), "030_fork": ("fork", 34),
    "031_spoon": ("spoon", 30), "032_knife": ("knife", 31), "035_power_drill": ("power drill", 895),
    "037_scissors": ("scissors", 82), "040_large_marker": ("dry-erase marker", 16),
    "042_adjustable_wrench": ("adjustable wrench", 247), "043_phillips_screwdriver": ("screwdriver", 70),
    "048_hammer": ("hammer", 688), "050_medium_clamp": ("black clamp", 59), "051_large_clamp": ("black clamp", 125),
    "054_softball": ("softball", 191), "056_tennis_ball": ("tennis ball", 58), "058_golf_ball": ("golf ball", 46),
    "061_foam_brick": ("red foam brick", 28),
}

# (scene, stage 8 name): (reference mass g or None, where it comes from). None: listed, not scored.
REFERENCE = {
    ("put_marker_in_cup_v2", "iter_0"): (82, "8-inch scissors; YCB's are 82 g"),
    ("put_marker_in_cup_v2", "iter_1"): (10, "12 oz paper cup, about 10 g"),
    ("put_marker_in_cup_v2", "iter_2"): (16, "dry-erase marker; YCB's large one is 16 g"),
    ("put_cup_in_bowl_v2", "iter_0"): (None, "26 cm plastic bowl, a few hundred g"),
    ("put_cup_in_bowl_v2", "iter_1"): (6, "ballpoint pen, about 6 g"),
    ("put_cup_in_bowl_v2", "iter_2"): (10, "paper cup, about 10 g"),
    ("serve_fruits", "iter_0"): (None, "authors' pear 240 g; ours was reconstructed smaller"),
    ("serve_fruits", "iter_1"): (120, "authors' banana 120 g"),
    ("serve_fruits", "iter_2"): (130, "authors' apple 130 g"),
    ("serve_fruits", "iter_4"): (None, "authors' two stacked plates 400 g; ours is one plate"),
    ("serve_fruits", "iter_5"): (140, "authors' orange 140 g"),
    ("serve_fruits", "iter_7"): (None, "authors' tray 500 g"),
}  # serve_fruits iter_3 and iter_6 are the ghost bananas of chapter 24


def ycb_items():
    for name, (phrase, mass_g) in YCB.items():
        obj = Path.home() / ".cache" / "ycb_google_16k" / name / "x" / name / "google_16k" / "textured.obj"
        yield {"id": f"ycb/{name}", "set": "ycb", "phrase": phrase, "ref_g": mass_g, "source": "YCB Table II",
               "image": str(RENDERS / f"{name}.png"), "mesh": str(obj), "scale": 1.0}


def scene_items():
    for (scene, name), (ref, source) in REFERENCE.items():
        d = bc.DATA / scene
        meta = json.load(open(d / "s5_scene" / "obj_cat_list" / f"{name}.json"))
        info = json.load(open(d / "s8_pose" / "info" / f"{name}.json"))
        scale = info["z_up"]["scale"]
        yield {"id": f"{scene}/{name}", "set": "scenes", "phrase": meta["removed_obj_phrase"], "ref_g": ref,
               "source": source, "image": str(d / "s6_upsample" / "upsampled" / f"{name}.png"),
               "mesh": str(d / "s8_pose" / "canonical_mesh" / f"{name}.glb"),
               "scale": float(np.mean(scale)) if isinstance(scale, list) else float(scale)}


def load_mesh(item):
    tm = trimesh.load(item["mesh"], force="mesh", process=True)
    tm.apply_scale(item["scale"])
    return tm


def ask(method, item, tm):
    from simfoundry.models.vlm import Gemini

    _, extent = trimesh.bounds.oriented_bounds(tm)
    geo = me.mesh_geometry(tm, extent)
    if method == "original":  # the repository's stage 11 before chapter 20: its prompt and volume rule, word for word
        volume_m3, _ = mass_prompt_volumes(tm, extent)
        prompt = prompt_object_mass_friction(obj_phrase=item["phrase"], bounding_box_cm=extent * 100,
                                             volume_cm=volume_m3 * 1e6)
    elif method == "current":
        volume_m3, material_m3 = mass_prompt_volumes(tm, extent, weld_seams=True)
        prompt = prompt_object_mass_friction(obj_phrase=item["phrase"], bounding_box_cm=extent * 100,
                                             volume_cm=volume_m3 * 1e6, envelope_note=True,
                                             material_volume_cm=None if material_m3 is None else material_m3 * 1e6)
    elif method == "typical":
        prompt = me.prompt_typical_mass(item["phrase"], extent * 100)
    else:
        prompt = me.prompt_material_mass(item["phrase"], extent * 100)
    vlm = Gemini(project="local", location="global", model=VLM_MODEL)  # the shim answers with SIMFOUNDRY_OLLAMA_MODEL
    res = vlm(prompt=prompt, image_paths=item["image"], temperature=0, top_p=0, seed=0, print_results=False)
    out = {"geometry": geo}
    try:
        # Stage 11 lets this raise: an answer cut off at the token limit (the shim retries once) stops the stage.
        out["text"] = text = vlm.get_result_text(res)
        ans = parse_json_response(text)
        out["answer"] = ans
        out["mass_g"] = me.material_mass_g(ans, geo) if method == "material" else float(ans["mass"]) * 1000
        out["friction"] = float(ans["friction"])
    except Exception as exc:  # scored as a failure, not dropped
        out["error"] = repr(exc)
    return out


def score(pairs):
    """pairs: [(predicted g, reference g)]. Factor = max(p/r, r/p)."""
    f = [max(p / r, r / p) for p, r in pairs if p and p > 0]
    if not f:
        return {}
    lg = [abs(math.log10(x)) for x in f]
    ok = [(p, r) for p, r in pairs if p and p > 0]
    return {"n": len(pairs), "answered": len(f), "median_factor": round(10 ** float(np.median(lg)), 2),
            "alde": round(float(np.mean([abs(math.log(p / r)) for p, r in ok])), 3),
            "mnre": round(float(np.mean([min(p / r, r / p) for p, r in ok])), 3),
            "within_25pct": round(sum(x <= 1.25 for x in f) / len(pairs), 2),
            "within_2x": round(sum(x <= 2 for x in f) / len(pairs), 2), "worst_factor": round(max(f), 1)}


def summarize(out_dir):
    rows = {}
    for f in sorted(out_dir.glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            rows.setdefault(r["id"], {"set": r["set"], "phrase": r["phrase"], "ref_g": r["ref_g"]})[r["method"]] = r
    methods = sorted({m for v in rows.values() for m in v if m not in ("set", "phrase", "ref_g")})

    def pred(v, m):
        if m in methods:
            return v.get(m, {}).get("mass_g")
        t, g = v.get("typical", {}).get("mass_g"), v.get("material", {}).get("mass_g")
        if m == "material_within_2x":  # stage 11's estimate: the computed mass held within a factor 2 of the typical
            return me.combine_masses(t, g)  # one, or whichever of the two answered
        if not (t and g):
            return None
        if m == "geomean":  # geometric mean of the typical and computed masses
            return math.sqrt(t * g)
    combos = ["geomean", "material_within_2x"] if {"typical", "material"} <= set(methods) else []
    summary = {}
    for s in ("ycb", "scenes"):
        sel = {k: v for k, v in rows.items() if v["set"] == s and v["ref_g"]}
        summary[s] = {m: score([(pred(v, m), v["ref_g"]) for v in sel.values()]) for m in methods + combos}
    table = [{"id": k, "ref_g": v["ref_g"], **{m: None if pred(v, m) is None else round(pred(v, m), 1)
                                                 for m in methods + combos}} for k, v in sorted(rows.items())]
    json.dump({"summary": summary, "table": table}, open(out_dir / "summary.json", "w"), indent=1, ensure_ascii=False)
    for s, by in summary.items():
        print(f"== {s}")
        for m, sc in by.items():
            print(f"  {m:20s} {sc}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--methods", default="current,typical,material")
    ap.add_argument("--sets", default="ycb,scenes")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        bc.check_setup()
        print(f"== {bc.lb.OLLAMA_MODEL} think={bc.lb.OLLAMA_THINK!r}", flush=True)
        bc.wait_for_idle_gpu()
        bc.warmup()
        items = [i for s in args.sets.split(",") for i in (ycb_items() if s == "ycb" else scene_items())]
        for method in args.methods.split(","):
            path = out / f"{method}.jsonl"
            done = {json.loads(line)["id"] for line in open(path)} if path.exists() else set()
            for item in items:
                if item["id"] in done:
                    continue
                res = ask(method, item, load_mesh(item))
                bc.append_jsonl(path, {"method": method, **item, **res})
                print(f"{method:9s} {item['id']:40s} ref {item['ref_g']} g -> "
                      f"{res.get('mass_g') and round(res['mass_g'], 1)} g, mu {res.get('friction')}"
                      f"{' ERROR ' + res['error'] if 'error' in res else ''}", flush=True)
    summarize(out)


if __name__ == "__main__":
    main()
