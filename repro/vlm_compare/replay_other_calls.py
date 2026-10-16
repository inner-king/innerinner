# SPDX-License-Identifier: Apache-2.0
"""Replay the pipeline's remaining VLM calls on saved scene outputs, through ``Gemini(...)`` and the shim.

  A11  mass and friction per object (11_make_objects_sim_ready.py: same mesh extents, volume rule,
       prompt and image as the stage)
  A8   coarse front pick over the saved yaw renders plus the photo (front_canonicalization.py)
  B1   the text steps of cousin prompting: canonical name, components, topology check and
       variations per component (1_prompt_object_cousins.py), stopping before image generation
  B7   task proposal (7_propose_scene_tasks.py, its own prompt builder and parser), at the seeds
       the shim gives repeats (0, 1000, 2000): the stage samples at temperature 0.7

None of these needs ComfyUI, and nothing is written under Data/: answers, parsed results and the
proposed task YAMLs go to <out_dir>. The Ollama model comes from the environment, as in the
pipeline (SIMFOUNDRY_OLLAMA_MODEL, SIMFOUNDRY_OLLAMA_THINK); run with the response cache off.

Usage: python repro/vlm_compare/replay_other_calls.py <out_dir> [--only A11,A8,B1,B7] [--data-root DIR]
"""
import argparse
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402  (puts the repository root on sys.path)

from omegaconf import OmegaConf  # noqa: E402

from simfoundry.models.vlm import Gemini  # noqa: E402
from simfoundry.pipeline.front_canonicalization import front_pick_prompt, parse_front_choice  # noqa: E402
from simfoundry.utils.prompt_utils import parse_json_response, prompt_object_mass_friction  # noqa: E402

STAGES = bc.REPO / "scripts/pipeline"
SCENES = ("put_marker_in_cup", "put_marker_in_cup_v2", "put_cup_in_bowl_v2")
CFG = OmegaConf.load(bc.REPO / "scripts/cfg/real2sim_cfg.yaml")


def stage_functions(rel_path, names):
    """Named top-level functions of a stage script, without running the script's module code.

    Importing 1_prompt_object_cousins.py would open an rembg session (a model load) at import.
    """
    src = (STAGES / rel_path).read_text()
    tree = ast.parse(src)
    ns = {"re": re, "json": json}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), rel_path, "exec"), ns)
    missing = set(names) - set(ns)
    assert not missing, f"{rel_path} no longer defines {missing}"
    return ns, src


def load_module(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, STAGES / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def objects_of(scene):
    s5 = bc.DATA / scene / "s5_scene"
    out = []
    for f in sorted((bc.DATA / scene / "s8_pose/canonical_mesh").glob("iter_*.glb")):
        idx = int(f.stem.split("iter_")[-1])
        meta = json.load(open(s5 / "obj_cat_list" / f"iter_{idx}.json"))
        out.append((idx, f.stem, meta))
    return out


# ---------------------------------------------------------------------------------------------- A11
def load_mesh_as_trimesh(mesh_path):
    """11_make_objects_sim_ready.py: load_mesh_as_trimesh."""
    tm = trimesh.load(mesh_path, process=True)
    if isinstance(tm, trimesh.Scene):
        tm = tm.to_geometry()
    return tm


def mass_friction_inputs(scene, img_name):
    """Extents (cm) and volume (cm^3) exactly as stage 11 computes them for a rigid object."""
    d = bc.DATA / scene
    info = json.load(open(d / "s8_pose/info" / f"{img_name}.json"))
    tf_scale = info["z_up"]["scale"]
    if isinstance(tf_scale, list):
        tf_scale = np.mean(tf_scale)
    tm = load_mesh_as_trimesh(str(d / "s8_pose/canonical_mesh" / f"{img_name}.glb"))
    tm.apply_scale(tf_scale)
    _, obb_extent = trimesh.bounds.oriented_bounds(tm)
    bbox_volume = np.prod(obb_extent)
    if tm.is_volume and tm.volume > bbox_volume * 0.1:
        volume_m3 = tm.volume
    else:
        volume_m3 = bbox_volume * 0.65
    return obb_extent * 100, volume_m3 * (100 ** 3)


def run_a11(scene, idx, img_name, meta):
    bbox_cm, volume_cm = mass_friction_inputs(scene, img_name)
    vlm = Gemini(project="local", location="global", model=CFG.s11_sim.vlm_model)
    result = vlm(prompt=prompt_object_mass_friction(obj_phrase=meta["removed_obj_phrase"], bounding_box_cm=bbox_cm,
                                                    volume_cm=volume_cm),
                 image_paths=str(bc.DATA / scene / "s6_upsample/upsampled" / f"{img_name}.png"),
                 temperature=0, top_p=0, seed=0, print_results=False)
    out = {"bbox_cm": [round(float(x), 2) for x in bbox_cm], "volume_cm3": round(float(volume_cm), 2)}
    text = vlm.get_result_text(result)  # stage 11 does not catch a refusal here either
    out["text"] = text
    j = parse_json_response(text)
    out.update(mass=j["mass"], friction=j["friction"])
    return out


# ---------------------------------------------------------------------------------------------- A8
def run_a8(scene, idx, img_name, meta):
    views = []
    for p in (bc.DATA / scene / "s8_pose/front_views" / img_name).glob("view_*_az*.png"):
        m = re.match(r"view_([A-Z])_az(\d+)\.png", p.name)
        views.append((m.group(1), int(m.group(2)), str(p)))
    views.sort()
    labels = [v[0] for v in views]
    photo = bc.DATA / scene / "s6_upsample/upsampled" / f"{img_name}.png"
    vlm = Gemini(project="local", location="global", model=CFG.s8_pose.get("front_pick_model", "gemini-2.5-flash"))
    result = vlm(prompt=front_pick_prompt(labels, category=meta["removed_obj_phrase"], has_photo=photo.is_file()),
                 image_paths=[v[2] for v in views] + [str(photo)], temperature=0, top_p=0, seed=0)
    text = vlm.get_result_text(result)
    choice = parse_front_choice(text, labels)
    return {"text": text, "choice": choice, "front_azimuth_deg": dict((v[0], v[1]) for v in views).get(choice)}


# ---------------------------------------------------------------------------------------------- B1
B1_PATH = "B_augmentation/stages/1_prompt_object_cousins.py"
B1, B1_SRC = stage_functions(B1_PATH, ["parse_component_list", "parse_yes_no_response", "parse_variation_options"])


def b1_prompts(obj_phrase, component, include_topology):
    """The stage's prompts, copied from 1_prompt_object_cousins.py main() (checked against it below)."""
    color_removal = f"""
        Given an object description, extract the canonical object name.

        Rules:
        - Remove all color descriptions.
        - Remove material, texture, size, and state descriptors.
        - Keep the full object name if it is a multi-word noun.
        - Do NOT invent or generalize the object category.
        - Return ONLY the object name, with no explanation and no punctuation.

        Object description: "{obj_phrase}"
        """
    components = (
        "With the given object image, decompose the object into functional components based on grasp affordance, list the component name only in number list, "
        "if there is a main body, treat its edge or bottom as a same component inside main body. Make sure to only generate the list of component names, don't generate any other additional text."
    )
    topology = (
        f"For the {component} of the object shown, is it reasonable to make variation on topology "
        f"such that the modified object still looks realistic and is reasonable enough that we would "
        f"see this object in our daily task? Answer with simply yes or no. If it is likely not to be "
        f"normal, make it a no."
    )
    pc = CFG.prompt_cousin_structured
    ng, nt, nv = pc.num_geometry_variation, pc.num_topology_variation, pc.num_visual_variation
    geometry_list = "\n".join([f"{j+1}. [variation description]" for j in range(ng)])
    topology_list = "\n".join([f"{j+1}. [variation description]" for j in range(nt)])
    visual_list = "\n".join([f"{j+1}. [variation description]" for j in range(nv)])
    if include_topology:
        variation = (
            f"For the component '{component}', list {ng} ways to generate variation "
            f"through Geometry dimension, {nt} ways through Topological dimension, "
            f"and {nv} ways through Visual dimension:\n"
            f"1. Geometry dimension\n"
            f"2. Topological dimension\n"
            f"3. Visual dimension\n\n"
            f"Make sure to be determined in the description, prevent using words like or, such as, like, or similar to in the description\n"
            f"Make sure to still maintain the object to have a realistic and reasonable look like something you would see in the daily life\n"
            f"For example, a red banana and a banana with a hole in it is a bad variation, because we rarely see those in daily life.\n"
            f"But a banana with green skin is a good varaition since the unripe banana is usually green.\n"
            f"Format your response as:\n"
            f"Component: {component}\n"
            f"Geometry:\n"
            f"{geometry_list}\n"
            f"Topology:\n"
            f"{topology_list}\n"
            f"Visual:\n"
            f"{visual_list}"
        )
    else:
        variation = (
            f"For the component '{component}', list {ng} ways to generate variation "
            f"through Geometry dimension and {nv} ways through Visual dimension:\n"
            f"1. Geometry dimension\n"
            f"2. Visual dimension\n\n"
            f"Make sure to be determined in the description, prevent using words like or, such as, like, or similar to in the description\n"
            f"Make sure to still maintain the object to have a realistic and reasonable look like something you would see in the daily life\n"
            f"For example, a red banana and a banana with a hole in it is a bad variation, because we rarely see those in daily life.\n"
            f"But a banana with green skin is a good varaition since the unripe banana is usually green, and a banana with dark brown speckles is also a good variation since it looks like a ripe banana\n"
            f"Format your response as:\n"
            f"Component: {component}\n"
            f"Geometry:\n"
            f"{geometry_list}\n"
            f"Visual:\n"
            f"{visual_list}"
        )
    return color_removal, components, topology, variation


# Fragments of each copied prompt that must still be in the stage, so a stage edit is noticed.
for _frag in ("Given an object description, extract the canonical object name.",
              "decompose the object into functional components based on grasp affordance",
              "is it reasonable to make variation on topology",
              "But a banana with green skin is a good varaition since the unripe banana is usually green.",
              "and a banana with dark brown speckles is also a good variation since it looks like a ripe banana"):
    assert _frag in B1_SRC, f"{B1_PATH} changed its prompts; update b1_prompts ({_frag!r})"


def b1_call(prompt, image=None, temperature=0, top_p=0):
    vlm = Gemini(project="local", location="global", model=CFG.prompt_cousin_structured.text_model)
    kwargs = {"image_paths": str(image)} if image else {}
    result = vlm(prompt=prompt, temperature=temperature, top_p=top_p, seed=CFG.seed, **kwargs)
    return vlm.get_result_text(result)


# ---------------------------------------------------------------------------------------------- B7
B7 = load_module("B_augmentation/stages/7_propose_scene_tasks.py", "propose_scene_tasks")


def run_b7(scene, seed=0):
    """7_propose_scene_tasks.py main(), with its inputs resolved the way it falls back to them."""
    pc = CFG.propose_scene_task
    s14 = bc.DATA / scene / "s14_og"
    scene_state_path = s14 / "reconstructed_og_scene.json"
    scene_image_path = next(p for p in (s14 / "auto_generation" / "scene_000.png",
                                        bc.DATA / scene / "s1_video/frames_all/frame_0001.png") if p.exists())
    objects = B7.load_scene_objects(str(scene_state_path))
    robot_info = B7.ROBOT_CONSTRAINTS.get(pc.robot_type.lower(), B7.ROBOT_CONSTRAINTS["franka"])
    prompt = B7.PROMPT_TEMPLATE.format(
        object_list=B7.format_object_list_for_prompt(objects), states=B7.OG_OBJECT_STATES, num_tasks=pc.num_tasks,
        object_constraints=B7.format_object_constraints([], [], []), robot_type=pc.robot_type,
        robot_constraint=pc.robot_constraint or robot_info["constraint"])
    vlm = Gemini(project="local", location="global", model=pc.vlm_model)
    result = vlm(prompt=prompt, image_paths=[str(scene_image_path)], temperature=pc.get("temperature", 0.7),
                 n_retries=int(pc.get("remote_retries", 3)), seed=seed)
    text = vlm.get_result_text(result)
    tasks = B7.parse_vlm_tasks(text)
    names = {info.get("name") for info in objects.values()} | set(objects)
    checks = []
    for t in tasks:
        mentioned = set(re.findall(r"\b(iter_\d+)\b", json.dumps(t)))
        checks.append({"task_name": t.get("task_name"), "objects": sorted(mentioned),
                       "unknown_objects": sorted(m for m in mentioned if m not in names)})
    return {"scene_image": str(scene_image_path), "objects": {k: v.get("category") for k, v in objects.items()},
            "text": text, "n_tasks": len(tasks), "tasks": tasks, "checks": checks}


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--only", default="A11,A8,B1,B7")
    ap.add_argument("--data-root", default=str(bc.DATA), help="scene outputs (default: <repo>/Data)")
    args = ap.parse_args()
    bc.DATA = Path(args.data_root)
    bc.check_setup()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "other_calls.jsonl"
    only = set(args.only.split(","))
    print(f"== {bc.lb.OLLAMA_MODEL} think={bc.lb.OLLAMA_THINK!r} keep_alive={bc.lb.OLLAMA_KEEP_ALIVE}", flush=True)
    bc.warmup()

    def record(case, res, meta, tail):
        bc.append_jsonl(results, {"case": case, **meta, **res})
        print(bc.summary_line(case["id"], meta, res.get("error") or tail), flush=True)

    if "A11" in only:
        for scene in SCENES:
            for idx, img_name, meta in objects_of(scene):
                res, m = bc.measured(lambda: run_a11(scene, idx, img_name, meta))
                record({"id": f"A11_{scene}_{img_name}", "stage": "A11", "scene": scene, "object": meta["removed_obj_phrase"]},
                       res, m, f"mass {res.get('mass')} friction {res.get('friction')} (bbox {res.get('bbox_cm')})")
    if "A8" in only:
        for scene in SCENES:
            for idx, img_name, meta in objects_of(scene):
                res, m = bc.measured(lambda: run_a8(scene, idx, img_name, meta))
                record({"id": f"A8_{scene}_{img_name}", "stage": "A8", "scene": scene, "object": meta["removed_obj_phrase"]},
                       res, m, f"front {res.get('choice')} ({res.get('front_azimuth_deg')} deg)")
    if "B1" in only:
        scene = "put_marker_in_cup_v2"
        upsampled = bc.DATA / scene / "s6_upsample/upsampled"
        n_objects = 0
        for f in sorted(upsampled.glob("*_transparent.png")):
            img_name = f.name.rsplit("_transparent.png", 1)[0]
            meta = json.load(open(bc.DATA / scene / "s5_scene/obj_cat_list" / f"{img_name}.json"))
            if not meta["is_valid_removed_obj"]:
                continue
            if n_objects >= 2:  # the B run in repro/README.md used prompt_cousin_structured.max_objects=2
                break
            n_objects += 1
            phrase = meta["removed_obj_phrase"]
            color_p, comp_p, _, _ = b1_prompts(phrase, "", False)
            base = {"stage": "B1", "scene": scene, "object": phrase}
            res, m = bc.measured(lambda: {"text": b1_call(color_p)})
            record({**base, "id": f"B1_{img_name}_name"}, res, m, repr((res.get("text") or "").strip()[:60]))
            res, m = bc.measured(lambda: {"text": b1_call(comp_p, f)})
            comps = B1["parse_component_list"](res.get("text") or "")
            res["components"] = comps
            record({**base, "id": f"B1_{img_name}_components"}, res, m, f"{comps}")
            for k, comp in enumerate(comps):
                _, _, topo_p, _ = b1_prompts(phrase, comp, False)
                res, m = bc.measured(lambda: {"text": b1_call(topo_p, f)})
                topo = B1["parse_yes_no_response"](res.get("text") or "")
                res["topology"] = topo
                record({**base, "id": f"B1_{img_name}_c{k}_topology", "component": comp}, res, m, f"{comp}: {topo}")
                _, _, _, var_p = b1_prompts(phrase, comp, topo)
                res, m = bc.measured(lambda: {"text": b1_call(var_p, f, temperature=0.7, top_p=0.95)})
                var = B1["parse_variation_options"](res.get("text") or "")
                res["variations"] = var
                counts = {c: {d: len(v) for d, v in dims.items()} for c, dims in var.items()}
                record({**base, "id": f"B1_{img_name}_c{k}_variations", "component": comp}, res, m, f"{counts}")
    if "B7" in only:
        task_dir = out / "proposed_tasks"
        task_dir.mkdir(exist_ok=True)
        for scene in ("put_marker_in_cup_v2", "put_cup_in_bowl_v2"):
            for seed in (0, 1000, 2000):
                res, m = bc.measured(lambda: run_b7(scene, seed))
                record({"id": f"B7_{scene}_seed{seed}", "stage": "B7", "scene": scene, "seed": seed}, res, m,
                       f"{res.get('n_tasks')} tasks {[c['task_name'] for c in res.get('checks', [])]}")
                for t in res.get("tasks", []):
                    name = re.sub(r"[^a-zA-Z0-9_-]", "_", t.get("task_name") or "task")
                    full = B7.build_full_task_yaml(t, B7.get_base_task_template(), scene)
                    OmegaConf.save(OmegaConf.create(full), task_dir / f"{scene}_seed{seed}_{name}.yaml")


if __name__ == "__main__":
    main()
