# SPDX-License-Identifier: Apache-2.0
"""Replay B7 with Pipeline A's two outputs crossed between the PutMarkerInCup scenes.

replay_b7_seeds.py found the marker-into-cup task in 21/30 draws on put_marker_in_cup_v2 and in 6/30 on
put_marker_in_cup_q35 (repro/README.md, chapter 16). The two calls differ in two inputs, both written by
Pipeline A: the object list in the prompt (s14_og/reconstructed_og_scene.json) and the scene image
(s14_og/auto_generation/scene_000.png). This sends the same call at the same seeds with the two inputs
crossed, and with the q35 list changed in one respect at a time:

  v2    v2 image,  v2 list    iter_0 blue_scissors, iter_1 white_cup_with_green_pattern, iter_2 red_marker
  q35   q35 image, q35 list   iter_0 white_cup,     iter_1 blue_scissors,                iter_2 red_marker
  a     q35 image, v2 list
  b     v2 image,  q35 list
  c     q35 image, q35 list with the cup named as in v2 (white_cup_with_green_pattern)
  d     q35 image, q35 list with the cup's and the scissors' ids swapped to v2's (iter_0 scissors, iter_1 cup)

The q35 image puts the scissors rather than the marker in the cup more often, whatever the list. That image
is B6's render of Pipeline A's meshes at B6's settled poses, and the scenes differ in both, so four more
conditions send renders of build_b7_hybrids.py's scene states (render_b7_scenes.py, B6's own camera and
take_scene_picture), all with v2's list:

  e     q35's meshes at v2's poses   (<renders>/q35_meshes_v2_poses.png)
  f     v2's meshes at q35's poses   (<renders>/v2_meshes_q35_poses.png)
  v2r   v2's meshes at v2's poses    (<renders>/v2.png): v2's condition, its image rendered again
  ar    q35's meshes at q35's poses  (<renders>/q35.png): a's condition, its image rendered again

and e2, f2, v2r2, ar2 the same with each state's second render (<renders>/<name>.repeat.png). The renderer
does not repeat itself to the pixel, and B7's answers turn on that noise: the same state rendered twice can
draw the marker-into-cup task 2 and 16 times in 30. So e3..e10, f3..f10, v2r3..v2r10 and ar3..ar10 send
eight more renders of each state (<renders>/<name>.r3.png ... r10.png, render_b7_scenes.py --takes r3,...),
meant for fewer seeds each, and the summary compares the states image by image: each image's rate is one
observation, and the poses and the meshes are tested by permuting images within the other factor's levels.

Each condition is a scene directory under <out_dir>/inputs/ holding copies of the two files, the list
edited in the copy, and run_b7 reads it in place of Data/<scene>; nothing under Data/ is written. v2 and
q35 go through copies too, so their answers check the copies against replay_b7_seeds.py's calls. Every
call records hashes of the prompt and image the shim actually sent, so what a condition sent is checked
rather than assumed. The calls, idle-GPU waits, discards and grading are replay_b7_seeds.py's.

Run with the response cache off and Pipeline B's shim settings:
  SIMFOUNDRY_LOCAL_BACKEND=1 SIMFOUNDRY_OLLAMA_MODEL=qwen3-vl:8b-instruct SIMFOUNDRY_OLLAMA_THINK=0
  SIMFOUNDRY_OLLAMA_KEEP_ALIVE=0 GCLOUD_PROJECT=local

Usage: python repro/vlm_compare/replay_b7_cross.py <out_dir> [--seeds N] [--conditions v2,q35,a,b,c,d]
           [--reference <replay_b7_seeds.py out_dir>/calls.jsonl] [--previous <an earlier out_dir>/calls.jsonl]
           [--renders DIR] [--data-root DIR] [--summarize-only]
A rerun with the same <out_dir> skips the (condition, seed) pairs already answered. A condition that an
earlier run (--previous) also made is checked to answer it word for word alike, seed by seed.
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_b7_seeds as rbs  # noqa: E402  (importing it installs its idle-GPU check into bench_common)

bc, roc = rbs.bc, rbs.roc
V2, Q35 = rbs.SCENES
SCENE_JSON = Path("s14_og/reconstructed_og_scene.json")
SCENE_IMAGE = Path("s14_og/auto_generation/scene_000.png")
KINDS = ("marker", "cup", "scissors")


def _sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def _id_of(init_info, kind):
    (key,) = [k for k, v in init_info.items() if kind in (v.get("args", {}).get("category") or "")]
    return key


def rename_cup(init_info, v2_info):
    """The cup is named as v2 names it; the ids stay q35's."""
    init_info[_id_of(init_info, "cup")]["args"]["category"] = v2_info[_id_of(v2_info, "cup")]["args"]["category"]


def v2_ids(init_info, v2_info):
    """The cup and the scissors trade ids, so each has the one it has in v2; the names stay q35's."""
    cup, scissors = (init_info[_id_of(init_info, k)]["args"] for k in ("cup", "scissors"))
    cup["category"], scissors["category"] = scissors["category"], cup["category"]
    assert all(_id_of(init_info, k) == _id_of(v2_info, k) for k in KINDS)


# name: (scene the image is from, or a render under --renders; scene the object list is from; edit made
# to the list's copy)
CONDITIONS = {
    "v2": (V2, V2, None),
    "q35": (Q35, Q35, None),
    "a": (Q35, V2, None),
    "b": (V2, Q35, None),
    "c": (Q35, Q35, rename_cup),
    "d": (Q35, Q35, v2_ids),
    "e": ("q35_meshes_v2_poses.png", V2, None),
    "f": ("v2_meshes_q35_poses.png", V2, None),
    "v2r": ("v2.png", V2, None),
    "ar": ("q35.png", V2, None),
    "e2": ("q35_meshes_v2_poses.repeat.png", V2, None),
    "f2": ("v2_meshes_q35_poses.repeat.png", V2, None),
    "v2r2": ("v2.repeat.png", V2, None),
    "ar2": ("q35.repeat.png", V2, None),
}
EXTRA_TAKES = range(3, 11)
for _base, _state in (("e", "q35_meshes_v2_poses"), ("f", "v2_meshes_q35_poses"), ("v2r", "v2"), ("ar", "q35")):
    for _k in EXTRA_TAKES:
        CONDITIONS[f"{_base}{_k}"] = (f"{_state}.r{_k}.png", V2, None)
CONTROL = {V2: "v2", Q35: "q35"}  # the condition that sends a scene as it is
# Every image of each scene state that is sent with v2's list, keyed (meshes from, poses from): B6's
# photograph where the state is a scene's own, then the renders.
STATE_IMAGES = {
    ("v2", "v2"): ["v2", "v2r", "v2r2", *(f"v2r{k}" for k in EXTRA_TAKES)],
    ("q35", "q35"): ["a", "ar", "ar2", *(f"ar{k}" for k in EXTRA_TAKES)],
    ("q35", "v2"): ["e", "e2", *(f"e{k}" for k in EXTRA_TAKES)],
    ("v2", "q35"): ["f", "f2", *(f"f{k}" for k in EXTRA_TAKES)],
}
# What the summary tests (Fisher): (conditions, conditions, count, what changes between them). The
# single pairs differ in one thing; on the q35 image, q35 -> c -> a and q35 -> d -> a change the list's
# two differences in either order. The pooled ones sum over the input that is not compared.
MIC = "marker_in_cup_runs"
COMPARISONS = [
    (("q35",), ("a",), MIC, "q35 image: q35 list -> v2 list"),
    (("b",), ("v2",), MIC, "v2 image: q35 list -> v2 list"),
    (("a",), ("v2",), MIC, "v2 list: q35 image -> v2 image"),
    (("q35",), ("b",), MIC, "q35 list: q35 image -> v2 image"),
    (("q35",), ("c",), MIC, "q35 image, q35 ids: white_cup -> white_cup_with_green_pattern"),
    (("d",), ("a",), MIC, "q35 image, v2 ids: white_cup -> white_cup_with_green_pattern"),
    (("q35",), ("d",), MIC, "q35 image, white_cup: q35 ids -> v2 ids"),
    (("c",), ("a",), MIC, "q35 image, white_cup_with_green_pattern: q35 ids -> v2 ids"),
    (("q35", "b"), ("v2", "a"), MIC, "q35 list -> v2 list, over both images"),
    (("q35", "a"), ("v2", "b"), MIC, "q35 image -> v2 image, over both lists"),
    (("q35",), ("c", "d", "a"), MIC, "q35 image: q35 list -> any of the changed lists"),
    (("q35", "b"), ("v2", "a", "c", "d"), "on_cup_runs", "q35 list -> the other lists"),
    (("q35", "a", "c", "d"), ("v2", "b"), "scissors_in_cup_runs", "q35 image -> v2 image"),
    # The q35 image's poses and meshes, one at a time (v2's list throughout): against B6's two images,
    (("a",), ("e",), MIC, "q35 meshes: q35 poses -> v2 poses"),
    (("f",), ("v2",), MIC, "v2 meshes: q35 poses -> v2 poses"),
    (("a",), ("f",), MIC, "q35 poses: q35 meshes -> v2 meshes"),
    (("e",), ("v2",), MIC, "v2 poses: q35 meshes -> v2 meshes"),
    (("a", "f"), ("e", "v2"), MIC, "q35 poses -> v2 poses, over both meshes"),
    (("a", "e"), ("f", "v2"), MIC, "q35 meshes -> v2 meshes, over both poses"),
    (("a", "f"), ("e", "v2"), "scissors_in_cup_runs", "q35 poses -> v2 poses, over both meshes"),
    (("a", "e"), ("f", "v2"), "scissors_in_cup_runs", "q35 meshes -> v2 meshes, over both poses"),
    # then within the renders alone, and each render against the image it renders again.
    (("ar",), ("e",), MIC, "renders, q35 meshes: q35 poses -> v2 poses"),
    (("f",), ("v2r",), MIC, "renders, v2 meshes: q35 poses -> v2 poses"),
    (("ar",), ("f",), MIC, "renders, q35 poses: q35 meshes -> v2 meshes"),
    (("e",), ("v2r",), MIC, "renders, v2 poses: q35 meshes -> v2 meshes"),
    (("ar", "f"), ("e", "v2r"), MIC, "renders, q35 poses -> v2 poses, over both meshes"),
    (("ar", "e"), ("f", "v2r"), MIC, "renders, q35 meshes -> v2 meshes, over both poses"),
    (("ar", "f"), ("e", "v2r"), "scissors_in_cup_runs", "renders, q35 poses -> v2 poses, over both meshes"),
    (("ar", "e"), ("f", "v2r"), "scissors_in_cup_runs", "renders, q35 meshes -> v2 meshes, over both poses"),
    (("a",), ("ar",), MIC, "q35 image: B6's -> rendered again"),
    (("v2",), ("v2r",), MIC, "v2 image: B6's -> rendered again"),
    # A state's two renders, and the renders' poses and meshes over both renders of each state.
    (("v2r",), ("v2r2",), MIC, "v2 state: first render -> second"),
    (("ar",), ("ar2",), MIC, "q35 state: first render -> second"),
    (("e",), ("e2",), MIC, "q35 meshes at v2 poses: first render -> second"),
    (("f",), ("f2",), MIC, "v2 meshes at q35 poses: first render -> second"),
    (("ar", "f", "ar2", "f2"), ("e", "v2r", "e2", "v2r2"), MIC, "both renders, q35 poses -> v2 poses, over both meshes"),
    (("ar", "e", "ar2", "e2"), ("f", "v2r", "f2", "v2r2"), MIC, "both renders, q35 meshes -> v2 meshes, over both poses"),
    (("ar", "f", "ar2", "f2"), ("e", "v2r", "e2", "v2r2"), "scissors_in_cup_runs",
     "both renders, q35 poses -> v2 poses, over both meshes"),
    (("ar", "e", "ar2", "e2"), ("f", "v2r", "f2", "v2r2"), "scissors_in_cup_runs",
     "both renders, q35 meshes -> v2 meshes, over both poses"),
]


def control_scene(cond):
    """The scene a condition sends unchanged, or None for a crossed or edited one."""
    image_from, list_from, edit = CONDITIONS[cond]
    return image_from if image_from == list_from and edit is None else None


def build_inputs(root, data, renders, conds):
    """A scene directory per condition with copies of its image and scene JSON; returns what each holds."""
    v2_info = json.loads((data / V2 / SCENE_JSON).read_text())["objects_info"]["init_info"]
    manifest = {}
    for name, (image_from, list_from, edit) in CONDITIONS.items():
        if name not in conds and not (root / name).exists():
            continue  # a render that this run neither sends nor sent before need not exist
        scene_dir = root / name
        (scene_dir / SCENE_IMAGE).parent.mkdir(parents=True, exist_ok=True)
        image_src = data / image_from / SCENE_IMAGE if image_from in CONTROL else renders / image_from
        shutil.copyfile(image_src, scene_dir / SCENE_IMAGE)
        if edit is None:
            shutil.copyfile(data / list_from / SCENE_JSON, scene_dir / SCENE_JSON)
        else:
            scene = json.loads((data / list_from / SCENE_JSON).read_text())
            edit(scene["objects_info"]["init_info"], v2_info)
            (scene_dir / SCENE_JSON).write_text(json.dumps(scene, indent=4) + "\n")
        objects = roc.B7.load_scene_objects(str(scene_dir / SCENE_JSON))
        image = (scene_dir / SCENE_IMAGE).read_bytes()
        manifest[name] = {
            "image_from": image_from, "list_from": list_from, "edit": edit.__doc__ if edit else None,
            "objects": {k: v["category"] for k, v in objects.items()},
            "object_list": roc.B7.format_object_list_for_prompt(objects),
            "image_sha256": _sha(image), "scene_json_sha256": _sha((scene_dir / SCENE_JSON).read_bytes()),
            # The shim sends the image downscaled and re-encoded, so this is what a call's hash should be.
            "sent_image_sha256": _sha(bc.lb._resize_for_vlm(image)),
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


# What the shim posts to Ollama, captured around bench_common's recorder (itself a wrap of requests.post).
SENT = []
_post = bc.requests.post


def _capturing_post(url, *args, **kwargs):
    if url.endswith("/api/chat"):
        user = [m for m in (kwargs.get("json") or {}).get("messages", []) if m.get("role") == "user"][-1]
        SENT.append({"prompt": user.get("content") or "", "images": [_sha(i) for i in user.get("images") or []]})
    return _post(url, *args, **kwargs)


bc.requests.post = _capturing_post


def call(cond, seed, manifest):
    SENT.clear()
    res, meta, contended = rbs.call(cond, seed)
    meta["sent"] = [{"prompt_sha256": _sha(s["prompt"]), "image_sha256": s["images"],
                     "object_list_in_prompt": manifest[cond]["object_list"] in s["prompt"]} for s in SENT]
    return res, meta, contended, [s["prompt"] for s in SENT]


# ---------------------------------------------------------------------------------------------- summary
def wilson(k, n, z=1.96):
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def puts(r, state, kinds):
    """Whether a written task holds ``state`` true for one of these kinds of object and the cup.

    The two other ways a run misses the marker-into-cup task: the scissors goes inside the cup in its
    place (InsideAABB), or the marker or scissors is put on the cup rather than in it (OnTopAABB).
    """
    cat = {kind: next(c for c in r["objects"].values() if kind in c) for kind in (*kinds, "cup")}
    return any(f"{state}({cat[k]}, {cat['cup']})" in g["predicates"] for k in kinds for g in r["graded"])


def summarize(out, records, manifest, reference, previous):
    for r in records:  # graded again, so a change to grade() applies to calls already made
        r["graded"] = rbs.grade(r.get("tasks", []), r["objects"]) if r.get("objects") else []
    ref_text = {(r["case"]["scene"], r["case"]["seed"]): r.get("text") for r in reference
                if r["case"]["kind"] == "seed" and not r["discarded"]}
    prev_text = {(r["case"]["condition"], r["case"]["seed"]): r.get("text") for r in previous if not r["discarded"]}
    prompts = {c: (out / "inputs" / c / "prompt.txt").read_text() for c in CONDITIONS
               if (out / "inputs" / c / "prompt.txt").exists()}
    conds = [c for c in CONDITIONS if any(r["case"]["condition"] == c for r in records)]
    summary = {}
    for c in conds:
        runs = sorted((r for r in records if r["case"]["condition"] == c and not r["discarded"]),
                      key=lambda r: r["case"]["seed"])
        attempts = [a for r in runs for a in r["attempts"]]
        written = [g for r in runs for g in r["graded"]]
        sent = [s for r in runs for s in r.get("sent", [])]
        hits = [r["case"]["seed"] for r in runs if any(g["marker_in_cup"] for g in r["graded"])]
        # v2 and q35 send what replay_b7_seeds.py sent from Data/, so they should answer word for word alike.
        scene = control_scene(c)
        same = [r.get("text") == ref_text[(scene, r["case"]["seed"])] for r in runs if (scene, r["case"]["seed"]) in ref_text]
        same_prev = [r.get("text") == prev_text[(c, r["case"]["seed"])] for r in runs if (c, r["case"]["seed"]) in prev_text]
        summary[c] = {
            "image_from": manifest[c]["image_from"], "list_from": manifest[c]["list_from"], "edit": manifest[c]["edit"],
            "objects": manifest[c]["objects"],
            "runs": len(runs),
            "discarded": sum(r["case"]["condition"] == c and r["discarded"] for r in records),
            "errors": sum(bool(r.get("error")) for r in runs),
            "marker_in_cup_runs": len(hits),
            "marker_in_cup_wilson95": wilson(len(hits), len(runs)),
            "marker_in_cup_seeds": hits,
            "marker_in_cup_runs_loadable": sum(any(g["marker_in_cup"] and not g["invalid"] for g in r["graded"])
                                               for r in runs),
            "scissors_in_cup_runs": sum(puts(r, "InsideAABB", ["scissors"]) for r in runs if r.get("objects")),
            "on_cup_runs": sum(puts(r, "OnTopAABB", ["marker", "scissors"]) for r in runs if r.get("objects")),
            "n_tasks_parsed": dict(sorted(Counter(r.get("n_tasks", 0) for r in runs).items())),
            "tasks_written": len(written),
            "invalid_tasks": sum(bool(g["invalid"]) for g in written),
            "mislabeled_tasks": sum(bool(g["mislabeled"]) for g in written),
            "task_kinds": Counter(" & ".join(g["predicates"]) for g in written).most_common(),
            "same_text_as_replay_b7_seeds": f"{sum(same)}/{len(same)}" if same else None,
            "same_text_as_previous_run": f"{sum(same_prev)}/{len(same_prev)}" if same_prev else None,
            "max_tokens_hits": sum(a.get("done_reason") == "length" for a in attempts),
            "prompt_tokens": sorted({a.get("prompt_tokens") for a in attempts}),
            "gen_tokens_mean": round(sum(a.get("gen_tokens") or 0 for a in attempts) / max(1, len(attempts)), 1),
            "loaded_mib_vram_ctx": sorted({(m[1] // 2**20, m[2] // 2**20, m[3]) for r in runs for m in r["ollama_ps"]}),
            "sent_prompts": sorted({s["prompt_sha256"] for s in sent}),
            "sent_images": sorted({i for s in sent for i in s["image_sha256"]}),
            "sent_list_in_prompt": bool(sent) and all(s["object_list_in_prompt"] for s in sent),
        }

    checks = []
    for c in conds:
        s = summary[c]
        checks.append((f"{c}: every call sent one prompt (the saved prompt.txt), with this condition's object list, "
                       "and the copied image", c in prompts and s["sent_prompts"] == [_sha(prompts[c])]
                       and s["sent_list_in_prompt"] and s["sent_images"] == [manifest[c]["sent_image_sha256"]]))
        if "q35" in prompts and c in prompts and c != "q35":
            checks.append((f"{c}: its prompt is q35's but for the object list",
                           prompts[c].replace(manifest[c]["object_list"], manifest["q35"]["object_list"]) == prompts["q35"]))
    for c in conds:
        twin = CONTROL[manifest[c]["list_from"]]
        if c != twin and manifest[c]["edit"] is None and twin in summary:
            checks.append((f"{c}: sent {twin}'s prompt word for word", summary[c]["sent_prompts"] == summary[twin]["sent_prompts"]))
    for c in conds:
        twin = CONTROL.get(manifest[c]["image_from"])
        if twin and c != twin and twin in summary:
            checks.append((f"{c}: sent {twin}'s image", summary[c]["sent_images"] == summary[twin]["sent_images"]))
    for c in conds:
        if summary[c]["same_text_as_previous_run"]:
            k, n = map(int, summary[c]["same_text_as_previous_run"].split("/"))
            checks.append((f"{c}: answered as the previous run did, seed for seed ({k}/{n})", k == n))
    summary["checks"] = [{"check": name, "ok": bool(ok)} for name, ok in checks]

    tests = []
    for xs, ys, count, change in COMPARISONS:
        if all(c in summary for c in (*xs, *ys)):
            (kx, nx), (ky, ny) = ((sum(summary[c][count] for c in cs), sum(summary[c]["runs"] for c in cs))
                                  for cs in (xs, ys))
            tests.append({"from": "+".join(xs), "to": "+".join(ys), "count": count, "change": change,
                          "k": [kx, ky], "n": [nx, ny], "fisher_p": rbs.fisher_two_sided(kx, nx - kx, ky, ny - ky)})
    summary["tests"] = tests
    rendered = any(summary[c]["image_from"] not in CONTROL for c in conds)
    summary["image_level"] = {f"{count}, B6's photographs {'in' if b6 else 'out'}": image_level(summary, count, b6)
                              for count in (MIC, "scissors_in_cup_runs") for b6 in (True, False)} if rendered else {}

    # Two images of one state (B6's and a render of it, or a state's two renders) at the same seed: how often
    # the very same answer comes back says how far render noise alone moves B7.
    text = {(r["case"]["condition"], r["case"]["seed"]): r.get("text") for r in records if not r["discarded"]}
    agree = []
    for x, y in (("v2", "v2r"), ("a", "ar"), ("v2r", "v2r2"), ("ar", "ar2"), ("e", "e2"), ("f", "f2")):
        seeds = sorted({sd for c, sd in text if c == x} & {sd for c, sd in text if c == y})
        if seeds:
            agree.append({"pair": f"{x} / {y}", "same_text": sum(text[(x, sd)] == text[(y, sd)] for sd in seeds),
                          "seeds": len(seeds)})
    summary["same_answer_across_images_of_a_state"] = agree
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")

    lines = ["| | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    row = lambda name, f: lines.append(f"| {name} | " + " | ".join(str(f(summary[c])) for c in conds) + " |")  # noqa: E731
    row("image from", lambda x: x["image_from"])
    row("object list", lambda x: ", ".join(f"{k} {v}" for k, v in sorted(x["objects"].items())))
    row("runs kept (discarded as contended)", lambda x: f"{x['runs']} ({x['discarded']})")
    row("marker into cup (InsideAABB)", lambda x: f"{x['marker_in_cup_runs']}/{x['runs']}")
    row("... Wilson 95%", lambda x: x["marker_in_cup_wilson95"])
    row("... of which the YAML would load", lambda x: f"{x['marker_in_cup_runs_loadable']}/{x['runs']}")
    row("seeds with it", lambda x: x["marker_in_cup_seeds"])
    row("scissors into cup (InsideAABB)", lambda x: f"{x['scissors_in_cup_runs']}/{x['runs']}")
    row("marker or scissors on the cup (OnTopAABB)", lambda x: f"{x['on_cup_runs']}/{x['runs']}")
    row("tasks parsed per run", lambda x: x["n_tasks_parsed"])
    row("written tasks: would not load / mislabeled", lambda x: f"{x['invalid_tasks']} / {x['mislabeled_tasks']} of {x['tasks_written']}")
    row("same text as replay_b7_seeds.py", lambda x: x["same_text_as_replay_b7_seeds"])
    row("same text as the previous run", lambda x: x["same_text_as_previous_run"])
    row("MAX_TOKENS, prompt tokens, mean gen tokens",
        lambda x: f"{x['max_tokens_hits']}, {x['prompt_tokens']}, {x['gen_tokens_mean']}")
    row("model size / in VRAM (MiB) / context", lambda x: x["loaded_mib_vram_ctx"])
    if summary["image_level"]:
        lines.append("\nImage level (one observation per image: its rate over the seeds sent with it):\n")
    for key, il in summary["image_level"].items():
        lines.append(f"- {key}:")
        lines += [f"  - {st}: {x['images']} images, mean {x['mean']}, sd {x['sd']}, rates {x['rates']}"
                  for st, x in il["states"].items()]
        lines += [f"  - {t['factor']} (q35 -> v2), within each level of the other: mean difference {t['difference']}, "
                  f"permutation p = {t['p']}" for t in il["tests"]]
    lines.append("\nFisher exact (two-sided), each draw as one observation:\n")
    lines += [f"- {t['count']}, {t['from']} -> {t['to']} ({t['change']}): {t['k'][0]}/{t['n'][0]} -> "
              f"{t['k'][1]}/{t['n'][1]}, p = {t['fisher_p']:.3g}" for t in tests]
    if agree:
        lines.append("\nSame answer at the same seed, two images of one state:\n")
        lines += [f"- {a['pair']}: {a['same_text']}/{a['seeds']}" for a in agree]
    lines.append("\nChecks:\n")
    lines += [f"- [{'x' if c['ok'] else ' '}] {c['check']}" for c in summary["checks"]]
    for c in conds:
        lines.append(f"\n{c} written tasks:\n" + "\n".join(f"- {n} x {k}" for k, n in summary[c]["task_kinds"]))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def image_level(summary, count, include_b6, n_perm=100000, seed=0):
    """Each image's rate per state, and the poses' and the meshes' effects on it, image by image.

    A draw is not an independent look at a state: the draws on one image share its render noise, which moves
    the rate as much as anything else here. So each image counts once, by its rate, and an effect is tested
    by permuting images between the two levels of the factor within each level of the other (two-sided).
    """
    rates = {st: [summary[c][count] / summary[c]["runs"] for c in cs if c in summary and summary[c]["runs"]
                  and (include_b6 or summary[c]["image_from"] not in CONTROL)] for st, cs in STATE_IMAGES.items()}
    out = {"states": {f"{m} meshes, {p} poses": {
        "images": len(r), "mean": round(float(np.mean(r)), 3) if r else None,
        "sd": round(float(np.std(r, ddof=1)), 3) if len(r) > 1 else None, "rates": [round(x, 2) for x in r]}
        for (m, p), r in rates.items()}, "tests": []}
    rng = np.random.default_rng(seed)
    for factor, i in (("poses", 1), ("meshes", 0)):
        strata = [(np.array(next(r for st, r in rates.items() if st[1 - i] == lvl and st[i] == "q35")),
                   np.array(next(r for st, r in rates.items() if st[1 - i] == lvl and st[i] == "v2"))) for lvl in ("v2", "q35")]
        if any(len(a) < 2 or len(b) < 2 for a, b in strata):
            continue
        observed = np.mean([b.mean() - a.mean() for a, b in strata])
        permuted = np.zeros(n_perm)
        for a, b in strata:
            pool = np.concatenate([a, b])
            order = np.argsort(rng.random((n_perm, len(pool))), axis=1)
            permuted += (pool[order[:, len(a):]].mean(1) - pool[order[:, :len(a)]].mean(1)) / len(strata)
        p = (np.sum(np.abs(permuted) >= abs(observed) - 1e-12) + 1) / (n_perm + 1)
        out["tests"].append({"factor": factor, "difference": round(float(observed), 3), "p": round(float(p), 4)})
    return out


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--seeds", type=int, default=30, help="seeds 0..N-1 per condition")
    ap.add_argument("--conditions", default=",".join(CONDITIONS), help="which of them to call, in this order")
    ap.add_argument("--reference", default=str(bc.REPO / "results/diagnostics/vlm_compare/b7_q35/calls.jsonl"),
                    help="replay_b7_seeds.py's calls, which v2 and q35 should repeat word for word")
    ap.add_argument("--previous", default=str(bc.REPO / "results/diagnostics/vlm_compare/b7_q35_cross/calls.jsonl"),
                    help="an earlier run's calls, whose conditions this run's should repeat word for word")
    ap.add_argument("--renders", default=None, help="render_b7_scenes.py's images (default: <out_dir>/renders)")
    ap.add_argument("--data-root", default=str(bc.DATA), help="scene outputs (default: <repo>/Data)")
    ap.add_argument("--summarize-only", action="store_true", help="grade the calls already made; no VLM calls")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    conds = [c for c in args.conditions.split(",") if c]
    assert set(conds) <= set(CONDITIONS), f"unknown conditions {set(conds) - set(CONDITIONS)}"
    calls = out / "calls.jsonl"
    records = [json.loads(line) for line in open(calls)] if calls.exists() else []
    reference = Path(args.reference)
    ref = [json.loads(line) for line in open(reference)] if reference.exists() else []
    previous = Path(args.previous)
    prev = [json.loads(line) for line in open(previous)] if previous.exists() and previous.resolve() != calls.resolve() else []
    if args.summarize_only:
        summarize(out, records, json.loads((out / "inputs/manifest.json").read_text()), ref, prev)
        return
    bc.check_setup()
    assert bc.lb.OLLAMA_KEEP_ALIVE in ("0", "0s"), "set SIMFOUNDRY_OLLAMA_KEEP_ALIVE=0, as the pipeline does"
    for scene in (V2, Q35):
        # The stage prefers assets/scenes/<scene>/; without it, it reads the two files copied here.
        assert not (bc.REPO / "assets/scenes" / scene).exists(), f"assets/scenes/{scene} would change B7's inputs"
    manifest = build_inputs(out / "inputs", Path(args.data_root), Path(args.renders or out / "renders").resolve(), conds)
    bc.DATA = out / "inputs"  # run_b7 reads bc.DATA / <scene>, here a condition
    print(f"== {bc.lb.OLLAMA_MODEL} think={bc.lb.OLLAMA_THINK!r} keep_alive={bc.lb.OLLAMA_KEEP_ALIVE} "
          f"num_tasks={rbs.NUM_TASKS} temperature={roc.CFG.propose_scene_task.get('temperature', 0.7)} "
          f"conditions={conds} reference={reference if ref else None} previous={previous if prev else None}", flush=True)
    for c in conds:
        print(f"   {c}: image {manifest[c]['image_from']}, list {manifest[c]['objects']}", flush=True)
    bc.warmup()

    done = {(r["case"]["condition"], r["case"]["seed"]) for r in records if not r["discarded"]}
    # Interleaved, so anything that drifts over the session falls on every condition alike.
    for seed in range(args.seeds):
        for cond in conds:
            if (cond, seed) in done:
                continue
            for attempt in range(rbs.MAX_TRIES):
                res, meta, contended, prompts = call(cond, seed, manifest)
                case = {"id": f"B7x_{cond}_seed{seed}", "condition": cond, "seed": seed, "kind": "seed", "try": attempt}
                graded = rbs.grade(res.get("tasks", []), res["objects"]) if "objects" in res else []
                rec = {"case": case, "discarded": contended, **meta, **res, "graded": graded}
                bc.append_jsonl(calls, rec)
                records.append(json.loads(json.dumps(rec, default=str)))
                if prompts and not (out / "inputs" / cond / "prompt.txt").exists():
                    (out / "inputs" / cond / "prompt.txt").write_text(prompts[0])
                hit = any(g["marker_in_cup"] for g in graded)
                print(bc.summary_line(case["id"], meta, res.get("error") or
                                      f"{res.get('n_tasks')} parsed {[g['task_name'] for g in graded]}"
                                      f"{' MARKER->CUP' if hit else ''}{' (discarded)' if contended else ''}"), flush=True)
                if not contended:
                    break
            else:
                print(f"   gave up on {cond} seed {seed}: every try overlapped another job", flush=True)
    summarize(out, records, manifest, ref, prev)


if __name__ == "__main__":
    main()
