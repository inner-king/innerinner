# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU replay of the stage-5 crop_source rules on the stored YouTube scenes (spec SPEC_crop_source_ko.v2, section 8-2).

Inputs (read only):
- the 57 objects of research 1 (objects_cpu.json);
- the iteration masks: stage 8's object points projected onto the depth grid (the same approximation research 1 used,
  score_objects.observed_mask), because the stored scenes have no removal masks;
- the original-frame SAM3 instances on the same depth grid (sam3_out/masks_<scene>.npz, rows of sam3_summary.json);
- the categories each iteration received (s5_scene/obj_cat_list/iter_k.json, vlm_categories, in order).

The pipeline order asks SAM3 for a category when its iteration first names it, and matches each object in iteration
order. The all-at-once order gives every category to the pool before the first match (research 1's order).

Expected numbers (spec 8-2 and the v3 notes): pipeline order 46 right, 6 traces (5 rescued, 1 lost), 1 revealed later,
4 no_original; all-at-once 47 / 6 / 1 / 3. Fake (shifted) masks accepted: 0.
"""

import ast
import importlib.util
import json
import os
import pathlib

import numpy as np
import pytest

from simfoundry.pipeline import crop_source as cs

# The research 1 folder that holds yt_fix/scorecard and yt/runs/Data. Unset: the replay is skipped.
FINAL_VEG_ENV = "SIMFOUNDRY_TEST_FINAL_VEG"
FINAL_VEG = pathlib.Path(os.environ.get(FINAL_VEG_ENV) or "final_veg_unset")
SCORECARD = FINAL_VEG / "yt_fix/scorecard"
SAM3_DIR = SCORECARD / "sam3_out"
OBJECTS = SCORECARD / "objects_cpu.json"
STUDY = SCORECARD / "match_rule_study_all.json"
DATA = FINAL_VEG / "yt/runs/Data"
SCORE_OBJECTS = SCORECARD / "score_objects.py"
STUDY_RULE = "IoU>=0.5 이름 무관, 원본 없는 범주는 유지"
PARAMS = cs.CropParams()
STAGE = pathlib.Path(__file__).resolve().parents[1] / "scripts/pipeline/A_reconstruction/stages/5_decompose_scene.py"


def _data_ready():
    if not os.environ.get(FINAL_VEG_ENV):
        return False
    if not (OBJECTS.is_file() and SAM3_DIR.joinpath("sam3_summary.json").is_file() and SCORE_OBJECTS.is_file()):
        return False
    try:
        import trimesh  # noqa: F401
    except ImportError:
        return False
    return DATA.is_dir()


pytestmark = pytest.mark.skipif(
    not _data_ready(),
    reason=f"set {FINAL_VEG_ENV}: the replay needs the research 1 SAM3 outputs, objects_cpu.json, score_objects.py (with trimesh) and the stored scenes",
)


def _observed_mask_module():
    spec = importlib.util.spec_from_file_location("score_objects_replay", SCORE_OBJECTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _primary_name_function():
    """`get_primary_object_name` of the stage, from its syntax tree (the stage cannot be imported)."""
    tree = ast.parse(STAGE.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_primary_object_name"]
    assert len(fns) == 1, f"expected one get_primary_object_name in the stage, found {len(fns)}"
    ns = {}
    exec(compile(ast.Module(body=fns, type_ignores=[]), "5_decompose_scene.py", "exec"), ns)
    return ns["get_primary_object_name"]


def _prompt_of(obj_cat, primary=None):
    """The SAM3 prompt of an obj_cat (research 2, Q3): its primary name, lower case; the whole obj_cat when that is empty."""
    primary = primary or _primary_name_function()
    name = primary(obj_cat)
    return (name if name else obj_cat).strip().lower()


def _unpack(z, key, gw, gh):
    return np.unpackbits(z[key])[: gw * gh].reshape(gh, gw).astype(bool)


def _scene_inputs(scene):
    sam3 = json.loads((SAM3_DIR / "sam3_summary.json").read_text())["scenes"][scene]
    gw, gh = sam3["grid"]
    z = np.load(SAM3_DIR / f"masks_{scene}.npz")
    K = np.asarray(np.load(DATA / scene / "s2_da/da/exports/npz/results.npz")["intrinsics"][0], float)
    objects = sorted((o for o in json.loads(OBJECTS.read_text()) if o["scene"] == scene),
                     key=lambda o: int(o["name"].split("_")[1]))
    so = _observed_mask_module()
    masks = {}
    for o in objects:
        masks[o["name"]] = so.observed_mask(DATA / scene / "s8_pose/pc" / f"{o['name']}.ply", K, gh, gw)
    lists = {}
    primary = _primary_name_function()
    for path in (DATA / scene / "s5_scene/obj_cat_list").glob("iter_*.json"):
        k = int(path.stem.split("_")[1])
        record = json.loads(path.read_text())
        full = record.get("vlm_categories_full") or record.get("vlm_categories") or []
        lists[k] = [_prompt_of(c, primary) for c in full]
    return dict(sam3=sam3, gw=gw, gh=gh, z=z, objects=objects, masks=masks, lists=lists)


def replay(scenes=None, order="pipeline", params=PARAMS):
    """Run the rules over the scenes. Returns totals, per-object outcomes and the fake-mask count."""
    scenes = scenes or [s for s in json.loads((SAM3_DIR / "sam3_summary.json").read_text())["scenes"]]
    totals = dict(objects=0, right=0, wrong=0, matched=0, revealed=0, trace=0, no_original=0,
                  rescued_own=0, rescued_extra=0, lost=0, fakes=0, fakes_accepted=0)
    per_object = {}
    for scene in scenes:
        inp = _scene_inputs(scene)
        if not inp["objects"]:
            continue
        pool = cs.OriginalPool()
        asked = set()

        def ask(prompt):
            if prompt in asked:
                return
            asked.add(prompt)
            rows = inp["sam3"]["prompts"].get(prompt, [])
            masks = [_unpack(inp["z"], f"{prompt}|{row['idx']}", inp["gw"], inp["gh"]) for row in rows]
            pool.add_prompt(prompt, [row["score"] for row in rows], masks)

        if order == "all_at_once":
            for k in sorted(inp["lists"]):
                for c in inp["lists"][k]:
                    ask(c)
        dropped_names, decisions = [], {}
        for o in inp["objects"]:
            k = int(o["name"].split("_")[1])
            if order == "pipeline":
                for kk in sorted(inp["lists"]):
                    if kk <= k:
                        for c in inp["lists"][kk]:
                            ask(c)
            category = _prompt_of(o["category"].replace("_", " "))
            d = cs.match_iteration_mask(inp["masks"][o["name"]], pool, params, category)
            decisions[o["name"]] = d
            if d.kind == "matched":
                pool.claim(d.index, k)
            elif d.kind == "trace":
                dropped_names.append(o["name"])
        plan = cs.rescue_plan(pool, [inp["masks"][n] for n in dropped_names], params)
        outcome_of = {name: plan.per_dropped[i] for i, name in enumerate(dropped_names)}
        entries = pool.entries()
        for o in inp["objects"]:
            name = o["name"]
            d = decisions[name]
            m = inp["masks"][name]
            ious = [cs._iou(m, e.mask) for e in entries]
            ref = int(np.argmax(ious)) if ious and max(ious) >= 0.1 else None
            totals["objects"] += 1
            record = {"kind": d.kind, "index": d.index, "ref": ref}
            if d.kind == "matched":
                record["matched_prompt"] = entries[d.index].prompt
            if d.kind == "matched":
                totals["matched"] += 1
                totals["right" if d.index == ref else "wrong"] += 1
                record["right"] = d.index == ref
            elif d.kind == "revealed_later":
                totals["revealed"] += 1
            elif d.kind == "no_original":
                totals["no_original"] += 1
            else:
                totals["trace"] += 1
                out = outcome_of[name]
                record["rescue"] = out.reason
                if out.rescued:
                    totals["rescued_own"] += 1
                else:
                    totals["lost"] += 1
            per_object[f"{scene}/{name}"] = record
        totals["rescued_extra"] += len(plan.extra)
        for shift in ((inp["gw"] // 3, 0), (-inp["gw"] // 3, 0), (0, inp["gh"] // 3), (0, -inp["gh"] // 3)):
            for name, m in inp["masks"].items():
                fake = _shift(m, *shift)
                if not fake.any():
                    continue
                totals["fakes"] += 1
                if any(cs._iou(fake, e.mask) >= 0.5 for e in entries):
                    totals["fakes_accepted"] += 1
    return totals, per_object


def _shift(mask, dx, dy):
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    ys2, xs2 = ys + dy, xs + dx
    ok = (ys2 >= 0) & (ys2 < h) & (xs2 >= 0) & (xs2 < w)
    out[ys2[ok], xs2[ok]] = True
    return out


def test_pipeline_order_matches_the_expected_numbers():
    totals, _ = replay(order="pipeline")
    assert totals["objects"] == 57
    assert (totals["matched"], totals["wrong"]) == (46, 0)
    assert (totals["trace"], totals["rescued_own"], totals["lost"]) == (6, 5, 1)
    assert totals["revealed"] == 1 and totals["no_original"] == 4
    assert totals["fakes_accepted"] == 0


def test_all_at_once_order_matches_the_expected_numbers():
    totals, _ = replay(order="all_at_once")
    assert totals["objects"] == 57
    assert (totals["matched"], totals["wrong"]) == (47, 0)
    assert (totals["trace"], totals["rescued_own"], totals["lost"]) == (6, 5, 1)
    assert totals["revealed"] == 1 and totals["no_original"] == 3
    assert totals["fakes_accepted"] == 0


def test_the_primary_names_of_the_full_categories_are_the_short_categories_of_the_stored_scenes():
    # Research 2 (Q3) asks SAM3 with the primary name of each obj_cat. The replay reads the full names and derives the same
    # strings; this checks that they agree with the short names the stored scenes also keep.
    primary = _primary_name_function()
    for scene_dir in sorted(p for p in DATA.iterdir() if (p / "s5_scene/obj_cat_list").is_dir()):
        for path in sorted((scene_dir / "s5_scene/obj_cat_list").glob("iter_*.json")):
            record = json.loads(path.read_text())
            if "vlm_categories_full" not in record:
                continue
            derived = [_prompt_of(c, primary) for c in record["vlm_categories_full"]]
            short = [c.strip().lower() for c in record["vlm_categories"]]
            assert derived == short, f"{scene_dir.name} {path.name}: {derived} != {short}"
