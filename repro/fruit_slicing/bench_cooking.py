# SPDX-License-Identifier: Apache-2.0
"""Time PhysX collider cooking for rigid pieces added while the simulation plays (repro/README.md, section 15.1).

A runtime cut swaps one rigid body for two new ones whose collision geometry nobody has cooked yet. This adds the
two halves of one cut (prepare_cut_pieces.py) as plain USD rigid bodies to a playing OmniGibson scene and times
each stage up to the first physics step and the first render:

- ``author_s``: authoring both pieces on the stage (visual mesh, rigid body, mass, collider prims)
- ``flush_s``: ``IPhysxSimulation.flush_changes()``, where omni.physx parses the new prims and cooks
- ``first_step_s``: the first physics step after that
- ``first_render_s``: the first render, which uploads the new visual meshes

and checks that the colliders are really there after the first step: omni.physx's cooking statistics (tasks
scheduled and finished, cache hits and misses), the PhysX scene's shape counts, and an overlap query over each
piece. A collider still missing keeps the loop stepping until it appears (``ready_steps``). Then 60 physics steps
and 8 full steps are timed with the pieces present, the pieces are checked for having fallen through the floor,
and they are removed. OmniGibson does not know these prims: this is the cost below OmniGibson, which
bench_swap.py measures on top.

Collision variants per half (cut_common.colliders):

- ``box``: an analytic box over the piece's bounds. Nothing to cook: the control.
- ``hull``: one convex hull of the 2k-triangle proxy half; ``hull_dense``: of the full-resolution visual half.
- ``clip``: the asset's own convex hulls clipped by the plane, each cooked as a convex hull.
- ``coacd8``: CoACD's decomposition of the proxy half into at most 8 parts, computed offline (its time is in
  manifest.json). A half CoACD failed on has none, and the variant is skipped for that cut.
- ``vhacd``: the proxy half with ``convexDecomposition`` at PhysX's defaults: PhysX runs V-HACD while cooking.
  ``vhacd4/8/16``: the same with at most 4/8/16 hulls and a 1 % error tolerance, so that the limit binds.
- ``sdf128/256``: the proxy half as an SDF triangle mesh; ``sdf256_dense`` the visual half (opt-in).

Every cut is a different mesh, and each process offsets its meshes by a few micrometres, so a first add never hits
a cooking cache an earlier run filled (omni.physx keeps one on disk). ``--repeat`` re-adds the same meshes to
measure a hit on purpose. The scene is built the way teleop builds it: ``apply_teleop_omnigibson_macros()`` (CPU
dynamics, CCD on, Fabric on), physics at 120 Hz, actions and rendering at 15 Hz.

Runs in the simfoundry env with this checkout first on PYTHONPATH, and boots Isaac Sim (about 90 s)::

    PYTHONPATH=$PWD ~/miniforge3/envs/simfoundry/bin/python repro/fruit_slicing/bench_cooking.py \\
        --scene /home/inner/real2sim/SimFoundry/assets/scenes/DROID/droid_desk_serve_fruits/droid_desk_serve_fruits_scene_state_latest.json \\
        --pieces results/diagnostics/fruit_slicing/pieces --out results/diagnostics/fruit_slicing/cooking.json
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from cut_common import VARIANTS, author_rigid_body, collider_triangles, colliders, half_masses

REPEAT = ("hull", "vhacd8", "sdf256")
SPOT_XY = (0.85, -0.45)  # free desk spot for the first half; the second goes beside it, GAP further along x
GAP = 0.03
GROUND_CLEARANCE = 0.002


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="scene state JSON the pieces are added to")
    ap.add_argument("--pieces", required=True, help="prepare_cut_pieces.py output directory")
    ap.add_argument("--out", required=True, help="JSON report")
    ap.add_argument("--fruits", nargs="+", default=None, help="default: every fruit in the manifest")
    ap.add_argument("--cuts", type=int, nargs="+", default=None, help="cut indices; default: all")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--repeat", nargs="*", default=list(REPEAT), help="variants re-added unchanged (cache hit)")
    ap.add_argument("--no-visual", action="store_true", help="author colliders only")
    ap.add_argument("--steady-steps", type=int, default=60)
    ap.add_argument("--max-wait-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=None, help="mesh offset and order seed; default: from the clock")
    args = ap.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    from simfoundry import configure_omnigibson_data_path

    configure_omnigibson_data_path(force=True)

    import omnigibson as og
    from omnigibson.macros import gm
    from simfoundry.utils.ground_plane_utils import apply_ground_plane_info
    from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros
    from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

    apply_teleop_omnigibson_macros()
    scene_json = load_json_with_absolute_usd_paths(args.scene)
    env = og.Environment(configs=dict(
        env=dict(action_frequency=15, rendering_frequency=15, physics_frequency=120),
        scene=dict(type="Scene", scene_file=scene_json, use_floor_plane=True, floor_plane_visible=True,
                   use_skybox=False, include_robots=True)))
    ground = apply_ground_plane_info(scene_json, og.sim.floor_plane)
    env.reset()
    for _ in range(30):
        og.sim.step()

    import carb
    import omni.physx as ophysx
    import omni.usd
    from omni.physx.bindings._physx import PhysicsSceneStats
    from pxr import PhysicsSchemaTools, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation

    stage = og.sim.stage
    stage_id = omni.usd.get_context().get_stage_id()
    sim_iface = ophysx.get_physx_simulation_interface()
    physx_iface = ophysx.get_physx_interface()
    cooking = ophysx.get_physx_cooking_private_interface()
    sq = ophysx.get_physx_scene_query_interface()
    stats_iface = ophysx.get_physx_statistics_interface()
    scene_path_int = PhysicsSchemaTools.sdfPathToInt(
        next(str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)))
    ground_z = float(ground["position"][2]) if ground else 0.0
    UsdGeom.Xform.Define(stage, "/World/cut_bench")

    def vec(v, n):
        """A carb float3/float4 (or tuple) as a numpy array."""
        try:
            return np.array([v[i] for i in range(n)], dtype=float)
        except TypeError:
            return np.array([getattr(v, a) for a in "xyzw"[:n]], dtype=float)

    def cook_stats():
        c = cooking.get_cooking_statistics()
        return {"scheduled": c.total_scheduled_tasks, "finished": c.total_finished_tasks,
                "hit": c.total_finished_cache_hit_tasks, "miss": c.total_finished_cache_miss_tasks}

    def scene_stats():
        st = PhysicsSceneStats()
        stats_iface.get_physx_scene_statistics(stage_id, scene_path_int, st)
        return {"dynamic": st.nb_dynamic_rigids, "convex": st.nb_convex_shapes, "trimesh": st.nb_trimesh_shapes,
                "box": st.nb_box_shapes}

    def delta(a, b):
        return {k: b[k] - a[k] for k in a}

    def collider_present(path, lo, hi):
        """Whether PhysX's scene holds a shape of @path: an overlap box over its current world bounds. A box,
        not a sphere inside the piece, since a triangle-mesh (SDF) shape is hit by its surface only."""
        tr = physx_iface.get_rigidbody_transformation(path)
        if not tr["ret_val"]:
            return False
        R = Rotation.from_quat(vec(tr["rotation"], 4)).as_matrix()
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = corners @ R.T + vec(tr["position"], 3)
        c, h = (w.min(0) + w.max(0)) / 2, (w.max(0) - w.min(0)) / 2 + 0.002
        found = []

        def report(hit):
            if hit.rigid_body == path:
                found.append(hit.collision)
            return True

        sq.overlap_box(carb.Float3(*h.tolist()), carb.Float3(*c.tolist()), carb.Float4(0, 0, 0, 1), report, False)
        return bool(found)

    def body_z(path):
        tr = physx_iface.get_rigidbody_transformation(path)
        return float(vec(tr["position"], 3)[2]) if tr["ret_val"] else float("nan")

    def time_calls(fn, n):
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return ts

    manifest = json.load(open(Path(args.pieces) / "manifest.json"))
    fruits = args.fruits or list(manifest["fruits"])
    seed = args.seed if args.seed is not None else time.time_ns() % (2 ** 32)
    rng = np.random.default_rng(seed)
    report = {"conditions": conditions(args, gm, og, ground, manifest, fruits, seed), "runs": [], "baseline": []}

    def baseline(tag):
        phys = time_calls(og.sim.step_physics, args.steady_steps)
        full = time_calls(og.sim.step, 8)
        render = time_calls(og.sim.render, 3)
        row = {"tag": tag, "physics_step_ms": 1e3 * statistics.median(phys),
               "sim_step_ms": 1e3 * statistics.median(full), "render_ms": 1e3 * statistics.median(render),
               "cook": cook_stats(), "shapes": scene_stats()}
        report["baseline"].append(row)
        return row

    b = baseline("start")
    print(f"[bench] baseline: physics step {b['physics_step_ms']:.2f} ms, sim step {b['sim_step_ms']:.1f} ms, "
          f"render {b['render_ms']:.1f} ms, shapes {b['shapes']}", flush=True)
    n_run = 0
    for fruit in fruits:
        info = manifest["fruits"][fruit]
        fz = np.load(Path(args.pieces) / f"{fruit}.npz")
        R_fruit = Rotation.from_quat(fz["ori"]).as_matrix()
        for k in (args.cuts if args.cuts is not None else [c["k"] for c in info["cuts"]]):
            z = np.load(Path(args.pieces) / f"{fruit}_cut{k}.npz")
            masses = half_masses(info["cuts"][k], float(fz["mass"]))
            # Each half keeps the fruit's orientation and rests on the floor; the two sit side by side.
            poses, x = [], SPOT_XY[0]
            for side in (0, 1):
                Vw = z[f"s{side}_visual_V"] @ R_fruit.T
                lo, hi = Vw.min(0), Vw.max(0)
                poses.append(np.array([x - lo[0], SPOT_XY[1] - (lo[1] + hi[1]) / 2,
                                       ground_z + GROUND_CLEARANCE - lo[2]]))
                x += (hi[0] - lo[0]) + GAP
            bounds = [(z[f"s{side}_visual_V"].min(0), z[f"s{side}_visual_V"].max(0)) for side in (0, 1)]
            visual = [None if args.no_visual else (z[f"s{side}_visual_V"], z[f"s{side}_visual_F"]) for side in (0, 1)]
            jitter = {v: rng.uniform(-5e-6, 5e-6, 3) for v in args.variants}
            order = list(args.variants)
            rng.shuffle(order)
            order += [f"{v}#2" for v in args.repeat if v in args.variants]  # same meshes again: a cache hit
            for run_variant in order:
                variant = run_variant.split("#")[0]
                specs = [colliders(variant, z, side, jitter[variant]) for side in (0, 1)]
                if not all(specs):
                    report.setdefault("skipped", []).append({"fruit": fruit, "cut": k, "variant": run_variant})
                    continue
                paths = [f"/World/cut_bench/run{n_run}_{side}" for side in (0, 1)]
                n_run += 1
                for _ in range(2):
                    og.sim.step_physics()
                c0, s0 = cook_stats(), scene_stats()
                t0 = time.perf_counter()
                # Plain Usd API, no Sdf.ChangeBlock: inside one, Usd defers recomposing, and prims it defines are
                # not valid until the block closes.
                for side in (0, 1):
                    author_rigid_body(stage, paths[side], masses[side], visual[side], specs[side],
                                      pos=poses[side], quat_xyzw=fz["ori"])
                t1 = time.perf_counter()
                sim_iface.flush_changes()
                t2 = time.perf_counter()
                c_flush = cook_stats()
                og.sim.step_physics()
                t3 = time.perf_counter()
                c_step, s_step = cook_stats(), scene_stats()
                present = [collider_present(p, *bd) for p, bd in zip(paths, bounds)]
                present_first, steps = list(present), 1
                while not all(present) and steps < args.max_wait_steps:
                    og.sim.step_physics()
                    steps += 1
                    present = [collider_present(p, *bd) for p, bd in zip(paths, bounds)]
                t_ready = time.perf_counter()
                t4 = time.perf_counter()
                og.sim.render()
                t5 = time.perf_counter()
                phys = time_calls(og.sim.step_physics, args.steady_steps)
                full = time_calls(og.sim.step, 8)
                z_end, s_end = [body_z(p) for p in paths], scene_stats()
                t6 = time.perf_counter()
                for p in paths:
                    stage.RemovePrim(p)
                sim_iface.flush_changes()
                og.sim.step_physics()
                t7 = time.perf_counter()
                row = {
                    "fruit": fruit, "cut": k, "variant": variant, "repeat": run_variant.endswith("#2"),
                    "collider_prims": [len(s) for s in specs],
                    "collider_triangles": [collider_triangles(s) for s in specs],
                    "visual_triangles": [len(z[f"s{side}_visual_F"]) for side in (0, 1)], "mass_kg": masses,
                    "author_s": t1 - t0, "flush_s": t2 - t1, "first_step_s": t3 - t2, "add_to_first_step_s": t3 - t0,
                    "cook_pending_after_flush": c_flush["scheduled"] - c_flush["finished"],
                    "cook_after_flush": delta(c0, c_flush), "cook_after_first_step": delta(c0, c_step),
                    "shapes_after_first_step": delta(s0, s_step), "shapes_steady": delta(s0, s_end),
                    "present_after_first_step": present_first, "present": present, "ready_steps": steps,
                    "add_to_ready_s": t_ready - t0, "first_render_s": t5 - t4,
                    "physics_step_ms": 1e3 * statistics.median(phys), "sim_step_ms": 1e3 * statistics.median(full),
                    "z_drop_m": [p[2] - ze for p, ze in zip(poses, z_end)], "remove_s": t7 - t6,
                }
                report["runs"].append(row)
                print(f"[bench] {fruit} cut{k} {run_variant:10s} prims {row['collider_prims']} "
                      f"author {row['author_s']*1e3:6.1f} flush {row['flush_s']*1e3:8.1f} step1 "
                      f"{row['first_step_s']*1e3:6.1f} ms | cook miss/hit {row['cook_after_first_step']['miss']}/"
                      f"{row['cook_after_first_step']['hit']} pending {row['cook_pending_after_flush']} | shapes "
                      f"{row['shapes_after_first_step']} present {present_first} ready@{steps} | render "
                      f"{row['first_render_s']*1e3:6.1f} ms | phys {row['physics_step_ms']:.2f} ms | z drop "
                      f"{[round(d * 1e3, 1) for d in row['z_drop_m']]} mm", flush=True)
            baseline(f"{fruit}_cut{k}")
    report["summary"] = summarize(report)
    report["conditions"]["nvidia_smi_end"] = gpu_apps()
    report["conditions"]["loadavg_end"] = os.getloadavg()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(json.dumps(report["summary"], indent=1), flush=True)
    print(f"[bench] wrote {args.out}", flush=True)
    os._exit(0)  # OmniGibson's atexit teardown is slow and sometimes hangs; the report is written


def gpu_apps():
    """Compute apps on the GPU, to show nothing else ran during the timing."""
    try:
        return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"nvidia-smi failed: {e}"


def conditions(args, gm, og, ground, manifest, fruits, seed):
    """Everything needed to compare these numbers with another run."""
    import carb
    import omni.kit.app

    settings = carb.settings.get_settings()
    mgr = omni.kit.app.get_app().get_extension_manager()
    exts = {e["name"]: e["version"] for e in mgr.get_extensions() if e["name"] in (
        "omni.physx", "omni.physx.cooking", "omni.usd.schema.physx", "omni.convexdecomposition")}
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "?")
    return {
        "scene": os.path.abspath(args.scene), "fruits": fruits, "variants": args.variants, "repeat": args.repeat,
        "visual_mesh": not args.no_visual, "seed": seed, "omnigibson": og.__version__,
        "isaacsim": metadata.version("isaacsim"), "extensions": {k: str(v) for k, v in exts.items()},
        "macros": {k: getattr(gm, k) for k in ("USE_GPU_DYNAMICS", "ENABLE_FLATCACHE", "ENABLE_CCD",
                                                "ENABLE_OBJECT_STATES", "ENABLE_TRANSITION_RULES", "HEADLESS")},
        "physics_dt": og.sim.get_physics_dt(), "rendering_dt": og.sim.get_rendering_dt(),
        "cooking_settings": {k: settings.get(k) for k in (
            "/physics/cooking/ujitsoCollisionCooking", "/persistent/physics/useLocalMeshCache",
            "/persistent/physics/localMeshCacheSizeMB", "/persistent/physics/cooking/ujitsoCookingMaxProcessCount")},
        "ground_plane": ground, "host": platform.node(), "cpu": cpu, "cpu_count": os.cpu_count(),
        "loadavg_start": os.getloadavg(), "nvidia_smi_start": gpu_apps(),
        "pieces": {f: {k: manifest["fruits"][f][k] for k in ("visual_faces", "proxy_faces", "asset_hulls")}
                   for f in fruits},
    }


def summarize(report):
    """Medians per variant (first adds and cache-hit repeats apart), next to the baseline without pieces."""
    def med(rows, key, f=lambda v: v):
        return statistics.median(f(r[key]) for r in rows)

    out = {"baseline_physics_step_ms": statistics.median(b["physics_step_ms"] for b in report["baseline"]),
           "baseline_sim_step_ms": statistics.median(b["sim_step_ms"] for b in report["baseline"]),
           "baseline_render_ms": statistics.median(b["render_ms"] for b in report["baseline"]), "variants": {}}
    for variant in dict.fromkeys(r["variant"] for r in report["runs"]):
        for repeat in (False, True):
            rows = [r for r in report["runs"] if r["variant"] == variant and r["repeat"] == repeat]
            if not rows:
                continue
            out["variants"][variant + ("#2" if repeat else "")] = {
                "n": len(rows), "collider_prims_per_cut": med(rows, "collider_prims", sum),
                "collider_triangles_per_cut": med(rows, "collider_triangles", sum),
                "author_ms": 1e3 * med(rows, "author_s"), "flush_ms": 1e3 * med(rows, "flush_s"),
                "first_step_ms": 1e3 * med(rows, "first_step_s"),
                "add_to_first_step_ms": 1e3 * med(rows, "add_to_first_step_s"),
                "add_to_first_step_ms_max": 1e3 * max(r["add_to_first_step_s"] for r in rows),
                "first_render_ms": 1e3 * med(rows, "first_render_s"),
                "runs_all_present_after_first_step": sum(all(r["present_after_first_step"]) for r in rows),
                "ready_steps_max": max(r["ready_steps"] for r in rows),
                "cook_pending_after_flush_max": max(r["cook_pending_after_flush"] for r in rows),
                "cache_miss": sum(r["cook_after_first_step"]["miss"] for r in rows),
                "cache_hit": sum(r["cook_after_first_step"]["hit"] for r in rows),
                "physics_step_ms": med(rows, "physics_step_ms"), "sim_step_ms": med(rows, "sim_step_ms"),
                "z_drop_mm_max": 1e3 * max(max(r["z_drop_m"]) for r in rows),
            }
    return out


if __name__ == "__main__":
    main()
