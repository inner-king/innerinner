# SPDX-License-Identifier: Apache-2.0
"""Time OmniGibson's mid-episode remove/add, and record and save across it (repro/README.md, section 15.1).

A cut implemented on OmniGibson replaces one object with two in the middle of an episode. This does exactly that,
through OmniGibson's own path, in an environment built like teleop's and wrapped in the HDF5CollectionWrapper
that teleop and policy evaluation record through. Each episode:

1. ``reset()`` (the wrapper flushes the previous episode; ``scene.reset()`` restores the original object set, so
   after a cut it removes the two pieces and adds the fruit back: timed as ``reset_s``)
2. ``--steps-before`` recorded env steps with a hold action (timed)
3. the cut: write both halves as USD files (``write_s``), ``og.sim.batch_remove_objects([fruit])``
   (``remove_s``: dump state, move to the graveyard, one physics step, view rebuild, state reload), then add the
   two pieces at the fruit's pose (``add_s``), either one ``scene.add_object()`` each as
   ``TransitionRuleAPI.execute_transition`` does (``--add-modes rule``) or one ``og.sim.batch_add_objects()``
   (``batch``), and the first env step, which initializes them (``first_step_s``)
4. ``--steps-after`` recorded env steps, plus 0, 5 or 10 so that episode lengths differ (timed)
5. ``scene.save()`` and ``og.sim.save()`` with the pieces in the scene (timed). ``og.sim.save()`` runs twice:
   as recording leaves it, with the wrapper's dump filter on (``saved_ep<N>_filtered.json``), and with the filter
   off as the wrapper's own checkpoints save (``saved_ep<N>.json``, which check_cut_reload.py loads)

Physics alone is timed too (``og.sim.step_physics()``, before step 2 and after step 4). After the last episode the
wrapper's file is closed and read back: per episode the recorded transitions and the serialized state size before
and after the cut.

Each episode uses the next cut of ``--fruit`` from prepare_cut_pieces.py, offset by a few micrometres so no
cooking cache serves it. Collision is ``--variant`` (cut_common.colliders), ``clip`` by default.

Runs in the simfoundry env with this checkout first on PYTHONPATH, and boots Isaac Sim (about 90 s)::

    PYTHONPATH=$PWD ~/miniforge3/envs/simfoundry/bin/python repro/fruit_slicing/bench_swap.py \\
        --scene /home/inner/real2sim/SimFoundry/assets/scenes/DROID/droid_desk_serve_fruits/droid_desk_serve_fruits_scene_state_latest.json \\
        --pieces results/diagnostics/fruit_slicing/pieces --out-dir results/diagnostics/fruit_slicing/swap
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from cut_common import colliders, half_masses, write_piece_usd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pieces", required=True, help="prepare_cut_pieces.py output directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fruit", default="red_apple_zlgmzo_2")
    ap.add_argument("--variant", default="clip")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--add-modes", nargs="+", default=["rule", "batch"], choices=["rule", "batch"])
    ap.add_argument("--steps-before", type=int, default=30)
    ap.add_argument("--steps-after", type=int, default=30)
    ap.add_argument("--physics-steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--profile", action="store_true",
                    help="profile the remove, the add and the first step (cProfile); its timings are inflated, so a "
                         "profiled run is for where the time goes, not how much there is")
    args = ap.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    from simfoundry import configure_omnigibson_data_path

    configure_omnigibson_data_path(force=True)

    import omnigibson as og
    import torch as th
    from omnigibson.envs import HDF5CollectionWrapper
    from omnigibson.macros import gm
    from omnigibson.objects import USDObject
    from omnigibson.sensors import VisionSensor
    from simfoundry.utils.ground_plane_utils import apply_ground_plane_info
    from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros
    from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

    out_dir = Path(args.out_dir).resolve()
    (out_dir / "pieces").mkdir(parents=True, exist_ok=True)
    apply_teleop_omnigibson_macros()
    scene_json = load_json_with_absolute_usd_paths(args.scene)
    env = og.Environment(configs=dict(
        env=dict(action_frequency=15, rendering_frequency=15, physics_frequency=120),
        scene=dict(type="Scene", scene_file=scene_json, use_floor_plane=True, floor_plane_visible=True,
                   use_skybox=False, include_robots=True)))
    ground = apply_ground_plane_info(scene_json, og.sim.floor_plane)
    env.reset()
    for _ in range(10):
        og.sim.step()

    # As teleop and policy evaluation wrap it (scripts/cfg/real2sim_cfg.yaml, s14_teleop.data_env_kwargs). A
    # viewport camera is what turns the per-step dump filter on; policy evaluation then turns its cameras back on.
    viewport = "/World/viewer_camera" if og.sim.viewer_camera is not None else None
    wrapper = HDF5CollectionWrapper(env=env, output_path=str(out_dir / "rollouts.hdf5"), only_successes=False,
                                    flush_every_n_traj=1, obj_attr_keys=["scale", "visible"],
                                    viewport_camera_path=viewport)
    if viewport is None:
        wrapper.enable_dump_filters()
    for sensor in VisionSensor.SENSORS.values():
        sensor.render_product.hydra_texture.set_updates_enabled(True)

    manifest = json.load(open(Path(args.pieces) / "manifest.json"))
    info = manifest["fruits"][args.fruit]
    fz = np.load(Path(args.pieces) / f"{args.fruit}.npz")
    category = env.scene.object_registry("name", args.fruit).category
    rng = np.random.default_rng(args.seed if args.seed is not None else time.time_ns() % (2 ** 32))
    robot = env.robots[0]
    action = th.zeros(robot.action_dim)  # the scene's IK arm controller takes pose deltas: zero holds

    def timed(fn):
        t = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - t

    def step_times(n):
        return [timed(lambda: wrapper.step(action))[1] for _ in range(n)]

    def physics_times(n):
        return [timed(og.sim.step_physics)[1] for _ in range(n)]

    report = {"conditions": {
        "scene": os.path.abspath(args.scene), "fruit": args.fruit, "variant": args.variant,
        "add_modes": args.add_modes, "episodes": args.episodes, "steps_before": args.steps_before,
        "steps_after": args.steps_after, "omnigibson": og.__version__,
        "macros": {k: getattr(gm, k) for k in ("USE_GPU_DYNAMICS", "ENABLE_FLATCACHE", "ENABLE_CCD",
                                                "ENABLE_OBJECT_STATES", "ENABLE_TRANSITION_RULES", "HEADLESS")},
        "physics_dt": og.sim.get_physics_dt(), "rendering_dt": og.sim.get_rendering_dt(),
        "ground_plane": ground, "viewport_camera_path": viewport,
        "sensors": {k: [s.image_width, s.image_height, list(s.modalities)] for k, s in VisionSensor.SENSORS.items()},
        "n_objects_before": len(env.scene.objects), "visual_faces_fruit": info["visual_faces"],
    }, "episodes": []}
    print(f"[swap] {len(env.scene.objects)} objects; sensors {report['conditions']['sensors']}", flush=True)

    n_cuts = len(info["cuts"])
    for ep in range(args.episodes):
        mode = args.add_modes[ep % len(args.add_modes)]
        k = ep % n_cuts
        rec = {"episode": ep, "cut": k, "add_mode": mode}
        _, rec["reset_s"] = timed(wrapper.reset)
        fruit = env.scene.object_registry("name", args.fruit)
        assert fruit is not None, "reset did not bring the fruit back"
        rec["n_objects_start"] = len(env.scene.objects)
        rec["physics_before_s"] = physics_times(args.physics_steps)
        rec["step_before_s"] = step_times(args.steps_before)
        rec["state_size_before"] = int(len(og.sim.dump_state(serialized=True)))

        # ---- the cut
        z = np.load(Path(args.pieces) / f"{args.fruit}_cut{k}.npz")
        masses = half_masses(info["cuts"][k], float(fz["mass"]))
        jitter = rng.uniform(-5e-6, 5e-6, 3)
        pos, ori = fruit.get_position_orientation()
        names = [f"{args.fruit}_ep{ep}_half{side}" for side in (0, 1)]
        paths = [out_dir / "pieces" / f"{name}.usd" for name in names]
        profiler = None
        if args.profile:
            import cProfile

            profiler = cProfile.Profile()
        t0 = time.perf_counter()
        for side in (0, 1):
            write_piece_usd(paths[side], masses[side], (z[f"s{side}_visual_V"], z[f"s{side}_visual_F"]),
                            colliders(args.variant, z, side, jitter), category=f"{category}_half")
        t1 = time.perf_counter()
        if profiler:
            profiler.enable()
        og.sim.batch_remove_objects([fruit])
        t2 = time.perf_counter()
        pieces = [USDObject(name=n, usd_path=str(p), category=f"{category}_half") for n, p in zip(names, paths)]
        if mode == "rule":
            for piece in pieces:
                env.scene.add_object(piece)
                piece.set_position_orientation(position=pos, orientation=ori)
        else:
            og.sim.batch_add_objects(pieces, [env.scene] * len(pieces))
            for piece in pieces:
                piece.set_position_orientation(position=pos, orientation=ori)
        t3 = time.perf_counter()
        wrapper.step(action)
        t4 = time.perf_counter()
        if profiler:
            profiler.disable()
            rec["profile"] = top_functions(profiler)
        rec.update(write_s=t1 - t0, remove_s=t2 - t1, add_s=t3 - t2, first_step_s=t4 - t3, cut_total_s=t4 - t0,
                   usd_bytes=[p.stat().st_size for p in paths],
                   collider_prims=[len(colliders(args.variant, z, side)) for side in (0, 1)])
        # Episode lengths differ (+0/5/10 steps), as teleop's do: what the wrapper records during the next reset
        # is keyed by this episode's last step, which only a longer next episode reaches on replay.
        rec["step_after_s"] = step_times(args.steps_after + 5 * (ep % 3))
        rec["state_size_after"] = int(len(og.sim.dump_state(serialized=True)))
        rec["physics_after_s"] = physics_times(args.physics_steps)
        rec["n_objects_end"] = len(env.scene.objects)

        # ---- saving with the pieces in the scene
        def scene_save():
            return json.loads(env.scene.save())

        saved, rec["scene_save_s"] = attempt(lambda: timed(scene_save), rec, "scene_save")
        if saved is not None:
            names_saved = set(saved["objects_info"]["init_info"])
            rec["scene_save_has_pieces"] = all(n in names_saved for n in names)
            rec["scene_save_has_fruit"] = args.fruit in names_saved
        # og.sim.save() as the recording leaves it: the wrapper's per-step dump filter keeps only awake, non-kinematic
        # objects in the dumped state. Then with the filter off for the save, as the wrapper's own
        # update_checkpoint() does; that file is the one check_cut_reload.py loads.
        filtered_path = out_dir / f"saved_ep{ep}_filtered.json"
        _, rec["sim_save_s"] = attempt(lambda: timed(lambda: og.sim.save(json_paths=[str(filtered_path)])), rec,
                                       "og_sim_save")
        json_path = out_dir / f"saved_ep{ep}.json"
        wrapper.disable_dump_filters()
        _, rec["sim_save_unfiltered_s"] = attempt(lambda: timed(lambda: og.sim.save(json_paths=[str(json_path)])),
                                                  rec, "og_sim_save_unfiltered")
        wrapper.enable_dump_filters()
        for key, path in (("filtered", filtered_path), ("unfiltered", json_path)):
            if path.exists():
                with open(path) as f:
                    doc = json.load(f)
                rec[f"saved_{key}"] = {"file": str(path), "objects": len(doc["objects_info"]["init_info"]),
                                       "objects_with_state": len(doc["state"]["registry"]["object_registry"])}
        rec["piece_poses"] = {p.name: [t.tolist() for t in p.get_position_orientation()] for p in pieces}
        rec["piece_z_drop_m"] = [float(pos[2] - p.get_position_orientation()[0][2]) for p in pieces]
        rec["transitions"] = json.loads(json.dumps(wrapper.current_transitions, default=str))
        report["episodes"].append(rec)
        med = statistics.median
        print(f"[swap] ep{ep} cut{k} {mode:5s} reset {rec['reset_s']:.2f} s | "
              f"step {med(rec['step_before_s'])*1e3:.1f} -> "
              f"{med(rec['step_after_s'])*1e3:.1f} ms | phys {med(rec['physics_before_s'])*1e3:.2f} -> "
              f"{med(rec['physics_after_s'])*1e3:.2f} ms | write {rec['write_s']*1e3:.0f} remove "
              f"{rec['remove_s']*1e3:.0f} add {rec['add_s']*1e3:.0f} first step {rec['first_step_s']*1e3:.0f} ms | "
              f"state {rec['state_size_before']} -> {rec['state_size_after']} | saves {rec.get('scene_save')}/"
              f"{rec.get('og_sim_save')}/{rec.get('og_sim_save_unfiltered')} | objects with state "
              f"{rec.get('saved_filtered', {}).get('objects_with_state')}/"
              f"{rec.get('saved_unfiltered', {}).get('objects_with_state')} of "
              f"{rec.get('saved_unfiltered', {}).get('objects')}", flush=True)

    _, report["close_s"] = attempt(lambda: timed(wrapper.save_data), report, "hdf5_close")
    report["hdf5"] = inspect_hdf5(out_dir / "rollouts.hdf5")
    report["summary"] = summarize(report)
    with open(out_dir / "swap.json", "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(json.dumps(report["summary"], indent=1), flush=True)
    print(f"[swap] wrote {out_dir / 'swap.json'}", flush=True)
    os._exit(0)  # OmniGibson's atexit teardown is slow and sometimes hangs; the report is written


def top_functions(profiler, n=25):
    """The @n functions with the most cumulative time, as (seconds, calls, "file:line(function)")."""
    import pstats

    stats = pstats.Stats(profiler)
    rows = sorted(stats.stats.items(), key=lambda kv: -kv[1][3])[:n]
    return [(round(ct, 4), nc, f"{os.path.basename(f)}:{line}({fn})") for (f, line, fn), (_, nc, _, ct, _) in rows]


def attempt(fn, rec, key):
    """Run @fn, recording "ok" or the exception under @key; returns fn's result split as (value, seconds)."""
    try:
        value, seconds = fn()
        rec[key] = "ok"
        return value, seconds
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        rec[key] = f"{type(e).__name__}: {e}"
        return None, None


def inspect_hdf5(path):
    """What the wrapper wrote: per demo the number of samples, the state sizes and the recorded transitions."""
    import h5py

    out = {"path": str(path), "demos": {}}
    try:
        with h5py.File(path, "r") as f:
            data = f["data"]
            out["n_episodes"] = int(data.attrs.get("n_episodes", -1))
            out["scene_file_objects"] = sorted(json.loads(data.attrs["scene_file"])["objects_info"]["init_info"])
            for name in data:
                grp = data[name]
                sizes = grp["state_size"][()].astype(int)
                trans = json.loads(grp.attrs["transitions"])
                out["demos"][name] = {
                    "num_samples": int(grp.attrs["num_samples"]), "state_shape": list(grp["state"].shape),
                    "state_size_first": int(sizes[0]), "state_size_last": int(sizes[-1]),
                    "state_size_changes_at": [int(i) for i in np.flatnonzero(np.diff(sizes)) + 1],
                    "transitions": {step: {kind: {op: [v if isinstance(v, str) else v["args"]["name"] for v in vals]
                                                  for op, vals in ops.items()}
                                           for kind, ops in t.items()} for step, t in trans.items()},
                }
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def summarize(report):
    """Medians over episodes, per add mode and pooled."""
    eps = report["episodes"]
    med = statistics.median

    def block(rows):
        if not rows:
            return None
        return {
            "n": len(rows),
            "step_before_ms": 1e3 * med(med(r["step_before_s"]) for r in rows),
            "step_after_ms": 1e3 * med(med(r["step_after_s"]) for r in rows),
            "physics_before_ms": 1e3 * med(med(r["physics_before_s"]) for r in rows),
            "physics_after_ms": 1e3 * med(med(r["physics_after_s"]) for r in rows),
            "write_ms": 1e3 * med(r["write_s"] for r in rows), "remove_ms": 1e3 * med(r["remove_s"] for r in rows),
            "add_ms": 1e3 * med(r["add_s"] for r in rows), "first_step_ms": 1e3 * med(r["first_step_s"] for r in rows),
            "cut_total_ms": 1e3 * med(r["cut_total_s"] for r in rows),
            "cut_total_ms_max": 1e3 * max(r["cut_total_s"] for r in rows),
            "reset_after_cut_s": med(r["reset_s"] for r in rows if r["episode"] > 0) if len(rows) > 1 else None,
            "scene_save_ms": 1e3 * med(r["scene_save_s"] for r in rows if r["scene_save_s"] is not None)
            if any(r["scene_save_s"] is not None for r in rows) else None,
            "sim_save_ms": 1e3 * med(r["sim_save_s"] for r in rows if r["sim_save_s"] is not None)
            if any(r["sim_save_s"] is not None for r in rows) else None,
        }

    return {"all": block(eps), **{mode: block([r for r in eps if r["add_mode"] == mode])
                                   for mode in dict.fromkeys(r["add_mode"] for r in eps)},
            "saves_ok": all(r.get("scene_save") == "ok" and r.get("og_sim_save") == "ok"
                            and r.get("og_sim_save_unfiltered") == "ok" for r in eps),
            "hdf5_close": report.get("hdf5_close")}


if __name__ == "__main__":
    main()
