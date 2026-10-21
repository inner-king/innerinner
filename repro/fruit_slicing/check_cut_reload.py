# SPDX-License-Identifier: Apache-2.0
"""Reload what bench_swap.py saved after a cut, and replay what it recorded (repro/README.md, section 15.1).

One mode per Isaac Sim process, since OmniGibson loads a scene once per process and replay needs transition rules
off (stages 3, 4 and 6 call ``apply_teleop_omnigibson_macros(enable_tr=False)``):

- ``env``: an og.Environment with a saved JSON as its scene_file, as teleop (2_teleop_og_scene.py) and policy
  evaluation load a scene.
- ``restore``: ``og.sim.restore()`` on a fresh simulator, as the light editor's settle.py and parity_check.py do.

  Both check that the pieces are there and the fruit is not, how far the loaded poses are from the saved ones,
  how far they move over 60 steps (4 s), that the scene saves again, and that a reset keeps the pieces. The same
  gaps are measured for the uncut props as a control (``--saved`` may also be an uncut scene).
  ``og.sim.save()`` does not write SimFoundry's ``ground_plane_info``, so the floor is put where ``--scene`` (the
  scene the cut was made in) says, as PickPlaceTask would for that scene.

- ``replay``: every recorded episode through OmniGibson's playback (``HDF5PlaybackWrapper``, whose
  ``playback_episode`` is the one stage 6's LeRobot wrapper inherits; it executes the recorded transitions), then
  through SimFoundry's ``MinimalPlaybackWrapper`` (stages 3 and 4; it skips them). Per episode: whether it ran to
  the end, the exception if not, and for a completed replay which objects are left and the largest gap between the
  replayed and recorded poses of the pieces.

Runs in the simfoundry env with this checkout first on PYTHONPATH, and boots Isaac Sim (about 90 s)::

    PYTHONPATH=$PWD ~/miniforge3/envs/simfoundry/bin/python repro/fruit_slicing/check_cut_reload.py env \\
        --saved results/diagnostics/fruit_slicing/swap/saved_ep0.json --scene <the scene bench_swap.py cut> \\
        --out results/diagnostics/fruit_slicing/swap/reload_env.json
    ... check_cut_reload.py restore --saved ... --scene ... --out .../reload_restore.json
    ... check_cut_reload.py replay --hdf5 results/diagnostics/fruit_slicing/swap/rollouts.hdf5 --scene ... \\
        --out .../replay.json
"""
import argparse
import json
import math
import os
import tempfile
import traceback


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["env", "restore", "replay"])
    ap.add_argument("--saved", help="og.sim.save() output (env, restore)")
    ap.add_argument("--scene", help="the scene the cut was made in, for its ground_plane_info")
    ap.add_argument("--hdf5", help="bench_swap.py's rollouts.hdf5 (replay)")
    ap.add_argument("--fruit", default="red_apple_zlgmzo_2")
    ap.add_argument("--settle-steps", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    from simfoundry import configure_omnigibson_data_path

    configure_omnigibson_data_path(force=True)

    import omnigibson as og  # noqa: F401 - boots Isaac Sim before the other omnigibson imports
    from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros

    apply_teleop_omnigibson_macros(enable_tr=args.mode != "replay")
    report = {"mode": args.mode}
    try:
        if args.mode == "replay":
            report.update(replay(args))
        else:
            report.update(reload(args))
    except Exception as e:  # noqa: BLE001 - the failure is the finding
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, indent=1, default=str), flush=True)
    os._exit(0)  # OmniGibson's atexit teardown is slow and sometimes hangs; the report is written


def pose_gap(obj, saved_pos, saved_ori):
    """(position gap in m, rotation gap in degrees) between an object's pose and a saved one."""
    import torch as th

    pos, ori = obj.get_position_orientation()
    dp = float(th.linalg.norm(pos - th.tensor(saved_pos, dtype=pos.dtype)))
    dot = abs(float(th.dot(ori, th.tensor(saved_ori, dtype=ori.dtype))))
    return dp, math.degrees(2 * math.acos(min(1.0, dot)))


def reload(args):
    """Load a saved cut scene the way a consumer does, and check what came back."""
    import omnigibson as og
    from omnigibson.robots import BaseRobot
    from simfoundry.utils.ground_plane_utils import apply_ground_plane_info
    from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

    saved = load_json_with_absolute_usd_paths(args.saved)  # a saved cut scene's paths are absolute already
    with open(args.scene) as f:
        original = json.load(f)
    reg = saved["state"]["registry"]["object_registry"]
    pieces = sorted(n for n in saved["objects_info"]["init_info"] if "_half" in n)
    out = {"saved": os.path.abspath(args.saved), "saved_has_ground_plane_info": "ground_plane_info" in saved,
           "objects_in_file": len(saved["objects_info"]["init_info"]), "objects_with_state": len(reg),
           "saved_top_level_keys": sorted(saved), "pieces_in_file": pieces,
           "fruit_in_file": args.fruit in saved["objects_info"]["init_info"]}

    if args.mode == "env":
        env = og.Environment(configs=dict(
            env=dict(action_frequency=15, rendering_frequency=15, physics_frequency=120),
            scene=dict(type="Scene", scene_file=saved, use_floor_plane=True, floor_plane_visible=True,
                       use_skybox=False, include_robots=True)))
        scene = env.scene
    else:
        og.launch()
        og.sim.restore(scene_files=[saved])
        env, scene = None, og.sim.scenes[0]
    out["ground_plane"] = apply_ground_plane_info(original, og.sim.floor_plane)
    if not og.sim.is_playing():
        og.sim.play()
    if env is not None:
        env.reset()  # as teleop does after building the environment: the scene file's state, on the moved floor

    names = {o.name for o in scene.objects}
    out["pieces_loaded"] = [p for p in pieces if p in names]
    out["fruit_loaded"] = args.fruit in names
    # The uncut props are the control: whatever moves them too is the loading path, not the cut.
    groups = {"pieces": [scene.object_registry("name", p) for p in out["pieces_loaded"]],
              "others": [o for o in scene.objects if o.name not in pieces and o.name in reg
                         and not isinstance(o, BaseRobot) and not o.fixed_base]}

    def gaps(tag):
        """Per object and largest pose gap to the saved state (mm, degrees), pieces and uncut props apart."""
        for group, objs in groups.items():
            g = {o.name: pose_gap(o, reg[o.name]["root_link"]["pos"], reg[o.name]["root_link"]["ori"]) for o in objs}
            out[f"{tag}_gap_{group}"] = {n: [round(1e3 * dp, 3), round(dr, 3)] for n, (dp, dr) in g.items()}
            out[f"{tag}_gap_max_{group}"] = ([max(1e3 * dp for dp, _ in g.values()), max(dr for _, dr in g.values())]
                                             if g else None)

    gaps("load")
    for _ in range(args.settle_steps):
        og.sim.step()
    out["settle_steps"] = args.settle_steps
    gaps("settled")
    try:
        resaved = json.loads(scene.save())
        out["resave"] = "ok"
        out["resave_has_pieces"] = all(p in resaved["objects_info"]["init_info"] for p in out["pieces_loaded"])
    except Exception as e:  # noqa: BLE001
        out["resave"] = f"{type(e).__name__}: {e}"
    try:
        if env is not None:
            env.reset()
        else:
            scene.reset()
        out["reset"] = "ok"
        out["pieces_after_reset"] = [p for p in pieces if scene.object_registry("name", p) is not None]
    except Exception as e:  # noqa: BLE001
        out["reset"] = f"{type(e).__name__}: {e}"
    return out


def replay(args):
    """Replay every episode with OmniGibson's playback, then with SimFoundry's MinimalPlaybackWrapper."""
    import h5py
    import omnigibson as og
    import torch as th
    from omnigibson.envs.data_wrapper import HDF5PlaybackWrapper
    from simfoundry.utils.annotation_utils import MinimalPlaybackWrapper

    out = {"hdf5": os.path.abspath(args.hdf5), "omnigibson_playback": {}, "minimal_playback": {}}
    with h5py.File(args.hdf5, "r") as f:
        n_episodes = int(f["data"].attrs["n_episodes"])
        recorded = {}
        for i in range(n_episodes):
            grp = f["data"][f"demo_{i}"]
            recorded[i] = (grp["state"][()], grp["state_size"][()].astype(int), json.loads(grp.attrs["transitions"]))
    out["n_episodes"] = n_episodes

    class Playback(HDF5PlaybackWrapper):
        """HDF5PlaybackWrapper whose constructor takes what create_from_hdf5 passes it.

        create_from_hdf5 hands every subclass ``robot_grasping_mode``, which only the LeRobot wrapper (stage 6's)
        accepts; HDF5PlaybackWrapper raises TypeError. It is None here, which DataPlaybackWrapper ignores anyway.
        """

        def __init__(self, *args, robot_grasping_mode=None, **kwargs):
            super().__init__(*args, **kwargs)

    tmp = tempfile.mkdtemp(prefix="cut_replay_")
    pb = Playback.create_from_hdf5(
        input_path=args.hdf5, output_path=os.path.join(tmp, "replay.hdf5"), robot_obs_modalities=["proprio"],
        n_render_iterations=1, only_successes=False, include_task_obs=False)
    if args.scene:
        # The recording's scene file is scene.save() output, which has no ground_plane_info; a SimFoundry task
        # (PickPlaceTask) would put the floor where the scene says, bench_swap.py's DummyTask does not.
        from simfoundry.utils.ground_plane_utils import apply_ground_plane_info

        with open(args.scene) as f:
            out["ground_plane"] = apply_ground_plane_info(json.load(f), og.sim.floor_plane)

    def final_piece_gap(episode):
        """Largest gap between where the pieces are now and where the last recorded state that holds them (the dump
        filter leaves sleeping objects out of a step's state) has them."""
        states, sizes, _ = recorded[episode]
        scene = og.sim.scenes[0]
        pieces = {}
        for state, size in zip(states[::-1], sizes[::-1]):
            want, _ = scene.deserialize(th.tensor(state[:size], dtype=th.float32))
            pieces = {n: s for n, s in want["registry"]["object_registry"].items() if "_half" in n}
            if pieces:
                break
        gaps = [pose_gap(scene.object_registry("name", n), s["root_link"]["pos"].tolist(),
                         s["root_link"]["ori"].tolist()) for n, s in pieces.items()]
        return (max(1e3 * g[0] for g in gaps), max(g[1] for g in gaps)) if gaps else (None, None)

    for i in range(n_episodes):
        rec = {"transition_steps": sorted(int(k) for k in recorded[i][2])}
        try:
            pb.playback_episode(i, record_data=False)
            rec["result"] = "ok"
            names = {o.name for o in og.sim.scenes[0].objects}
            rec["pieces_after"] = sorted(n for n in names if "_half" in n)
            rec["fruit_after"] = args.fruit in names
            rec["final_piece_gap_mm"], rec["final_piece_gap_deg"] = final_piece_gap(i)
        except Exception as e:  # noqa: BLE001 - the failure is the finding
            rec["result"] = f"{type(e).__name__}: {e}"
        out["omnigibson_playback"][i] = rec
        print(f"[replay] OmniGibson playback demo_{i}: {rec}", flush=True)

    with h5py.File(args.hdf5, "r") as f:
        minimal = MinimalPlaybackWrapper(pb.env, f)
        for i in range(n_episodes):
            rec = {}
            try:
                minimal.playback_episode(i)
                rec["result"] = "ok"
            except Exception as e:  # noqa: BLE001
                rec["result"] = f"{type(e).__name__}: {e}"
            out["minimal_playback"][i] = rec
            print(f"[replay] MinimalPlaybackWrapper demo_{i}: {rec}", flush=True)
    return out


if __name__ == "__main__":
    main()
