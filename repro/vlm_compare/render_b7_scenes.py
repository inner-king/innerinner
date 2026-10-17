# SPDX-License-Identifier: Apache-2.0
"""Render scene states the way B6 renders the scene image B7 is shown (s14_og/auto_generation/scene_000.png).

B6 (scripts/pipeline/B_augmentation/stages/6_sample_reconstructed_scene.py) builds its environment from
s14_og's config (floor plane and skybox, the Franka at its configured pose, the nv_franka_droid external
cameras) and photographs its first randomized layout after letting it settle (og.sim.step, each one
rendering the viewer camera at OmniGibson's default pose), one more physics step, and its own
take_scene_picture: the viewer camera jumps to s14_og.auto_camera_pos/ori and five frames are rendered.

This builds the same environment once and adds every prop mesh the given scene states use. For each state
it shows only that state's three props, at its saved poses, renders SETTLE_STEPS steps from the default
viewer pose (a pose set while the simulation plays reaches the renderer only through steps), puts the props
back exactly on the saved poses, takes B6's one physics step and calls the same take_scene_picture. The
props are read back after the photograph, so how far that one step moved them is recorded.

Each state is photographed once per --takes entry, every state in turn before the next take: by default
"first" (<name>.png) and "repeat" (<name>.repeat.png), which measures the renderer's own variation; any
other take t is <name>.t.png, so more photographs can be added later without replacing these. A state whose
name is a scene under --data-root (put_marker_in_cup_v2 -> "v2") is also compared with B6's original image.
render_log.json gathers every take made into <out_dir>.

Boots Isaac Sim (~90 s, GPU) and waits first until no other job is on the GPU.
Usage: python repro/vlm_compare/render_b7_scenes.py <scenes_dir> <out_dir> [--names v2,q35,...] [--takes first,repeat]
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

import numpy as np
import pynvml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # B6 imports simfoundry; take this checkout's
STAGE = REPO / "scripts/pipeline/B_augmentation/stages/6_sample_reconstructed_scene.py"
ORIGINALS = {"v2": "put_marker_in_cup_v2", "q35": "put_marker_in_cup_q35"}
SCENE_IMAGE = Path("s14_og/auto_generation/scene_000.png")
FAR = 20.0  # metres from the scene that a prop not in the current state is parked, hidden
SETTLE_STEPS = 20  # steps rendered from the default viewer pose before each photograph (B6 settles 100)


def _cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
    except OSError:
        return ""


def other_gpu_jobs(gpu):
    """Jobs on the GPU, counted as replay_b7_seeds.other_gpu_jobs counts them: compute processes other than
    Ollama's and an idle ComfyUI (models unloaded, under 512 MiB), and graphics clients that are Python or
    Kit (an Isaac Sim render) rather than the desktop's."""
    jobs = []
    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu):
        cmd = _cmdline(proc.pid)
        if not cmd or "ollama" in cmd or ("comfyui" in cmd.lower() and (proc.usedGpuMemory or 0) < 512 * 2**20):
            continue
        jobs.append((proc.pid, cmd[:100]))
    for proc in pynvml.nvmlDeviceGetGraphicsRunningProcesses(gpu):
        cmd = _cmdline(proc.pid)
        if "comfyui" not in cmd.lower() and ("python" in cmd or re.search(r"(^|/)kit(\s|$)", cmd) or "isaac" in cmd.lower()):
            jobs.append((proc.pid, cmd[:100]))
    return sorted(set(jobs))


def wait_for_idle_gpu():
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    t0, jobs = time.time(), other_gpu_jobs(gpu)
    if jobs:
        print(f"waiting for the GPU: {jobs}", flush=True)
    while jobs:
        time.sleep(30)
        jobs = other_gpu_jobs(gpu)
    print(f"GPU idle after {time.time() - t0:.0f} s", flush=True)


def load_stage():
    """B6's module: importing it points OmniGibson at this checkout's datasets and boots it, as B6 does."""
    spec = importlib.util.spec_from_file_location("b6_sample_reconstructed_scene", STAGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def props(state):
    """(object id, category, model, root_link) of each DatasetObject in a saved scene state."""
    reg = state["state"]["registry"]["object_registry"]
    return [(name, info["args"]["category"], info["args"]["model"], reg[name]["root_link"])
            for name, info in state["objects_info"]["init_info"].items() if info["class_name"] == "DatasetObject"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--names", default=None, help="states to render, in this order (default: every <name>.json)")
    ap.add_argument("--takes", default="first,repeat", help="photographs of each state: first, repeat, or any other name")
    ap.add_argument("--data-root", default=str(REPO / "Data"))
    args = ap.parse_args()
    takes = args.takes.split(",")
    # Resolved now: importing B6 changes directory.
    scenes_dir, out, data_root = (Path(p).resolve() for p in (args.scenes_dir, args.out_dir, args.data_root))
    out.mkdir(parents=True, exist_ok=True)
    names = args.names.split(",") if args.names else sorted(p.stem for p in scenes_dir.glob("*.json")
                                                           if p.stem != "registration")
    states = {n: json.loads((scenes_dir / f"{n}.json").read_text()) for n in names}

    wait_for_idle_gpu()
    b6 = load_stage()  # chdirs into the config directory, as B6 does
    og, th, OmegaConf = b6.og, b6.th, b6.OmegaConf
    cfg = OmegaConf.load(REPO / "scripts/cfg/real2sim_cfg.yaml").s14_og
    cam_pos, cam_ori = th.tensor(list(cfg.auto_camera_pos)), th.tensor(list(cfg.auto_camera_ori))

    # B6's environment (its main(), include_gs=false, include_table=false, include_robot=true).
    robot_cfg = OmegaConf.to_container(cfg.robot_config, resolve=True)
    robot_cfg["controller_config"] = {
        "arm_0": {"name": "InverseKinematicsController", "command_input_limits": None, "mode": "pose_delta_ori",
                  "smoothing_filter_size": 5},
        "gripper_0": {"name": "MultiFingerGripperController", "command_input_limits": (0.0, 1.0), "mode": "smooth"},
    }
    sensors = b6.parse_config(f"{b6.SIMFOUNDRY_CFG_DIR}/external_sensors/{cfg.external_sensors_cfg}.yaml")["external_sensors"]
    env = og.Environment(configs={
        "env": {"external_sensors": sensors},
        "scene": {"type": "Scene", "use_floor_plane": True, "floor_plane_visible": True, "use_skybox": True},
        "robots": [robot_cfg]})
    env.reset()
    og.sim.play()
    for _ in range(10):
        og.sim.step()

    meshes = sorted({(cat, model) for n in names for _, cat, model, _ in props(states[n])})
    objs = {}
    with og.sim.stopped():
        for i, (cat, model) in enumerate(meshes):
            obj = b6.DatasetObject(name=f"{cat}__{model}", dataset_name="real2sim-assets", category=cat, model=model)
            env.scene.add_object(obj)
            obj.set_position_orientation(position=th.tensor([FAR + i, FAR, 0.1]), orientation=th.tensor([0.0, 0, 0, 1]))
            objs[(cat, model)] = obj

    default_view = og.sim.viewer_camera.get_position_orientation()  # where B6's settling was rendered from
    robot = env.robots[0]
    og.sim.play()
    robot.reset()
    robot.keep_still()
    robot.set_position_orientation(position=th.tensor(list(cfg.robot_config.position)),
                                   orientation=th.tensor(list(cfg.robot_config.orientation)))
    for _ in range(20):
        og.sim.step()

    def show(name):
        state = states[name]
        used = {(cat, model): root for _, cat, model, root in props(state)}
        for i, (key, obj) in enumerate(objs.items()):
            if key in used:
                obj.set_position_orientation(position=th.tensor(used[key]["pos"]), orientation=th.tensor(used[key]["ori"]))
            else:
                obj.set_position_orientation(position=th.tensor([FAR + i, FAR, 0.1]), orientation=th.tensor([0.0, 0, 0, 1]))
            obj.visible = key in used
            obj.keep_still()
        rob = state["state"]["registry"]["object_registry"][robot.name]
        robot.set_position_orientation(position=th.tensor(rob["root_link"]["pos"]), orientation=th.tensor(rob["root_link"]["ori"]))
        robot.set_joint_positions(th.tensor(rob["joint_pos"]))
        robot.keep_still()
        return used

    def readback(used):
        """How far each shown prop, and the robot's joints, are from the state when photographed."""
        worst = {}
        for key, root in used.items():
            pos, ori = objs[key].get_position_orientation()
            dq = min(np.abs(ori.numpy() - np.array(root["ori"])).max(), np.abs(ori.numpy() + np.array(root["ori"])).max())
            worst["/".join(key)] = {"pos_mm": round(float(np.abs(pos.numpy() - np.array(root["pos"])).max()) * 1000, 3),
                                    "quat": round(float(dq), 6)}
        return worst

    def photograph(name, path):
        show(name)
        og.sim.viewer_camera.set_position_orientation(*default_view)
        for _ in range(SETTLE_STEPS):
            og.sim.step()
        used = show(name)  # back on the saved poses, which the steps above may have nudged
        og.sim.step_physics()  # B6's last step before its photograph
        for obj in objs.values():
            obj.keep_still()
        b6.take_scene_picture(str(path), camera_pos=cam_pos, camera_ori=cam_ori)
        return used

    # The first photographs of a session can come back dark or blank (turntable.py); throw one away.
    photograph(names[0], out / "_warmup.png")
    log_path = out / "render_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"renders": {}}
    log.update({"camera_pos": list(cfg.auto_camera_pos), "camera_ori": list(cfg.auto_camera_ori),
                "settle_steps": SETTLE_STEPS})
    image = lambda name, tag: out / (f"{name}.png" if tag == "first" else f"{name}.{tag}.png")  # noqa: E731
    for tag in takes:
        for name in names:
            path = image(name, tag)
            used = photograph(name, path)
            log["renders"].setdefault(name, {})[tag] = {"image": path.name, "pose_error": readback(used),
                                                        "robot_joint_error": float((robot.get_joint_positions() - th.tensor(
                                                            states[name]["state"]["registry"]["object_registry"][robot.name]["joint_pos"])).abs().max())}
            print(name, tag, json.dumps(log["renders"][name][tag]), flush=True)

    def diff(a, b):
        x = np.asarray(Image.open(a).convert("RGB"), np.int16)
        y = np.asarray(Image.open(b).convert("RGB"), np.int16)
        d = np.abs(x - y)
        return {"mean_abs": round(float(d.mean()), 3), "max_abs": int(d.max()),
                "pixels_differing_by_over_8": int((d.max(-1) > 8).sum())}

    for name in names:
        r = log["renders"][name]
        if image(name, "first").exists() and image(name, "repeat").exists():
            r["first_vs_repeat"] = diff(image(name, "first"), image(name, "repeat"))
        if name in ORIGINALS:
            for tag in takes:
                r[f"{tag}_vs_b6_original"] = diff(image(name, tag), data_root / ORIGINALS[name] / SCENE_IMAGE)
        print(name, {k: v for k, v in r.items() if re.search("_vs_", k)}, flush=True)
    log_path.write_text(json.dumps(log, indent=1) + "\n")
    og.shutdown()


if __name__ == "__main__":
    main()
