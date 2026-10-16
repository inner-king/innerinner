# SPDX-License-Identifier: Apache-2.0
"""Render an auto_bg scene (Gaussian-splat background) from the views that decide whether it is usable.

The splat is trained on one handheld sweep, so it can match the video from the video's own camera and
fall apart from anywhere else (repro/README.md, 17.4). This loads the step-7 scene state with the robot
moved in front of the objects, as repro/scripted_pick_place.py does, and renders one still per camera:
  real       the video frame stage 4 anchored the world to, at its own lens
  droid_*    the two external cameras of the DROID layout (nv_franka_droid_high_res)
  teleop_*   the two external cameras of the teleop layout (sim_franka_teleop), the demo videos' views
into one labelled grid. No physics is stepped; the objects stay where the reconstruction put them.

Usage (simfoundry env):
  python repro/g4/render_bg_views.py --scene <scene> --scene-file <abs path to ..._scene_state_auto_bg.json> \
      --pick iter_N --place iter_M --out <abs path to views.jpg> [--label TEXT]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Importing the demo boots OmniGibson the way the demo does (data path, headless, dependencies).
from scripted_pick_place import CFG_DIR, DATA, load_json_with_absolute_usd_paths, np_, real_camera  # noqa: E402

import omnigibson as og  # noqa: E402
import omnigibson.lazy as lazy  # noqa: E402
from omnigibson.utils.config_utils import parse_config  # noqa: E402

LAYOUTS = (("droid", "nv_franka_droid_high_res"), ("teleop", "sim_franka_teleop"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scene-file", required=True)
    ap.add_argument("--pick", required=True)
    ap.add_argument("--place", required=True)
    ap.add_argument("--standoff", type=float, default=0.55)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    os.chdir(CFG_DIR)
    scene_json = load_json_with_absolute_usd_paths(args.scene_file)
    cams, labels = [], {}
    for prefix, layout in LAYOUTS:
        for c in parse_config(str(Path(CFG_DIR) / "external_sensors" / f"{layout}.yaml"))["external_sensors"]:
            c = dict(c)
            labels[f"{prefix}_{c['name']}"] = f"{prefix}: {c['name']}"
            c["relative_prim_path"] = c["relative_prim_path"].rsplit("/", 1)[0] + f"/{prefix}_{c['name']}"
            c["name"] = f"{prefix}_{c['name']}"
            c["modalities"] = ["rgb"]
            # 640x360: five RTX cameras at 1280x720 run a 12 GB GPU out of memory.
            c["sensor_kwargs"] = {**c.get("sensor_kwargs", {}), "image_height": 360, "image_width": 640}
            cams.append(c)
    real = real_camera(args.scene, width=640)
    cams.append({"sensor_type": "VisionSensor", "name": "real_view_cam", "modalities": ["rgb"],
                 "sensor_kwargs": {"image_height": real["height"], "image_width": real["width"],
                                   "clipping_range": [0.01, 100.0]},
                 "position": real["position"], "orientation": real["orientation"]})
    labels["real_view_cam"] = f"real: video frame {real['frame_idx']}, hfov {real['hfov_deg']:.0f} deg"

    # Robot in front of the objects, as the demo places it (the scene-editing step between B and C).
    reg = scene_json["state"]["registry"]["object_registry"]
    mid = (np.array(reg[args.pick]["root_link"]["pos"]) + np.array(reg[args.place]["root_link"]["pos"])) / 2
    base = reg["robot0"]["root_link"]
    base["pos"] = [float(mid[0] - args.standoff), float(mid[1]), base["pos"][2]]
    base["ori"] = [0.0, 0.0, 0.0, 1.0]
    base["lin_vel"] = base["ang_vel"] = [0.0, 0.0, 0.0]

    env = og.Environment(configs={
        "env": {"external_sensors": cams, "action_frequency": 15, "rendering_frequency": 15, "physics_frequency": 120},
        "scene": {"type": "Scene", "scene_file": scene_json, "use_floor_plane": True, "floor_plane_visible": False,
                  "use_skybox": True, "include_robots": True},
        "task": {"type": "DummyTask"},
    })
    env.reset()
    # Same tonemapping as the demo for a splat background (repro/README.md, 13.3).
    lazy.carb.settings.get_settings().set("/rtx/post/tonemap/op", 1)
    cam = env.external_sensors["real_view_cam"]
    cam.horizontal_aperture = real["aperture"]
    cam.focal_length = real["focal"]
    for _ in range(30):
        og.sim.render()

    tiles = []
    for name in ["real_view_cam"] + [c["name"] for c in cams if c["name"] != "real_view_cam"]:
        rgb = np_(env.external_sensors[name].get_obs()[0]["rgb"])[:, :, :3].astype(np.uint8)
        im = Image.fromarray(rgb).resize((640, 360))
        ImageDraw.Draw(im).rectangle([0, 0, 640, 22], fill=(0, 0, 0))
        ImageDraw.Draw(im).text((6, 5), f"{args.label}  {labels[name]}".strip(), fill=(255, 255, 0))
        tiles.append(im)
    cols, rows = 2, (len(tiles) + 1) // 2
    sheet = Image.new("RGB", (640 * cols, 360 * rows))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * 640, (k // cols) * 360))
    sheet.save(args.out, quality=90)
    print(f"saved {args.out}", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
