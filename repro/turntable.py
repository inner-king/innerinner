# SPDX-License-Identifier: Apache-2.0
"""360-degree turntable renders of each reconstructed object, in the same OmniGibson/Isaac renderer.

Loads the reconstructed scene without the robot, shows one object at a time, and orbits a camera around
it (the objects stay put; no physics is stepped). Writes one MP4 with every object in sequence plus a
JSON sidecar giving each object's frame range, category and size, for compose_real_vs_sim.py.

Usage: python repro/turntable.py <scene> <out.mp4> [--frames-per-object 96] [--elevation 20]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

import imageio
import numpy as np
import torch as th

REPO = Path(__file__).resolve().parents[1]  # repro/<this file> -> repository root
sys.path.insert(0, str(REPO))
from simfoundry import CFG_DIR, configure_omnigibson_data_path, import_og_dependencies  # noqa: E402
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths  # noqa: E402

configure_omnigibson_data_path(force=True)
import_og_dependencies()

import omnigibson as og  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402

DATA = REPO / "Data"
APERTURE = 20.955


def np_(x):
    return x.detach().cpu().numpy().astype(np.float64) if isinstance(x, th.Tensor) else np.asarray(x, np.float64)


def look_at(eye, target):
    """xyzw quaternion for a USD camera (looks down -z, +y up) at @eye facing @target."""
    f = target - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    R = np.stack([right, up, -f], axis=1)
    return np_(T.mat2quat(th.tensor(R, dtype=th.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("out")
    ap.add_argument("--frames-per-object", type=int, default=96)
    ap.add_argument("--elevation", type=float, default=30.0)
    ap.add_argument("--hfov", type=float, default=50.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fill", type=float, default=0.85, help="fraction of the frame height the bounding sphere spans")
    args = ap.parse_args()

    os.chdir(CFG_DIR)
    scene_json = load_json_with_absolute_usd_paths(str(DATA / args.scene / "s14_og" / "reconstructed_og_scene.json"))
    cam_cfg = [{"sensor_type": "VisionSensor", "name": "turntable_cam", "modalities": ["rgb"],
                "sensor_kwargs": {"image_height": args.height, "image_width": args.width,
                                  "clipping_range": [0.005, 50.0]},
                "position": [0.0, -1.0, 0.5], "orientation": [0.0, 0.0, 0.0, 1.0]}]
    env = og.Environment(configs={
        "env": {"external_sensors": cam_cfg, "action_frequency": 15, "rendering_frequency": 15, "physics_frequency": 120},
        "scene": {"type": "Scene", "scene_file": scene_json, "use_floor_plane": True, "floor_plane_visible": True,
                  "use_skybox": True, "include_robots": False},
    })
    env.reset()
    cam = env.external_sensors["turntable_cam"]
    cam.horizontal_aperture = APERTURE
    cam.focal_length = APERTURE / 2 / np.tan(np.radians(args.hfov) / 2)

    info = json.load(open(DATA / args.scene / "s14_og" / "reconstructed_og_scene.json"))["objects_info"]["init_info"]
    objects = sorted((o for o in env.scene.objects if o.name.startswith("iter_")), key=lambda o: o.name)
    frames, index = [], []
    for _ in range(10):                 # the first renders after reset come back empty
        og.sim.render()
    for obj in objects:
        for other in objects:
            other.visible = other is obj
        lo, hi = (np_(v) for v in obj.aabb)
        center = (lo + hi) / 2
        size = hi - lo
        # Fit the bounding sphere inside the narrower (vertical) field of view: framing by width alone
        # cropped the top and bottom of a tall cup in a 16:9 frame.
        vfov = 2 * np.arctan(np.tan(np.radians(args.hfov) / 2) * args.height / args.width)
        dist = (np.linalg.norm(size) / 2) / np.tan(vfov / 2) / args.fill
        el = np.radians(args.elevation)
        start = len(frames)
        for k in range(args.frames_per_object):
            az = 2 * np.pi * k / args.frames_per_object - np.pi / 2
            eye = center + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            cam.set_position_orientation(position=th.tensor(eye, dtype=th.float32),
                                         orientation=th.tensor(look_at(eye, center), dtype=th.float32))
            # Annotator output trails the render by a frame or two, and the very first pose needs longer.
            for _ in range(15 if not frames else 3):
                og.sim.render()
            rgb = np_(cam.get_obs()[0]["rgb"])[:, :, :3].astype(np.uint8)
            frames.append(rgb)
        meta_bbox = obj.native_bbox * obj.scale
        index.append({"name": obj.name, "category": info[obj.name]["args"]["category"],
                      "size_cm": [round(float(v) * 100, 1) for v in sorted(np_(meta_bbox), reverse=True)],
                      "start": start, "end": len(frames)})
        print(f"{obj.name} {index[-1]['category']} {index[-1]['size_cm']} cm: frames {start}-{len(frames)}", flush=True)

    imageio.mimwrite(args.out, frames, fps=15, macro_block_size=2, quality=8)
    json.dump(index, open(Path(args.out).with_suffix(".json"), "w"), indent=1)
    print(f"saved {args.out} ({len(frames)} frames)", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
