# SPDX-License-Identifier: Apache-2.0
"""Product-shot renders and geometry of YCB objects, for the stage 11 mass benchmark (repro/README.md, chapter 29).

Stage 11 asks its VLM for a mass from an object's photo (stage 6's studio shot on white), its oriented bounding
box and a volume. This gives a YCB object the same inputs: a 1024x1024 render of the textured google_16k mesh on
white, three-quarter view, and the mesh's extents, volume and surface area. YCB publishes each object's measured
mass (Calli et al., arXiv:1502.03143, Table II), so the stage's estimates can be scored against real masses.

    PYOPENGL_PLATFORM=egl python -I repro/mass_friction/render_ycb.py <ycb_dir> <out_dir> 037_scissors 025_mug ...

<ycb_dir>/<name>/x/<name>/google_16k/textured.obj is where ycb_fetch.sh unpacks each object.
"""
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import pyrender
import trimesh
from PIL import Image


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera-to-world pose for a camera at eye looking at target (OpenGL: the camera looks down -z)."""
    eye, target, up = map(np.asarray, (eye, target, up))
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = s, u, -f, eye
    return pose


def render(mesh, size=1024, elev_deg=35.0, azim_deg=-60.0, fov_deg=30.0):
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.45, 0.45, 0.45])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    center = mesh.bounding_box.centroid
    radius = np.linalg.norm(mesh.extents) / 2
    dist = radius / np.sin(np.radians(fov_deg) / 2) * 1.05
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    eye = center + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    pose = look_at(eye, center)
    scene.add(pyrender.PerspectiveCamera(yfov=np.radians(fov_deg)), pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=2.0), pose=look_at(center + np.array([0, 0, dist]), center, up=(0, 1, 0)))
    r = pyrender.OffscreenRenderer(size, size)
    color, _ = r.render(scene)
    r.delete()
    return color


def main():
    ycb_dir, out_dir, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        obj = os.path.join(ycb_dir, name, "x", name, "google_16k", "textured.obj")
        mesh = trimesh.load(obj, force="mesh", process=True)
        Image.fromarray(render(mesh)).save(os.path.join(out_dir, f"{name}.png"))
        # Geometry as stage 11 measures it: oriented extents, and the volume and area of the mesh
        # (google_16k meshes are watertight scans in metres).
        _, extent = trimesh.bounds.oriented_bounds(mesh)
        geo = {"extent_m": [float(x) for x in extent], "is_volume": bool(mesh.is_volume),
               "volume_m3": float(mesh.volume) if mesh.is_volume else None, "area_m2": float(mesh.area),
               "hull_volume_m3": float(mesh.convex_hull.volume)}
        json.dump(geo, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1)
        print(name, [round(100 * x, 1) for x in extent], "cm", "watertight" if geo["is_volume"] else "open", flush=True)


if __name__ == "__main__":
    main()
