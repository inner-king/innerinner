# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Does Hunyuan.generate_texture give each object its own metallic-roughness map?

Runs the real Hunyuan.generate_texture three times in one process, on three objects that were given
different base colours, metallic values and roughness values. Only the paint pipeline is a stand-in.
The real one needs the Hunyuan paint models and a GPU. The stand-in writes each object's textures under
the same file names the dependency's save_obj_mesh uses (textured_mesh.jpg, _metallic.jpg, _roughness.jpg),
then calls the dependency's real convert_obj_to_glb, which runs Blender. The stale maps come from
Blender's OBJ import and glTF export, which keep and reuse the images they loaded for the previous
object, so that part has to be real. The stand-in ignores the shape path it is given.

The check reads the metallicRoughness texture back out of each GLB and compares it with what that object
was given. Hunyuan3D-2.1 is only read, never written.

Usage (hunyuan env, no GPU needed; the work dir should be empty):

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree> python -B repro/hunyuan_stale_maps_check.py <work dir>

Where deps/ is not inside this checkout, pass the dependency explicitly, for example
`--hy3dpaint /path/to/Hunyuan3D-2.1/hy3dpaint`. Without the flag the script uses deps/ at the repository root.

Exit status 1 when an object came back with another object's map, 0 when every object has its own.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HY3DPAINT = REPO_ROOT / "deps" / "Hunyuan3D-2.1" / "hy3dpaint"

OBJ = "mtllib textured_mesh.mtl\nusemtl Material\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n"
MTL = ("newmtl Material\nKd 0.8 0.8 0.8\nd 1.0\nillum 2\nmap_Kd textured_mesh.jpg\n"
       "map_Pm textured_mesh_metallic.jpg\nmap_Pr textured_mesh_roughness.jpg\n")
# (base colour, metallic, roughness) given to object 0, 1, 2
OBJECTS = [((255, 0, 0), 255, 30), ((0, 255, 0), 0, 200), ((0, 0, 255), 128, 100)]


class StandInPaintPipeline:
    """The part of Hunyuan3DPaintPipeline that matters here.

    Every call writes the object's textures under the same names the real pipeline uses, then runs the
    dependency's convert_obj_to_glb on them, the same export the real pipeline ends with.
    """

    def __init__(self, paint_dir, convert_obj_to_glb):
        self.paint_dir = paint_dir
        self.convert_obj_to_glb = convert_obj_to_glb
        self.calls = 0

    def __call__(self, shape_fpath, image_path=None, use_remesh=True):
        base, metallic, roughness = OBJECTS[self.calls]
        self.calls += 1
        obj_path = self.paint_dir / "textured_mesh.obj"
        obj_path.write_text(OBJ)
        (self.paint_dir / "textured_mesh.mtl").write_text(MTL)
        Image.new("RGB", (64, 64), base).save(self.paint_dir / "textured_mesh.jpg", quality=100)
        Image.new("L", (64, 64), metallic).save(self.paint_dir / "textured_mesh_metallic.jpg", quality=100)
        Image.new("L", (64, 64), roughness).save(self.paint_dir / "textured_mesh_roughness.jpg", quality=100)
        assert self.convert_obj_to_glb(str(obj_path), str(self.paint_dir / "textured_mesh.glb"))
        return str(obj_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("work_dir", type=Path, help="folder for the three GLBs and the stand-in's paint files")
    parser.add_argument("--hy3dpaint", type=Path, default=DEFAULT_HY3DPAINT,
                        help="Hunyuan3D-2.1 hy3dpaint folder whose convert_obj_to_glb is run, read only "
                             "(default: %(default)s)")
    args = parser.parse_args(argv)
    if not (args.hy3dpaint / "DifferentiableRenderer" / "mesh_utils.py").is_file():
        parser.error(f"no Hunyuan3D-2.1 hy3dpaint folder at {args.hy3dpaint}")

    sys.path.insert(0, str(args.hy3dpaint))
    from DifferentiableRenderer.mesh_utils import convert_obj_to_glb  # the dependency's real Blender export

    import simfoundry
    from simfoundry.models.mesh_generator import Hunyuan

    paint_dir = args.work_dir / "paint"
    paint_dir.mkdir(parents=True, exist_ok=True)
    generator = Hunyuan.__new__(Hunyuan)  # skip __init__: it would load the models
    generator.texture_pipeline = StandInPaintPipeline(paint_dir, convert_obj_to_glb)
    generator.low_vram = False

    print("simfoundry from:", simfoundry.__file__)
    first_md5 = None
    bad = 0
    for i, (base, metallic, roughness) in enumerate(OBJECTS):
        out = args.work_dir / f"object_{i}.glb"
        generator.generate_texture(str(paint_dir / "shape.glb"), str(out))
        material = trimesh.load(out, force="mesh", process=False).visual.material
        mr = np.asarray(material.metallicRoughnessTexture.convert("RGB"))
        got_rough, got_metal = float(mr[..., 1].mean()), float(mr[..., 2].mean())
        md5 = hashlib.md5(mr.tobytes()).hexdigest()[:8]
        first_md5 = first_md5 or md5
        ok = abs(got_rough - roughness) <= 4 and abs(got_metal - metallic) <= 4
        if not ok:
            bad += 1
        same = " (same md5 as object 0)" if i and md5 == first_md5 else ""
        print(f"object {i}: gave metallic {metallic:3d} roughness {roughness:3d} | GLB MR map has metallic "
              f"{got_metal:5.1f} roughness {got_rough:5.1f} md5 {md5}{same} -> {'ok' if ok else 'STALE'}")
    print("RESULT:", "every object has its own map" if not bad
          else f"{bad} object(s) came back with another object's map")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
