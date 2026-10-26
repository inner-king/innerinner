# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 14 attaches stage 5c's background mesh to the OmniGibson scene as `mesh_background_0`.

Stage 5c writes `background_mesh.glb` in glTF's Y-up convention. Stage 14 turns it into a USD in the
stage-4 world (Z-up, metres) and loads that as a visual-only object, so the scene JSON carries the same
kind of entry the light editor writes when a room is attached by hand (`light_editor/background_io.py`).
Nothing is attached by hand: the entry comes out of the pipeline.

The object has no collision. The stage-14 floor plane already holds the objects, and a collision mesh
would be a second floor at whatever height the reconstruction put it.
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from simfoundry.pipeline.depth_mesh import WORLD_TO_GLTF
from simfoundry.utils.python_utils import atomic_output_path

BACKGROUND_NAME = "mesh_background_0"
BACKGROUND_CATEGORY = "mesh_background"
# The USD prim path of the background inside its own file.
USD_ROOT = "/mesh_background"

#: Stage-4 world -> glTF is WORLD_TO_GLTF; the editor's import undoes it, so this is the rotation the USD needs.
GLTF_TO_WORLD = np.linalg.inv(WORLD_TO_GLTF)


def make_scene_cfg(include_gs: bool, bg_mesh_attached: bool) -> dict:
    """The scene config stage 14 builds the OmniGibson scene from.

    With neither background on, this is the config stage 14 always used. A background mesh that
    is attached hides the white floor plane, which would otherwise show through it at z = 0 and
    fight it where the sink and the floor meet. The plane stays a physics support.

    Args:
        include_gs: Whether the 3DGS background is in the scene.
        bg_mesh_attached: Whether the stage-5c mesh is actually attached (after the GLB and the
            include_gs checks), not merely requested.

    Returns:
        dict: The `scene_cfg` for `env`'s scene.
    """
    return {
        "type": "Scene",
        "use_floor_plane": True,
        "floor_plane_visible": not (include_gs or bg_mesh_attached),
        "use_skybox": not include_gs,
    }


def _load_world_mesh(glb_path):
    """The stage-5c GLB as a trimesh in the stage-4 world, textured if the GLB is."""
    import trimesh

    scene = trimesh.load(str(glb_path), force="scene")
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    mesh.apply_transform(GLTF_TO_WORLD)
    return mesh


def write_background_usd(glb_path, usd_path) -> dict:
    """Write the background GLB as a USD in the stage-4 world, with its texture beside it.

    Args:
        glb_path: The stage-5c `background_mesh.glb` (Y-up, as glTF requires).
        usd_path: Where to write the `.usd`. Its texture goes in a `textures/` folder next to it.

    Returns:
        dict: `usd_path` (absolute), `texture` (the texture's path relative to the USD, or None),
        `vertices`, `faces`, and `bounds` ([min, max] corners in metres, world frame).
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    usd_path = Path(usd_path)
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = _load_world_mesh(glb_path)
    points = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    # trimesh flips V on load, so `uv` is already in the USD convention (origin bottom-left).
    uv = getattr(mesh.visual, "uv", None)
    material = getattr(mesh.visual, "material", None)
    texture_image = getattr(material, "baseColorTexture", None)
    if texture_image is None:
        texture_image = getattr(material, "image", None)

    texture_rel = None
    if texture_image is not None:
        texture_dir = usd_path.parent / "textures"
        texture_dir.mkdir(parents=True, exist_ok=True)
        texture_file = texture_dir / f"{usd_path.stem}_basecolor.png"
        texture_image.convert("RGB").save(texture_file, format="PNG")
        texture_rel = f"./textures/{texture_file.name}"

    diffuse = (0.8, 0.8, 0.8)
    factor = getattr(material, "baseColorFactor", None)
    if texture_rel is None and factor is not None:
        diffuse = tuple(float(c) / 255.0 for c in np.asarray(factor)[:3])

    with atomic_output_path(usd_path) as tmp:
        stage = Usd.Stage.CreateNew(str(tmp))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, USD_ROOT)
        stage.SetDefaultPrim(root.GetPrim())

        mesh_prim = UsdGeom.Mesh.Define(stage, f"{USD_ROOT}/mesh")
        mesh_prim.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
        mesh_prim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).tolist()))
        mesh_prim.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, n)) for n in normals]))
        mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        # Triangles stay triangles: a subdivision scheme would round off the plane's edges.
        mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh_prim.CreateDoubleSidedAttr().Set(True)

        if uv is not None:
            primvar = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            primvar.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, t)) for t in np.asarray(uv, dtype=np.float64)]))

        material_prim = UsdShade.Material.Define(stage, f"{USD_ROOT}/material")
        surface = UsdShade.Shader.Define(stage, f"{USD_ROOT}/material/surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        # A building surface: matte, not metal.
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material_prim.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        if texture_rel is not None:
            reader = UsdShade.Shader.Define(stage, f"{USD_ROOT}/material/st")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            texture = UsdShade.Shader.Define(stage, f"{USD_ROOT}/material/basecolor")
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_rel))
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                texture.ConnectableAPI(), "rgb")
        else:
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(material_prim)

        stage.GetRootLayer().Save()

    return {
        "usd_path": str(usd_path.resolve()),
        "texture": texture_rel,
        "vertices": int(len(points)),
        "faces": int(len(faces)),
        "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
    }


def relativize_background_path(scene_json_path, name=BACKGROUND_NAME) -> bool:
    """Write the background's `usd_path` in a saved scene JSON relative to that JSON.

    OmniGibson's restore resolves a relative path against the process's working directory, so the
    absolute path written while building the scene would not survive a move, and the relative one
    is what makes a scene directory portable. Everything else in the file is written back exactly
    as it was loaded, with the same layout OmniGibson saves it in.

    Args:
        scene_json_path: The scene JSON that `og.sim.save` wrote.
        name: The object's name in `objects_info.init_info`.

    Returns:
        bool: True if the entry exists (already relative counts), False if there is no such entry,
        in which case nothing is written.
    """
    scene_json_path = Path(scene_json_path)
    scene = json.loads(scene_json_path.read_text())
    entry = scene.get("objects_info", {}).get("init_info", {}).get(name)
    if entry is None:
        return False
    args = entry.get("args", {})
    usd = args.get("usd_path")
    if usd is None:
        return False
    if not os.path.isabs(usd):
        return True
    args["usd_path"] = os.path.relpath(usd, scene_json_path.parent)
    with atomic_output_path(scene_json_path) as tmp:
        Path(tmp).write_text(json.dumps(scene, indent=4))
    return True


def add_background_object(env, usd_path, name=BACKGROUND_NAME):
    """Add the background USD to a running OmniGibson environment as a fixed, visual-only object.

    Visual-only because it is not a collider: the floor plane is what the objects rest on. Its pose is
    the identity, since the mesh is already in the world frame.

    Args:
        env: The OmniGibson environment whose scene gets the object.
        usd_path: The `.usd` from `write_background_usd`.
        name: The object's name in the scene.

    Returns:
        The added object.
    """
    import omnigibson as og
    from omnigibson.objects import USDObject

    obj = USDObject(
        name=name,
        usd_path=str(Path(usd_path).resolve()),
        category=BACKGROUND_CATEGORY,
        fixed_base=True,
        visual_only=True,
    )
    env.scene.add_object(obj)
    og.sim.step()
    return obj
