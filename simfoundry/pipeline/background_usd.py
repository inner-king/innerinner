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
import math
import numbers
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from simfoundry.pipeline.depth_mesh import WORLD_TO_GLTF
from simfoundry.utils.python_utils import atomic_output_path

BACKGROUND_NAME = "mesh_background_0"
BACKGROUND_CATEGORY = "mesh_background"
# The USD prim path of the background inside its own file.
USD_ROOT = "/mesh_background"
# OmniGibson wants exactly one rigid-body root link per object, fixed and visual-only or not, so the mesh hangs off a body.
USD_BASE_LINK = f"{USD_ROOT}/base_link"

#: Stage-4 world -> glTF is WORLD_TO_GLTF; the editor's import undoes it, so this is the rotation the USD needs.
GLTF_TO_WORLD = np.linalg.inv(WORLD_TO_GLTF)

#: The shading options stage 14 accepts. Anything else is refused before Isaac Sim starts.
SHADINGS = ("lit", "unlit")
#: The unlit background's emissive intensity, provisional. It is the value that brings the unlit background's gain
#: to the photo to 1 under the engine's default exposure (filmIso 100, fNumber 5, 1/50 s) and tone mapper (ACES):
#: 2776 (marker), 2700 (bowl), 2700 (LiDAR). A first guess of 2700 gave mean gains within 1 +- 0.02 on three scenes.
#: It is bound to that camera exposure and tone mapper, so another camera or tone mapper needs its own value.
UNLIT_DEFAULT_EMISSIVE_INTENSITY = 2700.0


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


def _positive_number(value, name):
    """A finite number above zero, as a float. Bools and strings are refused rather than coerced.

    Args:
        value: The value to check.
        name: The option's name, for the message.

    Returns:
        float: `value` as a float.

    Raises:
        ValueError: If `value` is not a real number above zero.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number above zero, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite number above zero, got {value!r}")
    return number


def _exposure_scale_rgb(value):
    """`exposure_scale` as three channel multipliers, or None when there is no multiplier.

    Args:
        value: None, one positive number (all three channels), or a sequence of three (R, G, B).

    Returns:
        tuple or None: Three floats, or None.

    Raises:
        ValueError: If `value` has any other form.
    """
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != 3:
            raise ValueError(f"exposure_scale as a sequence needs exactly three numbers (R, G, B), got {value!r}")
        return tuple(_positive_number(channel, "exposure_scale channel") for channel in value)
    number = _positive_number(value, "exposure_scale")
    return (number, number, number)


def resolve_background_exposure(shading, exposure_scale, emissive_intensity):
    """Check the background's shading options and fill in what they mean.

    `exposure_scale` is a multiplier on the texture and belongs to `lit`; `emissive_intensity` is the emission
    of the unlit material and belongs to `unlit`. Each is refused under the other shading, so a setting that
    would do nothing fails here instead of silently.

    Args:
        shading: `"lit"` or `"unlit"`, spelled exactly.
        exposure_scale: None, one positive number, or three positive numbers (R, G, B). `lit` only.
        emissive_intensity: None or one positive number. `unlit` only.

    Returns:
        tuple: `(shading, scale_rgb, intensity)`. `scale_rgb` is three floats for `lit` with a multiplier and
        None otherwise; `intensity` is a float for `unlit` (the default when none is given) and None otherwise.

    Raises:
        ValueError: If a value is not one of the forms above, or is set under the wrong shading.
    """
    if not isinstance(shading, str) or shading not in SHADINGS:
        raise ValueError(f"shading must be one of {SHADINGS}, got {shading!r}")
    if shading == "lit":
        if emissive_intensity is not None:
            raise ValueError("emissive_intensity applies to unlit shading only")
        return shading, _exposure_scale_rgb(exposure_scale), None
    if exposure_scale is not None:
        raise ValueError("exposure_scale applies to lit shading only")
    if emissive_intensity is None:
        return shading, None, UNLIT_DEFAULT_EMISSIVE_INTENSITY
    return shading, None, _positive_number(emissive_intensity, "emissive_intensity")


def _author_lit_material(stage, material_prim, texture_rel, diffuse, scale_rgb):
    """The lit background's UsdPreviewSurface: the texture on diffuseColor, times `scale_rgb` if given."""
    from pxr import Gf, Sdf, UsdShade

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
        if scale_rgb is not None:
            # Alpha stays 1, so the multiplier does not touch the texture's own alpha.
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale_rgb, 1.0))
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb")
    else:
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))


def _author_unlit_material(stage, material_prim, texture_rel, intensity):
    """The unlit background: one OmniPBR (MDL) shader whose emission is the texture.

    UsdPreviewSurface cannot give this. With diffuseColor at zero it still took about 45% of its brightness from
    the skybox at the default skybox strength, while the MDL emission does not depend on the skybox.
    """
    from pxr import Gf, Sdf, UsdShade

    mdl = UsdShade.Shader.Define(stage, f"{USD_ROOT}/material/omnipbr")
    # OmniPBR is found through its MDL source, not an info:id, so there is no CreateIdAttr here.
    mdl.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    # No diffuse: the texture is emitted, not lit. Diffuse is what would make it depend on the scene light.
    mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    mdl.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
    # White, so the emission does not tint the texture (OmniPBR's own default is reddish).
    mdl.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
    mdl.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_rel))
    mdl.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(intensity)
    mdl.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.0)
    mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
    mdl.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    material_prim.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")


def write_background_usd(glb_path, usd_path, *, shading="lit", exposure_scale=None, emissive_intensity=None) -> dict:
    """Write the background GLB as a USD in the stage-4 world, with its texture beside it.

    The defaults write the material this module always wrote. `shading="unlit"` writes one OmniPBR (MDL)
    material whose emission is the texture instead, and `exposure_scale` (lit only) multiplies the texture.

    Args:
        glb_path: The stage-5c `background_mesh.glb` (Y-up, as glTF requires).
        usd_path: Where to write the `.usd`. Its texture goes in a `textures/` folder next to it.
        shading: `"lit"` (the default) or `"unlit"`.
        exposure_scale: For `lit`: None, one positive number, or three (R, G, B) multiplying the texture.
        emissive_intensity: For `unlit`: None (the provisional default) or one positive number.

    Returns:
        dict: `usd_path` (absolute), `texture` (the texture's path relative to the USD, or None),
        `vertices`, `faces`, and `bounds` ([min, max] corners in metres, world frame), and the options as
        written: `shading`, `exposure_scale` (three floats or None), and `emissive_intensity` (a float or None).

    Raises:
        ValueError: If an option is invalid (see `resolve_background_exposure`), or `unlit` or `exposure_scale`
            is asked for on a GLB that has no base-colour texture.
    """
    shading, scale_rgb, intensity = resolve_background_exposure(shading, exposure_scale, emissive_intensity)
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

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

    # Refused before anything is written: without a texture there is nothing for the emission or the multiplier.
    if texture_rel is None and shading == "unlit":
        raise ValueError("unlit shading needs a base-colour texture, the GLB has none")
    if texture_rel is None and scale_rgb is not None:
        raise ValueError("exposure_scale needs a base-colour texture, the GLB has none")

    with atomic_output_path(usd_path) as tmp:
        stage = Usd.Stage.CreateNew(str(tmp))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, USD_ROOT)
        stage.SetDefaultPrim(root.GetPrim())
        # The one rigid body: no collision and no mass on it, the object is fixed and visual-only.
        base_link = UsdGeom.Xform.Define(stage, USD_BASE_LINK)
        UsdPhysics.RigidBodyAPI.Apply(base_link.GetPrim())

        mesh_prim = UsdGeom.Mesh.Define(stage, f"{USD_BASE_LINK}/mesh")
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
        if shading == "unlit":
            _author_unlit_material(stage, material_prim, texture_rel, intensity)
        else:
            _author_lit_material(stage, material_prim, texture_rel, diffuse, scale_rgb)
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(material_prim)

        stage.GetRootLayer().Save()

    return {
        "usd_path": str(usd_path.resolve()),
        "texture": texture_rel,
        "vertices": int(len(points)),
        "faces": int(len(faces)),
        "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "shading": shading,
        "exposure_scale": None if scale_rgb is None else [float(c) for c in scale_rgb],
        "emissive_intensity": intensity,
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
