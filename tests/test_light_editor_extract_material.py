# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The light editor's texture lookup on the background USD: the lit texture and the unlit emission texture (B19).

It needs pxr, trimesh and PIL, so it skips where any of them is missing; the simfoundry-editor env has all three.
"""

import pathlib

import numpy as np
import pytest

pytest.importorskip("pxr")
pytest.importorskip("trimesh")
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from simfoundry.pipeline import background_usd as bu  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
LIGHT_EDITOR = REPO / "scripts/interactive/light_editor"


@pytest.fixture
def extract(monkeypatch):
    monkeypatch.syspath_prepend(str(LIGHT_EDITOR))
    import extract as module

    return module


def _textured_glb(tmp_path):
    """One triangle with a 16 x 16 texture: the smallest GLB that the background writer textures."""
    import trimesh
    from trimesh.visual.material import SimpleMaterial
    from trimesh.visual.texture import TextureVisuals

    from simfoundry.pipeline.depth_mesh import export_glb

    pixels = (np.arange(16 * 16 * 3) % 251).astype(np.uint8).reshape(16, 16, 3)
    mesh = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]), process=False,
        visual=TextureVisuals(uv=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                              material=SimpleMaterial(image=Image.fromarray(pixels))))
    glb = tmp_path / "background_mesh.glb"
    export_glb(mesh, glb)
    return glb, pixels


def _mesh_prim(usd_path):
    """The stage and its mesh prim; the caller keeps the stage alive while it reads the prim."""
    stage = Usd.Stage.Open(str(usd_path))
    return stage, stage.GetPrimAtPath("/mesh_background/base_link/mesh")


def _reference(extract, usd_dir, texture="./textures/mesh_background_0_basecolor.png"):
    """The texture as the extractor reads the file itself. It re-encodes every texture, so this is the fair comparison."""
    return extract._open_asset_image(Sdf.AssetPath(texture), usd_dir)


def _same(found, reference):
    return found is not None and reference is not None and np.array_equal(np.asarray(found), np.asarray(reference))


def _mdl_background(path, *, colour=None, **textures):
    """A USD whose mesh is bound to one OmniPBR material with the given inputs (the names are the MDL's own)."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/mesh_background")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/mesh_background/base_link/mesh")
    material = UsdShade.Material.Define(stage, "/mesh_background/material")
    mdl = UsdShade.Shader.Define(stage, "/mesh_background/material/omnipbr")
    mdl.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    if colour is not None:
        mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    for name, value in textures.items():
        mdl.CreateInput(name, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(value))
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def test_the_lit_texture_is_found_and_is_the_texture_file(tmp_path, extract):
    glb, pixels = _textured_glb(tmp_path)
    info = bu.write_background_usd(glb, tmp_path / "lit" / "mesh_background_0.usd")
    usd_dir = pathlib.Path(info["usd_path"]).parent
    # The texture file is lossless; the extractor's copy is re-encoded, so it is compared with the extractor's reading.
    texture_file = Image.open(usd_dir / "textures" / "mesh_background_0_basecolor.png").convert("RGB")
    assert np.array_equal(np.asarray(texture_file), pixels)
    _, prim = _mesh_prim(info["usd_path"])
    found, colour = extract._bound_texture(prim, usd_dir)
    assert colour is None
    assert _same(found, _reference(extract, usd_dir))


def test_the_unlit_texture_is_found_through_the_emission_texture(tmp_path, extract):
    glb, _ = _textured_glb(tmp_path)
    info = bu.write_background_usd(glb, tmp_path / "unlit" / "mesh_background_0.usd", shading="unlit")
    usd_dir = pathlib.Path(info["usd_path"]).parent
    _, prim = _mesh_prim(info["usd_path"])
    found, colour = extract._bound_texture(prim, usd_dir)
    assert colour is None
    assert _same(found, _reference(extract, usd_dir))


def test_the_emission_intensity_is_not_applied_to_the_texture(tmp_path, extract):
    glb, _ = _textured_glb(tmp_path)
    info = bu.write_background_usd(glb, tmp_path / "bright" / "mesh_background_0.usd", shading="unlit",
                                   emissive_intensity=50000)
    usd_dir = pathlib.Path(info["usd_path"]).parent
    _, prim = _mesh_prim(info["usd_path"])
    found, _ = extract._bound_texture(prim, usd_dir)
    assert _same(found, _reference(extract, usd_dir))


def test_a_lit_multiplier_is_not_applied_to_the_texture(tmp_path, extract):
    glb, _ = _textured_glb(tmp_path)
    info = bu.write_background_usd(glb, tmp_path / "scaled" / "mesh_background_0.usd", exposure_scale=0.2)
    usd_dir = pathlib.Path(info["usd_path"]).parent
    _, prim = _mesh_prim(info["usd_path"])
    found, _ = extract._bound_texture(prim, usd_dir)
    assert _same(found, _reference(extract, usd_dir))


def test_a_diffuse_texture_wins_over_an_emission_texture(tmp_path, extract):
    textures = tmp_path / "textures"
    textures.mkdir()
    Image.fromarray(np.full((4, 4, 3), 40, np.uint8)).save(textures / "a.png")
    Image.fromarray(np.full((4, 4, 3), 200, np.uint8)).save(textures / "b.png")
    usd = tmp_path / "both.usd"
    _mdl_background(usd, diffuse_texture="./textures/a.png", emissive_color_texture="./textures/b.png")
    _, prim = _mesh_prim(usd)
    found, colour = extract._bound_texture(prim, tmp_path)
    assert colour is None
    assert _same(found, _reference(extract, tmp_path, "./textures/a.png"))
    assert not _same(found, _reference(extract, tmp_path, "./textures/b.png"))


def test_a_surface_with_no_texture_still_falls_back_to_the_flat_colour(tmp_path, extract):
    usd = tmp_path / "flat.usd"
    _mdl_background(usd, colour=(0.2, 0.3, 0.4))
    _, prim = _mesh_prim(usd)
    found, colour = extract._bound_texture(prim, tmp_path)
    assert found is None
    assert colour == pytest.approx((0.2, 0.3, 0.4), abs=1e-6)


def test_an_empty_emission_texture_is_not_opened(tmp_path, extract, monkeypatch):
    usd = tmp_path / "empty.usd"
    _mdl_background(usd, colour=(0.2, 0.3, 0.4), emissive_color_texture="")
    opened = []
    real_open = extract._open_asset_image

    def recording(asset, usd_dir):
        opened.append(asset)
        return real_open(asset, usd_dir)

    monkeypatch.setattr(extract, "_open_asset_image", recording)
    _, prim = _mesh_prim(usd)
    found, colour = extract._bound_texture(prim, tmp_path)
    assert found is None
    assert colour == pytest.approx((0.2, 0.3, 0.4), abs=1e-6)
    assert opened == []
