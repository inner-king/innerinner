# SPDX-License-Identifier: Apache-2.0
"""
Round trip of a cut piece through USD (simfoundry/cutting/usd_io.py).

Needs a standalone ``pxr`` (usd-core; the 3dgrut env has one) and skips without it. Isaac Sim's own ``pxr`` is
importable only once Kit runs, so these do not run in the simfoundry env.
"""

import hashlib
import os

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

from simfoundry.cutting.geometry import MeshPart, PieceGeometry, convex_hull  # noqa: E402
from simfoundry.cutting.usd_io import piece_source, read_source, write_piece  # noqa: E402


def box(lo, hi):
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def write_parent(path):
    """A SimFoundry-style asset: textured skin with face-varying st, a scaled convex collider, authored inertia."""
    (path.parent / "material").mkdir(parents=True, exist_ok=True)
    (path.parent / "material" / "skin.png").write_bytes(b"not really a png")
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/fruit")
    stage.SetDefaultPrim(root.GetPrim())
    mat = UsdShade.Material.Define(stage, "/fruit/Looks/skin")
    shader = UsdShade.Shader.Define(stage, "/fruit/Looks/skin/skin")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("material/skin.png"))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.CreateOutput("out", Sdf.ValueTypeNames.Token))
    link = UsdGeom.Xform.Define(stage, "/fruit/link_iter_0_canonical_mesh_obj")
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(link.GetPrim())
    mass.CreateMassAttr(0.2)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.001, 0, 0))
    V, F = box([-0.03, -0.02, 0], [0.03, 0.02, 0.05])
    vis = UsdGeom.Mesh.Define(stage, "/fruit/link_iter_0_canonical_mesh_obj/visuals/skin")
    vis.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
    vis.CreateFaceVertexCountsAttr([3] * len(F))
    vis.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.reshape(-1).astype(np.int32)))
    st = UsdGeom.PrimvarsAPI(vis).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray.FromNumpy((V[F][:, :, :2] * 10).reshape(-1, 2).astype(np.float32)))
    UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(mat)
    # a unit cube scaled into place, as the scanned assets store their hulls
    U, G = box([0, 0, 0], [1, 1, 1])
    col = UsdGeom.Mesh.Define(stage, "/fruit/link_iter_0_canonical_mesh_obj/collisions/collision_0")
    col.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(U.astype(np.float32)))
    col.CreateFaceVertexCountsAttr([3] * len(G))
    col.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(G.reshape(-1).astype(np.int32)))
    col.AddTranslateOp().Set(Gf.Vec3d(-0.03, -0.02, 0))
    col.AddScaleOp().Set(Gf.Vec3f(0.06, 0.04, 0.05))
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    stage.GetRootLayer().Save()


@pytest.fixture
def parent(tmp_path):
    path = tmp_path / "assets" / "fruit" / "fruit.usd"
    path.parent.mkdir(parents=True)
    write_parent(path)
    return path


def half_piece(source):
    """The x >= 0 half of the box parent, built by hand (cut_geometry needs shapely, absent beside usd-core)."""
    skin = source.visuals[0]
    V, F = box([0.0, -0.02, 0], [0.03, 0.02, 0.05])
    st = V[F][:, :, :2] * 10
    hull = convex_hull(V)
    return PieceGeometry([MeshPart(V, F, {"st": st}, skin.name, skin.material)],
                         (V[[0, 1, 3, 2]], np.array([[0, 2, 1], [0, 3, 2]])), [hull], hull.volume, hull.centroid)


def test_read_source_bakes_collider_transforms_and_keeps_face_varying_st(parent):
    source, info = read_source(parent)
    assert info.link_name == "link_iter_0_canonical_mesh_obj" and info.root_name == "fruit"
    assert info.mass == pytest.approx(0.2)
    assert info.materials == {"skin": "/fruit/Looks/skin"}
    assert len(source.hulls) == 1
    assert np.allclose(source.hulls[0].min(0), [-0.03, -0.02, 0]) and np.allclose(source.hulls[0].max(0),
                                                                                 [0.03, 0.02, 0.05])
    skin = source.visuals[0]
    assert skin.corner_attrs["st"].shape == (12, 3, 2)
    assert np.allclose(skin.corner_attrs["st"], skin.points[skin.faces][:, :, :2] * 10, atol=1e-6)


def test_write_piece_round_trip(parent, tmp_path):
    source, info = read_source(parent)
    piece = half_piece(source)
    out = tmp_path / "pieces"
    path, digest, materials = write_piece(out, "half_fruit_0", piece, info, category="half_fruit", mass=0.1,
                                          interior_color=(0.9, 0.8, 0.6), generation=1)
    # named and hashed by its own bytes, which is what OmniGibson's expected_file_hash checks
    assert os.path.basename(path) == f"half_fruit_0_{digest[:12]}.usd"
    assert hashlib.md5(open(path, "rb").read()).hexdigest() == digest
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    assert root.GetAttribute("ig:category").Get() == "half_fruit"
    link = root.GetChild("link_iter_0_canonical_mesh_obj")
    assert link.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(link).GetMassAttr().Get() == pytest.approx(0.1)
    assert not UsdPhysics.MassAPI(link).GetCenterOfMassAttr().HasAuthoredValue()  # PhysX derives it from the hulls
    # the skin's material is the parent's, by reference: its texture still resolves next to the parent
    skin = stage.GetPrimAtPath(f"{link.GetPath()}/visuals/skin")
    bound = UsdShade.MaterialBindingAPI(skin).ComputeBoundMaterial()[0]
    tex = stage.GetPrimAtPath(f"{bound.GetPath()}/skin").GetAttribute("inputs:diffuse_texture").Get()
    assert os.path.samefile(tex.resolvedPath, parent.parent / "material" / "skin.png")
    assert materials == {"/fruit/Looks/skin": str(bound.GetPath())}
    cap = stage.GetPrimAtPath(f"{link.GetPath()}/visuals/cut_face_1")
    assert UsdShade.MaterialBindingAPI(cap).ComputeBoundMaterial()[0].GetPath().name == "cut_face"
    hull = stage.GetPrimAtPath(f"{link.GetPath()}/collisions/hull_0")
    assert UsdPhysics.MeshCollisionAPI(hull).GetApproximationAttr().Get() == "convexHull"
    # reading it back gives what piece_source() says without reading
    read, read_info = read_source(path)
    kept, _ = piece_source(piece, path, info, 1, materials)
    assert [p.name for p in read.visuals] == [p.name for p in kept.visuals] == ["skin", "cut_face_1"]
    assert [p.material for p in read.visuals][0] == kept.visuals[0].material
    assert np.allclose(read.visuals[0].corner_attrs["st"], piece.visuals[0].corner_attrs["st"], atol=1e-6)
    assert np.allclose(read.hulls[0], piece.hulls[0].points, atol=1e-7)
    assert read_info.link_name == info.link_name


def test_identical_cut_reuses_its_file_and_another_never_overwrites(parent, tmp_path):
    source, info = read_source(parent)
    piece = half_piece(source)
    out = tmp_path / "pieces"
    kwargs = dict(category="half_fruit", interior_color=(0.9, 0.8, 0.6), generation=1)
    a = write_piece(out, "p", piece, info, mass=0.1, **kwargs)
    b = write_piece(out, "p", piece, info, mass=0.1, **kwargs)
    c = write_piece(out, "p", piece, info, mass=0.2, **kwargs)
    assert a[0] == b[0] and a[1] == b[1]
    assert c[0] != a[0]
    assert sorted(os.listdir(out)) == sorted({os.path.basename(a[0]), os.path.basename(c[0])})


def test_a_piece_cut_again_references_its_own_materials(parent, tmp_path):
    source, info = read_source(parent)
    first, _, _ = write_piece(tmp_path / "pieces", "p", half_piece(source), info, category="half_fruit",
                                      mass=0.1, interior_color=(0.9, 0.8, 0.6), generation=1)
    again, again_info = read_source(first)
    second, _, _ = write_piece(tmp_path / "pieces", "q", half_piece(again), again_info, category="half_fruit",
                               mass=0.05, interior_color=(0.9, 0.8, 0.6), generation=2)
    stage = Usd.Stage.Open(second)
    link = stage.GetDefaultPrim().GetChild("link_iter_0_canonical_mesh_obj")
    skin = stage.GetPrimAtPath(f"{link.GetPath()}/visuals/skin")
    bound = UsdShade.MaterialBindingAPI(skin).ComputeBoundMaterial()[0]
    tex = stage.GetPrimAtPath(f"{bound.GetPath()}/skin").GetAttribute("inputs:diffuse_texture").Get()
    assert os.path.samefile(tex.resolvedPath, parent.parent / "material" / "skin.png")
