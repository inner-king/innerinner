# SPDX-License-Identifier: Apache-2.0
"""
Read a cuttable object's geometry out of its USD asset, and write each cut piece as a new USD asset.

Needs ``pxr``: Isaac Sim's own, once Kit runs, or a standalone ``usd-core``
(the 3dgrut env has one). Nothing here imports OmniGibson.

The pieces follow the layout SimFoundry's assets use and OmniGibson's
``USDObject`` loads: a default prim holding one rigid link, with ``visuals``
and ``collisions`` under it. The link keeps its parent's name, so a scene's
``link_physics_materials`` (keyed by link name) applies to the pieces as it
did to the parent. Only ``physics:mass`` is authored on the link; PhysX derives
the centre of mass and inertia from the piece's colliders. The parent's
``centerOfMass``/``diagonalInertia``/``principalAxes`` would be wrong for half
of it.

A piece never rewrites an asset (AGENTS.md, saved-scene rule 1). Its file is
named after the MD5 of its own bytes, which is also the ``expected_file_hash``
OmniGibson checks at load, so a saved scene always points at exactly the
geometry it was saved with, and two cuts can never overwrite each other's file.

The skin's material is referenced from the parent's file rather than copied:
its texture paths are relative to that file and stay resolvable from there, and
nothing is duplicated. A piece therefore needs its parent's asset to exist,
as the uncut scene does. The cut face gets a flat OmniPBR material of its own.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
import uuid

import numpy as np

from simfoundry.cutting.geometry import MeshPart, SourceGeometry

#: Name of the flat material every cut face of a piece is bound to.
CUT_FACE_MATERIAL = "cut_face"

#: Prim name prefix of the cut-face meshes. A piece cut again recognizes its earlier cut faces by it, so that their
#: clipped remains keep the flat material instead of referencing the parent's.
CUT_FACE_PREFIX = "cut_face"


class UncuttableAsset(ValueError):
    """The asset's structure is not one a plane cut can be authored for."""


@dataclass
class SourceInfo:
    """What a cut needs from a cuttable asset besides its geometry.

    Attributes:
        usd_path (str): absolute path of the asset
        root_name (str): name of its default prim
        link_name (str): name of its one rigid link
        mass (None or float): ``physics:mass`` authored on the link, if any
        materials (dict): visual mesh name -> bound material prim path in the asset (None if unbound)
        up_axis (str): the asset's stage up axis
        meters_per_unit (float): the asset's stage units
    """

    usd_path: str
    root_name: str
    link_name: str
    mass: object
    materials: dict
    up_axis: str
    meters_per_unit: float


def _triangulate(counts, indices, corner_attrs):
    """Fan-triangulate polygons; per-corner attributes follow their corners."""
    counts = np.asarray(counts, np.int64)
    indices = np.asarray(indices, np.int64)
    if np.all(counts == 3):
        return indices.reshape(-1, 3), {k: v.reshape(len(counts), 3, -1) for k, v in corner_attrs.items()}
    if np.any(counts < 3):
        counts_ok = counts >= 3
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        keep = np.concatenate([np.arange(s, s + c) for s, c, ok in zip(starts, counts, counts_ok) if ok])
        indices = indices[keep]
        corner_attrs = {k: v[keep] for k, v in corner_attrs.items()}
        counts = counts[counts_ok]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n_tris = counts - 2
    face_of_tri = np.repeat(np.arange(len(counts)), n_tris)
    k = np.arange(n_tris.sum()) - np.repeat(np.cumsum(n_tris) - n_tris, n_tris)
    corners = np.stack([starts[face_of_tri], starts[face_of_tri] + k + 1, starts[face_of_tri] + k + 2], 1)
    return indices[corners], {name: v[corners] for name, v in corner_attrs.items()}


def read_source(usd_path):
    """The geometry of a single-link rigid asset, in its link's frame.

    Visual meshes are every mesh under the link without a ``CollisionAPI``;
    colliders are every mesh with one, read as convex hulls of their points. Mesh
    transforms below the link are baked into the points.

    Args:
        usd_path (str or Path): the asset

    Returns:
        tuple: (SourceGeometry, SourceInfo)

    Raises:
        UncuttableAsset: no single rigid link, no visual mesh or no collider
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    usd_path = os.path.abspath(str(usd_path))
    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    if not root:
        raise UncuttableAsset(f"{usd_path}: no default prim")
    links = [c for c in root.GetChildren() if c.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(links) != 1:
        raise UncuttableAsset(f"{usd_path}: {len(links)} rigid links; only single-link objects can be cut")
    link = links[0]
    cache = UsdGeom.XformCache()
    visuals, hulls, materials = [], [], {}
    for prim in Usd.PrimRange(link):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        M = np.array(cache.ComputeRelativeTransform(prim, link)[0], np.float64)
        pts = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
        pts = pts @ M[:3, :3] + M[3, :3]  # USD matrices act on row vectors
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            hulls.append(pts)
            continue
        if mesh.GetPurposeAttr().Get() == UsdGeom.Tokens.guide:
            continue
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64)
        corner = {}
        st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if st and st.HasValue():
            values = np.asarray(st.ComputeFlattened(), np.float64).reshape(-1, 2)
            interp = st.GetInterpolation()
            if interp == UsdGeom.Tokens.faceVarying:
                corner["st"] = values
            elif interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
                corner["st"] = values[indices]
            elif interp == UsdGeom.Tokens.uniform:
                corner["st"] = np.repeat(values, counts, axis=0)
        faces, corner = _triangulate(counts, indices, corner)
        # Kept in the rule's cache, 290k triangles each: 32-bit indices and UVs (USD stores float UVs anyway)
        faces = faces.astype(np.int32)
        corner = {k: v.astype(np.float32) for k, v in corner.items()}
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        materials[prim.GetName()] = str(bound.GetPath()) if bound else None
        visuals.append(MeshPart(pts, faces, corner, name=prim.GetName(), material=materials[prim.GetName()]))
    if not visuals or not hulls:
        raise UncuttableAsset(f"{usd_path}: {len(visuals)} visual meshes and {len(hulls)} colliders under "
                              f"{link.GetPath()}; a cut needs both")
    mass_attr = UsdPhysics.MassAPI(link).GetMassAttr()
    mass = mass_attr.Get() if mass_attr and mass_attr.HasAuthoredValue() else None
    info = SourceInfo(usd_path=usd_path, root_name=root.GetName(), link_name=link.GetName(),
                      mass=float(mass) if mass else None, materials=materials,
                      up_axis=str(UsdGeom.GetStageUpAxis(stage)),
                      meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)))
    return SourceGeometry(visuals, hulls), info


def _identity_ops(xformable):
    from pxr import Gf, UsdGeom

    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 1, 1))


def _author_mesh(stage, path, V, F, purpose=None):
    from pxr import Gf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V, np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(F, np.int32).reshape(-1)))
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*np.min(V, 0).tolist()), Gf.Vec3f(*np.max(V, 0).tolist())]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if purpose is not None:
        mesh.CreatePurposeAttr(purpose)
    _identity_ops(mesh)
    return mesh


def _define_flat_material(stage, path, color, roughness=0.6):
    """An OmniPBR material of one constant colour (no textures)."""
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(out)
    material.CreateDisplacementOutput("mdl").ConnectToSource(out)
    material.CreateVolumeOutput("mdl").ConnectToSource(out)
    return material


def is_cut_face(part):
    """Whether a visual mesh is a cut face authored by write_piece()."""
    return part.name.startswith(CUT_FACE_PREFIX)


def write_piece(directory, stem, piece, source_info, *, category, mass, interior_color, generation):
    """Write one piece of a cut as a new USD asset.

    Args:
        directory (str or Path): where piece files go (created if missing)
        stem (str): file name prefix; the MD5 of the file is appended
        piece (geometry.PieceGeometry): the piece, in the parent's link frame
        source_info (SourceInfo): the parent asset
        category (str): written as ``ig:category``
        mass (float): kg, authored as the link's ``physics:mass``
        interior_color (sequence): RGB in [0, 1] of the cut faces
        generation (int): how many cuts led to this piece; names this cut's face mesh

    Returns:
        tuple: (absolute path of the file, its MD5 hex digest, {parent material path: material path in the piece})
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".{stem}.{uuid.uuid4().hex}.usd"
    stage = Usd.Stage.CreateNew(str(tmp))
    UsdGeom.SetStageUpAxis(stage, source_info.up_axis)
    UsdGeom.SetStageMetersPerUnit(stage, source_info.meters_per_unit)
    root_path = f"/{source_info.root_name}"
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    _identity_ops(root)
    root.GetPrim().CreateAttribute("ig:category", Sdf.ValueTypeNames.String).Set(category)

    # Materials: the skin's referenced from the parent's file, the cut faces' flat and local
    UsdGeom.Scope.Define(stage, f"{root_path}/Looks")
    flat = _define_flat_material(stage, f"{root_path}/Looks/{CUT_FACE_MATERIAL}", interior_color)
    relative_parent = os.path.relpath(source_info.usd_path, directory)
    referenced = {}
    for part in piece.visuals:
        if is_cut_face(part) or part.material is None or part.material in referenced:
            continue
        name = Sdf.Path(part.material).name
        while name == CUT_FACE_MATERIAL or name in {Sdf.Path(p).name for p in referenced.values()}:
            name += "_"
        path = f"{root_path}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        material.GetPrim().GetReferences().AddReference(relative_parent, Sdf.Path(part.material))
        referenced[part.material] = path

    link_path = f"{root_path}/{source_info.link_name}"
    link = UsdGeom.Xform.Define(stage, link_path)
    _identity_ops(link)
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(float(mass))

    UsdGeom.Xform.Define(stage, f"{link_path}/visuals")
    names = set()
    extent = []
    for part in piece.visuals:
        name = part.name
        while name in names:
            name += "_"
        names.add(name)
        mesh = _author_mesh(stage, f"{link_path}/visuals/{name}", part.points, part.faces)
        st = part.corner_attrs.get("st")
        if st is not None:
            pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            pv.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(st.reshape(-1, 2), np.float32)))
        if is_cut_face(part):
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(flat)
            mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, interior_color))]))
        elif part.material in referenced:
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                UsdShade.Material(stage.GetPrimAtPath(referenced[part.material])))
        extent.append(part.points)
    if piece.cap is not None:
        name = f"{CUT_FACE_PREFIX}_{generation}"
        while name in names:
            name += "_"
        mesh = _author_mesh(stage, f"{link_path}/visuals/{name}", *piece.cap)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(flat)
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, interior_color))]))
        extent.append(piece.cap[0])

    UsdGeom.Xform.Define(stage, f"{link_path}/collisions")
    for i, hull in enumerate(piece.hulls):
        mesh = _author_mesh(stage, f"{link_path}/collisions/hull_{i}", hull.points, hull.faces,
                            purpose=UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")

    allV = np.concatenate(extent)
    root.GetPrim().CreateAttribute("ig:nativeBB", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3f(*(allV.max(0) - allV.min(0)).tolist()))
    stage.GetRootLayer().Save()
    del stage

    digest = hashlib.md5(tmp.read_bytes()).hexdigest()
    final = directory / f"{stem}_{digest[:12]}.usd"
    if final.exists():
        tmp.unlink()  # same bytes, same name: an identical cut was written before
    else:
        os.replace(tmp, final)
    return str(final.resolve()), digest, referenced


def piece_source(piece, usd_path, source_info, generation, materials):
    """The SourceGeometry and SourceInfo of a piece just written, without reading the file back.

    The cut face authored by write_piece() becomes one more visual mesh, so a
    second cut clips it like the skin.

    Args:
        piece (geometry.PieceGeometry): the piece written
        usd_path (str): its file
        source_info (SourceInfo): its parent's
        generation (int): as passed to write_piece()
        materials (dict): write_piece()'s parent -> piece material paths
    """
    visuals = [MeshPart(p.points, p.faces, p.corner_attrs, p.name, materials.get(p.material, p.material))
               for p in piece.visuals]
    if piece.cap is not None:
        visuals.append(MeshPart(piece.cap[0], piece.cap[1], {}, name=f"{CUT_FACE_PREFIX}_{generation}",
                                material=None))
    info = SourceInfo(usd_path=usd_path, root_name=source_info.root_name, link_name=source_info.link_name,
                      mass=None, materials={p.name: p.material for p in visuals}, up_axis=source_info.up_axis,
                      meters_per_unit=source_info.meters_per_unit)
    return SourceGeometry(visuals, [h.points for h in piece.hulls]), info
