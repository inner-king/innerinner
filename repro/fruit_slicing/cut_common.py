# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the runtime-cutting benchmarks (bench_cooking.py, bench_swap.py).

The pieces come from prepare_cut_pieces.py (``<object>_cut<k>.npz``); this turns one half and a collision variant
into USD: a rigid body prim with a visual mesh and collider prims, in the layout SimFoundry's assets use
(``<link>/visuals``, ``<link>/collisions``). pxr is imported inside the functions: it exists only once Isaac Sim
runs.
"""
import numpy as np

#: Collision variants, in the order the README reports them.
VARIANTS = ("box", "hull", "hull_dense", "clip", "coacd8", "vhacd", "vhacd4", "vhacd8", "vhacd16", "sdf128", "sdf256")

#: V-HACD's error tolerance for ``vhacd<N>``. At PhysX's default of 10 % it keeps a near-convex fruit half as one
#: hull, so the hull limit never binds; at 1 % it splits the half up to the limit. ``vhacd`` keeps the defaults.
VHACD_ERROR_PERCENTAGE = 1.0


def unpack(z, prefix):
    """Ragged (V, F) parts stored by prepare_cut_pieces.pack_parts -> list of (V, F)."""
    V, VO, F, FO = (z[f"{prefix}_{k}"] for k in ("V", "VO", "F", "FO"))
    return [(V[VO[i]:VO[i + 1]], F[FO[i]:FO[i + 1]]) for i in range(len(VO) - 1)]


def colliders(variant, z, side, jitter=None):
    """Collider specs for one half of a cut.

    Args:
        variant (str): one of VARIANTS, or ``sdf<res>_dense`` (SDF of the full-resolution visual half)
        z: the cut's npz
        side (int): 0 or 1
        jitter (None or np.ndarray): (3,) offset added to every collider vertex, so that a mesh differs from any
            cooked in an earlier process and the cooking caches cannot serve it

    Returns:
        list: ("box", lo, hi) or ("mesh", V, F, approximation, {attribute name: value}) per collider prim
    """
    s = f"s{side}"
    d = np.zeros(3) if jitter is None else np.asarray(jitter)
    if variant == "box":
        V = z[f"{s}_proxy_V"]
        return [("box", V.min(0), V.max(0))]
    if variant in ("hull", "hull_dense"):
        kind = "visual" if variant == "hull_dense" else "proxy"
        return [("mesh", z[f"{s}_{kind}_V"] + d, z[f"{s}_{kind}_F"], "convexHull", {})]
    if variant == "clip" or variant.startswith("coacd"):
        return [("mesh", V + d, F, "convexHull", {}) for V, F in unpack(z, f"{s}_{variant}")]
    if variant.startswith("vhacd"):
        attrs = {} if variant == "vhacd" else {
            "physxConvexDecompositionCollision:maxConvexHulls": int(variant[5:]),
            "physxConvexDecompositionCollision:errorPercentage": VHACD_ERROR_PERCENTAGE}
        return [("mesh", z[f"{s}_proxy_V"] + d, z[f"{s}_proxy_F"], "convexDecomposition", attrs)]
    if variant.startswith("sdf"):
        dense = variant.endswith("_dense")
        res = int(variant[3:].removesuffix("_dense"))
        kind = "visual" if dense else "proxy"
        return [("mesh", z[f"{s}_{kind}_V"] + d, z[f"{s}_{kind}_F"], "sdf",
                 {"physxSDFMeshCollision:sdfResolution": res})]
    raise ValueError(f"unknown collision variant {variant}")


def collider_triangles(specs):
    """Triangles PhysX is handed to cook, over all collider prims of one half."""
    return int(sum(len(c[2]) for c in specs if c[0] == "mesh"))


def _set_mesh(mesh, V, F):
    from pxr import Gf, Vt

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V, np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(F, np.int32).reshape(-1)))
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*V.min(0).tolist()), Gf.Vec3f(*V.max(0).tolist())]))


def author_rigid_body(stage, path, mass, visual, specs, pos=None, quat_xyzw=None, color=(0.85, 0.75, 0.55)):
    """Author one rigid body: an Xform with RigidBodyAPI and MassAPI, a visual mesh and its collider prims.

    Only the mass is authored; PhysX derives the centre of mass and inertia from the colliders.

    Args:
        stage (Usd.Stage): stage to author on (the live one, or a new file's)
        path (str): prim path of the rigid body
        mass (float): kg
        visual (None or tuple): (V, F) of the visual mesh, or None for colliders only
        specs (list): collider specs from colliders()
        pos (None or sequence): translation to author on the body, if any
        quat_xyzw (None or sequence): orientation to author on the body, if any
        color (tuple): display color of the visual mesh
    """
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    body = UsdGeom.Xform.Define(stage, path)
    if pos is not None:
        body.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
    if quat_xyzw is not None:
        x, y, z, w = map(float, quat_xyzw)
        body.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(float(mass))
    if visual is not None:
        UsdGeom.Xform.Define(stage, f"{path}/visuals")
        mesh = UsdGeom.Mesh.Define(stage, f"{path}/visuals/mesh")
        _set_mesh(mesh, *visual)
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    UsdGeom.Xform.Define(stage, f"{path}/collisions")
    for j, spec in enumerate(specs):
        cpath = f"{path}/collisions/collision_{j}"
        if spec[0] == "box":
            lo, hi = np.asarray(spec[1]), np.asarray(spec[2])
            cube = UsdGeom.Cube.Define(stage, cpath)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2).tolist()))
            cube.AddScaleOp().Set(Gf.Vec3f(*(hi - lo).tolist()))
            cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            continue
        _, V, F, approx, attrs = spec
        mesh = UsdGeom.Mesh.Define(stage, cpath)
        _set_mesh(mesh, V, F)
        mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approx)
        if approx in ("convexDecomposition", "sdf"):
            from pxr import PhysxSchema  # Isaac Sim's pxr only; standalone usd-core lacks it

            api = (PhysxSchema.PhysxConvexDecompositionCollisionAPI if approx == "convexDecomposition"
                   else PhysxSchema.PhysxSDFMeshCollisionAPI)
            api.Apply(prim)
        for name, value in attrs.items():
            prim.GetAttribute(name).Set(value)


def write_piece_usd(usd_path, mass, visual, specs, category):
    """Write one half as a standalone asset OmniGibson's USDObject can load: a default prim ``/piece`` holding one
    link, ``/piece/base_link``, which is the rigid body.

    Args:
        usd_path (str or Path): file to create
        mass (float): kg
        visual (tuple): (V, F) of the visual mesh
        specs (list): collider specs from colliders()
        category (str): written as ``ig:category`` on the default prim

    Returns:
        str: the link name
    """
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/piece")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().CreateAttribute("ig:category", Sdf.ValueTypeNames.String).Set(category)
    author_rigid_body(stage, "/piece/base_link", mass, visual, specs)
    stage.GetRootLayer().Save()
    return "base_link"


def half_masses(cut_manifest, total_mass):
    """Split the fruit's mass between the two halves by their proxy volumes."""
    vols = [h["proxy_volume_m3"] or 0.0 for h in cut_manifest["halves"]]
    if sum(vols) <= 0:
        return [total_mass / 2, total_mass / 2]
    return [total_mass * v / sum(vols) for v in vols]
