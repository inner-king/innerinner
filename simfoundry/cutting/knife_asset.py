# SPDX-License-Identifier: Apache-2.0
"""
A procedural chef's knife as a USD asset OmniGibson's ``USDObject`` loads, with its blade marked for the cutting rule.

One rigid link holds three convex colliders, ``blade``, ``bolster`` and
``handle``; the rule tells the blade apart by its collider name, so a knife
pressed onto a fruit by its handle or bolster does not cut. The geometry is the
knife repro/fruit_slicing/make_slicing_scene.py designed: a 16 x 4.5 cm wedge
blade, 2.2 mm at the spine and 0.4 mm at the edge, its edge rising toward the
tip.

Knife frame: x from the handle to the tip, y the blade's normal, z up to the
spine. The edge lies at z = 0 from the bolster (x = 0) to 60 % of the blade, so a
knife held with z up cuts with its edge down and its plane vertical.

Needs ``pxr``: Isaac Sim's own, or a standalone usd-core.
"""

from pathlib import Path

import numpy as np

BLADE_LEN, BLADE_H = 0.16, 0.045
SPINE_T, EDGE_T = 0.0022, 0.0004
BOLSTER_LEN = 0.012
HANDLE_LEN, HANDLE_H, HANDLE_T = 0.11, 0.024, 0.016
KNIFE_MASS = 0.18

#: The ``cutting`` entry that marks this knife (objects_info.init_info[name].args.cutting).
CUTTING_SPEC = {"role": "knife", "blade": "blade"}


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def knife_meshes():
    """Blade, bolster and handle as (V, F) triangle meshes in the knife frame, outward-wound."""
    xs = np.linspace(0, BLADE_LEN, 33)
    t = xs / BLADE_LEN
    edge_z = 0.016 * (np.clip(t - 0.6, 0, None) / 0.4) ** 2
    spine_z = BLADE_H - (BLADE_H - 0.016) * (np.clip(t - 0.5, 0, None) / 0.5) ** 1.3
    thick = SPINE_T * (1 - 0.6 * t)
    V, F = [], []
    for i, x in enumerate(xs):
        V += [[x, -EDGE_T / 2, edge_z[i]], [x, EDGE_T / 2, edge_z[i]],
              [x, thick[i] / 2, spine_z[i]], [x, -thick[i] / 2, spine_z[i]]]
    for i in range(len(xs) - 1):
        a, b = 4 * i, 4 * (i + 1)
        for k in range(4):
            k2 = (k + 1) % 4
            F += [[a + k, b + k2, b + k], [a + k, a + k2, b + k2]]
    e = 4 * (len(xs) - 1)
    F += [[0, 2, 1], [0, 3, 2], [e, e + 1, e + 2], [e, e + 2, e + 3]]
    blade = (np.array(V), np.array(F))
    bolster = _box([-BOLSTER_LEN, -0.006, 0.012], [0.0, 0.006, BLADE_H])
    handle = _box([-BOLSTER_LEN - HANDLE_LEN, -HANDLE_T / 2, BLADE_H - HANDLE_H], [-BOLSTER_LEN, HANDLE_T / 2, BLADE_H])
    return {"blade": blade, "bolster": bolster, "handle": handle}


def write_knife_usd(path, mass=KNIFE_MASS):
    """Write the knife asset.

    Args:
        path (str or Path): file to (over)write
        mass (float): kg

    Returns:
        str: the path written
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from scipy.spatial import ConvexHull

    from simfoundry.cutting.usd_io import _author_mesh, _define_flat_material, _identity_ops

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/knife")
    stage.SetDefaultPrim(root.GetPrim())
    _identity_ops(root)
    root.GetPrim().CreateAttribute("ig:category", Sdf.ValueTypeNames.String).Set("knife")
    UsdGeom.Scope.Define(stage, "/knife/Looks")
    looks = {"blade": _define_flat_material(stage, "/knife/Looks/steel", (0.78, 0.8, 0.82), roughness=0.25),
             "bolster": None, "handle": _define_flat_material(stage, "/knife/Looks/handle", (0.12, 0.08, 0.06))}
    looks["bolster"] = looks["blade"]
    link = UsdGeom.Xform.Define(stage, "/knife/base_link")
    _identity_ops(link)
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(float(mass))
    UsdGeom.Xform.Define(stage, "/knife/base_link/visuals")
    UsdGeom.Xform.Define(stage, "/knife/base_link/collisions")
    allV = []
    for name, (V, F) in knife_meshes().items():
        # Unwelded triangles shade flat, as a machined blade and a box handle should look
        Vu = V[F.reshape(-1)]
        mesh = _author_mesh(stage, f"/knife/base_link/visuals/{name}", Vu, np.arange(len(Vu)).reshape(-1, 3))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(looks[name])
        hull = ConvexHull(V)
        Vh = V[hull.vertices]
        remap = np.full(len(V), -1)
        remap[hull.vertices] = np.arange(len(hull.vertices))
        Fh = remap[hull.simplices]
        inner = Vh.mean(0)
        tri = Vh[Fh]
        flip = np.einsum("ij,ij->i", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), tri[:, 0] - inner) < 0
        Fh[flip] = Fh[flip][:, ::-1]
        coll = _author_mesh(stage, f"/knife/base_link/collisions/{name}", Vh, Fh, purpose=UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(coll.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(coll.GetPrim()).CreateApproximationAttr("convexHull")
        allV.append(V)
    allV = np.concatenate(allV)
    root.GetPrim().CreateAttribute("ig:nativeBB", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3f(*(allV.max(0) - allV.min(0)).tolist()))
    stage.GetRootLayer().Save()
    return str(path.resolve())
