# SPDX-License-Identifier: Apache-2.0
"""The OmniGibson side of the food-knife co-simulation (``cosim``): Isaac Sim owns the knife.

* ``write_knife_usd`` builds a knife asset in the solver's knife frame, with the blade the solver
  cuts with (``knife.KnifeGeometry``: the Chef'd knife's belly and spine), so the body PhysX moves
  and the blade the food meets are one shape. OmniGibson's ``USDObject`` loads it.
* ``PhysicsStepCoupler`` hooks the coupling into PhysX's step: after every physics step it reads
  the knife's rigid body where PhysX has it (not OmniGibson's Xform, which lags by a step) and
  advances the food; before the next step it applies the food's force at the edge midpoint, and
  its moment about that point, to the body.
* ``FoodPoints`` shows the food as a USD points prim, one sphere per particle.
* ``collider_hulls`` and ``hull_top`` read where an asset's colliders stand, which is what the knife
  meets: a reconstructed board's convex hulls stand a couple of millimetres above its visual mesh.

The USD and PhysX parts need ``pxr`` and ``omni.physx``, i.e. a running Isaac Sim (``og.launch()``);
``knife_parts``, ``knife_mass_properties`` and ``hull_top`` are plain numpy.
"""

import math
from pathlib import Path

import numpy as np

from simfoundry.cutting.knife import CHEF_KNIFE, KnifeGeometry
from simfoundry.cutting.render import HANDLE_RGB, STEEL_RGB

#: Densities of the knife's parts [kg/m^3]: a steel blade and bolster, a plastic handle. A 20 cm chef's
#: knife weighs 150-250 g; these give 175 g, balanced near the bolster.
STEEL_DENSITY = 7850.0
HANDLE_DENSITY = 1300.0
#: Bolster and handle, as render.py draws them past the heel (u < -half_length).
BOLSTER_LENGTH = 0.012
HANDLE_HALF_THICKNESS = 0.010
HANDLE_HEIGHT = 0.025


def srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def knife_parts(geometry: KnifeGeometry = CHEF_KNIFE, sections: int = 48):
    """Blade, bolster and handle as (vertices, triangles) in the knife frame.

    The knife frame is the solver's: origin at the edge's midpoint, x along the edge to the tip,
    y the blade's normal, z towards the spine. The blade's outline is the solver's
    (``edge_height``, ``spine_height``); it thickens from the edge over ``bevel_height`` as the
    solver's blade does, and the edge and spine meet at the tip.
    """
    g = geometry
    # Cross-sections from the heel to just short of the tip, denser towards the tip where the belly
    # curves; the tip itself is one vertex, where the edge and the spine meet.
    s = np.linspace(0.0, 1.0, sections + 1)[:-1]
    u = -g.half_length + 2.0 * g.half_length * (1.0 - (1.0 - s) ** 1.5)
    we, ws = g.edge_height(u), g.spine_height(u)
    wb = we + np.minimum(g.bevel_height, 0.5 * (ws - we))  # the bevel's top, below the spine

    def thickness(h):
        return g.edge_thickness + (g.spine_thickness - g.edge_thickness) * np.clip(h / g.bevel_height, 0.0, 1.0)

    t_b, t_s = thickness(wb - we), thickness(ws - we)
    ring = []
    for i in range(len(u)):
        ring.append([
            [u[i], -0.5 * g.edge_thickness, we[i]], [u[i], 0.5 * g.edge_thickness, we[i]],
            [u[i], 0.5 * t_b[i], wb[i]], [u[i], 0.5 * t_s[i], ws[i]],
            [u[i], -0.5 * t_s[i], ws[i]], [u[i], -0.5 * t_b[i], wb[i]],
        ])
    V = np.concatenate([np.asarray(ring, dtype=float).reshape(-1, 3),
                        [[g.half_length, 0.0, float(g.edge_height(g.half_length))]]])
    tip = len(V) - 1
    F = []
    for i in range(len(u) - 1):
        a, b = 6 * i, 6 * (i + 1)
        for k in range(6):
            k2 = (k + 1) % 6
            F += [[a + k, b + k2, b + k], [a + k, a + k2, b + k2]]
    last = 6 * (len(u) - 1)
    for k in range(6):
        F.append([last + k, last + (k + 1) % 6, tip])
    for k in range(1, 5):
        F.append([0, k + 1, k])
    blade = (V, np.asarray(F))
    top = g.height - 0.001
    lo_b = np.array([-g.half_length - BOLSTER_LENGTH, -0.004, max(0.0, g.height - 0.03)])
    hi_b = np.array([-g.half_length, 0.004, top])
    lo_h = np.array([-g.half_length - BOLSTER_LENGTH - g.handle_length, -HANDLE_HALF_THICKNESS, top - HANDLE_HEIGHT])
    hi_h = np.array([-g.half_length - BOLSTER_LENGTH, HANDLE_HALF_THICKNESS, top])
    return {"blade": blade, "bolster": _box(lo_b, hi_b), "handle": _box(lo_h, hi_h)}


def handle_centre(geometry: KnifeGeometry = CHEF_KNIFE) -> np.ndarray:
    """Middle of the handle in the knife frame: where a hand holds the knife."""
    g = geometry
    top = g.height - 0.001
    return np.array([-g.half_length - BOLSTER_LENGTH - 0.5 * g.handle_length, 0.0, top - 0.5 * HANDLE_HEIGHT])


def _box(lo, hi):
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def _identity_ops(xformable):
    from pxr import Gf, UsdGeom

    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 1, 1))


def _mesh(stage, path, V, F, purpose=None):
    from pxr import Gf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V, np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(F, np.int32).reshape(-1)))
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*np.min(V, 0).tolist()), Gf.Vec3f(*np.max(V, 0).tolist())]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    if purpose is not None:
        mesh.CreatePurposeAttr(purpose)
    _identity_ops(mesh)
    return mesh


def _omni_pbr(stage, path, color, roughness, metallic=0.0):
    """An OmniPBR material of one colour (linear RGB)."""
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for o in (material.CreateSurfaceOutput("mdl"), material.CreateDisplacementOutput("mdl"),
              material.CreateVolumeOutput("mdl")):
        o.ConnectToSource(out)
    return material


def knife_mass_properties(geometry: KnifeGeometry = CHEF_KNIFE):
    """Mass [kg], centre of mass, principal moments of inertia and principal axes (3x3, columns) of the knife.

    From its parts' volumes at ``STEEL_DENSITY`` and ``HANDLE_DENSITY``. PhysX would otherwise
    spread the mass evenly over the colliders, which puts most of it in the handle, the largest.
    """
    import trimesh

    parts = knife_parts(geometry)
    density = {"blade": STEEL_DENSITY, "bolster": STEEL_DENSITY, "handle": HANDLE_DENSITY}
    masses, centres, inertias = [], [], []
    for name, (V, F) in parts.items():
        m = trimesh.Trimesh(V, F, process=True)
        m.fix_normals()
        m.density = density[name]
        masses.append(m.mass)
        centres.append(m.center_mass)
        inertias.append(m.moment_inertia)  # about its own centre of mass
    mass = float(sum(masses))
    com = sum(mi * ci for mi, ci in zip(masses, centres)) / mass
    I = np.zeros((3, 3))
    for mi, ci, Ii in zip(masses, centres, inertias):
        d = ci - com
        I += Ii + mi * (float(d @ d) * np.eye(3) - np.outer(d, d))
    moments, axes = np.linalg.eigh(I)
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] = -axes[:, 2]
    return mass, com, moments, axes


def write_knife_usd(path, geometry: KnifeGeometry = CHEF_KNIFE, steel_rgb=None):
    """Writes the knife as a one-link rigid USD asset whose link frame is the solver's knife frame.

    Its colliders are convex hulls of the blade, the bolster and the handle; its mass and inertia
    are ``knife_mass_properties``'.

    Args:
        path: File to (over)write.
        geometry: The blade, as the solver cuts with it.
        steel_rgb: sRGB colour of the blade; render.STEEL_RGB if None.

    Returns:
        str: The absolute path written.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

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
    steel = srgb_to_linear(STEEL_RGB if steel_rgb is None else steel_rgb)
    looks = {
        "blade": _omni_pbr(stage, "/knife/Looks/steel", steel, roughness=0.25, metallic=0.9),
        "handle": _omni_pbr(stage, "/knife/Looks/handle", srgb_to_linear(HANDLE_RGB), roughness=0.5),
    }
    looks["bolster"] = looks["blade"]
    link = UsdGeom.Xform.Define(stage, "/knife/base_link")
    _identity_ops(link)
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    mass, com, moments, axes = knife_mass_properties(geometry)
    mass_api = UsdPhysics.MassAPI.Apply(link.GetPrim())
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, moments)))
    qx, qy, qz, qw = Rotation.from_matrix(axes).as_quat()
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
    UsdGeom.Xform.Define(stage, "/knife/base_link/visuals")
    UsdGeom.Xform.Define(stage, "/knife/base_link/collisions")
    every = []
    for name, (V, F) in knife_parts(geometry).items():
        # Unwelded triangles shade flat, as a ground blade and a box handle should look.
        Vu = V[F.reshape(-1)]
        mesh = _mesh(stage, f"/knife/base_link/visuals/{name}", Vu, np.arange(len(Vu)).reshape(-1, 3))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(looks[name])
        hull = ConvexHull(V)
        Vh = V[hull.vertices]
        remap = np.full(len(V), -1)
        remap[hull.vertices] = np.arange(len(hull.vertices))
        Fh = remap[hull.simplices]
        tri = Vh[Fh]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        outward = np.einsum("ij,ij->i", normals, tri[:, 0] - Vh.mean(0)) >= 0
        Fh[~outward] = Fh[~outward][:, ::-1]
        coll = _mesh(stage, f"/knife/base_link/collisions/{name}", Vh, Fh, purpose=UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(coll.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(coll.GetPrim()).CreateApproximationAttr("convexHull")
        every.append(V)
    every = np.concatenate(every)
    root.GetPrim().CreateAttribute("ig:nativeBB", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3f(*(every.max(0) - every.min(0)).tolist()))
    stage.GetRootLayer().Save()
    return str(path.resolve())


def collider_hulls(usd_path):
    """Points of each collision mesh of a one-link asset, in its link's frame (needs ``pxr``)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    hulls = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            link = prim.GetParent().GetParent()  # <link>/collisions/<mesh>
            M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            L = np.array(UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            T = np.linalg.inv(L) @ M
            pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            hulls.append(pts @ T[:3, :3].T + T[:3, 3])
    return hulls


def hull_top(hulls, xy):
    """Height of the top of a union of convex hulls above each (x, y), NaN where none is."""
    from scipy.spatial import ConvexHull

    xy = np.atleast_2d(np.asarray(xy, float))
    top = np.full(len(xy), -np.inf)
    for pts in hulls:
        eq = ConvexHull(pts).equations  # n . p + d <= 0 inside
        up, down, side = eq[eq[:, 2] > 1e-9], eq[eq[:, 2] < -1e-9], eq[np.abs(eq[:, 2]) <= 1e-9]
        z_top = (-(up[:, 3][None, :] + xy @ up[:, :2].T) / up[:, 2][None, :]).min(axis=1)
        z_bot = (-(down[:, 3][None, :] + xy @ down[:, :2].T) / down[:, 2][None, :]).max(axis=1)
        inside = z_top >= z_bot
        if len(side):
            inside &= np.all(xy @ side[:, :2].T + side[:, 3][None, :] <= 1e-12, axis=1)
        top = np.where(inside, np.maximum(top, z_top), top)
    return np.where(np.isfinite(top), top, np.nan)


def rigid_body_pose(prim_path):
    """World (position, xyzw quaternion) of a rigid body as PhysX has it now."""
    import omnigibson.lazy as lazy

    t = lazy.omni.physx.get_physx_interface().get_rigidbody_transformation(prim_path)
    if not t.get("ret_val"):
        raise RuntimeError(f"PhysX has no rigid body at {prim_path}")
    return np.asarray(t["position"], dtype=float), np.asarray(t["rotation"], dtype=float)


class PhysicsStepCoupler:
    """Runs a ``cosim.KnifeCoupling`` in PhysX's step: food after each step, its wrench before the next.

    Args:
        coupling: The coupling.
        knife_body: Prim path of the knife's rigid body.
        apply_wrench: Apply the food's wrench to the knife; False only measures it (a control run).
        order: Order of the step callbacks among PhysX's (OmniGibson's own run at 0).
    """

    def __init__(self, coupling, knife_body, apply_wrench=True, order=10):
        import omnigibson.lazy as lazy

        self.coupling = coupling
        self.knife_body = str(knife_body)
        self.apply_wrench = apply_wrench
        self.order = order
        self._body_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(lazy.pxr.Sdf.Path(self.knife_body))
        self._subs = []
        self.wrench = None  # (force, point, moment), world frame, for the next physics step
        self.begin_cut = None  # passed to KnifeCoupling.step on the next physics step, then reset
        self.records = []
        self.physics_steps = 0

    def start(self):
        import omnigibson as og

        pi = og.sim.pi
        self._subs = [
            pi.subscribe_physics_on_step_events(self._pre_step, pre_step=True, order=self.order),
            pi.subscribe_physics_on_step_events(self._post_step, pre_step=False, order=self.order),
        ]

    def stop(self):
        self._subs = []
        self.wrench = None

    def _pre_step(self, dt):
        if self.wrench is None or not self.apply_wrench:
            return
        import omnigibson as og

        force, point, moment = self.wrench
        og.sim.psi.apply_force_at_pos(og.sim.stage_id, self._body_id, tuple(map(float, force)),
                                      tuple(map(float, point)))
        og.sim.psi.apply_torque(og.sim.stage_id, self._body_id, tuple(map(float, moment)))

    def _post_step(self, dt):
        import omnigibson as og

        pos, quat = rigid_body_pose(self.knife_body)
        begin, self.begin_cut = self.begin_cut, None
        step = self.coupling.step(float(dt), pos, quat, begin_cut=begin)
        self.physics_steps += 1
        self.wrench = (step.force, step.point, step.moment) if step.advanced else None
        self.records.append(dict(
            t=float(og.sim.current_time), solver_t=float(self.coupling.sim.time), pos=pos.tolist(), quat=quat.tolist(),
            force=step.force.tolist(), moment=step.moment.tolist(), advanced=step.advanced, cut=step.cut,
            wall_s=step.wall_s, applied=bool(self.apply_wrench and step.advanced),
        ))


class FoodPoints:
    """The food as a USD points prim: a sphere per particle, coloured by where it started.

    Particles that started on the food's skin take ``skin_rgb``, the rest ``cut_rgb``, which shows
    wherever a cut exposes them. Colours are sRGB, given to the renderer linear.

    Args:
        stage: The USD stage.
        path: Prim path of the points prim.
        skin: (N,) bool, whether each particle started on the skin.
        width: Sphere diameter [m]; a little over the particle spacing closes the surface.
        skin_rgb, cut_rgb: Colours.
        albedo: Optional (N, 3) sRGB colour per particle instead.
    """

    def __init__(self, stage, path, skin, width, skin_rgb, cut_rgb, albedo=None):
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        self.points = UsdGeom.Points.Define(stage, path)
        n = len(skin)
        self.points.CreateWidthsAttr(Vt.FloatArray([float(width)]))
        self.points.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        if albedo is None:
            albedo = np.where(np.asarray(skin, bool)[:, None], np.asarray(skin_rgb)[None, :],
                              np.asarray(cut_rgb)[None, :])
        colors = srgb_to_linear(albedo).astype(np.float32)
        primvar = UsdGeom.PrimvarsAPI(self.points.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        primvar.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(colors)))
        # A preview surface reading the colour primvar.
        mat = UsdShade.Material.Define(stage, f"{path}_look")
        shader = UsdShade.Shader.Define(stage, f"{path}_look/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        reader = UsdShade.Shader.Define(stage, f"{path}_look/color")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
        result = reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(result)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        surface = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(surface)
        UsdShade.MaterialBindingAPI.Apply(self.points.GetPrim()).Bind(mat)
        self._gf = Gf
        self._vt = Vt
        self.n = n

    def update(self, x_world):
        x = np.ascontiguousarray(np.asarray(x_world, dtype=np.float32))
        self.points.GetPointsAttr().Set(self._vt.Vec3fArray.FromNumpy(x))
        lo, hi = x.min(axis=0), x.max(axis=0)
        extent = [self._gf.Vec3f(*lo.tolist()), self._gf.Vec3f(*hi.tolist())]
        self.points.GetExtentAttr().Set(self._vt.Vec3fArray(extent))


def knife_yaw_pitch_roll_deg(quat_xyzw):
    """Yaw, pitch and roll (degrees) of a knife body, as ``KnifePose`` counts them, for logs."""
    from simfoundry.cutting.cosim import pose_from_axes, quat_xyzw_to_matrix

    R = quat_xyzw_to_matrix(quat_xyzw)
    p = pose_from_axes(np.zeros(3), R[:, 0], R[:, 1], R[:, 2])
    return math.degrees(p.yaw), math.degrees(p.pitch), math.degrees(p.roll)
