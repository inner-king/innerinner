# SPDX-License-Identifier: Apache-2.0
"""
A knife that cuts where it is: a transition rule that replaces an object by two
rigid pieces, cut by the plane of the blade at the moment the blade presses on it.

OmniGibson's own ``SlicingRule`` swaps a ``sliceable`` object for two
pre-authored halves from the BEHAVIOR-1K dataset whenever any part of a
``slicer`` touches it. SimFoundry's objects are ``USDObject``s with categories
outside the BEHAVIOR taxonomy and have no halves, and a cut should happen where
the knife actually is. This rule does that instead:

- **Who.** Objects opt in through an optional ``cutting`` entry in their init
  info args (``objects_info.init_info[name].args.cutting``), which OmniGibson
  passes to the constructor's ``**kwargs``, records in the object's init info and
  writes back on every save, like ``args.mass``. ``{"role": "knife", "blade":
  "blade"}`` marks a knife whose blade is the collision mesh (or link) named
  ``blade``; ``{"role": "cuttable"}`` marks an object it may cut. A cuttable's
  entry may override ``min_force_n``, ``max_hulls``, ``min_piece_fraction``,
  ``min_piece_volume_m3`` and ``interior_color`` for itself, a knife's
  ``min_force_n`` and ``rearm_s`` (CuttingSettings).
- **When.** Only the blade counts. PhysX's contact report gives every contact's
  collider, so a contact with the handle is never mistaken for one with the
  blade, whether the blade is a link of its own or a collider of the knife's one
  link. The rule takes, over the physics steps of each simulator step, the largest
  normal force the blade exerts on a cuttable (summed normal impulse / physics
  dt), and cuts above ``min_force_n`` (10 N: iGibson 2.0's threshold on its
  ``slicer`` link). After a cut the knife is disarmed until its blade has touched
  no cuttable for ``rearm_s``, as OmniGibson's ``SlicerActive`` re-arms a slicer.
  PhysX reports only what it simulates: a kinematic knife moved by setting its
  pose does not wake a sleeping fruit, and passes through it unreported, so
  whatever drives one wakes the cuttables it moves toward.
- **Where.** The cut plane is the blade's mid-plane at that moment, taken from
  the knife's pose; the pieces are what geometry.cut_geometry() makes of the
  object's own visual meshes and convex colliders.
- **How the stroke gets through.** PhysX stops a rigid blade at a rigid fruit's
  surface, so the swap happens on the first qualifying contact, and the slab of
  the object the blade occupies is left out: the two pieces sit on either side
  of the blade, ``blade thickness / 2 + gap_margin_m`` from its mid-plane, and the
  blade finishes its stroke between them while they fall apart. The alternative,
  letting the blade pass through the uncut object by filtering their collision
  during the stroke, would need a runtime change to PhysX's filtered pairs per cut
  and leaves nothing to hold the object against the blade until the swap. See
  repro/README.md section 15.2.
- **The swap.** The pieces are written as new USD files (usd_io.write_piece),
  the parent is removed with ``og.sim.batch_remove_objects`` and the pieces added
  with one ``og.sim.batch_add_objects``. They are initialized at once, inside this
  step: OmniGibson would otherwise simulate them uninitialized through the next
  step's physics and then ``keep_still()`` them, discarding the parent's motion.
  They start at the parent's pose with the velocity of the parent's material at
  their centroids, and carry ``args.mass`` (split by collider volume),
  ``args.expected_file_hash`` (their file's MD5) and a ``cutting`` entry, so a
  piece can be cut again.

Import this module before the scene is created: ``BaseTransitionRule``
registers subclasses on definition, and a scene instantiates the rules
registered when it loads. Transition rules must be on
(``gm.ENABLE_TRANSITION_RULES``, which ``apply_teleop_omnigibson_macros`` sets
unless asked not to).
"""

from collections import OrderedDict
import copy
from dataclasses import dataclass, field, fields
import os
import time

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.objects.usd_object import USDObject
from omnigibson.transition_rules import BaseTransitionRule, ObjectCandidateFilter, RuleCondition, TransitionResults
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.ui_utils import create_module_logger

from simfoundry.cutting import geometry as geo
from simfoundry.cutting import usd_io

log = create_module_logger(module_name=__name__)

#: Init-info args key that marks knives and cuttable objects.
CUTTING_KEY = "cutting"

#: Assets whose geometry the rule keeps read (a cut piece's is kept from its cut, so cutting it again reads nothing).
SOURCE_CACHE_SIZE = 16


@dataclass
class CuttingSettings:
    """Defaults for every knife and cuttable object; a ``cutting`` entry may override the per-object ones.

    Attributes:
        pieces_dir (str): where piece USD files are written. Saved scenes point at them, so it should outlive the run
        min_force_n (float): normal force the blade must exert to cut (N)
        rearm_s (float): seconds the blade must touch no cuttable before it cuts again
        max_hulls (int): convex colliders per piece (OmniGibson's add costs 6-8 ms per collider prim)
        min_piece_fraction (float): smallest share of the object's collider volume a piece may get; a cut that
            would leave less is not made, which is also what keeps the flat of a blade from cutting
        min_piece_volume_m3 (float): smallest piece volume in the world, however often an object has been cut
        gap_margin_m (float): clearance between the blade's faces and each piece's cut face, on each side
        interior_color (tuple): RGB of the flat material on cut faces
        contact_separation_m (float): a blade contact this close counts as touching for re-arming
    """

    pieces_dir: str = os.environ.get("SIMFOUNDRY_CUT_PIECES_DIR",
                                     os.path.join(os.path.expanduser("~"), ".cache", "simfoundry", "cut_pieces"))
    min_force_n: float = 10.0
    rearm_s: float = 1.0
    max_hulls: int = 8
    min_piece_fraction: float = 0.03
    min_piece_volume_m3: float = 1e-6
    gap_margin_m: float = 5e-4
    interior_color: tuple = (0.93, 0.87, 0.72)
    contact_separation_m: float = 1e-3


SETTINGS = CuttingSettings()


def configure(**kwargs):
    """Change the defaults in SETTINGS, e.g. ``configure(pieces_dir=...)``."""
    names = {f.name for f in fields(CuttingSettings)}
    for key, value in kwargs.items():
        if key not in names:
            raise KeyError(f"unknown cutting setting {key}; known: {sorted(names)}")
        setattr(SETTINGS, key, value)


def cutting_spec(obj):
    """The object's ``cutting`` entry, or None."""
    spec = obj.get_init_info()["args"].get(CUTTING_KEY)
    return spec if isinstance(spec, dict) else None


def setting(name, *specs):
    """``name`` from the first spec that sets it, else from SETTINGS."""
    for spec in specs:
        if spec and name in spec:
            return spec[name]
    return getattr(SETTINGS, name)


def get_cutting_rule(scene):
    """The scene's PlaneCuttingRule instance (None if transition rules are off)."""
    api = scene.transition_rule_api
    if api is None:
        return None
    return next((r for r in api.all_rules if isinstance(r, PlaneCuttingRule)), None)


class CuttingRoleFilter(ObjectCandidateFilter):
    """Objects whose ``cutting`` entry has the given role."""

    def __init__(self, role):
        self.role = role

    def __call__(self, obj):
        spec = cutting_spec(obj)
        return spec is not None and spec.get("role") == self.role


def _path_id(path):
    return int(lazy.pxr.PhysicsSchemaTools.sdfPathToInt(path))


def _np(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64)


@dataclass
class Window:
    """What the blades did since the rule last looked.

    Attributes:
        peak (dict): (knife, cuttable) -> (largest normal force in one physics step in N, world contact point)
        touching (set): knives whose blade touched a cuttable
    """

    peak: dict = field(default_factory=dict)
    touching: set = field(default_factory=set)


class BladeContactMonitor:
    """Reads PhysX's contact report after every physics step, keeping only blade-on-cuttable contacts.

    OmniGibson's RigidContactAPI aggregates contacts per rigid body and keeps only the last physics step of a
    simulator step; this keeps the collider, so a knife's blade and handle are told apart even in one rigid body,
    and the largest force over all physics steps.
    """

    def __init__(self):
        self._subscription = None
        self._blades = {}  # collider path id -> knife
        self._bodies = {}  # rigid body path id -> cuttable
        self._window = Window()

    def track(self, blade_colliders, cuttable_bodies):
        """Watch these colliders and bodies from now on.

        Args:
            blade_colliders (dict): collider prim path -> knife object
            cuttable_bodies (dict): rigid body prim path -> cuttable object
        """
        self._blades = {_path_id(p): k for p, k in blade_colliders.items()}
        self._bodies = {_path_id(p): c for p, c in cuttable_bodies.items()}
        if self._subscription is None:
            self._subscription = lazy.omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
                self._on_contact)

    def take(self):
        """The window since the last call; a new one starts."""
        window, self._window = self._window, Window()
        return window

    def _on_contact(self, headers, data):
        if not self._blades or not self._bodies:
            return
        lost = lazy.omni.physx.bindings._physx.ContactEventType.CONTACT_LOST
        impulses = {}
        for h in headers:
            knife, cuttable = self._blades.get(h.collider0), self._bodies.get(h.actor1)
            if knife is None or cuttable is None:
                knife, cuttable = self._blades.get(h.collider1), self._bodies.get(h.actor0)
                if knife is None or cuttable is None:
                    continue
            if h.type == lost:
                continue
            total, point, near = 0.0, None, False
            for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                c = data[i]
                n, j = c.normal, c.impulse
                jn = abs(n[0] * j[0] + n[1] * j[1] + n[2] * j[2])
                total += jn
                near = near or c.separation <= SETTINGS.contact_separation_m
                if point is None or jn > 0:
                    point = (c.position[0], c.position[1], c.position[2])
            if total > 0 or near:
                self._window.touching.add(knife)
            key = (knife, cuttable)
            j0, p0 = impulses.get(key, (0.0, None))
            impulses[key] = (j0 + total, point if p0 is None else p0)
        if not impulses:
            return
        dt = og.sim.get_physics_dt()
        for key, (impulse, point) in impulses.items():
            force = impulse / dt
            if force > self._window.peak.get(key, (0.0, None))[0]:
                self._window.peak[key] = (force, point)


class BladeForceCondition(RuleCondition):
    """Keeps knife/cuttable pairs whose blade pressed hard enough, and updates each knife's re-arming."""

    def __init__(self, rule):
        self._rule = rule

    def __call__(self, object_candidates):
        rule = self._rule
        window = rule.last_window = rule.monitor.take()
        dt = og.sim.get_sim_step_dt()
        knives = set(object_candidates["knife"])
        cuttables = set(object_candidates["cuttable"])
        for knife in knives:
            state = rule.knife_state.setdefault(knife, {"armed": True, "clear_s": 0.0})
            if knife in window.touching:
                state["clear_s"] = 0.0
            else:
                state["clear_s"] += dt
                if not state["armed"] and state["clear_s"] >= setting("rearm_s", cutting_spec(knife)):
                    state["armed"] = True
        best = {}
        for (knife, cuttable), (force, point) in window.peak.items():
            if knife not in knives or cuttable not in cuttables or not rule.knife_state[knife]["armed"]:
                continue
            threshold = setting("min_force_n", cutting_spec(cuttable), cutting_spec(knife))
            rule.last_forces[(knife.name, cuttable.name)] = force
            if force >= threshold and force > best.get(cuttable, (None, 0.0))[1]:
                best[cuttable] = (knife, force, point)
        rule.pending = [(knife, cuttable, force, point) for cuttable, (knife, force, point) in best.items()]
        object_candidates["knife"] = list({k for k, _, _, _ in rule.pending})
        object_candidates["cuttable"] = [c for _, c, _, _ in rule.pending]
        return len(rule.pending) > 0

    @property
    def modifies_filter_names(self):
        return {"knife", "cuttable"}


class PlaneCuttingRule(BaseTransitionRule):
    """Cuts a cuttable object by a knife's blade plane when the blade presses on it. See the module docstring."""

    def __init__(self, scene):
        super().__init__(scene)
        self.monitor = BladeContactMonitor()
        #: knife -> {"armed": bool, "clear_s": seconds since the blade last touched a cuttable}
        self.knife_state = {}
        #: (knife, cuttable, force, point) pairs the condition passed this step
        self.pending = []
        #: (knife name, cuttable name) -> last peak force seen while the knife was armed (N), for diagnostics
        self.last_forces = {}
        #: the blade contacts of the last simulator step (a Window), whatever the knives' arming
        self.last_window = Window()
        #: one record per cut made or refused, in order (see _plan())
        self.cut_log = []
        self._blades = {}  # knife -> (blade link, Plane in its frame, thickness in its units)
        self._sources = OrderedDict()  # usd path -> (SourceGeometry, SourceInfo), most recently used last
        self._counters = {}  # source name -> next piece number

    @classproperty
    def candidate_filters(cls):
        return {"knife": CuttingRoleFilter("knife"), "cuttable": CuttingRoleFilter("cuttable")}

    def _generate_conditions(self):
        return [BladeForceCondition(self)]

    def refresh(self, object_candidates):
        super().refresh(object_candidates)
        blade_colliders, bodies = {}, {}
        for knife in object_candidates["knife"]:
            for path in self._blade(knife)[3]:
                blade_colliders[path] = knife
        for cuttable in object_candidates["cuttable"]:
            for link in cuttable.links.values():
                bodies[link.prim_path] = cuttable
            # Read the asset now, when the object enters the scene, rather than inside the step that first cuts it
            # (the apple's 290k-triangle mesh takes ~0.1 s); a piece's geometry is kept from its own cut. Not past
            # the cache's size, or every refresh (each cut, each reset) would read assets again.
            if len(self._sources) < SOURCE_CACHE_SIZE:
                try:
                    self._source(cuttable)
                except usd_io.UncuttableAsset as e:
                    log.warning(f"{cuttable.name} is marked cuttable but cannot be cut: {e}")
        self.monitor.track(blade_colliders, bodies)
        for knife in list(self.knife_state):
            if knife not in object_candidates["knife"]:
                del self.knife_state[knife]

    # ------------------------------------------------------------------------------------------------ the blade
    def _blade(self, knife):
        """(blade link, blade mid-plane in the link frame, thickness in link units, blade collider prim paths)."""
        if knife not in self._blades:
            name = (cutting_spec(knife) or {}).get("blade", "blade")
            stage = og.sim.stage
            cache = lazy.pxr.UsdGeom.XformCache()
            found = []
            for link_name, link in knife.links.items():
                for mesh in link.collision_meshes.values():
                    prim_name = mesh.prim_path.rsplit("/", 1)[-1]
                    if link_name != name and not prim_name.startswith(name):
                        continue
                    prim = stage.GetPrimAtPath(mesh.prim_path)
                    M = np.array(cache.ComputeRelativeTransform(prim, stage.GetPrimAtPath(link.prim_path))[0])
                    pts = np.asarray(prim.GetAttribute("points").Get(), np.float64)
                    found.append((link, pts @ M[:3, :3] + M[3, :3], mesh.prim_path))
            if not found:
                raise ValueError(f"knife {knife.name}: no collider or link named {name!r} for the blade")
            links = {id(f[0]) for f in found}
            if len(links) != 1:
                raise ValueError(f"knife {knife.name}: blade colliders {name!r} span {len(links)} links")
            plane, thickness = geo.blade_plane_from_points(np.concatenate([f[1] for f in found]))
            self._blades[knife] = (found[0][0], plane, thickness, [f[2] for f in found])
        return self._blades[knife]

    def blade_plane(self, knife):
        """The blade's mid-plane in the world and the blade's thickness (m), from the knife's current pose."""
        link, plane, thickness, _ = self._blade(knife)
        pos, quat = link.get_position_orientation()
        world, q = geo.plane_to_world(plane, _np(pos), _np(quat), _np(knife.scale))
        return world, thickness / q

    # ------------------------------------------------------------------------------------------------ the cut
    def _source(self, obj):
        path = os.path.abspath(obj._usd_path)
        if getattr(obj, "_encrypted", False):
            raise usd_io.UncuttableAsset(f"{obj.name}: encrypted assets cannot be read for cutting")
        if path not in self._sources:
            self._remember(path, usd_io.read_source(path))
        self._sources.move_to_end(path)
        return self._sources[path]

    def _remember(self, path, source):
        # A source holds a full visual mesh (~15 MB for a scanned fruit); keep the most recent few
        self._sources[path] = source
        while len(self._sources) > SOURCE_CACHE_SIZE:
            self._sources.popitem(last=False)

    def transition(self, object_candidates):
        t_start = time.perf_counter()
        plans = []
        for knife, cuttable, force, point in self.pending:
            self.knife_state[knife]["armed"] = False  # refused or failed too: no retrying every step of the stroke
            self.knife_state[knife]["clear_s"] = 0.0
            try:
                plan = self._plan(knife, cuttable, force, point)
            except Exception as e:  # noqa: BLE001 - a cut that cannot be planned must not stop the simulation
                log.error(f"cutting {cuttable.name} failed: {type(e).__name__}: {e}", exc_info=True)
                self.cut_log.append({"knife": knife.name, "cuttable": cuttable.name, "force_n": float(force),
                                     "refused": f"error: {type(e).__name__}: {e}"})
                plan = None
            if plan is not None:
                plans.append(plan)
        self.pending = []
        if not plans:
            return TransitionResults(add=[], remove=[])

        t0 = time.perf_counter()
        og.sim.batch_remove_objects([p["parent"] for p in plans])
        t1 = time.perf_counter()
        pieces = [USDObject(**args) for p in plans for args in p["args"]]
        og.sim.batch_add_objects(pieces, [self.scene] * len(pieces))
        t2 = time.perf_counter()
        self._initialize_now(pieces)
        t3 = time.perf_counter()
        it = iter(pieces)
        for plan in plans:
            for spec in plan["placement"]:
                piece = next(it)
                piece.set_position_orientation(position=spec["pos"], orientation=spec["quat"])
                piece.set_linear_velocity(th.tensor(spec["lin_vel"], dtype=th.float32))
                piece.set_angular_velocity(th.tensor(spec["ang_vel"], dtype=th.float32))
        t4 = time.perf_counter()
        self.monitor.take()  # the swap stepped physics once; nothing in that window is a stroke
        swap = {"remove_s": t1 - t0, "add_s": t2 - t1, "initialize_s": t3 - t2, "place_s": t4 - t3}
        for plan in plans:
            plan["record"]["timings"].update(swap)
            plan["record"]["timings"]["rule_step_s"] = time.perf_counter() - t_start
            plan["record"]["n_cuts_in_step"] = len(plans)
            self.cut_log.append(plan["record"])
            log.info(f"cut {plan['record']['cuttable']} with {plan['record']['knife']} at "
                     f"{plan['record']['force_n']:.1f} N into {plan['record']['pieces']}")
        return TransitionResults(add=[], remove=[])

    def _plan(self, knife, cuttable, force, point):
        """Everything the swap needs for one cut, with the piece files written; None if the cut is refused."""
        timings = {}
        spec = cutting_spec(cuttable) or {}
        record = {"knife": knife.name, "cuttable": cuttable.name, "force_n": float(force),
                  "contact_point": None if point is None else list(point), "sim_time_s": float(og.sim.current_time),
                  "timings": timings}
        t = time.perf_counter()
        plane_w, thickness = self.blade_plane(knife)
        pos, quat = (_np(x) for x in cuttable.get_position_orientation())
        scale = _np(cuttable.scale)
        plane_l, k = geo.plane_to_local(plane_w, pos, quat, scale)
        half_gap = (thickness / 2 + SETTINGS.gap_margin_m) / k
        record.update(plane_origin=plane_w.origin.tolist(), plane_normal=plane_w.normal.tolist(),
                      blade_thickness_m=float(thickness), half_gap_m=float(half_gap * k), parent_pos=pos.tolist(),
                      parent_quat=quat.tolist(), parent_scale=scale.tolist())
        timings["plane_s"] = time.perf_counter() - t
        try:
            t = time.perf_counter()
            source, info = self._source(cuttable)
            timings["read_source_s"] = time.perf_counter() - t
            result = geo.cut_geometry(source, plane_l, half_gap, max_hulls=setting("max_hulls", spec),
                                      min_fraction=setting("min_piece_fraction", spec))
        except usd_io.UncuttableAsset as e:
            record["refused"] = str(e)
            self.cut_log.append(record)
            log.warning(f"not cutting {cuttable.name}: {e}")
            return None
        timings.update(result.timings)
        record["fractions"] = [float(f) for f in result.fractions]
        volume_scale = float(np.prod(np.abs(scale)))
        if result.reason is None and min(p.volume for p in result.pieces) * volume_scale < setting(
                "min_piece_volume_m3", spec):
            result.reason = f"a piece would be smaller than {setting('min_piece_volume_m3', spec)} m^3"
        if result.reason is not None:
            record["refused"] = result.reason
            self.cut_log.append(record)
            return None

        # Mass and motion
        masses = geo.split_mass(cuttable.mass, [p.volume for p in result.pieces])
        R = geo.quat_xyzw_to_matrix(quat)
        lin = _np(cuttable.get_linear_velocity())
        ang = _np(cuttable.get_angular_velocity())
        com_parent = pos + R @ (scale * source.whole[1])

        # Names, files, init info
        parent_args = cuttable.get_init_info()["args"]
        root = spec.get("source", cuttable.name)
        root_category = spec.get("source_category", cuttable.category)
        generation = int(spec.get("generation", 0)) + 1
        t = time.perf_counter()
        args_list, placement, names = [], [], []
        for piece, mass in zip(result.pieces, masses):
            name = self._piece_name(root)
            path, digest, materials = usd_io.write_piece(
                SETTINGS.pieces_dir, name, piece, info, category=f"half_{root_category}", mass=float(mass),
                interior_color=setting("interior_color", spec), generation=generation)
            self._remember(path, usd_io.piece_source(piece, path, info, generation, materials))
            piece_spec = {k: copy.deepcopy(v) for k, v in spec.items()}
            piece_spec.update(role="cuttable", source=root, source_category=root_category, parent=cuttable.name,
                              generation=generation)
            args = {
                "name": name, "usd_path": path, "category": f"half_{root_category}",
                "scale": parent_args.get("scale"), "visible": parent_args.get("visible", True),
                "visual_only": parent_args.get("visual_only", False),
                "self_collisions": parent_args.get("self_collisions", False),
                "include_default_states": parent_args.get("include_default_states", True),
                # A category's taxonomy abilities assume the dataset's own assets; a piece has only what was asked for
                "abilities": copy.deepcopy(parent_args.get("abilities")) or {},
                "expected_file_hash": digest, "mass": float(mass), CUTTING_KEY: piece_spec,
            }
            materials_arg = parent_args.get("link_physics_materials")
            if materials_arg:
                args["link_physics_materials"] = copy.deepcopy(materials_arg)
            args_list.append(args)
            com = pos + R @ (scale * piece.centroid)
            placement.append({"pos": pos.tolist(), "quat": quat.tolist(),
                              "lin_vel": geo.velocity_at(lin, ang, com_parent, com).tolist(), "ang_vel": ang.tolist()})
            names.append(name)
            record.setdefault("piece_files", []).append({"name": name, "usd_path": path, "md5": digest,
                                                         "bytes": os.path.getsize(path)})
        timings["write_s"] = time.perf_counter() - t
        record.update(pieces=names, masses=[float(m) for m in masses], parent_mass=float(cuttable.mass),
                      parent_lin_vel=lin.tolist(), parent_ang_vel=ang.tolist(), generation=generation,
                      collider_prims=[len(p.hulls) for p in result.pieces],
                      visual_faces=[int(sum(len(v.faces) for v in p.visuals)) for p in result.pieces],
                      piece_volumes_m3=[p.volume * volume_scale for p in result.pieces])
        return {"parent": cuttable, "args": args_list, "placement": placement, "record": record}

    def _piece_name(self, root):
        n = self._counters.get(root, 0)
        while self.scene.object_registry("name", f"half_{root}_{n}") is not None:
            n += 1
        self._counters[root] = n + 1
        return f"half_{root}_{n}"

    def _initialize_now(self, objs):
        """Initialize objects added in this step now, as Simulator._non_physics_step() would at the next one.

        Left to OmniGibson, they would go through the next step's physics uninitialized and then be keep_still()ed,
        losing the velocity they were given; and og.sim.save() refuses to save while objects wait to be initialized.
        This repeats that method's initialization block (simulator.py, OmniGibson 3.8.0).
        """
        sim = og.sim
        sim.pi.update_simulation(elapsedStep=0, currentTime=sim.current_time)
        for obj in objs:
            obj.initialize()
            sim._objects_to_initialize.remove(obj)
            if len(obj.states.keys() & sim.object_state_types_on_contact) > 0:
                sim._objects_require_contact_callback = True
            if len(obj.states.keys() & sim.object_state_types_on_joint_break) > 0:
                sim._objects_require_joint_break_callback = True
        sim.update_handles()
        if gm.ENABLE_TRANSITION_RULES:
            self.scene.transition_rule_api.refresh_all_rules()
