# SPDX-License-Identifier: Apache-2.0
"""The MPM food coupled to a rigid knife that a robot simulator moves.

The robot simulator (OmniGibson's PhysX, see ``og_cosim``) owns the knife. After every physics
step it reports where the knife's rigid body is, and ``KnifeCoupling.step`` advances the food over
that step with the knife moving in a straight line between the body's last two poses
(``KnifeTrajectory.linear``). PhysX moves a body along exactly that line within a step
(semi-implicit Euler), so the solver sees the path PhysX took. The step returns the force and
moment of the food on the knife, which the simulator applies to the body over its next step: an
explicit coupling, the wrench one physics step late.

The solver works in its own frame, an axis-aligned box on a board at a height of its own
(``MPMCuttingSim``); ``sim_from_world`` places it in the world, as
``real2sim.FoodPlacement.sim_from_world`` does for food that Pipeline A reconstructed.

Two things the robot simulator would otherwise have to know about the solver are done here:

* **When a stroke starts a cut.** ``MPMCuttingSim.begin_cut`` splits the food by the blade's plane
  once per stroke. By default a cut begins when the edge, coming down over the food, gets within
  ``cut_height`` of its top; the next one only after the edge has risen ``rearm_height`` above
  it. A caller that knows its strokes can say so instead (``begin_cut`` of ``step``).
* **When the solver can rest.** While the knife is away from the solver's box, nothing it does
  reaches the food, so the solver idles once the food has had ``settle`` to settle under its
  weight and ``linger`` to come to rest after the knife left. An idle step costs nothing and
  returns no wrench; the solver's clock then stops, and the viscoelastic food relaxes no further.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from simfoundry.cutting.knife import KnifePose, KnifeTrajectory


def quat_xyzw_to_matrix(q) -> np.ndarray:
    """3x3 rotation of a unit quaternion given as (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_xyzw(R) -> np.ndarray:
    """Unit quaternion (x, y, z, w) of a 3x3 rotation, w >= 0."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def pose_from_axes(origin, e_u, e_v, e_w) -> KnifePose:
    """The ``KnifePose`` whose axes are (e_u, e_v, e_w): yaw, pitch and roll as ``knife._axes`` composes them.

    e_u is the edge (towards the tip), e_v the blade's normal and e_w from the edge towards the
    spine, a right-handed frame. A blade pitched to the vertical (e_u straight up or down) has no
    yaw; the solver's strokes never get there.
    """
    e_u = np.asarray(e_u, dtype=float)
    e_v = np.asarray(e_v, dtype=float)
    yaw = math.atan2(e_u[1], e_u[0])
    pitch = math.atan2(-e_u[2], math.hypot(e_u[0], e_u[1]))
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    # Roll turns e_v from its level direction towards the pitched e_w (knife._axes).
    v0 = np.array([-sy, cy, 0.0])
    w1 = np.array([sp * cy, sp * sy, cp])
    roll = math.atan2(float(e_v @ w1), float(e_v @ v0))
    return KnifePose(origin=np.asarray(origin, dtype=float), yaw=yaw, pitch=pitch, roll=roll)


def _unwrap(angle, near):
    return near + (angle - near + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class CouplingStep:
    """What one physics step of the coupling gives the robot simulator.

    Attributes:
        force: Mean force of the food on the knife over the step, world frame [N].
        moment: Mean moment of the food on the knife about ``point``, world frame [N m].
        point: World position of the knife frame's origin (the edge midpoint) at the step's end.
        advanced: Whether the solver ran this step; an idle step has no wrench.
        cut: Index of the cut this step began, or None.
        wall_s: Wall time of the solver's step [s].
        substeps: Solver time steps taken.
    """

    force: np.ndarray
    moment: np.ndarray
    point: np.ndarray
    advanced: bool
    cut: int = None
    wall_s: float = 0.0
    substeps: int = 0


@dataclass
class KnifeCoupling:
    """Advances an ``MPMCuttingSim`` along a knife that another simulator moves.

    Args:
        sim: The solver, its food at rest on its board.
        sim_from_world: 4x4 rigid map from the world into the solver's frame.
        knife_in_link: 4x4 pose of the solver's knife frame (origin at the edge's midpoint, x along
            the edge to the tip, z towards the spine) in the frame of the rigid body that the robot
            simulator reports; identity for a knife asset built in that frame (``og_cosim``).
        auto_cut: Begin cuts by the rule in the module docstring when ``step`` is not told.
        cut_height: The edge, over the food and coming down, begins a cut this close to its top [m].
        rearm_height: After a cut began, the edge must rise this far above the food's top before
            it can begin another [m].
        wake_margin: The solver runs while any part of the blade is within this distance of its
            box [m].
        settle: Solver time the food first spends settling under its weight, knife or not [s].
        linger: Solver time the food goes on moving after the knife left its box [s].
    """

    sim: object
    sim_from_world: np.ndarray
    knife_in_link: np.ndarray = None
    auto_cut: bool = True
    cut_height: float = 3.0e-3
    rearm_height: float = 0.01
    wake_margin: float = 0.01
    settle: float = 0.3
    linger: float = 1.0
    log: list = field(default_factory=list)

    def __post_init__(self):
        S = np.asarray(self.sim_from_world, dtype=float)
        self.R = S[:3, :3]
        self.t = S[:3, 3]
        self.knife_in_link = np.eye(4) if self.knife_in_link is None else np.asarray(self.knife_in_link, float)
        g = self.sim.knife_geometry
        # The blade's outline in the knife frame, for the solver box test: edge and spine along its length.
        u = np.linspace(-g.half_length, g.half_length, 21)
        self._outline = np.concatenate([
            np.stack([u, np.zeros_like(u), g.edge_height(u)], axis=1),
            np.stack([u, np.zeros_like(u), g.spine_height(u)], axis=1),
        ])
        self._edge_u = np.linspace(-g.half_length, g.half_length, 81)
        self._edge = np.stack([self._edge_u, np.zeros_like(self._edge_u), g.edge_height(self._edge_u)], axis=1)
        self.prev = None  # the knife's last pose, solver frame
        self.armed = True
        self.awake_until = self.settle  # solver time
        self.food_top = float(self.sim.block_max[2])
        self.food_lo = np.asarray(self.sim.block_min, dtype=float)
        self.food_hi = np.asarray(self.sim.block_max, dtype=float)
        self.box_hi = np.asarray(self.sim.domain_size, dtype=float)
        self.steps = 0
        self.idle_steps = 0

    # -- frames ------------------------------------------------------------------------

    def knife_pose(self, link_pos, link_quat_xyzw) -> KnifePose:
        """The solver-frame knife pose of a rigid body at this world pose."""
        L = np.eye(4)
        L[:3, :3] = quat_xyzw_to_matrix(link_quat_xyzw)
        L[:3, 3] = np.asarray(link_pos, dtype=float)
        K = L @ self.knife_in_link
        R = self.R @ K[:3, :3]
        pose = pose_from_axes(self.R @ K[:3, 3] + self.t, R[:, 0], R[:, 1], R[:, 2])
        if self.prev is not None:
            # Keep the angles continuous, or linear interpolation would spin the blade round.
            pose = KnifePose(pose.origin, _unwrap(pose.yaw, self.prev.yaw), pose.pitch,
                             _unwrap(pose.roll, self.prev.roll))
        return pose

    def to_world_point(self, p):
        return self.R.T @ (np.asarray(p, dtype=float) - self.t)

    def to_world_vector(self, v):
        return self.R.T @ np.asarray(v, dtype=float)

    def world_from_sim(self) -> np.ndarray:
        """4x4 map from the solver's frame to the world (positions of particles, say)."""
        W = np.eye(4)
        W[:3, :3] = self.R.T
        W[:3, 3] = -self.R.T @ self.t
        return W

    # -- stepping ----------------------------------------------------------------------

    def _points(self, pose, local):
        e_u, e_v, e_w = pose.axes()
        return np.asarray(pose.origin)[None, :] + local[:, :1] * e_u + local[:, 1:2] * e_v + local[:, 2:3] * e_w

    def _near(self, pose) -> bool:
        p = self._points(pose, self._outline)
        m = self.wake_margin
        return bool(np.all(p.max(axis=0) >= -m) and np.all(p.min(axis=0) <= self.box_hi + m))

    def _edge_over_food(self, pose):
        """Heights of the edge points over the food's footprint (empty if none is over it)."""
        p = self._points(pose, self._edge)
        over = np.all((p[:, :2] >= self.food_lo[:2]) & (p[:, :2] <= self.food_hi[:2]), axis=1)
        return p[over, 2], over

    def _auto_begin(self, start: KnifePose, end: KnifePose) -> bool:
        z, over = self._edge_over_food(end)
        if not self.armed:
            if z.size == 0 or z.min() > self.food_top + self.rearm_height:
                self.armed = True
            return False
        if z.size == 0 or z.min() > self.food_top + self.cut_height:
            return False
        if abs(end.axes()[1][2]) > 0.9:
            return False  # a blade lying flat cuts no plane
        z0 = self._points(start, self._edge)[over, 2]
        return bool(np.mean(z - z0) < 0.0)

    def step(self, dt: float, link_pos, link_quat_xyzw, begin_cut=None) -> CouplingStep:
        """Advances the food over one physics step that ended with the knife's body at this pose.

        Args:
            dt: Length of the physics step [s].
            link_pos: World position of the knife's rigid body at the step's end.
            link_quat_xyzw: Its world orientation (x, y, z, w).
            begin_cut: True to begin a cut with this step, False not to; None leaves it to the
                rule in the module docstring (if ``auto_cut``).

        Returns:
            The step's ``CouplingStep``.
        """
        sim = self.sim
        end = self.knife_pose(link_pos, link_quat_xyzw)
        start = end if self.prev is None else self.prev
        self.prev = end
        self.steps += 1
        point = self.to_world_point(end.origin)
        if begin_cut is None:
            begin_cut = self.auto_cut and self._auto_begin(start, end)
        if begin_cut:
            self.armed = False
        near = self._near(start) or self._near(end)
        if near:
            self.awake_until = max(self.awake_until, sim.time + dt + self.linger)
        if not (near or begin_cut or sim.time < self.awake_until):
            self.idle_steps += 1
            sim.set_knife(end)
            return CouplingStep(np.zeros(3), np.zeros(3), point, advanced=False)
        cuts = sim.n_cuts
        t0 = time.perf_counter()
        n0 = sim.time
        traj = KnifeTrajectory.linear(sim.time, sim.time + dt, start, end, begins_cut=bool(begin_cut))
        try:
            force, moment = sim.advance(sim.time + dt, traj)
        except RuntimeError as e:  # too many cuts or pieces: go on without this cut
            self.log.append(dict(time=n0, error=str(e)))
            traj = KnifeTrajectory.linear(sim.time, sim.time + dt, start, end)
            force, moment = sim.advance(sim.time + dt, traj)
        wall = time.perf_counter() - t0
        cut = sim.n_cuts - 1 if sim.n_cuts > cuts else None
        if cut is not None:
            self.log.append(dict(time=n0, cut=cut, origin=[float(v) for v in start.origin]))
        return CouplingStep(
            force=self.to_world_vector(force), moment=self.to_world_vector(moment), point=point,
            advanced=True, cut=cut, wall_s=wall, substeps=int(math.ceil(dt / sim.dt - 1e-9)),
        )
