# SPDX-License-Identifier: Apache-2.0
"""Kitchen-knife geometry, pose and scripted cutting strokes.

The knife frame has its origin at the midpoint of the cutting edge's straight part:

    e_u  along the edge (blade length), towards the tip
    e_w  from the edge towards the spine
    e_v  = e_w x e_u, the blade normal (the thickness direction)

A chef's knife's edge is straight from the heel and curves up towards the tip (the belly), and
its spine curves down to meet it at the point; ``KnifeGeometry.edge_height`` and
``spine_height`` give both, and the solver and the renderer use the same profile.

A blade held vertically with yaw 0 cuts along the plane y = const. Pitch rotates the blade
within its own plane (a rocking cut: the tip goes down), which leaves the cut plane where it
is; roll tilts the plane about the edge (twisting a cut slab off the block).
"""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class KnifeGeometry:
    """Blade shape in metres.

    Attributes:
        half_length: Half the edge length. The blade must be longer than the food is
            wide along the edge, or the crack front stops at the blade's ends.
        height: Edge-to-spine height of the blade.
        edge_thickness: Thickness at the edge. Far below what the grid resolves; the
            solver dilates the blade where the material is still intact (see mpm.py).
        spine_thickness: Thickness at the spine.
        bevel_height: Height over which the blade thickens from edge to spine.
        handle_length: Length of the handle, for rendering only.
        tip_rise: How far the tip stands above the edge's straight part (the belly) [m];
            0 keeps the edge straight to the tip.
        belly_length: Length over which the edge curves up to the tip [m].
        spine_taper: Length over which the spine curves down to the tip [m]; None takes
            38 % of the blade, as on a chef's knife.
    """

    half_length: float = 0.10
    height: float = 0.045
    edge_thickness: float = 0.3e-3
    spine_thickness: float = 2.0e-3
    bevel_height: float = 0.02
    handle_length: float = 0.11
    tip_rise: float = 0.0
    belly_length: float = 0.0
    spine_taper: float = None

    @property
    def taper(self) -> float:
        """Length over which the spine curves down to the tip [m]."""
        return 0.38 * 2.0 * self.half_length if self.spine_taper is None else self.spine_taper

    def edge_height(self, u):
        """Height of the edge above its straight part at ``u`` along the blade [m]."""
        u = np.asarray(u, dtype=float)
        if self.belly_length <= 0.0 or self.tip_rise <= 0.0:
            return np.zeros_like(u)
        s = np.clip((u - (self.half_length - self.belly_length)) / self.belly_length, 0.0, 1.0)
        return self.tip_rise * s * s

    def spine_height(self, u):
        """Height of the spine above the edge's straight part at ``u`` [m]."""
        u = np.asarray(u, dtype=float)
        if self.taper <= 0.0:
            return np.full_like(u, self.height)
        s = np.clip((u - (self.half_length - self.taper)) / self.taper, 0.0, 1.0)
        return self.tip_rise + (self.height - self.tip_rise) * (1.0 - s * s)

    def lowest_edge_point(self, pose: "KnifePose", samples: int = 400) -> np.ndarray:
        """World position of the edge's lowest point for a pose."""
        e_u, _, e_w = pose.axes()
        u = np.linspace(-self.half_length, self.half_length, samples)
        pts = np.asarray(pose.origin, dtype=float)[None, :] + u[:, None] * e_u + self.edge_height(u)[:, None] * e_w
        return pts[np.argmin(pts[:, 2])]


# The Chef'd cook's knife, as the clip shows it held flat: a 20 cm chef's knife, the blade at the
# heel 4.6 cm tall, the edge curving up over its last 40 % to a tip 0.43 blade heights above the
# straight part, the spine curving down over 37 %.
CHEF_KNIFE = KnifeGeometry(half_length=0.10, height=0.046, spine_thickness=2.2e-3, bevel_height=0.02,
                           tip_rise=0.020, belly_length=0.08, spine_taper=0.075)


@dataclass(frozen=True)
class KnifePose:
    """Rigid pose of a knife.

    Attributes:
        origin: World position of the edge midpoint, shape (3,).
        yaw: Rotation of e_u about +z [rad].
        pitch: Rotation within the blade's plane, about the blade normal [rad]; positive
            lowers the tip.
        roll: Rotation of the blade's plane about the edge [rad]; positive moves the spine
            towards -e_v.
    """

    origin: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def axes(self):
        """Returns (e_u, e_v, e_w) as unit vectors in the world frame."""
        return _axes(self.yaw, self.pitch, self.roll)


def _axes(yaw, pitch, roll):
    e_u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    e_w = np.array([0.0, 0.0, 1.0])
    e_v = np.cross(e_w, e_u)
    cp, sp = math.cos(pitch), math.sin(pitch)
    e_u, e_w = cp * e_u - sp * e_w, sp * e_u + cp * e_w
    cr, sr = math.cos(roll), math.sin(roll)
    e_v, e_w = cr * e_v + sr * e_w, -sr * e_v + cr * e_w
    return e_u, e_v, e_w


@dataclass(frozen=True)
class _Segment:
    t0: float
    t1: float
    p0: np.ndarray
    p1: np.ndarray
    yaw0: float
    yaw1: float
    saw_amplitude: float = 0.0
    saw_frequency: float = 0.0
    begins_cut: bool = False
    pitch0: float = 0.0
    pitch1: float = 0.0
    roll0: float = 0.0
    roll1: float = 0.0

    def angles(self, a):
        return (
            self.yaw0 + a * (self.yaw1 - self.yaw0),
            self.pitch0 + a * (self.pitch1 - self.pitch0),
            self.roll0 + a * (self.roll1 - self.roll0),
        )


def _move(t0, t1, a: "KnifePose", b: "KnifePose", begins_cut=False):
    return _Segment(
        t0, t1, np.asarray(a.origin, float), np.asarray(b.origin, float), a.yaw, b.yaw,
        begins_cut=begins_cut, pitch0=a.pitch, pitch1=b.pitch, roll0=a.roll, roll1=b.roll,
    )


@dataclass
class KnifeTrajectory:
    """Piecewise-linear knife motion with optional sawing, plus cut events.

    A segment flagged ``begins_cut`` is a downward stroke; the solver splits the pieces
    under the blade's plane when that segment starts (see ``MPMCuttingSim.begin_cut``).
    """

    segments: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.segments[-1].t1 if self.segments else 0.0

    @property
    def cut_times(self):
        """Start times of the downward strokes."""
        return [s.t0 for s in self.segments if s.begins_cut]

    def _segment(self, t: float) -> _Segment:
        for s in self.segments:
            if t < s.t1:
                return s
        return self.segments[-1]

    def state(self, t: float):
        """Knife pose and velocity at time t.

        Returns:
            (KnifePose, linear velocity (3,), angular velocity (3,)).
        """
        s = self._segment(t)
        span = max(s.t1 - s.t0, 1e-9)
        a = min(max((t - s.t0) / span, 0.0), 1.0)
        moving = s.t0 <= t <= s.t1
        origin = s.p0 + a * (s.p1 - s.p0)
        yaw, pitch, roll = s.angles(a)
        lin = (s.p1 - s.p0) / span if moving else np.zeros(3)
        ang = np.zeros(3)
        if moving and (s.yaw0 != s.yaw1 or s.pitch0 != s.pitch1 or s.roll0 != s.roll1):
            # Angular velocity from dR/dt R^T, by a central difference of the interpolated angles.
            h = 1e-4
            r_m = np.stack(_axes(*s.angles(a - h)), axis=1)
            r_p = np.stack(_axes(*s.angles(a + h)), axis=1)
            r_0 = np.stack(_axes(yaw, pitch, roll), axis=1)
            w = (r_p - r_m) / (2.0 * h * span) @ r_0.T
            ang = 0.5 * np.array([w[2, 1] - w[1, 2], w[0, 2] - w[2, 0], w[1, 0] - w[0, 1]])
        if s.saw_amplitude > 0.0 and moving:
            # Sawing slides the blade along its own edge, so the cut plane does not move.
            e_u = _axes(yaw, pitch, roll)[0]
            phase = 2.0 * math.pi * s.saw_frequency * (t - s.t0)
            origin = origin + s.saw_amplitude * math.sin(phase) * e_u
            lin = lin + s.saw_amplitude * 2.0 * math.pi * s.saw_frequency * math.cos(phase) * e_u
        return KnifePose(origin=origin, yaw=yaw, pitch=pitch, roll=roll), lin, ang

    @classmethod
    def linear(cls, t0: float, t1: float, start: KnifePose, end: KnifePose, begins_cut: bool = False):
        """One straight move from ``start`` to ``end`` over [t0, t1] - one step of a co-simulation."""
        return cls(segments=[_move(t0, t1, start, end, begins_cut)])

    @classmethod
    def strokes(
        cls,
        cuts,
        *,
        top_z: float,
        board_z: float,
        start_time: float = 0.15,
        clearance: float = 0.012,
        board_gap: float = 0.3e-3,
        descend_speed: float = 0.05,
        ascend_speed: float = 0.10,
        travel_speed: float = 0.15,
        dwell: float = 0.08,
        saw_amplitude: float = 0.0,
        saw_frequency: float = 0.0,
        settle: float = 0.25,
        retreat: float = 0.0,
    ) -> "KnifeTrajectory":
        """Builds a sequence of vertical cuts.

        Args:
            cuts: Iterable of (x, y, yaw): the edge midpoint over the food and the
                blade yaw for each cut, in order.
            top_z: Height of the food's top surface.
            board_z: Height of the cutting board.
            start_time: Time before the first motion, so the food settles under gravity.
            clearance: Height of the edge above the food before and after a stroke.
            board_gap: The stroke stops this far above the board.
            descend_speed: Edge speed while cutting [m/s].
            ascend_speed: Speed of the lift after a cut [m/s].
            travel_speed: Speed of the moves between cuts [m/s].
            dwell: Pause at the bottom of a stroke [s].
            saw_amplitude: Amplitude of a sawing motion along the edge [m] (0 = press cut).
            saw_frequency: Frequency of that motion [Hz].
            settle: Time after the last lift, so the pieces come to rest.
            retreat: After the last cut, lift the knife this much higher (out of a camera's way).
        """
        segs = []
        t = start_time
        cuts = list(cuts)
        hover_z = top_z + clearance
        bottom_z = board_z + board_gap
        x0, y0, yaw0 = cuts[0]
        pos = np.array([x0, y0, hover_z])
        yaw = yaw0
        segs.append(_Segment(0.0, t, pos, pos, yaw, yaw))
        for x, y, cut_yaw in cuts:
            target = np.array([x, y, hover_z])
            dist = float(np.linalg.norm(target - pos))
            turn = abs(cut_yaw - yaw)
            move_time = max(dist / travel_speed, turn / 1.5)
            if move_time > 0.0:
                segs.append(_Segment(t, t + move_time, pos, target, yaw, cut_yaw))
                t += move_time
            pos, yaw = target, cut_yaw
            bottom = np.array([x, y, bottom_z])
            down_time = (hover_z - bottom_z) / descend_speed
            segs.append(
                _Segment(t, t + down_time, pos, bottom, yaw, yaw, saw_amplitude, saw_frequency, begins_cut=True)
            )
            t += down_time
            segs.append(_Segment(t, t + dwell, bottom, bottom, yaw, yaw))
            t += dwell
            up_time = (hover_z - bottom_z) / ascend_speed
            segs.append(_Segment(t, t + up_time, bottom, pos, yaw, yaw))
            t += up_time
        if retreat > 0.0:
            away = pos + np.array([0.0, 0.0, retreat])
            segs.append(_Segment(t, t + retreat / travel_speed, pos, away, yaw, yaw))
            t += retreat / travel_speed
            pos = away
        segs.append(_Segment(t, t + settle, pos, pos, yaw, yaw))
        return cls(segments=segs)

    @classmethod
    def rock_strokes(
        cls,
        cuts,
        *,
        top_z: float,
        board_z: float,
        knife_half_length: float,
        tip_reach: float,
        tilt: float = math.radians(30.0),
        start_time: float = 0.15,
        clearance: float = 0.006,
        board_gap: float = 0.3e-3,
        descend_speed: float = 0.06,
        rock_rate: float = math.radians(70.0),
        dwell: float = 0.06,
        ascend_speed: float = 0.12,
        travel_speed: float = 0.2,
        retreat: float = 0.08,
        settle: float = 0.3,
        rock_steps: int = 24,
        slide: float = 0.0,
        slide_speed: float = 0.3,
        geometry: KnifeGeometry = None,
        glide: float = 0.0,
        roll_glide: float = 0.0,
    ) -> "KnifeTrajectory":
        """Rocking cuts as a cook makes them (the Chef'd reference clip).

        The blade comes down tilted, tip first, until the tip touches the board just beyond
        the food; then it rocks down until the whole straight edge lies on the board. With a
        belly (``geometry.tip_rise``) the edge's lowest point is what stays on the board, so
        the knife rolls on the belly and its tip ends up raised, as a chef's knife lies. Both
        phases keep the blade in the cut's plane. ``glide`` and ``roll_glide`` push the knife
        forward along its length while it comes down and while it rocks: the Chef'd cook
        does (4.2 and 2.4 cm), and the sliding is what makes it cut easily (see mpm.py on
        slicing). With ``slide`` the knife is then drawn back along its edge, towards the
        handle, before it lifts: the Chef'd cook pulls it out of the cut that way rather than
        lifting it through the food.

        Args:
            cuts: Iterable of (x, y, yaw): a point of the cut's plane over the food and the
                blade's yaw; e_u (yaw) points from the heel to the tip.
            top_z: Height of the food's top surface.
            board_z: Height of the cutting board.
            knife_half_length: Half the edge length; the tip is at origin + half_length e_u.
            tip_reach: Distance along e_u from (x, y) to where the tip lands on the board,
                a little beyond the food's far side.
            tilt: Pitch of the blade while it comes down [rad].
            rock_rate: Angular speed of the rock [rad/s].
            rock_steps: Straight segments approximating the rock about the tip.
            slide: How far the knife is drawn back along its edge at the board [m].
            slide_speed: Speed of that draw [m/s].
            geometry: The blade, for its belly; a straight edge of ``knife_half_length`` if None.
            glide: Forward travel along the blade while it comes down [m].
            roll_glide: Forward travel along the blade while it rocks [m].
            Other arguments: as in ``strokes``.
        """
        segs = []
        g = geometry if geometry is not None else KnifeGeometry(half_length=knife_half_length)

        def tip_pose(x, y, yaw, low_z, pitch, forward=0.0):
            # The tip above (x, y) + (tip_reach + forward) along the yaw, the edge's lowest point at low_z.
            e_u, _, e_w = _axes(yaw, pitch, 0.0)
            level = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            origin = np.array([x, y, 0.0]) + (tip_reach + forward) * level - (g.half_length * e_u + g.tip_rise * e_w)
            origin[2] = 0.0
            origin[2] = low_z - g.lowest_edge_point(KnifePose(origin=origin, yaw=yaw, pitch=pitch))[2]
            return KnifePose(origin=origin, yaw=yaw, pitch=pitch)

        cuts = list(cuts)
        hover_z = top_z + clearance
        bottom_z = board_z + board_gap
        x0, y0, yaw0 = cuts[0]
        pose = tip_pose(x0, y0, yaw0, hover_z, tilt)
        segs.append(_move(0.0, start_time, pose, pose))
        t = start_time
        for x, y, yaw in cuts:
            start = tip_pose(x, y, yaw, hover_z, tilt)
            dist = float(np.linalg.norm(start.origin - pose.origin))
            if dist > 1e-9 or yaw != pose.yaw:
                dt = max(dist / travel_speed, abs(yaw - pose.yaw) / 1.5, 1e-3)
                segs.append(_move(t, t + dt, pose, start))
                t += dt
            down = tip_pose(x, y, yaw, bottom_z, tilt, glide)
            dt = (hover_z - bottom_z) / descend_speed
            segs.append(_move(t, t + dt, start, down, begins_cut=True))
            t += dt
            prev = down
            for k in range(1, rock_steps + 1):
                nxt = tip_pose(x, y, yaw, bottom_z, tilt * (1.0 - k / rock_steps), glide + roll_glide * k / rock_steps)
                dt = tilt / rock_rate / rock_steps
                segs.append(_move(t, t + dt, prev, nxt))
                t += dt
                prev = nxt
            segs.append(_move(t, t + dwell, prev, prev))
            t += dwell
            if slide > 0.0:
                back = KnifePose(prev.origin - slide * prev.axes()[0], yaw)
                segs.append(_move(t, t + slide / slide_speed, prev, back))
                t += slide / slide_speed
                prev = back
            up = KnifePose(prev.origin + np.array([0.0, 0.0, hover_z - bottom_z]), yaw)
            dt = (hover_z - bottom_z) / ascend_speed
            segs.append(_move(t, t + dt, prev, up))
            t += dt
            pose = up
        return cls(segments=_finish(segs, t, pose, retreat, travel_speed, settle))

    @classmethod
    def press_twist(
        cls,
        cut,
        *,
        top_z: float,
        board_z: float,
        twist: float = math.radians(15.0),
        start_time: float = 0.15,
        clearance: float = 0.006,
        board_gap: float = 0.3e-3,
        descend_speed: float = 0.04,
        push_height: float = 0.025,
        push_distance: float = 0.03,
        twist_time: float = 0.6,
        hold: float = 0.2,
        ascend_speed: float = 0.12,
        travel_speed: float = 0.2,
        retreat: float = 0.08,
        settle: float = 0.8,
    ) -> "KnifeTrajectory":
        """A straight press cut, then the blade tips the cut slab away from the block (the HDMD
        reference clip).

        A slab only goes over when it is pushed sideways above its centre of mass. A blade
        twisted about an edge left on the board presses the slab down into the board instead
        and only bends it, so the blade first rises to ``push_height`` inside the cut, then
        moves ``push_distance`` towards the slab while it tilts by ``twist``.

        Args:
            cut: (x, y, yaw) of the cut, as in ``strokes``.
            twist: Roll while pushing [rad]; positive moves the spine towards -e_v, and the push
                goes towards -e_v too, so it tips over a slab on that side (negative: +e_v).
            push_height: Height of the edge above the board while it pushes [m].
            push_distance: Sideways travel of the push [m].
            twist_time: Duration of the push [s].
            hold: Pause after the push before the knife lifts.
            Other arguments: as in ``strokes``.
        """
        x, y, yaw = cut
        hover_z = top_z + clearance
        bottom_z = board_z + board_gap
        hover = KnifePose(np.array([x, y, hover_z]), yaw)
        bottom = KnifePose(np.array([x, y, bottom_z]), yaw)
        raised = KnifePose(np.array([x, y, board_z + push_height]), yaw)
        e_v = hover.axes()[1]
        side = -1.0 if twist >= 0.0 else 1.0
        pushed = KnifePose(raised.origin + side * push_distance * e_v, yaw, roll=twist)
        lifted = KnifePose(pushed.origin + np.array([0.0, 0.0, hover_z - pushed.origin[2]]), yaw, roll=twist)
        segs = [_move(0.0, start_time, hover, hover)]
        t = start_time
        dt = (hover_z - bottom_z) / descend_speed
        segs.append(_move(t, t + dt, hover, bottom, begins_cut=True))
        t += dt
        dt = (push_height - board_gap) / descend_speed
        segs.append(_move(t, t + dt, bottom, raised))
        t += dt
        segs.append(_move(t, t + twist_time, raised, pushed))
        t += twist_time
        segs.append(_move(t, t + hold, pushed, pushed))
        t += hold
        dt = (hover_z - pushed.origin[2]) / ascend_speed
        segs.append(_move(t, t + dt, pushed, lifted))
        t += dt
        return cls(segments=_finish(segs, t, lifted, retreat, travel_speed, settle))


def _finish(segs, t, pose, retreat, travel_speed, settle):
    """Lifts the knife out of the way and lets the pieces settle."""
    if retreat > 0.0:
        away = KnifePose(pose.origin + np.array([0.0, 0.0, retreat]), pose.yaw, pose.pitch, pose.roll)
        segs.append(_move(t, t + retreat / travel_speed, pose, away))
        t += retreat / travel_speed
        pose = away
    segs.append(_move(t, t + settle, pose, pose))
    return segs
