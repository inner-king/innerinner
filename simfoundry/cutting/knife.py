# SPDX-License-Identifier: Apache-2.0
"""Kitchen-knife geometry, pose and scripted cutting strokes.

The knife frame has its origin at the midpoint of the cutting edge:

    e_u  along the edge (blade length), towards the tip
    e_w  from the edge towards the spine
    e_v  = e_w x e_u, the blade normal (the thickness direction)

A blade held vertically with yaw 0 cuts along the plane y = const.
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
    """

    half_length: float = 0.10
    height: float = 0.045
    edge_thickness: float = 0.3e-3
    spine_thickness: float = 2.0e-3
    bevel_height: float = 0.02
    handle_length: float = 0.11


@dataclass(frozen=True)
class KnifePose:
    """Rigid pose of a knife whose edge is horizontal and blade vertical.

    Attributes:
        origin: World position of the edge midpoint, shape (3,).
        yaw: Rotation of e_u about +z [rad].
    """

    origin: np.ndarray
    yaw: float = 0.0

    def axes(self):
        """Returns (e_u, e_v, e_w) as unit vectors in the world frame."""
        e_u = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
        e_w = np.array([0.0, 0.0, 1.0])
        e_v = np.cross(e_w, e_u)
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
        yaw = s.yaw0 + a * (s.yaw1 - s.yaw0)
        lin = (s.p1 - s.p0) / span if moving else np.zeros(3)
        ang = np.array([0.0, 0.0, (s.yaw1 - s.yaw0) / span]) if moving else np.zeros(3)
        if s.saw_amplitude > 0.0 and moving:
            # Sawing slides the blade along its own edge, so the cut plane does not move.
            e_u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            phase = 2.0 * math.pi * s.saw_frequency * (t - s.t0)
            origin = origin + s.saw_amplitude * math.sin(phase) * e_u
            lin = lin + s.saw_amplitude * 2.0 * math.pi * s.saw_frequency * math.cos(phase) * e_u
        return KnifePose(origin=origin, yaw=yaw), lin, ang

    @classmethod
    def linear(cls, t0: float, t1: float, start: KnifePose, end: KnifePose, begins_cut: bool = False):
        """One straight move from ``start`` to ``end`` over [t0, t1] - one step of a co-simulation."""
        seg = _Segment(
            t0, t1, np.asarray(start.origin, float), np.asarray(end.origin, float),
            start.yaw, end.yaw, begins_cut=begins_cut,
        )
        return cls(segments=[seg])

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
