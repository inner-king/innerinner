# SPDX-License-Identifier: Apache-2.0
"""
The geometry of a runtime cut: no USD, no OmniGibson, no Isaac Sim.

A cut replaces one rigid body by two. Everything here works in one rigid
link's frame, in the asset's own (unscaled) units, so that a piece can be
authored as a USD asset that is placed at its parent's pose with its parent's
scale and lines up with it exactly:

- :func:`blade_plane_from_points` and :func:`plane_to_world` turn the blade's
  collision vertices and the knife's pose into the cut plane, and
  :func:`plane_to_local` brings that plane into the object's frame.
- :func:`clip_mesh` cuts the visual mesh. Triangles that cross the plane are
  split, not assigned to a side by their centroid, and face-varying primvars
  (the ``st`` the textured skin needs) are interpolated along the cut edges, so
  the skin keeps its texture up to the cut.
- :func:`cap_polygon` closes the cut face: the cross-section bounded by the cut
  edges, triangulated as a polygon with holes.
- :func:`clip_convex` and :func:`merge_hulls` give the piece colliders. A convex
  polytope cut by a plane stays convex, so the asset's own decomposition, clipped,
  is a decomposition of each piece; runtime decomposition (V-HACD 0.45-5.5 s,
  CoACD 8-35 s per half, repro/README.md section 15.1) never runs.
- :func:`split_mass` and :func:`velocity_at` carry the parent's mass and motion
  over to the pieces.

:func:`cut_geometry` is the whole cut. It removes a slab as thick as the blade
(``half_gap`` on either side of the plane) rather than splitting at the plane:
the swap happens on the blade's first qualifying contact, and the blade has to
fit between the two pieces it leaves, or PhysX pushes them apart from inside.
"""

from dataclasses import dataclass, field
import time

import numpy as np

#: Clipped hulls smaller than this fraction of the whole object's hull volume are
#: dropped: a sliver the plane barely shaves off a hull adds a collider prim
#: (6-8 ms of OmniGibson's add each, section 15.1) and nothing to the shape.
MIN_HULL_FRACTION = 1e-4

#: Rim endpoints closer than this (in the asset's units, metres here) are one
#: vertex. The cut edges of two meshes of one link (a piece's skin and its
#: earlier cut face) meet at the same points only up to rounding.
WELD_TOLERANCE = 1e-7

#: Vertices closer to a cut plane than this (asset units) lie on it.
PLANE_TOLERANCE = 1e-12

#: How far outside every collider (asset units, metres here) a point of the cut face may be and still count as
#: inside the object: decomposition hulls leave seams between them, and a scan's skin bulges past its hulls.
CAP_HULL_TOLERANCE = 1e-3

#: Half-widths (asset units) of the tube _filled_outline() draws along a degenerate rim, tried in turn: gaps in the
#: rim narrower than twice the width are closed.
FILL_TUBE_WIDTHS = (2e-5, 1e-4, 3e-4, 1e-3)


# ----------------------------------------------------------------------------------------------- frames and planes
@dataclass(frozen=True)
class Plane:
    """An oriented plane: the points ``x`` with ``(x - origin) . normal == 0``.

    Attributes:
        origin (np.ndarray): (3,) a point on the plane
        normal (np.ndarray): (3,) unit normal; "inside" for a clip is the side it points to
    """

    origin: np.ndarray
    normal: np.ndarray

    @staticmethod
    def make(origin, normal):
        """A plane through ``origin`` with ``normal`` normalized."""
        n = np.asarray(normal, np.float64)
        length = np.linalg.norm(n)
        if not length > 0:
            raise ValueError(f"plane normal must be non-zero, got {normal}")
        return Plane(np.asarray(origin, np.float64).copy(), n / length)

    def signed_distance(self, points):
        """Signed distance of ``points`` (..., 3) along the normal."""
        return (np.asarray(points, np.float64) - self.origin) @ self.normal

    def offset(self, distance):
        """The same plane moved ``distance`` along its normal."""
        return Plane(self.origin + distance * self.normal, self.normal)

    def flipped(self):
        """The same plane with the normal reversed."""
        return Plane(self.origin, -self.normal)


def quat_xyzw_to_matrix(quat):
    """Rotation matrix of a quaternion in XYZW order (normalized first)."""
    x, y, z, w = np.asarray(quat, np.float64) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _scale_vector(scale):
    if scale is None:
        return np.ones(3)
    s = np.broadcast_to(np.asarray(scale, np.float64), (3,)).copy()
    if np.any(s == 0):
        raise ValueError(f"scale must be non-zero, got {scale}")
    return s


def blade_plane_from_points(points):
    """The mid-plane of a blade and its thickness, from the blade's collision vertices.

    The blade normal is the direction in which the vertices spread least (the
    smallest principal axis): a blade is long, tall and thin. The plane sits
    halfway between the outermost vertices along it, so a wedge-section blade
    (thick spine, thin edge) is centred on it. The normal's sign is arbitrary for
    a cut; it is fixed so that its largest component is positive, which keeps the
    result deterministic.

    Args:
        points (np.ndarray): (n, 3) blade collision vertices in the knife's link frame

    Returns:
        tuple: (Plane in the link frame, thickness along the normal in the same units)
    """
    P = np.asarray(points, np.float64)
    if len(P) < 4:
        raise ValueError("a blade needs at least four collision vertices")
    c = P.mean(0)
    _, vecs = np.linalg.eigh(np.cov((P - c).T))
    n = vecs[:, 0]
    if n[np.argmax(np.abs(n))] < 0:
        n = -n
    proj = (P - c) @ n
    lo, hi = proj.min(), proj.max()
    return Plane(c + n * (lo + hi) / 2, n), float(hi - lo)


def plane_to_world(plane_local, position, quat_xyzw, scale=None):
    """A plane given in a link's unscaled frame, in the world.

    The link maps a local point ``x`` to ``position + R @ (scale * x)``. Normals
    transform with the inverse transpose, ``R @ (n / scale)``.

    Args:
        plane_local (Plane): plane in the link frame
        position (array-like): (3,) link position
        quat_xyzw (array-like): (4,) link orientation
        scale (None or float or array-like): the object's scale

    Returns:
        tuple: (Plane in the world, factor q such that a local distance along the
            plane's normal is ``1 / q`` times the world distance)
    """
    R = quat_xyzw_to_matrix(quat_xyzw)
    s = _scale_vector(scale)
    origin = np.asarray(position, np.float64) + R @ (s * plane_local.origin)
    m = plane_local.normal / s
    q = float(np.linalg.norm(m))
    return Plane(origin, R @ (m / q)), q


def plane_to_local(plane_world, position, quat_xyzw, scale=None):
    """A world plane in a link's unscaled frame.

    Args:
        plane_world (Plane): plane in the world
        position (array-like): (3,) link position
        quat_xyzw (array-like): (4,) link orientation
        scale (None or float or array-like): the object's scale

    Returns:
        tuple: (Plane in the link frame, factor k such that a world distance ``d``
            along the world normal is ``d / k`` along the local normal)
    """
    R = quat_xyzw_to_matrix(quat_xyzw)
    s = _scale_vector(scale)
    origin = (R.T @ (plane_world.origin - np.asarray(position, np.float64))) / s
    m = s * (R.T @ plane_world.normal)
    k = float(np.linalg.norm(m))
    return Plane(origin, m / k), k


def plane_basis(normal):
    """Two unit vectors ``u, v`` with ``u x v == normal``."""
    n = np.asarray(normal, np.float64)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


# -------------------------------------------------------------------------------------------------- visual meshes
@dataclass
class MeshPart:
    """One triangle mesh of a link, in the link frame.

    Attributes:
        points (np.ndarray): (n, 3) vertices
        faces (np.ndarray): (m, 3) vertex indices
        corner_attrs (dict): name -> (m, 3, k) per-face-corner values (face-varying primvars such as ``st``)
        name (str): mesh prim name
        material (None or str): what the mesh is bound to, carried through untouched
    """

    points: np.ndarray
    faces: np.ndarray
    corner_attrs: dict = field(default_factory=dict)
    name: str = "mesh"
    material: object = None


def clip_mesh(part, plane):
    """The part of a triangle mesh on the ``+normal`` side of ``plane`` (``d >= 0``).

    A triangle crossing the plane is cut at its edges: one corner kept gives one
    triangle, two kept give two. The crossing point of an edge is computed once,
    from the edge's lower vertex index, and shared by both triangles of that edge,
    so a welded input stays welded and its cut edges form closed rims. Corner
    attributes are interpolated along each triangle's own edges, so a UV seam that
    the cut crosses stays a seam.

    Args:
        part (MeshPart): mesh to clip
        plane (Plane): kept side is where the normal points

    Returns:
        tuple: (MeshPart or None if nothing is kept, (r, 2, 3) cut-edge segments on the plane)
    """
    P = np.asarray(part.points, np.float64)
    F = np.asarray(part.faces, np.int64)
    n = len(P)
    d = plane.signed_distance(P)
    # A vertex within rounding of the plane is on it; one 1e-18 off would get a crossing vertex 1e-18 away
    d[np.abs(d) <= PLANE_TOLERANCE] = 0.0
    inside = d >= 0.0
    ins = inside[F]
    n_in = ins.sum(1)
    full = n_in == 3
    cross = np.flatnonzero((n_in == 1) | (n_in == 2))
    if not full.any() and len(cross) == 0:
        return None, np.zeros((0, 2, 3))
    if len(cross) == 0:
        kept = MeshPart(P, F[full], {k: v[full] for k, v in part.corner_attrs.items()}, part.name, part.material)
        return _compact(kept), np.zeros((0, 2, 3))

    # Roll each crossing triangle so its odd corner (alone on its side) comes first; winding is kept.
    one = n_in[cross] == 1
    odd = np.where(one, np.argmax(ins[cross], 1), np.argmin(ins[cross], 1))
    roll = (odd[:, None] + np.arange(3)[None, :]) % 3
    fc = np.take_along_axis(F[cross], roll, 1)
    a, b, c = fc[:, 0], fc[:, 1], fc[:, 2]

    # One vertex per crossed edge, keyed by the undirected edge. A kept endpoint lying on the plane (d == 0) is
    # its own crossing point: a second vertex there would leave zero-area triangles and a doubled rim.
    k = len(cross)
    lo = np.minimum(np.concatenate([a, a]), np.concatenate([b, c]))
    hi = np.maximum(np.concatenate([a, a]), np.concatenate([b, c]))
    keys, inv = np.unique(lo * n + hi, return_inverse=True)
    klo, khi = keys // n, keys % n
    t = d[klo] / (d[klo] - d[khi])
    on_lo, on_hi = d[klo] == 0, d[khi] == 0
    fresh = ~(on_lo | on_hi)
    ids = np.empty(len(keys), np.int64)
    ids[on_lo], ids[on_hi] = klo[on_lo], khi[on_hi]
    ids[fresh] = n + np.arange(np.count_nonzero(fresh))
    new_points = P[klo[fresh]] + t[fresh, None] * (P[khi[fresh]] - P[klo[fresh]])
    pb = ids[inv[:k]]
    pc = ids[inv[k:]]

    tris = [F[full], np.stack([a, pb, pc], 1)[one],
            np.stack([pb, b, c], 1)[~one], np.stack([pb, c, pc], 1)[~one]]
    faces = np.concatenate(tris)
    valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[valid]
    rim = np.concatenate([np.stack([pb, pc], 1)[one], np.stack([pc, pb], 1)[~one]])
    rim = rim[rim[:, 0] != rim[:, 1]]

    attrs = {}
    if part.corner_attrs:
        dc = np.take_along_axis(d[F[cross]], roll, 1)
        s_ab = (dc[:, 0] / (dc[:, 0] - dc[:, 1]))[:, None]
        s_ac = (dc[:, 0] / (dc[:, 0] - dc[:, 2]))[:, None]
        for name, A in part.corner_attrs.items():
            Ac = np.take_along_axis(A[cross], roll[:, :, None], 1).astype(np.float64)
            Aa, Ab, Acc = Ac[:, 0], Ac[:, 1], Ac[:, 2]
            Apb = Aa + s_ab * (Ab - Aa)
            Apc = Aa + s_ac * (Acc - Aa)
            attrs[name] = np.concatenate([
                A[full],
                np.stack([Aa, Apb, Apc], 1)[one].astype(A.dtype),
                np.stack([Apb, Ab, Acc], 1)[~one].astype(A.dtype),
                np.stack([Apb, Acc, Apc], 1)[~one].astype(A.dtype),
            ])[valid]

    all_points = np.concatenate([P, new_points])
    segments = all_points[rim]
    return _compact(MeshPart(all_points, faces, attrs, part.name, part.material)), segments


def _compact(part):
    """Drop vertices no face uses."""
    used = np.zeros(len(part.points), bool)
    used[part.faces.ravel()] = True
    if used.all():
        return part
    remap = np.cumsum(used) - 1
    return MeshPart(part.points[used], remap[part.faces], part.corner_attrs, part.name, part.material)


def _snap(points, tolerance=WELD_TOLERANCE):
    return np.round(np.asarray(points, np.float64) / tolerance) * tolerance


def _even_odd_inside(point, segments_2d):
    """Whether ``point`` (2,) lies inside the region bounded by ``segments_2d`` (r, 2, 2), by ray parity."""
    x, y = point
    (x1, y1), (x2, y2) = segments_2d[:, 0].T, segments_2d[:, 1].T
    straddle = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return bool(np.count_nonzero(straddle & (x < xc)) % 2)


def _closed_walks(seg2):
    """Split 2D segments (r, 2, 2) into walks that use every segment once, as (k, 2) vertex arrays.

    Where every vertex has an even degree (a closed cross-section, crossings included) every walk is closed.
    """
    pts, inv = np.unique(seg2.reshape(-1, 2), axis=0, return_inverse=True)
    inv = inv.reshape(-1, 2)
    incident = [[] for _ in range(len(pts))]
    for e, (a, b) in enumerate(inv):
        incident[a].append(e)
        incident[b].append(e)
    used = np.zeros(len(inv), bool)
    walks = []
    for e0 in range(len(inv)):
        if used[e0]:
            continue
        used[e0] = True
        start, cur = inv[e0]
        walk = [start, cur]
        while cur != start:
            e = next((e for e in incident[cur] if not used[e]), None)
            if e is None:
                break
            used[e] = True
            a, b = inv[e]
            cur = b if a == cur else a
            walk.append(cur)
        walks.append(pts[walk])
    return walks


def _sample_points(polygon, grid=6):
    """A point guaranteed inside ``polygon`` and the points of a ``grid`` x ``grid`` lattice over it that are inside."""
    import shapely

    x0, y0, x1, y1 = polygon.bounds
    gx, gy = np.meshgrid(np.linspace(x0, x1, grid + 2)[1:-1], np.linspace(y0, y1, grid + 2)[1:-1])
    gx, gy = gx.ravel(), gy.ravel()
    inside = shapely.contains_xy(polygon, gx, gy)
    return np.concatenate([np.asarray(polygon.representative_point().coords), np.c_[gx[inside], gy[inside]]])


def _filled_outline(seg2, expected_area=None):
    """Everything the 2D segments enclose, holes included.

    The segments are thickened into a tube, each tube filled to its outer ring and
    shrunk back, which closes gaps narrower than the tube. The tube is widened
    through FILL_TUBE_WIDTHS until the fill covers half of ``expected_area``: a
    scanned skin can be open along a slit (0.2-0.6 mm on the apple) where its two
    layers join, and then no rim encloses anything. Concave corners sharper than
    the final width are rounded; the rest of the outline is kept.
    """
    import shapely

    lines = shapely.simplify(shapely.line_merge(shapely.multilinestrings(shapely.linestrings(seg2))),
                             FILL_TUBE_WIDTHS[0])
    filled = shapely.Polygon()
    for width in FILL_TUBE_WIDTHS:
        tube = shapely.buffer(lines, width, quad_segs=2)
        filled = shapely.buffer(shapely.union_all([shapely.Polygon(p.exterior) for p in shapely.get_parts(tube)]),
                                -width, quad_segs=2)
        if expected_area is None or filled.area >= 0.5 * expected_area:
            break
    return filled


def cap_polygon(segments, plane, inside=None, expected_area=None):
    """Triangulate the cut face bounded by ``segments``, facing away from the kept side (``-plane.normal``).

    The segments are the cut edges of every mesh of the piece. They are split
    into closed walks, each walk is made a valid polygon, and each polygon is
    voted solid or hole by points sampled inside it: the cap is the solids minus
    the holes, triangulated by constrained Delaunay. So a section that is not
    star-shaped (a banana cut lengthwise) or has a hole (a ring) is capped as it
    is; a fan from each loop's centre, which the offline cutter used
    (prepare_cut_pieces.close_rim), covers neither.

    The scanned fruits' visual meshes need every part of this. They are
    thousands of UV charts whose windings disagree (welded, the apple encloses
    0.2 cm^3 of its ~150) and in places two layers about 0.1 mm apart, so a cut's
    rim can wind twice around the section, crossing itself where the layers swap.
    A parity count over such a rim calls the middle of the apple outside, and
    GEOS's polygonizer drops the ring outright. A walk made valid by the
    "structure" method keeps the whole area it winds around.

    "Inside the object" is ``inside`` when given; the cut passes it the object's
    colliders, so the visual cap covers what the collision shape fills. Without
    it a sample point is inside by ray parity over the segments, which is exact
    for a clean closed surface.

    One case defeats the walks: where the scanned skin is open along a slit whose
    edges join its two layers, the rim is a C traced out along one layer and back
    along the other, winding zero times around anything (one side of one of
    twelve test cuts of the scene's fruits). So the result is checked against
    ``expected_area``, the colliders' own section; if it falls short by half, the
    cap is everything the rim encloses once its gaps are closed
    (_filled_outline), which also covers any hole in the section.

    Args:
        segments (np.ndarray): (r, 2, 3) cut-edge segments lying on ``plane``
        plane (Plane): the clip plane; the piece is on its ``+normal`` side
        inside (None or callable): (p, 3) points -> (p,) bool, whether each is inside the object
        expected_area (None or float): area the cap should roughly have (the colliders' section)

    Returns:
        None or tuple: (V (p, 3), F (q, 3)) of the cap, or None if the segments enclose nothing
    """
    import shapely

    seg = np.asarray(segments, np.float64)
    if len(seg) < 3:
        return None
    u, v = plane_basis(plane.normal)
    rel = seg - plane.origin
    seg2 = _snap(np.stack([rel @ u, rel @ v], -1))
    keep = np.any(seg2[:, 0] != seg2[:, 1], axis=1)
    seg, seg2 = seg[keep], seg2[keep]
    # Each cut edge once: a surface authored twice with opposite windings (the scanned fruits' visual meshes
    # are) gives every edge twice, which the parity count would read as outside
    a, b = seg2[:, 0], seg2[:, 1]
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    canonical = np.where(swap[:, None], np.c_[b, a], np.c_[a, b])
    first = np.sort(np.unique(canonical, axis=0, return_index=True)[1])
    seg, seg2 = seg[first], seg2[first]
    if len(seg2) < 3:
        return None
    # The triangulation only uses the (snapped) rim endpoints; put each back where the skin has it, so the cap
    # shares the skin's rim exactly instead of to within the snapping grid
    exact = dict(zip(map(tuple, seg2.reshape(-1, 2)), seg.reshape(-1, 3)))
    solids, holes = [], []
    for walk in _closed_walks(seg2):
        if len(walk) < 4:
            continue
        polygon = shapely.make_valid(shapely.Polygon(walk), method="structure", keep_collapsed=False)
        if polygon.is_empty or not polygon.area > 0:
            continue
        # Voted over several points: a decomposition's hulls leave seams (the orange's two meet with a 0.6 mm gap
        # through its middle), and one point can fall into one
        samples = _sample_points(polygon)
        if inside is None:
            votes = [_even_odd_inside(p, seg2) for p in samples]
        else:
            votes = inside(plane.origin + samples[:, :1] * u + samples[:, 1:] * v)
        (solids if np.mean(votes) > 0.5 else holes).append(polygon)
    region = shapely.union_all(solids) if solids else shapely.Polygon()
    if holes and not region.is_empty:
        region = shapely.difference(region, shapely.union_all(holes))
    if expected_area is not None and region.area < 0.5 * expected_area:
        region = _filled_outline(seg2, expected_area)
    if region.is_empty:
        return None
    tris = shapely.get_parts(shapely.constrained_delaunay_triangles(region))
    if len(tris) == 0:
        return None
    T = shapely.get_coordinates(tris).reshape(len(tris), 4, 2)[:, :3]  # each triangle's ring repeats its start
    area2 = ((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1])
             - (T[:, 1, 1] - T[:, 0, 1]) * (T[:, 2, 0] - T[:, 0, 0]))
    T = T[area2 != 0]
    ccw = area2[area2 != 0] > 0
    T[ccw] = T[ccw][:, ::-1]  # counter-clockwise in (u, v) faces +normal; the cap faces -normal
    flat, inv = np.unique(T.reshape(-1, 2), axis=0, return_inverse=True)
    V = plane.origin + flat[:, :1] * u + flat[:, 1:] * v
    for i, key in enumerate(map(tuple, flat)):
        if key in exact:  # not where the noding split two crossing rim segments
            V[i] = exact[key]
    return V, inv.reshape(-1, 3)


# ---------------------------------------------------------------------------------------------------- colliders
@dataclass
class Hull:
    """A convex collider: its hull mesh, volume and centroid (link frame, asset units)."""

    points: np.ndarray
    faces: np.ndarray
    volume: float
    centroid: np.ndarray


def convex_hull(points):
    """The convex hull of ``points`` as an outward-wound triangle mesh with its volume and centroid.

    Returns:
        Hull or None: None when the points span no volume (the plane only grazed a hull)
    """
    from scipy.spatial import ConvexHull, QhullError

    P = np.asarray(points, np.float64)
    if len(P) < 4:
        return None
    try:
        hull = ConvexHull(P)
    except (QhullError, ValueError):
        return None
    if not hull.volume > 0:
        return None
    used = hull.vertices
    remap = np.full(len(P), -1)
    remap[used] = np.arange(len(used))
    V = P[used]
    F = remap[hull.simplices]
    inner = V.mean(0)
    tri = V[F]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", normals, tri[:, 0] - inner) < 0
    F[flip] = F[flip][:, ::-1]
    # Centroid of the solid: tetrahedra from an interior point to every face
    tri = V[F]
    vols = np.einsum("ij,ij->i", tri[:, 0] - inner, np.cross(tri[:, 1] - inner, tri[:, 2] - inner)) / 6.0
    cents = (tri.sum(1) + inner) / 4.0
    centroid = (vols[:, None] * cents).sum(0) / vols.sum()
    return Hull(V, F, float(hull.volume), centroid)


def clip_convex(points, plane):
    """A convex collider intersected with the half-space ``d >= 0``.

    The result is the hull of the vertices on that side and the points where the
    hull's edges cross the plane, which is convex again, so the asset's convex
    decomposition stays one after a cut.

    Args:
        points (np.ndarray): (n, 3) vertices of a convex collider
        plane (Plane): kept side is where the normal points

    Returns:
        Hull or None: None when nothing (or only a flat sliver) is on the kept side
    """
    from scipy.spatial import ConvexHull, QhullError

    P = np.asarray(points, np.float64)
    d = plane.signed_distance(P)
    if np.all(d >= 0):
        return convex_hull(P)
    if not np.any(d > 0):
        return None
    try:
        simplices = ConvexHull(P).simplices
    except (QhullError, ValueError):
        return None
    edges = np.unique(np.sort(np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]],
                                              simplices[:, [2, 0]]]), axis=1), axis=0)
    i, j = edges[:, 0], edges[:, 1]
    crossing = (d[i] >= 0) != (d[j] >= 0)
    i, j = i[crossing], j[crossing]
    t = d[i] / (d[i] - d[j])
    cut = P[i] + t[:, None] * (P[j] - P[i])
    return convex_hull(np.concatenate([P[d >= 0], cut]))


def merge_hulls(hulls, max_count, neighbours=4):
    """Merge convex colliders until there are at most ``max_count``.

    OmniGibson's add costs 6-8 ms per collider prim (README section 15.1: 0.34 s
    for the 34-38 prims of an apple's two pieces), so a piece is held to a few
    hulls. Each round merges the smallest hull with whichever of its nearest
    neighbours (by centroid) adds the least volume to their union's hull. Merging
    two hulls of a piece keeps every vertex on the piece's side of the cut, so the
    cut face stays flat.

    Args:
        hulls (list of Hull): colliders of one piece
        max_count (None or int): at most this many remain; None keeps all
        neighbours (int): candidates considered per round

    Returns:
        list of Hull
    """
    hulls = list(hulls)
    if max_count is None or max_count < 1:
        return hulls
    while len(hulls) > max_count:
        i = int(np.argmin([h.volume for h in hulls]))
        others = [j for j in range(len(hulls)) if j != i]
        dist = [np.linalg.norm(hulls[j].centroid - hulls[i].centroid) for j in others]
        best = None
        for j in [others[k] for k in np.argsort(dist)[:neighbours]]:
            merged = convex_hull(np.concatenate([hulls[i].points, hulls[j].points]))
            if merged is None:
                continue
            cost = merged.volume - hulls[i].volume - hulls[j].volume
            if best is None or cost < best[0]:
                best = (cost, j, merged)
        if best is None:
            break
        _, j, merged = best
        hulls = [h for k, h in enumerate(hulls) if k not in (i, j)] + [merged]
    return hulls


# ----------------------------------------------------------------------------------------------- mass and motion
def split_mass(total_mass, volumes):
    """Split ``total_mass`` in proportion to ``volumes`` (equal shares if they are all zero)."""
    vols = np.clip(np.asarray(volumes, np.float64), 0.0, None)
    if not vols.sum() > 0:
        return np.full(len(vols), float(total_mass) / len(vols))
    return float(total_mass) * vols / vols.sum()


def section_area(hulls, plane, tolerance=1e-9):
    """Area of the faces that hulls clipped by ``plane`` have on it (summed: overlaps count twice)."""
    from scipy.spatial import ConvexHull, QhullError

    u, v = plane_basis(plane.normal)
    total = 0.0
    for h in hulls:
        on = h.points[np.abs(plane.signed_distance(h.points)) <= tolerance] - plane.origin
        if len(on) < 3:
            continue
        try:
            total += ConvexHull(np.c_[on @ u, on @ v]).volume  # a 2D hull's volume is its area
        except (QhullError, ValueError):
            continue
    return total


def hulls_volume_centroid(hulls):
    """Total volume and volume-weighted centroid of a set of convex colliders.

    Overlapping hulls are counted twice; a decomposition's overlaps are thin, and a
    mass split only needs the ratio between the two pieces.
    """
    vol = np.array([h.volume for h in hulls])
    if not len(vol) or not vol.sum() > 0:
        return 0.0, np.zeros(3)
    cent = np.array([h.centroid for h in hulls])
    return float(vol.sum()), (vol[:, None] * cent).sum(0) / vol.sum()


def velocity_at(linear_velocity, angular_velocity, from_point, to_point):
    """Velocity of the point ``to_point`` of a rigid body moving with ``linear_velocity`` at ``from_point``."""
    return np.asarray(linear_velocity, np.float64) + np.cross(
        np.asarray(angular_velocity, np.float64), np.asarray(to_point, np.float64) - np.asarray(from_point, np.float64))


# -------------------------------------------------------------------------------------------------------- a cut
@dataclass
class SourceGeometry:
    """What a cut reads from the object being cut, in its link frame and asset units.

    Attributes:
        visuals (list of MeshPart): visual meshes
        hulls (list of np.ndarray): (k, 3) vertices of each convex collider
    """

    visuals: list
    hulls: list

    def __post_init__(self):
        self._whole = None
        self._planes = None

    @property
    def whole(self):
        """(volume, centroid) of the uncut colliders, computed once."""
        if self._whole is None:
            self._whole = hulls_volume_centroid([h for h in map(convex_hull, self.hulls) if h is not None])
        return self._whole

    def contains(self, points, tolerance=1e-9):
        """Whether each of ``points`` (p, 3) is inside any of the colliders."""
        from scipy.spatial import ConvexHull, QhullError

        if self._planes is None:
            self._planes = []
            for H in self.hulls:
                try:
                    self._planes.append(ConvexHull(np.asarray(H, np.float64)).equations)
                except (QhullError, ValueError):
                    continue
        P = np.asarray(points, np.float64)
        result = np.zeros(len(P), bool)
        for E in self._planes:
            result |= np.all(P @ E[:, :3].T + E[:, 3] <= tolerance, axis=1)
        return result


@dataclass
class PieceGeometry:
    """One side of a cut.

    Attributes:
        visuals (list of MeshPart): the clipped visual meshes
        cap (None or tuple): (V, F) of the new cut face, facing out of the piece
        hulls (list of Hull): convex colliders
        volume (float): collider volume (asset units cubed)
        centroid (np.ndarray): (3,) collider centroid
    """

    visuals: list
    cap: object
    hulls: list
    volume: float
    centroid: np.ndarray


@dataclass
class CutResult:
    """The outcome of :func:`cut_geometry`.

    Attributes:
        pieces (list of PieceGeometry): two pieces, the ``+normal`` side first; empty if the cut was refused
        reason (None or str): why the cut was refused
        fractions (tuple): each side's share of the whole object's collider volume
        timings (dict): seconds spent per step
    """

    pieces: list
    reason: object
    fractions: tuple
    timings: dict


def cut_geometry(source, plane, half_gap=0.0, max_hulls=None, min_fraction=0.03):
    """Cut an object by ``plane``, leaving a slab ``2 * half_gap`` thick between the two pieces.

    The colliders are clipped first, because they decide whether the cut happens:
    a side holding less than ``min_fraction`` of the object's volume refuses it.
    That is also what keeps the flat of a blade from cutting: pressed face-on, its
    mid-plane only grazes the object.

    Args:
        source (SourceGeometry): the object, in its link frame
        plane (Plane): the cut plane in the same frame
        half_gap (float): half the slab removed around the plane, in the same units
        max_hulls (None or int): colliders per piece, merged down with merge_hulls()
        min_fraction (float): smallest share of the object's collider volume a piece may have

    Returns:
        CutResult
    """
    timings = {}
    t0 = time.perf_counter()
    whole_volume, _ = source.whole
    sides = (plane.offset(half_gap), plane.flipped().offset(half_gap))
    tiny = MIN_HULL_FRACTION * whole_volume
    side_hulls = []
    for side_plane in sides:
        hulls = [h for h in (clip_convex(H, side_plane) for H in source.hulls) if h is not None and h.volume > tiny]
        side_hulls.append(hulls)
    fractions = tuple(hulls_volume_centroid(h)[0] / whole_volume if whole_volume > 0 else 0.0 for h in side_hulls)
    timings["clip_hulls_s"] = time.perf_counter() - t0
    if min(fractions) < min_fraction:
        return CutResult([], f"a piece would hold {min(fractions):.1%} of the volume (< {min_fraction:.0%})",
                         fractions, timings)

    t0 = time.perf_counter()
    side_hulls = [merge_hulls(h, max_hulls) for h in side_hulls]
    timings["merge_hulls_s"] = time.perf_counter() - t0
    pieces = []
    timings["clip_visual_s"] = timings["cap_s"] = 0.0
    for side_plane, hulls in zip(sides, side_hulls):
        t0 = time.perf_counter()
        visuals, segments = [], []
        for part in source.visuals:
            clipped, seg = clip_mesh(part, side_plane)
            if clipped is not None and len(clipped.faces):
                visuals.append(clipped)
            segments.append(seg)
        t1 = time.perf_counter()
        cap = (cap_polygon(np.concatenate(segments), side_plane,
                           inside=lambda p: source.contains(p, tolerance=CAP_HULL_TOLERANCE),
                           expected_area=section_area(hulls, side_plane)) if segments else None)
        timings["clip_visual_s"] += t1 - t0
        timings["cap_s"] += time.perf_counter() - t1
        volume, centroid = hulls_volume_centroid(hulls)
        pieces.append(PieceGeometry(visuals, cap, hulls, volume, centroid))
    return CutResult(pieces, None, fractions, timings)
