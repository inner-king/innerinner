# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the runtime-cut geometry (simfoundry/cutting/geometry.py).

Pure numpy/scipy/trimesh/shapely; no USD, OmniGibson or Isaac Sim. Each test
checks the geometry against an answer known in closed form: a cube or sphere
clipped by a plane, a linear function interpolated along cut edges, the area of
an annulus or a crescent.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("shapely")
pytest.importorskip("scipy")

from simfoundry.cutting.geometry import (  # noqa: E402
    MeshPart,
    Plane,
    SourceGeometry,
    blade_plane_from_points,
    cap_polygon,
    clip_convex,
    clip_mesh,
    convex_hull,
    cut_geometry,
    hulls_volume_centroid,
    merge_hulls,
    plane_to_local,
    plane_to_world,
    quat_xyzw_to_matrix,
    split_mass,
    velocity_at,
)


def quat_about(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[axis * np.sin(angle / 2), np.cos(angle / 2)]


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def box_points(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def blade_wedge(length=0.16, height=0.045, spine=0.0022, edge=0.0004):
    """Vertices of a wedge-section blade in the knife frame: x along the blade, y its normal, z up to the spine."""
    pts = []
    for x in (0.0, length):
        pts += [[x, -edge / 2, 0.0], [x, edge / 2, 0.0], [x, -spine / 2, height], [x, spine / 2, height]]
    return np.array(pts)


def closed(part, cap):
    """The clipped skin and its cap as one trimesh (vertices welded by position)."""
    V = np.concatenate([part.points, cap[0]])
    F = np.concatenate([part.faces, cap[1] + len(part.points)])
    mesh = trimesh.Trimesh(V, F, process=False)
    mesh.merge_vertices(digits_vertex=9)
    return mesh


# ------------------------------------------------------------------------------------------------ plane from knife
def test_blade_plane_from_points_finds_the_thin_axis_and_spine_thickness():
    plane, thickness = blade_plane_from_points(blade_wedge())
    assert np.allclose(np.abs(plane.normal), [0, 1, 0], atol=1e-9)
    assert abs(plane.signed_distance([0.08, 0.0, 0.02])) < 1e-12  # the wedge is centred on y = 0
    assert thickness == pytest.approx(0.0022)


def test_blade_plane_normal_sign_is_deterministic():
    pts = blade_wedge()
    a, _ = blade_plane_from_points(pts)
    b, _ = blade_plane_from_points(pts[::-1].copy())
    assert np.allclose(a.normal, b.normal)
    assert a.normal[np.argmax(np.abs(a.normal))] > 0


def test_plane_follows_the_knife_pose():
    rng = np.random.default_rng(0)
    plane_l, _ = blade_plane_from_points(blade_wedge())
    for _ in range(20):
        pos, quat = rng.normal(size=3), random_quat(rng)
        plane_w, q = plane_to_world(plane_l, pos, quat)
        R = quat_xyzw_to_matrix(quat)
        assert np.allclose(plane_w.normal, R @ plane_l.normal, atol=1e-12)
        # every blade vertex keeps its distance to the mid-plane
        blade_w = pos + blade_wedge() @ R.T
        assert np.allclose(plane_w.signed_distance(blade_w), plane_l.signed_distance(blade_wedge()), atol=1e-12)
        assert q == pytest.approx(1.0)


def test_plane_to_world_with_uniform_scale_scales_the_thickness():
    plane_l, thickness = blade_plane_from_points(blade_wedge())
    _, q = plane_to_world(plane_l, [0, 0, 0], [0, 0, 0, 1], scale=2.0)
    assert thickness / q == pytest.approx(2 * thickness)


def test_plane_to_local_round_trips_and_scales_distances():
    rng = np.random.default_rng(1)
    for _ in range(20):
        pos, quat, scale = rng.normal(size=3), random_quat(rng), rng.uniform(0.3, 3.0, size=3)
        plane_w = Plane.make(rng.normal(size=3), rng.normal(size=3))
        plane_l, k = plane_to_local(plane_w, pos, quat, scale)
        # local points mapped to the world keep the sign of their distance, scaled by k
        x_l = rng.normal(size=(50, 3))
        x_w = pos + (scale * x_l) @ quat_xyzw_to_matrix(quat).T
        assert np.allclose(plane_w.signed_distance(x_w), k * plane_l.signed_distance(x_l), atol=1e-9)
        back, _ = plane_to_world(plane_l, pos, quat, scale)
        assert np.allclose(back.normal, plane_w.normal, atol=1e-9)
        assert abs(plane_w.signed_distance(back.origin)) < 1e-9


# ------------------------------------------------------------------------------------------------------ colliders
def test_clip_convex_cube():
    cube = box_points([0, 0, 0], [1, 1, 1])
    hull = clip_convex(cube, Plane.make([0.25, 0, 0], [1, 0, 0]))
    assert hull.volume == pytest.approx(0.75)
    assert np.allclose(hull.centroid, [0.625, 0.5, 0.5])
    assert hull.points[:, 0].min() == pytest.approx(0.25)
    # the mesh is outward-wound: trimesh agrees on the volume
    assert trimesh.Trimesh(hull.points, hull.faces).volume == pytest.approx(0.75)


def test_clip_convex_oblique_plane_keeps_a_convex_hull_on_one_side():
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(60, 3))
    plane = Plane.make([0.1, -0.2, 0.0], [1, 2, 3])
    hull = clip_convex(pts, plane)
    assert np.all(plane.signed_distance(hull.points) >= -1e-12)
    a, b = clip_convex(pts, plane), clip_convex(pts, plane.flipped())
    assert a.volume + b.volume == pytest.approx(convex_hull(pts).volume)


def test_clip_convex_misses():
    cube = box_points([0, 0, 0], [1, 1, 1])
    assert clip_convex(cube, Plane.make([2, 0, 0], [1, 0, 0])) is None
    assert clip_convex(cube, Plane.make([-1, 0, 0], [1, 0, 0])).volume == pytest.approx(1.0)
    # a plane through a face keeps nothing of positive volume on the outer side
    assert clip_convex(cube, Plane.make([1, 0, 0], [1, 0, 0])) is None


def test_merge_hulls_caps_the_count_and_covers_the_parts():
    hulls = [convex_hull(box_points([i, 0, 0], [i + 1, 1, 1])) for i in range(6)]
    merged = merge_hulls(hulls, 2)
    assert len(merged) == 2
    assert sum(h.volume for h in merged) == pytest.approx(6.0)  # side by side boxes merge without added volume
    allpts = np.concatenate([h.points for h in merged])
    assert allpts[:, 0].min() == 0 and allpts[:, 0].max() == 6
    assert merge_hulls(hulls, None) == hulls


# ----------------------------------------------------------------------------------------------------- mass, motion
def test_split_mass_is_proportional_and_conserved():
    m = split_mass(0.2, [1.0, 3.0])
    assert m.sum() == pytest.approx(0.2)
    assert m[1] == pytest.approx(3 * m[0])
    assert np.allclose(split_mass(0.2, [0.0, 0.0]), [0.1, 0.1])


def test_symmetric_cut_splits_mass_evenly():
    source = SourceGeometry([], [box_points([-1, -1, -1], [1, 1, 1])])
    result = cut_geometry(source, Plane.make([0, 0, 0], [0, 0, 1]), half_gap=0.0)
    m = split_mass(0.3, [p.volume for p in result.pieces])
    assert np.allclose(m, [0.15, 0.15])


def test_velocity_at_a_piece_centroid():
    v = velocity_at([1, 0, 0], [0, 0, 2], from_point=[0, 0, 0], to_point=[0, 0.5, 0])
    assert np.allclose(v, [0.0, 0, 0])  # w x r = (0,0,2) x (0,0.5,0) = (-1, 0, 0)
    assert np.allclose(velocity_at([1, 2, 3], [0, 0, 0], [5, 5, 5], [0, 0, 0]), [1, 2, 3])


# --------------------------------------------------------------------------------------------------- visual meshes
def sphere_part(subdivisions=4):
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.04)
    F = sphere.faces
    # a face-varying attribute that is linear in position: interpolation along an edge must reproduce it exactly
    st = sphere.vertices[F][:, :, :2] * 3.0 + 0.5
    return MeshPart(np.asarray(sphere.vertices), np.asarray(F), {"st": st}, name="skin", material="/m"), sphere


def test_clip_mesh_cuts_triangles_and_closes_with_the_cap():
    part, sphere = sphere_part()
    plane = Plane.make([0, 0, 0.013], [0.2, -0.3, 1.0])
    clipped, segments = clip_mesh(part, plane)
    assert np.all(plane.signed_distance(clipped.points) >= -1e-12)
    assert np.allclose(plane.signed_distance(segments.reshape(-1, 3)), 0, atol=1e-12)
    cap = cap_polygon(segments, plane)
    mesh = closed(clipped, cap)
    assert mesh.is_watertight
    # the icosphere is convex, so the clipped solid is exactly the clipped hull
    assert mesh.volume == pytest.approx(clip_convex(sphere.vertices, plane).volume, rel=1e-6)
    assert clipped.material == "/m" and clipped.name == "skin"


def test_clip_mesh_interpolates_face_varying_attributes_along_the_cut():
    part, _ = sphere_part()
    plane = Plane.make([0.01, 0, 0], [1, 0.5, -0.2])
    clipped, _ = clip_mesh(part, plane)
    st = clipped.corner_attrs["st"]
    assert st.shape == (len(clipped.faces), 3, 2)
    expected = clipped.points[clipped.faces][:, :, :2] * 3.0 + 0.5
    assert np.allclose(st, expected, atol=1e-12)


def test_clip_mesh_keeps_welding_and_drops_unused_vertices():
    part, _ = sphere_part(3)
    clipped, _ = clip_mesh(part, Plane.make([0, 0, 0], [0, 0, 1]))
    assert np.unique(clipped.faces).size == len(clipped.points)
    mesh = trimesh.Trimesh(clipped.points, clipped.faces, process=False)
    # welded: the only open edges are the rim on the plane
    edges = mesh.edges_sorted
    counts = np.unique(edges, axis=0, return_counts=True)[1]
    rim_vertices = np.unique(mesh.edges_sorted[trimesh.grouping.group_rows(edges, require_count=1)])
    assert set(counts) <= {1, 2}
    assert np.allclose(clipped.points[rim_vertices, 2], 0, atol=1e-12)


def test_cap_of_a_surface_authored_twice():
    # the scanned fruits' visual meshes carry every face twice, once per winding: each cut edge comes back twice
    part, _ = sphere_part()
    doubled = MeshPart(part.points, np.concatenate([part.faces, part.faces[:, ::-1]]))
    plane = Plane.make([0, 0, 0.005], [0, 0, 1])
    _, segments = clip_mesh(doubled, plane)
    V, F = cap_polygon(segments, plane)
    single = cap_polygon(clip_mesh(part, plane)[1], plane)
    assert trimesh.Trimesh(V, F, process=False).area == pytest.approx(
        trimesh.Trimesh(*single, process=False).area, rel=1e-9)


def test_cap_of_a_rim_open_along_a_slit_falls_back_to_the_enclosed_area():
    # a scan's two skin layers joined around a 0.3 mm slit: the rim is a C traced out along one layer and back
    # along the other, 0.05 mm inside it, and winds around nothing
    r, gap = 0.035, 3e-4
    theta = np.linspace(gap / r, 2 * np.pi, 400)
    outer = np.c_[np.cos(theta), np.sin(theta)] * r
    inner = (np.c_[np.cos(theta), np.sin(theta)] * (r - 5e-5))[::-1]
    walk = np.concatenate([outer, inner, outer[:1]])
    segments = np.stack([walk[:-1], walk[1:]], 1)
    segments = np.concatenate([segments, np.zeros((len(segments), 2, 1))], axis=2)  # z = 0
    plane = Plane.make([0, 0, 0], [0, 0, -1])
    assert cap_polygon(segments, plane) is None or trimesh.Trimesh(*cap_polygon(segments, plane)).area < 1e-4
    V, F = cap_polygon(segments, plane, expected_area=np.pi * r ** 2)
    assert trimesh.Trimesh(V, F, process=False).area == pytest.approx(np.pi * r ** 2, rel=0.01)
    assert np.allclose(trimesh.Trimesh(V, F, process=False).face_normals, [0, 0, 1], atol=1e-9)


def test_clip_mesh_whole_and_nothing():
    part, _ = sphere_part(2)
    kept, segments = clip_mesh(part, Plane.make([0, 0, -1], [0, 0, 1]))
    assert len(kept.faces) == len(part.faces) and len(segments) == 0
    gone, segments = clip_mesh(part, Plane.make([0, 0, 1], [0, 0, 1]))
    assert gone is None and len(segments) == 0


def torus_mesh(major=0.05, minor=0.02, n_major=96, n_minor=48):
    """A watertight torus about z, built on its parameter grid (trimesh's needs a triangulation engine)."""
    a, b = np.meshgrid(np.linspace(0, 2 * np.pi, n_major, endpoint=False),
                       np.linspace(0, 2 * np.pi, n_minor, endpoint=False), indexing="ij")
    V = np.stack([(major + minor * np.cos(b)) * np.cos(a), (major + minor * np.cos(b)) * np.sin(a),
                  minor * np.sin(b)], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n_major), np.arange(n_minor), indexing="ij")
    q = [i * n_minor + j, ((i + 1) % n_major) * n_minor + j, ((i + 1) % n_major) * n_minor + (j + 1) % n_minor,
         i * n_minor + (j + 1) % n_minor]
    F = np.concatenate([np.stack([q[0], q[1], q[2]], -1).reshape(-1, 3), np.stack([q[0], q[2], q[3]], -1).reshape(-1, 3)])
    mesh = trimesh.Trimesh(V, F, process=False)
    trimesh.repair.fix_normals(mesh)
    return mesh


def c_prism(r_in=0.03, r_out=0.05, height=0.04, gap=np.pi / 6, n=64):
    """A watertight prism whose section is a C (an annular sector), and that section as a shapely polygon."""
    from shapely.geometry import Polygon

    theta = np.linspace(gap, 2 * np.pi - gap, n)
    ring = np.concatenate([np.c_[np.cos(theta), np.sin(theta)] * r_out, (np.c_[np.cos(theta), np.sin(theta)] * r_in)[::-1]])
    m = len(ring)  # outer arc 0..n-1, inner arc n..2n-1 (reversed)
    V = np.concatenate([np.c_[ring, np.zeros(m)], np.c_[ring, np.full(m, height)]])
    walls = [[k, (k + 1) % m, m + (k + 1) % m] for k in range(m)] + [[k, m + (k + 1) % m, m + k] for k in range(m)]
    caps = []
    for k in range(n - 1):  # strip between the outer arc k..k+1 and the inner arc at the same angles
        o0, o1, i0, i1 = k, k + 1, 2 * n - 1 - k, 2 * n - 2 - k
        caps += [[o0, i1, o1], [o0, i0, i1], [m + o0, m + o1, m + i1], [m + o0, m + i1, m + i0]]
    mesh = trimesh.Trimesh(V, np.array(walls + caps), process=False)
    trimesh.repair.fix_normals(mesh)
    assert mesh.is_watertight
    return mesh, Polygon(ring)


def test_cap_of_an_annulus_leaves_the_hole_open():
    torus = torus_mesh()
    plane = Plane.make([0, 0, 0], [0, 0, 1])
    clipped, segments = clip_mesh(MeshPart(torus.vertices, torus.faces), plane)
    V, F = cap_polygon(segments, plane)
    area = trimesh.Trimesh(V, F, process=False).area
    assert area == pytest.approx(np.pi * (0.07 ** 2 - 0.03 ** 2), rel=0.01)
    hole = trimesh.Trimesh(V, F, process=False).triangles_center
    assert np.linalg.norm(hole[:, :2], axis=1).min() > 0.029  # nothing covers the hole
    # the cap faces away from the kept (+z) side
    normals = trimesh.Trimesh(V, F, process=False).face_normals
    assert np.allclose(normals, [0, 0, -1], atol=1e-9)
    assert closed(clipped, (V, F)).is_watertight


def test_cap_of_a_crescent_is_not_a_fan():
    # a C-shaped prism: its section is not star-shaped about its centroid, which lies in the hollow of the C
    prism, poly = c_prism()
    plane = Plane.make([0, 0, 0.02], [0, 0, 1])
    clipped, segments = clip_mesh(MeshPart(prism.vertices, prism.faces), plane)
    V, F = cap_polygon(segments, plane)
    assert trimesh.Trimesh(V, F, process=False).area == pytest.approx(poly.area, rel=1e-6)
    assert closed(clipped, (V, F)).volume == pytest.approx(poly.area * 0.02, rel=1e-6)


# --------------------------------------------------------------------------------------------------------- a cut
def box_source(lo=(-0.03, -0.02, 0.0), hi=(0.03, 0.02, 0.05)):
    box = trimesh.creation.box(bounds=[lo, hi])
    sub = box.subdivide().subdivide()
    return SourceGeometry([MeshPart(sub.vertices, sub.faces, name="skin")], [box_points(lo, hi)])


def test_cut_leaves_a_slab_as_thick_as_the_blade():
    source = box_source()
    plane = Plane.make([0.005, 0, 0], [1, 0, 0])
    result = cut_geometry(source, plane, half_gap=0.0015)
    assert result.reason is None and len(result.pieces) == 2
    plus, minus = result.pieces
    assert min(h.points[:, 0].min() for h in plus.hulls) == pytest.approx(0.0065)
    assert max(h.points[:, 0].max() for h in minus.hulls) == pytest.approx(0.0035)
    assert plus.visuals[0].points[:, 0].min() == pytest.approx(0.0065)
    assert plus.volume == pytest.approx(0.0235 * 0.04 * 0.05)
    assert minus.volume == pytest.approx(0.0335 * 0.04 * 0.05)
    for piece, side in zip(result.pieces, (plane.offset(0.0015), plane.flipped().offset(0.0015))):
        mesh = closed(piece.visuals[0], piece.cap)
        assert mesh.is_watertight
        assert np.allclose(trimesh.Trimesh(*piece.cap, process=False).face_normals, -side.normal, atol=1e-9)
        assert mesh.volume == pytest.approx(piece.volume, rel=1e-6)


def test_cut_refuses_a_sliver():
    source = box_source()
    result = cut_geometry(source, Plane.make([0.029, 0, 0], [1, 0, 0]), half_gap=0.0, min_fraction=0.03)
    assert result.pieces == [] and "1.7%" in result.reason
    # the flat of a blade pressed onto the top face: its mid-plane only grazes the object
    assert cut_geometry(source, Plane.make([0, 0, 0.0495], [0, 0, 1])).pieces == []


def test_cutting_a_piece_again_caps_across_the_old_cut_face():
    source = box_source()
    first = cut_geometry(source, Plane.make([0, 0, 0], [1, 0, 0]), half_gap=0.001)
    half = first.pieces[0]
    cap_part = MeshPart(*half.cap, name="cut_face")
    again = SourceGeometry(half.visuals + [cap_part], [h.points for h in half.hulls])
    second = cut_geometry(again, Plane.make([0, 0, 0.02], [0, 0, 1]), half_gap=0.001)
    assert second.reason is None
    for piece in second.pieces:
        parts = piece.visuals
        assert {p.name for p in parts} == {"skin", "cut_face"}
        V = np.concatenate([p.points for p in parts] + [piece.cap[0]])
        offsets = np.cumsum([0] + [len(p.points) for p in parts])
        F = np.concatenate([p.faces + o for p, o in zip(parts, offsets[:-1])] + [piece.cap[1] + offsets[-1]])
        mesh = trimesh.Trimesh(V, F, process=False)
        mesh.merge_vertices(digits_vertex=9)
        assert mesh.is_watertight
        assert mesh.volume == pytest.approx(piece.volume, rel=1e-6)


def test_cut_merges_hulls_per_piece():
    hulls = [box_points([x, 0, 0], [x + 0.01, 0.04, 0.05]) for x in np.arange(-0.03, 0.03, 0.01)]
    box = trimesh.creation.box(bounds=[[-0.03, 0, 0], [0.03, 0.04, 0.05]])
    source = SourceGeometry([MeshPart(box.vertices, box.faces)], hulls)
    result = cut_geometry(source, Plane.make([0.004, 0, 0], [1, 0, 0]), max_hulls=2)
    assert [len(p.hulls) for p in result.pieces] == [2, 2]
    vols = [hulls_volume_centroid(p.hulls)[0] for p in result.pieces]
    assert sum(vols) == pytest.approx(0.06 * 0.04 * 0.05)
