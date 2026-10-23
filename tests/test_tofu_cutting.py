# SPDX-License-Identifier: Apache-2.0
"""Physics checks for the MPM knife-cutting solver (simfoundry/cutting), on the CPU at coarse resolution."""

import math
from dataclasses import replace

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from simfoundry.cutting.knife import KnifePose, KnifeTrajectory  # noqa: E402
from simfoundry.cutting.materials import FIRM_TOFU, PRESETS, SILKEN_TOFU  # noqa: E402
from simfoundry.cutting.mpm import MPMCuttingSim, SimConfig  # noqa: E402

DX = 4e-3
BLOCK = (0.032, 0.024, 0.016)
CONFIG = SimConfig(dx=DX, side_margin=0.012, top_margin=0.016)
ACROSS = -math.pi / 2.0  # blade along y, cutting plane x = const


def _sim(material=FIRM_TOFU):
    return MPMCuttingSim(material, BLOCK, config=CONFIG, device="cpu")


def _stroke(sim, depth=None, x_offset=1.0e-3, speed=0.08):
    """One vertical stroke through the block centre (or only ``depth`` into it), then a lift."""
    c = 0.5 * (sim.block_min + sim.block_max)
    top = sim.block_max[2]
    bottom = sim.board_z if depth is None else top - depth - 0.3e-3
    return KnifeTrajectory.strokes(
        [(c[0] + x_offset, c[1], ACROSS)], top_z=top, board_z=bottom,
        start_time=0.05, clearance=0.006, descend_speed=speed, settle=0.1,
    )


def _run(sim, traj, frame=0.01):
    """Advances through the whole trajectory and returns the knife force per frame."""
    forces = []
    t = 0.0
    while t < traj.duration - 1e-9:
        t = min(t + frame, traj.duration)
        f, _ = sim.advance(t, traj)
        forces.append((t, f))
    return forces


def test_materials_are_consistent():
    for m in PRESETS.values():
        assert m.shear_modulus > 0.0 and m.lame_lambda > 0.0
        assert 0.0 < m.poisson_ratio < 0.5
        assert math.isclose(
            m.youngs_modulus, 2.0 * m.shear_modulus * (1.0 + m.poisson_ratio), rel_tol=1e-12
        )
    assert SILKEN_TOFU.youngs_modulus < FIRM_TOFU.youngs_modulus
    assert SILKEN_TOFU.wave_speed < FIRM_TOFU.wave_speed


def test_stroke_keeps_the_cut_plane_while_sawing():
    traj = KnifeTrajectory.strokes(
        [(0.05, 0.04, ACROSS)], top_z=0.03, board_z=0.01, saw_amplitude=0.01, saw_frequency=2.0
    )
    (t_cut,) = traj.cut_times
    pose0, lin0, _ = traj.state(t_cut)
    assert pose0.origin[2] == pytest.approx(0.03 + 0.012)
    _, e_v, _ = KnifePose(origin=pose0.origin, yaw=ACROSS).axes()
    for t in np.linspace(t_cut, t_cut + 0.3, 7):
        pose, lin, _ = traj.state(t)
        # Sawing slides the blade along its edge: the plane (normal e_v) stays where it was.
        assert np.dot(pose.origin - pose0.origin, e_v) == pytest.approx(0.0, abs=1e-12)
        assert lin[2] < 0.0


def test_block_rests_on_the_board():
    sim = _sim()
    x0 = sim.positions()
    sim.advance(0.15)
    x = sim.positions()
    assert not np.isnan(x).any()
    # Gravity sags 1.6 cm of firm tofu by a fraction of a millimetre, no more.
    assert np.abs(x - x0).max() < 1.0e-3
    assert x[:, 2].min() > sim.board_z - 0.25 * DX


@pytest.fixture(scope="module")
def cut_once():
    """A block after one full stroke through it, and the knife force per frame."""
    sim = _sim()
    traj = _stroke(sim)
    forces = _run(sim, traj)
    return sim, forces


def test_viscoelastic_tofu_keeps_settling_under_its_weight():
    def sag(material, t_end):
        sim = _sim(material)
        top = sim.x0[:, 2] > sim.block_max[2] - DX
        sim.advance(t_end)
        return float((sim.x0[top, 2] - sim.positions()[top, 2]).mean())

    elastic = replace(FIRM_TOFU, relaxed_modulus=0.0)
    quick = replace(FIRM_TOFU, relaxation_time=0.25)
    assert quick.viscoelastic and not elastic.viscoelastic
    s_elastic = sag(elastic, 1.0)
    s_visco = sag(quick, 1.0)
    # Four times softer once relaxed: the block keeps sinking well past the elastic sag.
    assert s_visco > 2.0 * s_elastic > 0.0


def test_one_stroke_cuts_the_block_in_two_that_stay_apart(cut_once):
    sim, forces = cut_once
    assert sim.n_pieces == 2
    assert sim.cut_complete(0)
    assert sim.connected_pieces() == 2
    x, pc = sim.positions(), sim.pieces()
    assert not np.isnan(x).any()
    frac = (pc == 0).mean()
    assert 0.35 < frac < 0.65
    # No interpenetration across the cut once the knife is out and the halves settled.
    cut_x = 0.5 * (sim.block_min[0] + sim.block_max[0]) + 1.0e-3
    left = x[pc == int(np.argmin([x[pc == p, 0].mean() for p in (0, 1)])), 0]
    right = x[pc == int(np.argmax([x[pc == p, 0].mean() for p in (0, 1)])), 0]
    assert left.max() < cut_x + 0.5 * DX
    assert right.min() > cut_x - 0.5 * DX
    # The tofu resists the knife: an upward force while the edge is in it.
    assert max(f[2] for _, f in forces) > 0.05
    # And nothing fell through the board.
    assert x[:, 2].min() > sim.board_z - 0.5 * DX


def test_cut_pieces_export_as_closed_meshes(cut_once, tmp_path):
    pytest.importorskip("skimage")
    pytest.importorskip("trimesh")
    from simfoundry.cutting.export import export_pieces

    sim, _ = cut_once
    spacing = DX / CONFIG.particles_per_axis
    summary = export_pieces(sim.positions(), sim.pieces(), spacing, tmp_path, density=FIRM_TOFU.density)
    assert len(summary) == 2
    for entry in summary:
        assert entry["watertight"]
        assert (tmp_path / entry["file"]).exists()
        # The surface encloses about the volume the particles stand for.
        assert entry["volume_cm3"] == pytest.approx(entry["particle_volume_cm3"], rel=0.25)


def test_a_shallow_press_dents_without_cutting():
    tough = replace(FIRM_TOFU, fracture_energy=200.0)
    sim = _sim(tough)
    traj = _stroke(sim, depth=3.0e-3)
    forces = _run(sim, traj)
    assert sim.crack_front(0).min() > 999.0
    assert not sim.cut_complete(0)
    assert max(f[2] for _, f in forces) > 0.0


def test_a_partial_cut_stops_with_the_knife():
    sim = _sim()
    depth = 8.0e-3
    traj = _stroke(sim, depth=depth)
    _run(sim, traj)
    assert sim.n_pieces == 2  # split into two fields, but still one block below the slit
    assert not sim.cut_complete(0)
    assert sim.connected_pieces() == 1
    o = sim._cut_frames[0][0]
    front = sim.crack_front(0)
    lowest = front.min() + o[2] - sim.board_z  # height of the deepest crack tip above the board
    block_h = BLOCK[2]
    # The crack ran to about the knife's depth, not further: a dilated edge plus one crack lead.
    assert block_h - depth - 2.0 * DX < lowest < block_h - depth + DX


def test_a_tougher_tofu_needs_more_force_to_cut():
    means = []
    for gc in (1.0, 8.0):
        sim = _sim(replace(FIRM_TOFU, fracture_energy=gc))
        traj = _stroke(sim)
        (t_cut,) = traj.cut_times
        forces = _run(sim, traj)
        top = sim.block_max[2]
        t_in = t_cut + (traj.state(t_cut)[0].origin[2] - top) / 0.08
        t_out = t_in + (top - sim.board_z) / 0.08
        cutting = [f[2] for t, f in forces if t_in + 0.02 < t < t_out - 0.02]
        means.append(np.mean(cutting))
        assert sim.cut_complete(0)
    assert means[1] > 1.2 * means[0]


def test_shaped_solid_fills_only_its_shape():
    # A block with a quarter cut away along its length: an L in cross-section.
    def inside(p):
        return ~((p[:, 1] > 0.5 * BLOCK[1]) & (p[:, 2] > 0.5 * BLOCK[2]))

    sim = MPMCuttingSim(FIRM_TOFU, BLOCK, config=CONFIG, device="cpu", inside=inside)
    full = _sim()
    volume = 0.75 * float(np.prod(BLOCK))
    assert sim.n_particles * sim.particle_volume == pytest.approx(volume, rel=0.05)
    assert sim.particle_volume == pytest.approx(full.particle_volume, rel=1e-9)
    local = sim.x0 - sim.block_min
    assert not ((local[:, 1] > 0.5 * BLOCK[1] + 0.2 * DX) & (local[:, 2] > 0.5 * BLOCK[2] + 0.2 * DX)).any()
    # The faces of the notch are skin, like the block's own faces; the core is not.
    notch_face = (np.abs(local[:, 2] - 0.5 * BLOCK[2]) < 0.5 * DX) & (local[:, 1] > 0.5 * BLOCK[1] + DX)
    assert notch_face.any() and sim.skin[notch_face].all()
    core = (np.abs(local - np.array([0.5, 0.25, 0.25]) * np.array(BLOCK)) < 0.6 * DX).all(axis=1)
    assert core.any() and not sim.skin[core].any()


def test_opencv_camera_projects_like_opencv():
    from simfoundry.cutting.render import Camera

    K = np.array([[800.0, 0.0, 300.0], [0.0, 760.0, 200.0], [0.0, 0.0, 1.0]])
    a = math.radians(20.0)
    # Camera axes (x right, y down, z forward) in the world: looking along +y, tipped down.
    R = np.array([[1.0, 0.0, 0.0], [0.0, -math.sin(a), math.cos(a)], [0.0, -math.cos(a), -math.sin(a)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.1, -0.5, 0.3]
    cam = Camera.from_opencv(T, K, 1200, 800, calib_size=(600, 400))
    pts = np.array([[0.12, 0.05, 0.02], [0.05, 0.1, 0.0], [0.2, -0.1, 0.1]])
    pc = (pts - T[:3, 3]) @ R  # camera coordinates
    u = 2.0 * (K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2])
    v = 2.0 * (K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2])
    px, py, z = cam.project(pts)
    np.testing.assert_allclose(px, u, atol=1e-6)
    np.testing.assert_allclose(py, v, atol=1e-6)
    np.testing.assert_allclose(z, pc[:, 2], atol=1e-9)
    # The ray of a pixel passes through what projects onto that pixel.
    i, j = int(px[0]), int(py[0])
    ray = cam.rays[j, i]
    d = pts[0] - cam.eye
    assert np.linalg.norm(np.cross(ray, d / np.linalg.norm(d))) < 2e-3


def test_food_from_a_mesh_is_placed_level_and_long_side_along_x():
    trimesh = pytest.importorskip("trimesh")
    from simfoundry.cutting.real2sim import place_food

    box = trimesh.creation.box(extents=(0.07, 0.10, 0.05))  # long side along y
    T = trimesh.transformations.rotation_matrix(math.radians(25.0), (0, 0, 1))
    T[:3, 3] = (0.3, -0.2, 0.025 + 0.012)  # resting on a board 12 mm above the table
    box.apply_transform(T)
    place = place_food(box)
    np.testing.assert_allclose(place.block_size, (0.10, 0.07, 0.05), atol=1e-9)
    assert math.degrees(place.yaw) == pytest.approx(25.0 + 90.0 - 180.0, abs=1e-6)
    assert place.inside(0.5 * place.block_size[None, :]).all()
    assert not place.inside(1.1 * place.block_size[None, :]).any()
    S = place.sim_from_world(np.array([0.02, 0.02, 0.01]))
    corners = box.vertices @ S[:3, :3].T + S[:3, 3]
    np.testing.assert_allclose(corners.min(axis=0), (0.02, 0.02, 0.01), atol=1e-9)
    np.testing.assert_allclose(corners.max(axis=0), (0.12, 0.09, 0.06), atol=1e-9)
    # A groove across the top, as an occluder leaves in a single-view reconstruction: the mesh
    # keeps it, its convex hull fills it. (Built from voxels: no triangulation engine needed.)
    pytest.importorskip("skimage")
    occ = np.zeros((42, 30, 22), dtype=bool)  # 2.5 mm voxels, one empty layer around
    occ[1:41, 1:29, 1:21] = True  # 10 x 7 x 5 cm
    occ[17:25, 1:29, 15:21] = False  # 2 cm wide, 1.5 cm deep, across the width
    grooved = trimesh.voxel.VoxelGrid(occ).marching_cubes  # in voxel units
    grooved.apply_scale(2.5e-3)
    as_is = place_food(grooved)
    filled = place_food(grooved, convex=True)
    groove = np.array([[0.05, 0.035, 0.045]])
    assert not as_is.inside(groove).any() and filled.inside(groove).all()
    assert filled.local_mesh.volume > as_is.local_mesh.volume + 0.8 * 0.02 * 0.07 * 0.015


def test_chef_knife_profile_and_rock_on_the_belly():
    from simfoundry.cutting.knife import CHEF_KNIFE

    g = CHEF_KNIFE
    # Edge and spine meet at the point; the edge is straight up to the belly.
    assert g.edge_height(g.half_length) == pytest.approx(g.tip_rise)
    assert g.spine_height(g.half_length) == pytest.approx(g.tip_rise)
    assert g.edge_height(g.half_length - g.belly_length - 1e-3) == 0.0
    assert g.spine_height(-g.half_length) == pytest.approx(g.height)
    # A rocking stroke keeps the edge's lowest point on the board and ends with the tip raised.
    board = 0.01
    traj = KnifeTrajectory.rock_strokes(
        [(0.05, 0.03, 0.0)], top_z=0.05, board_z=board, knife_half_length=g.half_length, tip_reach=0.04,
        geometry=g, glide=0.04, roll_glide=0.02, slide=0.0, settle=0.0, retreat=0.0,
    )
    lowest = []
    for t in np.linspace(0.0, traj.duration, 400):
        pose, _, _ = traj.state(t)
        lowest.append(g.lowest_edge_point(pose)[2])
    assert min(lowest) > board  # never through the board
    rock_end = traj.cut_times[0] + (0.05 + 0.006 - (board + 0.3e-3)) / 0.06 + math.radians(30.0) / math.radians(70.0)
    pose, _, _ = traj.state(rock_end)
    assert pose.pitch == pytest.approx(0.0, abs=1e-6)
    e_u, _, e_w = pose.axes()
    tip = pose.origin + g.half_length * e_u + g.tip_rise * e_w
    assert tip[2] - board == pytest.approx(g.tip_rise, abs=1e-3)


def _press_or_slice(slide, fracture_energy=8.0):
    """Knife force and first-crack depth of one stroke 8 mm into the block, sliding ``slide`` m per m down."""
    sim = MPMCuttingSim(replace(FIRM_TOFU, fracture_energy=fracture_energy), BLOCK, config=CONFIG, device="cpu")
    c = 0.5 * (sim.block_min + sim.block_max)
    top = sim.block_max[2]
    depth = 0.008
    hover = KnifePose(np.array([c[0] + 1.0e-3, c[1], top + 0.002]), ACROSS)
    e_u = hover.axes()[0]
    travel = top + 0.002 - (top - depth)
    down = KnifePose(hover.origin + np.array([0.0, 0.0, -travel]) + slide * travel * e_u, ACROSS)
    traj = KnifeTrajectory.linear(0.0, travel / 0.05, hover, down, begins_cut=True)
    forces = [f[2] for _, f in _run(sim, traj, frame=0.005)]
    onset = sim.crack_onset_height(0)
    return max(forces), (top - onset if np.isfinite(onset) else float("inf"))


def test_slicing_cuts_with_less_force_than_pressing():
    # A tough tofu: pressed straight down it is only dented in the first 8 mm; sliding the edge
    # along itself twice as fast as it goes down (Atkins's slice-to-push ratio of 2) cuts it.
    press_force, press_onset = _press_or_slice(0.0)
    slice_force, slice_onset = _press_or_slice(2.0)
    assert slice_onset < press_onset
    assert slice_onset < 3.0e-3
    assert slice_force < 0.8 * press_force
