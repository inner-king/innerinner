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


def _glide(sim, xi, speed=0.08):
    """A level stroke through the block centre whose edge slides along itself at xi times its descent."""
    c = 0.5 * (sim.block_min + sim.block_max)
    hover, bottom = sim.block_max[2] + 0.006, sim.board_z + 0.3e-3
    e_u = KnifePose(np.zeros(3), ACROSS).axes()[0]
    run = xi * (hover - bottom)
    p0 = np.array([c[0] + 1.0e-3, c[1], hover]) - 0.5 * run * e_u
    p1 = np.array([c[0] + 1.0e-3, c[1], bottom]) + 0.5 * run * e_u
    up = p1 + np.array([0.0, 0.0, hover - bottom])
    t0, t1 = 0.05, 0.05 + (hover - bottom) / speed
    poses = [KnifePose(p, ACROSS) for p in (p0, p1, up)]
    moves = [
        KnifeTrajectory.linear(0.0, t0, poses[0], poses[0]),
        KnifeTrajectory.linear(t0, t1, poses[0], poses[1], begins_cut=True),
        KnifeTrajectory.linear(t1, t1 + 0.2, poses[1], poses[2]),
        KnifeTrajectory.linear(t1 + 0.2, t1 + 0.3, poses[2], poses[2]),
    ]
    return KnifeTrajectory(segments=[s for m in moves for s in m.segments])


def _cutting_force(sim, traj, forces, speed=0.08):
    """Mean upward knife force while the edge is inside the block, its first and last 20 ms left out."""
    (t_cut,) = traj.cut_times
    top = sim.block_max[2]
    t_in = t_cut + (traj.state(t_cut)[0].origin[2] - top) / speed
    t_out = t_in + (top - sim.board_z) / speed
    return float(np.mean([f[2] for t, f in forces if t_in + 0.02 < t < t_out - 0.02]))


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
        # Every preset tears, at about 10 % stretch.
        assert m.tensile_strength == pytest.approx(0.1 * m.youngs_modulus)
        assert m.critical_opening == pytest.approx(2.0 * m.fracture_energy / m.tensile_strength)
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
    # Taller than BLOCK: the crack may run a dilated edge plus a quarter cell (4 mm here) ahead
    # of the knife, which would leave little of BLOCK's 16 mm under it.
    block = (BLOCK[0], BLOCK[1], 0.024)
    sim = MPMCuttingSim(FIRM_TOFU, block, config=CONFIG, device="cpu")
    depth = 8.0e-3
    traj = _stroke(sim, depth=depth)
    _run(sim, traj)
    assert sim.n_pieces == 2  # split into two fields, but still one block below the slit
    assert not sim.cut_complete(0)
    assert sim.connected_pieces() == 1
    o = sim._cut_frames[0][0]
    front = sim.crack_front(0)
    lowest = front.min() + o[2] - sim.board_z  # height of the deepest crack tip above the board
    block_h = block[2]
    # The crack ran to about the knife's depth, not further: a dilated edge plus one crack lead.
    assert block_h - depth - 2.0 * DX < lowest < block_h - depth + DX


def test_a_tougher_tofu_needs_more_force_to_cut():
    means = []
    for gc in (1.0, 8.0):
        sim = _sim(replace(FIRM_TOFU, fracture_energy=gc))
        traj = _stroke(sim)
        means.append(_cutting_force(sim, traj, _run(sim, traj)))
        assert sim.cut_complete(0)
    assert means[1] > 1.2 * means[0]


def test_the_cutting_force_is_the_fracture_energy_on_any_grid():
    # A straight press: in steady cutting the push does G_c of work per unit of new crack area,
    # so the force is G_c times the block's width, whatever the grid (Atkins). A tough tofu,
    # because at 4 mm cells in a block four cells tall the coarse dilated edge alone pushes with
    # about 0.09 N, which would hide the 0.07 N of firm tofu's G_c.
    tough = replace(FIRM_TOFU, fracture_energy=8.0)
    ratios = []
    for dx in (4.0e-3, 3.0e-3):
        config = SimConfig(dx=dx, side_margin=0.012, top_margin=0.016)
        sim = MPMCuttingSim(tough, BLOCK, config=config, device="cpu")
        traj = _stroke(sim)
        force = _cutting_force(sim, traj, _run(sim, traj))
        ratios.append(force / (tough.fracture_energy * BLOCK[1]))
        assert sim.cut_complete(0)
    assert all(0.75 < r < 1.15 for r in ratios)
    assert abs(ratios[0] - ratios[1]) < 0.1


def test_a_cooks_slicing_stroke_cuts_more_easily_than_a_straight_press():
    # The same tofu and descent, pressed straight down and with the edge sliding along itself at
    # twice the descent speed, as a cook's push cut does (slice/push ratio xi = 2).
    onset, force = {}, {}
    for xi in (0.0, 2.0):
        sim = _sim()
        traj = _glide(sim, xi)
        forces = _run(sim, traj)
        assert sim.cut_complete(0)
        # How far the edge had gone below the top when the tofu gave way (negative: not yet in).
        onset[xi] = sim.block_max[2] - sim.crack_onset_height(0)
        force[xi] = _cutting_force(sim, traj, forces)
    # The straight press dents the tofu further before it cuts, and pushes harder while it cuts
    # (1 + xi^2 = 5 times, by Atkins's energy balance).
    assert onset[0.0] > onset[2.0] + 0.4e-3
    assert force[0.0] > 2.5 * force[2.0]


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
    # A tough tofu: pressed straight down it dents further before it cracks and pushes back
    # harder; sliding the edge along itself twice as fast as it goes down (Atkins's slice-to-push
    # ratio of 2) cracks it sooner, with a fraction of the force.
    press_force, press_onset = _press_or_slice(0.0)
    slice_force, slice_onset = _press_or_slice(2.0)
    assert slice_onset < press_onset
    assert slice_onset < 3.0e-3
    assert slice_force < 0.8 * press_force


def _ligament(sim):
    """Height of the uncut material under the crack front, in each bin the block crosses [m]."""
    o, e_u, e_v, _ = sim._cut_frames[0]
    rel = sim.positions() - o
    near = np.abs(rel @ e_v) < 2.0 * DX
    bins = np.unique(np.floor((rel[near] @ e_u - sim.bin_umin) / DX).astype(int))
    return sim.crack_front(0)[bins] + o[2] - sim.board_z


def _cut_and_pry(sim, depth, rest=0.0):
    """Cuts ``depth`` straight down along x through the block's middle, then levers the slit open.

    With ``rest`` the blade first comes out and the block is left alone that long before the
    blade goes back into the slit. The lever pushes one side outward from inside the cut (the
    blade moves 3 mm sideways while it tilts 8 degrees), lets go and lifts out.

    Returns the uncut ligament (``_ligament``) and the number of pieces after the stroke, after
    the rest and at the end.
    """
    c = 0.5 * (sim.block_min + sim.block_max)
    top = sim.block_max[2]
    hover = KnifePose(np.array([c[0], c[1], top + 0.004]), 0.0)  # blade along x: plane y = c_y
    down = KnifePose(np.array([c[0], c[1], top - depth]), 0.0)
    pried = KnifePose(down.origin - np.array([0.0, 0.003, 0.0]), 0.0, roll=math.radians(8.0))
    moves = [(hover, down, (0.004 + depth) / 0.08, True)]
    if rest > 0.0:
        moves += [(down, hover, 0.1, False), (hover, hover, rest, False), (hover, down, 0.2, False)]
    moves += [(down, pried, 0.3, False), (pried, down, 0.3, False), (down, hover, 0.1, False),
              (hover, hover, 0.3, False)]
    segs, t, ends = [], 0.05, []
    for a, b, dt, cut in moves:
        segs += KnifeTrajectory.linear(t, t + dt, a, b, begins_cut=cut).segments
        t += dt
        ends.append(t)
    traj = KnifeTrajectory(segments=segs)
    states = []
    for t_end in (ends[0], ends[2] if rest > 0.0 else ends[0], traj.duration):
        t = sim.time
        while t < t_end - 1e-9:
            t = min(t + 0.01, t_end)
            sim.advance(t, traj)
        states.append((_ligament(sim), sim.connected_pieces()))
    return states


def test_a_half_cut_block_tears_partway_when_pried_open():
    # Cut halfway and lever the slit open with the blade: the crack tears on into the uncut half,
    # about a cell, but the rest of the ligament holds the two sides together.
    sim = _sim()
    (stroke, _), _, (end, pieces) = _cut_and_pry(sim, 0.5 * BLOCK[2])
    # Along at least a third of the 32 mm cut (the bins at its ends, by the free faces, may not).
    assert (stroke - end > 0.9 * DX).sum() >= 3
    assert end.max() > DX
    assert not sim.cut_complete(0)
    assert pieces == 1


def test_a_thin_ligament_holds_until_it_is_pried_apart():
    # A cut three quarters deep leaves a ligament thinner than a cell and a half under the crack.
    sim = _sim()
    (stroke, p_stroke), (rested, p_rested), (end, p_end) = _cut_and_pry(sim, 0.75 * BLOCK[2], rest=0.5)
    assert 0.0 < stroke.min() and stroke.max() < 1.5 * DX
    # Left alone, nothing pulls it open: the crack stays where the knife left it, and the block
    # stays one piece (it used to count as cut through: the front was within 1.5 cells of the board).
    np.testing.assert_allclose(rested, stroke)
    assert p_stroke == p_rested == 1
    # The same lever that only tears a half cut a cell deeper tears this one through.
    assert sim.cut_complete(0)
    assert p_end == 2
