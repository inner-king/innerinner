# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 8's size check compares a placed mesh with its depth, in the camera frame, on synthetic data.

The camera sits at the origin. A world mesh (floor at z = 0, world up = +z) is placed by a rotation that takes
world up to the up direction the camera reports, and a little way in front of the camera. Depth points are surface
samples on the camera-facing faces; the ring is floor points around the object.
"""

import ast
import json
import pathlib

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation as R

from simfoundry.pipeline import size_check as sc
from simfoundry.utils.python_utils import atomic_output_path

REPO = pathlib.Path(__file__).resolve().parents[1]
STAGE8 = REPO / "scripts/pipeline/A_reconstruction/stages/8_match_object_poses.py"
CFG = REPO / "scripts/cfg/real2sim_cfg.yaml"

SETTINGS = {"size_warn_ratio": 0.15, "visible_only": True, "height_percentile": 95, "min_ring_points": 50}
HEIGHT_FIT_OFF = {"enabled": True, "threshold": 0.25, "min_scale": 0.5, "max_scale": 2.0}


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _camera(up_cam, dist=0.6):
    """World -> camera, with world up at `up_cam` in the camera frame and the object `dist` ahead."""
    rot = R.align_vectors([_unit(up_cam)], [[0.0, 0.0, 1.0]])[0].as_matrix()
    offset = np.array([0.0, 0.0, dist])
    return (lambda points: np.asarray(points) @ rot.T + offset), rot


def _box(extents=(0.2, 0.1, 0.05), z_offset=0.0):
    box = trimesh.creation.box(extents=extents)
    box.apply_translation([0.0, 0.0, extents[2] / 2 + z_offset])
    return box


def _depth(mesh, to_camera, n=20000):
    """Depth-like points: surface samples of the world mesh that lie on camera-facing faces, in the camera frame."""
    np.random.seed(0)
    points, faces_hit = trimesh.sample.sample_surface(mesh, n)
    cam = to_camera(mesh.vertices)
    tri = cam[mesh.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    facing = np.einsum("ij,ij->i", normals, tri.mean(axis=1)) < 0
    return to_camera(points)[facing[faces_hit]]


def _ring(floor_z=0.0, half=0.5, step=0.02, clear=(0.12, 0.08)):
    """Floor points around the object, outside a rectangle `clear` (half-widths) that holds the object."""
    axis = np.arange(-half, half + 1e-9, step)
    grid_x, grid_y = np.meshgrid(axis, axis)
    points = np.stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, floor_z)], axis=1)
    outside = (np.abs(points[:, 0]) > clear[0]) | (np.abs(points[:, 1]) > clear[1])
    return points[outside]


def _report(world_mesh, up_cam, settings=SETTINGS, ring=None, depth=None, dist=0.6):
    to_camera, _ = _camera(up_cam, dist)
    placed = trimesh.Trimesh(vertices=to_camera(world_mesh.vertices), faces=world_mesh.faces, process=False)
    depth_points = _depth(world_mesh, to_camera) if depth is None else depth
    ring_points = to_camera(_ring() if ring is None else ring)
    return sc.size_report(placed.vertices, placed.faces, depth_points, ring_points, _unit(up_cam), settings)


def _scaled_report(scaled_mesh, up_cam, settings=SETTINGS):
    """A mesh whose size is wrong, measured against the depth of the correct box (not of the scaled one)."""
    to_camera, _ = _camera(up_cam)
    return _report(scaled_mesh, up_cam, settings, depth=_depth(_box(), to_camera))


OVERHEAD = np.array([0.0, 0.0, -1.0])
TILTED = _unit([0.3, -0.5, -0.8])


def test_a_mesh_that_matches_the_depth_reports_no_warning():
    report = _report(_box(), OVERHEAD)
    assert report["warnings"] == []
    assert report["footprint"]["ratios"] == pytest.approx([1.0, 1.0], abs=0.03)
    assert report["height"]["ratio"] == pytest.approx(1.0, abs=0.05)


def test_a_wider_footprint_warns_about_the_footprint_and_not_the_height():
    scaled = _box()
    scaled.apply_scale([1.3, 1.3, 1.0])
    report = _scaled_report(scaled, OVERHEAD)
    assert report["footprint"]["ratios"] == pytest.approx([1.3, 1.3], abs=0.03)
    assert report["warnings"] == ["footprint"]
    assert report["height"]["ratio"] == pytest.approx(1.0, abs=0.05)


def test_a_narrower_footprint_warns_too():
    scaled = _box()
    scaled.apply_scale([0.8, 0.8, 1.0])
    report = _scaled_report(scaled, OVERHEAD)
    assert report["footprint"]["max_deviation"] == pytest.approx(0.2, abs=0.03)
    assert report["warnings"] == ["footprint"]


def test_a_taller_mesh_warns_about_the_height_and_not_the_footprint():
    stretched = _box()
    stretched.apply_scale([1.0, 1.0, 1.4])
    report = _scaled_report(stretched, OVERHEAD)
    assert report["height"]["ratio"] == pytest.approx(1.4, abs=0.03)
    assert report["warnings"] == ["height"]
    assert report["footprint"]["max_deviation"] == pytest.approx(0.0, abs=0.03)


def test_the_visible_footprint_does_not_warn_for_a_round_object_seen_from_below_its_top():
    # A cylinder seen from a camera below its top: the far half of its footprint is hidden from the depth.
    cylinder = trimesh.creation.cylinder(radius=0.05, height=0.05, sections=48)
    cylinder.apply_translation([0.0, 0.0, 0.025])
    low = _unit([0.0, -1.0, -0.02])
    assert _report(cylinder, low, dict(SETTINGS, visible_only=True))["warnings"] == []
    full = _report(cylinder, low, dict(SETTINGS, visible_only=False))
    assert "footprint" in full["warnings"]
    assert full["footprint"]["max_deviation"] > 1.0


def test_the_footprint_sides_do_not_depend_on_how_the_object_is_turned_or_tilted():
    turned = _box()
    turned.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0.0, 0.0, 1.0]))
    report = _report(turned, TILTED)
    assert report["warnings"] == []
    assert report["footprint"]["mesh_sides_m"] == pytest.approx([0.2, 0.1], abs=0.005)
    assert report["footprint"]["depth_sides_m"] == pytest.approx([0.2, 0.1], abs=0.01)


def test_the_height_is_read_from_the_surface_the_object_stands_on(tmp_path):
    on_floor = _report(_box(), OVERHEAD)
    # The same box on a tray 2 cm up: the tray bottom is the surface around it.
    tray_box = _box(z_offset=0.02)
    tray = _report(tray_box, OVERHEAD, ring=_ring(floor_z=0.02))
    assert on_floor["height"]["depth_m"] == pytest.approx(0.05, abs=0.005)
    assert tray["height"]["depth_m"] == pytest.approx(on_floor["height"]["depth_m"], abs=0.002)
    assert tray["height"]["floor_m"] - on_floor["height"]["floor_m"] == pytest.approx(0.02, abs=0.002)
    assert tray["height"]["ratio"] == pytest.approx(1.0, abs=0.05)


def test_too_few_points_around_the_object_skip_the_height_with_the_reason():
    report = _report(_box(), OVERHEAD, ring=_ring()[:30])
    assert "skipped" in report["height"]
    assert "30 points" in report["height"]["skipped"]
    assert report["warnings"] == []


def test_a_few_depth_returns_below_the_surface_do_not_move_the_floor():
    ring = _ring()
    below = ring.copy()
    below[: len(below) // 20, 2] = -0.02  # 5 % of the ring is depth noise under the surface
    report = _report(_box(), OVERHEAD, ring=below)
    assert report["height"]["ratio"] == pytest.approx(1.0, abs=0.05)
    # The floor is 0.6 m along the camera's up axis (the camera is 0.6 m from the object, looking at it from above).
    assert report["height"]["floor_m"] == pytest.approx(-0.6, abs=1e-3)


def test_a_few_depth_returns_above_the_top_do_not_set_the_height():
    to_camera, _ = _camera(OVERHEAD)
    depth = _depth(_box(), to_camera)
    stray = depth[: len(depth) // 50] + np.array([0.0, 0.0, -0.03])  # 2 % of points 3 cm above the top
    report = _report(_box(), OVERHEAD, depth=np.vstack([depth, stray]))
    assert report["height"]["ratio"] == pytest.approx(1.0, abs=0.05)


def test_a_warning_fires_only_above_the_threshold_and_not_at_it():
    scaled = _box()
    scaled.apply_scale([1.3, 1.3, 1.0])
    measured = _scaled_report(scaled, OVERHEAD, dict(SETTINGS, size_warn_ratio=0.5))
    deviation = measured["footprint"]["max_deviation"]
    at = _scaled_report(scaled, OVERHEAD, dict(SETTINGS, size_warn_ratio=deviation))
    just_below = _scaled_report(scaled, OVERHEAD, dict(SETTINGS, size_warn_ratio=deviation * (1 - 1e-9)))
    assert "footprint" not in at["warnings"]
    assert "footprint" in just_below["warnings"]


def test_a_degenerate_depth_skips_the_footprint_with_the_reason():
    report = _report(_box(), OVERHEAD, depth=np.zeros((2, 3)))
    assert "skipped" in report["footprint"]
    assert "footprint" not in report["warnings"]


def test_the_report_serialises_to_json_in_both_its_forms():
    assert json.loads(json.dumps(_report(_box(), OVERHEAD)))["warnings"] == []
    skipped = _report(_box(), OVERHEAD, ring=_ring()[:10])
    assert json.loads(json.dumps(skipped))["height"]["skipped"]


def test_only_faces_that_look_at_the_camera_contribute_their_vertices():
    box = _box()
    to_camera, _ = _camera(OVERHEAD)
    cam_vertices = to_camera(box.vertices)
    visible = sc.visible_vertices(cam_vertices, box.faces)
    top = cam_vertices[cam_vertices[:, 2] < cam_vertices[:, 2].min() + 1e-6]
    assert len(visible) == 4
    assert all(np.isclose(point[2], top[:, 2].min()) for point in visible)


def test_fit_height_keeps_the_lowest_point_and_stretches_only_along_up():
    rng = np.random.default_rng(1)
    vertices = rng.normal(size=(200, 3))
    for up in ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], _unit(rng.normal(size=3))):
        direction = _unit(up)
        stretched = sc.fit_height(vertices, up, 1.7)
        before = vertices @ direction
        after = stretched @ direction
        assert after.min() == pytest.approx(before.min(), abs=1e-9)
        assert (after.max() - after.min()) == pytest.approx(1.7 * (before.max() - before.min()), rel=1e-9)
        across_before = vertices - np.outer(before, direction)
        across_after = stretched - np.outer(after, direction)
        assert np.allclose(across_before, across_after, atol=1e-12)


def test_fit_height_with_a_factor_of_one_changes_nothing():
    vertices = np.random.default_rng(2).normal(size=(30, 3))
    assert np.array_equal(sc.fit_height(vertices, [0.0, 0.0, 1.0], 1.0), vertices)


@pytest.mark.parametrize("report, settings, applied, reason_fragment", [
    ({"height": {"mesh_m": 0.05, "depth_m": 0.07, "deviation": 0.4}}, dict(HEIGHT_FIT_OFF, enabled=False),
     False, "disabled"),
    ({"height": {"skipped": "only 4 points"}}, HEIGHT_FIT_OFF, False, "not comparable"),
    ({"height": {"mesh_m": 0.05, "depth_m": 0.055, "deviation": 0.1}}, HEIGHT_FIT_OFF, False, "within the threshold"),
    ({"height": {"mesh_m": 0.05, "depth_m": 0.01, "deviation": 0.8}}, HEIGHT_FIT_OFF, False, "outside"),
    ({"height": {"mesh_m": 0.05, "depth_m": 0.07, "deviation": 0.4}}, HEIGHT_FIT_OFF, True, None),
])
def test_the_height_fit_decision_applies_only_inside_every_limit(report, settings, applied, reason_fragment):
    decision = sc.decide_height_fit(report, settings)
    assert decision["applied"] is applied
    if applied:
        assert decision["scale"] == pytest.approx(0.07 / 0.05)
        assert decision["skipped"] is None
    else:
        assert decision["scale"] is None
        assert reason_fragment in decision["skipped"]


def test_a_deviation_exactly_at_the_threshold_is_not_stretched():
    report = {"height": {"mesh_m": 0.05, "depth_m": 0.0625, "deviation": 0.25}}
    assert sc.decide_height_fit(report, HEIGHT_FIT_OFF)["applied"] is False


def _stage8_function(name):
    """The function as the stage defines it, compiled on its own so the stage's module-level imports are not needed."""
    tree = ast.parse(STAGE8.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np, "trimesh": trimesh, "atomic_output_path": atomic_output_path, "fit_height": sc.fit_height}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(STAGE8), "exec"), namespace)
    return namespace[name]


def test_the_canonical_file_is_rewritten_with_its_height_stretched_and_its_footprint_kept(tmp_path):
    stretch = _stage8_function("_stretch_canonical_height")
    box = _box()
    path = tmp_path / "canonical.glb"
    trimesh.Scene(box).export(path)
    assert stretch(path, np.array([0.0, 0.0, 1.0]), 2.0) is None
    loaded = trimesh.load(path, force="mesh")
    assert loaded.extents == pytest.approx([0.2, 0.1, 0.1], abs=1e-6)
    assert loaded.bounds[0][2] == pytest.approx(0.0, abs=1e-6)  # the bottom stays where it was


def test_a_file_with_several_geometries_is_left_alone_with_the_reason(tmp_path):
    stretch = _stage8_function("_stretch_canonical_height")
    path = tmp_path / "parts.glb"
    trimesh.Scene([_box(), _box(z_offset=0.2)]).export(path)
    before = path.read_bytes()
    reason = stretch(path, np.array([0.0, 0.0, 1.0]), 2.0)
    assert "2 geometries" in reason
    assert path.read_bytes() == before


def test_the_warning_text_names_the_object_and_the_threshold():
    text_of = _stage8_function("_size_warning_text")
    report = {"warnings": ["footprint", "height"],
              "footprint": {"mesh_sides_m": [0.26, 0.13], "depth_sides_m": [0.2, 0.1], "ratios": [1.3, 1.3]},
              "height": {"mesh_m": 0.07, "depth_m": 0.05, "ratio": 1.4}}
    text = text_of("iter_3", report, 0.15)
    assert "iter_3" in text and "0.15" in text
    assert "footprint" in text and "height" in text


def test_the_defaults_are_the_ones_the_check_was_written_with():
    yaml = pytest.importorskip("yaml")
    s8 = yaml.safe_load(CFG.read_text())["s8_pose"]
    assert s8["size_warn_ratio"] == 0.15
    assert s8["size_check_visible_only"] is True
    assert s8["height_percentile"] == 95
    assert s8["height_ring_px"] == 6
    assert s8["height_ring_gap_px"] == 3
    assert s8["height_fit"] == {"enabled": False, "threshold": 0.25, "min_scale": 0.5, "max_scale": 2.0}


def _stage8_main():
    tree = ast.parse(STAGE8.read_text())
    return tree, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Name) and n.func.id == name)
                 or (isinstance(n.func, ast.Attribute) and n.func.attr == name))]


def _ifs_on(main, name):
    return [n for n in ast.walk(main) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == name]


def test_the_size_check_is_imported_at_module_level():
    tree, _ = _stage8_main()
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "simfoundry.pipeline.size_check"]
    assert len(imports) == 1
    assert {a.name for a in imports[0].names} == {"decide_height_fit", "fit_height", "ring_mask", "size_report"}


def test_the_switch_is_defined_before_the_object_loop_and_guards_every_size_output():
    _, main = _stage8_main()
    switches = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "size_check_on" for t in n.targets)]
    first_guard = min(i.lineno for i in _ifs_on(main, "size_check_on"))
    assert switches and max(n.lineno for n in switches) < first_guard
    # Off exactly when size_warn_ratio is null, and on for any number (0 included).
    assert all("size_warn_ratio" in ast.unparse(n.value) and "is not None" in ast.unparse(n.value) for n in switches)
    guarded_keys = set()
    for guard in _ifs_on(main, "size_check_on"):
        for n in ast.walk(guard):
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript) \
                    and isinstance(n.targets[0].slice, ast.Constant):
                guarded_keys.add(n.targets[0].slice.value)
    assert {"size_check", "height_fit"} <= guarded_keys


def test_the_summary_carries_size_warnings_only_when_the_check_is_on():
    _, main = _stage8_main()
    finalize = [c for c in _calls(main, "finalize_stage")][0]
    result = next(k.value for k in finalize.keywords if k.arg == "result")
    info = next(k.value for k in result.keywords if k.arg == "additional_info")
    assert isinstance(info, ast.IfExp) and isinstance(info.test, ast.Name) and info.test.id == "size_check_on"
    keys = {k.value for k in info.body.keys}
    assert "size_warnings" in keys


def test_the_stretched_mesh_is_measured_again_before_its_record_is_written():
    _, main = _stage8_main()
    applied = [i for i in ast.walk(main) if isinstance(i, ast.If) and "applied" in ast.dump(i.test)]
    assert len(applied) == 1
    recomputed = [n for n in ast.walk(applied[0]) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "size_check_record" for t in n.targets)
                  and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", None) == "size_report"]
    assert len(recomputed) == 1


def test_the_stage_passes_the_depth_the_ring_and_the_camera_up_to_the_check():
    _, main = _stage8_main()
    calls = _calls(main, "size_report")
    assert len(calls) == 2
    for call in calls:
        args = [ast.unparse(a) for a in call.args]
        assert len(args) == 6
        assert args[2] == "size_depth_points"
        assert args[3] == "size_ring_points"
        assert args[4] in {"up_cam", "gravity_up_cam"}
        assert args[5] == "size_settings"


# --------------------------------------------------------------------------------------
# Round 2: the ring around an object keeps off every removed object; the survivors of the first spec
# --------------------------------------------------------------------------------------

def _two_squares():
    own = np.zeros((60, 80), dtype=bool)
    own[20:30, 30:40] = True
    other = np.zeros((60, 80), dtype=bool)
    other[20:30, 50:60] = True
    return own, other


def test_a_neighbours_mask_is_kept_out_of_the_ring():
    own, other = _two_squares()
    ring = sc.ring_mask(own, own | other, ring_px=6, gap_px=3)
    assert not ring[other].any()


def test_the_ring_reaches_the_surface_near_the_object_but_not_the_gap_beside_a_mask():
    own, other = _two_squares()
    ring = sc.ring_mask(own, own | other, ring_px=6, gap_px=3)
    assert ring[25, 44]  # 5 px from the object, 6 px from the neighbour: clear of both gaps
    assert not ring[25, 41]  # 2 px from the object's own edge: inside its gap
    assert not ring[25, 48]  # 2 px from the neighbour's edge: inside the gap


def test_the_ring_does_not_overlap_any_removed_mask():
    own, other = _two_squares()
    ring = sc.ring_mask(own, own | other, ring_px=6, gap_px=3)
    assert not (ring & (own | other)).any()
    assert ring.any()


def test_with_no_neighbour_the_ring_is_the_band_beyond_the_gap():
    own, _ = _two_squares()
    ring = sc.ring_mask(own, own, ring_px=6, gap_px=3)
    assert ring[25, 43] and not ring[25, 41]
    assert ring.sum() > 0 and not (ring & own).any()


def test_a_zero_gap_keeps_the_ring_off_the_masks_only():
    own, other = _two_squares()
    ring = sc.ring_mask(own, own | other, ring_px=6, gap_px=0)
    assert ring[25, 41]  # no gap: the pixel beside the object's own edge is in the ring
    assert not ring[other].any()
    assert not (ring & (own | other)).any()


@pytest.mark.parametrize("up", [(1.0, 0.0, 0.0), (0.99, 0.1, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)])
def test_the_footprint_is_the_rectangle_for_any_up_direction(up):
    up = _unit(up)
    _, _, basis = np.linalg.svd(up[None, :])
    xs, ys = np.meshgrid(np.linspace(0.0, 0.2, 40), np.linspace(0.0, 0.1, 40))
    rectangle = xs.ravel()[:, None] * basis[1] + ys.ravel()[:, None] * basis[2]
    sides = sc.footprint_sides(rectangle, up)
    assert sides is not None and np.all(np.isfinite(sides))
    assert sides == pytest.approx((0.2, 0.1), abs=0.005)


def test_the_stretch_is_by_the_factor_for_an_up_of_any_length():
    vertices = np.random.default_rng(3).normal(size=(100, 3))
    for up in ([0.0, 0.0, 2.0], [0.0, 3.0, 0.0], [0.5, 0.5, 0.5]):
        stretched = sc.fit_height(vertices, up, 2.0)
        along = _unit(up)
        before = vertices @ along
        after = stretched @ along
        assert after.min() == pytest.approx(before.min(), abs=1e-9)
        assert after.max() - after.min() == pytest.approx(2.0 * (before.max() - before.min()), rel=1e-9)


def test_by_default_a_ring_of_fewer_than_fifty_points_skips_the_height():
    settings = {"size_warn_ratio": 0.15, "visible_only": True, "height_percentile": 95}  # no min_ring_points key
    short = _report(_box(), OVERHEAD, settings, ring=_ring()[:49])
    enough = _report(_box(), OVERHEAD, settings, ring=_ring()[:50])
    assert "skipped" in short["height"] and "need 50" in short["height"]["skipped"]
    assert "skipped" not in enough["height"]


def test_the_mesh_height_counts_every_vertex_not_only_the_first_half():
    box = _box()
    middle = np.tile([0.0, 0.0, 0.025], (60, 1))  # the first 60 vertices are in the middle of the box
    vertices = np.vstack([middle, box.vertices])
    faces = box.faces + 60  # the box's own corners, the top and bottom among them, come last
    to_camera, _ = _camera(OVERHEAD)
    report = sc.size_report(to_camera(vertices), faces, _depth(_box(), to_camera), to_camera(_ring()),
                            _unit(OVERHEAD), SETTINGS)
    assert report["height"].get("mesh_m") == pytest.approx(0.05, abs=1e-6)


def test_a_height_off_by_more_than_the_threshold_warns_even_when_under_twice_it():
    stretched = _box()
    stretched.apply_scale([1.0, 1.0, 1.2])
    report = _scaled_report(stretched, OVERHEAD, dict(SETTINGS, size_warn_ratio=0.15))
    assert 0.15 < report["height"]["deviation"] < 0.25
    assert "height" in report["warnings"]


def test_the_stage_builds_the_union_of_every_removed_mask_once_before_the_object_loop():
    tree = ast.parse(STAGE8.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loops = [n for n in ast.walk(main) if isinstance(n, ast.For)]
    loop = next(n for n in loops if "mesh_dir" in ast.unparse(n.iter))
    inside_loop = [n for n in ast.walk(loop) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "all_object_px" for t in n.targets)]
    assert inside_loop == []
    union_loops = [n for n in loops if "removal_mask" in ast.unparse(n.iter)]
    assert len(union_loops) == 1 and union_loops[0].lineno < loop.lineno
    # Every removal mask, not a slice of them: the iterable is the glob itself.
    assert isinstance(union_loops[0].iter, ast.Call) and ast.unparse(union_loops[0].iter).startswith("sorted(")
    assert "[" not in ast.unparse(union_loops[0].iter)


def test_the_ring_is_computed_from_the_union_with_the_configured_gap():
    _, main = _stage8_main()
    rings = _calls(main, "ring_mask")
    assert len(rings) == 1
    args = [ast.unparse(a) for a in rings[0].args]
    assert args[0] == "object_px"
    assert "all_object_px" in args[1]
    assert args[3] == "gap_px"
    gaps = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "gap_px" for t in n.targets)]
    assert len(gaps) == 1 and "height_ring_gap_px" in ast.unparse(gaps[0].value)
