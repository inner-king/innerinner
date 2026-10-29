# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 8's size check compares a placed mesh with its depth, in the camera frame, on synthetic data.

The camera sits at the origin. A world mesh (floor at z = 0, world up = +z) is placed by a rotation that takes
world up to the up direction the camera reports, and a little way in front of the camera. Depth points are surface
samples on the camera-facing faces; the ring is floor points around the object.
"""

import ast
import io
import json
import pathlib

import numpy as np
import pytest
import trimesh
from PIL import Image
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
    assert s8["height_fit"] == {
        "enabled": False, "threshold": 0.25, "min_scale": 0.5, "max_scale": 2.0,
        "skip_rod_like": False, "rod_min_elongation": 3.0, "rod_min_roundness": 0.6,
        "min_axis_alignment": None, "min_result_extent_m": None,
    }


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
    assert {a.name for a in imports[0].names} == {"decide_height_fit", "final_scale_record", "fit_height", "ring_mask",
                                                  "size_report", "validate_height_fit_safeguards"}


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
    # Two branches rewrite the canonical mesh: the height fit and the column fit (column_fit.py). Each measures the
    # mesh again, once, before the record is written.
    applied = [i for i in ast.walk(main) if isinstance(i, ast.If) and "applied" in ast.dump(i.test)]
    assert len(applied) == 2
    for branch in applied:
        recomputed = [n for n in ast.walk(branch) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "size_check_record" for t in n.targets)
                      and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", None) == "size_report"]
        assert len(recomputed) == 1


def test_the_stage_passes_the_depth_the_ring_and_the_camera_up_to_the_check():
    _, main = _stage8_main()
    calls = _calls(main, "size_report")
    assert len(calls) == 3          # the check, and the measurement again after the height fit and after the column fit
    for call in calls:
        args = [ast.unparse(a) for a in call.args]
        assert len(args) == 7
        assert args[2] == "size_depth_points"
        assert args[3] == "size_ring_points"
        assert args[4] in {"up_cam", "gravity_up_cam"}
        assert args[5] == "size_settings"
        assert args[6] == "size_fallback_points"


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
    assert len(rings) == 2  # the ring, and the fallback band with no neighbour excluded
    args = [ast.unparse(a) for a in rings[0].args]
    assert args[0] == "object_px"
    assert "all_object_px" in args[1]
    assert args[3] == "gap_px"
    band = [ast.unparse(a) for a in rings[1].args]
    assert band[0] == "object_px" and band[1] == "object_px" and band[3] == "gap_px"
    gaps = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "gap_px" for t in n.targets)]
    assert len(gaps) == 1 and "height_ring_gap_px" in ast.unparse(gaps[0].value)


# The node transform the Hunyuan GLBs carry: Y-up to Z-up, with a translation. The stage-8 canonical file carries the
# same kind of rotation with a uniform scale (0.16), so both are tested.
Y_TO_Z_NODE = np.array([[1.0, 0.0, 0.0, 0.1], [0.0, 0.0, -1.0, 0.2], [0.0, 1.0, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0]])
SCALED_Y_TO_Z_NODE = np.array([[0.16, 0.0, 0.0, 0.05], [0.0, 0.0, -0.16, -0.1], [0.0, 0.16, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0]])
HUNYUAN_GLB_DIR = pathlib.Path("/home/inner/simfoundry_logs/lidar_rerun_2026-10-26/Data/lidar_kitchen_full_lidar"
                               "/s7_mesh/textured_mesh/hunyuan")


def _single_geometry(scene):
    nodes = list(scene.graph.nodes_geometry)
    assert len(nodes) == 1
    transform, name = scene.graph[nodes[0]]
    return nodes[0], transform, scene.geometry[name]


def _assert_stretched_along_up_only(before, after, up, scale):
    """`after` is `before` with every point moved along `up` only, the extent along `up` multiplied by `scale`, and the
    lowest point along `up` where it was."""
    up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    along_before, along_after = before @ up, after @ up
    assert along_after.max() - along_after.min() == pytest.approx(scale * (along_before.max() - along_before.min()),
                                                                  abs=1e-6)
    assert along_after.min() == pytest.approx(along_before.min(), abs=1e-6)
    moved = after - before
    across = moved - (moved @ up)[:, None] * up[None, :]
    assert np.allclose(across, 0.0, atol=1e-6)  # GLB stores float32, so this is float32 precision


@pytest.mark.parametrize("node", [Y_TO_Z_NODE, SCALED_Y_TO_Z_NODE], ids=["unit", "scaled"])
def test_a_rotated_node_transform_is_baked_stretched_and_reset(tmp_path, node):
    stretch = _stage8_function("_stretch_canonical_height")
    geometry = trimesh.creation.box(extents=(0.2, 0.1, 0.05))
    geometry.apply_translation([0.03, -0.02, 0.07])  # off-centre, so a wrong product order moves the points elsewhere
    scene = trimesh.Scene()
    scene.add_geometry(geometry, node_name="part", transform=node)
    path = tmp_path / "rotated.glb"
    scene.export(path)
    _, _, raw = _single_geometry(trimesh.load(path))
    world_before = trimesh.load(path).to_geometry().vertices.copy()
    assert np.allclose(world_before, (node[:3, :3] @ raw.vertices.T).T + node[:3, 3], atol=1e-9)
    up = np.array([0.0, 0.0, 1.0])
    assert stretch(path, up, 1.3) is None
    back = trimesh.load(path)
    _, transform_after, _ = _single_geometry(back)
    assert np.allclose(transform_after, np.eye(4), atol=1e-9)
    world_after = back.to_geometry().vertices
    _assert_stretched_along_up_only(world_before, world_after, up, 1.3)
    assert np.allclose(world_after, sc.fit_height(world_before, up, 1.3), atol=1e-6)


@pytest.mark.parametrize("index", range(8))
def test_the_real_hunyuan_canonical_meshes_stretch_in_the_world_frame_with_everything_else_kept(tmp_path, index):
    source = HUNYUAN_GLB_DIR / f"iter_{index}_mesh.glb"
    if not source.exists():
        pytest.skip(f"{source} is not present on this machine (the lidar rerun's Hunyuan output)")
    stretch = _stage8_function("_stretch_canonical_height")
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    up = np.array([0.0, 0.0, 1.0])
    before_scene = trimesh.load(path)
    _, node, before_geom = _single_geometry(before_scene)
    world_before = before_scene.to_geometry().vertices.copy()
    assert not np.allclose(node, np.eye(4))  # the case the old helper refused
    assert stretch(path, up, 1.3) is None
    back = trimesh.load(path)
    _, node_after, after_geom = _single_geometry(back)
    world_after = back.to_geometry().vertices
    _assert_stretched_along_up_only(world_before, world_after, up, 1.3)
    assert np.allclose(world_after, sc.fit_height(world_before, up, 1.3), atol=1e-6)
    assert np.allclose(node_after, np.eye(4), atol=1e-9)
    assert np.array_equal(np.asarray(after_geom.faces), np.asarray(before_geom.faces))
    assert np.allclose(np.asarray(after_geom.visual.uv), np.asarray(before_geom.visual.uv), atol=1e-9)
    # trimesh always re-encodes a JPEG texture on export, at PIL's default quality (the stage's identity path does the
    # same). The texture is therefore exactly one such re-encode of the original pixels, and nothing else changes.
    texture_before = before_geom.visual.material.baseColorTexture
    buffer = io.BytesIO()
    texture_before.save(buffer, format="JPEG")
    reference = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert(texture_before.mode))
    assert np.array_equal(np.asarray(after_geom.visual.material.baseColorTexture), reference)


# --- the floor's noise (height_noise_sigmas) and the neighbour floor (height_floor_fallback) -------------------------

NOISE_SETTINGS = {"size_warn_ratio": 0.15, "visible_only": False, "height_percentile": 95, "min_ring_points": 50,
                  "height_noise_sigmas": 2.0, "floor_fallback": True}


def _box_points(extents=(0.2, 0.1), z=0.0, n=(21, 11)):
    """A flat grid over the footprint at height `z`: the depth returns of a top face, or the floor under an object."""
    grid_x, grid_y = np.meshgrid(np.linspace(-extents[0] / 2, extents[0] / 2, n[0]),
                                 np.linspace(-extents[1] / 2, extents[1] / 2, n[1]))
    return np.stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z)], axis=1)


def _ring_with_spread(floor_z, spread, n=60):
    """`n` floor points alternating `spread` above and below `floor_z`: median `floor_z`, MAD `spread`."""
    offsets = np.where(np.arange(n) % 2 == 0, spread, -spread)
    return np.stack([np.arange(n) * 0.01, np.zeros(n), floor_z + offsets], axis=1)


def _height_case(mesh_height, depth_height, ring, fallback=None, settings=NOISE_SETTINGS, floor_z=0.0):
    """A box `mesh_height` tall on the floor, depth points on a top face `depth_height` above the floor. Up is +Z."""
    box = trimesh.creation.box(extents=(0.2, 0.1, mesh_height))
    box.apply_translation([0.0, 0.0, floor_z + mesh_height / 2])
    depth = _box_points(z=floor_z + depth_height)
    return sc.size_report(box.vertices, box.faces, depth, ring, np.array([0.0, 0.0, 1.0]), settings, fallback)


def test_robust_spread_is_1_4826_times_the_median_absolute_deviation():
    # median 2.5; the absolute deviations 2.5, 1.5, 0.5, 0.5, 1.5, 97.5 have median 1.5. The outlier does not count.
    assert sc.robust_spread([0.0, 1.0, 2.0, 3.0, 4.0, 100.0]) == pytest.approx(1.4826 * 1.5)
    assert sc.robust_spread([0.0, 1.0, 2.0, 3.0, 4.0, -50.0]) == pytest.approx(1.4826 * 1.5)


def test_dominant_floor_is_the_median_of_the_most_populated_five_millimetre_bin():
    # bins 0 (four points), 2 (two) and 4 (one): the median of bin 0
    assert sc.dominant_floor([0.001, 0.002, 0.003, 0.004, 0.011, 0.012, 0.021]) == pytest.approx(0.0025)
    # a tie between bins 0 and 2 goes to the lower bin
    assert sc.dominant_floor([0.001, 0.002, 0.011, 0.012]) == pytest.approx(0.0015)


def test_a_height_off_by_more_than_the_ratio_but_within_the_floors_noise_is_not_warned():
    # depth height 0.02 m: the ratio alone allows 0.003 m. Floor spread 0.002 m, so the noise allows 2 x 0.00297 m.
    ring = _ring_with_spread(0.0, 0.002)
    assert _height_case(0.024, 0.02, ring)["warnings"] == []
    assert _height_case(0.028, 0.02, ring)["warnings"] == ["height"]


def test_the_noise_rule_is_strict_at_its_limit_and_needs_both_limits():
    # exceeds: both the ratio and the noise must be cleared, each strictly
    assert not sc._height_exceeds({"deviation": 0.5, "deviation_m": 0.01, "floor_spread_m": 0.005}, 0.15, 2.0)
    assert sc._height_exceeds({"deviation": 0.5, "deviation_m": 0.0101, "floor_spread_m": 0.005}, 0.15, 2.0)
    assert not sc._height_exceeds({"deviation": 0.15, "deviation_m": 0.1, "floor_spread_m": 0.01}, 0.15, 2.0)
    assert not sc._height_exceeds({"deviation": 0.1, "deviation_m": 0.05, "floor_spread_m": 0.001}, 0.15, 2.0)
    assert not sc._height_exceeds({"deviation": 0.5, "deviation_m": 0.0015, "floor_spread_m": 0.001}, 0.15, 2.0)
    assert sc._height_exceeds({"deviation": 0.5, "deviation_m": 0.0015, "floor_spread_m": 0.001}, 0.15, None)
    assert not sc._height_exceeds({"deviation": 0.15, "deviation_m": 0.1, "floor_spread_m": 0.01}, 0.15, None)


def test_a_thick_ring_reads_the_floor_itself_even_when_a_fallback_is_given():
    fallback = _box_points(z=0.05)  # far from the ring's floor: must not be used
    report = _height_case(0.02, 0.02, _ring_with_spread(0.0, 0.001), fallback=fallback)
    assert report["height"]["floor_source"] == "ring"
    assert report["height"]["floor_m"] == pytest.approx(0.0)


def test_a_thin_ring_reads_the_floor_from_the_neighbour_pixels_as_their_dominant_bin():
    ring = _ring_with_spread(0.0, 0.001, n=10)  # fewer than 50: the ring is too thin
    # 120 points on one floor height, and 180 spread over higher heights: the median (about 13 mm) is not the floor
    fallback = np.stack([np.zeros(300), np.zeros(300), np.concatenate([np.zeros(120), np.linspace(0.006, 0.05, 180)])],
                        axis=1)
    height = _height_case(0.02, 0.02, ring, fallback=fallback)["height"]
    assert height["floor_source"] == "neighbour_masks"
    assert height["floor_m"] == pytest.approx(0.0)
    assert height["floor_spread_m"] == pytest.approx(sc.robust_spread(fallback[:, 2]))


def test_a_thin_ring_is_skipped_with_the_reason_when_the_fallback_is_off_or_thin_too():
    ring = _ring_with_spread(0.0, 0.001, n=10)
    off = {**NOISE_SETTINGS, "floor_fallback": False}
    reason = _height_case(0.02, 0.02, ring, fallback=_box_points(), settings=off)["height"]["skipped"]
    assert reason == "only 10 points around the object to read the surface from, need 50"
    thin = _height_case(0.02, 0.02, ring, fallback=_box_points(n=(3, 3)))["height"]["skipped"]
    assert thin == "only 10 points around the object to read the surface from, need 50"


def test_the_fit_is_not_applied_inside_the_floors_noise_and_is_applied_without_it():
    report = {"height": {"deviation": 0.3, "deviation_m": 0.0015, "floor_spread_m": 0.001, "depth_m": 0.005,
                         "mesh_m": 0.0065}}
    inside = sc.decide_height_fit(report, {**HEIGHT_FIT_OFF, "height_noise_sigmas": 2.0})
    assert inside["applied"] is False
    assert "within the floor's noise" in inside["skipped"]
    assert sc.decide_height_fit(report, {**HEIGHT_FIT_OFF, "height_noise_sigmas": None})["applied"] is True


def test_the_warning_text_says_where_the_floor_came_from_only_for_a_height_warning():
    text_of = _stage8_function("_size_warning_text")
    height = {"mesh_m": 0.038, "depth_m": 0.014, "ratio": 2.7, "floor_source": "neighbour_masks"}
    assert text_of("iter_5", {"warnings": ["height"], "height": height}, 0.15).endswith(
        "threshold 0.15; floor read from the neighbouring object's pixels")
    assert "neighbouring" not in text_of("iter_5", {"warnings": ["height"], "height": {**height, "floor_source": "ring"}},
                                         0.15)


def test_the_stage_passes_the_fallback_floor_to_both_size_checks_and_reads_the_noise_settings():
    _, main = _stage8_main()
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "size_report"]
    assert len(calls) == 3
    for call in calls:
        assert len(call.args) == 7 and ast.unparse(call.args[6]) == "size_fallback_points"
    source = ast.unparse(main)
    assert "height_noise_sigmas" in source and "height_floor_fallback" in source
    assert "neighbouring object's pixels" in source


def test_the_defaults_of_the_noise_and_the_fallback_are_in_the_config():
    yaml = pytest.importorskip("yaml")
    s8 = yaml.safe_load(CFG.read_text())["s8_pose"]
    assert s8["height_noise_sigmas"] == 2.0
    assert s8["height_floor_fallback"] is True


# --- the floor's median and bin width, the fallback's boundary, the sign of the height deviation, and the defaults ---

def test_dominant_floor_is_the_median_of_its_bin_not_the_mean():
    # One 5 mm bin holds three values at 0.1 mm and one at 4.9 mm: the median (0.1 mm) is not the mean (1.3 mm).
    assert sc.dominant_floor([0.0001, 0.0001, 0.0001, 0.0049]) == pytest.approx(0.0001)


def test_the_dominant_bin_is_five_millimetres_wide():
    # Five millimetre bins: three values in the first and four in the second, so the second wins, with median 7.5 mm.
    # Ten millimetre bins would put all seven in one bin, whose median is 6 mm.
    assert sc.dominant_floor([0.001, 0.002, 0.003, 0.006, 0.007, 0.008, 0.009]) == pytest.approx(0.0075)


def test_the_fallback_is_used_with_exactly_the_minimum_number_of_points_and_not_with_one_fewer():
    # The ring minimum is also the fallback's minimum: 50 points are enough, 49 are not.
    ring = _ring_with_spread(0.0, 0.001, n=10)  # too thin, so the fallback decides
    exactly = _height_case(0.02, 0.02, ring, fallback=_box_points(n=(5, 10)))["height"]  # 50 points
    assert exactly["floor_source"] == "neighbour_masks"
    one_fewer = _height_case(0.02, 0.02, ring, fallback=_box_points(n=(7, 7)))["height"]  # 49 points
    assert one_fewer["skipped"] == "only 10 points around the object to read the surface from, need 50"


def test_a_mesh_shorter_than_the_depth_is_judged_by_its_size_against_the_noise_too():
    # The mesh is 8 mm shorter than the depth: 40 % off. The floor's noise is about 1.5 mm, so 2 sigmas is about 3 mm,
    # and 8 mm is beyond that.
    assert _height_case(0.012, 0.02, _ring_with_spread(0.0, 0.001))["warnings"] == ["height"]
    # The same shortfall inside a floor noise of about 15 mm (2 sigmas about 30 mm) is not reported.
    assert _height_case(0.012, 0.02, _ring_with_spread(0.0, 0.01))["warnings"] == []


def test_the_size_report_does_not_read_the_fallback_when_the_setting_is_missing():
    # The function's own default is off, so a caller that leaves the key out keeps the old behaviour.
    no_key = {k: v for k, v in NOISE_SETTINGS.items() if k != "floor_fallback"}
    ring = _ring_with_spread(0.0, 0.001, n=10)  # too thin
    reason = _height_case(0.02, 0.02, ring, fallback=_box_points(), settings=no_key)["height"]["skipped"]
    assert reason == "only 10 points around the object to read the surface from, need 50"


def test_the_stage_defaults_for_the_noise_and_the_fallback_are_the_config_values():
    yaml = pytest.importorskip("yaml")
    s8 = yaml.safe_load(CFG.read_text())["s8_pose"]
    _, main = _stage8_main()
    defaults = {}
    for call in ast.walk(main):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "get"
                and len(call.args) == 2 and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[1], ast.Constant)):
            defaults.setdefault(call.args[0].value, call.args[1].value)

    assert defaults["height_noise_sigmas"] == s8["height_noise_sigmas"] == 2.0
    assert defaults["height_floor_fallback"] is True and s8["height_floor_fallback"] is True


# --- the total scale stage 8 placed the object at, recorded before FoundationPose resets the pose's scale ---------------

def test_the_final_scale_is_the_pre_scale_times_the_cpd_scale_for_a_scalar():
    assert sc.final_scale_record(0.5, 2.0) == {"cpd_scale": 2.0, "final_scale": 1.0}


def test_the_final_scale_is_the_pre_scale_times_each_axis_of_the_cpd_scale():
    assert sc.final_scale_record(0.5, np.array([1.0, 2.0, 4.0])) == {"cpd_scale": [1.0, 2.0, 4.0],
                                                                     "final_scale": [0.5, 1.0, 2.0]}


def test_a_unit_cpd_scale_leaves_the_final_scale_at_the_pre_scale():
    assert sc.final_scale_record(0.8, np.ones(3))["final_scale"] == pytest.approx([0.8, 0.8, 0.8])


def test_the_scale_record_is_plain_numbers_and_lists_so_it_serialises_to_json():
    record = sc.final_scale_record(np.float64(0.5), np.array([1.0, 2.0, 4.0]))
    assert json.loads(json.dumps(record)) == record
    assert all(type(v) is float for v in record["final_scale"])
    assert type(sc.final_scale_record(0.5, 2.0)["final_scale"]) is float


def test_the_scale_record_is_made_from_the_final_pre_scale_and_the_cpd_scale_before_foundationpose_resets_it():
    _, main = _stage8_main()
    records = _calls(main, "final_scale_record")
    assert len(records) == 1
    args = [ast.unparse(a) for a in records[0].args]
    assert args == ["final_pre_scale_factor", "top_info['tf_z_up'].scale"]
    # The same expression FoundationPose's export applies to the mesh, so the record and the mesh read one factor.
    applied = [c for c in _calls(main, "apply_scale") if isinstance(c.func, ast.Attribute)
               and ast.unparse(c.func.value) == "fp_scaled_mesh"]
    assert [ast.unparse(a) for a in applied[0].args] == [args[1]]
    # Recorded before the FoundationPose block, and so before the pose is overwritten with a unit scale.
    fp_block = [n.lineno for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == "use_foundationpose"]
    overwrite = [n.lineno for n in ast.walk(main) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "top_info['tf_z_up']" and ast.unparse(n.value) == "foundationpose_tf_rt"]
    assert len(fp_block) == 1 and len(overwrite) == 1
    assert records[0].lineno < fp_block[0] < overwrite[0]


def test_the_info_file_records_the_cpd_and_final_scales_beside_the_pre_scale_factor():
    _, main = _stage8_main()
    out = [n.value for n in ast.walk(main) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
           and any(isinstance(t, ast.Name) and t.id == "out_info" for t in n.targets)]
    assert len(out) == 1
    entries = {k.value: ast.unparse(v) for k, v in zip(out[0].keys, out[0].values)}
    assert entries["pre_scale_factor"] == "float(pre_scale_factor)"
    assert entries["cpd_scale"] == "scale_record['cpd_scale']"
    assert entries["final_scale"] == "scale_record['final_scale']"


def test_the_height_fit_factor_is_recorded_apart_and_is_one_unless_a_stretch_was_applied():
    _, main = _stage8_main()
    recorded = {}
    for guard in _ifs_on(main, "size_check_on"):
        for n in ast.walk(guard):
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript) \
                    and isinstance(n.targets[0].slice, ast.Constant):
                recorded[n.targets[0].slice.value] = n.value
    value = recorded["height_fit_scale"]
    assert isinstance(value, ast.IfExp)
    assert ast.unparse(value.test) == "height_fit_record['applied']"
    assert ast.unparse(value.body) == "height_fit_record['scale']" and ast.unparse(value.orelse) == "1.0"


def test_no_log_says_a_scale_grid_search_runs_while_its_loop_is_commented_out():
    _, main = _stage8_main()
    text = ast.unparse(main)
    assert "Running scale grid search" not in text
    assert "scale_multiplier=" not in text
    assert "Scale grid search is off" in text


def test_the_readme_names_the_scale_keys_the_info_file_records():
    readme = (REPO / "scripts/pipeline/README.md").read_text()
    assert "`cpd_scale`" in readme and "`final_scale`" in readme and "`height_fit_scale`" in readme
