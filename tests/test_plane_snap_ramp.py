# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tapered move of a one-sided plane snap (`s_bg_mesh.plane_snap_ramp_px`, B18c): the weights, the validation, the
default, and what the ramp does to a depression with soft edges.

Synthetic depth maps on the camera of test_plane_snap.py, so the expected values are known in closed form. None needs a
GPU; the config and README tests read the repository's files and skip nothing.
"""

import ast
import json
import pathlib

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt, gaussian_filter

from simfoundry.pipeline import depth_mesh as dm
from simfoundry.pipeline.depth_mesh import DEFAULT_PARAMS, build_background_mesh, snap_ramp_weights

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "scripts/cfg/real2sim_cfg.yaml"
README = REPO_ROOT / "scripts/pipeline/README.md"

H, W, FOCAL, CAM_HEIGHT = 48, 64, 80.0, 0.7
K = np.array([[FOCAL, 0.0, 32.0], [0.0, FOCAL, 24.0], [0.0, 0.0, 1.0]])
CAM2WORLD = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, CAM_HEIGHT], [0.0, 0.0, 0.0, 1.0]])
DEEP = (slice(16, 32), slice(24, 40))
NEW_STATS = {"plane_snap_ramp_px_used", "vertices_snap_ramped"}
NEW_PARAMS = {"plane_snap_ramp_px"}


def depression_depth():
    """The 4 cm depression with soft edges of test_plane_snap.py, on a flat support plane."""
    mask = np.zeros((H, W))
    mask[DEEP] = 1.0
    return np.full((H, W), CAM_HEIGHT) + 0.04 * gaussian_filter(mask, sigma=2.0)


def plate():
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return np.stack([rows * 5 % 256, cols * 4 % 256, np.full_like(rows, 7)], axis=-1).astype(np.uint8)


def build(depth, params=None):
    return build_background_mesh(depth, K, CAM2WORLD, plate(), None, params)


def without_new_keys(stats):
    kept = {k: v for k, v in stats.items() if k not in NEW_STATS and k != "params"}
    kept["params"] = {k: v for k, v in stats["params"].items() if k not in NEW_PARAMS}
    return kept


def rectangle(rows, cols, shape=(9, 13)):
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True
    return mask


# 1. the weights -----------------------------------------------------------------------------------------------------------

def test_a_zero_ramp_is_the_mask_as_floats():
    mask = rectangle(slice(2, 7), slice(2, 11))
    weights = snap_ramp_weights(mask, 0)
    assert weights.dtype == np.float64 and np.array_equal(weights, mask.astype(np.float64))


def test_the_weights_are_finite_floats_in_the_unit_interval_with_the_shape_of_the_mask():
    mask = rectangle(slice(1, 8), slice(1, 12))
    mask[4, 6] = False  # a hole
    weights = snap_ramp_weights(mask, 4)
    assert weights.shape == mask.shape and weights.dtype == np.float64
    assert np.isfinite(weights).all() and weights.min() >= 0.0 and weights.max() <= 1.0


def test_the_weights_do_not_fall_as_the_pixel_goes_deeper_into_the_region():
    mask = rectangle(slice(0, 9), slice(0, 13))
    mask[0:3, 0:4] = False
    mask[6:9, 9:13] = False
    weights = snap_ramp_weights(mask, 3)
    dist = distance_transform_edt(mask)
    inside = np.flatnonzero(mask.reshape(-1))
    order = inside[np.argsort(dist.reshape(-1)[inside], kind="stable")]
    assert np.all(np.diff(weights.reshape(-1)[order]) >= 0.0)


def test_a_region_that_fills_the_array_gets_the_plain_mask_whatever_the_ramp():
    full = np.ones((3, 4), dtype=bool)
    for ramp in (1, 4, 9):
        assert np.array_equal(snap_ramp_weights(full, ramp), np.ones((3, 4)))


def test_pixels_outside_the_region_are_exactly_zero():
    mask = rectangle(slice(2, 7), slice(2, 11))
    weights = snap_ramp_weights(mask, 3)
    assert np.all(weights[~mask] == 0.0)


def test_a_single_pixel_region_has_no_move_at_all():
    single = rectangle(slice(2, 3), slice(2, 3), shape=(5, 5))
    assert snap_ramp_weights(single, 4)[2, 2] == 0.0


def test_the_rim_of_a_hole_starts_at_zero_too():
    region = rectangle(slice(0, 9), slice(0, 9), shape=(9, 9))
    region[4, 4] = False
    weights = snap_ramp_weights(region, 4)
    for row, col in ((3, 4), (5, 4), (4, 3), (4, 5)):
        assert weights[row, col] == 0.0
    assert weights[3, 3] > 0.0  # the diagonal neighbours are one step further in


def test_a_region_nine_pixels_thick_ramps_from_its_rim_to_its_middle_in_quarters():
    # The weight measures the nearest pixel outside the region in any direction. In an 11 x 15 array the region below is
    # nine rows tall, so its centre column is four pixels from the rim above and below, and the weight reaches 1 there.
    region = rectangle(slice(1, 10), slice(1, 14), shape=(11, 15))
    column = snap_ramp_weights(region, 4)[1:10, 7]
    assert np.allclose(column, [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0])


def test_a_five_row_rectangle_stops_at_one_half_because_the_nearest_outside_pixel_is_three_rows_away():
    # The centre row lies three pixels from the outside above and below, whatever the columns, so the distance stops
    # growing at 3 and the weight at one half. The rule is the nearest non-snapped pixel in any direction, not the sides.
    region = rectangle(slice(2, 7), slice(2, 11))
    row = snap_ramp_weights(region, 4)[4, 2:11]
    assert np.allclose(row, [0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.0])


def by_hand(depth, below, above, camera=K):
    """Pre-snap points, the plane mask and the ray hits, from the helpers and the rules of the build (no ramp)."""
    rays_cam = dm.pixel_rays((H, W), camera)
    points = dm.to_world(rays_cam * depth[..., None], CAM2WORLD)
    rays_world = rays_cam @ CAM2WORLD[:3, :3].T
    in_range = np.isfinite(depth) & (depth > 0)
    trim = int(DEFAULT_PARAMS["border_trim_px"])
    in_range[:trim] = in_range[-trim:] = False
    in_range[:, :trim] = in_range[:, -trim:] = False
    plane_flag = in_range & dm.plane_snap_mask(points[..., 2], below, above)
    _, hits = dm.plane_intersections(rays_world, CAM2WORLD[:3, 3])
    return points, rays_world, plane_flag, hits


def mesh_height_by_pixel(mesh):
    uv = np.asarray(mesh.visual.uv)
    cols = np.rint(uv[:, 0] * W - 0.5).astype(int)
    rows = np.rint((1.0 - uv[:, 1]) * H - 0.5).astype(int)
    height = np.full((H, W), np.nan)
    height[rows, cols] = np.asarray(mesh.vertices)[:, 2]
    return height


def test_a_pixel_whose_ray_misses_the_plane_keeps_its_height_scaled_by_the_ramp(monkeypatch):
    # Every ray of the test camera reaches the plane, so the miss branch needs rays that do not: the hits of the left
    # half are cut off here, and that branch must then take its height from (1 - weight), not from 0.
    depth = depression_depth()
    band = {"plane_snap_below_m": 0.03, "plane_snap_above_m": 0.01}
    real_intersections = dm.plane_intersections

    def misses_on_the_left(rays_world, centre, height=0.0, min_slope=1e-6):
        on_plane, hits = real_intersections(rays_world, centre, height, min_slope)
        hits = hits.copy()
        hits[:, :32] = False
        return on_plane, hits

    monkeypatch.setattr(dm, "plane_intersections", misses_on_the_left)
    mesh, _, _ = build(depth, {**band, "plane_snap_ramp_px": 6})
    points, _, plane_flag, _ = by_hand(depth, 0.03, 0.01)
    weight = snap_ramp_weights(plane_flag, 6)
    columns = np.arange(W)[None, :]
    partial = plane_flag & (columns < 32) & (weight > 0.0) & (weight < 1.0)
    assert partial.any()
    height = mesh_height_by_pixel(mesh)
    shown = partial & np.isfinite(height)
    assert shown.any()
    assert np.allclose(height[shown], points[..., 2][shown] * (1.0 - weight[shown]), atol=1e-12)


def test_a_fully_weighted_vertex_is_moved_along_its_ray_in_x_and_y_as_well_as_z():
    depth = depression_depth()
    band = {"plane_snap_below_m": 0.03, "plane_snap_above_m": 0.01}
    mesh, _, _ = build(depth, {**band, "plane_snap_ramp_px": 6})
    points, rays_world, plane_flag, _ = by_hand(depth, 0.03, 0.01)
    on_plane, hits = dm.plane_intersections(rays_world, CAM2WORLD[:3, 3])
    weight = snap_ramp_weights(plane_flag, 6)
    full = plane_flag & hits & (weight == 1.0)
    uv = np.asarray(mesh.visual.uv)
    cols = np.rint(uv[:, 0] * W - 0.5).astype(int)
    rows = np.rint((1.0 - uv[:, 1]) * H - 0.5).astype(int)
    vertex_at = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    checked = 0
    for r, c in zip(*np.nonzero(full)):
        if (r, c) in vertex_at:
            assert np.allclose(np.asarray(mesh.vertices)[vertex_at[(r, c)]], on_plane[r, c], atol=1e-9)
            checked += 1
    assert checked > 0


# 2. validation and the default ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, 2.5, "4", True, -1, float("nan"), [4]])
def test_a_value_that_is_not_a_whole_number_of_pixels_at_least_zero_is_an_error_naming_the_key(value):
    with pytest.raises(ValueError, match="plane_snap_ramp_px must be an integer >= 0"):
        build(np.full((H, W), CAM_HEIGHT), {"plane_snap_ramp_px": value})


@pytest.mark.parametrize("value", [0, 1, 4, np.int64(4)])
def test_zero_one_and_whole_numbers_are_accepted_and_recorded_as_ints(value):
    _, _, stats = build(np.full((H, W), CAM_HEIGHT), {"plane_snap_ramp_px": value})
    assert stats["plane_snap_ramp_px_used"] == int(value)


def test_the_default_is_zero_and_the_config_says_so():
    assert DEFAULT_PARAMS["plane_snap_ramp_px"] == 0


# 3. the default is the old move, byte for byte -----------------------------------------------------------------------------

@pytest.mark.parametrize("band", [{}, {"plane_snap_below_m": 0.03, "plane_snap_above_m": 0.01}])
def test_zero_written_out_and_zero_left_out_give_the_same_mesh_texture_and_stats(band):
    depth = depression_depth()
    left_out = build(depth, band)
    written = build(depth, {**band, "plane_snap_ramp_px": 0})
    assert np.asarray(left_out[0].vertices).tobytes() == np.asarray(written[0].vertices).tobytes()
    assert np.asarray(left_out[0].faces).tobytes() == np.asarray(written[0].faces).tobytes()
    assert np.asarray(left_out[0].visual.uv).tobytes() == np.asarray(written[0].visual.uv).tobytes()
    assert np.asarray(left_out[1]).tobytes() == np.asarray(written[1]).tobytes()
    assert without_new_keys(left_out[2]) == without_new_keys(written[2])
    assert written[2]["vertices_snap_ramped"] == 0 and written[2]["plane_snap_ramp_px_used"] == 0


def test_the_zero_branch_is_the_three_old_snap_lines_word_for_word():
    tree = ast.parse((REPO_ROOT / "simfoundry/pipeline/depth_mesh.py").read_text())
    build_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_background_mesh")
    branches = [n for n in ast.walk(build_fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "ramp_px == 0"]
    assert len(branches) == 1
    assert [ast.unparse(statement) for statement in branches[0].body] == [
        "use = plane_flag & hits_plane",
        "points = np.where(use[..., None], on_plane, points)",
        "points[..., 2] = np.where(plane_flag & ~hits_plane, 0.0, points[..., 2])",
    ]
    assert branches[0].orelse, "the ramp is the other branch, not a replacement of the old lines"


# 4. the stats are an independent count -------------------------------------------------------------------------------------

def ramped_mesh_count(mesh, depth, below, above, ramp):
    """The mesh's plane vertices whose move the ramp takes only part of the way, from the helpers and the rules of the build."""
    _, _, plane_flag, _ = by_hand(depth, below, above)
    weight = snap_ramp_weights(plane_flag, ramp)
    uv = np.asarray(mesh.visual.uv)
    cols = np.rint(uv[:, 0] * W - 0.5).astype(int)
    rows = np.rint((1.0 - uv[:, 1]) * H - 0.5).astype(int)
    return int(np.count_nonzero(plane_flag[rows, cols] & (weight[rows, cols] < 1.0)))


@pytest.mark.parametrize("ramp", [1, 4, 6])
def test_the_ramped_count_is_the_mesh_plane_vertices_whose_weight_is_below_one(ramp):
    depth = depression_depth()
    mesh, _, stats = build(depth, {"plane_snap_below_m": 0.03, "plane_snap_above_m": 0.01, "plane_snap_ramp_px": ramp})
    assert stats["plane_snap_ramp_px_used"] == ramp
    assert stats["vertices_snap_ramped"] == ramped_mesh_count(mesh, depth, 0.03, 0.01, ramp) > 0


def test_a_plane_pixel_that_no_triangle_reaches_is_not_counted_among_the_mesh_vertices():
    # One plane pixel with invalid neighbours on every side: no triangle has it as a corner, so it is not a mesh vertex
    # even though the ramp gives it a weight below one. Counting the whole grid would count it.
    depth = depression_depth()
    depth[4:7, 4:7] = np.nan
    depth[5, 5] = CAM_HEIGHT
    mesh, _, stats = build(depth, {"plane_snap_below_m": 0.03, "plane_snap_above_m": 0.01, "plane_snap_ramp_px": 6})
    _, _, plane_flag, _ = by_hand(depth, 0.03, 0.01)
    weight = snap_ramp_weights(plane_flag, 6)
    assert plane_flag[5, 5] and weight[5, 5] < 1.0
    grid_count = int(np.count_nonzero(plane_flag & (weight < 1.0)))
    assert stats["vertices_snap_ramped"] == ramped_mesh_count(mesh, depth, 0.03, 0.01, 6)
    assert stats["vertices_snap_ramped"] < grid_count


def test_a_numpy_integer_setting_is_recorded_as_a_plain_int_and_the_new_stats_serialise():
    # stage_info.json is written from these stats, and a numpy integer there would stop the write.
    _, _, stats = build(depression_depth(), {"plane_snap_below_m": 0.03, "plane_snap_above_m": 0.01,
                                              "plane_snap_ramp_px": np.int64(4)})
    assert type(stats["plane_snap_ramp_px_used"]) is int and stats["plane_snap_ramp_px_used"] == 4
    assert type(stats["vertices_snap_ramped"]) is int
    assert type(stats["params"]["plane_snap_ramp_px"]) is int
    json.dumps({key: stats[key] for key in NEW_STATS})
    json.dumps(stats["params"])


# 4b. the effect on the depression with soft edges -----------------------------------------------------------------------

#: The focal length of the LiDAR input, for the effect test only. A pixel covers 0.7 / focal metres at this depth: 8.75 mm
#: at the 80 px of the other tests, where the 3 cm cliff of the snap is under four pixel footprints, so no triangle can pass
#: the limit of 8 and the problem does not show. At 320 px a pixel covers 2.2 mm, and the same cliff is about fourteen
#: footprints. The depth map is the depression of test_plane_snap.py, unchanged.
K_LIDAR_FOCAL = np.array([[320.0, 0.0, 32.0], [0.0, 320.0, 24.0], [0.0, 0.0, 1.0]])


def build_with(depth, camera, params):
    return build_background_mesh(depth, camera, CAM2WORLD, plate(), None, params)


def stretched_count(mesh, camera):
    """grid_triangles' own criterion on the final mesh: a triangle counts when the largest of its three edge ratios is over
    8, an edge ratio being its length over the mean camera depth of its two vertices divided by the focal length, times
    sqrt(2) for a diagonal edge. Positions and depths are the ones after the snap."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    uv = np.asarray(mesh.visual.uv)
    cols = np.rint(uv[:, 0] * W - 0.5).astype(int)
    rows = np.rint((1.0 - uv[:, 1]) * H - 0.5).astype(int)
    depth_v = dm.to_camera(verts, CAM2WORLD)[:, 2]
    focal = 0.5 * (camera[0, 0] + camera[1, 1])
    worst = np.zeros(len(faces))
    for i, j in ((0, 1), (1, 2), (0, 2)):
        u, v = faces[:, i], faces[:, j]
        steps = np.where(np.abs(rows[u] - rows[v]) + np.abs(cols[u] - cols[v]) == 2, np.sqrt(2.0), 1.0)
        footprint = 0.5 * (depth_v[u] + depth_v[v]) / focal * steps
        worst = np.maximum(worst, np.linalg.norm(verts[u] - verts[v], axis=1) / footprint)
    return int(np.count_nonzero(worst > 8.0))


def test_the_ramp_removes_the_triangles_a_one_sided_cliff_stretches_and_leaves_the_rim_where_it_was():
    depth = depression_depth()
    band = {"plane_snap_below_m": 0.03, "plane_snap_above_m": 0.01}
    unramped = build_with(depth, K_LIDAR_FOCAL, {**band, "plane_snap_ramp_px": 0})
    # The scene has to show the problem before anything else is asserted: without the ramp the cliff stretches triangles.
    assert stretched_count(unramped[0], K_LIDAR_FOCAL) > 0
    ramped = build_with(depth, K_LIDAR_FOCAL, {**band, "plane_snap_ramp_px": 6})
    assert stretched_count(ramped[0], K_LIDAR_FOCAL) == 0
    points, _, plane_flag, _ = by_hand(depth, 0.03, 0.01, K_LIDAR_FOCAL)
    weight = snap_ramp_weights(plane_flag, 6)
    height = mesh_height_by_pixel(ramped[0])
    # Seven or more pixels in from the rim the weight is 1: those vertices are on the plane. The rim itself has weight 0,
    # so its vertices keep the height they were measured at.
    interior = plane_flag & (weight == 1.0) & np.isfinite(height)
    rim = plane_flag & (weight == 0.0) & np.isfinite(height)
    assert interior.any() and rim.any()
    assert np.all(np.abs(height[interior]) < 1e-9)
    assert np.allclose(height[rim], points[..., 2][rim], atol=1e-12)


# 5. the config and the README ------------------------------------------------------------------------------------------

def test_the_yaml_default_is_the_code_default():
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load(CONFIG.read_text())["s_bg_mesh"]["plane_snap_ramp_px"] == DEFAULT_PARAMS["plane_snap_ramp_px"]


def test_the_readme_names_the_key_the_numbers_and_the_limits():
    text = README.read_text()
    start = text.index("Tapered move (candidate, off by default)")
    section = text[start:text.index("\n### ", start)]
    for phrase in ("plane_snap_ramp_px", "465", "1 at 4", "1 at 6", "not been checked", "byte-identical",
                   "as the stage's own stretched-edge test measures it"):
        assert phrase in section
