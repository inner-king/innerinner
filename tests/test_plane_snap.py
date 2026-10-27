# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one-sided plane snap of the depth-mesh background (B18): the band below and above z = 0, its defaults, and what it
does to a lowered region, a raised structure, a depression and an object standing in a lowered region.

All tests use synthetic depth maps on the camera of test_depth_mesh.py (0.7 m above a flat support plane), so the expected
heights are known in closed form. None needs a GPU; the config test needs the pipeline environment and skips elsewhere.
"""

import ast
import pathlib
import re

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from simfoundry.pipeline import depth_mesh
from simfoundry.pipeline.depth_mesh import DEFAULT_PARAMS, build_background_mesh, plane_snap_mask

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "scripts/cfg/real2sim_cfg.yaml"
STAGE = REPO_ROOT / "scripts/pipeline/A_reconstruction/stages/5c_depth_mesh_background.py"

H, W, FOCAL, CAM_HEIGHT = 48, 64, 80.0, 0.7
K = np.array([[FOCAL, 0.0, 32.0], [0.0, FOCAL, 24.0], [0.0, 0.0, 1.0]])
# A camera looking straight down: image right = world +x, image down = world -y, optical axis = world -z.
CAM2WORLD = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, CAM_HEIGHT], [0.0, 0.0, 0.0, 1.0]])
BOX_ROWS, BOX_COLS, BOX_HEIGHT = (18, 30), (26, 44), 0.1

# Regions of the depth grid, all inside the 4-pixel border trim. A 2 cm lowered region is wide enough that its
# triangles stay joined to the plane; the depressions are large enough that their floor survives the 200-triangle drop.
LOWERED = (slice(12, 36), slice(16, 48))
DEEP = (slice(16, 32), slice(24, 40))
DEEP_CORE = (slice(20, 28), slice(28, 36))
WIDE = (slice(2, 46), slice(6, 58))           # lowered everywhere around the object below
LEFT_HALF = (slice(2, 46), slice(6, 44))      # lowered on the left only: the ring around the object mixes two heights
OBJECT_CORE = (slice(21, 27), slice(29, 35))  # the middle of the object mask, away from its edge
RAISED_SMALL = (slice(4, 10), slice(50, 60))  # clear of the lowered region, so the two offsets do not add up

NEW_KEYS = {"vertices_snapped_from_below", "vertices_snapped_from_above", "plane_snap_below_m_used", "plane_snap_above_m_used"}
NEW_PARAMS = {"plane_snap_below_m", "plane_snap_above_m"}


def flat(with_box=False):
    depth = np.full((H, W), CAM_HEIGHT)
    if with_box:
        depth[BOX_ROWS[0]:BOX_ROWS[1], BOX_COLS[0]:BOX_COLS[1]] = CAM_HEIGHT - BOX_HEIGHT
    return depth


def region_mask(region):
    mask = np.zeros((H, W), dtype=bool)
    mask[region] = True
    return mask


def box_mask():
    return region_mask((slice(BOX_ROWS[0], BOX_ROWS[1]), slice(BOX_COLS[0], BOX_COLS[1])))


def plate():
    """A plate whose colour names its pixel, so a texture lookup says which pixel it came from."""
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return np.stack([rows * 5 % 256, cols * 4 % 256, np.full_like(rows, 7)], axis=-1).astype(np.uint8)


def build(depth, params=None, mask=None):
    return build_background_mesh(depth, K, CAM2WORLD, plate(), mask, params)


def grid_of(mesh):
    """The height of each depth pixel's vertex (NaN where the mesh has none), read from the vertex's texture coordinate."""
    uv = np.asarray(mesh.visual.uv)
    cols = np.round(uv[:, 0] * W - 0.5).astype(int)
    rows = np.round((1.0 - uv[:, 1]) * H - 0.5).astype(int)
    grid = np.full((H, W), np.nan)
    grid[rows, cols] = np.asarray(mesh.vertices)[:, 2]
    return grid


def without_new_keys(stats):
    """The stats without the keys this change added (the snap counts, the limits used, and the two new settings)."""
    kept = {k: v for k, v in stats.items() if k not in NEW_KEYS and k != "params"}
    kept["params"] = {k: v for k, v in stats["params"].items() if k not in NEW_PARAMS}
    return kept


# 1. the band -------------------------------------------------------------------------------------------------------------

def test_the_band_is_strict_at_both_ends():
    z = np.array([-0.031, -0.03, -0.029, -0.02, -0.0, 0.0, 0.009, 0.01, 0.011, 0.5, np.nan])
    expected = [False, False, True, True, True, True, True, False, False, False, False]
    assert plane_snap_mask(z, 0.03, 0.01).tolist() == expected
    assert not plane_snap_mask(np.array([-0.03]), 0.03, 0.01)[0]  # the lower end on its own
    assert not plane_snap_mask(np.array([0.01]), 0.03, 0.01)[0]  # the upper end on its own


def test_with_equal_sides_the_band_is_the_old_absolute_value_test_for_every_input():
    rng = np.random.default_rng(18)
    for tol in (0.0, 0.01, 0.03):
        z = np.concatenate([rng.uniform(-0.1, 0.1, 100_000), [tol, -tol, 0.0, -0.0, np.nan, np.inf, -np.inf]])
        assert np.array_equal(plane_snap_mask(z, tol, tol), np.abs(z) < tol)


# 2. the defaults and the sides -------------------------------------------------------------------------------------------

def test_the_defaults_are_the_symmetric_band_and_are_recorded_as_null():
    assert DEFAULT_PARAMS["plane_snap_below_m"] is None and DEFAULT_PARAMS["plane_snap_above_m"] is None
    _, _, stats = build(flat())
    assert stats["params"]["plane_snap_below_m"] is None and stats["params"]["plane_snap_above_m"] is None
    assert stats["plane_snap_below_m_used"] == stats["plane_snap_above_m_used"] == DEFAULT_PARAMS["plane_snap_tol_m"]


def test_a_null_side_takes_the_symmetric_tolerance_and_says_so_in_the_stats():
    depth = flat()
    depth[LOWERED] += 0.02
    below_only = build(depth, {"plane_snap_below_m": 0.03})
    explicit = build(depth, {"plane_snap_below_m": 0.03, "plane_snap_above_m": 0.01})
    assert below_only[2]["plane_snap_below_m_used"] == 0.03 and below_only[2]["plane_snap_above_m_used"] == 0.01
    assert np.array_equal(below_only[0].vertices, explicit[0].vertices)
    assert np.array_equal(below_only[0].faces, explicit[0].faces)
    above_only = build(flat(), {"plane_snap_above_m": 0.03})[2]
    assert above_only["plane_snap_below_m_used"] == 0.01 and above_only["plane_snap_above_m_used"] == 0.03


def test_the_config_has_both_sides_null_with_a_candidate_comment():
    lines = [line for line in CONFIG.read_text().splitlines()
             if line.startswith("  plane_snap_below_m:") or line.startswith("  plane_snap_above_m:")]
    assert len(lines) == 2
    for line in lines:
        value, comment = line.split("#", 1)
        assert value.split(":", 1)[1].strip() == "null" and "candidate" in comment


def test_the_stage_reads_the_new_keys_from_the_config_through_its_own_comprehension():
    omegaconf = pytest.importorskip("omegaconf")
    s_bg = omegaconf.OmegaConf.load(CONFIG).s_bg_mesh
    params = {k: s_bg[k] for k in DEFAULT_PARAMS if k in s_bg}
    assert params["plane_snap_below_m"] is None and params["plane_snap_above_m"] is None
    assert "params = {k: cfg.s_bg_mesh[k] for k in DEFAULT_PARAMS if k in cfg.s_bg_mesh}" in STAGE.read_text()


# 3. the lowered region, the raised structure and the depressions ----------------------------------------------------

def test_a_two_centimetre_lowered_region_is_flattened_from_below_and_counted_so():
    depth = flat()
    depth[LOWERED] += 0.02
    default_grid = grid_of(build(depth)[0])
    one_sided_mesh, _, one_sided_stats = build(depth, {"plane_snap_below_m": 0.03})
    one_sided_grid = grid_of(one_sided_mesh)
    region = region_mask(LOWERED)
    assert np.allclose(default_grid[region & np.isfinite(default_grid)], -0.02)
    assert np.allclose(one_sided_grid[region & np.isfinite(one_sided_grid)], 0.0)
    n_region = int((region & np.isfinite(one_sided_grid)).sum())
    assert n_region > 0
    assert one_sided_stats["vertices_snapped_from_below"] == n_region
    assert one_sided_stats["vertices_snapped_from_above"] == 0


def test_a_two_centimetre_raised_structure_is_left_alone_by_the_one_sided_band():
    depth = flat()
    depth[LOWERED] -= 0.02
    default_mesh, _, _ = build(depth)
    one_sided_mesh, _, one_sided_stats = build(depth, {"plane_snap_below_m": 0.03})
    assert np.array_equal(grid_of(default_mesh), grid_of(one_sided_mesh), equal_nan=True)
    assert one_sided_stats["vertices_snapped_from_above"] == 0
    symmetric_grid = grid_of(build(depth, {"plane_snap_tol_m": 0.03})[0])
    region = region_mask(LOWERED)
    assert np.allclose(symmetric_grid[region & np.isfinite(symmetric_grid)], 0.0)


def test_a_six_centimetre_depression_stays_under_the_one_sided_band():
    depth = flat()
    depth[DEEP] += 0.06
    grid = grid_of(build(depth, {"plane_snap_below_m": 0.03})[0])
    assert np.isfinite(grid[DEEP_CORE]).all()
    assert np.allclose(grid[DEEP_CORE], -0.06)


def test_a_two_and_a_half_centimetre_depression_is_flat_under_the_one_sided_band_and_kept_by_the_symmetric_one():
    depth = flat()
    depth[DEEP] += 0.025
    kept = grid_of(build(depth)[0])
    flattened = grid_of(build(depth, {"plane_snap_below_m": 0.03})[0])
    assert np.allclose(kept[DEEP_CORE], -0.025)
    assert np.allclose(flattened[DEEP_CORE], 0.0)


def test_a_four_centimetre_depression_with_soft_edges_loses_its_shallow_edge_but_not_its_middle():
    # The sensor smears an edge over a couple of pixels; the shallow part of that edge is what the band flattens.
    mask = region_mask(DEEP).astype(float)
    depth = flat() + 0.04 * gaussian_filter(mask, sigma=2.0)
    unsnapped = grid_of(build(depth, {"plane_snap_tol_m": 0.0})[0])
    snapped = grid_of(build(depth, {"plane_snap_below_m": 0.03})[0])
    below_half = int(np.count_nonzero(unsnapped < -0.02))
    still_below_half = int(np.count_nonzero(snapped < -0.02))
    assert 0 < still_below_half < below_half
    assert np.nanmax(snapped[DEEP_CORE]) < -0.035


# 4. an object standing in a lowered region -------------------------------------------------------------------------

def test_an_object_on_a_flat_lowered_ring_is_filled_at_the_ring_height_and_follows_the_band():
    depth = flat()
    depth[WIDE] += 0.02
    default_mesh, _, default_stats = build(depth, mask=box_mask())
    one_sided_mesh, _, _ = build(depth, {"plane_snap_below_m": 0.03}, mask=box_mask())
    object_core = region_mask(OBJECT_CORE)
    assert default_stats["object_pixels_local_surface"] > 0
    assert np.allclose(grid_of(default_mesh)[object_core & np.isfinite(grid_of(default_mesh))], -0.02)
    assert np.allclose(grid_of(one_sided_mesh)[object_core & np.isfinite(grid_of(one_sided_mesh))], 0.0)


def test_an_object_on_a_ring_that_mixes_two_heights_is_filled_at_the_dominant_one():
    depth = flat()
    depth[LEFT_HALF] += 0.02
    default_mesh, _, default_stats = build(depth, mask=box_mask())
    one_sided_mesh, _, _ = build(depth, {"plane_snap_below_m": 0.03}, mask=box_mask())
    object_core = region_mask(OBJECT_CORE)
    assert default_stats["object_pixels_ring_level"] > 0  # the ring is two surfaces, so it is the dominant height (kind 2)
    assert default_stats["object_pixels_local_surface"] == 0
    assert np.allclose(grid_of(default_mesh)[object_core & np.isfinite(grid_of(default_mesh))], -0.02)
    assert np.allclose(grid_of(one_sided_mesh)[object_core & np.isfinite(grid_of(one_sided_mesh))], 0.0)


# 5. the counts ----------------------------------------------------------------------------------------------------

def test_the_snapped_counts_split_the_snapped_plane_vertices_by_their_side():
    depth = flat()
    depth[LOWERED] += 0.02          # 2 cm under the plane: snapped by the 3 cm band below
    depth[RAISED_SMALL] -= 0.005    # 5 mm over the plane: snapped by the 1 cm band above
    _, _, stats = build(depth, {"plane_snap_below_m": 0.03})
    assert stats["vertices_snapped_from_below"] > 0 and stats["vertices_snapped_from_above"] > 0
    assert stats["vertices_snapped_from_below"] + stats["vertices_snapped_from_above"] <= stats["vertices_snapped_to_plane"]


# 6. validation and zero ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["plane_snap_below_m", "plane_snap_above_m"])
@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf"), float("-inf"), True, False, "0.03", [0.03]])
def test_a_value_that_is_not_null_or_a_finite_number_of_at_least_zero_is_an_error_naming_the_key(key, value):
    with pytest.raises(ValueError) as error:
        build(flat(), {key: value})
    assert f"s_bg_mesh.{key}" in str(error.value)
    assert repr(value) in str(error.value)


@pytest.mark.parametrize("key", ["plane_snap_below_m", "plane_snap_above_m"])
@pytest.mark.parametrize("value", [None, 0, 0.0, 0.03, 1])
def test_null_zero_and_non_negative_numbers_are_accepted(key, value):
    mesh, _, _ = build(flat(), {key: value})
    assert len(mesh.vertices) > 0


def test_a_zero_below_turns_the_lower_band_off():
    # 5 mm under the plane: the 1 cm tolerance a null side would use snaps it, a zero lower band keeps it.
    depth = flat()
    depth[LOWERED] += 0.005
    region = region_mask(LOWERED)
    default_grid = grid_of(build(depth)[0])
    off_grid = grid_of(build(depth, {"plane_snap_below_m": 0})[0])
    assert np.allclose(default_grid[region & np.isfinite(default_grid)], 0.0)
    assert np.allclose(off_grid[region & np.isfinite(off_grid)], -0.005)


def test_a_zero_above_turns_the_upper_band_off():
    depth = flat()
    depth[LOWERED] -= 0.005  # 5 mm over the plane: snapped by the 1 cm band, kept by a zero upper band
    default_grid = grid_of(build(depth)[0])
    no_upper = grid_of(build(depth, {"plane_snap_above_m": 0})[0])
    region = region_mask(LOWERED)
    assert np.allclose(default_grid[region & np.isfinite(default_grid)], 0.0)
    assert np.allclose(no_upper[region & np.isfinite(no_upper)], 0.005)


# 7. the defaults change nothing ------------------------------------------------------------------------------------

@pytest.mark.parametrize("with_mask", [False, True])
def test_the_defaults_and_the_explicit_symmetric_band_give_the_same_mesh_and_stats(with_mask):
    depth = flat(with_box=True)
    mask = box_mask() if with_mask else None
    settings = [{}, {"plane_snap_below_m": None, "plane_snap_above_m": None},
                {"plane_snap_below_m": 0.01, "plane_snap_above_m": 0.01}]
    reference_mesh, reference_texture, reference_stats = build(depth, settings[0], mask)
    for setting in settings[1:]:
        mesh, texture, stats = build(depth, setting, mask)
        assert np.array_equal(mesh.vertices, reference_mesh.vertices)
        assert np.array_equal(mesh.faces, reference_mesh.faces)
        assert np.array_equal(mesh.visual.uv, reference_mesh.visual.uv)
        assert np.array_equal(np.asarray(texture), np.asarray(reference_texture))
        assert without_new_keys(stats) == without_new_keys(reference_stats)


def test_the_plane_flag_is_the_valid_pixels_inside_the_band_of_the_moved_heights():
    # Wiring, read from the source. Only valid pixels reach a triangle, so the flag matters through them alone, and it
    # must read the heights after the object pixels moved (the flag is computed after the object fill, from the moved z).
    tree = ast.parse(pathlib.Path(depth_mesh.__file__).read_text())
    build_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_background_mesh")
    flags = [ast.unparse(n.value) for n in ast.walk(build_function) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "plane_flag" for t in n.targets)]
    assert flags == ["in_range & plane_snap_mask(points[..., 2], below, above)"]


def test_the_snap_judges_an_object_pixel_where_it_lands_not_where_it_was_measured():
    # The box top is 10 cm up, so its pixels are measured far above the plane; the ring fill puts them at -2 cm, where
    # the band snaps them. Judging the measured height would leave them at -2 cm.
    depth = flat(with_box=True)
    depth[WIDE] += 0.02
    default_grid = grid_of(build(depth, mask=box_mask())[0])
    one_sided_grid = grid_of(build(depth, {"plane_snap_below_m": 0.03}, mask=box_mask())[0])
    object_core = region_mask(OBJECT_CORE)
    assert np.allclose(default_grid[object_core & np.isfinite(default_grid)], -0.02)
    assert np.allclose(one_sided_grid[object_core & np.isfinite(one_sided_grid)], 0.0)


def test_the_old_symmetric_key_and_both_sides_at_the_same_value_give_the_same_mesh():
    depth = flat(with_box=True)
    old = build(depth, {"plane_snap_tol_m": 0.03})
    both = build(depth, {"plane_snap_below_m": 0.03, "plane_snap_above_m": 0.03})
    assert np.array_equal(old[0].vertices, both[0].vertices)
    assert np.array_equal(old[0].faces, both[0].faces)
    assert {k: v for k, v in old[2].items() if k not in NEW_KEYS and k != "params"} == \
        {k: v for k, v in both[2].items() if k not in NEW_KEYS and k != "params"}
