# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 8's column fit (simfoundry/pipeline/column_fit.py) on synthetic meshes.

Every mesh is a closed height field built on a grid: a top surface, a bottom surface and the side walls between them.
The world is Z-up with the support at z = 0 unless a test says otherwise; depth points are samples of the top surface
the sensor would see.
"""

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation as R

from simfoundry.pipeline import column_fit as cf

UP = np.array([0.0, 0.0, 1.0])


def heightfield_mesh(top, bottom, half=0.05, n=24):
    """A closed mesh between `bottom(x, y)` and `top(x, y)` over the square [-half, half]^2."""
    xs = np.linspace(-half, half, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    t = np.stack([gx, gy, top(gx, gy)], -1).reshape(-1, 3)
    b = np.stack([gx, gy, bottom(gx, gy)], -1).reshape(-1, 3)
    vertices = np.concatenate([t, b])
    idx = np.arange(n * n).reshape(n, n)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b_, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            faces += [[a, b_, c], [a, c, d]]                                   # top, facing up
            faces += [[a + n * n, c + n * n, b_ + n * n], [a + n * n, d + n * n, c + n * n]]   # bottom, facing down
    rim = list(idx[:, 0]) + list(idx[-1, 1:]) + list(idx[::-1, -1][1:]) + list(idx[0, ::-1][1:-1])
    for k in range(len(rim)):
        p, q = rim[k], rim[(k + 1) % len(rim)]
        faces += [[p, p + n * n, q], [q, p + n * n, q + n * n]]
    mesh = trimesh.Trimesh(vertices, np.array(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def top_points(top, half=0.05, step=0.002):
    xs = np.arange(-half, half + 1e-9, step)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([gx, gy, top(gx, gy)], -1).reshape(-1, 3)


ON = cf.resolve_settings({"enabled": True})


def test_a_floating_bowed_plate_is_laid_between_the_support_and_the_depth_top():
    bow = lambda x, y: 1.6 * (x ** 2 + y ** 2)          # 8 mm higher at the rim than in the middle
    plate = heightfield_mesh(lambda x, y: 0.020 + bow(x, y), lambda x, y: 0.010 + bow(x, y))
    new, rec = cf.column_fit(plate.vertices, plate.faces, top_points(lambda x, y: np.full_like(x, 0.023)), UP, 0.0, ON)
    assert rec["applied"], rec
    assert rec["thickness_uniformity"] > 0.95
    fitted = trimesh.Trimesh(new, plate.faces)
    z = new[:, 2]
    assert abs(z.min()) < 1e-6
    # The rim columns read the depth through the eroded, gap-filled top, so allow a little at the edge.
    assert np.percentile(z, 99) == pytest.approx(0.023, abs=1e-3)
    assert fitted.volume == pytest.approx(0.1 * 0.1 * 0.023, rel=0.03)
    assert plate.volume == pytest.approx(0.1 * 0.1 * 0.010, rel=0.03)


def test_the_uv_order_of_vertices_and_the_faces_are_kept():
    plate = heightfield_mesh(lambda x, y: np.full_like(x, 0.015), lambda x, y: np.full_like(x, 0.008))
    new, rec = cf.column_fit(plate.vertices, plate.faces, top_points(lambda x, y: np.full_like(x, 0.030)), UP, 0.0, ON)
    assert rec["applied"]
    assert new.shape == plate.vertices.shape
    assert np.allclose(new[:, :2], plate.vertices[:, :2])        # only the height moves


def test_the_frame_does_not_matter():
    plate = heightfield_mesh(lambda x, y: np.full_like(x, 0.015), lambda x, y: np.full_like(x, 0.008))
    pts = top_points(lambda x, y: np.full_like(x, 0.030))
    rot = R.from_euler("xyz", [30, -20, 75], degrees=True).as_matrix()
    shift = np.array([0.1, -0.2, 0.7])
    new_w, _ = cf.column_fit(plate.vertices, plate.faces, pts, UP, 0.0, ON)
    new_c, rec = cf.column_fit(plate.vertices @ rot.T + shift, plate.faces, pts @ rot.T + shift, rot @ UP,
                               float((rot @ UP) @ shift), ON)
    assert rec["applied"]
    assert np.allclose((new_c - shift) @ rot, new_w, atol=1e-4)


def test_a_mound_thinning_to_its_edge_is_not_slab_like_and_is_left_alone():
    mound = lambda x, y: 0.001 + 0.03 * np.cos(np.pi / 2 * np.maximum(np.abs(x), np.abs(y)) / 0.05)
    dome = heightfield_mesh(mound, lambda x, y: np.zeros_like(x))
    new, rec = cf.column_fit(dome.vertices, dome.faces, top_points(lambda x, y: 2.0 * mound(x, y)), UP, 0.0, ON)
    assert new is None
    assert "not slab-like" in rec["skipped"]


def test_a_tray_is_a_container_and_is_left_alone():
    rim = lambda x, y: np.where(np.maximum(np.abs(x), np.abs(y)) > 0.04, 0.030, 0.004)
    tray = heightfield_mesh(rim, lambda x, y: np.zeros_like(x), n=41)
    new, rec = cf.column_fit(tray.vertices, tray.faces, top_points(lambda x, y: rim(x, y) + 0.003), UP, 0.0, ON)
    assert new is None
    assert "container" in rec["skipped"]


def test_a_bowed_plate_the_sensor_sees_flat_is_not_a_container():
    curl = lambda x, y: 4.0 * np.maximum(np.abs(x), np.abs(y)) ** 2      # the rim curls up 10 mm
    plate = heightfield_mesh(lambda x, y: 0.012 + curl(x, y), lambda x, y: 0.006 + curl(x, y))
    new, rec = cf.column_fit(plate.vertices, plate.faces, top_points(lambda x, y: np.full_like(x, 0.023)), UP, 0.0, ON)
    assert rec["mesh_depression"] > ON["max_depression"]
    assert rec["applied"], rec


def test_a_mesh_the_depth_agrees_with_is_left_alone():
    block = heightfield_mesh(lambda x, y: np.full_like(x, 0.022), lambda x, y: np.zeros_like(x))
    new, rec = cf.column_fit(block.vertices, block.faces, top_points(lambda x, y: np.full_like(x, 0.023)), UP, 0.0, ON)
    assert new is None
    assert "agrees" in rec["skipped"]
    assert rec["relative_difference"] < ON["threshold"]


def test_an_unknown_support_is_left_alone():
    plate = heightfield_mesh(lambda x, y: np.full_like(x, 0.020), lambda x, y: np.full_like(x, 0.012))
    pts = top_points(lambda x, y: np.full_like(x, 0.023))
    under = top_points(lambda x, y: np.full_like(x, 0.015))
    new, rec = cf.column_fit(plate.vertices, plate.faces, pts, UP, 0.0, ON, under_points=under)
    assert new is None
    assert "support is not known" in rec["skipped"]
    agreeing = top_points(lambda x, y: np.full_like(x, 0.002))
    new, rec = cf.column_fit(plate.vertices, plate.faces, pts, UP, 0.0, ON, under_points=agreeing)
    assert rec["applied"]


def test_too_few_depth_columns_are_refused():
    plate = heightfield_mesh(lambda x, y: np.full_like(x, 0.020), lambda x, y: np.full_like(x, 0.012))
    pts = top_points(lambda x, y: np.full_like(x, 0.023), half=0.004)
    new, rec = cf.column_fit(plate.vertices, plate.faces, pts, UP, 0.0, ON)
    assert new is None
    assert "columns keep a depth top" in rec["skipped"]


def test_two_crossings_are_counted_per_column_of_a_closed_slab():
    block = heightfield_mesh(lambda x, y: np.full_like(x, 0.02), lambda x, y: np.zeros_like(x))
    columns = cf.mesh_columns(block.vertices, block.faces, UP, 0.003)
    crossed = columns["hits"] > 0
    assert (columns["hits"][crossed] == 2).mean() > 0.95
    assert np.nanmax(columns["top"]) == pytest.approx(0.02)
    assert np.nanmin(columns["bottom"]) == pytest.approx(0.0, abs=1e-9)


def test_lowest_mode_takes_the_lower_of_two_populated_surfaces_wherever_they_sit():
    rng = np.random.default_rng(0)
    floor = rng.normal(0.011, 0.0008, 300)
    shelf = rng.normal(0.018, 0.0008, 320)
    for shift in (0.0, 0.0013, 0.0027, -0.6925):
        assert cf.lowest_mode(np.concatenate([floor, shelf]) + shift) == pytest.approx(0.011 + shift, abs=0.001)


def test_lowest_mode_ignores_a_thin_lower_tail():
    rng = np.random.default_rng(1)
    heights = np.concatenate([rng.normal(0.010, 0.0008, 500), rng.normal(-0.02, 0.0008, 20)])
    assert cf.lowest_mode(heights) == pytest.approx(0.010, abs=0.001)


def test_settings_default_off_and_validated():
    assert cf.resolve_settings(None)["enabled"] is False
    assert cf.resolve_settings({"enabled": True, "cell_m": 0.004})["cell_m"] == 0.004
    with pytest.raises(ValueError, match="cell_m"):
        cf.resolve_settings({"cell_m": 0})
    with pytest.raises(ValueError, match="top_erode_cells"):
        cf.resolve_settings({"top_erode_cells": 1.5})
    with pytest.raises(ValueError, match="min_top_cells"):
        cf.resolve_settings({"min_top_cells": True})


def test_the_config_default_is_the_code_default_and_off():
    yaml = pytest.importorskip("yaml")
    import pathlib
    cfg = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "scripts/cfg/real2sim_cfg.yaml").read_text())
    assert cfg["s8_pose"]["column_fit"] == cf.DEFAULT_COLUMN_FIT
    assert cfg["s8_pose"]["column_fit"]["enabled"] is False


def test_stage_8_calls_the_fit_and_records_it_only_when_enabled():
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts/pipeline/A_reconstruction/stages/8_match_object_poses.py").read_text()
    tree = ast.parse(src)
    guarded = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "column_fit_settings[\"enabled\"]" in ast.unparse(node.test).replace("'", "\""):
            body = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
            guarded += ("_column_fit_canonical(" in body) + ("out_info[\"column_fit\"]" in body.replace("'", "\""))
    assert guarded == 2
    calls = src.count("_column_fit_canonical(")
    assert calls == 2          # the definition and the one guarded call
