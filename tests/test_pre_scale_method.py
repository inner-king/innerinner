# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pre-scale method of stage 8 (`s8_pose.pre_scale_method`, B12), on synthetic clouds (CPU, no GPU).

The pure functions are in simfoundry/pipeline/pre_scale.py and run on Open3D's real boxes. Stage 8 is read as source for
its wiring, because it needs the GPU stack to import. The degenerate and non-orthogonal cases substitute a box where Open3D
does not produce one, by monkeypatching the point cloud's method.
"""

import ast
import json
import pathlib

import numpy as np
import open3d as o3d
import pytest
import yaml
from scipy.spatial.transform import Rotation

from simfoundry.pipeline import pre_scale as ps

REPO = pathlib.Path(__file__).resolve().parents[1]
STAGE8 = REPO / "scripts/pipeline/A_reconstruction/stages/8_match_object_poses.py"
CFG = REPO / "scripts/cfg/real2sim_cfg.yaml"
README = REPO / "scripts/pipeline/README.md"

BOX_SIZE = (0.10, 0.099, 0.02)  # a 10.0 x 9.9 x 2.0 cm plate: nearly square, so its PCA axes are ill-conditioned


def cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def mesh_of(vertices):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    return mesh


def box_corners():
    a, b, c = BOX_SIZE
    return np.array([[x, y, z] for x in (0.0, a) for y in (0.0, b) for z in (0.0, c)])


def box_surface(n, seed, size=BOX_SIZE):
    """Points spread uniformly over the six faces of a box, so every face is present."""
    rng = np.random.default_rng(seed)
    sizes = np.asarray(size)
    areas = np.array([sizes[1] * sizes[2], sizes[1] * sizes[2], sizes[0] * sizes[2],
                      sizes[0] * sizes[2], sizes[0] * sizes[1], sizes[0] * sizes[1]])
    face = rng.choice(6, size=n, p=areas / areas.sum())
    uv = rng.uniform(size=(n, 2))
    points = np.zeros((n, 3))
    for f in range(6):
        axis, side = f // 2, f % 2
        sel = face == f
        others = [d for d in range(3) if d != axis]
        points[sel, axis] = side * sizes[axis]
        points[sel, others[0]] = uv[sel, 0] * sizes[others[0]]
        points[sel, others[1]] = uv[sel, 1] * sizes[others[1]]
    return points


def flat_plane(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 1.0, size=(n, 2))
    return np.column_stack([xy, np.zeros(n)])


def stage_main():
    tree = ast.parse(STAGE8.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


# 1. the instability this method exists for -----------------------------------------------------------------------------

def test_the_pca_diagonal_of_a_near_square_plate_moves_with_the_sample_and_the_minimal_one_does_not():
    base = box_surface(2000, seed=0)
    rng = np.random.default_rng(1)
    pca, minimal = [], []
    for trial in range(100):
        subset = base[rng.uniform(size=len(base)) < 0.8]  # a different 80 % of the same plate
        rotated = subset @ Rotation.random(random_state=trial).as_matrix().T
        pca.append(ps.pca_diagonal(cloud(rotated)))
        minimal.append(ps.minimal_diagonal(cloud(rotated))[0])
    pca_width = max(pca) / min(pca) - 1.0
    minimal_width = max(minimal) / min(minimal) - 1.0
    assert pca_width > 0.05, f"the PCA case is not unstable here (width {pca_width:.3f}); the test would prove nothing"
    assert minimal_width < 0.01


# 2. the default is the old computation, to the bit ---------------------------------------------------------------------

def test_current_gives_the_inline_ratio_of_the_pca_diagonals_exactly():
    source = cloud(box_surface(3000, seed=1) * 1.3)
    target = cloud(box_surface(10000, seed=2) * 0.7)
    expected = np.linalg.norm(source.get_oriented_bounding_box().extent) / np.linalg.norm(
        target.get_oriented_bounding_box().extent)
    result = ps.select_pre_scale("current", source, mesh_of(box_corners()), target)
    assert result.factor == expected
    assert result.fallback == {}


# 3. obb_minimal does not depend on the seeds -------------------------------------------------------------------------

def test_obb_minimal_gives_the_same_factor_under_different_random_seeds():
    source = cloud(box_surface(3000, seed=3))
    mesh = mesh_of(box_surface(3000, seed=4))
    target = cloud(box_surface(10000, seed=5))
    values = []
    for seed in (0, 1):
        np.random.seed(seed)
        o3d.utility.random.seed(seed)
        values.append(ps.select_pre_scale("obb_minimal", source, mesh, target).factor)
    assert values[0] == values[1]


# 4. the known box ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rotation_seed", [None, 5])
def test_the_minimal_diagonal_of_a_box_is_the_length_of_its_space_diagonal(rotation_seed):
    corners = box_corners()
    if rotation_seed is not None:
        corners = corners @ Rotation.random(random_state=rotation_seed).as_matrix().T
    diagonal, reason = ps.minimal_diagonal(cloud(corners))
    assert reason is None
    assert abs(diagonal - np.sqrt(sum(s * s for s in BOX_SIZE))) < 1e-6


# 5. the ratio definition -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["current", "obb_minimal"])
@pytest.mark.parametrize("scale", [0.5, 1.7, 2.0])
def test_a_source_scaled_uniformly_by_s_gives_a_factor_of_s(method, scale):
    points = box_surface(4000, seed=6)
    result = ps.select_pre_scale(method, cloud(points * scale), mesh_of(points), cloud(points))
    # Open3D's minimal box is computed in float32, so a uniform scale comes back within about 1e-7 relative, not exactly.
    assert result.factor == pytest.approx(scale, rel=1e-6)


# 6. robustness: the fallbacks ------------------------------------------------------------------------------------------

DEGENERATE_SOURCES = {
    "a flat plane": lambda: flat_plane(2000, seed=7),
    "three points": lambda: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    "three distinct points repeated": lambda: np.tile([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], (20, 1)),
}


@pytest.mark.parametrize("name", list(DEGENERATE_SOURCES))
def test_a_degenerate_source_takes_the_pca_diagonal_and_the_fallback_is_recorded(name):
    target = cloud(box_surface(10000, seed=8))
    result = ps.select_pre_scale("obb_minimal", cloud(DEGENERATE_SOURCES[name]()), mesh_of(box_corners()), target)
    assert list(result.fallback) == ["source"]
    assert np.isfinite(result.factor) and result.factor > 0.0
    assert np.isfinite(result.source_diag) and result.source_diag > 0.0


def test_a_flat_mesh_takes_the_pca_diagonal_of_the_target_for_its_mesh_side():
    target = cloud(box_surface(10000, seed=9))
    flat_mesh = mesh_of(flat_plane(2000, seed=10))
    result = ps.select_pre_scale("obb_minimal", cloud(box_surface(3000, seed=11)), flat_mesh, target)
    assert list(result.fallback) == ["mesh"]
    assert result.mesh_diag == ps.pca_diagonal(target)  # the diagonal `current` takes for the mesh side
    assert np.isfinite(result.factor) and result.factor > 0.0


def test_a_cloud_with_no_extent_at_all_is_an_error_that_names_the_side():
    with pytest.raises(ValueError, match="source cloud has no extent"):
        ps.select_pre_scale("obb_minimal", cloud(np.tile([[0.2, 0.3, 0.4]], (50, 1))), mesh_of(box_corners()),
                            cloud(box_surface(10000, seed=12)))


def test_a_non_orthogonal_rotation_falls_back_on_both_sides_and_then_gives_the_current_factor(monkeypatch):
    class SkewedBox:
        extent = np.array([0.1, 0.1, 0.1])
        R = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    monkeypatch.setattr(o3d.geometry.PointCloud, "get_minimal_oriented_bounding_box", lambda self: SkewedBox())
    source = cloud(box_surface(3000, seed=13))
    mesh = mesh_of(box_surface(3000, seed=14))
    target = cloud(box_surface(10000, seed=15))
    result = ps.select_pre_scale("obb_minimal", source, mesh, target)
    assert set(result.fallback) == {"source", "mesh"}
    assert all("not orthogonal" in reason for reason in result.fallback.values())
    assert result.factor == ps.select_pre_scale("current", source, mesh, target).factor


@pytest.mark.parametrize("box, reason_text", [
    ("extent_inf", "not finite and positive"),
    ("rotation_nan", "rotation is not finite"),
    ("extent_zero", "not finite and positive"),
    ("raises", "minimal box raised"),
])
def test_a_minimal_box_that_cannot_be_trusted_is_reported_with_its_reason(monkeypatch, box, reason_text):
    class Box:
        def __init__(self, extent, R):
            self.extent, self.R = np.asarray(extent, dtype=float), np.asarray(R, dtype=float)

    eye = np.eye(3)
    fake = {
        "extent_inf": Box([np.inf, 0.1, 0.1], eye),
        "rotation_nan": Box([0.1, 0.1, 0.1], np.full((3, 3), np.nan)),
        "extent_zero": Box([0.1, 0.1, 0.0], eye),
    }

    def minimal(self):
        if box == "raises":
            raise RuntimeError("probe")
        return fake[box]

    monkeypatch.setattr(o3d.geometry.PointCloud, "get_minimal_oriented_bounding_box", minimal)
    diagonal, reason = ps.minimal_diagonal(cloud(box_corners()))
    assert diagonal is None and reason_text in reason


def test_obb_minimal_is_the_ratio_of_the_minimal_diagonals_and_not_of_the_pca_ones():
    source_points = box_surface(3000, seed=21)[np.random.default_rng(21).uniform(size=3000) < 0.8]
    source_points = source_points @ Rotation.random(random_state=21).as_matrix().T
    mesh_points = box_surface(3000, seed=22)
    target = cloud(box_surface(10000, seed=23))
    result = ps.select_pre_scale("obb_minimal", cloud(source_points), mesh_of(mesh_points), target)
    expected = ps.minimal_diagonal(cloud(source_points))[0] / ps.minimal_diagonal(cloud(mesh_points))[0]
    assert result.factor == expected
    current = ps.select_pre_scale("current", cloud(source_points), mesh_of(mesh_points), target).factor
    assert abs(result.factor / current - 1.0) > 1e-3  # the two methods really differ on this pair


def test_a_source_with_non_finite_points_gives_the_factor_of_the_clean_source():
    clean = box_surface(3000, seed=16)
    dirty = np.vstack([clean, [[np.nan, 0.0, 0.0], [0.0, np.inf, 0.0]]])
    mesh = mesh_of(box_surface(3000, seed=17))
    target = cloud(box_surface(10000, seed=18))
    assert ps.select_pre_scale("obb_minimal", cloud(dirty), mesh, target).factor == \
        ps.select_pre_scale("obb_minimal", cloud(clean), mesh, target).factor


# 7. validation ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["Obb_Minimal", " obb_minimal", "obb_minimal ", "", "minimal", None, 1, ["current"]])
def test_an_unknown_method_is_a_value_error_that_lists_the_allowed_ones(value):
    with pytest.raises(ValueError) as error:
        ps.validate_pre_scale_method(value)
    assert "current, obb_minimal" in str(error.value)
    assert repr(value) in str(error.value)


@pytest.mark.parametrize("value", ["current", "obb_minimal"])
def test_the_two_allowed_methods_are_accepted(value):
    assert ps.validate_pre_scale_method(value) == value


def test_select_pre_scale_refuses_an_unknown_method_before_computing_anything():
    with pytest.raises(ValueError):
        ps.select_pre_scale("Obb_Minimal", cloud(box_surface(100, seed=19)), mesh_of(box_corners()),
                            cloud(box_surface(100, seed=20)))


# 8. the record ---------------------------------------------------------------------------------------------------------

def test_current_records_nothing_and_obb_minimal_records_its_method_and_a_fallback_only_when_one_happened():
    fallback = {"mesh": "minimal box raised: probe; replaced by the PCA box"}
    assert ps.pre_scale_record("current", {}) == {}
    assert ps.pre_scale_record("current", fallback) == {}
    assert ps.pre_scale_record("obb_minimal", {}) == {"pre_scale_method": "obb_minimal"}
    record = ps.pre_scale_record("obb_minimal", fallback)
    assert record == {"pre_scale_method": "obb_minimal", "pre_scale_fallback": fallback}
    assert json.loads(json.dumps(record)) == record


# 9. the stage's wiring, read from its source ----------------------------------------------------------------------------

def test_the_stage_takes_the_factor_from_the_new_function_with_method_source_mesh_and_target_in_that_order():
    main = stage_main()
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "select_pre_scale"]
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args] == ["pre_scale_method", "source", "mesh", "target"]
    factors = [ast.unparse(n.value) for n in ast.walk(main) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "pre_scale_factor" for t in n.targets)]
    assert factors == ["prescale.factor"]


def test_the_voxel_size_stays_on_the_pca_source_diagonal_under_either_method():
    main = stage_main()
    values = {}
    for n in ast.walk(main):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            values.setdefault(n.targets[0].id, []).append(ast.unparse(n.value))
    assert values["source_obb_diag"] == ["np.linalg.norm(source.get_oriented_bounding_box().extent)"]
    assert values["voxel_size"] == ["source_obb_diag / 30"]


def test_the_stage_records_the_method_and_warns_about_a_fallback():
    main = stage_main()
    records = [n for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func) == "out_info.update"
               and n.args and isinstance(n.args[0], ast.Call) and ast.unparse(n.args[0].func) == "pre_scale_record"]
    assert len(records) == 1
    assert [ast.unparse(a) for a in records[0].args[0].args] == ["pre_scale_method", "prescale.fallback"]


def test_the_fallback_warning_is_logged_only_under_the_fallback_check_and_names_the_fallback():
    # A substring check passes for `if False:` as well, so the warning is found in the tree: it has to be the one call
    # under the `prescale.fallback` check, with the fallback text and the fallback itself in its message.
    guards = [n for n in ast.walk(ast.parse(STAGE8.read_text())) if isinstance(n, ast.If)
              and ast.unparse(n.test) == "prescale.fallback"]
    assert len(guards) == 1
    warnings = [c for c in ast.walk(guards[0]) if isinstance(c, ast.Call) and ast.unparse(c.func) == "logger.warning"]
    assert len(warnings) == 1
    message = warnings[0].args[0]
    assert isinstance(message, ast.JoinedStr)
    texts = "".join(v.value for v in message.values if isinstance(v, ast.Constant))
    assert "pre_scale fallback for " in texts
    assert [ast.unparse(v.value) for v in message.values if isinstance(v, ast.FormattedValue)] == [
        "img_name", "prescale.fallback"]


def test_stage_8_validates_the_pre_scale_method_once_outside_every_loop_and_before_the_object_loop():
    main = stage_main()
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "validate_pre_scale_method"]
    assert len(calls) == 1
    parents = {child: node for node in ast.walk(main) for child in ast.iter_child_nodes(node)}
    node = calls[0]
    while node is not main:
        node = parents[node]
        assert not isinstance(node, (ast.For, ast.While)), "validate_pre_scale_method sits inside a loop"
    object_loop = next(n for n in ast.walk(main) if isinstance(n, ast.For)
                       and ast.unparse(n.iter).startswith("sorted(os.listdir(mesh_dir)"))
    assert calls[0].lineno < object_loop.lineno


def test_the_stage_reads_the_method_from_the_config_with_the_default_current_that_the_yaml_has():
    main = stage_main()
    reads = [n for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func) == "cfg.s8_pose.get"
             and n.args and ast.literal_eval(n.args[0]) == "pre_scale_method"]
    assert len(reads) == 1
    default = ast.literal_eval(reads[0].args[1])
    assert default == "current"
    assert yaml.safe_load(CFG.read_text())["s8_pose"]["pre_scale_method"] == default


def test_the_readme_names_the_key_its_values_and_the_fallback_record():
    text = README.read_text()
    for phrase in ("pre_scale_method", "obb_minimal", "pre_scale_fallback", "PCA"):
        assert phrase in text


# 7. numpy PCA: the fallback diagonal, checked on a rectangle in a tilted plane ----------------------------------------

@pytest.mark.parametrize("euler", [(35.0, 60.0, 15.0), (-70.0, 20.0, 110.0)])
def test_numpy_pca_takes_the_diagonal_along_the_principal_axes_of_a_rectangle_in_a_tilted_plane(euler):
    # Sides a and b, sides and corners sampled, lying in a plane that is not one of the coordinate planes. The diagonal
    # is sqrt(a^2 + b^2) along the rectangle's own axes; measured along the world axes it is not.
    a, b = 0.10, 0.04
    t = np.linspace(-0.5, 0.5, 11)
    zeros = np.zeros_like(t)
    local = np.concatenate([
        np.column_stack([t * a, np.full_like(t, -b / 2), zeros]),
        np.column_stack([t * a, np.full_like(t, b / 2), zeros]),
        np.column_stack([np.full_like(t, -a / 2), t * b, zeros]),
        np.column_stack([np.full_like(t, a / 2), t * b, zeros]),
    ])
    points = Rotation.from_euler("xyz", euler, degrees=True).apply(local) + np.array([0.3, -0.2, 0.7])
    assert ps.numpy_pca_diagonal(points) == pytest.approx(float(np.hypot(a, b)), abs=1e-9)
