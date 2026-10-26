# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism of the plane fits of stages 3 and 5's frame scoring, and of stage 5's random draws (CPU, synthetic).

Open3D's RANSAC (`segment_plane`) runs in parallel and stops early with `probability` (0.99999999 by default in Open3D
0.20), so the same seed could give different planes at different thread counts; `probability=1.0` switches the early
stop off. Stage 5's removal order and tie-breaking noise drew from unseeded generators; they now draw from generators
seeded by the run. The stage scripts need the GPU, so the stages are checked by ast and the helpers are run directly.
"""

import ast
import collections
import logging
import pathlib

import numpy as np
import open3d as o3d
import torch
import trimesh

from simfoundry.pipeline import frame_selection as FS
from simfoundry.pipeline.support_graph import support_levels

REPO = pathlib.Path(__file__).resolve().parents[1]
FRAME_SELECTION = REPO / "simfoundry/pipeline/frame_selection.py"
STAGE3 = REPO / "scripts/pipeline/A_reconstruction/stages/3_segment_ground_plane.py"
STAGE5 = REPO / "scripts/pipeline/A_reconstruction/stages/5_decompose_scene.py"
README = REPO / "scripts/pipeline/README.md"


def _floor_with_object(seed=0, n=30000):
    """A floor (z = 0 with 5 mm noise) and a block above it making up 30 % of the points, as the support mask has."""
    rng = np.random.default_rng(seed)
    n_floor = int(0.7 * n)
    floor = np.column_stack([rng.uniform(-1, 1, n_floor), rng.uniform(-1, 1, n_floor), rng.normal(0, 0.005, n_floor)])
    n_block = n - n_floor
    block = np.column_stack([rng.uniform(-0.2, 0.2, n_block), rng.uniform(-0.2, 0.2, n_block),
                             rng.uniform(0.05, 0.25, n_block)])
    return np.vstack([floor, block])


def _clean_floor(seed=1, n=20000):
    """A level floor with 5 mm noise and nothing else."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.normal(0, 0.005, n)])


def _angle_deg(a, b):
    a = np.asarray(a)[:3] / np.linalg.norm(np.asarray(a)[:3])
    b = np.asarray(b)[:3] / np.linalg.norm(np.asarray(b)[:3])
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0.0, 1.0))))


def _module_function(path, name):
    tree = ast.parse(path.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _main_of(path):
    return _module_function(path, "main")


def _calls_named(node, attr):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr]


def _stage5_functions(*names):
    """The named top-level functions of the stage 5 file, compiled alone (the stage's imports are heavy)."""
    tree = ast.parse(STAGE5.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "np": np,
        "torch": torch,
        "trimesh": trimesh,
        "defaultdict": collections.defaultdict,
        "support_levels": support_levels,
        "logger": logging.getLogger("5_decompose_scene"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(STAGE5), "exec"), namespace)
    return tuple(namespace[n] for n in names)


def _cube_hulls():
    """Three convex objects standing side by side, as the removal order sees them (convex hulls)."""
    return {
        "a": trimesh.creation.box(extents=(0.20, 0.20, 0.20)),
        "b": trimesh.creation.box(extents=(0.30, 0.20, 0.10)).apply_translation([0.25, 0.0, 0.0]),
        "c": trimesh.creation.icosphere(subdivisions=2, radius=0.08).apply_translation([-0.2, 0.1, 0.1]),
    }


def _camera():
    cam2world = np.eye(4)
    cam2world[:3, 3] = [0.0, 0.0, 1.5]
    return cam2world


# --- the plane fits -----------------------------------------------------------------------------------------------------

def test_the_frame_plane_is_the_same_at_every_thread_count_for_the_same_seed():
    points = _floor_with_object()
    original = o3d.utility.get_max_threads()
    fits = []
    try:
        for threads in (1, 2, 4, 8):
            for _ in range(2):
                o3d.utility.set_max_threads(threads)
                o3d.utility.random.seed(0)
                fits.append(FS._fit_support_plane(points))
    finally:
        o3d.utility.set_max_threads(original)
    normal0, d0, inliers0 = fits[0]
    for normal, d, inliers in fits[1:]:
        assert np.array_equal(normal, normal0)
        assert d == d0
        assert np.array_equal(inliers, inliers0)


def test_the_ground_fit_repeats_for_a_seed_and_moves_little_between_seeds_or_against_the_early_stop():
    points = _clean_floor()
    plane0, inliers0 = FS.fit_ground_plane(points, 0)
    plane0_again, inliers0_again = FS.fit_ground_plane(points, 0)
    assert np.array_equal(plane0, plane0_again)
    assert np.array_equal(inliers0, inliers0_again)
    plane1, _ = FS.fit_ground_plane(points, 1)
    assert _angle_deg(plane0, plane1) < 1.0
    # The early stop the stage ran with before: same seed, default probability. On a clean floor it lands close.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.utility.random.seed(0)
    default_plane, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    assert _angle_deg(plane0, default_plane) < 0.5


def test_the_ground_fit_pins_the_ransac_settings_and_the_early_stop():
    # stage 3 calls fit_ground_plane, so the RANSAC settings are pinned on that one call, in frame_selection.py
    calls = _calls_named(_module_function(FRAME_SELECTION, "fit_ground_plane"), "segment_plane")
    assert len(calls) == 1
    keywords = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert keywords == {"distance_threshold": "0.01", "ransac_n": "3", "num_iterations": "1000", "probability": "1.0"}


def test_the_ground_fit_returns_the_flat_plane_that_stage_3_unpacks():
    # Open3D's segment_plane returns a flat (4,) array (its "[4, 1]" annotation is not the shape); stage 3 unpacks it as
    # [a, b, c, d], which a (4, 1) array would break
    points = _clean_floor()
    plane, inliers = FS.fit_ground_plane(points, 0)
    assert plane.shape == (4,) and plane.dtype == np.float64
    a, b, c, d = plane
    assert np.array([a, b, c]).shape == (3,)
    assert inliers.dtype.kind == "i" and len(inliers) > 0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    assert len(cloud.select_by_index(inliers).points) == len(inliers)


def test_the_ground_fit_seeds_open3d_with_the_seed_it_is_given():
    points = _floor_with_object()

    def direct(seed):
        o3d.utility.random.seed(seed)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        plane, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000, probability=1.0)
        return np.asarray(plane, dtype=np.float64).reshape(-1)

    # the input must be sensitive to the seed, or the equalities below could not tell seed s from seed s + 1
    assert not np.array_equal(direct(0), direct(1))
    for seed in (0, 1, 2):
        assert np.array_equal(direct(seed), FS.fit_ground_plane(points, seed)[0])


def test_every_plane_fit_turns_the_early_stop_off():
    frame_calls = _calls_named(ast.parse(FRAME_SELECTION.read_text()), "segment_plane")
    assert len(frame_calls) == 2  # the frame-scoring fit and the ground fit
    for call in frame_calls:
        assert {k.arg: ast.unparse(k.value) for k in call.keywords}.get("probability") == "1.0"
    # stage 3 fits its floor through fit_ground_plane and calls segment_plane nowhere itself
    assert not _calls_named(ast.parse(STAGE3.read_text()), "segment_plane")


# --- the seeds around them --------------------------------------------------------------------------------------------

def test_stage_3_fits_the_floor_through_the_seeded_ground_fit_on_every_path():
    main = _main_of(STAGE3)
    fits = [n for n in main.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name) and n.value.func.id == "fit_ground_plane"]
    # one call, a statement of main itself (not inside the img_idx branch), so the pinned path fits the floor too
    assert len(fits) == 1
    args = fits[0].value.args
    assert len(args) == 2
    assert ast.unparse(args[0]) == "pts"
    assert ast.unparse(args[1]) == "int(OmegaConf.select(cfg, 'seed') or 0)"
    assert not fits[0].value.keywords
    # stage 3 seeds the engine nowhere itself: fit_ground_plane does, right before its fit
    assert not [c for c in ast.walk(main) if isinstance(c, ast.Call)
                and ast.unparse(c.func) == "o3d.utility.random.seed"]


def test_frame_scoring_seeds_the_engine_before_the_candidates_are_scored():
    func = _module_function(FRAME_SELECTION, "select_canonical_frame")
    seeds = [n for n in func.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and ast.unparse(n.value.func) == "o3d.utility.random.seed"]
    scoring = [n for n in func.body if isinstance(n, ast.Assign) and "score_one" in ast.unparse(n.value)]
    assert len(seeds) == 1 and len(scoring) == 1
    assert seeds[0].lineno < scoring[0].lineno


# --- stage 5: the removal order and the tie-breaking noise ------------------------------------------------------------

def test_the_removal_order_repeats_for_a_seed_and_leaves_numpys_global_state_alone():
    removal_order, _ = _stage5_functions("compute_object_removal_order", "sample_interior_points")
    hulls = _cube_hulls()
    state_before = np.random.get_state()
    first = removal_order(obj_chs=hulls, cam2world_tf=_camera(), seed=0)
    second = removal_order(obj_chs=hulls, cam2world_tf=_camera(), seed=0)
    state_after = np.random.get_state()
    assert first == second
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_the_removal_order_requires_a_seed():
    removal_order, _ = _stage5_functions("compute_object_removal_order", "sample_interior_points")
    try:
        removal_order(obj_chs=_cube_hulls(), cam2world_tf=_camera())
    except TypeError as error:
        assert "seed" in str(error)
    else:
        raise AssertionError("compute_object_removal_order accepted a call without a seed")


def test_the_seed_changes_the_interior_sample_points_and_the_same_seed_repeats_them():
    (sample,) = _stage5_functions("sample_interior_points")
    hull = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(hull)
    state_before = np.random.get_state()[1].copy()
    same_a = sample(intersector, hull.bounds, 1000, np.random.default_rng(0))
    same_b = sample(intersector, hull.bounds, 1000, np.random.default_rng(0))
    other = sample(intersector, hull.bounds, 1000, np.random.default_rng(1))
    assert np.array_equal(same_a, same_b)
    assert not (same_a.shape == other.shape and np.array_equal(same_a, other))
    assert np.array_equal(np.random.get_state()[1], state_before)
    assert len(same_a) > 0 and np.all(hull.contains(same_a))


def test_the_interior_samples_are_n_ray_samples_and_the_seeded_draw_when_the_hull_fills_its_box():
    (sample,) = _stage5_functions("sample_interior_points")
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))  # a convex hull that fills its own bounding box
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(box)
    n = 1000
    result = sample(intersector, box.bounds, n, np.random.default_rng(7))
    assert len(result) == n  # every candidate is inside the hull, so none is dropped and none is added
    direct = np.random.default_rng(7).uniform(low=box.bounds[0], high=box.bounds[1], size=(n, 3))
    assert np.array_equal(result, direct)


def test_the_tie_break_noise_repeats_for_a_seed_is_tiny_and_keeps_the_input_shape_device_and_dtype():
    (noise,) = _stage5_functions("tie_break_noise")
    like = torch.linspace(-1.0, 1.0, 24, dtype=torch.float32).reshape(4, 6)
    a = noise(like, torch.Generator().manual_seed(0))
    b = noise(like, torch.Generator().manual_seed(0))
    c = noise(like, torch.Generator().manual_seed(1))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == like.shape and a.dtype == like.dtype and a.device == like.device
    assert float(a.min()) >= 0.0 and float(a.max()) < 1e-5
    d = noise(like.double(), torch.Generator().manual_seed(0))
    assert d.dtype == torch.float64 and torch.equal(d, a.double())


def test_stage_5_draws_no_random_number_from_an_unseeded_source():
    text = ast.unparse(ast.parse(STAGE5.read_text()))
    assert "np.random.seed(" not in text
    assert "np.random.uniform(" not in text
    assert "rand_like(" not in text
    removal = _module_function(STAGE5, "compute_object_removal_order")
    names = [a.arg for a in removal.args.args]
    assert "seed" in names
    assert names.index("seed") < len(names) - len(removal.args.defaults)  # required: it has no default


def test_stage_5_passes_the_runs_seed_to_its_random_draws():
    main = _main_of(STAGE5)
    run_seed = [n for n in main.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "run_seed" for t in n.targets)]
    assert len(run_seed) == 1
    assert ast.unparse(run_seed[0].value) == "int(OmegaConf.select(cfg, 'seed') or 0)"
    noise_generator = [n for n in main.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "noise_generator" for t in n.targets)]
    assert len(noise_generator) == 1
    assert ast.unparse(noise_generator[0].value) == "torch.Generator().manual_seed(run_seed)"
    removal_calls = [c for c in ast.walk(main) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                     and c.func.id == "compute_object_removal_order"]
    assert len(removal_calls) == 1
    seed_kw = [k for k in removal_calls[0].keywords if k.arg == "seed"]
    assert len(seed_kw) == 1 and ast.unparse(seed_kw[0].value) == "run_seed"
    noise_calls = [c for c in ast.walk(main) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                   and c.func.id == "tie_break_noise"]
    assert len(noise_calls) == 1
    assert [ast.unparse(a) for a in noise_calls[0].args] == ["k_pred_targets", "noise_generator"]


def test_the_readme_describes_the_reproducibility_of_stages_3_and_5():
    text = README.read_text()
    assert "Reproducibility of stages 3 and 5" in text
    assert "probability=1.0" in text


def test_the_removal_order_draws_its_sample_points_from_its_seed():
    removal_order, sample = _stage5_functions("compute_object_removal_order", "sample_interior_points")
    recorded = []

    def record(*args):
        points = sample(*args)
        recorded.append(points)
        return points

    removal_order.__globals__["sample_interior_points"] = record
    try:
        removal_order(obj_chs=_cube_hulls(), cam2world_tf=_camera(), seed=0)
        seed0 = list(recorded)
        recorded.clear()
        removal_order(obj_chs=_cube_hulls(), cam2world_tf=_camera(), seed=0)
        seed0_again = list(recorded)
        recorded.clear()
        removal_order(obj_chs=_cube_hulls(), cam2world_tf=_camera(), seed=1)
        seed1 = list(recorded)
    finally:
        removal_order.__globals__["sample_interior_points"] = sample
    assert len(seed0) == len(seed0_again) == len(seed1) == 3
    for a, b in zip(seed0, seed0_again):
        assert np.array_equal(a, b)
    assert any(not (a.shape == b.shape and np.array_equal(a, b)) for a, b in zip(seed0, seed1))
