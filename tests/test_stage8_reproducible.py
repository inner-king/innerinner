# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 8 seeds each object from the config seed and its index, so an object's draws do not depend on the other objects.

Stage 8 draws random numbers from numpy's global state (the candidate rotations of `sample_fits`, and the start of
`quaternion_kmeans`) and from Open3D's sampler (the Poisson-disk target). Python's `random` is seeded as well. The pool
helper `sample_fits_multiprocess` seeds each batch separately, so scheduling cannot change its answer.

The pool is tested with forkserver children. Under fork, the children stalled on their first candidate in this
environment, and the pool's workers return Open3D point clouds and probreg transformations, which cannot be pickled.
So the pool workers here return the same numbers (distance, rotation, translation, scale) as plain arrays. The pool's
own code, the batch seeds and the gathering of the results, is the stage's and is not replaced.

The stage is loaded with importlib on the CPU: faiss asks for a GPU unless ENFORCE_FAISS_GPU=0.
"""

import ast
import importlib.util
import multiprocessing
import os
import pathlib
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial.transform import Rotation as R

STAGE = pathlib.Path(__file__).resolve().parents[1] / "scripts/pipeline/A_reconstruction/stages/8_match_object_poses.py"
SEED = 0
# Batches 0 and 1 wait first in the delayed run, so the later batches finish before them.
DELAYS = {0: 1.0, 1: 0.5}
_CHILD = {"stage": None}


def _load_stage(name):
    # Importing the stage calls bootstrap_hydra_workdir, which changes the working directory, so it is restored.
    cwd = os.getcwd()
    spec = importlib.util.spec_from_file_location(name, STAGE)
    module = importlib.util.module_from_spec(spec)
    # Pool children and pickling find the module by its name.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


@pytest.fixture(scope="module")
def stage8():
    with pytest.MonkeyPatch.context() as mp:
        # faiss_utils reads ENFORCE_FAISS_GPU on each search, so it stays set for the whole module.
        mp.setenv("ENFORCE_FAISS_GPU", "0")
        mp.setenv("CUDA_VISIBLE_DEVICES", "")
        yield _load_stage("stage8_under_test")


@pytest.fixture(scope="module")
def clouds():
    """A 10 x 6 x 4 cm box: 600 target points, and a source made of half its top face, scaled by 1.1 and moved.

    The box is not symmetric, so another seed gives another candidate set and another fit.
    """
    box = o3d.geometry.TriangleMesh.create_box(0.10, 0.06, 0.04)
    o3d.utility.random.seed(SEED)
    target = box.sample_points_uniformly(600)
    points = np.asarray(target.points)
    top = points[np.isclose(points[:, 2], points[:, 2].max())]
    half = top[top[:, 0] < np.median(top[:, 0])]
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(half * 1.1 + np.array([0.01, 0.005, 0.02]))
    return source, target


def _same_fit(a, b):
    return (a["dist"] == b["dist"] and np.array_equal(a["tf_z_up"].rot, b["tf_z_up"].rot)
            and np.array_equal(a["tf_z_up"].t, b["tf_z_up"].t) and a["tf_z_up"].scale == b["tf_z_up"].scale)


def _same_projection(a, b):
    return a["dist"] == b["dist"] and np.array_equal(a["rot"], b["rot"]) and np.array_equal(a["t"], b["t"]) and a["scale"] == b["scale"]


def _poisson(box, seed):
    o3d.utility.random.seed(seed)
    return np.asarray(box.sample_points_poisson_disk(number_of_points=10000).points)


# ---------------------------------------------------------------------------
# The random sources, each repeated for the same seed
# ---------------------------------------------------------------------------

def test_quaternion_kmeans_repeats_after_seeding_and_the_seed_changes_its_start(stage8):
    rng = np.random.default_rng(SEED)
    quats = rng.normal(size=(60, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    np.random.seed(SEED)
    centers_a, clusters_a = stage8.quaternion_kmeans(quats, 4)
    np.random.seed(SEED)
    centers_b, clusters_b = stage8.quaternion_kmeans(quats, 4)

    assert clusters_a == clusters_b
    assert len(centers_a) == len(centers_b)
    assert all(np.array_equal(x, y) for x, y in zip(centers_a, centers_b))
    outcomes = set()
    for seed in range(5):
        np.random.seed(seed)
        _, clusters = stage8.quaternion_kmeans(quats, 4)
        outcomes.add(tuple(tuple(c) for c in clusters))
    assert len(outcomes) > 1


def test_sample_fits_repeats_for_the_same_seed_and_changes_with_the_seed(stage8, clouds):
    source, target = clouds
    np.random.seed(SEED)
    first = stage8.sample_fits(source, target, 10, update_scale=True, return_inverse=True)
    np.random.seed(SEED)
    second = stage8.sample_fits(source, target, 10, update_scale=True, return_inverse=True)
    np.random.seed(SEED + 1)
    other = stage8.sample_fits(source, target, 10, update_scale=True, return_inverse=True)

    assert [f["dist"] for f in first] == [s["dist"] for s in second]
    assert _same_fit(first[0], second[0])
    assert [f["dist"] for f in other] != [f["dist"] for f in first]


def test_poisson_target_repeats_after_seeding_open3d():
    box = o3d.geometry.TriangleMesh.create_box(0.10, 0.06, 0.04)
    first = _poisson(box, SEED)
    again = _poisson(box, SEED)
    other = _poisson(box, SEED + 1)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


# ---------------------------------------------------------------------------
# The pool: the same answer for each thread count and each scheduling
# ---------------------------------------------------------------------------

def stage_worker(source_pts, target_pts, ori_mats, process_id, seed):
    """The stage's own batch worker, run in a pool child, with its result reduced to picklable arrays."""
    if _CHILD["stage"] is None:
        _CHILD["stage"] = _load_stage("stage8_pool_child")
    info = _CHILD["stage"]._sample_fits(source_pts, target_pts, ori_mats, process_id, seed)
    tf = info["tf_z_up"]
    return {"dist": float(info["dist"]), "rot": tf.rot, "t": tf.t, "scale": float(tf.scale)}


def delayed_stage_worker(source_pts, target_pts, ori_mats, process_id, seed):
    time.sleep(DELAYS.get(process_id, 0.0))
    return stage_worker(source_pts, target_pts, ori_mats, process_id, seed)


class _RecordingPool:
    """The stage's pool, wrapped so the parent sees each batch's seed and candidates, and the order the batches finish."""

    def __init__(self, pool, seeds, finished, batches):
        self._pool, self._seeds, self._finished, self._batches = pool, seeds, finished, batches

    def apply_async(self, func, args=(), **kwargs):
        process_id, batch_seed = args[3], args[4]
        self._seeds.append((process_id, batch_seed))
        self._batches.append(args[2])
        return self._pool.apply_async(func, args=args, callback=lambda _: self._finished.append(process_id))

    def close(self):
        self._pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return self._pool.__exit__(*exc_info)


def _use_forkserver_pool(monkeypatch, stage8, seeds=None, finished=None, batches=None):
    """Makes the stage start its pool with forkserver children, recording what it sends if asked."""
    context = multiprocessing.get_context("forkserver")

    def pool(processes):
        real = context.Pool(processes=processes)
        return real if seeds is None else _RecordingPool(real, seeds, finished, [] if batches is None else batches)

    monkeypatch.setattr(stage8, "multiprocessing", SimpleNamespace(Pool=pool))


@pytest.mark.parametrize("n_threads", [2, 4])
def test_pool_fit_repeats_for_each_thread_count(stage8, clouds, monkeypatch, n_threads):
    source, target = clouds
    _use_forkserver_pool(monkeypatch, stage8)
    monkeypatch.setattr(stage8, "_sample_fits", stage_worker)
    first = stage8.sample_fits_multiprocess(source, target, 8, n_threads, seed=SEED)
    second = stage8.sample_fits_multiprocess(source, target, 8, n_threads, seed=SEED)

    assert _same_projection(first, second)


def test_pool_answer_does_not_depend_on_the_thread_count(stage8, clouds, monkeypatch):
    source, target = clouds
    _use_forkserver_pool(monkeypatch, stage8)
    monkeypatch.setattr(stage8, "_sample_fits", stage_worker)
    two = stage8.sample_fits_multiprocess(source, target, 8, 2, seed=SEED)
    four = stage8.sample_fits_multiprocess(source, target, 8, 4, seed=SEED)

    assert _same_projection(two, four)


def test_the_pool_candidates_are_the_rotations_drawn_from_the_seed(stage8, clouds, monkeypatch):
    source, target = clouds
    batches = []
    _use_forkserver_pool(monkeypatch, stage8, [], [], batches)
    monkeypatch.setattr(stage8, "_sample_fits", stage_worker)
    stage8.sample_fits_multiprocess(source, target, 8, 2, seed=SEED)

    expected = np.array_split(R.random(8, random_state=SEED).as_matrix(), 2)
    assert len(batches) == 2
    assert all(np.array_equal(got, want) for got, want in zip(batches, expected))


def test_each_batch_has_its_own_seed_and_the_finish_order_does_not_change_the_answer(stage8, clouds, monkeypatch):
    source, target = clouds
    _use_forkserver_pool(monkeypatch, stage8)
    monkeypatch.setattr(stage8, "_sample_fits", stage_worker)
    reference = stage8.sample_fits_multiprocess(source, target, 8, 4, seed=SEED)

    seeds, finished = [], []
    _use_forkserver_pool(monkeypatch, stage8, seeds, finished)
    monkeypatch.setattr(stage8, "_sample_fits", delayed_stage_worker)
    delayed = stage8.sample_fits_multiprocess(source, target, 8, 4, seed=SEED)

    assert sorted(seeds) == [(process_id, SEED + process_id) for process_id in range(4)]
    assert finished != [0, 1, 2, 3]  # the delays really changed the order in which the batches finished
    assert _same_projection(reference, delayed)


def test_a_worker_starts_from_its_own_seed_whatever_the_parent_state(stage8, clouds):
    source, target = clouds
    ori_mats = R.random(2, random_state=SEED).as_matrix()
    args = (np.asarray(source.points), np.asarray(target.points), ori_mats, 0)

    np.random.seed(SEED + 1)  # the parent's state, which must not matter
    stage8._sample_fits(*args, 5)
    after = np.random.get_state()

    # The worker draws no numbers of its own, so numpy is left in exactly the state its batch seed (5) gives.
    expected = np.random.RandomState(5).get_state()
    assert after[0] == expected[0] and np.array_equal(after[1], expected[1]) and after[2] == expected[2]


def test_results_are_gathered_in_batch_order_whatever_their_readiness(stage8):
    class Pending:
        def __init__(self, value, ready):
            self.value, self._ready = value, ready

        def ready(self):
            return self._ready

        def get(self):
            return self.value

    results = [Pending("batch0", False), Pending("batch1", True), Pending("batch2", True)]
    assert stage8._gather_in_batch_order(results) == ["batch0", "batch1", "batch2"]


# ---------------------------------------------------------------------------
# One seed for every source
# ---------------------------------------------------------------------------

def test_seed_everything_sets_python_numpy_and_open3d_and_the_value_changes_the_state(stage8):
    box = o3d.geometry.TriangleMesh.create_box(0.10, 0.06, 0.04)

    stage8.seed_everything_stage8(7)
    assert random.getstate() == random.Random(7).getstate()
    numpy_state = np.random.get_state()
    expected = np.random.RandomState(7).get_state()
    assert numpy_state[2] == expected[2] and np.array_equal(numpy_state[1], expected[1])
    first = np.asarray(box.sample_points_poisson_disk(number_of_points=10000).points)

    stage8.seed_everything_stage8(7)
    again = np.asarray(box.sample_points_poisson_disk(number_of_points=10000).points)
    assert np.array_equal(first, again)

    stage8.seed_everything_stage8(8)
    assert random.getstate() == random.Random(8).getstate()
    assert random.getstate() != random.Random(7).getstate()
    other = np.asarray(box.sample_points_poisson_disk(number_of_points=10000).points)
    assert not np.array_equal(first, other)


def test_open3d_is_seeded_with_the_config_value_itself(stage8):
    box = o3d.geometry.TriangleMesh.create_box(0.10, 0.06, 0.04)
    stage8.seed_everything_stage8(SEED + 2)
    via_stage = np.asarray(box.sample_points_poisson_disk(number_of_points=2000).points)
    o3d.utility.random.seed(SEED + 2)
    direct = np.asarray(box.sample_points_poisson_disk(number_of_points=2000).points)

    assert np.array_equal(via_stage, direct)


def test_a_numpy_integer_seed_is_cast_to_a_plain_int_first(stage8):
    # Python's random accepts no numpy integers, so the cast has to come before it.
    stage8.seed_everything_stage8(np.int64(SEED + 7))

    assert random.getstate() == random.Random(SEED + 7).getstate()


# ---------------------------------------------------------------------------
# Wiring, read from the source
# ---------------------------------------------------------------------------

def _main_node():
    tree = ast.parse(STAGE.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name]


def test_main_takes_the_seed_from_the_config_and_seeds_before_any_draw():
    main = _main_node()
    (seed_assign,) = [n for n in main.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "seed"]
    seed_calls = [n for n in main.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                  and getattr(n.value.func, "id", None) == "seed_everything_stage8"]

    assert ast.unparse(seed_assign.value) == "int(cfg.seed)"
    assert len(seed_calls) == 1
    assert [a.id for a in seed_calls[0].value.args] == ["seed"]
    first_fit = min(c.lineno for c in _calls(main, "sample_fits"))
    poisson = [c for c in ast.walk(main) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
               and c.func.attr == "sample_points_poisson_disk"]
    assert seed_calls[0].lineno < first_fit
    assert all(seed_calls[0].lineno < c.lineno for c in poisson)


def test_the_foundationpose_seed_is_the_stage_seed_not_a_constant():
    (call,) = _calls(_main_node(), "set_seed")

    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "seed"


# ---------------------------------------------------------------------------
# Per object: each object reseeds from the seed and its index
# ---------------------------------------------------------------------------

def _object_loop(main):
    return next(n for n in main.body if isinstance(n, ast.For) and getattr(n.target, "id", None) == "filename")


def test_each_object_reseeds_from_the_seed_and_its_index_after_the_skip_and_before_its_first_draw():
    loop = _object_loop(_main_node())
    (skip,) = [n for n in loop.body if isinstance(n, ast.If)
               and any(isinstance(x, ast.Name) and x.id == "requested_indices" for x in ast.walk(n.test))]
    reseeds = [c for c in ast.walk(loop) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "seed_everything_stage8"]
    draws = [c for c in ast.walk(loop) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
             and c.func.attr == "sample_points_poisson_disk"]

    assert len(reseeds) == 1
    (reseed,) = reseeds
    offset = reseed.args[0]
    assert isinstance(offset, ast.BinOp) and isinstance(offset.op, ast.Add)
    assert isinstance(offset.left, ast.Name) and isinstance(offset.right, ast.Name)
    assert {offset.left.id, offset.right.id} == {"seed", "idx"}
    assert skip.end_lineno < reseed.lineno < min(c.lineno for c in draws)


def _draws_of(stage8, idx, box):
    """One object's draws in the order the loop makes them: its reseed, then numpy, Python and Open3D."""
    stage8.seed_everything_stage8(SEED + idx)
    return (np.random.random(4), random.random(),
            np.asarray(box.sample_points_poisson_disk(number_of_points=2000).points))


def test_an_object_draws_the_same_whether_or_not_earlier_objects_ran(stage8):
    box = o3d.geometry.TriangleMesh.create_box(0.10, 0.06, 0.04)

    alone = _draws_of(stage8, 5, box)
    _draws_of(stage8, 3, box)  # object 3 runs first and consumes numbers
    np.random.random(100)      # and more unrelated draws come in between
    after_three = _draws_of(stage8, 5, box)

    assert np.array_equal(alone[0], after_three[0]) and alone[1] == after_three[1]
    assert np.array_equal(alone[2], after_three[2])


def test_one_seed_for_the_whole_run_lets_an_earlier_object_change_the_next_one(stage8):
    # Why each object reseeds: with one seed for the run, object 5 draws different numbers after object 3.
    box = o3d.geometry.TriangleMesh.create_box(0.10, 0.06, 0.04)
    stage8.seed_everything_stage8(SEED)
    np.random.random(4)
    box.sample_points_poisson_disk(number_of_points=2000)
    after_three = np.random.random(4)
    stage8.seed_everything_stage8(SEED)
    alone = np.random.random(4)

    assert not np.array_equal(after_three, alone)
