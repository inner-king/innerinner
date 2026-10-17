# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the volumes stage 11 states in its mass prompt (simfoundry/utils/prompt_utils.py).

A textured mesh keeps one copy of each vertex per texture coordinate, so a closed surface loads with open
seams. ``unmerge_vertices`` gives every face its own vertices, the extreme case of that.
"""

import numpy as np
import pytest

from simfoundry.utils.prompt_utils import mass_prompt_volumes, welded_volume

trimesh = pytest.importorskip("trimesh")


def obb_extent(tm):
    return trimesh.bounds.oriented_bounds(tm)[1]


def seamed(tm):
    tm = tm.copy()
    tm.unmerge_vertices()
    assert not tm.is_volume
    return tm


def thin_shell(outer=0.1, wall=0.001):
    """A closed box shell: the outer box plus an inward-facing inner one, like a cup reconstructed with its cavity."""
    inner = trimesh.creation.box(extents=[outer - 2 * wall] * 3)
    inner.invert()
    return trimesh.util.concatenate([trimesh.creation.box(extents=[outer] * 3), inner])


def test_seamed_solid_states_its_welded_volume():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    tm = seamed(sphere)
    ext = obb_extent(tm)

    volume, material = mass_prompt_volumes(tm, ext, weld_seams=True)

    assert volume == pytest.approx(sphere.volume)
    assert material is None


def test_default_keeps_the_box_estimate_for_an_open_mesh():
    """The replays in repro/vlm_compare depend on the original volumes."""
    tm = seamed(trimesh.creation.icosphere(subdivisions=3, radius=0.05))
    ext = obb_extent(tm)

    assert mass_prompt_volumes(tm, ext) == (pytest.approx(float(np.prod(ext)) * 0.65), None)


def test_closed_mesh_is_unaffected_by_welding():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    ext = obb_extent(sphere)

    assert mass_prompt_volumes(sphere, ext, weld_seams=True) == mass_prompt_volumes(sphere, ext)
    assert mass_prompt_volumes(sphere, ext)[0] == pytest.approx(sphere.volume)


def test_closed_thin_shell_states_the_box_estimate_and_its_material():
    shell = thin_shell()
    ext = obb_extent(shell)

    volume, material = mass_prompt_volumes(shell, ext, weld_seams=True)

    assert volume == pytest.approx(float(np.prod(ext)) * 0.65)
    assert material == pytest.approx(shell.volume)
    assert material < 0.1 * float(np.prod(ext))


def test_seamed_thin_shell_keeps_the_box_estimate_without_its_material():
    """Its welded walls would overstate a thin real wall (a paper cup's) several times over."""
    shell = thin_shell()
    tm = seamed(shell)
    ext = obb_extent(tm)

    assert welded_volume(tm) == pytest.approx(shell.volume)
    assert mass_prompt_volumes(tm, ext, weld_seams=True) == (pytest.approx(float(np.prod(ext)) * 0.65), None)


def test_seamed_curved_solid_states_its_welded_volume():
    """A bowed rod, like the banana: it fills about half its convex hull, and 65% of its box overstates it."""
    rod = trimesh.creation.capsule(height=0.15, radius=0.015)
    rod.vertices[:, 0] += 16.0 * rod.vertices[:, 2] ** 2
    tm = seamed(rod)
    ext = obb_extent(tm)

    assert 0.5 < rod.volume / rod.convex_hull.volume < 0.6
    assert mass_prompt_volumes(tm, ext)[0] > 1.5 * rod.volume
    assert mass_prompt_volumes(tm, ext, weld_seams=True) == (pytest.approx(rod.volume), None)


def test_single_missing_triangle_is_filled():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    holed = trimesh.Trimesh(vertices=sphere.vertices, faces=sphere.faces[1:])

    assert welded_volume(seamed(holed)) == pytest.approx(sphere.volume, rel=1e-3)


def test_real_hole_stays_open():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    keep = ~np.isin(sphere.faces, [0, 1]).any(axis=1)  # every face around two neighbouring vertices
    tm = seamed(trimesh.Trimesh(vertices=sphere.vertices, faces=sphere.faces[keep]))
    ext = obb_extent(tm)

    assert welded_volume(tm) is None
    assert mass_prompt_volumes(tm, ext, weld_seams=True) == mass_prompt_volumes(tm, ext)
