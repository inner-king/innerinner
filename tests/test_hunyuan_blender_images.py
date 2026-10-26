# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest

from simfoundry.models.mesh_generator import Hunyuan, clear_blender_images


class _Image:
    """Stand-in for a bpy image datablock. Equality is identity, as it is for datablocks."""

    def __init__(self, name):
        self.name = name


@pytest.fixture
def fake_bpy(monkeypatch):
    bpy = types.ModuleType("bpy")
    bpy.data = types.SimpleNamespace(images=[])
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    return bpy


def _load_images(bpy, count):
    """What the OBJ import does: one image datablock per texture file it reads."""
    bpy.data.images.extend(_Image(f"textured_mesh_{i}.jpg") for i in range(count))


class _Pipeline:
    """Stands in for the paint pipeline: loads images the way the OBJ import does, writes the GLB, returns the OBJ path."""

    def __init__(self, folder, bpy, images_to_load=0, error=None):
        self.folder = folder
        self.bpy = bpy
        self.images_to_load = images_to_load
        self.error = error
        self.images_at_call = None
        self.glb_bytes = b"glb written by the paint pipeline"

    def __call__(self, shape_fpath, image_path=None, use_remesh=True):
        self.images_at_call = len(self.bpy.data.images)
        _load_images(self.bpy, self.images_to_load)
        if self.error is not None:
            raise self.error
        (self.folder / "textured_mesh.glb").write_bytes(self.glb_bytes)
        return str(self.folder / "textured_mesh.obj")


def _hunyuan(pipeline):
    generator = Hunyuan.__new__(Hunyuan)  # skip __init__: it would load the models
    generator.texture_pipeline = pipeline
    generator.low_vram = False
    return generator


def _generate(tmp_path, pipeline):
    _hunyuan(pipeline).generate_texture(str(tmp_path / "shape.glb"), str(tmp_path / "object.glb"))


def test_clear_removes_every_image_and_counts_them(fake_bpy):
    # Five, not two: removing from the list being iterated skips every other image.
    _load_images(fake_bpy, 5)
    assert clear_blender_images() == 5
    assert fake_bpy.data.images == []


class _ForbidBpy:
    """Meta-path finder that fails the test if anything tries to import bpy."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if name == "bpy":
            raise AssertionError("clear_blender_images imported bpy")
        return None


def test_without_blender_loaded_nothing_is_cleared_and_blender_is_not_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "bpy", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_ForbidBpy] + sys.meta_path)
    assert clear_blender_images() == 0
    assert "bpy" not in sys.modules


def test_images_left_from_an_earlier_object_are_gone_before_the_paint_pipeline_runs(tmp_path, fake_bpy):
    _load_images(fake_bpy, 3)
    pipeline = _Pipeline(tmp_path, fake_bpy)
    _generate(tmp_path, pipeline)
    assert pipeline.images_at_call == 0


def test_images_the_paint_pipeline_loads_are_gone_once_generate_texture_returns(tmp_path, fake_bpy):
    pipeline = _Pipeline(tmp_path, fake_bpy, images_to_load=4)
    _generate(tmp_path, pipeline)
    assert fake_bpy.data.images == []


def test_images_are_cleared_and_the_error_propagates_when_the_paint_pipeline_fails(tmp_path, fake_bpy):
    pipeline = _Pipeline(tmp_path, fake_bpy, images_to_load=4, error=RuntimeError("paint failed"))
    with pytest.raises(RuntimeError, match="paint failed"):
        _generate(tmp_path, pipeline)
    assert fake_bpy.data.images == []


def test_the_glb_is_still_copied_to_out_fpath(tmp_path, fake_bpy):
    pipeline = _Pipeline(tmp_path, fake_bpy)
    _generate(tmp_path, pipeline)
    assert (tmp_path / "object.glb").read_bytes() == pipeline.glb_bytes
