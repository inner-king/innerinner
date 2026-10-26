# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that the configured seed reaches Hunyuan's shape pipeline as a CPU torch.Generator.

Needs no GPU and no model weights: Hunyuan.__new__ skips the constructor that loads the
pipelines, and fake pipelines stand in for them.
"""

import inspect

import torch
from omegaconf import OmegaConf

from simfoundry.models.mesh_generator import (
    RESERVED_GENERATION_KWARGS,
    Hunyuan,
    filter_generation_kwargs,
)
from simfoundry.pipeline.stage_utils import resolve_generation_kwargs


class _FakeMesh:
    def export(self, path):
        with open(path, "wb") as f:
            f.write(b"glb")

    def show(self):
        pass


class _FakeShapePipeline:
    """Records each call's kwargs and the first draw from the generator it was handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        generator = kwargs.get("generator")
        draw = None if generator is None else torch.randn(5, generator=generator)
        self.calls.append({"kwargs": kwargs, "draw": draw})
        return [_FakeMesh()]


class _FakeTexturePipeline:
    """Writes the textured .glb that generate_texture copies, and records what it was called with."""

    def __init__(self, glb_path, obj_path):
        self.glb_path = glb_path
        self.obj_path = obj_path
        self.received_args = None
        self.received_kwargs = None

    def __call__(self, *args, **kwargs):
        self.received_args = args
        self.received_kwargs = kwargs
        with open(self.glb_path, "wb") as f:
            f.write(b"glb")
        return str(self.obj_path)


def _hunyuan(shape_pipeline, texture_pipeline=None):
    gen = Hunyuan.__new__(Hunyuan)
    gen.shape_pipeline = shape_pipeline
    gen.texture_pipeline = texture_pipeline
    gen.low_vram = False
    return gen


def test_seed_becomes_a_cpu_generator(tmp_path):
    shape = _FakeShapePipeline()
    _hunyuan(shape).generate_shape(out_fpath=str(tmp_path / "shape.glb"), seed=123)

    call = shape.calls[0]
    generator = call["kwargs"]["generator"]
    assert isinstance(generator, torch.Generator)
    assert generator.device.type == "cpu"
    assert generator.initial_seed() == 123
    expected = torch.randn(5, generator=torch.Generator().manual_seed(123))
    assert torch.equal(call["draw"], expected)


def test_seed_zero_is_honoured(tmp_path):
    # The configured default seed is 0, so a truthiness test would silently drop it.
    shape = _FakeShapePipeline()
    _hunyuan(shape).generate_shape(out_fpath=str(tmp_path / "shape.glb"), seed=0)

    generator = shape.calls[0]["kwargs"]["generator"]
    assert generator is not None
    assert generator.initial_seed() == 0


def test_no_seed_and_no_generator_leaves_the_pipeline_default(tmp_path):
    shape = _FakeShapePipeline()
    _hunyuan(shape).generate_shape(out_fpath=str(tmp_path / "shape.glb"))

    assert shape.calls[0]["kwargs"]["generator"] is None


def test_same_seed_repeats_and_other_seeds_differ(tmp_path):
    shape = _FakeShapePipeline()
    gen = _hunyuan(shape)
    for seed in (1, 1, 2):
        gen.generate_shape(out_fpath=str(tmp_path / "shape.glb"), seed=seed)

    first, second, third = (call["draw"] for call in shape.calls)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_explicit_generator_takes_precedence_over_seed(tmp_path):
    shape = _FakeShapePipeline()
    explicit = torch.Generator().manual_seed(9)
    _hunyuan(shape).generate_shape(out_fpath=str(tmp_path / "shape.glb"), generator=explicit, seed=123)

    assert shape.calls[0]["kwargs"]["generator"] is explicit


def test_generate_mesh_forwards_seed_to_shape_only(tmp_path, capsys):
    shape = _FakeShapePipeline()
    texture = _FakeTexturePipeline(tmp_path / "textured_mesh.glb", tmp_path / "textured_mesh.obj")
    gen = _hunyuan(shape, texture)
    rng_before = torch.get_rng_state()

    gen.generate_mesh(
        out_fpath=str(tmp_path / "out.glb"),
        shape_image_path="x.png",
        texture_image_path="x.png",
        seed=5,
    )

    assert shape.calls[0]["kwargs"]["generator"].initial_seed() == 5
    assert "seed" not in texture.received_kwargs
    assert "seed" not in texture.received_args
    assert "dropped from generation_kwargs" not in capsys.readouterr().out
    assert (tmp_path / "out.glb").exists()
    # Seeding goes through the generator only; the process-wide RNG is left alone.
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_unsupported_option_is_still_dropped_with_a_warning(tmp_path, capsys):
    shape = _FakeShapePipeline()
    texture = _FakeTexturePipeline(tmp_path / "textured_mesh.glb", tmp_path / "textured_mesh.obj")

    _hunyuan(shape, texture).generate_mesh(
        out_fpath=str(tmp_path / "out.glb"),
        shape_image_path="x.png",
        texture_image_path="x.png",
        resolution=1,
    )

    out = capsys.readouterr().out
    assert "dropped from generation_kwargs" in out
    assert "resolution" in out
    assert (tmp_path / "out.glb").exists()


def test_configured_seed_reaches_shape_only_through_the_stage_resolver():
    cfg = OmegaConf.create({"seed": 7, "s7_mesh": {}})
    kwargs = resolve_generation_kwargs(cfg, cfg.s7_mesh)

    shape_kwargs = filter_generation_kwargs(
        Hunyuan.generate_shape, kwargs, reserved=RESERVED_GENERATION_KWARGS["generate_shape"]
    )
    texture_kwargs = filter_generation_kwargs(
        Hunyuan.generate_texture, kwargs, reserved=RESERVED_GENERATION_KWARGS["generate_texture"]
    )

    assert shape_kwargs == {"seed": 7}
    assert texture_kwargs == {}


def test_seed_is_appended_last_so_positional_callers_keep_their_meaning():
    params = list(inspect.signature(Hunyuan.generate_shape).parameters)

    assert params[-2:] == ["mask", "seed"]


def test_generate_texture_takes_no_seed():
    assert "seed" not in inspect.signature(Hunyuan.generate_texture).parameters
