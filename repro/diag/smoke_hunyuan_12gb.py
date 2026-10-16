# SPDX-License-Identifier: Apache-2.0
"""Smoke test: can stage 7's Hunyuan3D-2.1 (shape + PBR texture) run on a 12 GB RTX 3060?"""
import os, sys, time
from pathlib import Path
import torch
from PIL import Image

REPO = str(Path(__file__).resolve().parents[2])  # repro/<dir>/<this file> -> repository root
sys.path.insert(0, REPO)
from simfoundry.models.mesh_generator import Hunyuan

img_in, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
Hunyuan.set_repo_path(f"{REPO}/deps/Hunyuan3D-2.1")

sys.path.insert(0, f"{REPO}/deps/Hunyuan3D-2.1/hy3dshape")
from hy3dshape.rembg import BackgroundRemover
cut = BackgroundRemover()(Image.open(img_in).convert("RGB"))
cut_path = f"{out_dir}/input_rgba.png"
cut.save(cut_path)

t0 = time.time()
gen = Hunyuan(low_vram=True)
print(f"init {time.time() - t0:.0f}s, alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
gen.generate_shape(f"{out_dir}/shape.glb", image_path=cut_path)
print(f"shape {time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
gen.generate_texture(f"{out_dir}/shape.glb", f"{out_dir}/textured.glb", image_path=cut_path)
print(f"texture {time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
