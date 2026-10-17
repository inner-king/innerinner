# Patch Provenance

SimFoundry distributes eleven patch files under [`patches/`](patches/). Each is a
unified diff that an installation script applies to a third-party project after it
is fetched into the ignored local `deps/` directory.

**These patch files are the one place where SimFoundry does redistribute
third-party source code.** A unified diff carries upstream code in its context
lines (` `) and removed lines (`-`), so each patch reproduces a fragment of the
project it targets. That code remains under its upstream license — it is **not**
covered by SimFoundry's Apache 2.0 license, and a patch is not Apache-2.0 merely
because NVIDIA authored the diff.

Three of the eleven carry fragments of source that is **not open source**. Delivered only for academic/research purposes:

| Patch | Upstream license of the carried source |
|---|---|
| `Any6D.patch` | Custom **non-commercial / academic-only** |
| `FoundationPose.patch` | **NVIDIA Source Code License** (non-commercial) |
| `Hunyuan3D-2.1.patch` | **Tencent Hunyuan 3D 2.1 Community License** (non-OSS) |

The full upstream license texts for the fragments carried by `Any6D.patch` and
`Hunyuan3D-2.1.patch` are preserved verbatim under
[`third_party_notices/`](third_party_notices/) — see the per-patch detail below.

## Summary

All base commits below are the exact revisions the installation scripts pin, so
each patch is recorded against the source it is actually applied to.

| Patch | Upstream | Base commit | Upstream license | Owner | Applied by | `git apply --check` |
|---|---|---|---|---|---|---|
| `3dgrut.patch` | [nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut) | `a37ef721012dea0f29c0fcfff2d525023b4e854a` | Apache-2.0 | SimFoundry team | `install_3dgrut.sh` | ✅ clean |
| `3dgrut_nounset.patch` | [nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut) | `a37ef721012dea0f29c0fcfff2d525023b4e854a` | Apache-2.0 | SimFoundry team | `install_3dgrut.sh` | ✅ clean |
| `BEHAVIOR-1K-link-physics-materials.patch` | [StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) | `d89aae4e0e9a1de3cf8285cb9669c11d8c8bb864` | MIT | this fork | `install_simfoundry.sh` (`git_safe_checkout_patched`) | ✅ clean |
| `Any6D.patch` | [taeyeopl/Any6D](https://github.com/taeyeopl/Any6D) | `80eb4866a1c96ecb18be18836aba4f4bd6e80e9e` | **Non-commercial / academic-only** | SimFoundry team | `install_any6d.sh` | ✅ clean |
| `FoundationPose.patch` | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) | `e3d597b8c6b851d053094ebd6fa240191c5238f8` | **NVIDIA Source Code License (non-commercial)** | SimFoundry team | `install_simfoundry.sh` | ✅ clean |
| `Hunyuan3D-2.1.patch` | [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | `82920d643c0dc2f7bfd7255f45f62d386edfe60c` | **Tencent Hunyuan 3D 2.1 Community License (non-OSS)** | SimFoundry team | `install_hunyuan.sh` | ✅ clean |
| `ml-depth-pro.patch` | [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro) | `9efe5c1def37a26c5367a71df664b18e1306c708` | Apple Sample Code License | SimFoundry team | `install_simfoundry.sh` | ✅ clean |
| `PriorDepthAnything.patch` | [SpatialVision/Prior-Depth-Anything](https://github.com/SpatialVision/Prior-Depth-Anything) | `8c029cbca669443fe0bbf8dcefb5f91ad531084d` | Apache-2.0 | SimFoundry team | `install_simfoundry.sh` | ✅ clean |
| `splatfacto_depth_loss.patch` | [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio) | `6b60855003011b2ca23c2fe3f8e2ca6314c69924` (tag `v1.1.5`) | Apache-2.0 | SimFoundry team | `install_nerfstudio.sh` | ✅ clean |
| `void-model.patch` | [netflix/void-model](https://github.com/netflix/void-model) | `e3914f8f551dd4b880661991fd6b28cd1699a97a` | Apache-2.0 | SimFoundry team | `install_void.sh` | ✅ clean |
| `void-model-low-vram.patch` | [netflix/void-model](https://github.com/netflix/void-model) | `e3914f8f551dd4b880661991fd6b28cd1699a97a` | Apache-2.0 | this fork | `install_void.sh`, after `void-model.patch` | ✅ clean |

Validation was performed on 2026-07-27 by fetching each upstream at the base commit
above and running:

```bash
git -C <upstream-checkout> apply --check patches/<patch>
```

All patches applied cleanly. Eight were validated on 2026-07-27 (nine including
`lerobot.patch`, which has since been removed); `3dgrut_nounset.patch` was added
later and validated the same way on 2026-08-13 against the same base commit.
`void-model-low-vram.patch` was added in this fork and validated on 2026-10-16 against
the same base commit with `void-model.patch` applied first, the order `install_void.sh` uses.
`BEHAVIOR-1K-link-physics-materials.patch` was added in this fork and validated on 2026-10-17
against its base commit.
Re-run this whenever a base commit is bumped.

## Upstream source carried per patch

"Upstream lines" counts context plus removed lines — the upstream code physically
present in the patch file. "Added" lines are NVIDIA-authored.

| Patch | Context | Removed | Upstream lines | Added (NVIDIA) | Files touched |
|---|---|---|---|---|---|
| `Any6D.patch` | 112 | 96 | **208** | 130 | 4 |
| `FoundationPose.patch` | 43 | 68 | **111** | 68 | 3 |
| `PriorDepthAnything.patch` | 32 | 27 | **59** | 28 | 5 |
| `void-model-low-vram.patch` | 24 | 4 | **28** | 16 (this fork) | 2 |
| `Hunyuan3D-2.1.patch` | 22 | 3 | **25** | 10 | 3 |
| `splatfacto_depth_loss.patch` | 13 | 0 | **13** | 46 | 1 |
| `ml-depth-pro.patch` | 9 | 2 | **11** | 2 | 1 |
| `3dgrut.patch` | 6 | 1 | **7** | 4 | 1 |
| `BEHAVIOR-1K-link-physics-materials.patch` | 6 | 1 | **7** | 8 (this fork) | 1 |
| `void-model.patch` | 6 | 0 | **6** | 13 | 1 |
| `3dgrut_nounset.patch` | 6 | 0 | **6** | 5 | 1 |

## Per-patch detail

### `3dgrut.patch`
- **Target**: `threedgrut/export/scripts/ply_to_usd.py`
- **Purpose**: Sets `export_cameras=False` for the PLY-only export path, which has no
  camera dataset. A background Gaussian-splat volume for OmniGibson/NuRec needs only
  geometry; without this the PLY→USDZ step fails with
  `ValueError: export_cameras=True requires a dataset`.
- **Upstream NOTICE**: none at base commit.

### `3dgrut_nounset.patch`
- **Target**: `scripts/create_conda.sh`
- **Purpose**: Disables bash `nounset` (`set +u`) before the conda operations in
  3DGRUT's environment-creation script. The script's `conda()` wrapper reactivates
  the env after every `conda install`, sourcing deactivate.d hooks that can
  reference unset `CONDA_BACKUP_*` variables and abort the script under `set -u`.
- **Upstream NOTICE**: none at base commit.

### `BEHAVIOR-1K-link-physics-materials.patch`
- **Target**: `OmniGibson/omnigibson/objects/object_base.py`
- **Purpose**: Added by this fork. `BaseObject.__init__` keeps the caller's
  `link_physics_materials` dict, and `_post_load` replaces each of its entries with the
  `PhysicsMaterial` it creates and binds. That dict is also the object's init info
  (`save_init_info` records the argument as given), and for a scene loaded from JSON it is the
  JSON's own dict. So an object built with the kwarg, which is how a saved scene carries a
  friction (`objects_info.init_info[<name>].args.link_physics_materials`), loads with the
  friction bound, but every later save raises
  `TypeError: Object of type PhysicsMaterial is not JSON serializable`: `og.sim.save()`,
  `scene.save()`, and the `HDF5CollectionWrapper` that teleop and policy evaluation record
  through. The patch copies the dict down to each link's kwargs, so the materials it creates go
  into the object's own copy. Nothing else in OmniGibson reads the attribute, and `ManipulationRobot`'s
  finger frictions are added to the copy as before. Upstream `main` still has the same code,
  moved into `usd_object.py`.
- **Applied by**: `git_safe_checkout_patched` (`scripts/installation/git_safe.sh`), which takes the
  patch out before moving `deps/BEHAVIOR-1K` to a new pin and puts it back after. An applied patch
  leaves the tree modified, which the plain pinned checkout reads as local work and would never
  move off.
- **Upstream license text**: MIT; the text is reproduced in
  [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) §7, where it already covers the OmniGibson
  code vendored in `simfoundry/utils/asset_conversion_utils.py`.
- **Upstream NOTICE**: none at base commit.

### `Any6D.patch`
- **Targets**: `estimater.py`, `foundationpose/Utils.py`,
  `foundationpose/bundlesdf/mycuda/common.cu`, `requirements.txt`
- **Purpose**: Rotation-grid tuning for pose hypotheses, `trimesh.visual.material`
  import fix, CUDA kernel dtype dispatch, dependency bounds.
- **Note**: Any6D vendors its own copy of FoundationPose, so this patch modifies
  **FoundationPose source nested inside Any6D**. Two upstream licenses apply to one
  patch file — Any6D's academic-only terms and the NVIDIA Source Code License.
- **Upstream license text**: preserved verbatim at
  [`third_party_notices/Any6D-LICENSE.txt`](third_party_notices/Any6D-LICENSE.txt)
  (SHA-256 `3caf5f91…d297`), satisfying the license's condition that the copyright
  notice and permission notice accompany all copies or substantial portions of the
  Software.
- **Upstream NOTICE**: none at base commit.

### `FoundationPose.patch`
- **Targets**: `bundlesdf/mycuda/common.cu`, `bundlesdf/mycuda/setup.py`, `requirements.txt`
- **Purpose**: Wraps CUDA kernels in `AT_DISPATCH_FLOATING_TYPES` for dtype dispatch,
  adjusts the build setup, and bounds dependencies.
- **Upstream NOTICE**: none at base commit.

### `Hunyuan3D-2.1.patch`
- **Targets**: `hy3dpaint/textureGenPipeline.py`, `hy3dpaint/utils/simplify_mesh_utils.py`,
  `hy3dshape/hy3dshape/pipelines.py`
- **Purpose**: Points the texture pipeline at the local `hunyuanpaintpbr` path and
  switches mesh simplification to quadric decimation with an explicit face-count target.
- **Upstream license text**: the Tencent Hunyuan 3D 2.1 Community License Agreement
  is preserved verbatim at
  [`third_party_notices/Hunyuan3D-2.1-LICENSE.txt`](third_party_notices/Hunyuan3D-2.1-LICENSE.txt)
  (SHA-256 `b79ac5e1…81c0c`), per §3(a) of the Agreement, which requires that a copy
  accompany any distribution of the Works. The Agreement's grant covers a Territory
  that **excludes the European Union, the United Kingdom, and South Korea** (§1(l)).
- **Upstream NOTICE**: `Notice.txt` (121 lines) exists at the base commit and **is
  preserved verbatim** at
  [`third_party_notices/Hunyuan3D-2.1-NOTICE.txt`](third_party_notices/Hunyuan3D-2.1-NOTICE.txt)
  (SHA-256 `ffccf6b5…0350`), referenced from
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### `ml-depth-pro.patch`
- **Target**: `pyproject.toml`
- **Purpose**: Sets the Python target to 3.10 (`pythonVersion`, `target-version`) to
  match the `simfoundry` environment.
- **Upstream NOTICE**: none at base commit.

### `PriorDepthAnything.patch`
- **Targets**: `prior_depth_anything/__init__.py`, `depth_completion.py`, `plugin.py`,
  `requirements.txt`, `setup.py`
- **Purpose**: Adds a `coarse_only` mode and `args` plumbing through the completion
  path, plus packaging adjustments.
- **Upstream NOTICE**: none at base commit.

### `splatfacto_depth_loss.patch`
- **Target**: `nerfstudio/models/splatfacto.py`
- **Purpose**: Adds env-var-gated L1 depth supervision. With `NERFSTUDIO_DEPTH_LOSS=1`,
  splatfacto reads per-frame depth maps from `NERFSTUDIO_DEPTH_DIR` and adds
  `NERFSTUDIO_DEPTH_LOSS_MULT * |pred_depth - gt_depth|` to the loss. Documented in
  [`auto_bg_reconstruction/README.md`](scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/README.md) §4.2.
- **Note**: This is the only patch carrying an explicit `Subject:`/`Target:` header.
- **Upstream NOTICE**: none at base commit.

### `void-model.patch`
- **Target**: `inference/cogvideox_fun/make_warped_noise.py`
- **Purpose**: Prevents the `rp` dependency from invoking `sudo pip` for its
  auto-installs, which would otherwise raise an interactive password prompt in the
  middle of an unattended VOID pass.
- **Upstream NOTICE**: none at base commit.

### `void-model-low-vram.patch`
- **Targets**: `inference/cogvideox_fun/predict_v2v.py` (pass 1),
  `inference/cogvideox_fun/inference_with_pass1_warped_noise.py` (pass 2)
- **Purpose**: Added by this fork to run VOID on a 12 GB GPU; active only with
  `VOID_LOW_VRAM=1`. Pass 1 decodes the VAE in tiles (the untiled decode of a
  160-frame chunk ran out of memory). Pass 2 keeps the transformer, VAE and T5 on the
  CPU and streams them layer by layer (`enable_sequential_cpu_offload`), with tiled VAE
  decoding, instead of moving whole models onto the GPU. Unset, both scripts behave as
  upstream.
- **Upstream NOTICE**: none at base commit.
