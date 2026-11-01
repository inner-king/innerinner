# Auto-Background Reconstruction

The auto-background flow adds a 3D Gaussian Splat background to an existing A reconstruction. It is useful when you want the final OmniGibson scene to include both reconstructed foreground objects and a realistic static background.

This is an optional add-on. Run the normal A reconstruction first.

## Quick Start

Step 1: run A reconstruction in splat-prep mode so foreground objects and background training share the same DA3 frame set.

```bash
OMNIGIBSON_HEADLESS=1 \
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name <scene> \
  --video-fpath /path/to/video.mov \
  --no-stream \
  -- s1_video.splat_prep=true \
     s1_video.n_subsampled_frames=400 \
     s1_video.target_w=672 \
     s1_video.target_h=384 \
     s5_scene.pda_geometric_backend=depth_pro \
     s11_sim.vlm_model=gemini-2.5-pro
```

Step 2: run the background add-on.

```bash
bash scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/run_auto_align.sh \
  <scene> /path/to/video.mov \
  --floor-category "desk, table, or counter"
```

The wrapper checks that the required A outputs exist and prints the expected Step 1 command if they do not.

## What It Runs

| Step | Script | Purpose | Main output |
|---|---|---|---|
| 1 | `1_generate_quadmask_for_void.py` | Build masks for foreground object removal. | `auto_bg/void/input/`, mask debug images |
| 2 | `2_run_void_pass1.py` | First VOID inpainting pass. | `auto_bg/void/pass1/` |
| 3 | `3_run_void_pass2.py` | Second VOID pass and cleaned frames. | `auto_bg/void/pass2/cleaned_frames/` |
| 4 | `4_build_seed_ply_from_void_da3.py` | Run DA3 on cleaned frames and build a seed point cloud. | `auto_bg/seed.ply` |
| 5 | `5_train_bg_splat.py` | Train a depth-supervised splat. | `auto_bg/splat/export/<scene>_bg.ply` |
| 6 | `6_bridge_bg_splat_to_og.py` | Align the splat to the OG world. | `<scene>_bg.ply.pose.json` |
| 7 | `7_build_og_scene_assets.py` | Build a loadable scene asset directory. | `assets/scenes/<scene>/` |

## Inputs

Required from the splat-prep A run:

- `Data/<scene>/s1_video/frames_subsampled_400/`
- `Data/<scene>/s1_video/input_video.mp4`
- `Data/<scene>/s2_da/da/exports/npz/results.npz`
- `Data/<scene>/s4_frame/image_0_cam2world.npy`
- `Data/<scene>/s14_og/reconstructed_og_scene.json`

Do not mix frames from one run with DA3 output from another run; the metric scale can differ.

## Outputs

```text
Data/<scene>/auto_bg/
  void/
  clean_frames -> void/pass2/cleaned_frames
  da3/void/da/exports/npz/results.npz
  seed.ply
  splat/export/<scene>_bg.ply
  splat/export/<scene>_bg.ply.pose.json

assets/scenes/<scene>/
  objects/
  <scene>_scene_state_auto_bg.json
```

## Setup Notes

Auto-background needs the standard SimFoundry setup plus:

- `void` environment for video inpainting
- `3dgrut` environment for PLY to USDZ conversion
- `nerfstudio_simfoundry` environment for splat training
- CUDA 12.x toolchain and a compatible host compiler for `gsplat`

Recommended pinned `nerfstudio_simfoundry` versions:

- Python 3.10
- `torch==2.1.2`
- `torchvision==0.16.2`
- `gsplat==1.4.0`
- `nerfstudio==1.1.5`
- `hydra-core>=1.3,<1.4` (required by the canonical Stage 2c entrypoint)

Create it manually:

```bash
mamba create -n nerfstudio_simfoundry python=3.10 -y
mamba run -n nerfstudio_simfoundry pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
mamba run -n nerfstudio_simfoundry pip install gsplat==1.4.0 nerfstudio==1.1.5 "hydra-core>=1.3,<1.4"
```

The depth-loss patch in `patches/splatfacto_depth_loss.patch` must be applied to the installed `nerfstudio` package if you use depth-supervised training.

## Practical Defaults

The defaults are tuned for a 24 GiB GPU:

- 400 frames
- 672x384 splat-prep frames
- DA3 resolution 448
- `splatfacto-big`
- 80k splat iterations
- depth loss enabled

Typical Step 2 wall time on a 24 GiB GPU is around two hours, plus the canonical A reconstruction.

## Cross-capture alignment (`bridge_mode=appearance`)

Set `s6_bridge.bridge_mode=appearance` to align a background splat to a scene that was captured separately, for example a phone video of the background and a LiDAR capture of the kitchen. It is off by default: `auto` never chooses it, and the default run is unchanged. The `single` and `umeyama` modes chain camera centres within one DA3 run, so they cannot join two captures that share no frame; this mode aligns the two point clouds instead, by what the counter top looks like. The alignment takes four steps. The largest plane of the seed cloud (`s4_seed_ply.out_ply`) is taken as the counter, with its normal pointing to the cameras, and moved to z = 0. Within 3 cm of that plane, a top-down brightness map is built for both clouds: the scene's background mesh with its texture, and every DA3 pixel colour of every frame. For each yaw in 1-degree steps, an FFT correlation gives a rough shift, which is then searched on a 1 cm grid and scored by a masked normalised cross-correlation (NCC) over the cells both clouds observe; the overlap must cover at least half of the scene's cells. The best yaw and shift are refined on a 0.5 cm grid in 0.1-degree steps. The scale stays at 1; a scale check is written to the record and never applied. The step writes `<bg_ply>.align.json` in every case: the plane, the yaw curve and the best peaks, the second peak and the gap, the overlap, the scale check, the height residual and the reasons for a refusal. It writes the pose sidecar `<bg_ply>.pose.json` (with an `alignment_record` key) only when the alignment is accepted; a refusal exits non-zero and leaves any earlier sidecar where it is. The refusal reasons are `no_gravity_plane`, `no_overlap`, `low_ncc`, and `ambiguous` (the best peak and the second independent peak are closer than `min_peak_gap`). The settings are under `s6_bridge.appearance`. The path keys `seed_ply`, `scene_mesh` and `scene_texture` default to the seed cloud, `Data/<target_scene>/s_bg_mesh/background_mesh.glb`, and the texture beside it.

Limits, as tested:

- (i) Only one scene and one background video have been tried: the LiDAR kitchen and the Background.mp4 DA3 run of 71 frames. On that pair the reproduction gives yaw 22.3 degrees, NCC 0.944 and overlap 0.83. It agrees with the research prototype's transform to 0.97 cm and 0.53 degrees; the prototype's yaw was 22.9 degrees.
- (ii) The counter top must show a coloured edge or pattern. Plain tops, other lighting, and many objects moved between the two captures have not been tested.
- (iii) The scale is fixed at 1, which assumes metric DA3 depth; the scale check is recorded only.
- (iv) The refusal thresholds `min_ncc`, `min_peak_gap` and `min_plane_share` are provisional. On this scene the yaw curve runs from 0.67 at its lowest to 0.944 at its best. The second independent peak is 0.851 at 202 degrees, so the gap is 0.093 (0.086 with a 0.5 cm coarse grid). The NCC floor only removes poor alignments; the gap check is what catches a near-symmetric scene. The peak window `peak_window_deg` (10 degrees) is provisional too.
- (v) The largest plane is taken as the counter. A scene where a wall spans more of the view than the counter has not been tested.
- (vi) This one real-scene check is a CPU reproduction. The induction-centre check the research run used carries the LiDAR's glass error (the LiDAR reads the glass about 2 cm low), so its residual cannot be separated from the alignment error. The median height difference between the two clouds over the matched cells is 6.2 mm; it is recorded only.

## Troubleshooting

- Blurry splat: confirm `splatfacto-big` and camera optimizer `SO3xR3` are enabled.
- Floaters above a flat surface: confirm depth loss is enabled and try increasing `depth_loss_mult`.
- Ghost objects remain after VOID: increase guidance scale or choose a more accurate `--floor-category`.
- DA3 scale drift: refilm with smoother camera motion or increase frame overlap.
- `gsplat` compile errors: verify CUDA 12.x, a compatible compiler, and no unsupported old GPU architecture in `TORCH_CUDA_ARCH_LIST`.
