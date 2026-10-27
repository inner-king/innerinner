# Pipeline Reference

SimFoundry is organized into three runnable sub-pipelines:

```bash
scripts/pipeline/A_reconstruction/run.sh
scripts/pipeline/B_augmentation/run.sh
scripts/pipeline/C_application/run.sh
```

The dispatcher forwards to the same scripts:

```bash
scripts/pipeline/run.sh A_reconstruction ...
scripts/pipeline/run.sh B_augmentation ...
scripts/pipeline/run.sh C_application ...
```

All runners accept `--scene-name`, `--root-dir`, `--include`, `--exclude`, `--dry-run`, and Hydra-style overrides after `--`.

## A Reconstruction

Builds an OmniGibson scene from video or ZED stereo capture.

### Example

```bash
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name pull_scene_2 \
  --video-fpath /path/to/video.mov
```

Example captures are in `docs/assets/example_videos/` (repo root), e.g.
`--video-fpath docs/assets/example_videos/Fruits.mp4`. Shooting your own? See
[Capturing an input video](#capturing-an-input-video).

Useful options:

- `--pipeline video|stereo|zed`: input mode; default is `video`.
- `--stream / --no-stream`: stream stages 5-8 together or run them one at a time.
- `--max-vram-frac F`: VRAM budget for streamed stages as a fraction of total GPU memory. Default `0.9`, so the same setting works across card sizes.
- `--max-vram-gb N`: opt-in absolute hard budget in GiB, overriding the fraction. Leave unset unless you need to pin it — with `hard_vram_cap` the budget counts *total* GPU usage, so a value too small for the card stalls stages.
- `--detect-articulation`: run stage 9 for automatic articulated-object generation. Requires the optional `articulate` environments.
- Stage 9 has two optional interactive checkpoints, enabled as Hydra overrides (after `--`):
  - `s9_articulate_objects.merge_interactive_correction=true` opens a browser UI to review and
    correct the mesh segmentation before joints are estimated.
  - `s9_articulate_objects.interactive_joint_refinement=true` opens a browser UI after
    articulation to refine each joint — type, axis, pivot, limits, and physics (per-joint
    damping/friction, per-part mass and surface friction) — with a live 3D motion preview.
    Saves rewrite `results/mobility.urdf` (the original is backed up) and user physics edits
    land in `physics_overrides.json`.

  Both UIs open in a browser tab and block the pipeline until you finish (or cancel), so use
  them for attended runs only. 
- `--env-nerfstudio NAME`: select the Nerfstudio environment used by stage 2c. Default: `nerfstudio_simfoundry`.
- `--env-b1k NAME`: env for the OmniGibson stages. Default: `simfoundry`; pass this only if OmniGibson lives in a separate env.

### Capturing an input video

Keep these in mind when taking an input video:

1. **One flat surface.** Place every object on the same flat surface (a desk, a counter).
   Keep other surfaces — a second table, shelves — out of frame: stage 3 fits a single
   support plane.
2. **Angle down at the surface.** Shoot at an oblique downward angle — not edge-on from
   the side, not straight top-down — so both the surface and the sides of the objects
   are visible.
3. **Translate, don't rotate.** Sweep the camera sideways around the scene. Panning in
   place gives the reconstruction no parallax.
4. **Keep every object visible in every frame.** An object that leaves the frame can come
   out missing or mis-posed.
5. **Close all articulated objects.** Make sure all objects with joints (trash cans, cabinets) are closed.

> See [`example_videos/`](../../docs/assets/example_videos)

<img src="../../docs/assets/capture_lateral_sweep.png" alt="Start, middle, and end frames of an example capture: one table, camera angled down, sweeping laterally" width="100%">

*Start, middle, and end of `docs/assets/example_videos/Fruits.mp4`: the
camera stays angled down at one table, translates laterally, and keeps every object in
frame throughout.*

### Stages

| ID | Script | Env | Purpose | Key outputs |
|---|---|---|---|---|
| `1a` | `A_reconstruction/stages/1a_take_stereo_images.py` | `simfoundry` | Capture ZED stereo images. | `s1_zed/` |
| `1b` | `A_reconstruction/stages/1b_process_raw_video.py` | `simfoundry` | Convert and sample a video. | `s1_video/frames_*`, `input_video.mp4` |
| `2c` | `A_reconstruction/stages/2c_train_bg_splat.py` | `nerfstudio_simfoundry` (`--env-nerfstudio`) | Train and export the background Gaussian splat. | `s2c_gs/` |
| `2` | `A_reconstruction/stages/2_run_depth.py` | `da3` or `simfoundry` | Run the selected depth backend. | `s2_da/` or `s2_fs/` |
| `3` | `A_reconstruction/stages/3_segment_ground_plane.py` | `simfoundry` | Pick the canonical frame and find the support plane. | `s3_ground/` |
| `4` | `A_reconstruction/stages/4_unify_world_frame.py` | `simfoundry` | Align the scene to a stable world frame. | `s4_frame/` |
| `5` | `A_reconstruction/stages/5_decompose_scene.py` | `simfoundry` | Detect objects and create object-removal crops. | `s5_scene/` |
| `5c` | `A_reconstruction/stages/5c_depth_mesh_background.py` | `simfoundry` | Optional (`--bg-mesh`): textured mesh of the surroundings from the canonical frame's depth and stage 5's clean plate. | `s_bg_mesh/` |
| `6` | `A_reconstruction/stages/6_upsample_object_images.py` | `simfoundry` | Create cleaner object images for mesh generation. | `s6_upsample/` |
| `7` | `A_reconstruction/stages/7_generate_object_meshes.py` | `hunyuan` (`--env-mesh`) | Generate 3D meshes. | `s7_mesh/` |
| `8` | `A_reconstruction/stages/8_match_object_poses.py` | `simfoundry` | Estimate object poses. | `s8_pose/` |
| `9` | `A_reconstruction/stages/9_articulate_objects.py` | `simfoundry` | Optional automatic articulation (`--detect-articulation`). Also the source of articulated-object physics: the workflow estimates per-part mass/surface friction and per-joint damping/friction (`results/physics_properties.json` + URDF `<dynamics>`), which stage 11 consumes instead of estimating its own; user edits from the refinement UI (`physics_overrides.json`) take precedence. | `s9_articulate_objects/` |
| `10` | `A_reconstruction/stages/10_compile_scene.py` | `simfoundry` | Compile object metadata. | `s10_compile/` |
| `11` | `A_reconstruction/stages/11_make_objects_sim_ready.py` | `simfoundry` | Build sim-ready URDF/collision assets. | `s11_sim/` |
| `12` | `A_reconstruction/stages/12_stabilize_physics.py` | `simfoundry` | Settle objects in physics. | `s12_physics/` |
| `13` | `A_reconstruction/stages/13_import_usd.py` | `simfoundry` | Import assets into USD datasets. | dataset USD assets |
| `14` | `A_reconstruction/stages/14_create_og_scene.py` | `simfoundry` | Create the final OG scene JSON and preview. | `s14_og/reconstructed_og_scene.json`, `reconstructed_scene.png` |

### Mesh Generators

Stage 7 supports three image-to-3D backends, selected with `s7_mesh.shape_model` and
`s7_mesh.texture_model` (set both to the same value):

- **Hunyuan3D-2.1** (`hunyuan`, the default). Runs in the `hunyuan` env built by
  `install_everything.sh` and needs the RealESRGAN checkpoint from
  `download_checkpoints.sh`. On cards below ~29 GB VRAM, set `s7_mesh.low_vram=true`
  (CPU offloading, ~6 GB). This is the default `run.sh` invocation — nothing to pass.

- **TRELLIS.2** (`trellis2`). Opt-in: install it with
  `bash scripts/installation/install_trellis.sh` (or `install_simfoundry.sh --trellis`),
  which adds it to the `simfoundry` env — so stage 7 must also be pointed at that env.
  Model weights (`microsoft/TRELLIS.2-4B`) download automatically on the first run.

  ```bash
  bash scripts/pipeline/A_reconstruction/run.sh --scene-name <scene> --video-fpath <video> \
    --env-mesh simfoundry \
    -- s7_mesh.shape_model=trellis2 s7_mesh.texture_model=trellis2
  ```

- **Pixal3D** (`pixal3d`). Opt-in, pixel-aligned generation built on the TRELLIS.2
  backbone. It needs its own env (its pins conflict with the shared `simfoundry` env):
  build the TRELLIS.2 stack into a dedicated env first, then install Pixal3D on top —

  ```bash
  bash scripts/installation/install_trellis.sh --env-name pixal3d
  bash scripts/installation/install_pixal3d.sh --env-name pixal3d
  ```
  
  Then run stage 7 with:

  ```bash
  bash scripts/pipeline/A_reconstruction/run.sh --scene-name <scene> --video-fpath <video> \
    --env-mesh pixal3d \
    -- s7_mesh.shape_model=pixal3d s7_mesh.texture_model=pixal3d
  ```

All three write to the same layout (`s7_mesh/textured_mesh/<backend>/iter_N_mesh.glb`), so
downstream stages need no other changes. To compare backends on one capture, run each into
a separate `--scene-name`.

### Canonical Frame Selection

Stages 3-14 reconstruct the scene from a *single* frame of the capture: stage 3 fits the
support plane in it, stage 4 makes its camera the world frame, and stage 5 crops every object
out of it for stages 6-8. A bad frame therefore caps the quality of everything downstream — a
blurry frame produces blurry meshes, and a frame shot from far away leaves small objects at too
few pixels to reconstruct.

By default `s3_ground.img_idx: auto`, so stage 3 scores the candidate frames and picks one.

`frame_selection.mode` controls the last step:

- `heuristic`: scoring only, no remote calls.
- `hybrid` (default): the heuristic short-lists `vlm_top_k` frames and a VLM makes the final
  call.
- `vlm`: the VLM chooses among every frame that passes the gates.

Pin a frame instead  with an integer:

```bash
bash scripts/pipeline/A_reconstruction/run.sh --scene-name pull_scene_2 \
  -- s3_ground.img_idx=4
```

The pinned index is written to `s3_ground/frame_selection.json` as well, so a stage that reads
`img_idx: auto` follows the same frame rather than a selection left by an earlier automatic run.

If every candidate is rejected, stage 3 fails with the per-frame reasons; loosen the gate it
names or pin a frame. The one exception is a run with a single candidate frame (a single-image
input, or `frame_selection.max_candidates: 1`): there is no other frame to prefer, so that frame
is kept even when its only failing gate is `max_clipped_frac`, with a warning. Its other gates
still fail the run. Changing the selection after a run invalidates the `image_<idx>_*`
artifacts stages 4-14 wrote for the previous frame, so rerun from stage 3.

### Background Mesh (optional stage 5c)

The pipeline reconstructs the objects of a scene and leaves its surroundings out: the support surface is a
plane and everything else is missing. `--bg-mesh` adds stage 5c, which builds a textured mesh of the surroundings
from what the run already has: the canonical frame's depth (stage 2), the world frame (stages 3-4) and the
clean plate stage 5 leaves behind (the frame repainted with every detected object removed).

```bash
bash scripts/pipeline/A_reconstruction/run.sh --scene-name <scene> --video-fpath <video> --bg-mesh
bash scripts/pipeline/A_reconstruction/run.sh --scene-name <scene> --bg-mesh --include 5c   # on a finished run
```

Every depth pixel becomes a vertex and neighbouring pixels are joined into triangles; the clean plate is the
texture. The depth under an object's removal mask is the object, so those pixels are moved onto the surface the
object stood on, read from the measured ring around it: a plane fitted to the ring when the ring is one flat
surface (`s_bg_mesh.mask_surface: local`), otherwise a horizontal plane at the ring's dominant height, and z = 0
only when the ring is too small to read. A triangle is judged by how much its edges stretch: one whose vertices
all lie on the support plane within `s_bg_mesh.plane_snap_tol_m` by `plane_max_edge_ratio`, any other by
`relief_max_edge_ratio`, so a depth jump (a bottle against the wall behind it) is dropped. Border pixels, and
connected pieces smaller than `min_component_triangles` triangles, are dropped too. The settings are in
`s_bg_mesh` of `scripts/cfg/real2sim_cfg.yaml`; all of them are provisional and none is fitted to more than one scene.

Output: `Data/<scene>/s_bg_mesh/background_mesh.glb`, `background_texture.png` and `stage_info.json` (counts,
settings, which stage-5 files were used). The GLB is Y-up, as glTF requires, with the support plane at y = 0.
The light editor's mesh import (`up axis: auto`) rotates a glTF by +90 degrees about X, which gives back the Z-up
world with the support plane at z = 0, so the background lands in the frame the objects are in. (Writing the
Z-up world unchanged would lie on its side after that import.) Registering the result as a room needs the
`.usd` and `.background.json` of `light_editor/background_io.py`, which this stage does not write.

Limits, from the method rather than from a measurement:

- One viewpoint, so a bottle is a flat cut-out and the space behind it is empty. The holes behind bottles and
  taps stay. Two options: (i) a thin back wall behind each relief layer, which hides the holes from other views
  but shows a false wall behind a bottle and collides in physics; (ii) leave the holes, which show nothing that
  is not there. This stage takes (ii).
- A glass cooktop lies lower than the counter, so the local surface fills it lower. Stage 14 places objects on
  z = 0, so a pot on that glass appears to float above the visible glass by about that height difference. This
  is only what is seen: the physics uses the floor plane, which is not visible here. With the candidate
  `s_bg_mesh.plane_snap_below_m: 0.03` (see *One-sided plane snap* below), the visible glass of the reference scene
  reads +0.14 cm against the counter instead of -0.83 cm, and the pot's support reads 0 instead of -2.23 cm.
- A ring that is not one surface is filled at its widest surface's height, so a ring that also holds a
  lower surface (a sink beside a counter) can leave a step on the other side of the mask edge.
- The measured counter of the reference scene sits +1.2 cm above z = 0 at its median, with a 5-95% spread of
  -1.9 to +2.0 cm. `plane_snap_tol_m` (1 cm) therefore puts only part of the counter on the plane; the rest is
  relief, and the front edge keeps its sawtooth and the curtain-like stretch of grazing surfaces.
- The 22 mm step between the counter and the cooktop glass is cut by the relief limit (`relief_max_edge_ratio` 4.0), so a low side view shows holes at the mask edge; measured on one scene (the rerun copy), where the glass box drops 502 triangles against 1050 for 894363c, and the provisional 2.0 removes the front-edge sawtooth but grows the largest dropped piece from 3423 to 10003 triangles, so the value is a trade-off, not a fix.
- The mesh is only as good as the depth.

#### One-sided plane snap (candidate settings, off by default)

`s_bg_mesh.plane_snap_below_m` and `s_bg_mesh.plane_snap_above_m` set the snap band below and above z = 0 separately.
A pixel is a plane pixel when `-below < z < above`, strict at both ends, after its object pixels have moved onto their
surface. A null side uses `plane_snap_tol_m`, so both null is the symmetric band of `plane_snap_tol_m` exactly (the
test compares the masks for every input, NaN and infinities included). Zero switches that side off. Each value must be
null or a finite number of metres, zero or more; anything else is an error that names the key. `stage_info.json` records
the limits used (`plane_snap_below_m_used`, `plane_snap_above_m_used`) and how many snapped vertices came from below and
from above (`vertices_snapped_from_below`, `vertices_snapped_from_above`). The defaults are null, so the output is the
same as before. 0.03 below and the 1 cm above are candidates for the reference scene, not defaults.

Evidence, all from one scene, one capture and one sensor. The table is from research 2's report
(`lidar_rerun_2026-10-26/hob_depth/report_ko.md`, sections 5.1 and 5.2, outside this repository), which measured it on a
mirror of the 5c height stage. The code below, run on the same inputs, gives the same region means as that mirror (the
one-sided row agrees with the raw values to float precision).

| Setting | Visible glass, mean (cm) | Glass within ±0.5 cm | Pot's support (cm) | Counter (cm) | Sink, as measured (cm) | New flat background pixels (cm²) / of which over +1 cm |
|---|---|---|---|---|---|---|
| symmetric ±1 cm (current) | -0.83 | 34% | -2.23 | -0.07 | -0.98 | 0 (0) / 0 (0) |
| symmetric 2 cm | -0.41 | 82% | -2.23 | -0.00 | -0.66 | 14398 (611) / 5758 (244) |
| symmetric 3 cm | -0.02 | 99% | 0.00 | -0.00 | -0.23 | 21471 (911) / 7731 (328) |
| one-sided -3 cm < z < +1 cm | 0.14 | 87% | 0.00 | -0.00 | -0.23 | 13740 (583) / 0 (0) |
| one-sided -4 cm < z < +1 cm | 0.16 | 88% | 0.00 | -0.00 | 0.00 | 14950 (635) / 0 (0) |
| one-sided -5 cm < z < +1 cm | 0.16 | 88% | 0.00 | -0.00 | 0.00 | 15053 (639) / 0 (0) |
| one-sided -3 cm / +2 cm | -0.02 | 99% | 0.00 | -0.00 | -0.23 | 19498 (828) / 5758 (244) |

Depressions (research 2, section 5.3): a square depression of depth D, 8 cm wide, with the edge blurred as this sensor
blurs it. The table gives the area in cm² of the depression's half-depth floor that survives each setting (nominal: the
depression's own half-depth area, before any setting).

| D (cm) | Nominal | Current | Symmetric 3 cm | One-sided -3 cm / +1 cm |
|---|---|---|---|---|
| 1 | 52.8 | 52.3 | 0.0 | 0.2 |
| 2 | 52.8 | 92.8 | 0.1 | 0.8 |
| 3 | 52.8 | 73.4 | 12.1 | 12.1 |
| 4 | 52.8 | 67.5 | 37.8 | 37.8 |
| 5 | 52.8 | 64.2 | 52.0 | 52.0 |
| 6 | 52.8 | 61.8 | 62.1 | 62.1 |
| 8 | 52.8 | 59.7 | 59.7 | 59.7 |
| 10 | 52.8 | 58.2 | 58.2 | 58.2 |

A depression keeps its half-depth floor from about D = 2T (T the band below, 6 cm for 3 cm); shallower ones are
flattened, so a 3 cm band removes a 2 cm depression and keeps a 6 cm one.

Cracks, measured with this code on the reference scene (the merge-preview depth, the same inputs as the rows above):
the final triangulation has 5 connected pieces with the current setting, with the one-sided setting, and with symmetric
3 cm and 2 cm; the largest detached piece has 14,443 triangles in each. Before the 200-triangle drop, the whole
triangulation has 7 pieces with the current and one-sided settings and 5 with symmetric 3 cm and 2 cm. The plane layer
alone splits into 20 pieces with the current setting (17 under 200 triangles), 4 one-sided (2 under 200), 7 symmetric
3 cm (5 under 200) and 5 symmetric 2 cm (2 under 200). Plane-layer triangles dropped for stretched edges rise from 1
(current) to 503 (one-sided) and 956 (symmetric 3 cm). An earlier run reported 25 cracked pieces with a 2 cm symmetric snap;
the counts above did not reproduce it, and how the 25 were counted is not recorded.

Generalisation: 3 cm and 1 cm come from one capture of one sensor. Real terrain rarely lies 3 cm under the support
plane (a sink lies deeper), and the sensor error is what reads in that range here, but that is an observation from one
scene, so both values are candidates until a second scene shows the same. DA3 (monocular) scenes were not checked.

Not known or not measured:

- The glass's real height. The research assumed it lies within ±0.5 cm of the counter.
- The front of the glass reads +1 to +1.8 cm above the counter (about 13% of the visible glass, 20 to 25% measured
  against the counter). The 1 cm upper band keeps these, on purpose: a higher upper band flattens raised structures
  (328 cm² at 3 cm), so this is a trade-off, not a fix.
- The cause of the offset: a weak lidar return or the depth completion of ARKit are guesses. The camera was fixed in
  the capture, so the dependence on viewpoint was not tested.
- The sink's real depth: this depth map cannot give it.

Tapered move (candidate, off by default): `s_bg_mesh.plane_snap_ramp_px` fades a one-sided snap's move in from the rim of
the snapped region over that many pixels, instead of moving every snapped vertex in full. At 0 (the default) the mesh and
the texture are byte-identical to the output before this setting existed. Why: the length filter in `grid_triangles` runs
before the snap, so a triangle that spans the rim keeps one corner moved and its neighbour in place, and then it is
stretched. On the LiDAR v2 scene with the 3 cm / 1 cm band, 465 triangles are stretched at 0, 1 at 4 and 1 at 6
(`plane_snap_ramp_px` 0, 4 and 6; one scene, one run of this code on the same inputs, CPU). Here 'stretched' means an edge
length over the pixel footprint of more than eight, as the stage's own stretched-edge test measures it, taken on the vertices
after the move. The share of the 1 to 3 cm low surface that ends within 3 mm of the plane goes from 100% to 87% and 82%.

Limits: (i) one scene and one sensor; 4 to 6 pixels are candidates, not a default; (ii) the numbers come from a CPU run of
this code on the stored inputs, and the RTX render that shows the rim has not been checked (views 1, 5, 6 and 7 are the ones
to look at); (iii) the ramp moves vertices only (at 4 and 6 the faces, uv and texture are identical to those at 0, 279145
triangles), so it cannot close a gap the triangulation leaves: the thin white seam at the front edge is still there in a CPU
picture of the mesh at 6, and whether it shows in the RTX render has not been checked; (iv) how much of the pot's floor gap
(+2.5 to +0.3 cm) the ramp leaves has not been measured; (v) the current symmetric band has the same rim problem on a smaller
scale (14 stretched triangles in the current output, by the same test), so the same ramp could apply there, but its effect has
not been measured.

### Inputs And Outputs

Inputs:

- Video mode: `--video-fpath /path/to/video.mov`
- ZED mode: connected ZED camera and `--pipeline zed`

Final outputs:

- `Data/<scene>/s14_og/reconstructed_og_scene.json`
- `Data/<scene>/s14_og/reconstructed_scene.png`
- `Data/<scene>/s14_og/settled_poses.json`

## B Augmentation

Generates digital cousin assets, samples scene variants, and proposes tasks for a reconstructed scene.

### Example

```bash
bash scripts/pipeline/B_augmentation/run.sh \
  --scene-name pull_scene_2 \
  -- prompt_cousin_structured.max_objects=2 \
       prompt_cousin_structured.max_generated_images_per_object=1
```

Run only part of B:

```bash
bash scripts/pipeline/B_augmentation/run.sh --phases object-cousins
bash scripts/pipeline/B_augmentation/run.sh --phases scene-variations,task-generation
bash scripts/pipeline/B_augmentation/run.sh --include-p2p
```

### Stages

| ID | Script | Env | Purpose | Key outputs |
|---|---|---|---|---|
| `1` | `B_augmentation/stages/1_prompt_object_cousins.py` | `simfoundry` | Ask a VLM/image model for object cousin images. | `prompt_cousin_structured/` |
| `2` | `B_augmentation/stages/2_generate_cousin_combinations.py` | `simfoundry` | Choose which cousins to use together. | `cousins_combination/combinations.json` |
| `3` | `B_augmentation/stages/3_generate_cousin_meshes.py` | `hunyuan` (`--env-mesh`) | Generate textured cousin meshes. | `cousin_generation/` |
| `4` | `B_augmentation/stages/4_make_cousins_sim_ready.py` | `simfoundry` | Convert cousin meshes to sim-ready URDFs. | `sim_cousins/` |
| `5` | `B_augmentation/stages/5_import_cousin_usd.py` | `simfoundry` | Import cousin URDFs as USD assets. | `usd_cousins/`, custom asset dataset entries |
| `6` | `B_augmentation/stages/6_sample_reconstructed_scene.py` | `simfoundry` | Swap cousins into the reconstructed scene and sample variants. | `s14_og/auto_generation/` |
| `7` | `B_augmentation/stages/7_propose_scene_tasks.py` | `simfoundry` | Propose simple task YAMLs for the scene. | `proposed_tasks/*.yaml` |
| `8` | `B_augmentation/stages/8_match_cousin_p2p.py` | `simfoundry` | Optional point-to-point correspondence between base and cousin meshes. | `cousin_p2p_match/` |

### Inputs And Outputs

B expects a completed A reconstruction, especially:

- `Data/<scene>/s6_upsample/`
- `Data/<scene>/s14_og/reconstructed_og_scene.json`
- imported dataset assets from A Stage 13

Important B config keys in `scripts/cfg/real2sim_cfg.yaml`:

- `prompt_cousin_structured.*`
- `generate_cousins_combination.*`
- `cousin_generation.*`
- `sim.*`
- `usd.*`
- `propose_scene_task.*`
- `cousin_p2p_match.*`

The wrapper intentionally sets small local defaults for cousin count and task count. Override them after `--` for larger runs.

## C Application

Loads a reconstructed scene for smoke tests, policy evaluation, teleoperation, annotation, demo generation, and replay.

### Examples

```bash
bash scripts/pipeline/C_application/run.sh --scene-name pull_scene_2 --mode smoke-random
bash scripts/pipeline/C_application/run.sh --scene-name pull_scene_2 --mode eval
bash scripts/pipeline/C_application/run.sh --scene-name pull_scene_2 --mode demo
```

Use a generated task YAML in smoke mode:

```bash
bash scripts/pipeline/C_application/run.sh \
  --scene-name pull_scene_2 \
  --mode smoke-random \
  -- application_smoke.task_config=/path/to/task.yaml
```

### Modes And Stages

| Mode / ID | Script | Purpose | Key outputs |
|---|---|---|---|
| `smoke` | `C_application/stages/0_smoke_random_actions.py` | Headless random-action load/step test. | `application_smoke/*.mp4` |
| `1` | `C_application/stages/1_eval_policy_og_scene.py` | Evaluate a policy in the OG scene. | `s15_eval/` |
| `2` | `C_application/stages/2_teleop_og_scene.py` | Collect teleop demonstrations. | `s14_teleop/` |
| `3` | `C_application/stages/3_annotate_src_demo.py` | Annotate source demos. | `s15_annotation/` |
| `3b` | `C_application/stages/3b_modify_annotations.py` | Adjust annotations. | `s15_annotation/` |
| `4` | `C_application/stages/4_extract_waypoints.py` | Extract object-centric waypoints. | `s16_waypoints/` |
| `5` | `C_application/stages/5_generate_demos.py` | Generate demos from waypoints. | `s17_generated_demos/` |
| `6` | `C_application/stages/6_replay_dataset.py` | Replay generated demos. | `s18_replay/` |

Mode mapping:

- `--mode smoke-random`: stage `smoke`
- `--mode eval`: stage `1`
- `--mode demo`: stages `2,3,3b,4,5,6`
- `--mode full`: stages `1,2,3,3b,4,5,6`

### Task Predicates

Task YAMLs (see `scripts/cfg/task/example.yaml` for a fully commented example) use two
predicate vocabularies:

**Spatial placement predicates** (`group_predicate_placement`, applied at each episode
reset; implemented in `simfoundry/utils/placement_utils.py`): `on_top`, `left_of`,
`right_of`, `behind`, `in_front_of`, `inside`, `near`, `between`, `inside_link`.
The horizontal predicates displace one axis by `gap` and sample the other; `near`
places at a random direction around the reference; `between` places on the segment
between two reference groups (`reference_groups: [a, b]`); `inside_link` places
within a named link's AABB (`link_name`, e.g. a shelf level or drawer).

```yaml
group_predicate_placement:
  cup:
    reference_group: plate
    predicates: [left_of, near]
    gap: {left_of: [0.03, 0.15], near: [0.02, 0.10]}
```

**Check predicates** (`init_predicates_*`, `goal_predicates_*`, `milestone_predicates`):
any OmniGibson object state (`OnTop`, `Touching`, ...) plus the SimFoundry special
states — `PlaceOnTop` (init-only AABB placement), `InsideAABB` (containment,
`volume_threshold`), `OnTopAABB` (`z_tolerance`, `xy_overlap_threshold`), `AboveAABB`
(`min_clearance`, optional `xy_overlap_threshold`), `Lifted` (unary, `min_height`
above the episode-start pose), and `IsGrasping` (milestones only). The AABB-based
states are recommended for reconstructed/custom assets, whose collision meshes often
break OmniGibson's sampling-based states.

## Cache And Test Mode

A and B can cache raw remote model responses:

```bash
--cache-mode --model-cache-dir .cache/simfoundry/model_calls
```

Replay cached responses:

```bash
--test-mode --model-cache-dir .cache/simfoundry/model_calls
```

Use this for reproducible debugging and CI-like checks when remote model calls are expensive or unstable.
