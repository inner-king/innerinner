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

Stage 6 checks each upsampled silhouette against the stage-5 crop it was made from (`s6_upsample.reference_check`,
`simfoundry/pipeline/upsample_check.py`). An image model that cuts a corner, fills a hole or redraws the object at
another size changes the silhouette, and the check records the IoU, the area ratio and the change in hole fraction
in `stage_info.json` under `reference_checks`, warning when a threshold is crossed. With `action: revert` a flagged
object keeps its stage-5 crop instead. The thresholds are provisional: one scene and 15 objects, with evidence only
for `min_iou`. With `infill: true` the model paints the hidden part of the object too, so the silhouette can change
for a good reason; then a flag means the picture differs from the crop, not that it is wrong. With the check disabled,
stage 6 writes nothing extra.

Two options against colour drift are off by default (`simfoundry/pipeline/upsample_color.py`). With
`s6_upsample.infill_mode: occluded_only` the crop's visible pixels are kept and the infill is taken only where the
object was hidden, with a seam `infill_feather_px` wide; by default the model repaints the visible surface too.
`s6_upsample.infill_rim_trim_px` (default 0, off) takes that many pixels off the infill's outer edge, which can be a
dark rim, and leaves the visible surface alone. With
`s6_upsample.color_match.enabled` the upsampled object's Lab colour statistics are moved onto the crop's visible
surface, and the match is skipped when more than `max_filled_fraction` of the object was painted. When either is on,
`stage_info.json` records the numbers under `infill_composites` and `color_matches`: the headline ΔE compares each
object's own alpha, and the visible-surface and per-pixel distances sit beside it.

### Asset names and re-imports

Stage 11 names each object's model from its content: the seed of the six-letter name carries an eight-digit hash of the stage-8 mesh (for an articulated object, of its parts and URDF) and of the scale it is written at (`s11_sim.model_name_content_hash`, on by default; `false` gives the old names, which depend on the scene, the category and the index only). The name does not cover what stage 11 derives from that mesh (collision meshes, the URDF of a rigid object) or the mass and friction the VLM proposes; a change to those keeps the name, and stage 13 then reports it as a geometry or physics change. Stage 13 imports each asset into the shared dataset, where an import under an existing name used to overwrite the asset that an earlier scene points at. Each imported asset now carries a manifest (`real2sim_asset.json`) with the hashes of its inputs, and a re-import reports one of: a geometry change (`s13_usd.on_conflict`: `warn` by default, `fail` to stop before the import, `overwrite` to log at INFO), a change of the import settings or the import code (a warning), an asset made before manifests existed (a warning), or a physics-only change (INFO, with the mass before and after). The decision is recorded per object in `stage_info.json` under `asset_imports`. `s13_usd.skip_unchanged` (off by default) skips an import when the manifest says nothing changed. The manifest hashes the stage-11 inputs, not the imported asset, so a hand edit to the dataset's USD is not seen, and a skip copies that edit into the scene. Each run keeps a copy of its finished assets in its own `s13_usd/objects/<category>/<model>`; the shared asset that an old name overwrote exists only in those copies.

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
  is only what is seen: the physics uses the floor plane, which is not visible here.
- A ring that is not one surface is filled at its widest surface's height, so a ring that also holds a
  lower surface (a sink beside a counter) can leave a step on the other side of the mask edge.
- The measured counter of the reference scene sits +1.2 cm above z = 0 at its median, with a 5-95% spread of
  -1.9 to +2.0 cm. `plane_snap_tol_m` (1 cm) therefore puts only part of the counter on the plane; the rest is
  relief, and the front edge keeps its sawtooth and the curtain-like stretch of grazing surfaces.
- The 22 mm step between the counter and the cooktop glass is cut by the relief limit (`relief_max_edge_ratio` 4.0), so a low side view shows holes at the mask edge; measured on one scene (the rerun copy), where the glass box drops 502 triangles against 1050 for 894363c, and the provisional 2.0 removes the front-edge sawtooth but grows the largest dropped piece from 3423 to 10003 triangles, so the value is a trade-off, not a fix.
- The mesh is only as good as the depth.

`--bg-mesh` also turns on stage 14 (`s14_og.include_bg_mesh`). Stage 14 writes the GLB as a USD in the stage-4
world (`s14_og/background/mesh_background_0.usd`, with its texture beside it), adds it to the scene as the
visual-only object `mesh_background_0`, and records it in `reconstructed_og_scene.json` by a path relative to that
file. Nobody attaches it by hand in the light editor. It has no collision: the objects rest on the floor plane,
which is hidden while the mesh is shown. It is not combined with `include_gs`; with both on, the mesh is skipped
with a warning. Without the GLB, stage 14 warns and goes on without it. The saved JSON keeps the relative path, so a consumer resolves it with `simfoundry.utils.scene_utils.load_json_with_absolute_usd_paths` before handing the file to OmniGibson, since restoring the JSON as it is fails (OmniGibson resolves a relative path against a temporary scene USD).

### Size Check

Stage 8 fits one scale per object, so a placed mesh can disagree with the depth and nothing says so. Stage 8 now
compares each placed mesh with its depth: the two sides of the minimum-area rectangle of the footprint on the plane
perpendicular to gravity, and the height against the depth's height above the surface the object stands on. A
footprint difference above `s8_pose.size_warn_ratio` (0.15, provisional) is logged and recorded in `info/<img>.json`
under `size_check`, and listed in `stage_info.json` under `size_warnings`. A height is reported only when it is also
outside the noise of its own floor: `s8_pose.height_noise_sigmas` (2.0, provisional) times the floor's robust spread
(`floor_spread_m`, 1.4826 times the median absolute deviation of the floor points). The floor is the median of the
object's ring; when the ring has fewer than 50 points (`size_check.DEFAULT_MIN_RING_POINTS`) and
`s8_pose.height_floor_fallback` is true (the default), the floor is read from the pixels around the object with no
neighbour excluded, as the most populated 5 mm height bin of them (`floor_source: "neighbour_masks"`, logged as
"floor read from the neighbouring object's pixels"); otherwise the height is skipped with the reason. The mesh is not
changed by the check. With `s8_pose.height_fit.enabled: true` (off by default) a height that differs by more than its
threshold, and more than the floor's noise, is stretched to the depth's height, provided the factor is within its
limits; the decision, and any refusal, is recorded under `height_fit`. The stretch works on a canonical GLB that
carries a node transform: the transform is baked into the vertices, the stretch is made there, and the node is reset,
so the file reads back in the frame the check measured. `size_warn_ratio: null` turns the check off entirely.
`info/<img>.json` records `cpd_scale` and `final_scale` (`pre_scale_factor` times `cpd_scale`, the placed size) on every
run; with the check on it also records `height_fit_scale`, the stretch along up (1.0 when none was applied), which is
not in `final_scale`.

Known false warnings, from the lidar run (`b2_check`), that the check does not detect yet:

- A container or tray whose rim is thin: the depth's 95th percentile reads the inner bottom, not the rim. The tray
  (iter_5) had a depth maximum of 41 mm against a mesh of 38 mm, but a 95th percentile of 14 mm, because about 1 % of
  its masked points lie above 23 mm.
- A mask covering two objects: the pot (iter_6) was segmented together with its glass lid, and its mesh contains
  both. The footprint then differs from the depth's by 65 %, and the height cannot be read as the pot's alone.
- A floor from a ring that also sees other surfaces: the tray's ring spread is 11 mm, so a 25 mm mismatch is only
  just outside the noise.

These are not handled by any rule here, so a warning on such an object is a prompt to look, not a measurement.

Stage 8 seeds Python's `random`, numpy and Open3D from the global `seed` once per object, offset by the object's index (`seed` + index), so the CPD pose search and the Poisson-disk target of an object do not depend on which other objects this run processed. The FoundationPose call, when enabled, is also seeded for each object, from the global `seed`; whether its GPU kernels are deterministic is not verified.

#### Height fit safeguards

Three settings under `s8_pose.height_fit` refuse some stretches that the height rule above would apply. All three are
off by default, so the defaults change nothing. Their numbers are checked at start-up even while their switch is off
(`validate_height_fit_safeguards`), so a bad value stops the run before any object is processed.

- `skip_rod_like` (the main rule, default `false`): an object is not stretched when its long/mid extent is at least
  `rod_min_elongation` (3.0) and its short/mid extent is at least `rod_min_roundness` (0.6).
- `min_axis_alignment` (default `null`): the stretch is applied only when the shortest axis of the section is within
  the given |cos| of vertical. A section whose short/mid is above 0.9 (`AXIS_UNDEFINED_SHORT_OVER_MID` in
  `size_check.py`) has no determinable shortest axis, and its stretch is refused.
- `min_result_extent_m` (default `null`): the stretch is refused when the thinnest extent of the stretched mesh would
  be below this many metres.

**Why the rod rule.** Most of the height of a thin, long object comes from the tilt of its long axis. Research 1's pen
in the new run (`bowl new`) is 126.6 mm long and its long axis is 13.0° above horizontal, so 126.6 mm × sin 13.0° =
28.5 mm of its 30.2 mm mesh height is tilt. Matching that height to the depth stretches the section, not the tilt, so
a round section becomes an ellipse. The old pen (`bowl old`) is the same kind: 126.5 mm × sin 8.4° = 18.5 mm of its
22.5 mm. A flat object (a knife blade, a cutting board, a tray) is the opposite case: its thickness is its height, and
the same stretch is the right direction.

**Why not the alignment rule alone.** The alignment rule separates the red marker (|cos| 0.66) and the new and nofit
pens (|cos| 0.21), but not the old pen. Its shortest axis is nearly vertical (|cos| 0.98), and its round section is
squashed all the same. In the table below the two auxiliary rules skip only rows that the rod rule also skips, so the
table alone does not show that they are needed. They are separate settings for two other failures: a shortest axis
that is not vertical, and a result that is too thin.

**What is measured, and what is not.**
- Measured: one harm. The stretched pen (`bowl new`) was physically unstable in research 1's settle run: `settle.py`
  reported `ok = False` over 240 steps and the pen moved 0.509 m. Its final pose in stage 14 was about 0.34 m beside the
  unstretched pose (y −0.431 m against −0.091 m), and its centre height in stage 12 was 12.9 mm against 4.8 mm. The old
  and unstretched pens settled (`ok`). Its stage-11 mass estimate was 3.25 g against 7.18 g unstretched, with the VLM's
  typical value clamped at the low end. The cause was not checked: the report lists the 2.6 mm thickness, the contact
  offset and the collision hull as candidates
  (`~/simfoundry_logs/second_scene_check/report_put_cup_in_bowl_ko.md`, outside this repository).
- Not measured: whether any row is harmful in a photo or visual check. The red marker and the old pen are judged from
  geometry alone; their photo columns in research 1's table are empty by design. Whether the twenty rows the rules
  leave alone gain from the stretch is not measured either. No photo or visual check says whether stretching the
  cutting boards and the tray helped.
- Candidates: every number on this page is a candidate, not a tested default. They come from one table of 24 rows of
  9 objects in 3 scenes. 19 of the 24 rows are one scene's four runs (the lidar kitchen), so the sample is small.

**The 24 rows.** These are the objects whose height fit ran, or would run, at research 1's settings (threshold 0.25,
`min_scale` 0.1, `max_scale` 2.0), out of the 45 rows of its table. The values were recomputed with `mesh_axis_profile`
from the written stage-8 meshes and agree with research 1's table to the precision shown. "Needed scale" is the factor
the stretch takes the height by (depth over mesh). "Actual" rows were stretched in their run; "offline" rows would be
stretched at those settings. "–" means the rule does not skip the row.

| scene / variant | object | actual or offline | needed scale | inside [0.5, 2.0] | abs cos | short/mid | long/mid | shortest extent, mm (before → after) | rod (3.0, 0.6) | alignment (0.9) | thickness (4 mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lidar a | chef_knife | offline | ×0.614 | yes | 0.9990 | 0.3291 | 5.729 | 18.04 → 11.09 | – | – | – |
| lidar a | cutting_board | offline | ×0.607 | yes | 0.9986 | 0.0457 | 1.201 | 11.04 → 6.71 | – | – | – |
| lidar a | onion | offline | ×1.450 | yes | 0.9741 | 0.4317 | 1.041 | 44.33 → 64.06 | – | – | – |
| lidar a | beige_tray | offline | ×0.345 | no | 0.9989 | 0.1409 | 1.175 | 39.45 → 13.60 | – | – | – |
| lidar a | pot_with_liquid | offline | ×0.670 | yes | 0.9823 | 0.2797 | 1.335 | 93.42 → 62.63 | – | – | – |
| lidar d | chef_knife | offline | ×0.561 | yes | 0.9984 | 0.3413 | 5.873 | 18.41 → 10.35 | – | – | – |
| lidar d | cutting_board | offline | ×0.202 | no | 0.9987 | 0.1651 | 1.063 | 52.74 → 10.67 | – | – | – |
| lidar d | onion | offline | ×1.446 | yes | 0.9818 | 0.6211 | 1.024 | 48.60 → 71.59 | – | – | – |
| lidar d | beige_tray | offline | ×0.298 | no | 0.9997 | 0.1379 | 1.046 | 39.95 → 11.94 | – | – | – |
| lidar d | pot_with_liquid | offline | ×0.680 | yes | 0.9855 | 0.3045 | 1.448 | 101.01 → 68.87 | – | – | – |
| lidar d | blue_sink_stopper | offline | ×1.681 | yes | 0.9995 | 0.1538 | 1.085 | 10.03 → 16.81 | – | – | – |
| lidar i v1 | chef_knife | actual | ×0.689 | yes | 0.9981 | 0.3367 | 5.816 | 18.26 → 12.59 | – | – | – |
| lidar i v1 | cutting_board | actual | ×0.187 | no | 0.9984 | 0.1595 | 1.051 | 54.03 → 10.15 | – | – | – |
| lidar i v1 | onion | actual | ×1.450 | yes | 0.9871 | 0.6601 | 1.042 | 51.08 → 74.46 | – | – | – |
| lidar i v1 | beige_tray | actual | ×0.260 | no | 0.9997 | 0.1660 | 1.035 | 48.12 → 12.53 | – | – | – |
| lidar v2 | chef_knife | actual | ×0.433 | no | 0.9933 | 0.3367 | 5.816 | 18.15 → 7.90 | – | – | – |
| lidar v2 | cutting_board | actual | ×0.187 | no | 0.9984 | 0.1595 | 1.051 | 54.25 → 10.19 | – | – | – |
| lidar v2 | onion | actual | ×1.449 | yes | 0.9933 | 0.6601 | 1.042 | 51.09 → 75.66 | – | – | – |
| lidar v2 | beige_tray | actual | ×0.257 | no | 0.9997 | 0.1660 | 1.035 | 48.59 → 12.51 | – | – | – |
| marker old | blue_scissors | offline | ×0.542 | yes | 0.9939 | 0.2233 | 2.243 | 20.10 → 11.12 | – | – | – |
| marker old | red_marker | offline | ×0.789 | yes | 0.6646 | 0.9412 | 6.233 | 19.84 → 15.73 | skip | skip | – |
| bowl old | black_pen | offline | ×0.469 | no | 0.9845 | 0.7708 | 14.905 | 6.54 → 3.12 | skip | – | skip |
| bowl new | black_pen | actual | ×0.348 | no | 0.2128 | 0.7059 | 16.776 | 5.33 → 2.63 | skip | skip | skip |
| bowl nofit | black_pen | offline | ×0.348 | no | 0.2128 | 0.7059 | 16.776 | 5.33 → 2.63 | skip | skip | skip |

**The margins** (candidates; they hold for these 24 rows only):
- Long/mid 3.0: the rows the rod rule must keep have long/mid at most 1.042 (onions, short/mid 0.62 to 0.66). The rows
  it must skip have long/mid at least 6.233 (the red marker). Any limit in (1.042, 6.233] splits them the same way.
- Short/mid 0.6 (with long/mid at least 3): the knives are at most 0.341 and the pens at least 0.706, so any limit in
  (0.341, 0.706] splits them. The onions are not caught by roundness alone; their long/mid of 1.04 keeps them out.
- |cos| 0.9 for alignment: the harmful rows have |cos| at most 0.665 (the red marker) and the benign rows at least
  0.974 (an onion). Any limit in (0.665, 0.974] splits them. 0.9 is the angle of 25.8° from vertical.
- Thickness 4 mm: after the stretch the benign rows are at least 6.71 mm thick (a cutting board, lidar a), and the
  thinnest row the rule refuses is 3.12 mm (the old pen). 4 mm lies between them. It is not derived; see the physics
  floor below.

**`min_scale` and the wider range.** Research 1 ran with `min_scale` 0.1. This stage's default, in
`real2sim_cfg.yaml`, is 0.5, and the default is unchanged here. The reason for 0.1 is the flat objects. The cutting
boards need ×0.187 to ×0.202 in three runs (their mesh is 63.8 to 68.7 mm tall against a 12.9 mm depth), the trays need
×0.257 to ×0.345, and the knife in the v2 run needs ×0.433. All of them are below 0.5. A range that reaches them also
admits the pens (×0.348 and ×0.469). The three rules are meant to make that wider range usable: on these 24 rows,
within [0.1, 2.0] they refuse the pens and the red marker and leave the flat objects to the stretch. Whether stretching the cutting boards and the tray
helped in photo metrics was not measured. At the default 0.5, 11 of the 24 rows are outside the range and are not
stretched. Of the 13 that reach the range, the rod rule refuses only the red marker (×0.789).

**What the rules do not stop.**
- A flat object with a wrong depth: the height rule cannot tell a wrong depth from a wrong mesh, and nothing here looks
  at that.
- A rod-like object with the wrong diameter: a vertical stretch changes only the height, so a wrong diameter stays
  wrong. Fixing it would need a change of the cross-section, which is not done here.
- A thin plate tilted in the scene: its tilt is not corrected.
- The criterion is geometry, not names. A long object with a round section is caught because of its shape, and a flat
  one is not. No object name or category is read.

**The physics floor (`min_result_extent_m`).** The facts below come from reading the code, not from running it.
- Stage 12 (PyBullet) sets `contactProcessingThreshold=0.0` and `collisionMargin=0.0` on each object's root link
  (`12_stabilize_physics.py`, lines 205–209 and 291–295). The ground is `plane.urdf` (line 157). The repository does not
  say what these two parameters do for a thin shape: not found.
- Stage 14 (OmniGibson, in `deps/BEHAVIOR-1K`, which this repository does not track) sets the default contact offset to
  0.001 m and the rest offset to 0.0 m (`OmniGibson/omnigibson/prims/rigid_prim.py`, lines 30–31). It applies both to
  every rigid collision mesh at load (lines 232–233), with a comment that the engine's own offsets let light objects
  miss contacts. No code in `scripts/pipeline` or `simfoundry` sets a contact or rest offset. The engine's own default
  values are not in the repository: not found.
- Stage 11 builds the collision shapes with `generate_collision_meshes` (`simfoundry/utils/asset_conversion_utils.py`,
  line 640). It returns the convex hull when the mesh is convex, or when its volume is more than 90% of its convex
  hull's. Otherwise CoACD runs with `max_convex_hull` set to the hull count, `decimate=True` and `max_ch_vertex=60`.
  Hulls that are not volumes are dropped, and the convex hull is used again when the CoACD hulls' total volume is
  within 5% of the convex hull's. Nothing in this path looks at thickness. The function has five call sites (lines
  919, 1188, 1234, 1268 and 2080); which one stage 11 reaches was not traced.

Not known: why the 2.6 mm pen was unstable (not confirmed; the report names candidates and did not check them);
whether CoACD's decimation to 60 vertices per hull keeps a 2 to 5 mm section (not checked); and which setting would
make a 2.6 mm pen stable (not known). So 0.004 m is a provisional constant taken from two points of one object: the
pen as unstretched, 5.3 mm across, was stable, and the stretched pen at 2.6 mm was not. The old pen's 3.12 mm lies
between them, and which side it is on is not known. The constant is not derived from any contact setting above.

### CPD scale limit

Stage 8 scales the mesh by `pre_scale_factor` and then lets CPD choose a further scale m (`tf_z_up.scale`). With `s8_pose.cpd_scale_limit` set to r, where 0 < r < 1, each scaled CPD fit whose m lies outside [1 - r, 1 + r] is moved to the nearest bound and refitted there by a fixed-scale rigid CPD; the fits inside the bounds are unchanged, and the fit with the lowest `dist` is chosen as before. The default is `null`, which keeps the current code path. The setting is off because its only evidence is one scene (research 2's B9 study, where onion and pot came out at m 1.27 and 1.29); it is provisional and not tuned on any object. With a limit set the info file also records `cpd_scale_limit`, `cpd_scale_clamped` and `cpd_scale_unclamped`, and the log says when the chosen fit was clamped. Known and not fixed: `dist` is measured after the source is scaled by 1/m, so the `dist` values of fits with different m are not in one unit; a unit correction (dist * m^2) was tried and withdrawn (`b9_scale/report_pre_ko.md`, research 2).

### Reproducibility of stages 3 and 5

Open3D's RANSAC (`segment_plane`) runs in parallel and stops early with `probability` (Open3D 0.20 defaults to 0.99999999), so
with the default the same seed could give different planes at different thread counts. Measured on one lidar scene
(research 2, `stage3_repro/report_ko.md`): the same seed gave 16 different frame-scoring planes in 200 repeats, the object
count came out 6, 7 or 8 (30, 77 and 93 repeats), and the two real runs' floor tilts differed by 0.112 degrees and their
origins by about 2.5 mm. The frame-scoring fit and the floor fit now pass `probability=1.0`, which switches the early stop off: the
same seed gives the same plane at 1 to 24 threads. The cost is time: on this machine the frame-scoring fit (2,000 iterations) takes about 27 ms at 24 threads and 0.47 s on
one thread, and the floor fit (1,000 iterations) about 14 ms and 0.23 s. Stage 3 also seeds the floor fit on every path (a pinned `img_idx` used to skip the seeding). Stage 5 draws the
interior sample points of the removal order and the tie-breaking noise of its scale-shift fit from generators seeded by
`seed` in the config, where it used the unseeded numpy and torch generators before. This is not bit-identical to earlier
runs: the planes, the object count, the floor and the removal order's samples can change (they now repeat for a seed,
which they did not before), and everything downstream of them follows.

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
