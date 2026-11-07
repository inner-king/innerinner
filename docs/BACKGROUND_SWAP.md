# Background swap: splat background, colour matrix and counter patch

For people who run the tofu policy runner (`scripts/cutting/tofu_policy_episodes.py`). The runner's default background is a
3D Gaussian splat (USDZ), set by the packaged spec `simfoundry/configs/visual_background/p1.json`. A spec replaces the 5c mesh
background and changes only what it names; collision and physics stay as they are. `--visual-background none` gives the old
5c mesh background. The code is `simfoundry/utils/splat_background.py`. Each number carries its date (and its seed, where one
applies). The design notes behind the numbers are kept outside this repository.

## The default (since 2026-11-07)

The splat background is on by default. Without `--visual-background`, the runner loads the packaged spec. To run the old
background, pass `--visual-background none`. That path makes the same calls as the runner before this change;
`tests/test_tofu_policy_visual_background.py` checks this on the runner's syntax tree.

### Assets are not in the repository

The splat (`PD_slab4_bg.usdz`, about 51 MB) and its pose file (`align_v4b.json`) come from a scan of the owner's kitchen. They are
not published here. Put both files in `Data/visual_background/` (the `Data` folder is gitignored), or name another folder with
`SIMFOUNDRY_VISUAL_BACKGROUND_ROOT` or `--visual-background-root`. Without the files the runner stops with one message. It names
the files, the folder it searched, and the two ways out: point it at a folder, or pass `--visual-background none`.

> **TBD (owner's decision, still open):** where the two files are hosted.

### The packaged spec fits one scene

The packaged spec is aligned to the LiDAR kitchen. It names that scene's mesh background and the object `iter_7`. Before it
changes anything, the guard checks `iter_7`'s `expected_file_hash` against the scene JSON. Any other scene stops with an error
that points at `--visual-background none`, the runner's default `--scene` (`Data/tofu_chefd`) included. If the scene JSON has no
hash to compare, the guard says it cannot verify the scene. A spec may carry the optional key `scene_requires`, which lists the
hashes the guard checks; a spec without it works as before.

## Spec file

A spec is one JSON object. `load_visual_background` reads it and runs `_checked_spec`, the same checks that a hand-built
`SplatBackgroundSpec` gets when `swap_visual_background` is called.

| key | required | rule |
|---|---|---|
| `schema` | yes | must be `1` (`SCHEMA`) |
| `name` | yes | non-empty string |
| `usdz` | yes | path ending in `.usdz`; the file must exist |
| `pose_json` or `matrix` | exactly one | `pose_json`: a file with `M_da3_to_scene`; `matrix`: a 4x4 similarity transform (`_checked_matrix`) |
| `dz_m` | no, `0.0` | finite, `abs <= 0.02` m (`MAX_ABS_DZ_M`) |
| `floor_plane_visible` | no, `false` | a boolean |
| `remove` | no, `["mesh_background_0"]` | distinct object names, not `gs_background` (`GS_NAME`) |
| `kit_settings` | no, `{}` | keys start with `/`; values are bool, number or string; NaN and infinity are refused |
| `color` | no | must be `null` |
| `ccm` | no | 3 rows of 4 finite numbers, `abs <= 4` (`_checked_ccm`) |
| `counter_patch` | no | see the next table (`_checked_counter_patch`) |
| `scene_requires` | no | object name to `expected_file_hash`; checked before anything changes (see "The packaged spec fits one scene") |

Unknown keys are refused. Relative `usdz` and `pose_json` paths are taken from the spec file's folder.

`counter_patch` is an object whose four keys are all required:

| key | rule |
|---|---|
| `rects` | 1 to 8 rectangles `[xmin, xmax, ymin, ymax]`; each number `abs <= 5` m; `xmin < xmax`, `ymin < ymax`; each side at least 1 mm (`MIN_RECT_SIDE_M`) |
| `z_top_m` | finite, `abs <= 0.02` m; the top face of every box |
| `thickness_m` | from `1e-4` to `0.05` m inclusive (`MIN_THICKNESS_M`, `_PATCH_MAX_THICKNESS_M`) |
| `rgb` | 3 numbers in `[0, 1]` |

Sides are compared after rounding to a picometre. Rectangles may overlap.

The packaged spec is `simfoundry/configs/visual_background/p1.json`. Its values, with the paths relative to the spec file, are
the example below. Its counter patch has five boxes, and its reflectance `(0.1875, 0.1825, 0.185)` is the seam calibration of
2026-11-06 14:36 (22 jobs of seed 1000; the front border's seam step is 0.5/255 against a limit of 6).

```json
{
  "schema": 1,
  "name": "P1",
  "usdz": "PD_slab4_bg.usdz",
  "pose_json": "align_v4b.json",
  "dz_m": 0.0,
  "floor_plane_visible": false,
  "remove": ["mesh_background_0", "iter_7"],
  "kit_settings": {},
  "color": null,
  "ccm": [[0.4614666887256109, 0.0, 0.0, 0.0], [0.0, 0.4619944845606822, 0.0, 0.0], [0.0, 0.0, 0.4748459849932909, 0.0]],
  "counter_patch": {
    "rects": [[-1.52, 0.37, -1.77, -0.09], [-0.34, 0.19, -0.09, 0.0], [-1.45, -0.8, 0.0, 0.29], [-0.34, -0.03, 0.0, 0.29], [-1.45, -0.03, 0.29, 0.34]],
    "z_top_m": -0.0005,
    "thickness_m": 0.004,
    "rgb": [0.1875, 0.1825, 0.185]
  },
  "scene_requires": {"mesh_background_0": "98ea90c59155fa834fd340d64ebc19b2", "iter_7": "db7a0f67450bd93abbc1c04f1a9b4306"}
}
```

## Running it

```bash
# the default: the packaged 3DGS spec (the two asset files must be in place, see above)
python scripts/cutting/tofu_policy_episodes.py --scene SCENE_DIR --out OUT_DIR

# another scene, or a run that must match the old 5c mesh look
python scripts/cutting/tofu_policy_episodes.py --scene SCENE_DIR --out OUT_DIR --visual-background none

# your own spec
python scripts/cutting/tofu_policy_episodes.py --scene SCENE_DIR --out OUT_DIR --visual-background SPEC.json
```

`--out` is the only required argument. `--scene` defaults to `Data/tofu_chefd`, which the packaged spec does not fit; pass the
LiDAR kitchen's folder with `--scene`, or pass `--visual-background none`. In `main`, the swap runs after the scene edit
and before the sensors. `main` then applies `kit_settings` with `og.app.set_setting` (a failure is recorded, and the run goes
on), builds the environment with the spec's `floor_plane_visible`, and, when the spec has a `ccm`, writes it to the splat and
renders three times.

`--probe-out DIR` (with `--probe-ticks`, default `0 25 50 75 99`) is a test-only probe. It saves the front and wrist frames of
those ticks as `DIR/ep<seed:05d>/t<tick:04d>_<camera>.npz`, with `rgb` and `seg_instance`, and `seg_semantic` when the read has it (`CameraProbe`). The arrays come from
the same `get_obs()` read as the policy's images (`EpisodeRun.capture`); the probe never reads a camera again.

## What the scene and the record get

- Each name in `remove` is deleted from `init_info` and `object_registry`; `gs_background` is added to both, as a `USDObject`
  with the splat's path, scale and MD5 (`gs_background_entry`).
- Each box follows the splat, in rectangle order, in both sections, named `counter_patch_<i>`. A name the scene already has is
  refused.
- `setup.json` gets `scene_changes.visual_background`. A swap writes `name`, `usdz`, `md5`, `removed`, `pos`, `ori_xyzw`, `scale`,
  `dz_m`, `floor_plane_visible`, `kit_settings`, and `kit_settings_applied` and `ccm_applied` when they run. The `ccm` and
  `counter_patch` keys appear only when the spec has them. A `none` run writes `{"name": "none"}`.
- `setup.json`'s `args` lists `visual_background`, and `probe_out` and `probe_ticks`, only when given (`setup_args`). A `none`
  run lists `visual_background` as `none`.

### For readers of setup.json

Both `args.visual_background` and `scene_changes.visual_background` are present in every run, swapped or not. The key's presence
does not say whether the background was swapped. The `name` does: `none` means the old 5c mesh background, any other name
means the splat was swapped in. Only a swapped record has the other keys.

The places that read or write `visual_background`, in this repository and in the cutting checks outside it:

| Where | What it reads or writes |
|---|---|
| `scripts/cutting/tofu_policy_episodes.py`, `load_run_visual_background` | reads `--visual-background` and the spec root; `none` loads no spec |
| `scripts/cutting/tofu_policy_episodes.py`, `setup_args` | records `args.visual_background` (`none` when off); the root and the probe options only when given |
| `scripts/cutting/tofu_policy_episodes.py`, the `scene_changes` writes | writes `scene_changes.visual_background`: `{"name": "none"}`, or the swap record |
| `simfoundry/utils/splat_background.py` | `load_visual_background` and `packaged_visual_background_root` read the spec and the root; `swap_visual_background` builds the record |
| `setup.py` | ships `configs/visual_background/*.json` as package data |
| `tests/test_tofu_policy_visual_background.py`, `tests/test_splat_background.py`, `tests/test_tofu_policy_probe.py` | check the options, the record and the runner's syntax tree |
| the cutting checks outside this repository | the default-run checker reads `args` and `scene_changes`; the run script sets the root variable for the default run and passes `--visual-background none` for the old-background run |

## The colour matrix (`ccm`)

A 3x4 matrix, written to the splat's prims as `omni:nurec:ccmR`, `ccmG` and `ccmB` (`CCM_ATTRIBUTES`). `apply_splat_ccm` refuses
a splat with none of the three attributes, and a prim with only some of them.

Why: a plain splat renders brighter than the training video. On the 71 training frames (2026-11-05 16:22, no seed) the
mean ratio of OmniGibson to video was 1.279, 1.308 and 1.325 per channel. A linear matrix brought the ratio to 0.998-1.001 and
raised the PSNR from 14.13 dB to 18.30 dB, worst frame 15.20 dB (2026-11-05 16:22; repeated on 2026-11-05 17:30).

## Counter patch

Why: the splat draws nothing on a strip of counter in front of the cutting board, on the robot's side, and there the wrist
camera sees the sky (2026-11-05 16:25).

What: each rectangle becomes an opaque, visual-only, fixed `PrimitiveObject` Cube named `counter_patch_<i>`
(`counter_patch_entry`). OmniGibson's default Cube is 1 m on a side (`PrimitiveObject.__init__` starts the extents at
`th.ones(3)`), so `scale` is the box size in metres, and `size` is not passed. The top face is at `z_top_m`; `rgba` is `rgb`
with alpha 1. The box's state (`counter_patch_state`) adds the keys OmniGibson writes for a Cube: `radius` and `height` set to
-1, and `size` set to 1.0.

OmniGibson's `PrimitiveObject._dump_state` writes these keys, and `_load_state` reads `size` for a Cube. A state without `size`
makes a scene restore raise `KeyError`; the keys were added with the commit "Counter-patch boxes: give the scene state the keys a Cube needs" (2026-11-05 18:53). Expected warning, once per box: `Custom
scale specified for primitive object, so ignoring radius, height, and size arguments!` It comes from `PrimitiveObject._post_load`
whenever a box has a scale. The seed-1000 run printed it five times, once per box (2026-11-05).

Inside the cleared slab the boxes show alone. Outside it, the splat's translucent top layer is drawn over them, which is why the
patch colour has to be calibrated on the seam step, not matched to the splat colour (2026-11-05 18:25). Without the derived splat,
the fog layer also covers the footprints of objects on the counter. Make the derived file with
`splat_derive.py --clear-slab` (in the cutting checks, outside this repository) and point `usdz` at it.

## Known limits

- **Brightness and PSNR.** PSNR is 18.30 dB, below the 28 dB target (71 training frames, 2026-11-05 17:30). It was measured at
  training time and does not depend on the reflectance. A per-channel curve gives 22.97 dB on the same frames. The causes are not
  separated: tone curve, splat texture, antialiasing, and dark spots on the sink and hob faces.
- **Side effects of the counter patch.** Against the old background, the share of pixels that differ by more than 4/255 is 0.753 %
  of the front frames and 0.911 % of the wrist frames (mean over the 22 render-substitute jobs of seed 1000, 2026-11-06 14:36).
  The worst job is 0.830 % (front) and 2.632 % (wrist). The limit is 0.1 %.
- **Render time.** A render with the packaged spec takes 1.5 to 1.9 times as long as the 5c background: median 15.7 ms against
  8.1 ms (1.94 times) in one slot, 12.4 ms (1.53 times) in an earlier one, both on 2026-11-06. It depends on the machine's load.
- **Brightness.** Against the old background, the mean brightness of the judged frames is 0.600 to 0.624 times in front and
  0.665 to 0.678 times on the wrist (policy-runner replay of seed 1000, 2026-11-06 14:36). Checkpoints trained on the old
  background see a different image by default. To evaluate one as it was trained, pass `--visual-background none`.
- **Policy evaluation on the new background (observed 2026-11-07; cause not decided).** In an evaluation of three checkpoints
  trained on the old background (one scene; training seeds 0, 1 and 2; 50 evaluation episodes per seed), all 150 episodes end
  in a basic and a strict success, and none fails. The policy still cuts every time, but on the new background it follows the
  tofu's position less well: the pooled b_s is -0.516 (interval -0.648 to -0.413) against -0.166 (-0.26 to -0.09) on the old
  background, and the standard deviation of v0 is 8.13 mm against 4.71 mm. On the new background, b_s is -0.470 (interval
  -0.620 to -0.353), -0.523 (interval -0.682 to -0.383) and -0.554 (interval -0.681 to -0.453) for seeds 0, 1 and 2, and the
  standard deviation of v0 is 7.86, 8.54 and 7.92 mm. The cut position of the same episode moves by a median of 5.3 to 8.3 mm
  between the new background and the control evaluation, and by at most 21 mm. The blade depth is normal: +0.70 to +0.81 mm,
  against +0.58 to +0.75 mm on the old background. Over the whole frame, the new background's brightness relative to the old
  one is 0.69 in front and 0.70 to 0.71 on the wrist. The cause is not separated: the look and brightness of the background and
  the dark wrist region are not separated from each other.
  A control evaluation uses the same runner on the same system (Ubuntu 26.04) with the background off (`--visual-background none`),
  one evaluation per training seed, about 18 to 19 minutes per training seed. Its b_s is -0.212 (interval -0.302 to -0.138),
  -0.162 (interval -0.255 to -0.074) and -0.189 (interval -0.320 to -0.070) for seeds 0, 1 and 2; its standard deviation of v0 is
  4.29, 4.40 and 5.23 mm; and its median blade depth is +0.57, +0.63 and +0.73 mm. The basic success is 50 of 50 episodes in
  every evaluation. The old-background evaluation gives b_s -0.208 (interval -0.297 to -0.138), -0.106 (interval -0.191 to
  -0.029) and -0.184 (interval -0.306 to -0.072), with standard deviations of v0 of 4.45, 3.74 and 5.21 mm. Paired by seed, the
  control minus the old background is -0.004 (interval -0.037 to +0.027), -0.056 (interval -0.101 to -0.011) and -0.005
  (interval -0.065 to +0.055), and the new background minus the control is -0.258 (interval -0.367 to -0.164), -0.361 (interval
  -0.485 to -0.254) and -0.365 (interval -0.460 to -0.275). The cut position of the same episode moves by a median of 0.8 to
  1.7 mm between the control and the old background, and by at most 6.3 mm. With the background switched off, the same runner
  on the same system gives back the old-background numbers for all three checkpoints and stays within 0.06 of the old
  background's b_s, so the loss on the new background is not an effect of the runner or of Ubuntu 26.04. The second checkpoint
  (seed 1) shows a small difference between the control and the old background (-0.056) that the data cannot tell from
  run-to-run variation. To evaluate a checkpoint as it was trained, pass `--visual-background none`.
- **Wrist edge.** The front border's judged boundary is also seen by the wrist camera. There its step is +9/255 whatever the
  reflectance; the wrist sample is below the rule's minimum, so this is a known limit outside the judged rule (2026-11-06 14:36).
- **Front seam.** At the tentative reflectance 0.244 the judged front border had a seam step of 12/255 against a limit of 6 (22
  jobs of seed 1000, 2026-11-05 17:56). At the packaged reflectance it is 0.5/255 (2026-11-06 14:36).
- **Dark wrist region (observed; cause open; 2026-11-07).** On every run with the new background, the wrist frames at ticks 30 to
  39 have pixels whose largest channel is below 25 on 15.9 to 21.7 % of the frame on average. The peak tick is 31 to 35 %, and 8
  or 9 ticks go above 10 %. The old background's recording of the same seed peaks at 4.8 % and has no tick above 10 %. The runs are
  seed 1000 (the probed p1 run and the blind replays) and seeds 1000 and 1001 (the policy runs). Of the dark pixels of the probed
  run, an estimate from the stored poses puts 99.8 % on the front counter-patch box 0. They are not on the induction hob or the sink, which are not in view at
  ticks 30 to 39. They are not the knife either: the knife's image region is fixed across ticks, and only 1 to 4 % of it is dark
  at ticks 30 to 39. The probe with instance labels at ticks 30, 33, 35, 37 and 39 (seed 1000, 2026-11-08) found 53,660 dark pixels that are
  not the knife; counting the labels puts 100.0 % of them on the front counter-patch box 0. The label says which object a pixel belongs to;
  it does not rule out a dark element of the splat drawn over the box. The cause is not decided: an arm shadow on the
  low-reflectance box is plausible, but other shading, or a dark element of the splat drawn over the box, has not been
  ruled out. A policy trained on the
  old background sees a much different wrist image during the cutting phase.
- **Two rendered states.** The same configuration can render in one of two slightly different states: the frames differ by about 2 levels of 255, mostly at edges, with zero mean. Both states were seen on Ubuntu 24.04 and on 26.04; the cause is not known.
- **Probe wrist frame at tick 0** shows the front camera's view (about 2026-11-05 18:15).
- **Probe segmentation.** OmniGibson adds `seg_semantic` whenever `seg_instance` is requested.
- **Replaying the 5c probe** is not solved.

## Tests

From the repository root, with the `simfoundry` environment's Python:

```bash
python -B -m pytest -p no:cacheprovider -q \
  tests/test_splat_background.py tests/test_tofu_policy_visual_background.py tests/test_tofu_policy_probe.py \
  tests/test_background_swap_doc.py tests/test_cutting_package_doc.py
```

Run on this branch with this change (2026-11-08): 427 passed and 11 skipped in the five files above.

The 11 skips are environment checks, not failures. Six cases of `tests/test_splat_background.py` read the scene JSON named by
`SIMFOUNDRY_TEST_SCENE_JSON`; one reads `SIMFOUNDRY_TEST_ALIGN_JSON` and one `SIMFOUNDRY_TEST_PD_USDZ`. Two need `pxr`, which the
`simfoundry` environment does not have (`test_usdz_is_a_nurec_volume_whose_extent_is_the_ply_aabb` and
`test_apply_on_a_real_usd_stage_writes_the_matrix_and_reads_it_back`). One rule check in `tests/test_cutting_package_doc.py`
imports OmniGibson and torch. The two runner test files compare the runner's syntax tree with the base
commit named by their `BASE_COMMIT`: `6cb6ad5` for `tests/test_tofu_policy_visual_background.py` and `c06e04d` for
`tests/test_tofu_policy_probe.py`. They skip when that commit is not in the clone.
