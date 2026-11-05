# Background swap: splat background, colour matrix and counter patch

For people who run the tofu policy runner (`scripts/cutting/tofu_policy_episodes.py`) with the splat-background options.
`--visual-background SPEC.json` replaces the 5c mesh background with a 3D Gaussian splat (USDZ), changing only what the
spec names; collision and physics stay as they are. The code is `simfoundry/utils/splat_background.py`. Everything here was
checked against that module, the runner and the tests at commit `938df9a`. Notes cited as `E<n>`
(`background_video/bgswap/DESIGN_ERRATA_ko.md`) are in Korean and live outside this repository; numbers taken from them carry
their date and seed.

The options are **off by default**: without `--visual-background` the runner records the run as it did before
(`tests/test_tofu_policy_visual_background.py` checks this on the runner's syntax tree). The final reflectance of the
counter patch is **not decided yet**; the numbers in the examples are placeholders.

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

Unknown keys are refused. Relative `usdz` and `pose_json` paths are taken from the spec file's folder.

`counter_patch` is an object whose four keys are all required:

| key | rule |
|---|---|
| `rects` | 1 to 8 rectangles `[xmin, xmax, ymin, ymax]`; each number `abs <= 5` m; `xmin < xmax`, `ymin < ymax`; each side at least 1 mm (`MIN_RECT_SIDE_M`) |
| `z_top_m` | finite, `abs <= 0.02` m; the top face of every box |
| `thickness_m` | from `1e-4` to `0.05` m inclusive (`MIN_THICKNESS_M`, `_PATCH_MAX_THICKNESS_M`) |
| `rgb` | 3 numbers in `[0, 1]` |

Sides are compared after rounding to a picometre. Rectangles may overlap.

Example (paths and reflectance are placeholders; it loads once the paths name real files, checked with stand-in files):

```json
{
  "schema": 1,
  "name": "PD",
  "usdz": "/path/to/PD_slab4_bg.usdz",
  "pose_json": "/path/to/align_v4b.json",
  "dz_m": 0.0,
  "floor_plane_visible": false,
  "remove": ["mesh_background_0", "iter_7"],
  "kit_settings": {},
  "color": null,
  "ccm": [[0.46, 0.0, 0.0, 0.0], [0.0, 0.46, 0.0, 0.0], [0.0, 0.0, 0.47, 0.0]],
  "counter_patch": {
    "rects": [[-0.62, 0.05, -0.62, -0.10]],
    "z_top_m": -0.0005,
    "thickness_m": 0.004,
    "rgb": [0.25, 0.25, 0.25]
  }
}
```

## Running it

```bash
python scripts/cutting/tofu_policy_episodes.py --scene SCENE_DIR --out OUT_DIR --visual-background SPEC.json
```

`--out` is the only required argument; `--scene` defaults to `Data/tofu_chefd`. In `main`, the swap runs after the scene edit
and before the sensors. `main` then applies `kit_settings` with `og.app.set_setting` (a failure is recorded, and the run goes
on), builds the environment with the spec's `floor_plane_visible`, and, when the spec has a `ccm`, writes it to the splat and
renders three times.

`--probe-out DIR` (with `--probe-ticks`, default `0 25 50 75 99`) is a test-only probe. It saves the front and wrist frames of
those ticks as `DIR/ep<seed:05d>/t<tick:04d>_<camera>.npz`, with `rgb` and `seg_instance` (`CameraProbe`). The arrays come from
the same `get_obs()` read as the policy's images (`EpisodeRun.capture`); the probe never reads a camera again.

## What the scene and the record get

- Each name in `remove` is deleted from `init_info` and `object_registry`; `gs_background` is added to both, as a `USDObject`
  with the splat's path, scale and MD5 (`gs_background_entry`).
- Each box follows the splat, in rectangle order, in both sections, named `counter_patch_<i>`. A name the scene already has is
  refused.
- `setup.json` gets `scene_changes.visual_background` (name, `usdz`, `md5`, `removed`, pose, `scale`, `dz_m`,
  `floor_plane_visible`, `kit_settings`, and `kit_settings_applied` and `ccm_applied` when they run). The `ccm` and
  `counter_patch` keys appear only when the spec has them.
- `setup.json`'s `args` lists `visual_background`, and `probe_out` and `probe_ticks`, only when given (`setup_args`).

## The colour matrix (`ccm`)

A 3x4 matrix, written to the splat's prims as `omni:nurec:ccmR`, `ccmG` and `ccmB` (`CCM_ATTRIBUTES`). `apply_splat_ccm` refuses
a splat with none of the three attributes, and a prim with only some of them.

Why: a plain splat renders brighter than the training video. On the 71 training frames (2026-11-05 16:22, `E19`, no seed) the
mean ratio of OmniGibson to video was 1.279, 1.308 and 1.325 per channel. A linear matrix brought the ratio to 0.998-1.001 and
raised the PSNR from 14.13 dB to 18.30 dB, worst frame 15.20 dB (`E19`; repeated in `E27`, 17:30).

## Counter patch

Why: the splat draws nothing on a strip of counter in front of the cutting board, on the robot's side, and there the wrist
camera sees the sky (`E23`, 16:25).

What: each rectangle becomes an opaque, visual-only, fixed `PrimitiveObject` Cube named `counter_patch_<i>`
(`counter_patch_entry`). OmniGibson's default Cube is 1 m on a side (`PrimitiveObject.__init__` starts the extents at
`th.ones(3)`), so `scale` is the box size in metres, and `size` is not passed. The top face is at `z_top_m`; `rgba` is `rgb`
with alpha 1. The box's state (`counter_patch_state`) adds the keys OmniGibson writes for a Cube: `radius` and `height` set to
-1, and `size` set to 1.0.

OmniGibson's `PrimitiveObject._dump_state` writes these keys, and `_load_state` reads `size` for a Cube. A state without `size`
makes a scene restore raise `KeyError`; the keys were added in `aaa36a5` (`E30`, 18:53). Expected warning, once per box: `Custom
scale specified for primitive object, so ignoring radius, height, and size arguments!` It comes from `PrimitiveObject._post_load`
whenever a box has a scale. The seed-1000 run printed it five times, once per box (`E30`).

Inside the cleared slab the boxes show alone. Outside it, the splat's translucent top layer is drawn over them, which is why the
patch colour has to be calibrated on the seam step, not matched to the splat colour (`E29`, 18:25). Without the derived splat,
the fog layer also covers the footprints of objects on the counter (`E17`; `E25`). Make the derived file with
`splat_derive.py --clear-slab` (in `background_video/bgswap/cpu_checks/`, outside this repository) and point `usdz` at it.

## Known limits

- **Brightness and PSNR.** PSNR is 18.30 dB, below the 28 dB target (71 training frames, 17:30; `E27`). A per-channel curve gives
  22.97 dB on the same frames. The causes are not separated: tone curve, splat texture, antialiasing, and dark spots on the sink
  and hob faces (`E24`).
- **Front seam.** At the tentative reflectance 0.244 the judged front border has a seam step of 12/255 against a limit of 6 (22
  jobs of episode seed 1000, 2026-11-05 17:56; `E29`, 18:25).
- **Probe wrist frame at tick 0** shows the front camera's view (`E28`, about 18:15).
- **Probe segmentation.** OmniGibson adds `seg_semantic` whenever `seg_instance` is requested (`E22`).
- **Replaying the 5c probe** is not solved (`E25`).

## Tests

From the repository root, with the `simfoundry` environment's Python:

```bash
python -B -m pytest -p no:cacheprovider -q \
  tests/test_splat_background.py tests/test_tofu_policy_visual_background.py tests/test_tofu_policy_probe.py
```

At `938df9a` this gives `315 passed, 2 skipped`. The skips are `test_usdz_is_a_nurec_volume_whose_extent_is_the_ply_aabb` and
`test_apply_on_a_real_usd_stage_writes_the_matrix_and_reads_it_back`, which need `pxr` (not in the `simfoundry` environment). The
two runner test files compare the runner's syntax tree with the base commit named by their `BASE_COMMIT` (`5b05fe3`, `b76779e`);
they skip when that commit is not in the clone.
