# Food cutting: MPM solver fixes and robot episodes for ten foods

Branch `research4/food-cutting`. This folder generates robot demonstrations of a Franka arm that holds a knife and
slices vegetables, processed foods and raw meat in OmniGibson (Isaac Sim 5.1), with the food simulated by the MPM
cutting solver in `simfoundry/cutting/mpm.py` in the same process. The episodes are written in the format of
`scripts/cutting/tofu_policy_episodes.py` and convert to LeRobot with `scripts/cutting/policy/npz_to_lerobot.py`.

Code comments and help texts are in Korean (they come from the author's research repository); this README is the English entry point.
What was done and why, with the results (more foods, knife adhesion, the knife force cap, unsticking slices from the knife), is in Korean in
[docs/FOOD_CUTTING_PROGRESS_ko.md](../../../docs/FOOD_CUTTING_PROGRESS_ko.md).

## What is in the branch

| Path | What |
|---|---|
| `simfoundry/cutting/mpm.py` | Solver options, **all off by default** (the default output is bit-identical to `integration`, see below): board contact only where the food touches the board, no contact between pieces across the blade, the knife side decided by particles, contact only where pieces touch, pushing pieces along the knife face, slip in the kerf under the edge, an earlier pinch to the board, re-deciding sides ahead of the crack, a per-particle modulus scale (green onion leaves), a skin toughness band (sausage casing), and a tough bottom layer that needs sawing (raw pork) |
| `og_veg_cut.py` | Ten foods (cucumber, zucchini, carrot, potato, garlic, half onion, green onion, luncheon meat, cased sausage, raw pork): materials, shapes filled with particles, knife paths (slices, then julienne of stacked slices), the solver set-up. Runs alone without OmniGibson (`--no_og`); the episode runner reuses its set-up part |
| `warp_fixes.py` | Wrappers around the solver step: a holding hand, damping beside the knife, knife adhesion (a water film: suction and shear), bounce damping, toppling slices that stand on the spine |
| `force_cap.py` | Knife force cap: the arm slows and stalls when the resisting force reaches the cap (`elrobot_console` 12.1 N, `elrobot_motor` 18.6 N or a number) |
| `food_robot_episodes.py` | The episode runner: cut / place / move tasks, slices stuck to the knife (tap on the board, scrape on a bowl wall, wipe with a helper arm, shake), pushing slices away, the success judge per slice (separated + thickness within 2 mm, column-median thickness) |
| `food_handoff.py` | Cut pieces to PhysX rigid bodies; a sliding glue joint for slices stuck to the knife; bowl and tray USDs |
| `food_mesh_prep.py`, `food_mesh_lib.py`, `food_mesh_view.py` | The food drawn as meshes (skin colours, cut-face textures) that follow the particles, then the rigid pieces |
| `food_look.py`, `make_look_textures.py` | Scene looks: `--look real` (white plastic board, granite counter, tiled walls, ceiling light with shadows, an indoor photo dome) and `--look random` (boards, counters, walls, lights, dome and front camera drawn per episode). Rendering only |
| `make_veg_meshes.py`, `veg_shape_prep.py`, `make_food_shapes.py`, `food_regions.py` | Food shapes (analytic, no scans) and the tough regions of raw pork |
| `../policy/food_npz_to_lerobot.py` | The LeRobot converter with our task sentences and an optional 10th action column for the gripper (`--gripper_action`) |

## Set-up

Use the repository's Isaac Sim / OmniGibson environment. The mesh view also needs `pip install manifold3d`
(Open3D is optional). From the repository root:

```bash
export PYTHONPATH=$PWD
# shapes (written to Data/food/; Data/ is not committed)
python scripts/cutting/food/make_veg_meshes.py                       # Data/food/meshes/{cucumber,carrot,zucchini,potato}.obj
for v in cucumber carrot zucchini; do
  python scripts/cutting/food/veg_shape_prep.py --veg $v --keep 0.040 --trim_end 0.002 --out Data/food/shapes/${v}_k40t
done
python scripts/cutting/food/veg_shape_prep.py --veg potato --keep 0.030 --trim_end 0.012 --out Data/food/shapes/potato_k30t
python scripts/cutting/food/make_food_shapes.py                      # garlic_clove, onion_half, green_onion_v3, luncheon_block, sausage_cased, pork_slab10, ...
# scene looks (optional): textures, and Poly Haven indoor HDRIs (CC0) as JPG in Data/food/hdri/ with a LIST.txt
python scripts/cutting/food/make_look_textures.py                    # Data/food/looks/textures/
```

## Running

The common solver set-up used for all results (one line; the food arguments follow):

```bash
COMMON="--no_og --dx 0.00167 --side_margin 0.045 --knife_height 0.05 --knife_side --blade_split --adhesion film3 --adh_release none \
 --bounce_damp 2000 --top_s 0.6 --gap_contact 1.0 --face_tan --knife_bevel 0.045 --release_outer --reside --pinch_extra 0.0035 \
 --end_damp 50000 --spine_topple 3 --knife_friction_post 0.1 --adh_anchor --adh_anchor_hi 0.001 --kerf_slip --kerf_depth 30 --adh_tau 30"
PLAN="--plan 3,3,3,3 --hold_follow 0.009 --rest_s 0.5 --lift 0 --lift_clear 0.015"
SAW="--saw_amp 0.006366 --saw_hz 1.5"   # sawing while pressing (the knife moves along its edge)

# cut four 3 mm slices, then move a piece to the plate (knife force capped at 18.6 N), the realistic look
OMNIGIBSON_HEADLESS=1 python scripts/cutting/food/food_robot_episodes.py --out results/food/cucumber --episodes 2 --seed0 0 \
  --task_kind move --shift 0.01 --yaw_deg 15 --wet_lognormal 0.837 0.228 --look real \
  -- --veg cucumber $COMMON --shape Data/food/shapes/cucumber_k40t --wet 0.837 $PLAN $SAW --force_cap elrobot_motor

# random looks and a thickness drawn per episode (3, 5 or 8 mm; the task sentence follows)
... --task_kind cut --look random --thick_choices 3,5,8 -- --veg cucumber $COMMON --shape Data/food/shapes/cucumber_k40t $PLAN $SAW --force_cap elrobot_motor

# LeRobot (success only; --all keeps failures), with the gripper column for the place/move tasks
python scripts/cutting/policy/food_npz_to_lerobot.py --src results/food/cucumber/move --root datasets/food_move_cucumber --gripper_action
```

Food arguments that differ from the cucumber line: zucchini `--wet 1.004 --Gc 125 --side_margin 0.050`, potato
`--wet 1.256 --top_margin 0.11 --side_margin 0.058`, garlic `--wet 1.256`, onion `--no_layers --wet 0.5 --top_margin 0.12`,
green onion `--dx 0.00125 --region_E_mul 0.4 --region_xmin 0.016`, luncheon `--plan 5,5,5,5`, sausage `--plan 5,5,5,5`
(casing defaults on), raw pork `--wet 0.5 --knife_friction 0.5` (tough bottom layer defaults on).
Tasks: `cut` (cut only), `move` (cut, then unstick slices from the knife — `--unstick tap` by default — and move a piece),
`place` (put the whole food on the board). Speed: about 7–13 min per cut episode on an 8 GB laptop GPU.

## Checks done for this branch

- **Default output**: a one-stroke tofu cut (the set-up of `tests/test_solver_instrumentation.py`, CPU) gives bit-identical
  particle positions, velocities, knife forces and piece labels with `integration`'s `mpm.py` and with this one.
  (The board contact option was on by default in the author's copy; it is off here and `warp_fixes.py` turns it on.)
- `tests/test_subgrid_blade_faces.py` calls `_grid_update` directly; the new arguments are passed with their off values.
- New `tests/test_food_cutting_tools.py` (CPU): slice thickness judge, thickness plans, the gripper column, looks, force cap, hull fallback.
- Food mesh pieces made with `food_mesh_lib.py` are identical to the original renderer's.

## Known limits

- The knife is held by assisted grasping with its gravity off (as in the tofu runner); the food is held by an invisible box in the solver.
- Raw pork move fails when the chooser picks a lying slice; garlic pieces are too low for this gripper.
- Look values are first guesses from a kitchen video; tofu is not among the ten foods yet.
