# Tofu-cutting policy tooling

Everything between the episode runner (`scripts/cutting/tofu_policy_episodes.py`) and a trained LeRobot policy:
picking demonstrations, turning them into LeRobot datasets, training on a rented GPU, serving the policy back to the
simulator, and tabulating evaluations. The files sit beside the episode runner because they read its episode files
and speak its policy protocol; a change to either side has to change both.

Two tasks share the tooling: a **single cut** (one stroke through the tofu, two pieces) and **dicing** (K1 + K2 level
strokes in two directions, a grid of cubes).

## Flow

```
tofu_policy_episodes.py --mode expert|dice      scripted demonstrations: ep_<seed>.npz + ep_<seed>.json
        │
select_demos.py / dice_select.py                 one acceptance rule per task -> accepted.txt (npz paths)
        │
npz_to_lerobot.py / dice_to_lerobot.py           LeRobot v3 dataset (state, front + wrist images, 9-number action)
        │
lerobot_env.sh, train_act.sh / train_dice.sh     LeRobot 0.6.1 training on a GPU VM (pi05_lora_train.sh: π0.5 LoRA)
        │
policy_server.py                                 serves a checkpoint over a local socket
        │
tofu_policy_episodes.py --mode policy            the simulator asks the server for an action every 0.1 s
        │
eval_summary.py -> results_table.py              success rates with Wilson intervals, edge depth, failure kinds
```

## Episode files (written by the runner, read here)

- `ep_<seed>.npz`:
  - `images_front`, `images_wrist`: (T, 224, 224, 3) uint8, one per policy tick (10 Hz).
  - `state`: (T, 16). Columns 7-15 are the policy's state: two finger widths, then the hand position and quaternion
    in the robot base frame.
  - `cmd`: (T, 9), the commanded end-effector pose per tick: position, then the first two rotation columns.
    The dataset action at tick i is `cmd[i + 1]`.
  - `block`: the tofu box's centre (3), yaw and size (3).
  - Dicing adds `tick_cuts_done`, the finished-cut count per tick.
- `ep_<seed>.json`: the seed, the random draw and the outcome. Dicing adds `dice`, `dice_basic`, `dice_strict`,
  `dice_plan` and `food_params` (mass, thickness and density with their sources).

## Datasets

```bash
python select_demos.py --runs <run dir> [...] --out <sel dir>        # single cut
python dice_select.py <sel dir> <run dir> [...]                       # dicing (strict only)
python npz_to_lerobot.py --list <sel dir>/accepted.txt --root <ds root> --kinds image --skip-ticks 5 --vcodec h264
python dice_to_lerobot.py --list <sel dir>/accepted.txt --root <ds root> --variant A   # base | A | B
```

The dicing variants differ only in their inputs, so one set of demonstrations serves all three:
- **A** adds the finished-cut count to the state.
- **B** adds the hand's state 0.5 s and 1 s ago and the wrist image 1 s ago. LeRobot's ACT takes one observation
  step, so the history goes in as extra inputs. `policy_server.py --dice-variant B` rebuilds the same inputs from what
  it is sent.

## Training (on a GPU VM, nothing here starts one)

```bash
bash lerobot_env.sh                                              # python 3.12, torch 2.11 cu128, lerobot 0.6.1
bash train_act.sh image <tag> <ds root> 30000 32 --dataset.image_transforms.enable=true
bash train_dice.sh A <tag> <ds root> 50000 32
```

- `LEROBOT_BIN`, `OUT_ROOT` and `MINIFORGE` override the Colab defaults (`/root/miniforge3`, `/content`).
- Single-cut ACT predicts 20 ticks and executes 10. Dicing ACT predicts 40 and executes 20, since one cut takes
  about 2.5 s.
- `pi05_lora_train.sh` fine-tunes π0.5 with LoRA r=16, batch 8 and mean/std normalisation. The default quantile
  normalisation blows up on near-constant state dimensions.
  - The PaliGemma tokenizer is a gated repository. Put it in `HF_HUB_CACHE` beforehand; the script runs offline and
    never needs a token on the VM.

## Serving and evaluating

```bash
python policy_server.py --ckpt <checkpoint>/pretrained_model --port 6000 --device cuda [--dice-variant A|B]
python scripts/cutting/tofu_policy_episodes.py --mode policy --policy-port 6000 --seeds 1000 50 --episode-s 10 \
    <scene arguments, the same as for the demonstrations> --out <eval dir>
```

**Protocol.** `multiprocessing.connection` on 127.0.0.1, authkey `b"tofu"`.
- `{"cmd": "reset"}` starts an episode.
- `{"cmd": "act", "state", "env_state", "front", "wrist"[, "cuts_done"]}` returns a packed float32 array of 9 numbers.
  Arrays travel as `(dtype.str, shape, bytes)`.

**Smoke tests** need no simulator:
- `server_check.py <port>` checks a running server.
- `dice_server_smoke.py <checkpoint> A|B` starts its own CPU server and checks 12 ticks.

**Evaluation seeds** are 1000-1049, never used for demonstrations.

**Options of the runner that change what is measured**, so report them with the result:
- `--edge-floor-mm X --floor-ref solver|collider`: a rule that keeps the commanded edge X mm above a known board
  height. It is not learnt; report it as a rule.
- `--n-action-steps`: executed actions per predicted chunk, set on the server.
- `--action-mode delta`: the actions are offsets from the current pose.

**Tables.** Run `EVAL_SUMMARY_OUT=<file> python eval_summary.py "<label>"=<merged eval dir>[+<repeat dir>] ...`,
then `python results_table.py <file>`.

## Not here

Checkpoints, datasets, demonstrations, evaluation results, VM session and SSH configuration, and the host-specific
launch scripts. `MANIFEST.md` lists what came in and what stayed out.
