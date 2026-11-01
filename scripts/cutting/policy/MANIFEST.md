# MANIFEST: what came into `scripts/cutting/policy/` and why

Branch `research3/policy-tooling`, from `codelead/integration-base` (e05ceb3). Sources were the working copies under
`simfoundry_logs/research3_tofu_policy/` (outside the repository). The 2026-11-01 checks were all on the CPU:
- `server_check.py` against `policy_server.py` serving the (나) image ACT checkpoint: 3 actions of 9 numbers.
- `dice_server_smoke.py` with the dicing A and B checkpoints: 12 finite actions each.
- `npz_to_lerobot.py`: 2 episodes, 124 frames.
- `dice_to_lerobot.py --variant A`: 2 episodes, 522 frames.

| File | Bytes | From | Why it is here |
|---|---:|---|---|
| `README.md` | 5535 | new | The flow from demonstrations to an evaluated policy, the episode-file contract and the server protocol. |
| `MANIFEST.md` | (this file) | new | What came in, what stayed out. |
| `npz_to_lerobot.py` | 7160 | `colab/lerobot_tools/` | Single-cut episodes to a LeRobot v3 dataset (state and image kinds). Needed for D1's converter option. |
| `dice_to_lerobot.py` | 5090 | `colab/lerobot_tools/` | Dicing episodes to a LeRobot dataset; variants base, A (cut count) and B (history). |
| `policy_server.py` | 6476 | `scripts/` (2026-10-28) | Serves an ACT or π0.5 LoRA checkpoint to the runner (loaded by the checkpoint's policy type; Diffusion Policy is untested, see D2). This is the current copy; the older one in `colab/lerobot_tools/` (2026-10-27) lacks `--dice-variant`, `--n-action-steps` and PEFT loading. |
| `server_check.py` | 920 | `scripts/` | Wire check against a running server, no simulator. |
| `dice_server_smoke.py` | 2209 | new (the earlier `server_dice_smoke.py` was lost with `dice/plumbing_test/`) | Starts a CPU server with a dicing checkpoint and checks 12 ticks; exits non-zero on failure. |
| `select_demos.py` | 4926 | `scripts/` | Single-cut acceptance rule and contact sheet. |
| `dice_select.py` | 2062 | `scripts/` | Dicing acceptance (strict only), `accepted.txt` and summary. |
| `eval_summary.py` | 8844 | `scripts/` | Evaluation rows: Wilson intervals, edge depth, failure kinds. The hard-coded output path now comes from `EVAL_SUMMARY_OUT` (default `./eval_summary.json`). |
| `results_table.py` | 4957 | `scripts/` | Markdown table from `eval_summary.json`. |
| `lerobot_env.sh` | 877 | `colab/lerobot_only.sh` | LeRobot 0.6.1 training env (python 3.12, torch 2.11 cu128). Output root and miniforge path come from `OUT_ROOT` and `MINIFORGE`; the installer goes to /tmp. |
| `train_act.sh` | 874 | `colab/train_r3.sh` | Single-cut ACT recipe (chunk 20, execute 10). `LEROBOT_BIN` and `OUT_ROOT` replace the fixed Colab paths. |
| `train_dice.sh` | 1055 | `colab/train_dice.sh` | Dicing ACT recipe (chunk 40, execute 20), same overrides. |
| `pi05_lora_train.sh` | 2331 | `colab/pi05_train.sh` | π0.5 LoRA recipe with mean/std normalisation. It downloads only the public base model by a fixed revision and needs no token on the VM. |

Total about 53 KB in 15 files (16 with this one), all text, none over 9 KB.

## Left out, and why

| What | Why |
|---|---|
| `envs/`, `ckpt*/`, `ds_*/`, `datasets_lidar/`, demonstrations (`runs*/`, `demos*/`, `dice/demos/`), evaluation results (`evals*/`, `dice/evals/`), caches, `__pycache__/` | Data, not code; several GB. |
| `colab/ssh_config_*`, `colab/keepalive*`, `colab/STOP_*`, `colab/launch_*.sh`, `colab/pull_ckpts.sh`, `colab/*memtest*`, `colab/*normtest*`, `colab/*_finish.sh` | VM session, SSH and account handling, runtime names, or one-off tests tied to a session. |
| `colab/r3data.tar`, `colab/sf_code.tar.gz` | Upload bundles, 225 MB and 66 MB. |
| `scripts/eval_lidar.sh`, `gen_lidar.sh`, `gen_dice.sh`, `dice_eval_smoke.sh`, `depth_c*.sh`, `repeat_image_lidar.sh`, `auto_*.sh`, `start_policy_eval*.sh`, `lidar_*.sh` | Fixed home paths and a switch between two named lab machines. The commands they run are in README.md with placeholders. |
| `scripts/tofu_policy_episodes.*.py` backups, `colab/lerobot_tools/policy_server.py` (old), `*.before_*` | Superseded copies. The runner itself is already tracked at `scripts/cutting/tofu_policy_episodes.py`. |
| Plot scripts (`edge_depth_plot.py`, `rollout_sheet.py`, `status_figure.py`, `learning_curve.py`, ...) | They read result folders by fixed names and load a Korean font from a fixed path. They are report tooling, not part of the policy flow; they can come later if wanted. |
| PNG/GIF/MP4 under `docs/` or elsewhere | This machine has no git-lfs, and the repository stores media through LFS. |
