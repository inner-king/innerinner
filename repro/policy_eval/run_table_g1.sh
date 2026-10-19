#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Table G.1, simulation side (repro/README.md, chapter 18): the public pi0-DROID and pi0.5-DROID joint-position
# checkpoints on the authors' five DROID scenes, through the repository's own eval stage
# (scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py, unmodified) and openpi's policy server; and
# DreamZero-DROID through DreamZero's own server (chapter 19.4).
# Runs on the Colab runtime set up by setup_colab.sh (and setup_dreamzero.sh for DreamZero). Per-task settings:
# repro/policy_eval/table_g1.yaml.
#
#   servers POLICY...              start the policies' servers (ports in table_g1.yaml) and warm them up: openpi's,
#                                  or DreamZero's for a policy of `kind: dreamzero` (setup_dreamzero.sh)
#   stop POLICY...                 stop them
#   init [TASK...]                 25 fixed starts per task (sample_init_states.py), in parallel
#   eval POLICY TASK [N] [TAG]     one run of N episodes (default 25); TAG names the run (default "main");
#                                  SCENE_JSON=<path under assets/> evaluates a copy of the scene instead,
#                                  PROMPT=<text> another instruction, SENSORS_CFG=<name> another camera config,
#                                  TONEMAP_OP=<op> another RTX tone mapper (1 = linear; default Isaac Sim's ACES)
#                                  GRASPING_MODE=physical|assisted overrides the scene robot's grasping
#                                  (s15_eval.grasping_mode; the scenes' own is assisted)
#                                  BASE_CAMERA_1=<external camera name> is what the policy gets as
#                                  exterior_image_1_left (s15_eval.base_camera_1_name; default external_cam0_left)
#   all POLICY [JOBS] [N] [TAG]    every task for one policy, JOBS eval processes at a time (default 2)
#   summary [TAG]                  summarize.py over the runs
#
# On the 24 GB L4 one server runs at a time: pi0's tasks, then pi0.5's.
#   run_table_g1.sh init && run_table_g1.sh servers pi0 && run_table_g1.sh all pi0 && run_table_g1.sh stop pi0
#   run_table_g1.sh servers pi05 && run_table_g1.sh all pi05 && run_table_g1.sh stop pi05 && run_table_g1.sh summary
#
# Output under ${EVAL_ROOT:-/content/eval}: init_states/, out/<task>/s15_eval_<policy>_<tag>/..., logs/.
set -uo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
# SPEC=<file> runs another spec of the same schema. Made absolute: some lookups run inside a run's work directory.
SPEC="$(readlink -f "${SPEC:-${HERE}/table_g1.yaml}")"
EVAL_ROOT="${EVAL_ROOT:-/content/eval}"
PY="${SIMFOUNDRY_PY:-/root/miniforge3/envs/simfoundry/bin/python}"
OPENPI_DIR="${OPENPI_DIR:-/content/openpi}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/content/openpi_cache}"
CKPT_ROOT="${OPENPI_DATA_HOME}/openpi-assets/checkpoints"
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES
# JAX otherwise preallocates 75% of the card, which the Isaac Sim processes need. The weights are about
# 6 GB in bf16; 0.4 of the L4's 23 GB leaves room for activations.
XLA_FRACTION="${XLA_FRACTION:-0.4}"
mkdir -p "${EVAL_ROOT}/logs" "${EVAL_ROOT}/init_states" "${EVAL_ROOT}/out"

spec() {  # spec <python expression over s (the spec dict)>
  "${PY}" -c "import yaml,sys; s=yaml.safe_load(open('${SPEC}')); print(eval(sys.argv[1]))" "$1"
}
# TASKS="a b" in the environment narrows init/all to those tasks.
if [[ -n "${TASKS:-}" ]]; then TASKS=(${TASKS}); else TASKS=($(spec "' '.join(s['tasks'])")); fi

install_tasks() {
  # The eval stage reads the task YAML from scripts/cfg/task itself (resolve_task_config_path), not only
  # through Hydra's search path, so the configs are copied into that tree for the run: this directory's,
  # and those next to another SPEC (repro/policy_eval_ours/tasks/ for ours.yaml).
  local d dirs=("${HERE}")
  d="$(cd "$(dirname "${SPEC}")" && pwd)"
  [[ "${d}" != "${HERE}" && -d "${d}/tasks" ]] && dirs+=("${d}")
  mkdir -p "${REPO}/scripts/cfg/task/policy_eval"
  for d in "${dirs[@]}"; do cp "${d}"/tasks/*.yaml "${REPO}/scripts/cfg/task/policy_eval/"; done
}

# A policy's `kind` is the eval stage's s15_eval.policy, and says which server serves it: openpi (the default)
# or dreamzero, DreamZero's own server (socket_test_optimized_AR.py) in the env setup_dreamzero.sh builds.
kind() { spec "s['policies']['$1'].get('kind', 'openpi')"; }
DZ_PY="${DZ_PY:-/root/miniforge3/envs/dreamzero/bin/python}"
DZ_DIR="${DZ_DIR:-/content/dreamzero}"
export HF_HOME="${HF_HOME:-/content/hf_home}"

port_open() { "${PY}" -c "import socket,sys; socket.create_connection(('localhost', int(sys.argv[1])), 2)" "$1" 2> /dev/null; }

dreamzero_server() {  # dreamzero_server POLICY: start DreamZero's server on the policy's port and warm it up
  local p="$1" port ckpt
  port=$(spec "s['policies']['${p}']['port']")
  ckpt=$(spec "s['policies']['${p}']['checkpoint']")
  # Its RoboArena websocket server answers no HTTP health check: an open port means the model is loaded.
  if port_open "${port}"; then
    echo "[servers] ${p} already up on ${port}"
  else
    # One process: ip_size 1 runs both classifier-free-guidance branches in turn, where the authors' two GPUs
    # run one each. DiT caching as the DreamZero README serves it.
    (cd "${DZ_DIR}" && setsid nohup "${DZ_PY}" -m torch.distributed.run --standalone --nproc_per_node=1 \
      socket_test_optimized_AR.py --port "${port}" --enable-dit-cache --model-path "/content/dz_ckpt/${ckpt}" \
      > "${EVAL_ROOT}/logs/server_${p}.log" 2>&1 < /dev/null &)
    echo "[servers] ${p}: DreamZero /content/dz_ckpt/${ckpt} on ${port}"
    # Loading 46 GB of weights (and the Wan2.1 encoders it reads first) takes minutes.
    for _ in $(seq 360); do port_open "${port}" && break; sleep 5; done
    port_open "${port}" || { echo "[servers] ${p} did not come up, see ${EVAL_ROOT}/logs/server_${p}.log"; return 1; }
  fi
  # The first calls compile (minutes each, one per new context length); do them here rather than inside an
  # episode, through the eval stage's own client, two episodes' worth.
  "${PY}" "${HERE}/ping_dreamzero.py" --port "${port}" --n "${DZ_WARMUP_CALLS:-8}" --episodes 2
}

servers() {
  local p port cfg ckpt
  for p in "$@"; do
    if [[ $(kind "${p}") == dreamzero ]]; then dreamzero_server "${p}" || return 1; continue; fi
    port=$(spec "s['policies']['${p}']['port']")
    cfg=$(spec "s['policies']['${p}']['config']")
    ckpt=$(spec "s['policies']['${p}']['checkpoint']")
    if curl -fsS "http://localhost:${port}/healthz" > /dev/null 2>&1; then echo "[servers] ${p} already up on ${port}"; continue; fi
    (cd "${OPENPI_DIR}" && XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_FRACTION}" setsid nohup uv run scripts/serve_policy.py \
      --port "${port}" policy:checkpoint --policy.config="${cfg}" --policy.dir="${CKPT_ROOT}/${ckpt}" \
      > "${EVAL_ROOT}/logs/server_${p}.log" 2>&1 < /dev/null &)
    echo "[servers] ${p}: ${cfg} ${CKPT_ROOT}/${ckpt} on ${port}"
  done
  for p in "$@"; do
    [[ $(kind "${p}") == dreamzero ]] && continue
    port=$(spec "s['policies']['${p}']['port']")
    for _ in $(seq 180); do curl -fsS "http://localhost:${port}/healthz" > /dev/null 2>&1 && break; sleep 2; done
    # The first call compiles (about 30 s); do it here rather than inside an episode.
    "${PY}" "${HERE}/ping_policy.py" --port "${port}" --n 3 || return 1
  done
}

stop() {
  local p port
  for p in "$@"; do
    port=$(spec "s['policies']['${p}']['port']")
    if [[ $(kind "${p}") == dreamzero ]]; then
      pkill -f "socket_test_optimized_AR.py --port ${port} " && echo "[servers] stopped ${p} (${port})"
    else
      pkill -f "serve_policy.py --port ${port} " && echo "[servers] stopped ${p} (${port})"
    fi
  done
  sleep 5
}

init() {
  local t pids=()
  local tasks=("$@")
  [[ ${#tasks[@]} -gt 0 ]] || tasks=("${TASKS[@]}")
  install_tasks
  for t in "${tasks[@]}"; do
    # One Isaac Sim process per task, at most INIT_JOBS at a time (the L4 has 12 vCPUs and 52 GB RAM).
    while [[ $(jobs -rp | wc -l) -ge ${INIT_JOBS:-2} ]]; do sleep 5; done
    mkdir -p "${EVAL_ROOT}/init_states/work_${t}"
    (cd "${EVAL_ROOT}/init_states/work_${t}" && "${PY}" "${HERE}/sample_init_states.py" --spec "${SPEC}" --task "${t}" \
      --out "${EVAL_ROOT}/init_states/${t}.json" > "${EVAL_ROOT}/logs/init_${t}.log" 2>&1) &
    pids+=($!)
  done
  local rc=0 i
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "[init] ${tasks[$i]} failed, see ${EVAL_ROOT}/logs/init_${tasks[$i]}.log"; rc=1; }
  done
  for t in "${tasks[@]}"; do grep -a "^\[init_states\]" "${EVAL_ROOT}/logs/init_${t}.log" | tail -1; done
  return ${rc}
}

eval_one() {  # eval_one POLICY TASK [N] [TAG]
  local p="$1" t="$2" n="${3:-25}" tag="${4:-main}"
  local port ckpt scene task prompt timeout horizon work log
  port=$(spec "s['policies']['${p}']['port']")
  ckpt=$(spec "s['policies']['${p}']['checkpoint']")
  # SCENE_JSON (under assets/) replaces the task's scene, e.g. for an ablation on a copy of it.
  scene="${SCENE_JSON:-$(spec "s['tasks']['${t}']['scene']")}"
  task=$(spec "s['tasks']['${t}']['task']")
  # PROMPT and SENSORS_CFG replace the task's instruction and the external-camera config (ablations).
  prompt="${PROMPT:-$(spec "s['tasks']['${t}']['prompt']")}"
  local sensors="${SENSORS_CFG:-nv_franka_droid}"
  timeout=$(spec "s['tasks']['${t}']['timeout_s']")
  horizon=$(spec "s['tasks']['${t}']['execute_horizon']")
  local init="${EVAL_ROOT}/init_states/${t}.json"
  [[ -s "${init}" ]] || { echo "[eval] no init states for ${t}; run init first"; return 1; }
  if [[ ${n} -lt 25 ]]; then
    # The eval stage sets n_episodes to the number of init states it is given, so a shorter run needs a
    # shorter file: the first N starts.
    "${PY}" -c "import json,sys; d=json.load(open(sys.argv[1])); d['init_states']=d['init_states'][:int(sys.argv[2])]; json.dump(d, open(sys.argv[3],'w'))" \
      "${init}" "${n}" "${EVAL_ROOT}/init_states/${t}_first${n}.json"
    init="${EVAL_ROOT}/init_states/${t}_first${n}.json"
  fi
  install_tasks
  # Hydra runs in the working directory and writes a fixed-name log there, so each run gets its own.
  work="${EVAL_ROOT}/work/${t}_${p}_${tag}"
  log="${EVAL_ROOT}/logs/eval_${t}_${p}_${tag}.log"
  mkdir -p "${work}"
  echo "[eval] $(date +%T) ${p} ${t} n=${n} tag=${tag} -> ${log}"
  (cd "${work}" && "${PY}" "${REPO}/scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py" \
    --config-name=real2sim_cfg "task=policy_eval/${task}" \
    "root_dir=${EVAL_ROOT}/out" "scene_name=${t}" "s15_eval.out_dirname=s15_eval_${p}_${tag}" \
    "s15_eval.policy=$(kind "${p}")" s15_eval.host=localhost "s15_eval.port=${port}" "s15_eval.checkpoint=${ckpt}" \
    "s15_eval.scene_json=${scene}" "s15_eval.prompt='${prompt}'" "s15_eval.external_sensors_cfg=${sensors}" \
    "s15_eval.tonemap_op=${TONEMAP_OP:-null}" \
    ${GRASPING_MODE:+"++s15_eval.grasping_mode=${GRASPING_MODE}"} \
    ${BASE_CAMERA_1:+"s15_eval.base_camera_1_name=${BASE_CAMERA_1}"} \
    "s15_eval.timeout_s=${timeout}" "s15_eval.execute_horizon=${horizon}" "s15_eval.n_episodes=${n}" \
    "s15_eval.init_states_path=${init}" \
    s15_eval.save_video=true s15_eval.video_fps=10 "+s15_eval.video_resolution=[180,320]" \
    > "${log}" 2>&1)
  local rc=$?
  echo "[eval] $(date +%T) ${p} ${t} rc=${rc}: $(grep -a 'Success rate:' "${log}" | tail -1)"
  return ${rc}
}

all() {  # all POLICY [JOBS] [N] [TAG]: every task for one policy, JOBS at a time
  local p="$1" jobs="${2:-2}" n="${3:-25}" tag="${4:-main}" t
  # DreamZero's server keeps one episode's frames and KV cache, and a second client's session id resets them.
  [[ $(kind "${p}") == dreamzero ]] && jobs=1
  for t in "${TASKS[@]}"; do
    # Counts every eval process on the machine, including runs started outside this loop.
    while [[ $(pgrep -fc "stages/1_eval_policy_og_scene.py") -ge ${jobs} ]]; do sleep 20; done
    eval_one "${p}" "${t}" "${n}" "${tag}" &
    sleep 30  # stagger the Isaac Sim boots
  done
  wait
}

case "${1:-}" in
  servers) shift; servers "$@" ;;
  stop) shift; stop "$@" ;;
  init) shift; init "$@" ;;
  eval) shift; eval_one "$@" ;;
  all) shift; all "$@" ;;
  summary) shift; "${PY}" "${HERE}/summarize.py" --spec "${SPEC}" --root "${EVAL_ROOT}/out" --tag "${1:-main}" \
    --json "${EVAL_ROOT}/summary_${1:-main}.json" ;;
  *) echo "usage: $0 servers POLICY...|stop POLICY...|init [TASK...]|eval POLICY TASK [N] [TAG]|all POLICY [JOBS] [N] [TAG]|summary [TAG]" >&2
     exit 2 ;;
esac
