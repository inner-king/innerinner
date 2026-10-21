#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# pi0-DROID and pi0.5-DROID on this machine's 12 GB GPU (repro/README.md, chapter 30), the runs that went to Colab
# before. The episodes go through run_table_g1.sh's eval, so the eval stage, its settings and the output layout are
# the Colab runs'; only the paths and the server differ:
#
#   - openpi at commit 15a9616 in ${OPENPI_DIR} (uv env), checkpoints under ${OPENPI_DATA_HOME} as
#     setup_colab.sh lays them out. The server runs from the env's own python: `uv run` would re-sync the env.
#   - the simfoundry env imports openpi_client from openpi's source tree (PYTHONPATH), not an install.
#   - one server and one Isaac Sim process share the card. A fixed JAX pool does not fit both: at
#     XLA_PYTHON_CLIENT_MEM_FRACTION 0.6 (7.3 GB) Isaac Sim ran out of device memory on its first frame, and at 0.52
#     pi0.5 did not load (its peak while loading is above 6.4 GB). JAX's platform allocator hands back what loading
#     frees: pi0.5 then holds 6.6 GB, with the same 0.34 s per call. XLA_FRACTION=<f> uses a fixed pool instead.
#   - the second exterior camera is left out (SENSORS_CFG=nv_franka_droid_single_cam: the first camera at the same
#     pose and lens, rgb only): openpi's DROID policies read exterior_image_1_left and the wrist image only
#     (droid_policy.py), so the inputs are the same; the saved videos lose that view. Isaac Sim needs less memory.
#   - the server starts only when no other session's job holds the GPU.
#
#   server POLICY                      start pi0 or pi05 (ports in the spec) and warm it up
#   stop POLICY                        stop it
#   all POLICY N TAG SPEC:TASK...      one task after another, N episodes each, from <EVAL_ROOT>/init_states/<task>.json
#                                      (copied from INIT_FROM when missing); a run that dies is retried once
#
# GRASPING_MODE, SCENE_JSON, PROMPT, SENSORS_CFG, TONEMAP_OP, BASE_CAMERA_1, TIMEOUT_S and IMAGE_GAMMA pass
# through to run_table_g1.sh.
#   repro/policy_eval_ours/run_local_openpi.sh server pi05
#   GRASPING_MODE=physical repro/policy_eval_ours/run_local_openpi.sh all pi05 25 main \
#     repro/policy_eval_ours/ours.yaml:cup_in_bowl_ours_real
#   repro/policy_eval_ours/run_local_openpi.sh stop pi05
set -uo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
export SIMFOUNDRY_PY="${SIMFOUNDRY_PY:-${HOME}/miniforge3/envs/simfoundry/bin/python}"
export OPENPI_DIR="${OPENPI_DIR:-/home/inner/real2sim/tools/openpi}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
export EVAL_ROOT="${EVAL_ROOT:-${REPO}/results/policy_eval_ours/openpi_local}"
export PYTHONPATH="${REPO}:${OPENPI_DIR}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
# The same 25 starts the GR00T runs on this machine used.
INIT_FROM="${INIT_FROM:-${REPO}/results/policy_eval_ours/gr00t/init_states}"
export SENSORS_CFG="${SENSORS_CFG:-nv_franka_droid_single_cam}"
if [[ -n "${XLA_FRACTION:-}" ]]; then
  JAX_MEM=(XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_FRACTION}")
else
  JAX_MEM=(XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform)
fi
R="${REPO}/repro/policy_eval/run_table_g1.sh"
mkdir -p "${EVAL_ROOT}/logs" "${EVAL_ROOT}/init_states"

spec() {  # spec <spec yaml> <python expression over s>
  "${SIMFOUNDRY_PY}" -c "import yaml,sys; s=yaml.safe_load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$1" "$2"
}

busy() {  # other sessions start GPU jobs at any time, and leave the card between pipeline stages
  pgrep -f '^[^ ]*python[^ ]* .*(run_reconstruction.py|stages/[0-9]+_|replay_other_calls|replay_b7|scripted_pick_place)' \
    > /dev/null && return 0
  [[ $(curl -s -m 5 localhost:11434/api/ps | grep -c '"name"') -gt 0 ]] && return 0
  [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -ge 1000 ]]
}

server() {  # server POLICY
  local p="$1" spec_file="${REPO}/repro/policy_eval/table_g1.yaml" port cfg ckpt
  port=$(spec "${spec_file}" "s['policies']['${p}']['port']")
  cfg=$(spec "${spec_file}" "s['policies']['${p}']['config']")
  ckpt=$(spec "${spec_file}" "s['policies']['${p}']['checkpoint']")
  if curl -fsS "http://localhost:${port}/healthz" > /dev/null 2>&1; then echo "[openpi] ${p} already up on ${port}"; return 0; fi
  [[ -d "${OPENPI_DATA_HOME}/openpi-assets/checkpoints/${ckpt}" ]] || { echo "[openpi] no checkpoint ${ckpt}"; return 1; }
  busy && echo "[openpi] $(date +%T) waiting for the GPU"
  while busy; do sleep 60; done
  (cd "${OPENPI_DIR}" && env "${JAX_MEM[@]}" setsid nohup .venv/bin/python \
    scripts/serve_policy.py --port "${port}" policy:checkpoint --policy.config="${cfg}" \
    --policy.dir="${OPENPI_DATA_HOME}/openpi-assets/checkpoints/${ckpt}" \
    > "${EVAL_ROOT}/logs/server_${p}.log" 2>&1 < /dev/null &)
  for _ in $(seq 300); do curl -fsS "http://localhost:${port}/healthz" > /dev/null 2>&1 && break; sleep 2; done
  # The first call compiles; do it here rather than inside an episode.
  "${SIMFOUNDRY_PY}" "${REPO}/repro/policy_eval/ping_policy.py" --port "${port}" --n 3 \
    || { echo "[openpi] ${p} did not answer, see ${EVAL_ROOT}/logs/server_${p}.log"; return 1; }
  echo "[openpi] $(date +%T) ${p} up on ${port}; GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

stop() {  # stop POLICY
  local port
  port=$(spec "${REPO}/repro/policy_eval/table_g1.yaml" "s['policies']['$1']['port']")
  pkill -f "serve_policy.py --port ${port} " && echo "[openpi] stopped $1 (${port})"
  sleep 5
}

all() {  # all POLICY N TAG SPEC:TASK...
  local p="$1" n="$2" tag="$3" item sp t; shift 3
  for item in "$@"; do
    sp="$(readlink -f "${item%%:*}")"; t="${item#*:}"
    [[ -s "${EVAL_ROOT}/init_states/${t}.json" ]] || cp "${INIT_FROM}/${t}.json" "${EVAL_ROOT}/init_states/" || return 1
    SPEC="${sp}" "${R}" eval "${p}" "${t}" "${n}" "${tag}" && continue
    echo "[eval] $(date +%T) ${t} failed; retrying once in 2 minutes"; sleep 120
    SPEC="${sp}" "${R}" eval "${p}" "${t}" "${n}" "${tag}"
  done
}

case "${1:-}" in
  server) shift; server "$@" ;;
  stop) shift; stop "$@" ;;
  all) shift; all "$@" ;;
  *) echo "usage: $0 server POLICY|stop POLICY|all POLICY N TAG SPEC:TASK..." >&2; exit 2 ;;
esac
