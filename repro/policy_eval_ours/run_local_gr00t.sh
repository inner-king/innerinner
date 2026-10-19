#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# GR00T N1.6-DROID, the third zero-shot policy of Table G.1, on this machine's 12 GB GPU (repro/README.md,
# chapter 19). Table G.1's per-task Pearson r is taken across the policies run on a task; with pi0 and
# pi0.5 alone it is +-1 by construction, so a third public checkpoint is what makes it a measurement.
#
# The policy runs as Isaac-GR00T's own server (commit e29d8fc, the one Isaac Lab Arena pins for
# nvidia/GR00T-N1.6-DROID) in that repository's uv env; the episodes run through the repository's eval stage
# (scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py, unmodified) with s15_eval.policy=gr00t,
# whose Gr00tClient detects the N1.6 modality config. Task settings come from the same specs as the pi0/pi0.5
# runs: repro/policy_eval/table_g1.yaml for the authors' scenes, repro/policy_eval_ours/ours.yaml for ours.
#
#   server                       start the GR00T server on ${GR00T_PORT:-5555} and wait until it answers
#   stop                         stop it
#   eval SPEC TASK [N] [TAG]     one run of N episodes (default 25) from <EVAL_ROOT>/init_states/<task>.json
#
# Setup (once): Isaac-GR00T at e29d8fc in ${GR00T_DIR}, `uv sync --python 3.10 --no-install-package flash-attn`,
# then the prebuilt flash-attn 2.8.0.post2 wheel for torch 2.7 / cxx11abi TRUE / cp310 and
# `uv pip install --no-deps -e .` (no nvcc here to build the pinned 2.7.4.post1; the GR00T README tested 2.8.0.post2).
# GRASPING_MODE=physical|assisted overrides the scene robot's grasping (s15_eval.grasping_mode), as in
# run_table_g1.sh.
# The eval stage imports openpi_client unconditionally (through simfoundry/policies/openpi.py and dreamzero.py); a
# GR00T run never calls it, so an import-only stub stands in for it instead of installing the client into the
# shared simfoundry env. The script writes one under EVAL_ROOT/work/; OPENPI_STUBS names another directory.
set -uo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
PY="${SIMFOUNDRY_PY:-${HOME}/miniforge3/envs/simfoundry/bin/python}"
# GR00T_VERSION=n17 serves GR00T N1.7-DROID instead (README 30.8): repro/policy_eval_ours/gr00t_n17_server.py, run in an
# Isaac-GR00T checkout at the N1.7 release (1a1837f, its own uv env), on port 5556; its runs are s15_eval_gr00t17_<tag>,
# which correlate.py reads as Table G.1's GR00T N1.7 column.
GR00T_VERSION="${GR00T_VERSION:-n16}"
if [[ ${GR00T_VERSION} == n17 ]]; then
  GR00T_DIR="${GR00T_DIR:-/home/inner/real2sim/tools/Isaac-GR00T-n17}"
  GR00T_PORT="${GR00T_PORT:-5556}"
  GR00T_MODEL="${GR00T_MODEL:-/data/models/gr00t/GR00T-N1.7-DROID}"
  GR00T_NAME=gr00t17 GR00T_CHECKPOINT=gr00t_n17_droid
else
  GR00T_NAME=gr00t GR00T_CHECKPOINT=gr00t_n16_droid
fi
GR00T_DIR="${GR00T_DIR:-/home/inner/real2sim/tools/Isaac-GR00T}"
GR00T_PORT="${GR00T_PORT:-5555}"
# A local copy: this link stalls on large Hugging Face files, so the shards were fetched with curl -C -.
GR00T_MODEL="${GR00T_MODEL:-/data/models/gr00t/GR00T-N1.6-DROID}"
EVAL_ROOT="${EVAL_ROOT:-${REPO}/results/policy_eval_ours/gr00t}"
# Every openpi_client module the eval stage imports at load time; dreamzero.py added msgpack_numpy.
if [[ -z "${OPENPI_STUBS:-}" ]]; then
  OPENPI_STUBS="${EVAL_ROOT}/work/openpi_stubs"
  mkdir -p "${OPENPI_STUBS}/openpi_client"
  for m in __init__ image_tools websocket_client_policy msgpack_numpy; do
    echo '"""Import-only stub for a GR00T run (run_local_gr00t.sh): never called."""' \
      > "${OPENPI_STUBS}/openpi_client/${m}.py"
  done
fi
# As run_table_g1.sh: no viewer. Next to the 6.8 GB server, the viewer's render products took Serve Fruits
# (the busiest scene) past 12 GB and the renderer segfaulted out of device memory.
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES
# EXTERNAL_SENSORS=nv_franka_droid_single_cam keeps external_cam0_left (same pose; the policy's exterior
# image) and drops the second exterior camera, which only goes into the videos, and the depth outputs.
# Serve Fruits needs it here: with both cameras its renderer ran out of the 12 GB next to the server.
mkdir -p "${EVAL_ROOT}/logs" "${EVAL_ROOT}/out" "${EVAL_ROOT}/work"

spec() {  # spec <spec yaml> <python expression over s>
  "${PY}" -c "import yaml,sys; s=yaml.safe_load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$1" "$2"
}

server_up() {
  # PolicyClient's REQ socket has no receive timeout, so a probe of a port nobody serves blocks for good:
  # the hard kill bounds it.
  timeout -s KILL 10 "${PY}" - "${GR00T_PORT}" <<'EOF' > /dev/null 2>&1
import sys
sys.path.insert(0, ".")
from simfoundry.policies.gr00t_utils import PolicyClient
sys.exit(0 if PolicyClient(host="localhost", port=int(sys.argv[1])).ping() else 1)
EOF
}

gpu_idle() {  # other sessions start GPU jobs at any time; take the card only when nothing else holds it
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [[ ${used} -lt ${GPU_IDLE_MIB:-1000} ]]
}

server() {
  if (cd "${REPO}" && server_up); then echo "[gr00t] server already up on ${GR00T_PORT}"; return 0; fi
  gpu_idle || echo "[gr00t] $(date +%T) waiting for the GPU to be free"
  until gpu_idle; do sleep 30; done
  if [[ ${GR00T_VERSION} == n17 ]]; then
    (cd "${GR00T_DIR}" && setsid nohup .venv/bin/python "${REPO}/repro/policy_eval_ours/gr00t_n17_server.py" \
      --model-path "${GR00T_MODEL}" --port "${GR00T_PORT}" > "${EVAL_ROOT}/logs/server_${GR00T_NAME}.log" 2>&1 < /dev/null &)
  else
    (cd "${GR00T_DIR}" && setsid nohup .venv/bin/python gr00t/eval/run_gr00t_server.py --embodiment-tag OXE_DROID \
      --use_sim_policy_wrapper --model-path "${GR00T_MODEL}" --device cuda --host 127.0.0.1 \
      --port "${GR00T_PORT}" > "${EVAL_ROOT}/logs/server_gr00t.log" 2>&1 < /dev/null &)
  fi
  for _ in $(seq 180); do (cd "${REPO}" && server_up) && { echo "[gr00t] server up on ${GR00T_PORT}"; return 0; }; sleep 5; done
  echo "[gr00t] server did not come up, see ${EVAL_ROOT}/logs/server_gr00t.log"; return 1
}

stop() { pkill -f "(run_gr00t_server|gr00t_n17_server).py .*--port ${GR00T_PORT}" && echo "[gr00t] stopped"; sleep 5; }

eval_one() {  # eval_one SPEC TASK [N] [TAG]
  local sp="$1" t="$2" n="${3:-25}" tag="${4:-main}"
  local scene task prompt timeout horizon init work log
  # SCENE_JSON (under assets/) replaces the task's scene, as in run_table_g1.sh (e.g. use_background.py's copies).
  scene="${SCENE_JSON:-$(spec "${sp}" "s['tasks']['${t}']['scene']")}"
  task=$(spec "${sp}" "s['tasks']['${t}']['task']")
  prompt=$(spec "${sp}" "s['tasks']['${t}']['prompt']")
  timeout=$(spec "${sp}" "s['tasks']['${t}']['timeout_s']")
  horizon=$(spec "${sp}" "s['tasks']['${t}']['execute_horizon']")
  init="${EVAL_ROOT}/init_states/${t}.json"
  [[ -s "${init}" ]] || { echo "[eval] no init states for ${t} at ${init}"; return 1; }
  if [[ ${n} -lt 25 ]]; then
    "${PY}" -c "import json,sys; d=json.load(open(sys.argv[1])); d['init_states']=d['init_states'][:int(sys.argv[2])]; json.dump(d, open(sys.argv[3],'w'))" \
      "${init}" "${n}" "${EVAL_ROOT}/init_states/${t}_first${n}.json"
    init="${EVAL_ROOT}/init_states/${t}_first${n}.json"
  fi
  # The eval stage reads the task YAML from scripts/cfg/task; the copy is for this run only (untracked).
  mkdir -p "${REPO}/scripts/cfg/task/policy_eval"
  cp "$(dirname "${sp}")/tasks/${task}.yaml" "${REPO}/scripts/cfg/task/policy_eval/"
  work="${EVAL_ROOT}/work/${t}_${GR00T_NAME}_${tag}"
  log="${EVAL_ROOT}/logs/eval_${t}_${GR00T_NAME}_${tag}.log"
  mkdir -p "${work}"
  echo "[eval] $(date +%T) ${GR00T_NAME} ${t} n=${n} tag=${tag} -> ${log}"
  (cd "${work}" && PYTHONPATH="${REPO}:${OPENPI_STUBS}" "${PY}" \
    "${REPO}/scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py" \
    --config-name=real2sim_cfg "task=policy_eval/${task}" \
    "root_dir=${EVAL_ROOT}/out" "scene_name=${t}" "s15_eval.out_dirname=s15_eval_${GR00T_NAME}_${tag}" \
    s15_eval.policy=gr00t s15_eval.host=localhost "s15_eval.port=${GR00T_PORT}" "s15_eval.checkpoint=${GR00T_CHECKPOINT}" \
    "s15_eval.external_sensors_cfg=${EXTERNAL_SENSORS:-nv_franka_droid}" "s15_eval.tonemap_op=${TONEMAP_OP:-null}" \
    ${GRASPING_MODE:+"++s15_eval.grasping_mode=${GRASPING_MODE}"} \
    "s15_eval.scene_json=${scene}" "s15_eval.prompt='${prompt}'" \
    "s15_eval.timeout_s=${timeout}" "s15_eval.execute_horizon=${horizon}" "s15_eval.n_episodes=${n}" \
    "s15_eval.init_states_path=${init}" \
    s15_eval.save_video=true s15_eval.video_fps=10 "+s15_eval.video_resolution=[180,320]" \
    > "${log}" 2>&1)
  local rc=$?
  echo "[eval] $(date +%T) ${GR00T_NAME} ${t} rc=${rc}: $(grep -a 'Success rate:' "${log}" | tail -1)"
  return ${rc}
}

all() {  # all N TAG SPEC:TASK...: one task after another. A run that dies (another session's job took the
         # card mid-run) is retried once from its first episode.
  local n="$1" tag="$2" item; shift 2
  for item in "$@"; do
    eval_one "${item%%:*}" "${item#*:}" "${n}" "${tag}" && continue
    echo "[eval] $(date +%T) ${item#*:} failed; retrying once in 2 minutes"; sleep 120
    eval_one "${item%%:*}" "${item#*:}" "${n}" "${tag}"
  done
}

case "${1:-}" in
  server) server ;;
  stop) stop ;;
  eval) shift; eval_one "$@" ;;
  all) shift; all "$@" ;;
  *) echo "usage: $0 server|stop|eval SPEC TASK [N] [TAG]|all N TAG SPEC:TASK..." >&2; exit 2 ;;
esac
