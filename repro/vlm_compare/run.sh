#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compare local VLMs on the pipeline's own VLM calls (repro/README.md, chapter 16): every replay runs
# once per model, each in a fresh process (the shim reads its settings at import), with the
# pipeline's shim settings (keep_alive 0, 4096-token cap) and the response cache off.
#
# Usage: repro/vlm_compare/run.sh <out_dir> [<model>[:think] ...]
#   default models: qwen3-vl:8b-instruct qwen3.5:9b:0; ":0" sends think=false, ":1" think=true.
#   DATA_ROOT (default <repo>/Data) holds put_marker_in_cup{,_v2,_bg} and put_cup_in_bowl_v2.
# Needs Ollama running (repro/start_ollama.sh) with the models pulled; not ComfyUI. Each model takes
# about 10 min; a thinking Qwen3.5 much longer (most calls run into the token cap).
set -uo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
[[ -f "${REPO}/local/env.sh" ]] && source "${REPO}/local/env.sh"
OUT="$(readlink -f "${1:?usage: run.sh <out_dir> [model[:think] ...]}")"; shift
MODELS=("$@")
[[ ${#MODELS[@]} -gt 0 ]] || MODELS=(qwen3-vl:8b-instruct qwen3.5:9b:0)
DATA_ROOT="${DATA_ROOT:-${REPO}/Data}"
PY="${MAMBA_ROOT_PREFIX:-${HOME}/miniforge3}/envs/simfoundry/bin/python"
unset CACHE_MODE TEST_MODE
export SIMFOUNDRY_LOCAL_BACKEND=1 SIMFOUNDRY_OLLAMA_KEEP_ALIVE=0 GCLOUD_PROJECT=local
curl -fsS http://127.0.0.1:11434/api/version >/dev/null || { echo "Ollama is not running (repro/start_ollama.sh)" >&2; exit 1; }

labels=()
for spec in "${MODELS[@]}"; do
  model="${spec}" think=""
  if [[ "${spec}" =~ ^(.+):([01])$ ]]; then model="${BASH_REMATCH[1]}"; think="${BASH_REMATCH[2]}"; fi
  tag="${model//[:\/]/_}${think:+_think${think}}"
  labels+=("${tag}")
  # The replays wait for an idle GPU before every call (bench_common.wait_for_idle_gpu).
  mkdir -p "${OUT}/${tag}"
  echo "=== ${tag} $(date +%T)"
  env SIMFOUNDRY_OLLAMA_MODEL="${model}" ${think:+SIMFOUNDRY_OLLAMA_THINK=${think}} \
    "${PY}" "${HERE}/replay_weak_calls.py" "${OUT}/${tag}" --data-root "${DATA_ROOT}" \
      --s1-frames 10,20,30,40,50,60,80,120 > "${OUT}/${tag}/weak_calls.log" 2>&1
  env SIMFOUNDRY_OLLAMA_MODEL="${model}" ${think:+SIMFOUNDRY_OLLAMA_THINK=${think}} \
    "${PY}" "${HERE}/replay_other_calls.py" "${OUT}/${tag}" --data-root "${DATA_ROOT}" > "${OUT}/${tag}/other_calls.log" 2>&1
done
weak=() other=()
for tag in "${labels[@]}"; do
  weak+=("${tag}=${OUT}/${tag}/results.jsonl")
  other+=("${tag}=${OUT}/${tag}/other_calls.jsonl")
done
"${PY}" "${HERE}/grade_weak_calls.py" "${OUT}/grades" "${weak[@]}" > /dev/null
"${PY}" "${HERE}/grade_other_calls.py" "${OUT}/grades/other_calls.md" "${other[@]}" > /dev/null
echo "tables: ${OUT}/grades/scores.md, ${OUT}/grades/other_calls.md (box sheets in ${OUT}/grades/sheets)"
