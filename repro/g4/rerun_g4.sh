#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The G4 rerun (repro/README.md, chapter 17): rerun_v2.sh's configuration on a Colab G4 runtime
# (RTX PRO 6000 Blackwell, 96 GB) with every 12 GB adaptation removed:
#   stage 5 PDA geometric backend   da3, the repository default         (v2: depth_pro)
#   stage 7 Hunyuan3D-2.1           low_vram=false, the default          (v2: true)
#   stages 5-8                      streamed, the default                (v2: --no-stream)
#   B cousin_generation             low_vram=false, the default          (v2: true)
#   local backends                  models stay loaded between calls     (v2: unloaded after each)
# Unchanged from v2: Qwen3-VL 8B and Qwen-Image-2.1-Turbo in place of Gemini, the default
# object-removal order, the official SAM3, DA3 inside the simfoundry env. The VLM is pinned because
# run_local_pipeline.sh has since picked qwen3.5:9b for A (repro/README.md, chapter 16).
#
#   warmup  A on PutMarkerInCup as scene g4_warmup, so the timed runs below do not include
#           Hugging Face downloads. Resumable (--skip-successful). Its model responses go to their
#           own cache directory: in cache mode a cached response is replayed, not recomputed.
#   timed   A and the C smoke test on both example videos, then B on PutMarkerInCup.
# Usage: repro/g4/rerun_g4.sh warmup|timed      (on the G4 runtime; logs in $SIMFOUNDRY_LOG_DIR)
set -uo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
cd "${REPO}"
export SIMFOUNDRY_IMAGE_BACKEND=qwen21 SIMFOUNDRY_OLLAMA_MODEL=qwen3-vl:8b-instruct
export SIMFOUNDRY_OLLAMA_KEEP_ALIVE=24h SIMFOUNDRY_COMFY_FREE_AFTER=0 SIMFOUNDRY_LOCAL_OFFLOAD=0
# No display on the runtime.
export OMNIGIBSON_HEADLESS=1
L="${SIMFOUNDRY_LOG_DIR:-/content/logs}"
mkdir -p "${L}"
V="${REPO}/docs/assets/example_videos"

run() {  # run <log name> <cmd...>
  local name="$1" t0=${SECONDS} rc
  shift
  echo "=== ${name} $(date +%T)"
  "$@" > "${L}/${name}.log" 2>&1
  rc=$?
  echo "    ${name} rc=${rc} ($((SECONDS - t0)) s)"
  return "${rc}"
}

A() {  # A <scene> <video> [run.sh args...]
  local scene="$1" video="$2"
  shift 2
  repro/run_local_pipeline.sh A_reconstruction --scene-name "${scene}" --video-fpath "${V}/${video}.mp4" \
    --env-da3 simfoundry "$@"
}

case "${1:-}" in
  warmup)
    run g4_A_warmup A g4_warmup PutMarkerInCup --skip-successful \
      --cache-mode --model-cache-dir .cache/simfoundry/model_calls_g4_warmup
    ;;
  timed)
    for pair in "put_marker_in_cup_g4:PutMarkerInCup" "put_cup_in_bowl_g4:PutCupInBowl"; do
      scene=${pair%%:*}; video=${pair##*:}
      run "g4_A_${scene}" A "${scene}" "${video}" --cache-mode --model-cache-dir .cache/simfoundry/model_calls_g4
      run "g4_C_${scene}" repro/run_local_pipeline.sh C_application --scene-name "${scene}" --mode smoke-random
    done
    run g4_B_put_marker_in_cup_g4 repro/run_local_pipeline.sh B_augmentation --scene-name put_marker_in_cup_g4 \
      --cache-mode --model-cache-dir .cache/simfoundry/model_calls_g4 -- \
      prompt_cousin_structured.max_objects=2 prompt_cousin_structured.max_generated_images_per_object=1
    ;;
  *) echo "usage: $0 warmup|timed" >&2; exit 2 ;;
esac
echo "=== done $(date +%T)"
