#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The v2 rerun (repro/README.md, chapter 11): both example scenes with the configuration closest to
# the original:
#   image model  Qwen-Image-2.1-Turbo (instead of Qwen-Image-Edit-2511)
#   removal      SimFoundry default prompt order [mask, outline, bbox] (no override)
#   SAM          official facebook/sam3 (run_local_pipeline.sh default)
#   VLM          qwen3-vl:8b-instruct, pinned: run_local_pipeline.sh now picks qwen3.5:9b for A
# Still adapted to 12 GB: depth_pro PDA backend, Hunyuan low_vram, no streaming.
# Usage: repro/rerun_v2.sh      (logs: ${SIMFOUNDRY_LOG_DIR:-~/simfoundry_logs}/v2_*.log)
set -uo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
[[ -f "${REPO}/local/env.sh" ]] && source "${REPO}/local/env.sh"
cd "${REPO}"
export SIMFOUNDRY_IMAGE_BACKEND=qwen21
# The v2 runs' VLM, whose answers the response cache below also holds (its keys name the Gemini model
# a stage asked for, not the Ollama model that answered).
export SIMFOUNDRY_OLLAMA_MODEL=qwen3-vl:8b-instruct
# Separate response cache: cache keys do not include the backend, so the default cache would
# replay Edit-2511 images for identical inputs.
CACHE=(--cache-mode --model-cache-dir .cache/simfoundry/model_calls_qwen21)
A_OVR=(s7_mesh.low_vram=true s5_scene.pda_geometric_backend=depth_pro)
V="${REPO}/docs/assets/example_videos"
L="${SIMFOUNDRY_LOG_DIR:-${HOME}/simfoundry_logs}"
mkdir -p "$L"

for pair in "put_marker_in_cup_v2:PutMarkerInCup" "put_cup_in_bowl_v2:PutCupInBowl"; do
  scene=${pair%%:*}; video=${pair##*:}
  echo "=== A $scene $(date +%T)"
  repro/run_local_pipeline.sh A_reconstruction --scene-name "$scene" --video-fpath "$V/$video.mp4" \
    --env-da3 simfoundry --no-stream "${CACHE[@]}" -- "${A_OVR[@]}" > "$L/v2_A_$scene.log" 2>&1
  echo "    A rc=$?"
  echo "=== C $scene $(date +%T)"
  repro/run_local_pipeline.sh C_application --scene-name "$scene" --mode smoke-random > "$L/v2_C_$scene.log" 2>&1
  echo "    C rc=$?"
done
echo "=== B put_marker_in_cup_v2 $(date +%T)"
repro/run_local_pipeline.sh B_augmentation --scene-name put_marker_in_cup_v2 "${CACHE[@]}" -- \
  prompt_cousin_structured.max_objects=2 prompt_cousin_structured.max_generated_images_per_object=1 \
  cousin_generation.low_vram=true > "$L/v2_B_put_marker_in_cup_v2.log" 2>&1
echo "    B rc=$?"
echo "=== done $(date +%T)"
