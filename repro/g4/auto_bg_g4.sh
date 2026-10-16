#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The automatic 3DGS background (paper App. E.5.1) on the G4 runtime at the repository's defaults,
# for both example videos (repro/README.md, chapter 17). Differences from the 12 GB run (chapter 13):
#   frames         400, the repository default               (12 GB: 160; VOID took ~25 s/frame)
#   VOID           VOID_LOW_VRAM=0, the unpatched code path   (12 GB: tiled VAE decode, CPU offload)
#   void-DA3       the repo's stage 2b chunk chain            (12 GB: anchored variant, see chapter 13)
#   splat-prep A   Hunyuan low_vram=false                     (12 GB: true)
# Kept as on the 12 GB run: the splat-prep command of auto_bg_reconstruction/README.md (depth_pro PDA
# backend, --no-stream), the local backends of rerun_g4.sh, and repro/run_auto_bg.sh's local-VLM
# handling of step 1 (object list from the reconstruction's stage 5).
# Usage: repro/g4/auto_bg_g4.sh [scene:Video ...]   (default: both example videos; resumable)
set -uo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
cd "${REPO}"
# The VLM of the chapter-17 runs; run_local_pipeline.sh has since picked qwen3.5:9b for A (chapter 16).
export SIMFOUNDRY_IMAGE_BACKEND=qwen21 SIMFOUNDRY_OLLAMA_MODEL=qwen3-vl:8b-instruct
export SIMFOUNDRY_OLLAMA_KEEP_ALIVE=24h SIMFOUNDRY_COMFY_FREE_AFTER=0 SIMFOUNDRY_LOCAL_OFFLOAD=0
export OMNIGIBSON_HEADLESS=1
L="${SIMFOUNDRY_LOG_DIR:-/content/logs}"
mkdir -p "${L}"
V="${REPO}/docs/assets/example_videos"
PAIRS=("$@")
[[ ${#PAIRS[@]} -gt 0 ]] || PAIRS=("put_marker_in_cup_bg_g4:PutMarkerInCup" "put_cup_in_bowl_bg_g4:PutCupInBowl")

for pair in "${PAIRS[@]}"; do
  scene=${pair%%:*}; video=${pair##*:}
  echo "=== splat-prep A ${scene} $(date +%T)"
  t0=${SECONDS}
  repro/run_local_pipeline.sh A_reconstruction --scene-name "${scene}" --video-fpath "${V}/${video}.mp4" \
    --env-da3 simfoundry --no-stream --skip-successful \
    --cache-mode --model-cache-dir .cache/simfoundry/model_calls_g4_bg -- \
    s1_video.splat_prep=true s1_video.n_subsampled_frames=400 s1_video.target_w=672 s1_video.target_h=384 \
    s5_scene.pda_geometric_backend=depth_pro > "${L}/g4_bgA_${scene}.log" 2>&1
  rc=$?
  echo "    rc=${rc} ($((SECONDS - t0)) s)"
  [[ ${rc} -eq 0 ]] || continue
  # AUTO_BG_PREP_ONLY=1 stops after the splat-prep A. PutCupInBowl's was made that way on the A/B/C
  # runtime while the auto_bg runtime, short of disk, worked on PutMarkerInCup; the scene directory
  # was then copied over, where --skip-successful skipped its A stages.
  [[ "${AUTO_BG_PREP_ONLY:-0}" == 1 ]] && continue
  echo "=== auto_bg ${scene} $(date +%T)"
  t0=${SECONDS}
  # Step 7 (PLY -> NuRec USDZ, 9 s) needs the 3dgrut env, which the auto_bg runtime left out for disk
  # (setup_g4.sh); AUTO_BG_STOP_BEFORE_ASSETS=1 stops after step 6 so step 7 can run elsewhere.
  SIMFOUNDRY_LOG_DIR="${L}" VOID_LOW_VRAM=0 VOID_DA3=canonical AUTO_BG_STOP_BEFORE_ASSETS="${AUTO_BG_STOP_BEFORE_ASSETS:-0}" \
    repro/run_auto_bg.sh "${scene}" "desk, table, or counter" > "${L}/g4_auto_bg_${scene}.log" 2>&1
  echo "    rc=$? ($((SECONDS - t0)) s)"
done
echo "=== done $(date +%T)"
