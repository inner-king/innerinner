#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The repo's automatic background pipeline (paper App. E.5.1; run_auto_align.sh steps 1-7), as run for
# the reproduction on a 12 GB GPU with the local backends (repro/README.md, chapter 13).
#
# Same step scripts and order as run_auto_align.sh. Differences, each forced by that setup:
#   - Local model backend (no Gemini): step 1's object proposal uses the text VLM
#     (s1_quadmask.gemini_model=gemini-3.1-pro-preview -> the local VLM) instead of the image model,
#     since the step only reads text (get_result_text) from it.
#     Its object list is the canonical stage-5 one (s1_quadmask.objects) because the local VLM can
#     miss an object on keyframe 0 that it finds on the decomposition frame: Qwen3-VL 8B missed the
#     marker there, and Qwen3.5-9B, which finds it there, missed it on 2 of 8 other far frames.
#   - void-DA3 ("canonical 2b") runs repro/da3_anchored_stage.py instead of 2b_run_da.py: the repo
#     chains 220-frame chunks (24 GB); 16-frame chunks that fit 12 GB drifted >20 deg, so chunks are
#     aligned to a joint reference call instead (same Umeyama fit, star instead of chain).
#   - VOID_LOW_VRAM=1: tiled VAE decode in passes 1-2 and sequential CPU offload in pass 2
#     (patches/void-model-low-vram.patch).
#   - The canonical reconstruction feeding it (Data/<scene>, splat-prep mode) was made the same way
#     for stage 2.
# Each step is skipped when its output exists, so a rerun resumes after the last finished step.
# Usage: repro/run_auto_bg.sh <scene> [floor category] [extra hydra overrides for step 5...]
set -uo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
# Machine-specific locations (Ollama binary, model stores, ...) come from local/env.sh when the
# checkout has one, as a machine's own branch does (e.g. hanbin5/local).
[[ -f "${REPO}/local/env.sh" ]] && source "${REPO}/local/env.sh"
SCENE="$1"; shift
FLOOR="${1:-desk, table, or counter}"; shift || true
S5_EXTRA=("$@")
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniforge3}"
# Steps run with the env python directly (not `mamba run`), so put the env bin (ffmpeg, ns-*) on PATH.
export PATH="${MAMBA_ROOT_PREFIX}/envs/simfoundry/bin:${MAMBA_ROOT_PREFIX}/bin:${PATH}"
export SIMFOUNDRY_LOCAL_BACKEND=1 SIMFOUNDRY_OLLAMA_KEEP_ALIVE=0
# Pipeline A's local VLM, as run_local_pipeline.sh picks it (the given object list means step 1 does
# not call it).
export SIMFOUNDRY_OLLAMA_MODEL="${SIMFOUNDRY_OLLAMA_MODEL:-qwen3.5:9b}"
export SIMFOUNDRY_OLLAMA_THINK="${SIMFOUNDRY_OLLAMA_THINK:-0}"
export SIMFOUNDRY_IMAGE_BACKEND="${SIMFOUNDRY_IMAGE_BACKEND:-qwen21}"
export VOID_LOW_VRAM="${VOID_LOW_VRAM:-1}"
export GCLOUD_PROJECT="${GCLOUD_PROJECT:-local}" OMNI_KIT_ACCEPT_EULA=YES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LOG_ROOT="${SIMFOUNDRY_LOG_DIR:-${HOME}/simfoundry_logs}"
mkdir -p "${LOG_ROOT}"
if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  nohup "${HERE}/start_ollama.sh" >> "${LOG_ROOT}/ollama.log" 2>&1 &
  ollama_pid=$!
  until curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; do
    kill -0 "${ollama_pid}" 2>/dev/null || {
      echo "Ollama did not start (set OLLAMA_BIN / OLLAMA_MODELS); see ${LOG_ROOT}/ollama.log" >&2; exit 1; }
    sleep 1
  done
fi

cd "${REPO}"
D="${REPO}/Data/${SCENE}"
N=$(ls "${D}"/s1_video/frames_subsampled_* -d | sed 's/.*_//')
PY="${MAMBA_ROOT_PREFIX}/envs/simfoundry/bin/python"
LOG="${LOG_ROOT}/auto_bg_${SCENE}"
mkdir -p "${LOG}" "${D}/auto_bg/void/input"
ln -sfn "${D}/s1_video/frames_subsampled_${N}" "${D}/auto_bg/void/input/subsampled"
ln -sfn "${D}/s1_video/input_video.mp4" "${D}/auto_bg/void/input/input_video.mp4"
S=scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction
COMMON=(scene_name="${SCENE}" num_frames="${N}")

step() {  # step <label> <done-marker> <cmd...>
  local label="$1" marker="$2"; shift 2
  if [[ -e "${marker}" ]]; then echo "[auto_bg] skip ${label} (exists)"; return 0; fi
  echo "[auto_bg] $(date +%T) start ${label}"
  if "$@" > "${LOG}/${label// /_}.log" 2>&1; then echo "[auto_bg] $(date +%T) done ${label}"
  else echo "[auto_bg] $(date +%T) FAILED ${label} (see ${LOG}/${label// /_}.log)"; exit 1; fi
}

# Objects to erase = the canonical reconstruction's stage-5 categories (same prompt and VLM, run on
# the decomposition frame). On keyframe 0 alone Qwen3-VL missed the marker next to the cup.
OBJS=$("${PY}" -c "import json,sys; print('[' + ','.join(repr(o) for o in json.load(open(sys.argv[1]))['vlm_categories_full']) + ']')" \
  "${D}/s5_scene/obj_cat_list/iter_0.json")
step "1 quadmask" "${D}/auto_bg/void/input/quadmask_0.mp4" \
  "${PY}" ${S}/1_generate_quadmask_for_void.py "${COMMON[@]}" floor_category="'${FLOOR}'" \
  s1_quadmask.gemini_model=gemini-3.1-pro-preview "s1_quadmask.objects=${OBJS}"
step "2 pass1" "${D}/auto_bg/void/pass1/pass1.mp4" "${PY}" ${S}/2_run_void_pass1.py "${COMMON[@]}"
# PASS2_REUSE=true resumes pass 2 keeping finished chunks (only valid if pass 1 and the mask are unchanged).
step "3 pass2" "${D}/auto_bg/void/pass2/pass2.mp4" "${PY}" ${S}/3_run_void_pass2.py "${COMMON[@]}" \
  s3_pass2.reuse_chunks="${PASS2_REUSE:-false}"
step "da3 void (anchored)" "${D}/auto_bg/da3/void/da/exports/npz/results.npz" \
  "${PY}" "${HERE}/da3_anchored_stage.py" "${D}/auto_bg/void/pass2/cleaned_frames" "${D}/auto_bg/da3/void"
step "4 seed ply" "${D}/auto_bg/seed.ply" "${PY}" ${S}/4_build_seed_ply_from_void_da3.py "${COMMON[@]}"
ln -sfn "void/pass2/cleaned_frames" "${D}/auto_bg/clean_frames"
step "5 train splat" "${D}/auto_bg/splat/export/${SCENE}_bg.ply" \
  "${PY}" ${S}/5_train_bg_splat.py "${COMMON[@]}" "${S5_EXTRA[@]}"
step "6 bridge" "${D}/auto_bg/splat/export/${SCENE}_bg.ply.pose.json" "${PY}" ${S}/6_bridge_bg_splat_to_og.py "${COMMON[@]}"
step "7 assets" "${REPO}/assets/scenes/${SCENE}/${SCENE}_scene_state_auto_bg.json" \
  "${PY}" ${S}/7_build_og_scene_assets.py "${COMMON[@]}"
echo "[auto_bg] SUCCESS -> ${REPO}/assets/scenes/${SCENE}/${SCENE}_scene_state_auto_bg.json"
