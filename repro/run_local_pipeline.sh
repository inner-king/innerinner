#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run a SimFoundry pipeline (A_reconstruction / B_augmentation / C_application) with the local
# backends instead of Gemini, on a 12 GB GPU.
#
# Substitutions (see simfoundry/models/local_backends.py):
#   Gemini text/vision  -> Ollama qwen3-vl:8b-instruct   (repro/start_ollama.sh, 127.0.0.1:11434)
#   Gemini image models -> ComfyUI Qwen-Image-Edit-2511  (must already be running on 127.0.0.1:8189)
#
# Usage: repro/run_local_pipeline.sh A_reconstruction --scene-name X --video-fpath V [run.sh args] [-- hydra overrides]
# Give --video-fpath as an absolute path: the stages do not run from the caller's directory.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
# Machine-specific locations (Ollama binary, model stores, ...) come from local/env.sh when the
# checkout has one, as a machine's own branch does (e.g. hanbin5/local).
[[ -f "${REPO}/local/env.sh" ]] && source "${REPO}/local/env.sh"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniforge3}"
export PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}"

export SIMFOUNDRY_LOCAL_BACKEND=1
# SAM3 loads its own checkpoint from Hugging Face (facebook/sam3, gated). The first runs, made
# before access was granted, used SAM 3.1 detector weights; to reproduce them, set
#   SIMFOUNDRY_SAM3_CHECKPOINT=/path/to/sam3.1_multiplex_fp16.pt
# Unload the VLM right after each answer: the same stage needs the GPU for SAM3 and ComfyUI next.
export SIMFOUNDRY_OLLAMA_KEEP_ALIVE="${SIMFOUNDRY_OLLAMA_KEEP_ALIVE:-0}"
# The config interpolates GCLOUD_PROJECT; no Google service is contacted in local mode.
export GCLOUD_PROJECT="${GCLOUD_PROJECT:-local}"
# 12 GB leaves little slack; expandable segments avoid OOMs from allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMNI_KIT_ACCEPT_EULA=YES

LOG_DIR="${SIMFOUNDRY_LOG_DIR:-${HOME}/simfoundry_logs}"
if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  mkdir -p "${LOG_DIR}"
  nohup "${HERE}/start_ollama.sh" >> "${LOG_DIR}/ollama.log" 2>&1 &
  ollama_pid=$!
  until curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; do
    kill -0 "${ollama_pid}" 2>/dev/null || {
      echo "Ollama did not start (set OLLAMA_BIN / OLLAMA_MODELS); see ${LOG_DIR}/ollama.log" >&2; exit 1; }
    sleep 1
  done
fi
COMFY_URL="${SIMFOUNDRY_COMFY_URL:-http://127.0.0.1:8189}"
curl -fsS "${COMFY_URL}/system_stats" >/dev/null || { echo "ComfyUI is not reachable at ${COMFY_URL}" >&2; exit 1; }

pipeline="$1"; shift
cd "${REPO}"
exec bash "scripts/pipeline/${pipeline}/run.sh" "$@"
