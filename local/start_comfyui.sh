#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ComfyUI for the local image backend: repro/run_local_pipeline.sh expects it on 127.0.0.1:8189.
# Code /home/inner/real2sim/tools/ComfyUI (v0.41.0, with custom_nodes/ComfyUI-GGUF), conda env
# `comfyui`, weights /data/models/comfyui (comfyui_models.yaml). See local/README.md.
#
#   nohup local/start_comfyui.sh >> ~/simfoundry_logs/comfyui.log 2>&1 &
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
COMFY_DIR="${COMFY_DIR:-/home/inner/real2sim/tools/ComfyUI}"
COMFY_PYTHON="${COMFY_PYTHON:-${HOME}/miniforge3/envs/comfyui/bin/python}"
cd "${COMFY_DIR}"
exec "${COMFY_PYTHON}" main.py --listen 127.0.0.1 --port 8189 \
  --extra-model-paths-config "${HERE}/comfyui_models.yaml" "$@"
