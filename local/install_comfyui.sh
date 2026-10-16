#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# (Re)install this machine's ComfyUI for the local image backend: code next to Ollama, conda env
# `comfyui`, and the Qwen-Image-2.1-Turbo weights (SIMFOUNDRY_IMAGE_BACKEND=qwen21) on /data.
# Re-running skips whatever is already in place. Start it with local/start_comfyui.sh; see
# local/README.md.
set -euo pipefail
COMFY_DIR="${COMFY_DIR:-/home/inner/real2sim/tools/ComfyUI}"
WEIGHTS="${COMFY_WEIGHTS:-/data/models/comfyui}"
export PATH="${HOME}/miniforge3/bin:${PATH}"
export PIP_NO_CACHE_DIR=1

# v0.41.0 is the release checked to have the Qwen-Image-2.1 nodes the shim's qwen21 graph uses
# (TextEncodeQwenImage21, QwenImage21Cache).
[[ -d "${COMFY_DIR}" ]] || git clone --depth 1 --branch v0.41.0 https://github.com/comfyanonymous/ComfyUI "${COMFY_DIR}"
[[ -d "${COMFY_DIR}/custom_nodes/ComfyUI-GGUF" ]] ||
  git clone https://github.com/city96/ComfyUI-GGUF "${COMFY_DIR}/custom_nodes/ComfyUI-GGUF"
py="${HOME}/miniforge3/envs/comfyui/bin/python"
if [[ ! -x "${py}" ]]; then
  mamba create -y -n comfyui python=3.13 pip
  "${py}" -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
  "${py}" -m pip install -r "${COMFY_DIR}/requirements.txt" -r "${COMFY_DIR}/custom_nodes/ComfyUI-GGUF/requirements.txt"
fi

# fetch REPO REVISION PATH DEST SIZE SHA256. This link stalls on large files, and `curl --retry`
# restarts from byte 0 over the partial file, so every attempt resumes from what is on disk.
fetch() {
  local url="https://huggingface.co/$1/resolve/$2/$3" dest="${WEIGHTS}/$4" size=$5 sha=$6
  [[ -f "${dest}" ]] && return  # checked before it was moved into place
  mkdir -p "$(dirname "${dest}")"
  while (( $(stat -c %s "${dest}.part" 2>/dev/null || echo 0) < size )); do
    curl -fL --connect-timeout 30 --speed-limit 102400 --speed-time 30 -C - -o "${dest}.part" "${url}" || sleep 3
  done
  echo "${sha}  ${dest}.part" | sha256sum -c -
  mv "${dest}.part" "${dest}"
}
fetch unsloth/Qwen-Image-2.1-Turbo-GGUF 5d1f9f759ea877ae556cda88dbec958b62ad3809 \
  qwen-image-2.1-turbo-Q8_0.gguf diffusion_models/qwen-image-2.1-turbo-Q8_0.gguf \
  7640860480 399d2ddc3636f36456e2f5bfa57187da4e38721bc046430317f42a9eab5b1712
fetch Comfy-Org/Qwen-Image-2.1 a8342359c61979b06a8dafb1a47991fe1f30e994 \
  text_encoders/qwen3vl_8b_int8_convrot.safetensors text_encoders/qwen3vl_8b_int8_convrot.safetensors \
  9350798360 8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f
fetch Comfy-Org/Qwen-Image-2.1 a8342359c61979b06a8dafb1a47991fe1f30e994 \
  vae/qwen_image_2.1_vae_bf16.safetensors vae/qwen_image_2.1_vae_bf16.safetensors \
  675509688 bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9
echo "ComfyUI in ${COMFY_DIR}, weights in ${WEIGHTS}"
