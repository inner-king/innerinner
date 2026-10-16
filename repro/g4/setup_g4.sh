#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Set up a Colab G4 runtime (RTX PRO 6000 Blackwell, 96 GB) for the reruns that the 12 GB RTX 3060
# could not do as the repository intends (repro/README.md, chapter 17). Runs on the VM as root,
# from the checkout copied to /content/SimFoundry.
#
#   envs     simfoundry, hunyuan, void, nerfstudio_simfoundry and 3dgrut, built by the repository's
#            own installers, then its checkpoint downloader. DA3 runs inside simfoundry, as on the
#            12 GB machine (--env-da3 simfoundry), so the da3 and any6d envs are not built.
#   servers  the same local model backends as the 3060 runs: Ollama 0.40.1 with qwen3-vl:8b-instruct,
#            and ComfyUI with Qwen-Image-2.1-Turbo Q8_0 (unsloth GGUF) plus the Comfy-Org text
#            encoder and VAE, under the file names simfoundry/models/local_backends.py expects.
#   vulkan   Colab mounts the driver's CUDA, EGL and GLX libraries (/usr/lib64-nvidia) but not
#            libnvidia-glcore / libnvidia-glvkspirv or a Vulkan ICD file, so Isaac Sim finds no
#            Vulkan device. The missing user-space libraries of the running driver version are
#            copied out of NVIDIA's own package for that version; the kernel module is untouched.
#
# The 236 GB disk does not hold every env plus the VOID weights (41 GB) and the model caches, so the
# reruns used two runtimes, and G4_ENVS picks the optional envs:
#   A/B/C runtime    G4_ENVS=""                   (simfoundry and hunyuan only)
#   auto_bg runtime  G4_ENVS="void nerfstudio"    (auto_bg step 7, the 3dgrut conversion, ran on the
#                                                  3060 machine, which has that env)
# Each step leaves a marker in /content/setup_state and is skipped on a rerun. Logs: /content/logs.
# Usage: G4_ENVS="..." bash repro/g4/setup_g4.sh [all|envs|servers|vulkan|splat]
set -uo pipefail
G4_ENVS="${G4_ENVS-void nerfstudio 3dgrut}"

REPO="${REPO:-/content/SimFoundry}"
LOG=/content/logs
ST=/content/setup_state
export HOME=/root
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/root/miniforge3}"
export PATH="/root/.local/bin:${MAMBA_ROOT_PREFIX}/bin:${PATH}"
export PIP_NO_CACHE_DIR=1
export OMNI_KIT_ACCEPT_EULA=YES
COMFY=/content/ComfyUI
COMFY_PY=/content/comfy-venv/bin/python
mkdir -p "${LOG}" "${ST}"

step() {  # step <name> <cmd...>: run once, log to $LOG/<name>.log, leave a marker on success
  local name="$1"; shift
  if [[ -e "${ST}/${name}.done" ]]; then echo "[setup] $(date +%T) skip ${name}"; return 0; fi
  echo "[setup] $(date +%T) start ${name}"
  local t0=${SECONDS}
  if "$@" > "${LOG}/${name}.log" 2>&1; then
    touch "${ST}/${name}.done"; echo "[setup] $(date +%T) done ${name} ($((SECONDS - t0)) s)"
  else
    echo "[setup] $(date +%T) FAILED ${name} ($((SECONDS - t0)) s), see ${LOG}/${name}.log"; return 1
  fi
}

# ---------------------------------------------------------------- envs
install_miniforge() {
  [[ -x "${MAMBA_ROOT_PREFIX}/bin/mamba" ]] && return 0
  curl -fsSL -o /content/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /content/miniforge.sh -b -p "${MAMBA_ROOT_PREFIX}"
}

# basicsr 1.4.2 imports torchvision.transforms.functional_tensor, gone since torchvision 0.17
# (the env has 0.22). Same one-line fix as on the 3060 machine (repro/README.md, chapter 2).
fix_basicsr() {
  local py="${MAMBA_ROOT_PREFIX}/envs/hunyuan/bin/python" f
  f="$("${py}" -c 'import importlib.util as u; print(u.find_spec("basicsr").submodule_search_locations[0])')/data/degradations.py"
  sed -i 's/torchvision\.transforms\.functional_tensor import/torchvision.transforms.functional import/' "${f}"
  "${py}" -c "import basicsr.data.degradations"
}

preclone() {  # preclone <url> <dest>: GitHub answered one anonymous clone with a 401 on 2026-10-16
  local url="$1" dest="$2" i
  [[ -d "${dest}/.git" ]] && return 0
  for i in 1 2 3 4 5; do git clone "${url}" "${dest}" && return 0; rm -rf "${dest}"; sleep $((i * 20)); done
  return 1
}

wants() { [[ " ${G4_ENVS} " == *" $1 "* ]]; }

# The package cache is not needed once an env exists, and the disk is short (see G4_ENVS).
clean_caches() { mamba clean -a -y > /dev/null 2>&1; uv cache clean > /dev/null 2>&1; rm -rf /root/.cache/pip; }

envs() {
  local I="${REPO}/scripts/installation" C="${REPO}/repro/g4/constraints_simfoundry.txt" e
  step miniforge install_miniforge || return 1
  # install_simfoundry.sh clones this itself, without retries, then checks out its pinned commit.
  step preclone_b1k preclone https://github.com/StanfordVL/BEHAVIOR-1K.git "${REPO}/deps/BEHAVIOR-1K" || return 1
  # Only this env gets pins: the other installers pin opencv and numpy themselves (constraints_simfoundry.txt).
  step env_simfoundry env PIP_CONSTRAINT="${C}" UV_CONSTRAINT="${C}" \
    bash "${I}/install_simfoundry.sh" --project-root "${REPO}" --env-name simfoundry --cuda-version 12.8 --default \
    || return 1
  clean_caches
  step env_hunyuan bash "${I}/install_hunyuan.sh" --project-root "${REPO}" --env-name hunyuan \
    --cuda-version 12.8 --default || return 1
  step fix_basicsr fix_basicsr || return 1
  clean_caches
  # FoundationPose weights come from Google Drive folders that hit download quotas; the 3060
  # machine's copy is uploaded to /content/fallback as the downloader's fallback root. The
  # downloader always fetches the VOID weights (41 GB); a runtime without the void env drops them.
  step checkpoints bash "${I}/download_checkpoints.sh" --project-root "${REPO}" --env-name simfoundry \
    --checkpoint-fallback-root /content/fallback --default || return 1
  wants void || rm -rf "${REPO}/deps/void-model"
  if wants void; then
    step env_void bash "${I}/install_void.sh" --project-root "${REPO}" --env-name void --default || return 1
    clean_caches
  fi
  if wants nerfstudio; then
    step env_nerfstudio bash "${I}/install_nerfstudio.sh" --project-root "${REPO}" \
      --env-name nerfstudio_simfoundry --default || return 1
    clean_caches
  fi
  if wants 3dgrut; then
    step env_3dgrut bash "${I}/install_3dgrut.sh" --project-root "${REPO}" --env-name 3dgrut \
      --cuda-version 12.8 --default || return 1
    clean_caches
  fi
  for e in simfoundry hunyuan void nerfstudio_simfoundry 3dgrut; do
    [[ -x "${MAMBA_ROOT_PREFIX}/envs/${e}/bin/python" ]] && \
      "${MAMBA_ROOT_PREFIX}/envs/${e}/bin/python" -m pip freeze > "${LOG}/freeze_${e}.txt" 2>/dev/null
  done
  echo "[setup] $(date +%T) envs finished: ${G4_ENVS}"
}

# ---------------------------------------------------------------- servers
install_uv() { command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; }

install_ollama() {
  command -v ollama >/dev/null && return 0
  apt-get install -y -qq zstd >/dev/null
  curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.1 sh
}

start_ollama() {
  curl -fsS http://127.0.0.1:11434/api/version && return 0
  setsid nohup env OLLAMA_HOST=127.0.0.1:11434 ollama serve >> "${LOG}/ollama_serve.log" 2>&1 < /dev/null &
  for _ in $(seq 120); do curl -fsS http://127.0.0.1:11434/api/version && return 0; sleep 1; done
  return 1
}

install_comfy() {
  [[ -d "${COMFY}" ]] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "${COMFY}"
  [[ -d "${COMFY}/custom_nodes/ComfyUI-GGUF" ]] || \
    git clone --depth 1 https://github.com/city96/ComfyUI-GGUF "${COMFY}/custom_nodes/ComfyUI-GGUF"
  [[ -x "${COMFY_PY}" ]] || uv venv --python 3.12 /content/comfy-venv
  uv pip install --python "${COMFY_PY}" torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "${COMFY_PY}" -r "${COMFY}/requirements.txt" \
    -r "${COMFY}/custom_nodes/ComfyUI-GGUF/requirements.txt" "huggingface_hub[hf_xet]"
}

download_q21() {
  local M="${COMFY}/models"
  /content/comfy-venv/bin/hf download unsloth/Qwen-Image-2.1-Turbo-GGUF qwen-image-2.1-turbo-Q8_0.gguf \
    --local-dir "${M}/unet"
  /content/comfy-venv/bin/hf download Comfy-Org/Qwen-Image-2.1 \
    text_encoders/qwen3vl_8b_int8_convrot.safetensors vae/qwen_image_2.1_vae_bf16.safetensors --local-dir "${M}"
}

start_comfy() {
  curl -fsS http://127.0.0.1:8189/system_stats && return 0
  (cd "${COMFY}" && setsid nohup "${COMFY_PY}" main.py --listen 127.0.0.1 --port 8189 \
    >> "${LOG}/comfy_serve.log" 2>&1 < /dev/null &)
  for _ in $(seq 300); do curl -fsS http://127.0.0.1:8189/system_stats && return 0; sleep 1; done
  return 1
}

servers() {
  step uv install_uv || return 1
  step ollama install_ollama || return 1
  start_ollama > /dev/null || { echo "[setup] Ollama did not start"; return 1; }
  step ollama_vlm ollama pull qwen3-vl:8b-instruct || return 1
  step comfy install_comfy || return 1
  step comfy_models download_q21 || return 1
  start_comfy > /dev/null || { echo "[setup] ComfyUI did not start"; return 1; }
  echo "[setup] $(date +%T) servers up: Ollama :11434, ComfyUI :8189"
}

# ---------------------------------------------------------------- vulkan
vulkan_libs() {
  local drv run L=/usr/lib64-nvidia
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  run="/content/NVIDIA-Linux-x86_64-${drv}.run"
  [[ -s "${run}" ]] || curl -fsSL -o "${run}" "https://us.download.nvidia.com/tesla/${drv}/NVIDIA-Linux-x86_64-${drv}.run" \
    || curl -fsSL -o "${run}" "https://us.download.nvidia.com/XFree86/Linux-x86_64/${drv}/NVIDIA-Linux-x86_64-${drv}.run"
  rm -rf /content/nvdrv && sh "${run}" --extract-only --target /content/nvdrv > /dev/null
  # Only libraries the runtime does not already mount, all of the running driver's version.
  local f b
  for f in /content/nvdrv/*.so."${drv}"; do
    b="$(basename "${f}")"
    [[ -e "${L}/${b}" ]] || { cp "${f}" "${L}/" && echo "added ${b}"; }
  done
  ldconfig -n "${L}"
  mkdir -p /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
  if [[ -f /content/nvdrv/nvidia_icd.json.template ]]; then
    sed 's/__NV_VK_ICD__/libGLX_nvidia.so.0/' /content/nvdrv/nvidia_icd.json.template > /etc/vulkan/icd.d/nvidia_icd.json
  else
    cp /content/nvdrv/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
  fi
  [[ -f /content/nvdrv/10_nvidia.json ]] && cp /content/nvdrv/10_nvidia.json /usr/share/glvnd/egl_vendor.d/
  rm -rf /content/nvdrv "${run}"
  # Other parts may hold the dpkg lock (Ollama's installer runs apt-get too). libGLU is the one system
  # library Isaac Sim needs that the Colab image lacks: without it libneuray (Iray/MDL) does not load,
  # no material shader can be generated, and the RTX renderer fails to start.
  apt-get -o DPkg::Lock::Timeout=600 install -y -qq vulkan-tools libglu1-mesa libxt6 > /dev/null
  vulkaninfo --summary
}

vulkan() { step vulkan vulkan_libs; }

# Only what retrain_bg.sh needs (repro/README.md, 17.7): the nerfstudio env, which trains the splat and
# also runs auto_bg steps 4-6. Those steps need nothing of the simfoundry env (Isaac Sim, 24 GB) but
# simfoundry.utils.processing_utils' top-level `import supervision`; that package and its two pure-Python
# imports go into a side directory (/content/pylib) with --no-deps, so the env's numpy and opencv stay
# as install_nerfstudio.sh pinned them. On the 3060 machine the bridge (step 6) run this way matched the
# simfoundry env's output to 1e-9.
splat() {
  local I="${REPO}/scripts/installation"
  step miniforge install_miniforge || return 1
  step env_nerfstudio bash "${I}/install_nerfstudio.sh" --project-root "${REPO}" \
    --env-name nerfstudio_simfoundry --default || return 1
  step pylib "${MAMBA_ROOT_PREFIX}/envs/nerfstudio_simfoundry/bin/python" -m pip install --no-deps \
    --target /content/pylib supervision==0.30.8 pyDeprecate==0.13.0 defusedxml==0.7.1 || return 1
  clean_caches
  echo "[setup] $(date +%T) splat envs finished"
}

case "${1:-all}" in
  envs) envs ;;
  splat) splat ;;
  servers) servers ;;
  vulkan) vulkan ;;
  all)
    # The three parts touch disjoint directories, so they run side by side.
    vulkan & v=$!
    servers & s=$!
    envs; e=$?
    wait "${v}"; wait "${s}"
    echo "[setup] $(date +%T) all parts finished (envs rc=${e})"
    ;;
  *) echo "usage: $0 [all|envs|servers|vulkan|splat]" >&2; exit 2 ;;
esac
