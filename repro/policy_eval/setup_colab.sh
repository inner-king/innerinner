#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Set up a Colab GPU runtime for the Table G.1 policy evaluation (repro/README.md, chapter 18). Runs on the
# VM as root, from the checkout copied to /content/SimFoundry, with repro/g4/ (chapter 17) next to it. The
# evaluation ran on an L4 (24 GB); the same steps were first run on a G4 (96 GB). Isaac Sim needs RT cores,
# which A100 and H100 runtimes lack.
#
#   vulkan      repro/g4/setup_g4.sh vulkan: the Vulkan ICD and libGLU that Colab lacks (chapter 17.2).
#   simfoundry  the repository's install_simfoundry.sh under repro/g4/constraints_simfoundry.txt, exactly
#               as setup_g4.sh builds it, without the hunyuan env and the model checkpoints, which
#               policy evaluation never touches.
#   openpi      openpi at the commit install_simfoundry.sh pins in its commented-out step 2.16, as its
#               own uv project (the policy server), and its websocket client installed into simfoundry
#               (1_eval_policy_og_scene.py imports it unconditionally; the installer leaves it out).
#   ckpts       the public DROID joint-position checkpoints of pi0 and pi0.5 from gs://openpi-assets,
#               fetched with openpi's own downloader into its cache, where serve_policy.py looks.
#
# Each step leaves a marker in /content/setup_state and is skipped on a rerun. Logs: /content/logs.
# Usage: bash repro/policy_eval/setup_colab.sh [all|vulkan|simfoundry|openpi|ckpts]
set -uo pipefail
REPO="${REPO:-/content/SimFoundry}"
LOG=/content/logs
ST=/content/setup_state
export HOME=/root
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/root/miniforge3}"
export PATH="/root/.local/bin:${MAMBA_ROOT_PREFIX}/bin:${PATH}"
export PIP_NO_CACHE_DIR=1
export OMNI_KIT_ACCEPT_EULA=YES
OPENPI_DIR=/content/openpi
OPENPI_COMMIT=15a9616a00943ada6c20a0f158e3adb39df2ccac
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/content/openpi_cache}"
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

install_miniforge() {
  [[ -x "${MAMBA_ROOT_PREFIX}/bin/mamba" ]] && return 0
  curl -fsSL -o /content/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /content/miniforge.sh -b -p "${MAMBA_ROOT_PREFIX}"
}

preclone() {  # preclone <url> <dest>: GitHub answered one anonymous clone with a 401 on 2026-10-16
  local url="$1" dest="$2" i
  [[ -d "${dest}/.git" ]] && return 0
  for i in 1 2 3 4 5; do git clone "${url}" "${dest}" && return 0; rm -rf "${dest}"; sleep $((i * 20)); done
  return 1
}

simfoundry() {
  local C="${REPO}/repro/g4/constraints_simfoundry.txt"
  step miniforge install_miniforge || return 1
  step preclone_b1k preclone https://github.com/StanfordVL/BEHAVIOR-1K.git "${REPO}/deps/BEHAVIOR-1K" || return 1
  # The installer pins CUDA 12.8 and stops when the image's nvcc reports another release (Colab images carry
  # 13.0 since 2026-10-17); with no nvcc on PATH it installs the 12.8 toolkit into the env itself.
  local path_no_cuda
  path_no_cuda="$(tr ':' '\n' <<< "${PATH}" | grep -v '/cuda' | paste -sd: -)"
  step env_simfoundry env PATH="${path_no_cuda}" PIP_CONSTRAINT="${C}" UV_CONSTRAINT="${C}" \
    bash "${REPO}/scripts/installation/install_simfoundry.sh" --project-root "${REPO}" --env-name simfoundry \
    --cuda-version 12.8 --default || return 1
  mamba clean -a -y > /dev/null 2>&1; rm -rf /root/.cache/pip
}

openpi_server() {
  preclone https://github.com/Physical-Intelligence/openpi.git "${OPENPI_DIR}" || return 1
  git -C "${OPENPI_DIR}" checkout --detach "${OPENPI_COMMIT}" || return 1
  # lerobot is a git dependency whose LFS files the server does not need.
  # --python: without it uv pip targets the first python on PATH, miniforge's base, whose Python 3.14
  # (miniforge's latest since 2026-10) has no torch 2.7.1 wheels.
  (cd "${OPENPI_DIR}" && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python -e .)
}

openpi_client() {
  "${MAMBA_ROOT_PREFIX}/envs/simfoundry/bin/python" -m pip install -e "${OPENPI_DIR}/packages/openpi-client" \
    -c "${REPO}/repro/g4/constraints_simfoundry.txt"
  "${MAMBA_ROOT_PREFIX}/envs/simfoundry/bin/python" -c "import openpi_client.websocket_client_policy"
}

openpi() {
  step openpi_server openpi_server || return 1
  step openpi_client openpi_client || return 1
}

# fetch_ckpt <name>: gs://openpi-assets/checkpoints/<name>, and the norm-stats directory that the matching
# <name>_polaris config reads while it is built, into $OPENPI_DATA_HOME. Anonymously: on a Colab VM gcsfs
# picks up the VM's metadata-server credentials and the bucket answers 401, and openpi only routes this
# bucket through gsutil, which is not on PATH there. openpi's tokenizer download passes the same option.
# The server then finds both in the cache and never goes to the network for them. The policy itself
# normalizes with the checkpoint's own assets/droid/norm_stats.json (policy_config.create_trained_policy).
fetch_ckpt() {
  (cd "${OPENPI_DIR}" && uv run python -c "
from openpi.shared import download
for url in ('gs://openpi-assets/checkpoints/$1', 'gs://openpi-assets/checkpoints/polaris/$1_polaris/assets/droid'):
    print(download.maybe_download(url, gs={'token': 'anon'}))")
}

ckpts() {
  step ckpt_pi0 fetch_ckpt pi0_droid_jointpos || return 1
  step ckpt_pi05 fetch_ckpt pi05_droid_jointpos || return 1
}

case "${1:-all}" in
  vulkan) bash "${REPO}/repro/g4/setup_g4.sh" vulkan ;;
  simfoundry) simfoundry ;;
  openpi) openpi ;;
  ckpts) ckpts ;;
  all)
    # The Vulkan libraries, the conda env and the uv project live in disjoint directories. The openpi
    # client goes into the conda env, so it waits for that.
    bash "${REPO}/repro/g4/setup_g4.sh" vulkan & v=$!
    (step openpi_server openpi_server && ckpts) & o=$!
    simfoundry; s=$?
    wait "${v}"; wait "${o}"; w=$?
    [[ ${s} -eq 0 && ${w} -eq 0 ]] && step openpi_client openpi_client
    echo "[setup] $(date +%T) all parts finished (simfoundry rc=${s}, openpi rc=${w})"
    ;;
  *) echo "usage: $0 [all|vulkan|simfoundry|openpi|ckpts]" >&2; exit 2 ;;
esac
