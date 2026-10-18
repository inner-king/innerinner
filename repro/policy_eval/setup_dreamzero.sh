#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DreamZero-DROID's inference server on a Colab G4 runtime (repro/README.md, chapter 19), the fourth zero-shot policy
# of Table G.1, next to the simfoundry env that setup_colab.sh builds for the episodes. Runs on the VM as root after
# `setup_colab.sh miniforge`; run_table_g1.sh then starts the server for a policy of `kind: dreamzero`.
#
#   conda  the Python 3.11 env alone, which `all` creates first. Run it before starting setup_colab.sh simfoundry
#          next to this script, so that the two never write miniforge's package cache at once.
#   env    dreamzero0/dreamzero at a fixed commit, installed into that env as its README says (pip install -e .
#          against PyTorch's cu129 index: torch 2.8.0, which has sm_120 kernels), with three differences.
#          - `gear` is dropped from its dependencies: the PyPI package of that name is an unrelated project that
#            DreamZero never imports (dreamzero issue #95).
#          - deepspeed, which it lists unpinned, is removed again after the install: 0.19.7 (2026-10) and the
#            pinned transformers 4.51.3 import each other in a circle, so the server died on its first
#            `import transformers`. Only the training code (groot/vla/experiment/) uses deepspeed.
#          - flash-attn and Transformer Engine are not installed: the README builds the first for H100 and the
#            second for GB200 only. Without either, the attention module falls back to torch's
#            scaled_dot_product_attention, which computes the same attention for the unpadded inputs the model
#            passes (it warns "Padding mask is disabled" if a call ever passes lengths).
#   ckpt   GEAR-Dreams/DreamZero-DROID (46 GB of bf16 safetensors) into /content/dz_ckpt, without tensorrt/ (the
#          GB200 engine). Its config.json leaves the action head's skip_component_loading at False, which makes
#          the head download Wan2.1-I2V-14B-480P's own DiT (about 65 GB) and load it into RAM before
#          from_pretrained overwrites it with the checkpoint's. The checkpoint holds every component (DiT, T5,
#          CLIP, VAE: model.safetensors.index.json), so the flag is set. The head still torch.loads Wan2.1's T5,
#          CLIP and VAE files before that (unconditionally), so those three are fetched into the HF cache too,
#          with the umt5-xxl tokenizer the transforms read. The checkpoint's own experiment_cfg/conf.yaml sets
#          eval_bf16, so the model goes to the GPU in bf16.
#
# Each step leaves a marker in /content/setup_state and is skipped on a rerun. Logs: /content/logs.
# Usage: bash repro/policy_eval/setup_dreamzero.sh [all|conda|env|ckpt]
set -uo pipefail
LOG=/content/logs
ST=/content/setup_state
export HOME=/root
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/root/miniforge3}"
export PATH="/root/.local/bin:${MAMBA_ROOT_PREFIX}/bin:${PATH}"
export PIP_NO_CACHE_DIR=1
# The HF cache the server reads the Wan2.1 files and the tokenizer from (run_table_g1.sh sets the same).
export HF_HOME="${HF_HOME:-/content/hf_home}"
DZ_DIR="${DZ_DIR:-/content/dreamzero}"
DZ_COMMIT=ab790c198fbce33503358efbbd4187ce9a89adf3
DZ_ENV="${MAMBA_ROOT_PREFIX}/envs/dreamzero"
CKPT_DIR=/content/dz_ckpt/DreamZero-DROID
HFVENV=/content/hfvenv
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

dz_conda() {
  [[ -x "${MAMBA_ROOT_PREFIX}/bin/mamba" ]] || { echo "no mamba: run setup_colab.sh miniforge first"; return 1; }
  [[ -x "${DZ_ENV}/bin/python" ]] || mamba create -y -n dreamzero python=3.11
}

dz_env() {
  local i
  # GitHub answered one anonymous clone with a 401 on 2026-10-16 (setup_colab.sh's preclone).
  for i in 1 2 3 4 5; do
    [[ -d "${DZ_DIR}/.git" ]] && break
    git clone https://github.com/dreamzero0/dreamzero.git "${DZ_DIR}" && break
    rm -rf "${DZ_DIR}"; sleep $((i * 20))
  done
  git -C "${DZ_DIR}" checkout --detach "${DZ_COMMIT}" || return 1
  sed -i '/^    "gear",$/d' "${DZ_DIR}/pyproject.toml"
  grep -q '"gear"' "${DZ_DIR}/pyproject.toml" && { echo "gear still listed"; return 1; }
  [[ -x "${DZ_ENV}/bin/python" ]] || { echo "no dreamzero env: run the conda step first"; return 1; }
  (cd "${DZ_DIR}" && "${DZ_ENV}/bin/python" -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129) \
    || return 1
  "${DZ_ENV}/bin/python" -m pip uninstall -y deepspeed || return 1
  "${DZ_ENV}/bin/python" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'archs', torch.cuda.get_arch_list())
assert 'sm_120' in torch.cuda.get_arch_list(), 'this torch build has no Blackwell (sm_120) kernels'
import groot.vla.model.n1_5.sim_policy, openpi_client, tyro  # what the server script imports
from groot.vla.model.dreamzero.base_vla import VLA  # pulls in transformers' PreTrainedModel, as loading does
"
}

# hf_fetch <python statements>: run huggingface_hub in its own small venv, so the download need not wait for
# the dreamzero env.
hf_fetch() {
  [[ -x "${HFVENV}/bin/python" ]] || { python3 -m venv "${HFVENV}" && "${HFVENV}/bin/pip" install -q "huggingface_hub[hf_xet]"; } \
    || return 1
  "${HFVENV}/bin/python" -c "from huggingface_hub import snapshot_download, hf_hub_download
$1"
}

dz_ckpt() {
  hf_fetch "
print(snapshot_download('GEAR-Dreams/DreamZero-DROID', local_dir='${CKPT_DIR}', ignore_patterns=['tensorrt/*'],
                        max_workers=16))
for f in ('models_t5_umt5-xxl-enc-bf16.pth', 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth', 'Wan2.1_VAE.pth'):
    print(hf_hub_download('Wan-AI/Wan2.1-I2V-14B-480P', f))
print(snapshot_download('google/umt5-xxl', allow_patterns=['*.json', '*.model', '*.txt']))
" || return 1
  python3 - "${CKPT_DIR}/config.json" <<'EOF'
import json, sys
path = sys.argv[1]
cfg = json.load(open(path))
cfg["action_head_cfg"]["config"]["skip_component_loading"] = True
json.dump(cfg, open(path, "w"), indent=2)
print("skip_component_loading:", json.load(open(path))["action_head_cfg"]["config"]["skip_component_loading"])
EOF
  du -sh "${CKPT_DIR}" "${HF_HOME}"; df -h /content | tail -1
}

case "${1:-all}" in
  conda) step dz_conda dz_conda ;;
  env) step dz_env dz_env ;;
  ckpt) step dz_ckpt dz_ckpt ;;
  all)
    step dz_conda dz_conda || exit 1
    step dz_ckpt dz_ckpt & c=$!
    step dz_env dz_env; e=$?
    wait "${c}"; k=$?
    echo "[setup] $(date +%T) dreamzero finished (env rc=${e}, ckpt rc=${k})"
    [[ ${e} -eq 0 && ${k} -eq 0 ]]
    ;;
  *) echo "usage: $0 [all|conda|env|ckpt]" >&2; exit 2 ;;
esac
