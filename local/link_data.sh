#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Recreate this machine's links into the /data partition (local/MOVED_FROM.txt) for a checkout and
# for the user caches. Data and weights were moved there on 2026-10-16 and replaced by symlinks; a
# fresh clone, or a reinstalled deps/, needs the same links again.
#
# A correct link is reported as ok, a missing one is created, and anything else in the way (a
# different link, a real file or directory) is reported and left alone. deps/ links are made only
# where that dependency is installed.
# Usage: local/link_data.sh [--dry-run] [checkout]      (default checkout: the one holding this file)
set -euo pipefail
DRY=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY=true; shift; fi
CHECKOUT="$(cd "${1:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}" && pwd)"

links=(  # <path>|<target>; relative paths are inside the checkout
  "checkpoints|/data/models/simfoundry-checkpoints"
  "Data|/data/datasets/simfoundry"
  "deps/void-model/CogVideoX-Fun-V1.5-5b-InP|/data/models/void-model/CogVideoX-Fun-V1.5-5b-InP"
  "deps/void-model/void_pass1.safetensors|/data/models/void-model/void_pass1.safetensors"
  "deps/void-model/void_pass2.safetensors|/data/models/void-model/void_pass2.safetensors"
  "deps/ml-depth-pro/checkpoints|/data/models/ml-depth-pro-checkpoints"
  "deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets|/data/datasets/behavior-1k/omnigibson-robot-assets"
  "deps/BEHAVIOR-1K/datasets/real2sim-assets|/data/datasets/behavior-1k/real2sim-assets"
  "deps/BEHAVIOR-1K/datasets/custom-assets|/data/datasets/behavior-1k/custom-assets"
  "${HOME}/.cache/huggingface/hub|/data/models/huggingface/hub"
  "${HOME}/.cache/hy3dgen|/data/models/hy3dgen"
  "${HOME}/.cache/torch|/data/models/torch"
)

declare -A reported=()
for entry in "${links[@]}"; do
  path="${entry%%|*}"; target="${entry#*|}"
  [[ "${path}" == /* ]] || path="${CHECKOUT}/${path}"
  parent="$(dirname "${path}")"
  if [[ ! -e "${target}" ]]; then
    echo "missing target  ${target} (for ${path})"
  elif [[ -L "${path}" ]]; then
    current="$(readlink "${path}")"
    if [[ "${current}" == "${target}" ]]; then echo "ok              ${path}"
    else echo "different link  ${path} -> ${current} (expected ${target})"; fi
  elif [[ -e "${path}" ]]; then
    echo "in the way      ${path} is a real $( [[ -d "${path}" ]] && echo directory || echo file ); left alone"
  elif [[ "${path}" == "${CHECKOUT}/deps/"* && ! -d "${parent}" ]]; then
    [[ -n "${reported[${parent}]:-}" ]] || echo "not installed   ${parent}"
    reported["${parent}"]=1
  elif ${DRY}; then
    echo "would link      ${path} -> ${target}"
  else
    mkdir -p "${parent}"
    ln -s "${target}" "${path}"
    echo "linked          ${path} -> ${target}"
  fi
done
