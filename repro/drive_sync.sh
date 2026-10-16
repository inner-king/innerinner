#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copy results and checkpoints between a checkout and the project's Google Drive folder, with rclone.
#
# Drive folder layout (repro/README.md, chapter 14)       local side
#   results/results_view/       final videos and images    <repo>/results/results_view/
#   results/diagnostics/        check images, comparisons  <repo>/results/diagnostics/
#   results/logs/               run logs                   ${SIMFOUNDRY_LOG_DIR:-~/simfoundry_logs}/
#   results/model_calls/        model response cache       <repo>/.cache/simfoundry/
#   results/scenes/<scene>.tar  one scene's outputs        <repo>/Data/<scene> (+ assets/scenes/<scene>)
#   checkpoints/                weights we trained         anywhere
#
# Usage:
#   repro/drive_sync.sh push results|logs|model_calls
#   repro/drive_sync.sh pull results|logs|model_calls [<subpath>]
#   repro/drive_sync.sh push scene <scene>
#   repro/drive_sync.sh pull scene <scene>
#   repro/drive_sync.sh push checkpoint <local file or dir> <path under checkpoints/>
#   repro/drive_sync.sh pull checkpoint <path under checkpoints/> <local dir>
#   repro/drive_sync.sh ls [<path>]
#
# Pushes copy and never delete: a file removed locally stays on the Drive. A scene tar leaves out
# *.ckpt (those go under checkpoints/) and replaces the previous tar whole; a pull does not overwrite
# an existing Data/<scene>. rclone verifies every transfer by size and MD5.
#
# INNERINNER_DRIVE is the rclone path of the folder. The default, innerking:innerinner, is the
# "innerinner" folder of an rclone remote named "innerking" whose root is the Inner-King folder;
# local/env.sh may set it.
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
[[ -f "${REPO}/local/env.sh" ]] && source "${REPO}/local/env.sh"
DRIVE="${INNERINNER_DRIVE:-innerking:innerinner}"
command -v rclone >/dev/null || { echo "rclone not found on PATH" >&2; exit 127; }
RCLONE=(rclone --stats-one-line --stats 30s)

usage() { grep '^#   repro/drive_sync.sh' "${BASH_SOURCE[0]}" | sed 's/^#  //' >&2; exit 2; }

local_dir() {
  case "$1" in
    results) echo "${REPO}/results" ;;
    logs) echo "${SIMFOUNDRY_LOG_DIR:-${HOME}/simfoundry_logs}" ;;
    model_calls) echo "${REPO}/.cache/simfoundry" ;;
  esac
}

remote_dir() {
  case "$1" in
    results) echo "${DRIVE}/results" ;;
    logs) echo "${DRIVE}/results/logs" ;;
    model_calls) echo "${DRIVE}/results/model_calls" ;;
  esac
}

tmp=""
trap '[[ -n "${tmp}" ]] && rm -f "${tmp}"' EXIT

[[ $# -ge 1 ]] || usage
action="$1"; kind="${2:-}"
case "${action} ${kind}" in
  "push results" | "push logs" | "push model_calls")
    [[ $# -eq 2 ]] || usage
    "${RCLONE[@]}" copy "$(local_dir "${kind}")" "$(remote_dir "${kind}")"
    ;;
  "pull results" | "pull logs" | "pull model_calls")
    [[ $# -le 3 ]] || usage
    sub="${3:-}"
    "${RCLONE[@]}" copy "$(remote_dir "${kind}")${sub:+/${sub}}" "$(local_dir "${kind}")${sub:+/${sub}}"
    ;;
  "push scene")
    [[ $# -eq 3 ]] || usage
    scene="$3"
    [[ -d "${REPO}/Data/${scene}" ]] || { echo "no ${REPO}/Data/${scene}" >&2; exit 1; }
    paths=("Data/${scene}")
    [[ -d "${REPO}/assets/scenes/${scene}" ]] && paths+=("assets/scenes/${scene}")
    tmp="$(mktemp --tmpdir "${scene}.XXXXXX.tar")"
    echo "packing ${paths[*]} (without *.ckpt)"
    tar -C "${REPO}" --exclude='*.ckpt' -cf "${tmp}" "${paths[@]}"
    "${RCLONE[@]}" copyto "${tmp}" "${DRIVE}/results/scenes/${scene}.tar"
    ;;
  "pull scene")
    [[ $# -eq 3 ]] || usage
    scene="$3"
    [[ ! -e "${REPO}/Data/${scene}" ]] || { echo "${REPO}/Data/${scene} exists; move it away first" >&2; exit 1; }
    tmp="$(mktemp --tmpdir "${scene}.XXXXXX.tar")"
    "${RCLONE[@]}" copyto "${DRIVE}/results/scenes/${scene}.tar" "${tmp}"
    mkdir -p "${REPO}/Data"
    tar -C "${REPO}" -xf "${tmp}"
    echo "extracted Data/${scene}$([[ -d "${REPO}/assets/scenes/${scene}" ]] && echo " and assets/scenes/${scene}") into ${REPO}"
    ;;
  "push checkpoint")
    [[ $# -eq 4 ]] || usage
    "${RCLONE[@]}" copy "$3" "${DRIVE}/checkpoints/$4"  # a file lands inside that folder
    ;;
  "pull checkpoint")
    [[ $# -eq 4 ]] || usage
    "${RCLONE[@]}" copy "${DRIVE}/checkpoints/$3" "$4"
    ;;
  "ls "*)
    [[ $# -le 2 ]] || usage
    rclone lsf --max-depth 2 "${DRIVE}${kind:+/${kind}}"
    ;;
  *) usage ;;
esac
