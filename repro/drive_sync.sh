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
# A scene tar keeps its symlinks as they are, and some stages link their inputs by absolute path
# (auto_bg/splat_30k/ns_data/images/frame_0107.png -> /home/<user>/<checkout>/Data/<scene>/...).
# The target travels in the same tar, but that path only resolves on the machine that packed it, so
# a pull turns every absolute link into the scene into a relative link to the copy it just
# extracted, and a push reports the absolute links it packs.
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

# scene_link_target <scene> <link target>: the checkout-relative path that an absolute link target
# names inside the scene's own trees (Data/<scene>, assets/scenes/<scene>), or nothing when it leads
# elsewhere. Only the part after the checkout counts, since the checkout's own path is per machine.
scene_link_target() {
  local scene="$1" target="$2" tree
  for tree in "Data/${scene}" "assets/scenes/${scene}"; do
    case "${target}" in
      */"${tree}") echo "${tree}"; return ;;
      */"${tree}"/*) echo "${tree}/${target#*/"${tree}"/}"; return ;;
    esac
  done
}

# relink_scene <scene>: after a pull, make each absolute link into the scene a relative link to the
# copy just extracted. Links that lead elsewhere are listed and left alone.
relink_scene() {
  local scene="$1" trees=() tree link target rel relinked=0 left=0 broken
  for tree in "Data/${scene}" "assets/scenes/${scene}"; do
    if [[ -d "${REPO}/${tree}" ]]; then trees+=("${REPO}/${tree}"); fi
  done
  (( ${#trees[@]} > 0 )) || return 0  # find with no start point would search the current directory
  while IFS= read -r -d '' link; do
    target="$(readlink "${link}")"
    rel="$(scene_link_target "${scene}" "${target}")"
    if [[ -z "${rel}" ]]; then
      echo "left as is: ${link#"${REPO}/"} -> ${target}" >&2
      left=$((left + 1))
      continue
    fi
    # -n replaces a link to a directory, instead of creating the new link inside that directory.
    ln -sfn -- "$(realpath -s -m --relative-to="$(dirname "${link}")" "${REPO}/${rel}")" "${link}"
    relinked=$((relinked + 1))
  done < <(find "${trees[@]}" -type l -lname '/*' -print0)
  broken="$(find "${trees[@]}" -xtype l | wc -l)"
  echo "relinked ${relinked} absolute link(s) to the extracted copy, left ${left} as is; ${broken} do not resolve"
  (( broken == 0 )) || find "${trees[@]}" -xtype l -printf '  %p -> %l\n' >&2
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
    inside=0; outside=()
    while IFS= read -r -d '' link; do
      target="$(readlink "${link}")"
      if [[ -n "$(scene_link_target "${scene}" "${target}")" ]]; then inside=$((inside + 1))
      else outside+=("  ${link#"${REPO}/"} -> ${target}"); fi
    done < <(find "${paths[@]/#/${REPO}/}" -type l -lname '/*' -print0)
    (( inside == 0 )) || echo "${inside} absolute link(s) into the scene are packed as they are; pull scene relinks them"
    if (( ${#outside[@]} > 0 )); then
      echo "warning: ${#outside[@]} absolute link(s) lead outside the scene and may not resolve on another machine:" >&2
      printf '%s\n' "${outside[@]}" >&2
    fi
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
    relink_scene "${scene}"
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
