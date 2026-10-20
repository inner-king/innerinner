#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Turn one retrained background splat (repro/g4/retrain_bg.sh) into a loadable scene and render it from
# the real camera and the robot's cameras (render_bg_views.py), on a machine with the 3dgrut and
# simfoundry envs. Expects Data/<scene>/auto_bg/splat_<variant>/export/<scene>_bg.ply and its step-6
# sidecar (.pose.json) copied from the runtime. Writes assets/scenes/<scene>_<variant>/ (step 7, with
# its own out dir so the scene's default background stays as it was) and the view grid. A splat cut down to the
# desk by desk_room_splat.py carries a third sidecar (.room.json); its plain room is then added to the scene
# (add_plain_room.py), sized for the robot where the views put it.
# Usage: repro/g4/check_bg_variant.sh <scene> <variant> <pick> <place> <out.jpg>
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
cd "${REPO}"
SCENE="$1"; V="$2"; PICK="$3"; PLACE="$4"; OUT="$(realpath -m "$5")"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniforge3}"
export PATH="${MAMBA_ROOT_PREFIX}/envs/simfoundry/bin:${MAMBA_ROOT_PREFIX}/bin:${PATH}"
export GCLOUD_PROJECT="${GCLOUD_PROJECT:-local}" OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1
PY="${MAMBA_ROOT_PREFIX}/envs/simfoundry/bin/python"
PLY="${REPO}/Data/${SCENE}/auto_bg/splat_${V}/export/${SCENE}_bg.ply"
N=$(ls -d "Data/${SCENE}"/s1_video/frames_subsampled_* | sed 's/.*_//')
[[ -s "${PLY}" && -s "${PLY}.pose.json" ]] || { echo "missing ${PLY} or its .pose.json" >&2; exit 1; }
STATE="${REPO}/assets/scenes/${SCENE}_${V}/${SCENE}_${V}_scene_state_auto_bg.json"
"${PY}" scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/7_build_og_scene_assets.py \
  scene_name="${SCENE}" num_frames="${N}" s7_build_assets.bg_splat_ply="${PLY}" \
  s7_build_assets.out_scene_dir="${REPO}/assets/scenes/${SCENE}_${V}" \
  s7_build_assets.out_state_name="$(basename "${STATE}")"
if [[ -s "${PLY}.room.json" ]]; then
  "${PY}" "${HERE}/add_plain_room.py" "${STATE}" "${PLY}.room.json" --pick "${PICK}" --place "${PLACE}"
fi
"${PY}" "${HERE}/render_bg_views.py" --scene "${SCENE}" --scene-file "${STATE}" \
  --pick "${PICK}" --place "${PLACE}" --out "${OUT}" --label "${SCENE} ${V}"
