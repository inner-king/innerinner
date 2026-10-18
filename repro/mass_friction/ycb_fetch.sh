#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetch YCB google_16k models for the stage 11 mass benchmark (repro/README.md, chapter 29), each archive unpacked
# into its own empty directory: <dir>/<name>/x/<name>/google_16k/textured.obj.
#   repro/mass_friction/ycb_fetch.sh ~/.cache/ycb_google_16k 037_scissors 025_mug ...
set -euo pipefail
dir="$1"; shift
base=http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google
for name in "$@"; do
  d="${dir}/${name}"
  [[ -s "${d}/x/${name}/google_16k/textured.obj" ]] && continue
  mkdir -p "${d}/x"
  curl -sf --retry 3 -o "${d}/model.tgz" "${base}/${name}_google_16k.tgz"
  tar -xzf "${d}/model.tgz" -C "${d}/x" && rm "${d}/model.tgz"
  echo "[ycb] ${name}"
done
