#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Held-out image metrics (PSNR / SSIM / LPIPS on the 10% of frames splatfacto did not train on) of an
# auto_bg background splat, with nerfstudio's ns-eval in the training environment (ns_run.sh).
# Usage: repro/g4/eval_splat.sh <Data/<scene>/auto_bg/splat[_30k]> <out.json>
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
SPLAT="$(cd "$1" && pwd)"
OUT="$(realpath -m "$2")"
CFG="$(ls -t "${SPLAT}"/outputs/ns_data/splatfacto/*/config.yml | head -1)"
cd "${SPLAT}"
"${HERE}/ns_run.sh" ns-eval --load-config "${CFG#"${SPLAT}"/}" --output-path "${OUT}"
