#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Retrain an auto_bg background splat so it holds up from the robot's cameras (repro/README.md, 17.7).
# Reuses the scene's VOID frames and both DA3 runs, and trains variants side by side, each in
# auto_bg/splat_<variant>, then bridges it into the OG world (step 6) and scores the held-out frames.
# Every variant first puts void-DA3 into the original DA3's scale (scale_void_da3.py) and reseeds.
#   sreg_sh0              scale regularization on (max/min axis 10), SH degree 0; else the recipe
#   sreg_sh0_depth2       the same, depth loss 0.5 -> 2.0, and splatfacto's (not -big's) culling of
#                         low-opacity Gaussians (alpha < 0.1 instead of 0.005, until densification stops)
#   sreg_sh0_depth2_30k   sreg_sh0_depth2 at 30k iterations instead of 80k
#   iso3_sh0_depth2_30k   sreg_sh0_depth2_30k with the regularization's max/min axis ratio 10 -> 3
# Steps 4-6 run with the nerfstudio env's python (setup_g4.sh splat explains why that is enough).
# Usage: repro/g4/retrain_bg.sh <scene> [variant...]   (on the runtime; logs in $SIMFOUNDRY_LOG_DIR)
set -uo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
cd "${REPO}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniforge3}"
export PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}" GCLOUD_PROJECT="${GCLOUD_PROJECT:-local}"
export PYTHONPATH="${REPO}:${PYLIB:-/content/pylib}${PYTHONPATH:+:${PYTHONPATH}}"
SCENE="$1"; shift
VARIANTS=("$@")
[[ ${#VARIANTS[@]} -gt 0 ]] || VARIANTS=(sreg_sh0 sreg_sh0_depth2 sreg_sh0_depth2_30k)
PY="${MAMBA_ROOT_PREFIX}/envs/nerfstudio_simfoundry/bin/python"
S=scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction
D="${REPO}/Data/${SCENE}"
AB="${D}/auto_bg"
L="${SIMFOUNDRY_LOG_DIR:-/content/logs}/retrain_${SCENE}"
mkdir -p "${L}"
N=$(ls -d "${D}"/s1_video/frames_subsampled_* | sed 's/.*_//')
COMMON=(scene_name="${SCENE}" num_frames="${N}")

[[ -s "${AB}/da3/void_sim3.npz" ]] || "${PY}" "${HERE}/scale_void_da3.py" "${D}" "${AB}/da3/void_sim3.npz" > "${L}/scale.log" 2>&1
[[ -s "${AB}/seed_sim3.ply" ]] || "${PY}" "${S}/4_build_seed_ply_from_void_da3.py" "${COMMON[@]}" \
  s4_seed_ply.void_da3_npz="${AB}/da3/void_sim3.npz" s4_seed_ply.out_ply="${AB}/seed_sim3.ply" > "${L}/seed.log" 2>&1
cat "${L}/scale.log"

model_args() {
  case "$1" in
    sreg_sh0) echo '["--pipeline.model.use-scale-regularization=True","--pipeline.model.sh-degree=0"]' ;;
    iso3_sh0_depth2*) echo '["--pipeline.model.use-scale-regularization=True","--pipeline.model.max-gauss-ratio=3.0","--pipeline.model.sh-degree=0","--pipeline.model.cull-alpha-thresh=0.1"]' ;;
    sreg_sh0_depth2*) echo '["--pipeline.model.use-scale-regularization=True","--pipeline.model.sh-degree=0","--pipeline.model.cull-alpha-thresh=0.1"]' ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}
depth_mult() { case "$1" in *_depth2*) echo 2.0 ;; *) echo 0.5 ;; esac; }
iters() { case "$1" in *_30k) echo 30000 ;; *) echo 80000 ;; esac; }

# gsplat JIT-compiles its CUDA extension on first use, into ~/.cache/torch_extensions. Variants
# started together on an empty cache build into the same directory at once and break each other's
# build (getcwd() failed, missing .o files), so build it once first, in ns-train's environment
# (ns_run.sh), under a lock that also covers a second scene's retrain_bg.sh started alongside.
flock /tmp/gsplat_build.lock "${HERE}/ns_run.sh" python -c \
  "from gsplat.cuda._backend import _C; assert _C is not None; print('gsplat CUDA extension ready')" \
  > "${L}/gsplat_build.log" 2>&1 || { echo "[retrain] gsplat build failed, see ${L}/gsplat_build.log"; exit 1; }

i=0
for v in "${VARIANTS[@]}"; do
  args="$(model_args "${v}")" || exit 2
  port=$((7100 + 10 * i)); i=$((i + 1))
  (
    t0=${SECONDS}
    "${PY}" "${S}/5_train_bg_splat.py" "${COMMON[@]}" s5_train_bg_splat.out_dirname="splat_${v}" \
      s5_train_bg_splat.depth_da3_npz="${AB}/da3/void_sim3.npz" s5_train_bg_splat.seed_ply_path="${AB}/seed_sim3.ply" \
      s5_train_bg_splat.depth_loss_mult="$(depth_mult "${v}")" s5_train_bg_splat.max_num_iterations="$(iters "${v}")" \
      "s5_train_bg_splat.extra_model_args=${args}" s5_train_bg_splat.viewer_port="${port}" > "${L}/train_${v}.log" 2>&1 \
    && "${PY}" "${S}/6_bridge_bg_splat_to_og.py" "${COMMON[@]}" \
      s6_bridge.in_ply="${AB}/splat_${v}/export/${SCENE}_bg.ply" > "${L}/bridge_${v}.log" 2>&1 \
    && bash "${HERE}/eval_splat.sh" "${AB}/splat_${v}" "${L}/eval_${v}.json" > "${L}/eval_${v}.log" 2>&1
    rc=$?  # before the echo: its $(date) would reset $?
    echo "[retrain] $(date +%T) ${SCENE} ${v} rc=${rc} ($((SECONDS - t0)) s)"
  ) &
done
wait
echo "[retrain] $(date +%T) ${SCENE} done"
