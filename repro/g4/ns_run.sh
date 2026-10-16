#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run a command in the nerfstudio_simfoundry env the way 5_train_bg_splat.py's _ns_env() runs ns-train:
# CUDA headers on the host compiler's path and an explicit arch list for gsplat's JIT build, no
# torch.compile, and torch.load with weights_only=False for nerfstudio 1.1.5 checkpoints under
# torch >= 2.6. Without the first two the build fails (cuda_runtime_api.h not found;
# cooperative_groups::labeled_partition missing for the pre-sm_70 targets of an empty arch list).
# Usage: repro/g4/ns_run.sh <command...>
set -euo pipefail
P="${MAMBA_ROOT_PREFIX:-${HOME}/miniforge3}"
E="${P}/envs/nerfstudio_simfoundry"
exec env -u CUDA_HOME -u CC -u CXX -u NVCC_PREPEND_FLAGS -u NVCC_APPEND_FLAGS -u CFLAGS -u CXXFLAGS \
  CPLUS_INCLUDE_PATH="${E}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}" \
  TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;12.0" TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  "${P}/bin/mamba" run -n nerfstudio_simfoundry "$@"
