#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# basicsr 1.4.2 in the hunyuan env imports torchvision.transforms.functional_tensor, which
# torchvision 0.17 removed (the env has 0.22), so importing it fails. Point that one import at
# torchvision.transforms.functional, where rgb_to_grayscale lives now. This edits the installed
# package, so a rebuilt hunyuan env needs it again; running it twice changes nothing.
# Usage: local/fix_hunyuan_basicsr.sh [env prefix]      (default ~/miniforge3/envs/hunyuan)
set -euo pipefail
ENV="${1:-${MAMBA_ROOT_PREFIX:-${HOME}/miniforge3}/envs/hunyuan}"
# find_spec locates the package without importing it (importing is what fails).
f="$("${ENV}/bin/python" -c 'import importlib.util as u; print(u.find_spec("basicsr").submodule_search_locations[0])')/data/degradations.py"
if grep -q "torchvision.transforms.functional_tensor import" "$f"; then
  sed -i 's/torchvision\.transforms\.functional_tensor import/torchvision.transforms.functional import/' "$f"
  echo "patched $f"
else
  echo "already fixed: $f"
fi
"${ENV}/bin/python" -c "import basicsr.data.degradations" && echo "basicsr imports"
