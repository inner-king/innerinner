#!/usr/bin/env bash
# Training-only L4: miniforge + the lerobot env (same pins as before). Last line LEROBOT-DONE rc=<n>.
set -u
exec > ${OUT_ROOT:-/content}/lerobot_setup.out 2>&1
M=${MINIFORGE:-/root/miniforge3}
[ -x $M/bin/mamba ] || { curl -fsSL -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash /tmp/mf.sh -b -p $M; }
$M/bin/mamba create -y -n lerobot -c conda-forge --override-channels python=3.12 "ffmpeg=7.1"
PY=$M/envs/lerobot/bin/python
$PY -m pip install -q --upgrade pip
$PY -m pip install -q "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -q "lerobot[smolvla,dataset,training]==0.6.1"
$PY -c "import torch, lerobot; print('torch', torch.__version__, torch.cuda.is_available(), '| lerobot', lerobot.__version__)"
echo "LEROBOT-DONE rc=$?"
