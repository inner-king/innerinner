#!/usr/bin/env bash
# π0.5 LoRA fine-tuning on an L4 (option B): LeRobot 0.6.1 pi05 from lerobot/pi05_base, MEAN_STD state/action normalisation
# (the default quantiles blow up on our near-constant dimensions), LoRA r=16, batch 8, 10k steps, checkpoint every 2000 steps.
# Resume after a lost runtime: same env + data + cache, copy ckpt_pull/<runtime>/pi05_lora back, then
#   lerobot-train --config_path=/content/train/pi05_lora/checkpoints/<step>/pretrained_model/train_config.json --resume=true
set -u
exec > /content/pi05_train.out 2>&1
M=/root/miniforge3
[ -x $M/bin/mamba ] || { curl -fsSL -o /content/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash /content/mf.sh -b -p $M; }
[ -x $M/envs/lerobot/bin/python ] || $M/bin/mamba create -y -q -n lerobot -c conda-forge --override-channels python=3.12 "ffmpeg=7.1" > /dev/null
PY=$M/envs/lerobot/bin/python
$PY -m pip install -q "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -q "lerobot[pi,peft,dataset,training]==0.6.1"
echo "[pi05] $(date +%T) env ready"
export HF_HUB_CACHE=/content/hf_cache TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/pi05_base', revision='22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0')" > /dev/null
echo "[pi05] $(date +%T) base downloaded"
export HF_HUB_OFFLINE=1
$M/envs/lerobot/bin/lerobot-train --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.pretrained_revision=22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0 --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true --policy.tokenizer_max_length=80 --policy.device=cuda --policy.push_to_hub=false \
  '--policy.normalization_mapping={"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}' \
  --policy.scheduler_decay_steps=10000 --policy.scheduler_warmup_steps=500 \
  --dataset.repo_id=local/tofu_image --dataset.root=/content/ds_img/tofu_image --dataset.image_transforms.enable=true \
  --batch_size=8 --steps=10000 --log_freq=50 --save_freq=2000 --num_workers=8 --seed=0 \
  --output_dir=/content/train/pi05_lora --wandb.enable=false --peft.method_type=LORA --peft.r=16 > /content/train_pi05.log 2>&1
echo "TRAIN-pi05-DONE rc=$? $(date +%T)"
