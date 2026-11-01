#!/usr/bin/env bash
# ACT training on the training L4: train_r3.sh <kind: state|image> <tag> <dataset root> <steps> <batch> [lerobot-train args...]
# Output $O/train/act_<kind><tag>, log $O/train_<kind><tag>.log (last line TRAIN-<kind><tag>-DONE rc=<n>).
set -u
kind=$1; tag=$2; root=$3; steps=$4; batch=$5; shift 5
B=${LEROBOT_BIN:-/root/miniforge3/envs/lerobot/bin}
O=${OUT_ROOT:-/content}
rm -rf $O/train/act_$kind$tag
$B/lerobot-train --dataset.repo_id=local/tofu_$kind --dataset.root=$root/tofu_$kind --policy.type=act \
  --policy.push_to_hub=false --policy.device=cuda --policy.chunk_size=20 --policy.n_action_steps=10 \
  --output_dir=$O/train/act_$kind$tag --steps=$steps --batch_size=$batch --save_freq=2000 --log_freq=200 \
  --wandb.enable=false --seed=0 "$@" > $O/train_$kind$tag.log 2>&1
echo "TRAIN-$kind$tag-DONE rc=$? $(date +%T)" >> $O/train_$kind$tag.log
