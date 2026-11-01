#!/usr/bin/env bash
# Dicing ACT on a training L4: train_dice.sh <variant: A|B> <tag> <dataset root> <steps> <batch> [lerobot-train args...]
# Output $O/train/act_dice<variant><tag>, log $O/train_dice<variant><tag>.log (last line TRAIN-dice<variant><tag>-DONE rc=<n>).
# Chunk 40 ticks predicted and 20 executed (a cut period is about 2.5 s), the rest as train_r3.sh (the single-cut recipe).
set -u
variant=$1; tag=$2; root=$3; steps=$4; batch=$5; shift 5
B=${LEROBOT_BIN:-/root/miniforge3/envs/lerobot/bin}
O=${OUT_ROOT:-/content}
rm -rf $O/train/act_dice$variant$tag
$B/lerobot-train --dataset.repo_id=local/dice_$variant --dataset.root=$root/dice_$variant --policy.type=act \
  --policy.push_to_hub=false --policy.device=cuda --policy.chunk_size=40 --policy.n_action_steps=20 \
  --output_dir=$O/train/act_dice$variant$tag --steps=$steps --batch_size=$batch --save_freq=2000 --log_freq=200 \
  --wandb.enable=false --seed=0 "$@" > $O/train_dice$variant$tag.log 2>&1
echo "TRAIN-dice$variant$tag-DONE rc=$? $(date +%T)" >> $O/train_dice$variant$tag.log
