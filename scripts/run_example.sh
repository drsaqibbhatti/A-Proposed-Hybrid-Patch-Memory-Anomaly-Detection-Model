#!/usr/bin/env bash
set -e

python train.py \
  --train-dir /path/to/normal_train \
  --val-normal-dir /path/to/normal_val \
  --out-dir runs/aegis_ad \
  --image-size 256 \
  --teacher wide_resnet50_2 \
  --epochs 20 \
  --batch-size 8

python eval.py \
  --ckpt runs/aegis_ad/best_model.pt \
  --defect-dir /path/to/defect_images \
  --out-dir runs/aegis_ad/eval_defects \
  --save-maps \
  --save-overlays
