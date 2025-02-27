#!/bin/bash

dataset="mimic_cxr"
annotation="data/mimic_cxr/mimic_annotation_cxr_view.json"
base_dir="data/mimic_cxr/images"
# delta_file=""

version="r2gen_align"
savepath="./save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python3 -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 12 \
    --val_batch_size 12 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 115 \
    --min_new_tokens 95 \
    --max_new_tokens 135 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 4 \
    --devices 2 \
    --max_epochs 10 \
    --limit_train_batches 1.0 \
    --limit_val_batches 1.0 \
    --val_check_interval 0.5 \
    # --num_sanity_val_steps 2 \
    2>&1 |tee -a ${savepath}/log_mimic.txt
