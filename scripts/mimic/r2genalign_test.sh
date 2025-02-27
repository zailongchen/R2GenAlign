#!/bin/bash

dataset="mimic_cxr"
annotation="/home/guest/czl/dataset/mimic_cxr/mimic_annotation_cxr_view.json"
base_dir="/home/guest/czl/dataset/mimic_cxr/images"
delta_file=""

version="v1_test"
savepath="./save/$dataset/$version"

python3 -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 12 \
    --max_length 115 \
    --min_new_tokens 95 \
    --max_new_tokens 135 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --num_workers 8 \
    --devices 1 \
    --limit_test_batches 1.0 \
    2>&1 |tee -a ${savepath}/log_mimic_test.txt
