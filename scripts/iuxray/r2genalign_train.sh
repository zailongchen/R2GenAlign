#!/bin/bash

dataset="iu_xray"
annotation="/home/guest/czl/dataset/iu_xray/annotation_cxr_view.json"
base_dir="/home/guest/czl/dataset/iu_xray/images"
# delta_file="/home/guest/czl/R2GenExplore/save/mimic_cxr/report/report_bir2gen_mae_full_model/checkpoints/checkpoint_epoch4_step152316_bleu0.171899_cider0.251643_ori.pth"

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
    --llm_use_lora True \
    --savedmodel_path ${savepath} \
    --max_length 75 \
    --min_new_tokens 55 \
    --max_new_tokens 115 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 4 \
    --devices 2 \
    --max_epochs 20 \
    --limit_train_batches 1.0 \
    --limit_val_batches 1.0 \
    --val_check_interval 1 \
    # --num_sanity_val_steps 2 \
    2>&1 |tee -a ${savepath}/log_mimic.txt
