#!/bin/bash

cd LLaVA


export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=3

python ../scripts/cal_visnec.py \
    --model_name_or_path ./llava-v1.5-7b_vicuna \
    --data_path ./llava_v1_5_mix665k.json \
    --image_folder ./llava-v1.5-instruct \
    --pretrain_mm_mlp_adapter ./mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./results \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --model_max_length 2048 \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
