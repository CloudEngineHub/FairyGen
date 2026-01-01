#!/bin/bash

# modify to your model path
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
DATASET_NAME="imagefolder"

# modify to your data path
TRAIN_DATA_DIR="./data/pig"
OUTPUT_PATH="./lora-libraries/pig"

[ ! -d "$OUTPUT_PATH" ] && mkdir -p "$OUTPUT_PATH"


cmd="accelerate launch train.py \
 --pretrained_model_name_or_path $MODEL_NAME \
 --pretrained_vae_model_name_or_path $VAE_NAME \
 --dataset_name $DATASET_NAME \
 --train_data_dir $TRAIN_DATA_DIR \
 --output_dir $OUTPUT_PATH \
 --resolution 1024 --random_flip \
 --rank 32 \
 --train_batch_size 1 \
 --learning_rate 1e-4 \
 --lr_scheduler constant \
 --lr_warmup_steps 0 \
 --max_train_steps 400 \
 --checkpointing_steps 400 \
 --seed 42 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --mixed_precision fp16"

echo "Running command: $cmd"
eval $cmd

echo -ne "-------------------- Finished executing script --------------------\n\n"