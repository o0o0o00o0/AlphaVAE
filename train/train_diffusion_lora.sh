TIME_STR=$(date "+%Y-%m-%d_%H-%M-%S")
MASTER_PORT=$((RANDOM % (30000 - 20000 + 1) + 20000))

echo MASTER_PORT=$MASTER_PORT

export NCCL_IB_DISABLE=1

VAE_dir="/data/raw/zhanjiabo/models/Z-Image/rgba_vae_zimage_v1.1"

# Model Configuration
MODEL_ARGS=(
    --pretrained_model_name_or_path "/data/raw/zhanjiabo/models/Z-Image"
    --pretrained_vae_model ${VAE_dir}
    --guidance_scale 5.0
)

output_dir=${VAE_dir}/trained-zimage-${TIME_STR}
# Output Configuration
OUTPUT_ARGS=(
    --output_dir "${output_dir}"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --instance_prompt "RGBA"
    --dataset_name "/data/raw/zhanjiabo/dataset/AlphaVAE"
    --caption_column "caption"
    --image_column "image"
    --resolution 1024
)

# Training Configuration
TRAIN_ARGS=(
    --rank 64
    --num_train_epochs 30
    --seed "42"
    --use_8bit_adam
    --optimizer "adamW"
    --learning_rate 1e-4
    --lr_scheduler "constant"
    --lr_warmup_steps 100
    --train_batch_size 1
    --gradient_accumulation_steps 4
    --mixed_precision "bf16" 
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 1000
)

# Validation Configuration
VALIDATION_ARGS=(
    --validation_steps 500
    --validation_prompt "Burning firewood"
)

accelerate launch --config_file=./configs/accelerate_config.yaml --main_process_port=$MASTER_PORT train/train_diffusion_lora.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"