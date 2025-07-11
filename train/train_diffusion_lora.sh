TIME_STR=$(date "+%Y-%m-%d_%H-%M-%S")
MASTER_PORT=$((RANDOM % (30000 - 20000 + 1) + 20000))

echo MASTER_PORT=$MASTER_PORT

export NCCL_IB_DISABLE=1

VAE_dir="/path/to/your/model"

# Model Configuration
MODEL_ARGS=(
    --pretrained_model_name_or_path "/path/to/your/pretrained_model"
    --pretrained_vae_model ${VAE_dir}
    --guidance_scale 1
)

output_dir=${VAE_dir}/trained-flux-${TIME_STR}
# Output Configuration
OUTPUT_ARGS=(
    --output_dir "${output_dir}"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --instance_prompt "RGBA"
    --dataset_name "/path/to/your/train_dataset"
    --caption_column "caption"
    --image_column "image"
    --resolution 1024
)

# Training Configuration
TRAIN_ARGS=(
    --rank 64
    --num_train_epochs 20
    --seed "42"
    --optimizer "prodigy"
    --learning_rate 1.
    --lr_scheduler "constant"
    --lr_warmup_steps 0

    --train_batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16" 
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 100000
)

# Validation Configuration
VALIDATION_ARGS=(
    --validation_steps 5000
    --validation_prompt "Burning firewood"
)

accelerate launch --config_file=./configs/accelerate_config.yaml --main_process_port=$MASTER_PORT train/train_diffusion_lora.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"