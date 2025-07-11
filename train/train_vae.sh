TIME_STR=$(date "+%Y-%m-%d_%H-%M-%S")
MASTER_PORT=$((RANDOM % (30000 - 20000 + 1) + 20000))

echo MASTER_PORT=$MASTER_PORT
export NCCL_IB_DISABLE=1

WANDB_MODE=offline \
setting=vae_main
VAE_dir="path/to/your/vae_model"
train_data_dir="/path/to/your/train_datasets"
pretrained_vae_path="/path/to/your/pretrained/vae"
accelerate launch --config_file=./configs/accelerate_config.yaml \
    --num_processes=8 --main_process_port=$MASTER_PORT train/train_vae.py \
    --pretrained_path ${pretrained_vae_path} \
    --train_data_dir ${train_data_dir} \
    --num_eval 8 \
    --output_dir ${VAE_dir} \
    --train_batch_size 2 \
    --num_train_epochs 20 \
    --gan_start_step 4000 \
    --learning_rate 1.5e-5 \
    --resolution 1024 \
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --checkpointing_steps 5000 \
    --validation_steps 2000 \
    --mixed_precision bf16 \
    --config exp/${setting}.yaml

    # ATTENTION: The following arguments are essential if using the corresponding module!
    #      --naive_mse_loss:  Replace our loss with naive MSE loss (default: false)
    #      --gan_start_step:  Start to use GAN loss at this step (default: None)
    #      --ref_kl_scale:  Scale for KL loss (default: None)
    #      --lpips_scale:  Scale for LPIPS loss (default: None)
    #      --kl_scale:  Scale for KL loss (default: None)