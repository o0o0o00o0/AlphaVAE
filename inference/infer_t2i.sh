# modify pretrained_model_name_or_path to your path
vae_dir="./models/finetune_VAE"
diffusion_dir="${vae_dir}/finetune_diffusion"

python inference/infer_t2i.py \
    --pretrained_model_name_or_path "./models/FLUX.1-dev" \
    --pretrained_vae_model "${vae_dir}" \
    --lora_path "${diffusion_dir}" \
    --prompts "A white cat with brown and black patches sits gracefully, gazing forward." \
    --output_dir "./results/t2i" \
    --num_images_per_prompt 3
    
    
