# modify pretrained_model_name_or_path to your path
vae_dir="/data/raw/zhanjiabo/models/Z-Image/rgba_vae_zimage_v1.1/"
diffusion_dir="/data/raw/zhanjiabo/models/Z-Image/rgba_vae_zimage_v1.1/trained-zimage-2026-03-01_23-10-25/checkpoint-15000"

python inference/infer_t2i.py \
    --pretrained_model_name_or_path "/data/raw/zhanjiabo/models/Z-Image" \
    --pretrained_vae_model "${vae_dir}" \
    --lora_path "${diffusion_dir}" \
    --prompts "A pair of sunglasses." \
    --output_dir "./results/t2i" \
    --num_images_per_prompt 3
    
    
