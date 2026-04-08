vae_dir="/data/raw/zhanjiabo/models/Z-Image/rgba_vae_zimage_v1.1/checkpoint-40000/hf"
input_dir="./assets/origin_images"
output_dir="./results/vae_results_v1.1_checkpoint-40000"

python inference/infer_vae.py \
  --pretrained_vae_path ${vae_dir}  \
  --input_dir ${input_dir} \
  --output_dir ${output_dir} \
  --resolution 1024 \
  --dtype bf16
