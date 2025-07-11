from diffusers import (
    AutoencoderKL,
    FluxPipeline,
)
import argparse
import torch
from PIL import Image
import os
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model",
        type=str,
        default=None,
        required=False,
        help="Path to RGBA VAE model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="Seed for the random number generator.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        nargs="+",
        required=False,
        help="Prompts for the image generation.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        required=False,
        help="Path to the file containing the prompts for the image generation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Output directory for the generated images.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        required=True,
        help="Path to the LORA weights.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        required=False,
        help="Number of images to generate per prompt.",
    )
    return parser.parse_args(input_args)

def main(args):
    if args.pretrained_vae_model is not None:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_vae_model,
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
        )

    weight_dtype = torch.bfloat16
    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        vae=vae,
    )
    # load attention processors
    pipeline.load_lora_weights(args.lora_path)
    pipeline.to(device='cuda', dtype=weight_dtype)
    generator = torch.Generator('cuda').manual_seed(args.seed) if args.seed is not None else None

    prompts = args.prompts
    if args.prompt_file is not None:
        with open(args.prompt_file, 'r') as f:
            prompts = f.readlines()
    if prompts is None:
        raise ValueError("Either --prompts or --prompt_file must be provided.")
        
    os.makedirs(args.output_dir, exist_ok=True)
    for idx, prompt in enumerate(prompts):
        for i in range(args.num_images_per_prompt):
            image = pipeline(
                prompt=prompt,
                prompt_2=prompt,
                generator=generator,
            ).images[0]
            if args.num_images_per_prompt == 1:
                output_path = f"{args.output_dir}/{idx:05d}.png"
            else:
                output_path = f"{args.output_dir}/{idx:05d}_prompt_{i}_sample.png"
            image.save(output_path)
            print(f"generated image is save in {output_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    