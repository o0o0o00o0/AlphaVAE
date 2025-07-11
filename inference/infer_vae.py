import os
import argparse
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from accelerate import Accelerator
from diffusers import AutoencoderKL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_vae_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["float32", "fp16", "bf16"])
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir) or len(os.listdir(args.input_dir)) == 0:
        print(f"No images found in ${args.input_dir}.")
        return

    dtype_map = {
        "float32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    weight_dtype = dtype_map[args.dtype]

    os.makedirs(args.output_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_path).to('cuda', dtype=weight_dtype)
    vae.eval()

    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    for image_path in os.listdir(args.input_dir):
        img = Image.open(os.path.join(args.input_dir, image_path)).convert("RGBA")
        x = transform(img).unsqueeze(0).to(device='cuda', dtype=weight_dtype)
        recon = vae(x).sample
        recon = (recon * 0.5 + 0.5).clamp(0, 1)

        # Save each reconstructed image
        save_path = os.path.join(args.output_dir, f"recon_{image_path}")
        save_image(recon.cpu(), save_path)

    print("✅ Inference complete.")


if __name__ == "__main__":
    main()
