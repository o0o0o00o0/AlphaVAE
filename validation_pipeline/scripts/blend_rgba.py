import argparse
import os
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

COLOR_MAP = {
    "gray": (128., 128., 128.),
    "black": (0., 0., 0.),
    "white": (255., 255., 255.),
    "red": (255., 0., 0.),
    "green": (0., 255., 0.),
    "blue": (0., 0., 255.),
    "yellow": (255., 255., 0.),
    "cyan": (0., 255., 255.),
    "magenta": (255., 0., 255.),
}

def blend_rgba_to_color(image_path, output_path, bg_color):
    try:
        img = Image.open(image_path).convert("RGBA")
        rgba = np.array(img).astype(np.float32) / 255.0  # (H, W, 4)
        rgb, alpha = rgba[..., :3], rgba[..., 3:]
        bg_rgb = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3) / 255.0
        blended = alpha * rgb + (1 - alpha) * bg_rgb
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(blended)
        out_img.save(output_path)
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")

def process_recursive(input_root, output_root, bg_name, num_workers=8):
    bg_color = COLOR_MAP[bg_name.lower()]
    tasks = []

    for root, _, files in os.walk(input_root):
        rel_path = os.path.relpath(root, input_root)
        target_dir = os.path.join(output_root, bg_name, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for f in files:
            if f.lower().endswith(".png"):
                in_path = os.path.join(root, f)
                out_path = os.path.join(target_dir, f.replace(".png", ".jpg"))
                tasks.append((in_path, out_path, bg_color))

    print(f"🚀 Starting blend with {min(num_workers, len(tasks))} processes...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(blend_rgba_to_color, *args) for args in tasks]
        for _ in as_completed(futures):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--color', type=str, default='white')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(), help='Number of processes to use')
    args = parser.parse_args()

    process_recursive(args.input_dir, args.output_dir, args.color, args.num_workers)