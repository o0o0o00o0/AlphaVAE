import os
import json
import torch
import warnings
import argparse
from tqdm import tqdm
import pyiqa

warnings.filterwarnings('ignore')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, required=True, help='Path to JSON config')
parser.add_argument('--blend_GT_dir', type=str, required=True, help='Path to blended GT root dir')
parser.add_argument('--blend_infer_dir', type=str, required=True, help='Path to blended inference root dir')
parser.add_argument('--output_file', type=str, required=True, help='Path to output result file')
args = parser.parse_args()

# Color list to iterate over
COLOR_LIST = [
    "gray",
    "black",
    "white",
    "red",
    "green",
    "blue",
    "yellow",
    "cyan",
    "magenta",
    "rgba"
]

# Create metrics
aes_metric = pyiqa.create_metric('laion_aes', device=device)
clip_metric = pyiqa.create_metric('clipscore', device=device)
fid_metric = pyiqa.create_metric('fid', device=device)

def get_image_paths(directory):
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

def compute_laion_aes(image_paths):
    total = 0
    for path in tqdm(image_paths, desc='laion_aes', leave=False):
        total += aes_metric(path).item()
    return total / len(image_paths) if image_paths else 0.0

def compute_clipscore(image_paths, prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines()]
    assert len(prompts) == len(image_paths), f"Prompt count mismatch: {prompt_file}"
    total = 0
    for path, prompt in tqdm(zip(image_paths, prompts), total=len(prompts), desc='clipscore', leave=False):
        total += clip_metric(path, caption_list=[prompt]).item()
    return total / len(image_paths) if image_paths else 0.0

# Load config
with open(args.config_file, 'r') as f:
    config = json.load(f)

all_results = {}
per_color_summary = {}

for color in COLOR_LIST:
    print(f"\n🎨 Processing color: {color}")
    color_results = []
    category_metrics = {}

    for item in config:
        category = os.path.basename(item['gen_file_path_dir'])
        print(f"\n🧪 Evaluating category: {category} under color: {color}")

        gt_dir = os.path.join(args.blend_GT_dir, color, category)
        gen_dir = os.path.join(args.blend_infer_dir, color, category)
        prompt_file = item['prompt_file']

        if not os.path.exists(gt_dir) or not os.path.exists(gen_dir):
            print(f"⚠️ Skipping category '{category}' for color '{color}' due to missing directory.")
            continue

        gt_images = get_image_paths(gt_dir)
        gen_images = get_image_paths(gen_dir)

        if len(gt_images) == 0 or len(gen_images) == 0:
            print(f"⚠️ Skipping category '{category}' for color '{color}' due to empty images.")
            continue

        gt_aes = compute_laion_aes(gt_images)
        gen_aes = compute_laion_aes(gen_images)
        avg_clip = compute_clipscore(gen_images, prompt_file)
        fid = max(fid_metric(gt_dir, gen_dir).item(), 0.0)

        metrics = {
            "category": category,
            "GT_laion_aes_score": round(gt_aes, 4),
            "gen_laion_aes_score": round(gen_aes, 4),
            "avg_clipscore": round(avg_clip, 4),
            "FID_score": round(fid, 4)
        }
        color_results.append(metrics)
        category_metrics[category] = metrics

    all_results[color] = color_results

    # Weighted average within this color
    categories = list(category_metrics.keys())
    if not categories:
        continue

    n = len([c for c in categories if c != "mixed"])
    weights = {}
    for cat in categories:
        if cat == "mixed":
            weights[cat] = 0.5
        else:
            weights[cat] = 0.5 / n if n > 0 else 0.0

    norm = sum(weights.values())
    for k in weights:
        weights[k] /= norm

    avg_clip = sum(category_metrics[c]["avg_clipscore"] * weights[c] for c in categories)
    avg_fid = sum(category_metrics[c]["FID_score"] * weights[c] for c in categories)
    avg_gen_aes = sum(category_metrics[c]["gen_laion_aes_score"] * weights[c] for c in categories)
    avg_gt_aes = sum(category_metrics[c]["GT_laion_aes_score"] * weights[c] for c in categories)

    per_color_summary[color] = {
        "weighted_avg_clipscore": round(avg_clip, 4),
        "weighted_avg_fid": round(avg_fid, 4),
        "weighted_avg_gen_aes": round(avg_gen_aes, 4),
        "weighted_avg_gt_aes": round(avg_gt_aes, 4)
    }
    
    # Additional separated averages
    if "mixed" in category_metrics:
        mixed_metrics = category_metrics["mixed"]
        per_color_summary[color]["avg_result_only_mixed"] = {
            "clipscore": round(mixed_metrics["avg_clipscore"], 4),
            "FID_score": round(mixed_metrics["FID_score"], 4),
            "gen_laion_aes_score": round(mixed_metrics["gen_laion_aes_score"], 4),
            "GT_laion_aes_score": round(mixed_metrics["GT_laion_aes_score"], 4)
        }

    if n > 0:
        avg_clip_non_mixed = sum(category_metrics[c]["avg_clipscore"] * (1.0 / n) for c in categories if c != "mixed")
        avg_fid_non_mixed = sum(category_metrics[c]["FID_score"] * (1.0 / n) for c in categories if c != "mixed")
        avg_gen_aes_non_mixed = sum(category_metrics[c]["gen_laion_aes_score"] * (1.0 / n) for c in categories if c != "mixed")
        avg_gt_aes_non_mixed = sum(category_metrics[c]["GT_laion_aes_score"] * (1.0 / n) for c in categories if c != "mixed")

        per_color_summary[color]["avg_result_only_non_mixed"] = {
            "clipscore": round(avg_clip_non_mixed, 4),
            "FID_score": round(avg_fid_non_mixed, 4),
            "gen_laion_aes_score": round(avg_gen_aes_non_mixed, 4),
            "GT_laion_aes_score": round(avg_gt_aes_non_mixed, 4)
        }

# Compose final output (no global summary)
final_output = {
    "metadata": {
        "config_file": args.config_file,
        "blend_GT_dir": args.blend_GT_dir,
        "blend_infer_dir": args.blend_infer_dir,
        "evaluated_colors": COLOR_LIST
    },
    "results": all_results,
    "per_color_summary": per_color_summary
}

# Save to file
with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done! Results saved to: {args.output_file}")