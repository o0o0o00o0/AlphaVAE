import os
import json
import torch
import warnings
import argparse
import numpy as np
import pyiqa
from typing import Dict

warnings.filterwarnings('ignore')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Setup argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_root', type=str, default='')
parser.add_argument('--output_root', type=str, default='')
parser.add_argument('--output_json', type=str, default='')
args = parser.parse_args()

# Initialize metrics
aes_metric = pyiqa.create_metric('laion_aes', device=device)
fid_metric = pyiqa.create_metric('fid', device=device)
lpips_metric = pyiqa.create_metric('lpips', device=device)
psnr_metric = pyiqa.create_metric('psnr', device=device)
ssim_metric = pyiqa.create_metric('ssim', device=device)

def get_image_paths(folder: str):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

def compute_scores(input_paths, output_paths, input_dir, output_dir) -> Dict:
    in_aes = np.mean([aes_metric(p).item() for p in input_paths])
    out_aes = np.mean([aes_metric(p).item() for p in output_paths])
    fid = max(fid_metric(input_dir, output_dir).item(), 0.0)
    lpips = np.mean([lpips_metric(i, o).item() for i, o in zip(input_paths, output_paths)])
    psnr = np.mean([psnr_metric(i, o).item() for i, o in zip(input_paths, output_paths)])
    ssim = np.mean([ssim_metric(i, o).item() for i, o in zip(input_paths, output_paths)])
    return {
        "input_laion_score": round(in_aes, 4),
        "output_laion_score": round(out_aes, 4),
        "fid_score": round(fid, 4),
        "average_lpips": round(lpips, 4),
        "average_psnr": round(psnr, 4),
        "average_ssim": round(ssim, 4),
    }

def evaluate(input_root: str, output_root: str) -> Dict:
    final_result = {}

    for background in os.listdir(input_root):
        bg_input_path = os.path.join(input_root, background)
        bg_output_path = os.path.join(output_root, background)
        print(f"bg_input_path:{bg_input_path}")
        if not os.path.isdir(bg_input_path):
            continue

        bg_result = {}
        subset_result_map = {}
        subset_avg_scores = []

        valid_subsets = [
            s for s in os.listdir(bg_input_path)
            if os.path.isdir(os.path.join(bg_input_path, s))
        ]
        num_subsets = len(valid_subsets)
        if num_subsets == 0:
            continue
        subset_weight = 1.0 / num_subsets

        for subset in valid_subsets:
            subset_input_path = os.path.join(bg_input_path, subset)
            subset_output_path = os.path.join(bg_output_path, subset)

            subset_result = {}
            subtype_result_map = {}
            subtype_avg_scores = []

            valid_subtypes = [
                t for t in os.listdir(subset_input_path)
                if os.path.isdir(os.path.join(subset_input_path, t))
            ]
            num_subtypes = len(valid_subtypes)
            if num_subtypes == 0:
                continue
            subtype_weight_in_subset = subset_weight / num_subtypes

            for subtype in valid_subtypes:
                subtype_input_dir = os.path.join(subset_input_path, subtype)
                subtype_output_dir = os.path.join(subset_output_path, subtype)

                input_paths = get_image_paths(subtype_input_dir)
                output_paths = get_image_paths(subtype_output_dir)

                if len(input_paths) == 0 or len(input_paths) != len(output_paths):
                    continue

                score = compute_scores(input_paths, output_paths, subtype_input_dir, subtype_output_dir)
                subtype_result_map[subtype] = {
                    "number": len(input_paths),
                    "weight": round(subtype_weight_in_subset, 6),
                    "scores": score
                }
                subtype_avg_scores.append((score, subtype_weight_in_subset))

            if subtype_avg_scores:
                keys = subtype_avg_scores[0][0].keys()
                averaged = {
                    k: round(sum(s[k] for s, w in subtype_avg_scores)/len(subtype_avg_scores), 4)
                    for k in keys
                }
                subset_result["overall"] = {
                    "number": sum([v["number"] for v in subtype_result_map.values()]),
                    "weight": round(subset_weight, 6),
                    "scores": averaged
                }
                subset_result["subtypes"] = subtype_result_map
                subset_result_map[subset] = subset_result
                subset_avg_scores.append((averaged, subset_weight))

        if subset_avg_scores:
            keys = subset_avg_scores[0][0].keys()
            averaged = {
                k: round(sum(s[k] for s, w in subset_avg_scores)/len(subset_avg_scores), 4)
                for k in keys
            }
            bg_result["overall"] = {
                "number": sum([subset_result_map[s]["overall"]["number"] for s in subset_result_map]),
                "weight": 1.0, 
                "scores": averaged
            }
            bg_result["subsets"] = subset_result_map
            final_result[background] = bg_result

    return final_result


results = evaluate(args.input_root, args.output_root)

results_with_metadata = {
    "metadata": {
        "input_root": os.path.abspath(args.input_root),
        "output_root": os.path.abspath(args.output_root),
        "output_json": os.path.abspath(args.output_json),
        "device": str(device),
    },
    "results": results
}

with open(args.output_json, 'w') as f:
    json.dump(results_with_metadata, f, indent=4)

print(f"✅ Finished testing, saved to {args.output_json}")