import json
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")
args = parser.parse_args()

rgb_colors = {"black", "white", "red", "green", "blue", "yellow", "cyan", "magenta", "gray"}
rgba_colors = {"rgba"}

group_scores = {
    "rgb": {},
    "rgba": {}
}

with open(args.input, "r") as f:
    data = json.load(f)

# 收集各 group 的每个 subset 分数
for color, color_data in data["results"].items():
    if color in rgb_colors:
        group = "rgb"
    elif color in rgba_colors:
        group = "rgba"
    else:
        continue

    for subset_name, subset_data in color_data.get("subsets", {}).items():
        scores = subset_data["overall"]["scores"]
        if subset_name not in group_scores[group]:
            group_scores[group][subset_name] = {
                "input_laion_score": [],
                "output_laion_score": [],
                "fid_score": [],
                "lpips": [],
                "psnr": [],
                "ssim": []
            }
        group_scores[group][subset_name]["input_laion_score"].append(scores["input_laion_score"])
        group_scores[group][subset_name]["output_laion_score"].append(scores["output_laion_score"])
        group_scores[group][subset_name]["fid_score"].append(scores["fid_score"])
        group_scores[group][subset_name]["lpips"].append(scores["average_lpips"])
        group_scores[group][subset_name]["psnr"].append(scores["average_psnr"])
        group_scores[group][subset_name]["ssim"].append(scores["average_ssim"])

# 写入 CSV
with open(args.output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "group", "subset", "input_laion_score", "output_laion_score", "fid_score", "lpips", "psnr", "ssim"
    ])
    writer.writeheader()

    for group, subsets in group_scores.items():
        all_rows = []

        for subset_name, metrics in subsets.items():
            row = {
                "group": group,
                "subset": subset_name,
                "input_laion_score": np.mean(metrics["input_laion_score"]),
                "output_laion_score": np.mean(metrics["output_laion_score"]),
                "fid_score": np.mean(metrics["fid_score"]),
                "lpips": np.mean(metrics["lpips"]),
                "psnr": np.mean(metrics["psnr"]),
                "ssim": np.mean(metrics["ssim"]),
            }
            writer.writerow(row)
            all_rows.append(row)

        # 计算该 group 所有 subset 的整体平均
        if all_rows:
            overall_avg = {
                "group": group,
                "subset": "overall_avg",
                "input_laion_score": np.mean([r["input_laion_score"] for r in all_rows]),
                "output_laion_score": np.mean([r["output_laion_score"] for r in all_rows]),
                "fid_score": np.mean([r["fid_score"] for r in all_rows]),
                "lpips": np.mean([r["lpips"] for r in all_rows]),
                "psnr": np.mean([r["psnr"] for r in all_rows]),
                "ssim": np.mean([r["ssim"] for r in all_rows]),
            }
            writer.writerow(overall_avg)