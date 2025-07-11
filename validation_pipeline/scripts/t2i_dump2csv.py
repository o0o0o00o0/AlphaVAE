import json
import csv
import argparse

def avg_dict(dicts, keys):
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}

def process_group(summary, group_name, color_list):
    group_data = [summary[color] for color in color_list]

    # 明确字段
    overall_keys = ['weighted_avg_gt_aes', 'weighted_avg_gen_aes', 'weighted_avg_fid', 'weighted_avg_clipscore']
    mixed_keys = ['GT_laion_aes_score', 'gen_laion_aes_score', 'FID_score', 'clipscore']

    overall = avg_dict(group_data, overall_keys)
    mixed = avg_dict([g['avg_result_only_mixed'] for g in group_data], mixed_keys)
    non_mixed = avg_dict([g['avg_result_only_non_mixed'] for g in group_data], mixed_keys)

    return [
        [group_name, 'overall_avg', overall['weighted_avg_gt_aes'], overall['weighted_avg_gen_aes'], overall['weighted_avg_fid'], overall['weighted_avg_clipscore']],
        [group_name, 'mixed', mixed['GT_laion_aes_score'], mixed['gen_laion_aes_score'], mixed['FID_score'], mixed['clipscore']],
        [group_name, 'non_mixed', non_mixed['GT_laion_aes_score'], non_mixed['gen_laion_aes_score'], non_mixed['FID_score'], non_mixed['clipscore']],
    ]

def main():
    parser = argparse.ArgumentParser(description="Dump LAION+FID+CLIP summary from JSON to CSV")
    parser.add_argument('--json_file', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output_csv', type=str, default='laion_fid_clip_summary.csv', help='Output CSV file name')

    args = parser.parse_args()

    with open(args.json_file, "r") as f:
        data = json.load(f)

    summary = data["per_color_summary"]

    rgb_colors = [
        'gray', 'black', 'white', 'red', 'green',
        'blue', 'yellow', 'cyan', 'magenta'
    ]

    all_rows = []

    # RGB group (averaged)
    all_rows.extend(process_group(summary, 'rgb', rgb_colors))

    # RGBA group (direct read)
    rgba = summary['rgba']
    all_rows.append(['rgba', 'overall_avg', rgba['weighted_avg_gt_aes'], rgba['weighted_avg_gen_aes'], rgba['weighted_avg_fid'], rgba['weighted_avg_clipscore']])
    all_rows.append(['rgba', 'mixed', rgba['avg_result_only_mixed']['GT_laion_aes_score'], rgba['avg_result_only_mixed']['gen_laion_aes_score'], rgba['avg_result_only_mixed']['FID_score'], rgba['avg_result_only_mixed']['clipscore']])
    all_rows.append(['rgba', 'non_mixed', rgba['avg_result_only_non_mixed']['GT_laion_aes_score'], rgba['avg_result_only_non_mixed']['gen_laion_aes_score'], rgba['avg_result_only_non_mixed']['FID_score'], rgba['avg_result_only_non_mixed']['clipscore']])

    with open(args.output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'subset', 'input_laion', 'output_laion', 'fid_score', 'clipscore'])
        writer.writerows(all_rows)

    print(f"✅ CSV saved to {args.output_csv}")

if __name__ == "__main__":
    main()