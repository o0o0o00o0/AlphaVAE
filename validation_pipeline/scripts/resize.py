import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        default='',
                        help="origin path of images")
    parser.add_argument("--output_dir", type=str,
                        default='',
                        help="target path to store processed images")
    parser.add_argument("--resolution", type=int,
                        default=1024, help="")
    parser.add_argument("--num_workers", type=int,
                        default=8, help="")
    return parser.parse_args()


def build_transform(resolution):
    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def collect_image_tasks(input_root, output_root):
    tasks = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_root)
                output_path = os.path.join(output_root, rel_path)
                tasks.append((input_path, output_path))
    return tasks


def process_image(task, transform):
    input_path, output_path = task
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image = Image.open(input_path).convert("RGBA")
        tensor_image = transform(image)
        tensor_image_save = (tensor_image * 0.5 + 0.5).clamp(0, 1)
        save_image(tensor_image_save, output_path)
        return f"✅ 处理完成: {os.path.basename(input_path)}"
    except Exception as e:
        return f"❌ 错误: {input_path} -> {e}"


def main():
    args = parse_args()
    transform = build_transform(args.resolution)
    tasks = collect_image_tasks(args.input_dir, args.output_dir)

    print(f"🔍 共找到 {len(tasks)} 张图片，使用 {args.num_workers} 个线程处理...")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_image, task, transform) for task in tasks]
        for future in as_completed(futures):
            print(future.result())

    print("🎉 所有图片处理完毕。")


if __name__ == "__main__":
    main()