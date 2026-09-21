#!/usr/bin/env python3
"""
YOLO-seg TXT → COCO JSON 转换脚本

将 YOLO segmentation 格式的标注转换为 SAM3 微调所需的 COCO JSON 格式。

输入格式（YOLO-seg）:
    每行: class_id x1 y1 x2 y2 x3 y3 ...
    坐标为归一化值 (0~1)

输出格式（COCO JSON）:
    标准 COCO 格式，包含 images / annotations / categories

用法:
    python yolo_to_coco.py \
        --images-dir /path/to/images \
        --labels-dir /path/to/labels \
        --output /path/to/_annotations.coco.json \
        --prompt "objects on a white table"
"""

import argparse
import json
import os
from pathlib import Path


def yolo_seg_to_coco(images_dir: str, labels_dir: str, prompt: str, output_path: str):
    """
    将 YOLO-seg 标注转换为 COCO JSON。

    Args:
        images_dir: 图片文件夹路径
        labels_dir: 标签文件夹路径（与图片同名，.txt 后缀）
        prompt: 统一的文本提示词（作为 category name）
        output_path: 输出 COCO JSON 文件路径
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    # 支持的图片格式
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # 收集所有图片
    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    )

    if len(image_files) == 0:
        raise FileNotFoundError(f"在 {images_dir} 中未找到任何图片文件")

    # COCO 结构
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": prompt, "supercategory": "objects"}],
    }

    ann_id = 1
    stats = {"images": 0, "annotations": 0, "skipped_labels": 0}

    for img_id, img_path in enumerate(image_files, start=1):
        # 读取图片尺寸（通过读取文件头，避免依赖 PIL）
        width, height = _get_image_size(img_path)

        coco["images"].append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )

        # 对应的标签文件
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            stats["skipped_labels"] += 1
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:  # 至少 class_id + 3个点(x,y)
                continue

            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            # 转换归一化坐标为像素坐标（polygon 格式）
            # COCO segmentation: [x1, y1, x2, y2, ...]（像素值）
            polygon = []
            for i in range(0, len(coords), 2):
                x_px = coords[i] * width
                y_px = coords[i + 1] * height
                polygon.extend([round(x_px, 2), round(y_px, 2)])

            # 计算 bbox（从 polygon 推导）
            xs = polygon[0::2]
            ys = polygon[1::2]
            x_min, y_min = min(xs), min(ys)
            bbox_w = max(xs) - x_min
            bbox_h = max(ys) - y_min

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,  # 统一为单一类别
                    "segmentation": [polygon],
                    "bbox": [
                        round(x_min, 2),
                        round(y_min, 2),
                        round(bbox_w, 2),
                        round(bbox_h, 2),
                    ],
                    "area": round(bbox_w * bbox_h, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            stats["annotations"] += 1

        stats["images"] += 1

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    # 打印统计
    print(f"转换完成！")
    print(f"  图片数量:   {stats['images']}")
    print(f"  标注数量:   {stats['annotations']}")
    print(f"  缺失标签:   {stats['skipped_labels']}")
    print(f"  输出路径:   {output_path}")


def _get_image_size(img_path: Path):
    """
    获取图片尺寸，优先用 PIL，回退用文件头解析。
    """
    try:
        from PIL import Image

        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except ImportError:
        pass

    # 回退：读取文件头
    with open(img_path, "rb") as f:
        header = f.read(32)

    # PNG
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        w = int.from_bytes(header[16:20], "big")
        h = int.from_bytes(header[20:24], "big")
        return w, h

    # JPEG (需要扫描 SOF marker)
    if header[:2] == b"\xff\xd8":
        with open(img_path, "rb") as f:
            f.read(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    break
                if marker[0] != 0xFF:
                    break
                if marker[1] in (0xC0, 0xC1, 0xC2):
                    f.read(3)
                    h = int.from_bytes(f.read(2), "big")
                    w = int.from_bytes(f.read(2), "big")
                    return w, h
                length = int.from_bytes(f.read(2), "big")
                f.read(length - 2)

    raise ValueError(f"无法读取图片尺寸: {img_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO-seg → COCO JSON 转换")
    parser.add_argument("--images-dir", required=True, help="图片文件夹路径")
    parser.add_argument("--labels-dir", required=True, help="YOLO-seg 标签文件夹路径")
    parser.add_argument("--output", required=True, help="输出 COCO JSON 路径")
    parser.add_argument(
        "--prompt",
        default="objects on a white table",
        help="统一的文本提示词（默认: 'objects on a white table'）",
    )
    args = parser.parse_args()

    yolo_seg_to_coco(args.images_dir, args.labels_dir, args.prompt, args.output)


if __name__ == "__main__":
    main()
