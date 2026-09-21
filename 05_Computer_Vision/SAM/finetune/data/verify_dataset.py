#!/usr/bin/env python3
"""
数据集验证脚本

在开始训练前，验证 COCO JSON 格式的数据集是否正确：
  1. JSON 结构完整性
  2. 图片文件是否全部存在
  3. 标注格式是否正确（polygon / bbox）
  4. 可视化抽样检查（可选）

用法:
    python verify_dataset.py --ann-file /path/to/_annotations.coco.json [--visualize 5]
"""

import argparse
import json
import os
from pathlib import Path


def verify_dataset(ann_file: str, visualize_count: int = 0, images_dir: str = None):
    """验证 COCO JSON 数据集的完整性和正确性。"""
    print(f"验证数据集: {ann_file}")
    print("=" * 60)

    errors = []
    warnings = []

    # ---- 1. JSON 结构 ----
    with open(ann_file, "r") as f:
        coco = json.load(f)

    for key in ("images", "annotations", "categories"):
        if key not in coco:
            errors.append(f"缺少顶层字段: '{key}'")

    if errors:
        _report(errors, warnings)
        return False

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    print(f"  图片数量:   {len(images)}")
    print(f"  标注数量:   {len(annotations)}")
    print(f"  类别数量:   {len(categories)}")

    # ---- 2. categories 检查 ----
    cat_ids = set()
    for cat in categories:
        if "id" not in cat or "name" not in cat:
            errors.append(f"category 缺少 'id' 或 'name': {cat}")
        cat_ids.add(cat["id"])
        print(f"  类别 [{cat['id']}]: \"{cat['name']}\"")

    # ---- 3. images 检查 ----
    img_dir = Path(images_dir) if images_dir else Path(ann_file).parent
    img_id_set = set()
    for img in images:
        img_id_set.add(img["id"])
        for key in ("id", "file_name", "width", "height"):
            if key not in img:
                errors.append(f"image {img.get('id', '?')} 缺少字段 '{key}'")
        img_path = img_dir / img["file_name"]
        if not img_path.exists():
            errors.append(f"图片文件不存在: {img['file_name']}")

    # ---- 4. annotations 检查 ----
    ann_img_ids = set()
    empty_seg = 0
    bad_bbox = 0
    for ann in annotations:
        for key in ("id", "image_id", "category_id", "segmentation", "bbox"):
            if key not in ann:
                errors.append(f"annotation {ann.get('id', '?')} 缺少字段 '{key}'")

        if ann.get("category_id") not in cat_ids:
            errors.append(
                f"annotation {ann['id']}: category_id={ann['category_id']} 不在 categories 中"
            )

        if ann.get("image_id") not in img_id_set:
            errors.append(
                f"annotation {ann['id']}: image_id={ann['image_id']} 不在 images 中"
            )

        ann_img_ids.add(ann.get("image_id"))

        seg = ann.get("segmentation")
        if seg is None or seg == [] or seg == [[]]:
            empty_seg += 1
        elif isinstance(seg, list):
            for poly in seg:
                if len(poly) < 6:  # 至少 3 个点
                    errors.append(
                        f"annotation {ann['id']}: polygon 点数不足 ({len(poly)//2} 个点)"
                    )

        bbox = ann.get("bbox")
        if bbox is not None:
            if len(bbox) != 4:
                errors.append(f"annotation {ann['id']}: bbox 应有 4 个值，实际 {len(bbox)}")
            elif bbox[2] <= 0 or bbox[3] <= 0:
                bad_bbox += 1

    # ---- 5. 统计汇总 ----
    if empty_seg > 0:
        warnings.append(f"{empty_seg} 个标注的 segmentation 为空")
    if bad_bbox > 0:
        warnings.append(f"{bad_bbox} 个标注的 bbox 宽/高 <= 0")

    imgs_without_ann = img_id_set - ann_img_ids
    if imgs_without_ann:
        warnings.append(f"{len(imgs_without_ann)} 张图片没有任何标注")

    # ---- 6. 输出 ----
    print()
    if errors:
        print("❌ 发现错误:")
        for e in errors:
            print(f"   • {e}")
    if warnings:
        print("⚠️  警告:")
        for w in warnings:
            print(f"   • {w}")
    if not errors and not warnings:
        print("✅ 数据集验证通过！")

    # ---- 7. 可视化 ----
    if visualize_count > 0 and not errors:
        _visualize_samples(coco, img_dir, visualize_count)

    _report(errors, warnings)
    return len(errors) == 0


def _visualize_samples(coco, img_dir, count):
    """可视化抽样图片及其标注。"""
    try:
        import numpy as np
    except ImportError:
        print("⚠️  跳过可视化（需要 numpy）")
        return

    # 按 image_id 分组 annotations
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    import random

    sample_imgs = random.sample(coco["images"], min(count, len(coco["images"])))

    for img_info in sample_imgs:
        img_path = img_dir / img_info["file_name"]
        anns = ann_by_img.get(img_info["id"], [])
        print(f"\n  图片: {img_info['file_name']}")
        print(f"  尺寸: {img_info['width']}x{img_info['height']}")
        print(f"  标注数: {len(anns)}")
        for ann in anns:
            bbox = ann["bbox"]
            seg = ann["segmentation"]
            n_points = len(seg[0]) // 2 if seg and isinstance(seg[0], list) else 0
            print(
                f"    ann[{ann['id']}]: "
                f"bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}], "
                f"polygon点数={n_points}"
            )


def _report(errors, warnings):
    print()
    print(f"错误: {len(errors)}, 警告: {len(warnings)}")
    if errors:
        print("请修复错误后再开始训练。")
    print()


def main():
    parser = argparse.ArgumentParser(description="COCO JSON 数据集验证")
    parser.add_argument("--ann-file", required=True, help="COCO JSON 文件路径")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="图片文件夹路径（默认: 与 COCO JSON 同目录）",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="可视化抽样图片数量（默认: 0，不可视化）",
    )
    args = parser.parse_args()

    success = verify_dataset(args.ann_file, args.visualize, args.images_dir)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
