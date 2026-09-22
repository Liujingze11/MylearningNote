#!/usr/bin/env python3
"""
微调后推理测试脚本

用微调后的 checkpoint 对测试图片进行推理，验证：
  1. 模型能否正常加载
  2. 固定文本 prompt 能否输出多个实例 Mask
  3. 输出 Mask 质量（可视化检查）

用法:
    python inference_test.py \
        --checkpoint /path/to/checkpoint.pt \
        --image /path/to/test_image.jpg \
        --prompt "objects on a white table" \
        --output-dir /path/to/output
"""

import argparse
import os
import sys


def run_inference(checkpoint_path: str, image_path: str, prompt: str, output_dir: str):
    """加载微调模型并进行推理。"""
    import torch
    from PIL import Image
    import numpy as np

    # 添加 sam3 到 path
    sam3_root = _find_sam3_root()
    if sam3_root:
        sys.path.insert(0, sam3_root)

    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. 加载模型 ----
    print("加载模型...")
    from sam3.model_builder import build_sam3_image_model

    # 方法：先建结构，再手动加载（避免 detector. 前缀问题）
    model = build_sam3_image_model(
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)
    result = model.load_state_dict(state_dict, strict=False)

    if result.missing_keys:
        print(f"  ⚠️  缺失 {len(result.missing_keys)} 个键（可能正常）")
    print(f"  ✅ 模型加载完成")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # ---- 2. 读取图片 ----
    print(f"读取图片: {image_path}")
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    print(f"  图片尺寸: {orig_w}x{orig_h}")

    # ---- 3. 推理 ----
    print(f"推理中（prompt: \"{prompt}\"）...")
    from sam3.data.transforms import ResizeLongestSide

    transform = ResizeLongestSide(1024)
    image_tensor = transform.apply_image(image)
    image_tensor = torch.as_tensor(image_tensor, dtype=torch.float32).permute(2, 0, 1)
    image_tensor = (image_tensor / 255.0 - torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)) / torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    # SAM3 推理接口（根据官方 API 调整）
    with torch.no_grad():
        inputs = {
            "images": [image_tensor.to(device)],
            "query_text": [prompt],
        }
        try:
            outputs = model(inputs)
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            print()
            print("可能原因：")
            print("  1. SAM3 推理 API 接口有变动，请参考官方 examples/")
            print("  2. checkpoint 权重与模型结构不匹配")
            print("  3. 输入格式不正确")
            return False

    # ---- 4. 解析输出 ----
    print("解析输出...")
    if "masks" in outputs:
        masks = outputs["masks"]
        scores = outputs.get("scores", [None] * len(masks))
        boxes = outputs.get("boxes", [None] * len(masks))

        print(f"  检测到 {len(masks)} 个实例")
        for i, (mask, score) in enumerate(zip(masks, scores)):
            score_str = f"{score:.3f}" if score is not None else "N/A"
            print(f"    实例 {i+1}: score={score_str}, pixels={mask.sum().item():.0f}")
    else:
        print(f"  输出键: {list(outputs.keys())}")
        print("  ⚠️  未找到 'masks' 键，请检查输出格式")

    # ---- 5. 保存可视化 ----
    print(f"保存结果到: {output_dir}")
    _save_visualization(image, masks if "masks" in outputs else [], output_dir)

    print()
    print("✅ 推理测试完成")
    return True


def _save_visualization(image, masks, output_dir):
    """保存可视化结果。"""
    try:
        import numpy as np
        from PIL import Image as PILImage

        # 保存原图
        image.save(os.path.join(output_dir, "original.jpg"))

        if not masks:
            return

        # 生成彩色 mask 叠加图
        overlay = np.array(image).copy()
        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255], [0, 255, 255],
            [128, 0, 0], [0, 128, 0], [0, 0, 128],
        ]

        for i, mask in enumerate(masks):
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask

            # 调整 mask 尺寸到原图
            if mask_np.shape != (image.size[1], image.size[0]):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.resize(image.size, PILImage.NEAREST)
                mask_np = np.array(mask_pil) > 127

            color = colors[i % len(colors)]
            overlay[mask_np > 0] = (
                overlay[mask_np > 0] * 0.5 + np.array(color) * 0.5
            ).astype(np.uint8)

        PILImage.fromarray(overlay).save(os.path.join(output_dir, "masks_overlay.jpg"))
        print(f"  保存: masks_overlay.jpg")

    except Exception as e:
        print(f"  ⚠️  可视化保存失败: {e}")


def _find_sam3_root():
    """尝试找到 SAM3 仓库根目录。"""
    candidates = [
        "/home/ljz/Jingze/MylearningNote/05_Computer_Vision/SAM/sam3",
        os.path.expanduser("~/sam3"),
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "sam3", "train")):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="SAM3 微调后推理测试")
    parser.add_argument("--checkpoint", required=True, help="微调后的 checkpoint 路径")
    parser.add_argument("--image", required=True, help="测试图片路径")
    parser.add_argument("--prompt", default="objects on a white table", help="文本提示词")
    parser.add_argument("--output-dir", default="./inference_output", help="输出目录")
    args = parser.parse_args()

    success = run_inference(args.checkpoint, args.image, args.prompt, args.output_dir)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
