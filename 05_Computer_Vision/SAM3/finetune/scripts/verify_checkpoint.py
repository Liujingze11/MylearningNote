#!/usr/bin/env python3
"""
Checkpoint 验证脚本

训练完成后，验证保存的 checkpoint 是否正确可加载。

检查内容：
  1. checkpoint 文件是否存在
  2. 文件结构是否正确（包含 'model' 键）
  3. 权重 key 与模型结构是否匹配
  4. 是否包含 segmentation head 权重
  5. 能否成功加载到模型中

用法:
    python verify_checkpoint.py --checkpoint /path/to/checkpoint.pt
"""

import argparse
import os
import sys


def verify_checkpoint(checkpoint_path: str):
    """验证 checkpoint 的完整性和可加载性。"""
    print(f"验证 checkpoint: {checkpoint_path}")
    print("=" * 60)

    errors = []
    warnings = []

    # ---- 1. 文件存在性 ----
    if not os.path.exists(checkpoint_path):
        errors.append(f"文件不存在: {checkpoint_path}")
        _report(errors, warnings)
        return False

    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"  文件大小: {file_size_mb:.1f} MB")

    if file_size_mb < 10:
        warnings.append(f"文件过小 ({file_size_mb:.1f} MB)，可能不是完整 checkpoint")

    # ---- 2. 加载 checkpoint ----
    import torch

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        errors.append(f"无法加载 checkpoint: {e}")
        _report(errors, warnings)
        return False

    # ---- 3. 检查结构 ----
    if isinstance(ckpt, dict):
        print(f"  顶层键: {list(ckpt.keys())}")
        if "model" in ckpt:
            state_dict = ckpt["model"]
            print(f"  model 键数量: {len(state_dict)}")
        else:
            state_dict = ckpt
            warnings.append("checkpoint 顶层没有 'model' 键，将尝试直接使用 state_dict")
    else:
        errors.append(f"checkpoint 格式异常，期望 dict，实际 {type(ckpt)}")
        _report(errors, warnings)
        return False

    # ---- 4. 检查 segmentation head 权重 ----
    seg_keys = [k for k in state_dict if "segmentation" in k.lower() or "mask" in k.lower()]
    if seg_keys:
        print(f"  segmentation 相关键: {len(seg_keys)} 个")
        for k in seg_keys[:5]:
            print(f"    • {k}  shape={tuple(state_dict[k].shape)}")
        if len(seg_keys) > 5:
            print(f"    ... 共 {len(seg_keys)} 个")
    else:
        warnings.append("未找到 segmentation 相关键——训练时 enable_segmentation 是否为 True？")

    # ---- 5. 检查关键模块权重 ----
    key_modules = {
        "backbone": [k for k in state_dict if k.startswith("backbone.")],
        "transformer": [k for k in state_dict if k.startswith("transformer.")],
        "input_geometry_encoder": [k for k in state_dict if k.startswith("input_geometry_encoder.")],
    }
    for name, keys in key_modules.items():
        if keys:
            print(f"  {name}: {len(keys)} 个参数")
        else:
            warnings.append(f"未找到 {name} 相关键")

    # ---- 6. 尝试加载到模型 ----
    print()
    print("尝试加载到 SAM3 模型...")
    try:
        # 添加 sam3 到 path
        sam3_root = _find_sam3_root()
        if sam3_root:
            sys.path.insert(0, sam3_root)

        from sam3.model_builder import build_sam3_image_model

        # 不加载预训练权重，只构建模型结构
        model = build_sam3_image_model(
            checkpoint_path=None,
            load_from_HF=False,
            enable_segmentation=True,
        )

        # 加载 checkpoint
        result = model.load_state_dict(state_dict, strict=False)

        if result.missing_keys:
            warnings.append(f"加载时缺失 {len(result.missing_keys)} 个键")
            for k in result.missing_keys[:5]:
                warnings.append(f"  缺失: {k}")
        if result.unexpected_keys:
            warnings.append(f"加载时多余 {len(result.unexpected_keys)} 个键")
            for k in result.unexpected_keys[:5]:
                warnings.append(f"  多余: {k}")

        if not result.missing_keys and not result.unexpected_keys:
            print("  ✅ 完美加载，所有键匹配！")
        else:
            print(f"  ⚠️  加载完成（缺失={len(result.missing_keys)}, 多余={len(result.unexpected_keys)}）")

        # 统计总参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  模型总参数: {total_params / 1e6:.1f}M")

    except ImportError as e:
        warnings.append(f"无法导入 SAM3 模型（可能未安装）: {e}")
    except Exception as e:
        errors.append(f"加载到模型失败: {e}")

    _report(errors, warnings)
    return len(errors) == 0


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


def _report(errors, warnings):
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
        print("✅ checkpoint 验证通过！")
    print()
    print(f"错误: {len(errors)}, 警告: {len(warnings)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="SAM3 checkpoint 验证")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 文件路径（.pt）")
    args = parser.parse_args()

    success = verify_checkpoint(args.checkpoint)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
