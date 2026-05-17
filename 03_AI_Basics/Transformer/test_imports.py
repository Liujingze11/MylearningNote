"""
测试导入是否正确

这个脚本用于验证文件结构和导入路径是否正确配置
"""

import sys
import os

print("=" * 60)
print("测试 Transformer 项目导入")
print("=" * 60)

# 检查目录结构
print("\n1. 检查目录结构...")
required_dirs = ['core', 'utils', 'examples']
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"   ✓ {dir_name}/ 存在")
    else:
        print(f"   ✗ {dir_name}/ 不存在")

# 检查核心文件
print("\n2. 检查核心文件...")
core_files = [
    'core/attention.py',
    'core/feed_forward.py',
    'core/layer_norm.py',
    'core/encoder_layer.py',
    'core/decoder_layer.py',
    'core/encoder.py',
    'core/decoder.py',
    'core/transformer.py',
    'core/masks_utils.py'
]

for file_path in core_files:
    if os.path.exists(file_path):
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path} 不存在")

# 检查工具文件
print("\n3. 检查工具文件...")
util_files = [
    'utils/embeddings.py',
    'utils/positional_encoding.py'
]

for file_path in util_files:
    if os.path.exists(file_path):
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path} 不存在")

# 检查示例文件
print("\n4. 检查示例文件...")
if os.path.exists('examples/main.py'):
    print(f"   ✓ examples/main.py")
else:
    print(f"   ✗ examples/main.py 不存在")

# 测试导入（需要 torch）
print("\n5. 测试导入...")
try:
    import torch
    print("   ✓ torch 已安装")

    # 测试导入 core 包
    try:
        from core import (
            MultiHeadAttention,
            PositionWiseFeedForward,
            LayerNorm,
            EncoderLayer,
            DecoderLayer,
            Encoder,
            Decoder,
            Transformer,
            create_transformer
        )
        print("   ✓ core 包导入成功")
    except Exception as e:
        print(f"   ✗ core 包导入失败: {e}")

    # 测试导入 utils 包
    try:
        from utils import Embeddings, PositionalEncoder
        print("   ✓ utils 包导入成功")
    except Exception as e:
        print(f"   ✗ utils 包导入失败: {e}")

except ModuleNotFoundError:
    print("   ⚠ torch 未安装，跳过导入测试")
    print("   提示：运行 'pip install torch' 安装 PyTorch")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
