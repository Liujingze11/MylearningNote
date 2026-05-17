# 项目结构说明

## 📂 完整目录结构

```
手撕Transformer/
│
├── 📄 README.md                      # 项目主文档
├── 📄 PROJECT_STRUCTURE.md           # 项目结构说明（本文件）
├── 📄 test_imports.py                # 导入测试脚本
│
├── 📁 core/                          # 核心组件包
│   ├── 📄 __init__.py               # 包初始化，导出所有组件
│   ├── 📄 attention.py              # 多头注意力机制
│   ├── 📄 feed_forward.py           # 前馈神经网络
│   ├── 📄 layer_norm.py             # 层归一化
│   ├── 📄 encoder_layer.py          # 编码器层
│   ├── 📄 decoder_layer.py          # 解码器层
│   ├── 📄 encoder.py                # 完整编码器
│   ├── 📄 decoder.py                # 完整解码器
│   ├── 📄 transformer.py            # 完整 Transformer 模型
│   └── 📄 masks_utils.py            # 掩码和工具函数
│
├── 📁 utils/                         # 工具包
│   ├── 📄 __init__.py               # 包初始化
│   ├── 📄 embeddings.py             # 词嵌入层
│   └── 📄 positional_encoding.py    # 位置编码
│
├── 📁 examples/                      # 示例代码
│   ├── 📄 __init__.py               # 包初始化
│   └── 📄 main.py                   # 完整示例（训练、推理）
│
└── 📁 [原有文件]                     # 原始学习文件
    ├── 📄 embedding_learn.py        # Embedding 学习示例
    ├── 📄 mps_drive.py              # 设备选择工具
    ├── 📄 input_part.py             # 词嵌入（原始版本）
    └── 📄 position.py               # 位置编码（原始版本）
```

## 🔗 模块依赖关系

### 依赖图

```
utils/
├── embeddings.py           (无依赖)
└── positional_encoding.py  (无依赖)

core/
├── attention.py            (无依赖)
├── feed_forward.py         (无依赖)
├── layer_norm.py           (无依赖)
├── masks_utils.py          (无依赖)
│
├── encoder_layer.py        依赖 ← attention.py
│                           依赖 ← feed_forward.py
│                           依赖 ← layer_norm.py
│
├── decoder_layer.py        依赖 ← attention.py
│                           依赖 ← feed_forward.py
│                           依赖 ← layer_norm.py
│
├── encoder.py              依赖 ← encoder_layer.py
│                           依赖 ← utils/embeddings.py
│                           依赖 ← utils/positional_encoding.py
│
├── decoder.py              依赖 ← decoder_layer.py
│
└── transformer.py          依赖 ← encoder.py
                            依赖 ← decoder.py

examples/
└── main.py                 依赖 ← core 包的所有组件
```

## 📥 导入方式

### 1. 从 core 包导入

```python
# 导入完整模型
from core import Transformer, create_transformer

# 导入各个组件
from core import (
    MultiHeadAttention,
    PositionWiseFeedForward,
    PositionWiseFeedForwardGELU,
    LayerNorm,
    EncoderLayer,
    EncoderLayerPreLN,
    DecoderLayer,
    DecoderLayerPreLN,
    Encoder,
    EncoderWithoutDuplicateScaling,
    Decoder
)

# 导入工具函数
from core import (
    create_padding_mask,
    create_causal_mask,
    create_decoder_mask,
    count_parameters,
    print_model_info,
    LabelSmoothing
)
```

### 2. 从 utils 包导入

```python
from utils import Embeddings, PositionalEncoder
```

### 3. 直接导入特定模块

```python
# 导入注意力机制
from core.attention import MultiHeadAttention

# 导入前馈网络
from core.feed_forward import PositionWiseFeedForward, PositionWiseFeedForwardGELU

# 导入层归一化
from core.layer_norm import LayerNorm

# 导入编码器层
from core.encoder_layer import EncoderLayer, EncoderLayerPreLN

# 导入解码器层
from core.decoder_layer import DecoderLayer, DecoderLayerPreLN

# 导入完整编码器
from core.encoder import Encoder, EncoderWithoutDuplicateScaling

# 导入完整解码器
from core.decoder import Decoder

# 导入 Transformer
from core.transformer import Transformer, create_transformer

# 导入掩码工具
from core.masks_utils import (
    create_padding_mask,
    create_causal_mask,
    create_decoder_mask
)

# 导入嵌入层
from utils.embeddings import Embeddings

# 导入位置编码
from utils.positional_encoding import PositionalEncoder
```

## 🎯 各文件功能说明

### Core 包（核心组件）

| 文件 | 主要类/函数 | 功能 | 行数 |
|-----|----------|------|------|
| `attention.py` | `MultiHeadAttention` | 实现多头注意力机制 | ~140 |
| `feed_forward.py` | `PositionWiseFeedForward` | 位置前馈网络 | ~80 |
| `layer_norm.py` | `LayerNorm` | 层归一化 | ~50 |
| `encoder_layer.py` | `EncoderLayer` | 编码器层 | ~120 |
| `decoder_layer.py` | `DecoderLayer` | 解码器层 | ~150 |
| `encoder.py` | `Encoder` | 完整编码器 | ~120 |
| `decoder.py` | `Decoder` | 完整解码器 | ~80 |
| `transformer.py` | `Transformer` | 完整模型 | ~130 |
| `masks_utils.py` | 多个工具函数 | 掩码生成和其他工具 | ~180 |

### Utils 包（工具模块）

| 文件 | 主要类/函数 | 功能 | 行数 |
|-----|----------|------|------|
| `embeddings.py` | `Embeddings` | 词嵌入层 | ~45 |
| `positional_encoding.py` | `PositionalEncoder` | 位置编码 | ~65 |

### Examples 包（示例代码）

| 文件 | 主要函数 | 功能 | 行数 |
|-----|---------|------|------|
| `main.py` | 3个示例函数 | 演示训练和推理 | ~200 |

## 🔄 与原有文件的关系

| 原有文件 | 新文件 | 关系 |
|---------|--------|------|
| `input_part.py` | `utils/embeddings.py` | 改进版本，添加更多注释 |
| `position.py` | `utils/positional_encoding.py` | 改进版本，修复bug |
| `embedding_learn.py` | - | 保留作为学习示例 |
| `mps_drive.py` | - | 保留作为工具 |

## ✅ 组织优势

1. **清晰的模块划分**
   - `core/`: 核心 Transformer 组件
   - `utils/`: 辅助工具
   - `examples/`: 示例代码

2. **避免命名冲突**
   - 使用包结构避免全局命名空间污染

3. **便于维护**
   - 相关功能集中在一起
   - 依赖关系清晰

4. **易于扩展**
   - 可以轻松添加新的组件
   - 可以独立测试每个模块

5. **学习友好**
   - 原有文件保留，可对比学习
   - 结构化的代码更容易理解

## 🧪 测试代码

运行以下命令测试项目结构：

```bash
cd ~/Desktop/手撕Transformer
python test_imports.py
```

## 📝 注意事项

1. **导入路径**：所有导入都使用相对导入（`.`），确保包结构正确
2. **Python 路径**：examples/main.py 会自动添加父目录到 sys.path
3. **原有文件**：保留在项目根目录，不影响新结构
4. **PyTorch 依赖**：需要安装 torch 才能运行示例代码

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install torch

# 2. 测试导入
python test_imports.py

# 3. 运行示例
python examples/main.py
```

---

**更新日期**: 2026-02-09
