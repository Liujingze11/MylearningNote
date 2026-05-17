# 手撕 Transformer 教程

这是一个完整的 Transformer 实现教程，从零开始构建 Transformer 模型的每个组件。

## 📁 项目结构

```
手撕Transformer/
├── README.md                    # 项目说明文档（本文件）
├── core/                        # 核心组件目录
│   ├── __init__.py             # 包初始化文件
│   ├── attention.py            # 多头注意力机制
│   ├── feed_forward.py         # 前馈神经网络
│   ├── layer_norm.py           # 层归一化
│   ├── encoder_layer.py        # 编码器层
│   ├── decoder_layer.py        # 解码器层
│   ├── encoder.py              # 完整编码器
│   ├── decoder.py              # 完整解码器
│   ├── transformer.py          # 完整 Transformer 模型
│   └── masks_utils.py          # 工具函数（掩码生成等）
├── utils/                       # 工具目录
│   ├── __init__.py             # 包初始化文件
│   ├── embeddings.py           # 词嵌入
│   └── positional_encoding.py  # 位置编码
├── examples/                    # 示例代码目录
│   ├── __init__.py             # 包初始化文件
│   └── main.py                 # 完整示例（训练、推理）
├── embedding_learn.py           # Embedding 学习示例（原有）
├── mps_drive.py                 # 设备选择工具（原有）
├── input_part.py                # 词嵌入（原有文件）
└── position.py                  # 位置编码（原有文件）
```

## 🧩 模块说明

### 📦 `core/` - 核心组件

#### 1. **基础组件**
- `attention.py`: **多头注意力机制**
  - `MultiHeadAttention`: 实现缩放点积注意力和多头机制
  - 支持掩码（mask）操作
  - 包含头分割与合并逻辑

- `feed_forward.py`: **位置前馈网络**
  - `PositionWiseFeedForward`: 使用 ReLU 激活的前馈网络
  - `PositionWiseFeedForwardGELU`: 使用 GELU 激活的前馈网络

- `layer_norm.py`: **层归一化**
  - `LayerNorm`: 实现层归一化，包含可学习的缩放和偏移参数

#### 2. **层组件**
- `encoder_layer.py`: **编码器层**
  - `EncoderLayer`: Post-LN 版本（先残差后归一化）
  - `EncoderLayerPreLN`: Pre-LN 版本（先归一化后残差）

- `decoder_layer.py`: **解码器层**
  - `DecoderLayer`: Post-LN 版本，包含掩码自注意力和交叉注意力
  - `DecoderLayerPreLN`: Pre-LN 版本

#### 3. **完整模型**
- `encoder.py`: **编码器**
  - `Encoder`: 标准编码器（N 层编码器层堆叠）
  - `EncoderWithoutDuplicateScaling`: 修正版编码器（避免重复缩放）

- `decoder.py`: **解码器**
  - `Decoder`: 完整解码器（N 层解码器层堆叠）

- `transformer.py`: **Transformer 模型**
  - `Transformer`: 完整的编码器-解码器架构
  - `create_transformer()`: 便捷的模型创建函数

#### 4. **工具函数**
- `masks_utils.py`: **掩码和辅助工具**
  - `create_padding_mask()`: 创建 padding 掩码
  - `create_causal_mask()`: 创建因果掩码
  - `create_decoder_mask()`: 创建解码器掩码（组合 padding 和因果掩码）
  - `count_parameters()`: 统计模型参数
  - `print_model_info()`: 打印模型信息
  - `LabelSmoothing`: 标签平滑损失函数

### 📦 `utils/` - 工具模块

- `embeddings.py`: **词嵌入**
  - `Embeddings`: 将 token id 映射到词向量

- `positional_encoding.py`: **位置编码**
  - `PositionalEncoder`: 使用正弦余弦函数生成位置编码

### 📦 `examples/` - 示例代码

- `main.py`: **完整示例**
  - `simple_example()`: 基本使用示例
  - `training_example()`: 训练流程示例
  - `inference_example()`: 推理流程（自回归生成）

## 🚀 快速开始

### 安装依赖

```bash
pip install torch
```

### 运行示例

```bash
cd ~/Desktop/手撕Transformer
python examples/main.py
```

或者从项目根目录运行：

```bash
cd ~/Desktop/手撕Transformer
python -m examples.main
```

### 基本使用

```python
import torch
from core import create_transformer, create_padding_mask, create_decoder_mask

# 创建模型
model = create_transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048
)

# 准备数据
src = torch.randint(1, 10000, (2, 10))  # (batch_size, src_seq_len)
tgt = torch.randint(1, 10000, (2, 8))   # (batch_size, tgt_seq_len)

# 创建掩码
src_mask = create_padding_mask(src, pad_idx=0)
tgt_mask = create_decoder_mask(tgt, pad_idx=0)

# 前向传播
output = model(src, tgt, src_mask, tgt_mask)
print(output.shape)  # (2, 8, 10000)
```

### 导入模块

```python
# 导入完整模型
from core import Transformer, create_transformer

# 导入各个组件
from core import (
    MultiHeadAttention,
    PositionWiseFeedForward,
    LayerNorm,
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder
)

# 导入工具函数
from core import (
    create_padding_mask,
    create_causal_mask,
    create_decoder_mask,
    print_model_info
)

# 导入嵌入和位置编码
from utils import Embeddings, PositionalEncoder
```

## 📚 Transformer 架构详解

### 核心思想

Transformer 是一个基于**自注意力机制**的序列到序列模型，完全抛弃了 RNN 和 CNN，通过并行化计算大幅提升训练速度。

### 关键组件

1. **自注意力（Self-Attention）**
   - 计算序列中每个位置与其他所有位置的相关性
   - 公式：Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V

2. **多头注意力（Multi-Head Attention）**
   - 将 Q、K、V 投影到多个子空间，并行计算注意力
   - 允许模型关注不同位置的不同表示子空间

3. **位置编码（Positional Encoding）**
   - 由于 Transformer 没有循环结构，需要手动注入位置信息
   - 使用正弦和余弦函数生成位置编码

4. **残差连接（Residual Connection）**
   - 缓解深度网络的梯度消失问题
   - 公式：output = LayerNorm(x + Sublayer(x))

5. **层归一化（Layer Normalization）**
   - 稳定训练过程，加速收敛

### 编码器-解码器结构

```
输入序列 → 编码器 → 编码表示
                     ↓
输出序列 ← 解码器 ← 编码表示
```

- **编码器**：理解输入序列的语义
- **解码器**：根据编码表示生成输出序列

## 🎯 应用场景

1. **机器翻译**：英语 → 中文
2. **文本摘要**：长文本 → 摘要
3. **问答系统**：问题 + 上下文 → 答案
4. **代码生成**：自然语言描述 → 代码
5. **对话系统**：上文 → 回复

## 📊 超参数说明

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `d_model` | 模型维度 | 512 |
| `num_layers` | 编码器/解码器层数 | 6 |
| `num_heads` | 注意力头数 | 8 |
| `d_ff` | 前馈网络中间层维度 | 2048 |
| `dropout` | Dropout 概率 | 0.1 |
| `max_seq_len` | 最大序列长度 | 512 |

## 🔍 关键概念

### 自注意力 vs 交叉注意力

- **自注意力**：Q、K、V 来自同一个序列
- **交叉注意力**：Q 来自解码器，K、V 来自编码器

### Post-LN vs Pre-LN

- **Post-LN**：x = LN(x + Sublayer(x))
- **Pre-LN**：x = x + Sublayer(LN(x))
- Pre-LN 训练更稳定，无需学习率预热

### 掩码类型

1. **Padding 掩码**：遮蔽填充的 token
2. **因果掩码**：遮蔽未来信息（自回归生成）

## 🛠️ 训练技巧

1. **学习率调度**：使用 warmup + decay
2. **标签平滑**：防止模型过度自信
3. **梯度裁剪**：防止梯度爆炸
4. **权重初始化**：Xavier 或 He 初始化
5. **数据增强**：随机删除、替换、打乱等

## 💡 学习建议

### 学习路径

1. **理解注意力机制**：先学习 `core/attention.py`
2. **理解辅助组件**：学习 `core/feed_forward.py` 和 `core/layer_norm.py`
3. **理解编码器**：学习 `core/encoder_layer.py` → `core/encoder.py`
4. **理解解码器**：学习 `core/decoder_layer.py` → `core/decoder.py`
5. **理解完整模型**：学习 `core/transformer.py`
6. **实践运行**：运行 `examples/main.py`

### 推荐顺序

```
1. utils/embeddings.py + utils/positional_encoding.py
   ↓
2. core/attention.py (核心！)
   ↓
3. core/feed_forward.py + core/layer_norm.py
   ↓
4. core/encoder_layer.py
   ↓
5. core/decoder_layer.py
   ↓
6. core/encoder.py + core/decoder.py
   ↓
7. core/transformer.py
   ↓
8. examples/main.py (实践)
```

## 📖 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## 🤝 贡献

欢迎提出问题和建议！

---

**作者**: 手撕 Transformer 教程
**最后更新**: 2026-02-09
