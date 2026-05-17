import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    """
    位置编码（Positional Encoding）

    作用：
        由于 Transformer 没有循环结构（如 RNN），模型无法感知序列中 token 的位置信息
        位置编码通过添加位置相关的向量来注入位置信息

    实现方式：
        使用正弦和余弦函数生成位置编码
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    参数：
        d_model: 模型维度（必须与词嵌入维度一致）
        max_seq_len: 最大序列长度（默认 80）
    """

    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # 创建位置编码矩阵 (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)

        # 为每个位置计算位置编码
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                # 偶数维度使用 sin
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                # 奇数维度使用 cos
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        # 增加 batch 维度：(1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        # 将位置编码注册为 buffer（不参与梯度更新）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播

        参数：
            x: 词嵌入张量 (batch_size, seq_len, d_model)

        返回：
            加上位置编码的张量 (batch_size, seq_len, d_model)
        """
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)

        # 增加位置编码到单词嵌入表示中
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]

        return x
