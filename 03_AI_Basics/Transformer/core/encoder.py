import torch
import torch.nn as nn
from .encoder_layer import EncoderLayer
from utils.embeddings import Embeddings
from utils.positional_encoding import PositionalEncoder


class Encoder(nn.Module):
    """
    Transformer 编码器

    结构：
        1. 词嵌入（Token Embedding）+ 位置编码（Positional Encoding）
        2. N 层编码器层（Encoder Layers）
        3. 最后的层归一化（可选）

    参数：
        vocab_size: 词表大小
        d_model: 模型维度
        num_layers: 编码器层数（如 6）
        num_heads: 注意力头数（如 8）
        d_ff: 前馈网络中间层维度（如 2048）
        max_seq_len: 最大序列长度
        dropout: Dropout 概率
    """

    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len=512, dropout=0.1):
        super(Encoder, self).__init__()

        # 词嵌入层
        self.embedding = Embeddings(vocab_size, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoder(d_model, max_seq_len)

        # N 层编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播

        参数：
            x: 输入 token ids (batch_size, seq_len)
            mask: 掩码张量（用于遮蔽 padding）

        返回：
            编码器输出 (batch_size, seq_len, d_model)
        """
        # 1. 词嵌入
        # 注意：Embeddings 类内部已经乘以了 sqrt(d_model)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)

        # 2. 位置编码
        # 注意：PositionalEncoder 内部也会乘以 sqrt(d_model)
        # 如果已经在 Embeddings 中乘过了，这里就不需要再乘了
        # 需要修改 PositionalEncoder 或 Embeddings 以避免重复缩放
        x = self.positional_encoding(x)

        # Dropout
        x = self.dropout(x)

        # 3. 通过 N 层编码器层
        for layer in self.layers:
            x = layer(x, mask)

        return x


class EncoderWithoutDuplicateScaling(nn.Module):
    """
    修正版编码器：避免重复缩放

    原问题：
        - Embeddings 内部已经乘以了 sqrt(d_model)
        - PositionalEncoder 内部又乘以了 sqrt(d_model)
        - 导致输入被缩放了两次

    解决方案：
        - 使用原始的 nn.Embedding（不缩放）
        - 在 forward 中统一缩放一次
    """

    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len=512, dropout=0.1):
        super(EncoderWithoutDuplicateScaling, self).__init__()

        self.d_model = d_model

        # 使用原始的 nn.Embedding（不缩放）
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 创建位置编码矩阵（不缩放）
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)

        # N 层编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_seq_len, d_model):
        """
        创建位置编码矩阵（不缩放）
        """
        import math
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x, mask=None):
        """
        前向传播（统一缩放版本）

        参数：
            x: 输入 token ids (batch_size, seq_len)
            mask: 掩码张量

        返回：
            编码器输出 (batch_size, seq_len, d_model)
        """
        import math

        # 1. 词嵌入
        x = self.embedding(x)  # (batch_size, seq_len, d_model)

        # 2. 统一缩放（只缩放一次）
        x = x * math.sqrt(self.d_model)

        # 3. 加上位置编码
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]

        # Dropout
        x = self.dropout(x)

        # 4. 通过 N 层编码器层
        for layer in self.layers:
            x = layer(x, mask)

        return x
