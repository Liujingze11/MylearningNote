import torch
import torch.nn as nn
import math
from .decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """
    Transformer 解码器

    结构：
        1. 词嵌入（Token Embedding）+ 位置编码（Positional Encoding）
        2. N 层解码器层（Decoder Layers）
        3. 最后的层归一化（可选）

    参数：
        vocab_size: 词表大小
        d_model: 模型维度
        num_layers: 解码器层数（如 6）
        num_heads: 注意力头数（如 8）
        d_ff: 前馈网络中间层维度（如 2048）
        max_seq_len: 最大序列长度
        dropout: Dropout 概率
    """

    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len=512, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 创建位置编码矩阵
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)

        # N 层解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_seq_len, d_model):
        """
        创建位置编码矩阵
        """
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播

        参数：
            x: 目标序列 token ids (batch_size, tgt_seq_len)
            encoder_output: 编码器输出 (batch_size, src_seq_len, d_model)
            src_mask: 源序列掩码（用于编码器-解码器注意力）
            tgt_mask: 目标序列掩码（包含因果掩码）

        返回：
            解码器输出 (batch_size, tgt_seq_len, d_model)
        """
        # 1. 词嵌入
        x = self.embedding(x)  # (batch_size, tgt_seq_len, d_model)

        # 2. 缩放
        x = x * math.sqrt(self.d_model)

        # 3. 加上位置编码
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]

        # Dropout
        x = self.dropout(x)

        # 4. 通过 N 层解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x
