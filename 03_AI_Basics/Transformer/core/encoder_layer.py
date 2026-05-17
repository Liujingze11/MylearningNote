import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward
from .layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    """
    Transformer 编码器层

    结构：
        1. 多头自注意力（Multi-Head Self-Attention）
        2. 残差连接 + 层归一化（Add & Norm）
        3. 位置前馈网络（Position-wise Feed-Forward）
        4. 残差连接 + 层归一化（Add & Norm）

    参数：
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络中间层维度
        dropout: Dropout 概率
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # 多头自注意力层
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # 位置前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # 两个层归一化
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播

        参数：
            x: 输入张量 (batch_size, seq_len, d_model)
            mask: 掩码张量（用于遮蔽 padding）

        返回：
            输出张量 (batch_size, seq_len, d_model)
        """
        # 1. 多头自注意力 + 残差连接 + 层归一化
        # 自注意力：Q、K、V 都来自输入 x
        attn_output = self.self_attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)

        # 残差连接 + 层归一化（Post-LN）
        # 注：也可以使用 Pre-LN（先 LN 再自注意力），效果略有不同
        x = self.norm1(x + attn_output)

        # 2. 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)

        # 残差连接 + 层归一化
        x = self.norm2(x + ff_output)

        return x


class EncoderLayerPreLN(nn.Module):
    """
    使用 Pre-LN 的编码器层

    Pre-LN vs Post-LN：
        - Post-LN：x = LN(x + Sublayer(x))  # 先残差后归一化
        - Pre-LN：x = x + Sublayer(LN(x))   # 先归一化后残差

    Pre-LN 的优点：
        - 训练更稳定，梯度流动更好
        - 不需要学习率预热（warmup）
        - GPT、BERT 等模型常用 Pre-LN
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayerPreLN, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播（Pre-LN 版本）

        参数：
            x: 输入张量 (batch_size, seq_len, d_model)
            mask: 掩码张量

        返回：
            输出张量 (batch_size, seq_len, d_model)
        """
        # 1. 层归一化 -> 多头自注意力 -> 残差连接
        norm_x = self.norm1(x)
        attn_output = self.self_attention(norm_x, norm_x, norm_x, mask)
        attn_output = self.dropout(attn_output)
        x = x + attn_output  # 残差连接

        # 2. 层归一化 -> 前馈网络 -> 残差连接
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        ff_output = self.dropout(ff_output)
        x = x + ff_output  # 残差连接

        return x
