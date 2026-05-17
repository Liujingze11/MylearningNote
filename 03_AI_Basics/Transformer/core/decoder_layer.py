import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward
from .layer_norm import LayerNorm


class DecoderLayer(nn.Module):
    """
    Transformer 解码器层

    结构：
        1. 掩码多头自注意力（Masked Multi-Head Self-Attention）
           - 确保预测第 i 个位置时只能看到前 i-1 个位置的信息
        2. 残差连接 + 层归一化（Add & Norm）
        3. 编码器-解码器注意力（Encoder-Decoder Attention，交叉注意力）
           - Query 来自解码器，Key 和 Value 来自编码器输出
        4. 残差连接 + 层归一化（Add & Norm）
        5. 位置前馈网络（Position-wise Feed-Forward）
        6. 残差连接 + 层归一化（Add & Norm）

    参数：
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络中间层维度
        dropout: Dropout 概率
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # 掩码自注意力（用于解码器自身）
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # 编码器-解码器注意力（交叉注意力）
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        # 位置前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # 三个层归一化
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播

        参数：
            x: 解码器输入 (batch_size, tgt_seq_len, d_model)
            encoder_output: 编码器输出 (batch_size, src_seq_len, d_model)
            src_mask: 源序列掩码（用于遮蔽编码器的 padding）
            tgt_mask: 目标序列掩码（用于遮蔽解码器的 padding 和未来信息）

        返回：
            输出张量 (batch_size, tgt_seq_len, d_model)
        """
        # 1. 掩码自注意力 + 残差连接 + 层归一化
        # 自注意力：Q、K、V 都来自解码器输入 x
        # tgt_mask 包含因果掩码（causal mask），确保不能看到未来信息
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        self_attn_output = self.dropout(self_attn_output)
        x = self.norm1(x + self_attn_output)

        # 2. 编码器-解码器注意力（交叉注意力）+ 残差连接 + 层归一化
        # Q 来自解码器，K 和 V 来自编码器输出
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        cross_attn_output = self.dropout(cross_attn_output)
        x = self.norm2(x + cross_attn_output)

        # 3. 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm3(x + ff_output)

        return x


class DecoderLayerPreLN(nn.Module):
    """
    使用 Pre-LN 的解码器层

    Pre-LN 的优点：
        - 训练更稳定
        - 梯度流动更好
        - 不需要学习率预热
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayerPreLN, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播（Pre-LN 版本）

        参数：
            x: 解码器输入 (batch_size, tgt_seq_len, d_model)
            encoder_output: 编码器输出 (batch_size, src_seq_len, d_model)
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码

        返回：
            输出张量 (batch_size, tgt_seq_len, d_model)
        """
        # 1. 层归一化 -> 掩码自注意力 -> 残差连接
        norm_x = self.norm1(x)
        self_attn_output = self.self_attention(norm_x, norm_x, norm_x, tgt_mask)
        self_attn_output = self.dropout(self_attn_output)
        x = x + self_attn_output

        # 2. 层归一化 -> 交叉注意力 -> 残差连接
        norm_x = self.norm2(x)
        cross_attn_output = self.cross_attention(norm_x, encoder_output, encoder_output, src_mask)
        cross_attn_output = self.dropout(cross_attn_output)
        x = x + cross_attn_output

        # 3. 层归一化 -> 前馈网络 -> 残差连接
        norm_x = self.norm3(x)
        ff_output = self.feed_forward(norm_x)
        ff_output = self.dropout(ff_output)
        x = x + ff_output

        return x
