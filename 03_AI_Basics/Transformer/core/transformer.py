import torch
import torch.nn as nn
from .encoder import EncoderWithoutDuplicateScaling
from .decoder import Decoder


class Transformer(nn.Module):
    """
    完整的 Transformer 模型

    结构：
        1. 编码器（Encoder）：处理源序列
        2. 解码器（Decoder）：生成目标序列
        3. 输出线性层：将解码器输出映射到词表大小

    应用场景：
        - 机器翻译（如英语 -> 中文）
        - 文本摘要
        - 问答系统
        - 代码生成

    参数：
        src_vocab_size: 源语言词表大小
        tgt_vocab_size: 目标语言词表大小
        d_model: 模型维度（如 512）
        num_layers: 编码器和解码器的层数（如 6）
        num_heads: 注意力头数（如 8）
        d_ff: 前馈网络中间层维度（如 2048）
        max_seq_len: 最大序列长度（如 512）
        dropout: Dropout 概率
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, max_seq_len=512, dropout=0.1):
        super(Transformer, self).__init__()

        # 编码器
        self.encoder = EncoderWithoutDuplicateScaling(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        # 解码器
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        # 输出线性层：将解码器输出映射到目标词表大小
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播

        参数：
            src: 源序列 token ids (batch_size, src_seq_len)
            tgt: 目标序列 token ids (batch_size, tgt_seq_len)
            src_mask: 源序列掩码（用于遮蔽 padding）
            tgt_mask: 目标序列掩码（包含 padding 掩码和因果掩码）

        返回：
            输出 logits (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 1. 编码器：处理源序列
        encoder_output = self.encoder(src, src_mask)  # (batch_size, src_seq_len, d_model)

        # 2. 解码器：生成目标序列
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)  # (batch_size, tgt_seq_len, d_model)

        # 3. 输出投影：映射到词表大小
        output = self.output_projection(decoder_output)  # (batch_size, tgt_seq_len, tgt_vocab_size)

        return output

    def encode(self, src, src_mask=None):
        """
        仅编码（推理时使用）

        参数：
            src: 源序列 token ids (batch_size, src_seq_len)
            src_mask: 源序列掩码

        返回：
            编码器输出 (batch_size, src_seq_len, d_model)
        """
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        仅解码（推理时使用）

        参数：
            tgt: 目标序列 token ids (batch_size, tgt_seq_len)
            encoder_output: 编码器输出 (batch_size, src_seq_len, d_model)
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码

        返回：
            输出 logits (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.output_projection(decoder_output)
        return output


def create_transformer(src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6,
                       num_heads=8, d_ff=2048, max_seq_len=512, dropout=0.1):
    """
    创建 Transformer 模型的便捷函数

    参数：
        src_vocab_size: 源语言词表大小
        tgt_vocab_size: 目标语言词表大小
        d_model: 模型维度
        num_layers: 编码器和解码器的层数
        num_heads: 注意力头数
        d_ff: 前馈网络中间层维度
        max_seq_len: 最大序列长度
        dropout: Dropout 概率

    返回：
        Transformer 模型实例
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout
    )

    # 参数初始化（Xavier 初始化）
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
