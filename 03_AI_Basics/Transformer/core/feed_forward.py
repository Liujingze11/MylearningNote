import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈神经网络（Position-wise Feed-Forward Network）

    结构：两层全连接网络 + ReLU 激活函数
    公式：FFN(x) = max(0, xW1 + b1)W2 + b2

    特点：
        - 对序列中的每个位置独立应用相同的前馈网络
        - 中间层维度通常是模型维度的 4 倍（如 d_model=512, d_ff=2048）

    参数：
        d_model: 模型维度（输入和输出维度）
        d_ff: 前馈网络中间层维度
        dropout: Dropout 概率
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()

        # 第一个线性层：d_model -> d_ff（扩展维度）
        self.linear1 = nn.Linear(d_model, d_ff)

        # 第二个线性层：d_ff -> d_model（恢复维度）
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量 (batch_size, seq_len, d_model)

        返回：
            输出张量 (batch_size, seq_len, d_model)
        """
        # x -> 第一个线性层 -> ReLU 激活 -> Dropout -> 第二个线性层
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.linear1(x)

        # ReLU 激活函数
        x = F.relu(x)

        # Dropout
        x = self.dropout(x)

        # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear2(x)

        return x


class PositionWiseFeedForwardGELU(nn.Module):
    """
    使用 GELU 激活函数的位置前馈神经网络

    GELU（Gaussian Error Linear Unit）是一种平滑的激活函数，
    在 BERT、GPT 等模型中广泛使用，效果通常优于 ReLU

    参数：
        d_model: 模型维度（输入和输出维度）
        d_ff: 前馈网络中间层维度
        dropout: Dropout 概率
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForwardGELU, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量 (batch_size, seq_len, d_model)

        返回：
            输出张量 (batch_size, seq_len, d_model)
        """
        # 使用 GELU 替代 ReLU
        x = self.linear1(x)
        x = F.gelu(x)  # GELU 激活函数
        x = self.dropout(x)
        x = self.linear2(x)

        return x
