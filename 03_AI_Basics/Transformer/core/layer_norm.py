import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）

    作用：对每个样本的特征维度进行归一化，稳定训练过程

    与 Batch Normalization 的区别：
        - BN：在 batch 维度上归一化（依赖 batch 中的其他样本）
        - LN：在特征维度上归一化（不依赖其他样本，适合序列模型）

    公式：
        LN(x) = γ * (x - μ) / (σ + ε) + β
        其中 μ 和 σ 是该样本在特征维度上的均值和标准差

    参数：
        features: 特征维度（通常是 d_model）
        eps: 防止除零的小常数
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        # 可学习的缩放参数 γ（gamma）
        self.gamma = nn.Parameter(torch.ones(features))

        # 可学习的偏移参数 β（beta）
        self.beta = nn.Parameter(torch.zeros(features))

        # 防止除零的小常数
        self.eps = eps

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量 (batch_size, seq_len, features)

        返回：
            归一化后的张量 (batch_size, seq_len, features)
        """
        # 计算均值（在最后一个维度上）
        mean = x.mean(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)

        # 计算标准差（在最后一个维度上）
        std = x.std(dim=-1, keepdim=True)    # (batch_size, seq_len, 1)

        # 归一化
        x_norm = (x - mean) / (std + self.eps)

        # 应用可学习的缩放和偏移
        return self.gamma * x_norm + self.beta
