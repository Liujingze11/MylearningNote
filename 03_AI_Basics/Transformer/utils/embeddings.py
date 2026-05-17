import torch
import torch.nn as nn
import math


class Embeddings(nn.Module):
    """
    词嵌入层（Token Embeddings）

    作用：将离散的 token id 映射到连续的向量空间

    参数：
        vocab_size: 词表大小
        d_model: 词向量维度
    """

    def __init__(self, vocab_size, d_model):
        # super() 是 Python 中调用父类方法的机制
        # 在 PyTorch 中必须通过 super().__init__() 触发 nn.Module 的内部初始化逻辑
        # 才能正确注册参数、子模块并支持自动求导
        super(Embeddings, self).__init__()

        # nn.Embedding 层：查表（Look-Up Table）
        # 将离散的 token id 映射到一个连续的 d_model 维向量空间中
        # 这个向量是可学习的参数，用于表示 token 的语义信息
        # Embedding 的向量在初始化时是随机的，默认服从均匀分布
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        前向传播

        参数：
            x: token id 张量 (batch_size, seq_len)

        返回：
            词嵌入张量 (batch_size, seq_len, d_model)
        """
        # Embedding 是一个对象，内部定义了 __call__
        # 乘以 sqrt(d_model) 是 Transformer 论文中的做法
        # 目的是让嵌入向量的值相对位置编码更大一些
        return self.lut(x) * math.sqrt(self.d_model)
