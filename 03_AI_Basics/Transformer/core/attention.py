import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    核心思想：将 Query、Key、Value 投影到多个子空间，并行计算注意力

    参数：
        d_model: 模型维度（如 512）
        num_heads: 注意力头的数量（如 8）
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        # 确保 d_model 能被 num_heads 整除
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 定义 Q、K、V 的线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影

        # 输出的线性变换层
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力

        公式：Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

        参数：
            Q: Query 张量 (batch_size, num_heads, seq_len, d_k)
            K: Key 张量 (batch_size, num_heads, seq_len, d_k)
            V: Value 张量 (batch_size, num_heads, seq_len, d_k)
            mask: 掩码张量，用于遮蔽某些位置（如 padding 或未来信息）

        返回：
            output: 注意力输出 (batch_size, num_heads, seq_len, d_k)
            attention_weights: 注意力权重 (batch_size, num_heads, seq_len, seq_len)
        """
        # 计算注意力分数：Q * K^T
        # K.transpose(-2, -1): 交换最后两个维度
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用 mask（如果提供）
        if mask is not None:
            # 将 mask 为 0 的位置填充为一个很小的负数，使其 softmax 后接近 0
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # 在最后一个维度上做 softmax

        # 用注意力权重对 Value 加权求和
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def split_heads(self, x):
        """
        将输入张量分割成多个头

        输入形状：(batch_size, seq_len, d_model)
        输出形状：(batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()

        # 重塑为 (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # 转置为 (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        合并多个头的输出

        输入形状：(batch_size, num_heads, seq_len, d_k)
        输出形状：(batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()

        # 转置回 (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)

        # 重塑为 (batch_size, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        前向传播

        参数：
            Q: Query 张量 (batch_size, seq_len_q, d_model)
            K: Key 张量 (batch_size, seq_len_k, d_model)
            V: Value 张量 (batch_size, seq_len_v, d_model)
            mask: 掩码张量

        返回：
            output: 多头注意力输出 (batch_size, seq_len_q, d_model)
        """
        # 1. 线性变换
        Q = self.W_q(Q)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(K)  # (batch_size, seq_len_k, d_model)
        V = self.W_v(V)  # (batch_size, seq_len_v, d_model)

        # 2. 分割成多个头
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len_v, d_k)

        # 3. 计算缩放点积注意力
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 4. 合并多个头
        output = self.combine_heads(attn_output)  # (batch_size, seq_len_q, d_model)

        # 5. 最后的线性变换
        output = self.W_o(output)

        return output
