import torch
import torch.nn as nn


def create_padding_mask(seq, pad_idx=0):
    """
    创建 Padding 掩码

    作用：在注意力计算中遮蔽 padding 位置，使模型不关注填充的 token

    参数：
        seq: 输入序列 (batch_size, seq_len)
        pad_idx: padding token 的索引（通常是 0）

    返回：
        掩码张量 (batch_size, 1, 1, seq_len)
        - 1 表示有效位置
        - 0 表示 padding 位置（需要遮蔽）
    """
    # seq != pad_idx 会返回一个布尔张量，True 表示非 padding 位置
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask


def create_causal_mask(size):
    """
    创建因果掩码（Causal Mask）/ 自回归掩码

    作用：在解码器的自注意力中，确保预测第 i 个位置时只能看到前 i-1 个位置
    这是自回归生成的关键：生成当前 token 时不能看到未来的 token

    参数：
        size: 序列长度

    返回：
        因果掩码张量 (1, size, size)
        - 下三角矩阵（包括对角线）为 1
        - 上三角矩阵为 0

    示例：
        size = 4 时，掩码为：
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    """
    # torch.tril 返回下三角矩阵
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0)  # (1, size, size)
    return mask


def create_decoder_mask(tgt_seq, pad_idx=0):
    """
    创建解码器掩码（组合 padding 掩码和因果掩码）

    解码器的自注意力需要同时满足：
        1. 不关注 padding 位置
        2. 不关注未来位置（因果性）

    参数：
        tgt_seq: 目标序列 (batch_size, tgt_seq_len)
        pad_idx: padding token 的索引

    返回：
        组合掩码 (batch_size, 1, tgt_seq_len, tgt_seq_len)
    """
    batch_size, tgt_seq_len = tgt_seq.size()

    # 1. 创建 padding 掩码
    tgt_padding_mask = (tgt_seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_seq_len)

    # 2. 创建因果掩码
    tgt_causal_mask = create_causal_mask(tgt_seq_len).to(tgt_seq.device)  # (1, tgt_seq_len, tgt_seq_len)

    # 3. 组合两个掩码（逻辑与操作）
    tgt_mask = tgt_padding_mask & tgt_causal_mask  # (batch_size, 1, tgt_seq_len, tgt_seq_len)

    return tgt_mask


def count_parameters(model):
    """
    统计模型的参数数量

    参数：
        model: PyTorch 模型

    返回：
        total_params: 总参数数
        trainable_params: 可训练参数数
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def print_model_info(model):
    """
    打印模型信息

    参数：
        model: PyTorch 模型
    """
    total_params, trainable_params = count_parameters(model)

    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    print(f"参数大小: {total_params * 4 / 1024 / 1024:.2f} MB（假设每个参数 4 字节）")


def generate_square_subsequent_mask(sz):
    """
    生成方形的后续掩码（PyTorch 官方风格）

    参数：
        sz: 序列长度

    返回：
        掩码张量 (sz, sz)
        - 对角线及以下为 0.0
        - 上三角为 -inf

    注：此掩码与 create_causal_mask 作用相同，但使用方式不同
    这个函数返回的掩码直接加到注意力分数上，而不是用 masked_fill
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class LabelSmoothing(nn.Module):
    """
    标签平滑（Label Smoothing）

    作用：
        - 防止模型过度自信（输出概率过于接近 0 或 1）
        - 提高模型的泛化能力
        - 在机器翻译等任务中常用

    原理：
        - 原始标签：[0, 0, 1, 0]（one-hot）
        - 平滑后：[ε/(N-1), ε/(N-1), 1-ε, ε/(N-1)]
        其中 ε 是平滑系数，N 是类别数

    参数：
        vocab_size: 词表大小
        padding_idx: padding token 的索引
        smoothing: 平滑系数（如 0.1）
    """

    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    def forward(self, x, target):
        """
        前向传播

        参数：
            x: 模型输出（log probabilities）(batch_size * seq_len, vocab_size)
            target: 目标标签 (batch_size * seq_len)

        返回：
            损失值
        """
        assert x.size(1) == self.vocab_size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
