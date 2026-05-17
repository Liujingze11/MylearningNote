import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# 添加父目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import create_transformer, create_padding_mask, create_decoder_mask, print_model_info


def simple_example():
    """
    简单示例：演示 Transformer 的基本使用
    """
    print("=" * 50)
    print("简单示例：Transformer 基本使用")
    print("=" * 50)

    # 1. 定义超参数
    src_vocab_size = 10000  # 源语言词表大小
    tgt_vocab_size = 10000  # 目标语言词表大小
    d_model = 512           # 模型维度
    num_layers = 6          # 编码器/解码器层数
    num_heads = 8           # 注意力头数
    d_ff = 2048             # 前馈网络中间层维度
    max_seq_len = 100       # 最大序列长度
    dropout = 0.1           # Dropout 概率

    # 2. 创建模型
    model = create_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout
    )

    # 打印模型信息
    print_model_info(model)
    print()

    # 3. 创建示例数据
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8

    # 源序列（随机 token ids）
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    print(f"源序列形状: {src.shape}")

    # 目标序列（随机 token ids）
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    print(f"目标序列形状: {tgt.shape}")

    # 4. 创建掩码
    src_mask = create_padding_mask(src, pad_idx=0)
    tgt_mask = create_decoder_mask(tgt, pad_idx=0)

    print(f"源序列掩码形状: {src_mask.shape}")
    print(f"目标序列掩码形状: {tgt_mask.shape}")
    print()

    # 5. 前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        output = model(src, tgt, src_mask, tgt_mask)

    print(f"输出形状: {output.shape}")  # (batch_size, tgt_seq_len, tgt_vocab_size)
    print(f"输出维度说明: (批次大小, 目标序列长度, 目标词表大小)")
    print()


def training_example():
    """
    训练示例：演示如何训练 Transformer
    """
    print("=" * 50)
    print("训练示例：Transformer 训练流程")
    print("=" * 50)

    # 1. 超参数
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 256
    num_layers = 3
    num_heads = 8
    d_ff = 1024
    dropout = 0.1

    # 2. 创建模型
    model = create_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout
    )

    print_model_info(model)
    print()

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # 4. 创建模拟数据
    batch_size = 4
    src_seq_len = 10
    tgt_seq_len = 8

    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt_input = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    tgt_output = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))

    # 5. 创建掩码
    src_mask = create_padding_mask(src, pad_idx=0)
    tgt_mask = create_decoder_mask(tgt_input, pad_idx=0)

    # 6. 训练一步
    model.train()

    # 前向传播
    output = model(src, tgt_input, src_mask, tgt_mask)  # (batch_size, tgt_seq_len, tgt_vocab_size)

    # 计算损失
    # 将 output 重塑为 (batch_size * tgt_seq_len, tgt_vocab_size)
    # 将 tgt_output 重塑为 (batch_size * tgt_seq_len)
    output = output.contiguous().view(-1, tgt_vocab_size)
    tgt_output = tgt_output.contiguous().view(-1)

    loss = criterion(output, tgt_output)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"训练损失: {loss.item():.4f}")
    print()


def inference_example():
    """
    推理示例：演示如何使用 Transformer 进行自回归生成
    """
    print("=" * 50)
    print("推理示例：Transformer 自回归生成")
    print("=" * 50)

    # 1. 创建模型
    model = create_transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=256,
        num_layers=3,
        num_heads=8,
        d_ff=1024
    )

    model.eval()

    # 2. 源序列
    src = torch.randint(1, 1000, (1, 10))  # (1, 10)
    src_mask = create_padding_mask(src, pad_idx=0)

    # 3. 编码源序列
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)

    print(f"编码器输出形状: {encoder_output.shape}")

    # 4. 自回归生成
    max_len = 20
    start_token = 1  # 开始 token
    end_token = 2    # 结束 token

    # 初始化目标序列（只有开始 token）
    tgt = torch.tensor([[start_token]], dtype=torch.long)  # (1, 1)

    print(f"\n开始生成...")
    print(f"初始序列: {tgt.tolist()}")

    for i in range(max_len):
        # 创建目标序列掩码
        tgt_mask = create_decoder_mask(tgt, pad_idx=0)

        # 解码
        with torch.no_grad():
            output = model.decode(tgt, encoder_output, src_mask, tgt_mask)  # (1, current_len, vocab_size)

        # 获取最后一个位置的预测
        next_token_logits = output[:, -1, :]  # (1, vocab_size)
        next_token = torch.argmax(next_token_logits, dim=-1)  # (1,)

        # 将预测的 token 添加到序列中
        tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)  # (1, current_len + 1)

        print(f"步骤 {i+1}: 生成 token {next_token.item()}, 当前序列: {tgt.tolist()}")

        # 如果生成了结束 token，停止
        if next_token.item() == end_token:
            print(f"\n生成结束（遇到结束 token）")
            break

    print(f"\n最终生成序列: {tgt.tolist()}")
    print(f"序列长度: {tgt.size(1)}")
    print()


if __name__ == "__main__":
    # 运行示例
    simple_example()
    training_example()
    inference_example()

    print("=" * 50)
    print("所有示例运行完成！")
    print("=" * 50)
