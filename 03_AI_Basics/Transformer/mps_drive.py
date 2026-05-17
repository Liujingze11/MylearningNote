import torch  # 导入 PyTorch 库

def pick_device():
    """
    自动选择运行设备（GPU、MPS 或 CPU）

    优先顺序：
    1. CUDA（NVIDIA GPU）
    2. MPS（Apple Silicon 芯片）
    3. CPU（无 GPU 时使用）
    """
    if torch.cuda.is_available():
        # 如果系统中可用 CUDA（如 NVIDIA GPU）
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # 如果使用的是 macOS 并支持 Metal Performance Shaders (MPS)
        return torch.device("mps")
    # 默认使用 CPU
    return torch.device("cpu")



# 调用函数，选择当前最优设备
device = pick_device()


print("Using device:", device)


# ================== 小测试部分 ==================

# 在选定设备上创建一个 1000x1000 的随机矩阵
x = torch.randn(1000, 1000, device=device)

# 在同一设备上进行矩阵乘法运算（x @ x）
y = torch.mm(x, x)

# 打印结果的形状，确认计算完成
print("ok:", y.shape)