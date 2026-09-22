from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =========================
# 参数
# =========================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "test.png"
CHECKPOINT_PATH = BASE_DIR / "sam3.pt"
PROMPT = "objects on a white table"
CONFIDENCE = 0.4


# =========================
# 1. 检查权重是否存在
# =========================

if not CHECKPOINT_PATH.exists():
    print(f"❌ 权重不存在: {CHECKPOINT_PATH}")
    exit(1)

size_gb = CHECKPOINT_PATH.stat().st_size / (1024 ** 3)
print(f"✅ 权重存在: {CHECKPOINT_PATH} ({size_gb:.1f} GB)")


# =========================
# 2. 选择设备
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备：", device)


# =========================
# 3. 加载模型
# =========================

print("正在加载 SAM3...")
model = build_sam3_image_model(
    checkpoint_path=str(CHECKPOINT_PATH),  # 权重文件路径
    load_from_HF=False,                    # 用本地文件，不从网上下载
    device=device,                         # "cuda" 或 "cpu"
    eval_mode=True,                        # 推理模式，关闭训练相关计算，节省显存
)

processor = Sam3Processor(
    model,                                 # 刚加载好的模型
    device=device,                         # 设备
    confidence_threshold=CONFIDENCE,       # 低于 0.4 的结果不要
)

print("SAM3 加载完成")


# =========================
# 4. 读取图片
# =========================

image = Image.open(IMAGE_PATH).convert("RGB")   # .convert("RGB") 强制转三通道
print("图片尺寸：", image.size)


# =========================
# 5. SAM3 推理（两段式）
# =========================

with torch.inference_mode():  # 关闭梯度计算，节省内存

    if device == "cuda":
        # GPU
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            
            state = processor.set_image(image)  # 第一步：喂图片，返回 state（图片的特征表示）
            output = processor.set_text_prompt(
                state=state,
                prompt=PROMPT,
            )   # 第二步：给文本提示，结合图片特征做分割，返回 output

    
    else:
        # CPU
        state = processor.set_image(image)
        output = processor.set_text_prompt(
            state=state,
            prompt=PROMPT,
        )


# =========================
# 6. 获取结果（只取 mask，不取 box）
# =========================

# output 由 SAM3 内部构造返回，用 ["key"] 取出列表，下标对应同一个物体
masks  = output["masks"]
scores = output["scores"]

print()
print("==========================")
print("Prompt:", PROMPT)
print("Confidence threshold:", CONFIDENCE)
print(f"检测实例数量: {len(masks)}  置信度: {' '.join(f'{float(s):.3f}' for s in scores)}")
print("==========================")


# =========================
# 7. 可视化（只画 mask，不画边界框）
# =========================

plt.figure(figsize=(12, 8))
plt.imshow(image)

for i, mask in enumerate(masks):

    # tensor → numpy 数组：去掉多余维度、脱离梯度、搬到 CPU、转换格式
    mask = mask.squeeze().detach().cpu().numpy()

    # -------- 半透明 mask 叠加 --------
    color = np.random.random(3)

    overlay = np.zeros(
        (mask.shape[0], mask.shape[1], 4),
        dtype=np.float32,
    )
    overlay[..., :3] = color                    # RGB = 随机颜色
    overlay[..., 3] = mask.astype(float) * 0.45 # A = mask 区域半透明（0.45）
    plt.imshow(overlay)

    # 不画边界框和置信度文字（对比 sam3_mask_box_confi_photo_single.py）


plt.axis("off")
plt.tight_layout()

OUTPUT_PATH = BASE_DIR / "sam3_result_simple.png"

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.show()

print()
print("结果已保存：", OUTPUT_PATH)
