from pathlib import Path
import shutil
import time

import torch
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")  # 无界面环境，不弹窗
import matplotlib.pyplot as plt
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =========================
# 参数
# =========================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "images"
CHECKPOINT_PATH = BASE_DIR / "sam3.pt"

OUTPUT_DIR = BASE_DIR / "output"
VIS_DIR = OUTPUT_DIR / "visualized"   # 推理叠加图
OUT_IMAGE_DIR = OUTPUT_DIR / "images"  # 原图拷贝
OUT_LABEL_DIR = OUTPUT_DIR / "labels"  # YOLO-seg 标注

PROMPT = "objects on a white table"
CONFIDENCE = 0.5
CLASS_ID = 1          # 唯一类别 ID
POLY_EPSILON = 1.0    # approxPolyDP 多边形简化精度（像素）


# =========================
# 工具函数
# =========================

def mask_to_polygon(mask: np.ndarray) -> np.ndarray | None:
    """二值 mask 转为简化后的多边形坐标。"""

    mask_u8 = (mask > 0).astype(np.uint8)   # 转成 OpenCV 使用的 uint8 二值图

    # 提取外轮廓
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)    # 只保留面积最大的轮廓

    # 简化轮廓点
    poly = cv2.approxPolyDP(contour, POLY_EPSILON, closed=True)
    poly = poly.squeeze(1)
    if poly.ndim != 2 or len(poly) < 3:
        return None
    return poly


def polygon_to_yolo_line(poly: np.ndarray, img_w: int, img_h: int) -> str:
    """简化多边形坐标转 YOLO 格式。"""
    coords = " ".join(f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in poly)
    return f"{CLASS_ID} {coords}"


def save_visualization(image, masks: np.ndarray, out_path: Path) -> None:
    """保存 Mask 可视化结果到 out_path。"""

    # 创建画布并显示原图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    for mask in masks:
        color = np.random.random(3) # 每个实例随机颜色

        # 半透明 mask 叠加在原图上
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        overlay[..., :3] = color
        overlay[..., 3] = mask.astype(float) * 0.45
        ax.imshow(overlay)

        # 绘制轮廓
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            c = c.squeeze(1)
            if c.ndim == 2 and len(c) > 2:
                ax.plot(c[:, 0], c[:, 1], color=color, linewidth=1.5)

    # 调整轴和布局
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================
# 1. 检查权重并创建输出目录
# =========================

if not CHECKPOINT_PATH.exists():
    print(f"❌ 权重不存在: {CHECKPOINT_PATH}")
    exit(1)

for d in (VIS_DIR, OUT_IMAGE_DIR, OUT_LABEL_DIR):
    d.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 加载模型
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {device} | 权重: {CHECKPOINT_PATH.name} ({CHECKPOINT_PATH.stat().st_size / 1e9:.2f} GB)")

model = build_sam3_image_model(
    checkpoint_path=str(CHECKPOINT_PATH),
    load_from_HF=False,
    device=device,
    eval_mode=True,
)

processor = Sam3Processor(
    model,
    device=device,
    confidence_threshold=CONFIDENCE,
)

print("模型加载完成，开始推理\n")


# =========================
# 3. 循环推理
# =========================

image_files = sorted(IMAGE_DIR.glob("*.png"))
total = len(image_files)
print(f"待处理图片: {total} 张")

total_start = time.time()

for idx, image_path in enumerate(image_files, start=1):

    name = image_path.stem
    img_start = time.time()
    print(f"[{idx}/{total}] {name}: 开始推理...")

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    with torch.inference_mode():

        # 如果使用 NVIDIA GPU，建议开启 bf16 autocast
        if device == "cuda":

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(image)

                output = processor.set_text_prompt(
                    state=state,
                    prompt=PROMPT,
                )

        else:

            state = processor.set_image(image)

            output = processor.set_text_prompt(
                state=state,
                prompt=PROMPT,
            )

    # 只取 mask
    masks = output["masks"]

    print(f"[{idx}/{total}] {name}: 推理完成，得到 {len(masks)} 个 mask"
        f"（耗时 {time.time() - img_start:.1f}s）")

    if len(masks) == 0:
        masks_np = np.zeros((0, img_h, img_w), dtype=np.uint8)
    else:
        masks_np = masks.detach().cpu().numpy().squeeze(1).astype(np.uint8)

    # mask -> YOLO-seg 多边形标注
    label_lines = []
    for mask in masks_np:
        poly = mask_to_polygon(mask)
        if poly is not None:
            label_lines.append(polygon_to_yolo_line(poly, img_w, img_h))

    (OUT_LABEL_DIR / f"{name}.txt").write_text("\n".join(label_lines), encoding="utf-8")

    # 原图拷贝 + 可视化
    shutil.copy2(image_path, OUT_IMAGE_DIR / image_path.name)
    save_visualization(image, masks_np, VIS_DIR / f"{name}.png")

    elapsed = time.time() - img_start
    avg_per_img = (time.time() - total_start) / idx
    eta_min = avg_per_img * (total - idx) / 60
    print(f"[{idx}/{total}] {name}: 完成，输出 {len(label_lines)} 条标注"
        f"（耗时 {elapsed:.1f}s，预计剩余 {eta_min:.1f} 分钟）")

print()
print(f"完成，总耗时 {(time.time() - total_start) / 60:.1f} 分钟，结果在 {OUTPUT_DIR}")
