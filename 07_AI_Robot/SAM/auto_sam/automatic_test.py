import os
import cv2
import torch
import numpy as np

from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator
)


# ============================================================
# 1. 参数
# ============================================================

IMAGE_PATH = "test.png"

CHECKPOINT_PATH = (
    "sam_vit_b_01ec64.pth"
)

MODEL_TYPE = "vit_b"

OUTPUT_PATH = (
    "automatic_result.jpg"
)


# ============================================================
# 2. 加载 SAM
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"当前设备: {device}"
)


print(
    "正在加载 SAM..."
)


sam = sam_model_registry[
    MODEL_TYPE
](
    checkpoint=CHECKPOINT_PATH
)


sam.to(
    device=device
)


print(
    "SAM 加载完成"
)


# ============================================================
# 3. 创建自动分割器
# ============================================================

mask_generator = (
    SamAutomaticMaskGenerator(
        sam
    )
)


# ============================================================
# 4. 读取图片
# ============================================================

image_bgr = cv2.imread(
    IMAGE_PATH
)


if image_bgr is None:

    raise FileNotFoundError(
        IMAGE_PATH
    )


image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)


print(
    "图片读取完成"
)


# ============================================================
# 5. 自动生成 Mask
# ============================================================

print(
    "开始自动分割..."
)


masks = mask_generator.generate(
    image_rgb
)


print(
    f"生成 Mask 数量: {len(masks)}"
)


# ============================================================
# 6. 查看每个 Mask 信息
# ============================================================

for i, mask in enumerate(masks):

    print(
        i,
        "面积:",
        mask["area"],
        "IoU:",
        round(
            mask["predicted_iou"],
            3
        ),
        "稳定性:",
        round(
            mask["stability_score"],
            3
        )
    )


# ============================================================
# 7. 可视化所有 Mask
# ============================================================

result = image_bgr.copy()


# 固定随机颜色
np.random.seed(0)


for mask in masks:

    segmentation = (
        mask["segmentation"]
    )


    color = np.random.randint(
        0,
        255,
        size=3
    )


    # BGR
    result[
        segmentation
    ] = (
        result[
            segmentation
        ]
        * 0.5
        +
        color
        * 0.5
    )


# ============================================================
# 8. 保存
# ============================================================

cv2.imwrite(
    OUTPUT_PATH,
    result
)


print(
    "保存完成:"
)

print(
    OUTPUT_PATH
)