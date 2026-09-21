from pathlib import Path
import shutil
import time
import traceback

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
VIS_DIR = OUTPUT_DIR / "visualized"   # 可视化叠加图（供检查）
OUT_IMAGE_DIR = OUTPUT_DIR / "images"  # 原图拷贝（供训练）
OUT_LABEL_DIR = OUTPUT_DIR / "labels"  # YOLO-seg 标注

PROMPT = "objects on a white table"
CONFIDENCE = 0.4

CLASS_ID = 1          # 唯一类别 ID（ISAT 默认 __background__ 为 0）
IOU_THRESH = 0.8      # 重复 mask 去重：IoU 超过该值只保留置信度高的
POLY_EPSILON = 1.0    # approxPolyDP 多边形简化精度（像素）
LIMIT = None          # 只处理前 N 张，用于验证（None = 全部）


# =========================
# 工具函数
# =========================

def log(*args) -> None:
    """带时间戳的实时输出（flush=True，输出重定向时也立即显示）。"""
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def dedup_masks(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """按置信度降序保留 mask，与已保留 mask 的 IoU 均小于阈值才留下。返回保留的下标。"""
    keep = []
    for i in np.argsort(-scores):
        if all(mask_iou(masks[i], masks[j]) < IOU_THRESH for j in keep):
            keep.append(int(i))
    return np.array(keep)


def mask_to_polygon(mask: np.ndarray) -> np.ndarray | None:
    """二值 mask -> 最大外轮廓 -> approxPolyDP 简化后的多边形 (N, 2) 像素坐标。"""
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(contour, POLY_EPSILON, closed=True)
    poly = poly.squeeze(1)
    if poly.ndim != 2 or len(poly) < 3:
        return None
    return poly


def polygon_to_yolo_line(poly: np.ndarray, img_w: int, img_h: int) -> str:
    coords = " ".join(f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in poly)
    return f"{CLASS_ID} {coords}"


def save_visualization(image, masks: np.ndarray, out_path: Path) -> None:
    """mask 半透明叠加 + 轮廓描边，保存到 out_path。masks: (N, H, W) 0/1。"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    for mask in masks:
        color = np.random.random(3)

        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        overlay[..., :3] = color
        overlay[..., 3] = mask.astype(float) * 0.45
        ax.imshow(overlay)

        # 轮廓描边
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            c = c.squeeze(1)
            if c.ndim == 2 and len(c) > 2:
                ax.plot(c[:, 0], c[:, 1], color=color, linewidth=1.5)

    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # =========================
    # 1. 检查权重与输入
    # =========================

    if not CHECKPOINT_PATH.exists():
        print(f"❌ 权重不存在: {CHECKPOINT_PATH}")
        print("   请先运行 python download_test.py 下载权重")
        return

    image_files = sorted(IMAGE_DIR.glob("*.png")) + sorted(IMAGE_DIR.glob("*.jpg"))
    if not image_files:
        print(f"❌ 图片目录为空: {IMAGE_DIR}")
        return

    if LIMIT is not None:
        image_files = image_files[:LIMIT]

    log(f"图片数量: {len(image_files)}")

    # =========================
    # 2. 创建输出目录
    # =========================

    for d in (VIS_DIR, OUT_IMAGE_DIR, OUT_LABEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # =========================
    # 3. 选择设备并加载模型
    # =========================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"设备: {device} | 权重: {CHECKPOINT_PATH.name} ({CHECKPOINT_PATH.stat().st_size / 1e9:.2f} GB)")

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

    log("模型加载完成，开始推理\n")

    # =========================
    # 4. 循环推理
    # =========================

    total_instances = 0
    zero_images = []
    failed_images = []

    total_start = time.time()

    for idx, image_path in enumerate(image_files, start=1):
        name = image_path.stem
        img_start = time.time()
        log(f"[{idx}/{len(image_files)}] {name}: 开始推理...")

        try:
            image = Image.open(image_path).convert("RGB")
            img_w, img_h = image.size

            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        state = processor.set_image(image)
                        output = processor.set_text_prompt(state=state, prompt=PROMPT)
                else:
                    state = processor.set_image(image)
                    output = processor.set_text_prompt(state=state, prompt=PROMPT)

            masks = output["masks"]          # (N, 1, H, W) bool
            scores = output["scores"]

            log(f"[{idx}/{len(image_files)}] {name}: 推理完成，得到 {len(scores)} 个候选 mask"
                f"（耗时 {time.time() - img_start:.1f}s）")

            if len(masks) == 0:
                masks_np = np.zeros((0, img_h, img_w), dtype=np.uint8)
            else:
                masks_np = masks.detach().cpu().numpy().squeeze(1).astype(np.uint8)

            scores_np = scores.detach().cpu().float().numpy()  # bf16 不能直接转 numpy

            # IoU 去重
            keep = dedup_masks(masks_np, scores_np) if len(masks_np) > 0 else np.array([], dtype=int)
            masks_np = masks_np[keep]
            log(f"[{idx}/{len(image_files)}] {name}: IoU 去重后保留 {len(masks_np)} 个 mask")

            # 写标注（YOLO-seg 多边形格式）
            label_lines = []
            for mask in masks_np:
                poly = mask_to_polygon(mask)
                if poly is not None:
                    label_lines.append(polygon_to_yolo_line(poly, img_w, img_h))

            (OUT_LABEL_DIR / f"{name}.txt").write_text("\n".join(label_lines), encoding="utf-8")

            # 原图拷贝 + 可视化
            shutil.copy2(image_path, OUT_IMAGE_DIR / image_path.name)
            save_visualization(image, masks_np, VIS_DIR / f"{name}.png")

            total_instances += len(label_lines)

            if len(label_lines) == 0:
                zero_images.append(image_path.name)

            elapsed = time.time() - img_start
            avg_per_img = (time.time() - total_start) / idx
            eta_min = avg_per_img * (len(image_files) - idx) / 60
            log(f"[{idx}/{len(image_files)}] {name}: 完成，输出 {len(label_lines)} 条标注"
                f"（耗时 {elapsed:.1f}s，预计剩余 {eta_min:.1f} 分钟）")

        except Exception as e:
            failed_images.append(image_path.name)
            log(f"[{idx}/{len(image_files)}] ❌ {image_path.name} 失败: {e}")
            print(traceback.format_exc())

    # =========================
    # 5. 汇总
    # =========================

    print()
    print("==========================")
    log(f"处理完成，总耗时 {(time.time() - total_start) / 60:.1f} 分钟")
    print(f"  成功: {len(image_files) - len(failed_images)}/{len(image_files)} 张")
    print(f"  总实例数: {total_instances}")
    print(f"  0 检测图片: {len(zero_images)} 张")
    if zero_images:
        print("   ", ", ".join(zero_images[:20]) + (" ..." if len(zero_images) > 20 else ""))
    print(f"  失败图片: {len(failed_images)} 张")
    if failed_images:
        print("   ", ", ".join(failed_images))
    print(f"  可视化: {VIS_DIR}")
    print(f"  标注:   {OUT_LABEL_DIR}")
    print("==========================")


if __name__ == "__main__":
    main()
