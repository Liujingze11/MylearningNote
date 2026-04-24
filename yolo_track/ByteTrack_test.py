import os
import re
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO


# =========================
# 1. 路径配置
# =========================

MODEL_PATH = "yolo模型位置"              # 你的 YOLO segment 模型
FRAMES_DIR = "771帧图片所在文件夹"               # 771帧图片所在文件夹
OUTPUT_VIDEO = "跟踪结果视频.mp4"
OUTPUT_CSV = "跟踪结果.csv"

IMG_SIZE = 640
CONF = 0.406
IOU = 0.5
FPS = 25


# =========================
# 2. 自然排序函数
# 防止 frame_10.jpg 排在 frame_2.jpg 前面
# =========================

def natural_sort_key(path):
    name = Path(path).name
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", name)]


# =========================
# 3. 读取图片列表
# =========================

image_paths = sorted(
    [
        str(p)
        for p in Path(FRAMES_DIR).glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ],
    key=natural_sort_key
)

if len(image_paths) == 0:
    raise RuntimeError("没有找到图片，请检查 FRAMES_DIR 路径。")

print(f"共找到 {len(image_paths)} 帧图片")


# =========================
# 4. 初始化模型
# =========================

model = YOLO(MODEL_PATH)


# =========================
# 5. 初始化视频写入器
# =========================

first_frame = cv2.imread(image_paths[0])
if first_frame is None:
    raise RuntimeError(f"无法读取第一张图片: {image_paths[0]}")

height, width = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    FPS,
    (width, height)
)


# =========================
# 6. 创建 CSV 结果文件
# =========================

csv_file = open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame_index",
    "image_name",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "mask_points"
])


# =========================
# 7. 逐帧跟踪
# =========================

for frame_index, image_path in enumerate(image_paths):
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"跳过无法读取的图片: {image_path}")
        continue

    result = model.track(
        source=frame,
        task="segment",
        tracker="bytetrack.yaml",
        persist=True,          # 关键：保持上一帧的跟踪状态
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        verbose=False
    )[0]

    # 画出带 mask + track_id 的结果
    annotated_frame = result.plot()
    video_writer.write(annotated_frame)

    # 如果这一帧没有检测结果
    if result.boxes is None or result.boxes.id is None:
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    # segmentation mask polygon
    # result.masks.xy 是每个 mask 的轮廓点，和 boxes 顺序通常对应
    masks = result.masks.xy if result.masks is not None else [None] * len(boxes)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        track_id = track_ids[i]
        class_id = class_ids[i]
        conf = confs[i]
        class_name = model.names[class_id]

        if masks is not None and masks[i] is not None:
            # mask 点很多，这里转成字符串保存
            mask_points = masks[i].astype(int).tolist()
        else:
            mask_points = ""

        csv_writer.writerow([
            frame_index,
            Path(image_path).name,
            track_id,
            class_id,
            class_name,
            float(conf),
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            mask_points
        ])

    print(f"已处理 {frame_index + 1}/{len(image_paths)}: {Path(image_path).name}")


# =========================
# 8. 释放资源
# =========================

csv_file.close()
video_writer.release()

print("跟踪完成！")
print(f"视频结果已保存到: {OUTPUT_VIDEO}")
print(f"CSV结果已保存到: {OUTPUT_CSV}")