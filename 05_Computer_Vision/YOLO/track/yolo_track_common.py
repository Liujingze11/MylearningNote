import os
import re
import cv2
import csv
import json
from pathlib import Path
from ultralytics import YOLO


# =========================
# 1. 路径配置
# =========================

MODEL_PATH = "/home/liubohan/lbh/Jingze_Liu/yolo_track/best.pt"
FRAMES_DIR = "/home/liubohan/lbh/Jingze_Liu/yolo_track/frames"
OUTPUT_ROOT = "/home/liubohan/lbh/Jingze_Liu/yolo_track/output"

EXPERIMENT_NAME = "botsort_test"
AUTO_ADD_TRACKER_NAME = True
# =========================
# 2. 跟踪参数配置
# =========================

# 可选：
# "bytetrack"
# "botsort"
TRACKER_TYPE = "botsort"

# 如果你想用自定义 yaml，就把下面改成 True
USE_CUSTOM_TRACKER = True

# 自定义 tracker yaml 路径
CUSTOM_TRACKER_YAML = "/home/liubohan/lbh/Jingze_Liu/yolo_track/botsort_custom.yaml"

# YOLO任务类型
# segment：分割模型
# detect：检测模型
TASK = "segment"

IMG_SIZE = 640
CONF = 0.406
IOU = 0.5
FPS = 25

# GPU 设置
# 0 表示第一张GPU
# "cpu" 表示CPU
# None 表示让 Ultralytics 自动选择
DEVICE = 0


# =========================
# 3. 根据参数选择 tracker
# =========================

def get_tracker_config():
    if USE_CUSTOM_TRACKER:
        return CUSTOM_TRACKER_YAML

    if TRACKER_TYPE.lower() == "bytetrack":
        return "bytetrack.yaml"

    if TRACKER_TYPE.lower() == "botsort":
        return "botsort.yaml"

    raise ValueError("TRACKER_TYPE 只能是 'bytetrack' 或 'botsort'")


TRACKER_CONFIG = get_tracker_config()


# =========================
# 4. 自动生成输出路径
# =========================

tracker_name = Path(TRACKER_CONFIG).stem

if AUTO_ADD_TRACKER_NAME:
    output_folder_name = f"{EXPERIMENT_NAME}_{tracker_name}"
else:
    output_folder_name = EXPERIMENT_NAME

OUTPUT_DIR = Path(OUTPUT_ROOT) / output_folder_name

OUTPUT_VIDEO = OUTPUT_DIR / "tracking_result.mp4"
OUTPUT_CSV = OUTPUT_DIR / "tracking_result.csv"
OUTPUT_FRAMES_DIR = OUTPUT_DIR / "tracking_frames"


# =========================
# 5. 自然排序函数
# 防止 frame_10.jpg 排在 frame_2.jpg 前面
# =========================

def natural_sort_key(path):
    name = Path(path).name
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", name)
    ]


# =========================
# 6. 读取图片列表
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
print(f"当前使用 tracker: {TRACKER_CONFIG}")


# =========================
# 7. 创建输出文件夹
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 8. 初始化模型
# =========================

model = YOLO(MODEL_PATH)


# =========================
# 9. 初始化视频写入器
# =========================

first_frame = cv2.imread(image_paths[0])

if first_frame is None:
    raise RuntimeError(f"无法读取第一张图片: {image_paths[0]}")

height, width = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    FPS,
    (width, height)
)

if not video_writer.isOpened():
    raise RuntimeError("视频写入器初始化失败，请检查 OUTPUT_VIDEO 路径或编码格式。")


# =========================
# 10. 创建 CSV 结果文件
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
# 11. 逐帧跟踪
# =========================

for frame_index, image_path in enumerate(image_paths):
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"跳过无法读取的图片: {image_path}")
        continue

    result = model.track(
        source=frame,
        task=TASK,
        tracker=TRACKER_CONFIG,
        persist=True,          # 关键：逐帧图片输入时必须保持跟踪状态
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        verbose=False
    )[0]

    # =========================
    # 11.1 生成可视化结果图
    # =========================

    annotated_frame = result.plot()

    if annotated_frame.shape[:2] != (height, width):
        annotated_frame = cv2.resize(annotated_frame, (width, height))

    # =========================
    # 11.2 保存每一帧结果图片
    # =========================

    output_image_path = OUTPUT_FRAMES_DIR / Path(image_path).name
    cv2.imwrite(str(output_image_path), annotated_frame)

    # =========================
    # 11.3 写入视频
    # =========================

    video_writer.write(annotated_frame)

    # =========================
    # 11.4 保存 CSV 数据
    # =========================

    if result.boxes is None or result.boxes.id is None:
        print(
            f"已处理 {frame_index + 1}/{len(image_paths)}: "
            f"{Path(image_path).name}，无检测结果或无 track_id"
        )
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    if result.masks is not None and result.masks.xy is not None:
        masks = result.masks.xy
    else:
        masks = [None] * len(boxes)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]

        track_id = track_ids[i]
        class_id = class_ids[i]
        conf_score = confs[i]
        class_name = model.names[class_id]

        if masks is not None and i < len(masks) and masks[i] is not None:
            mask_points = masks[i].astype(int).tolist()
            mask_points = json.dumps(mask_points, ensure_ascii=False)
        else:
            mask_points = ""

        csv_writer.writerow([
            frame_index,
            Path(image_path).name,
            track_id,
            class_id,
            class_name,
            float(conf_score),
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            mask_points
        ])

    print(f"已处理 {frame_index + 1}/{len(image_paths)}: {Path(image_path).name}")


# =========================
# 12. 释放资源
# =========================

csv_file.close()
video_writer.release()

print("跟踪完成！")
print(f"使用 tracker: {TRACKER_CONFIG}")
print(f"视频结果已保存到: {OUTPUT_VIDEO}")
print(f"CSV 结果已保存到: {OUTPUT_CSV}")
print(f"逐帧结果图片已保存到: {OUTPUT_FRAMES_DIR}")