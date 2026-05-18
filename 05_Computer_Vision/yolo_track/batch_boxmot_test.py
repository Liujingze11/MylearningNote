import os
import re
import cv2
import csv
import json
import time
import torch
import numpy as np

from pathlib import Path
from ultralytics import YOLO
from boxmot.trackers.tracker_zoo import create_tracker


# =========================
# 1. 路径配置
# =========================

MODEL_PATH = "/home/liubohan/lbh/Jingze_Liu/yolo_track/best.pt"
FRAMES_DIR = "/home/liubohan/lbh/Jingze_Liu/yolo_track/frames"
OUTPUT_ROOT = "/home/liubohan/lbh/Jingze_Liu/yolo_track/output"

# 总实验名称：会自动加上模型名
EXPERIMENT_NAME = "boxmot_batch"

# YOLO segment 参数
TASK = "segment"
IMG_SIZE = 640
CONF = 0.406
IOU = 0.5
FPS = 25

# YOLO 使用的设备
YOLO_DEVICE = 0

# BoxMOT 使用的设备
BOXMOT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HALF = True if torch.cuda.is_available() else False

# ReID 模型
# 常用轻量模型：osnet_x0_25_msmt17.pt
# 首次运行可能会自动下载；自动下载失败时，把权重文件手动放到当前目录或指定绝对路径。
REID_WEIGHTS = Path("/home/liubohan/lbh/Jingze_Liu/yolo_track/weights/osnet_x0_25_msmt17.pt")


# =========================
# 2. 批量 Tracker 配置
# =========================
# BoxMOT 支持 boosttrack、deepocsort、botsort 等 tracker。
# 这里每个方法都会单独生成一个文件夹。

TRACKER_EXPERIMENTS = [
    {
        "name": "boosttrack_osnet_x0_25_msmt17",
        "tracker_type": "boosttrack",
        "params": {
            # 遮挡场景可以适当加大 max_age
            "max_age": 90,
            "min_hits": 2,
            "det_thresh": 0.35,
            "iou_threshold": 0.3,

            # BoostTrack + ReID
            "with_reid": True,

            # BoostTrack++ 相关增强
            "use_rich_s": True,
            "use_sb": True,
            "use_vt": True,
        }
    },
    {
        "name": "deepocsort_osnet_x0_25_msmt17",
        "tracker_type": "deepocsort",
        "params": {
            "max_age": 90,
            "min_hits": 2,
            "det_thresh": 0.35,
            "iou_thresh": 0.3,

            # False 表示不要关闭 embedding，也就是启用 ReID 外观特征
            "embedding_off": False,
            "cmc_off": False,
            "aw_off": False,
        }
    },
    {
        "name": "botsort_reid_osnet_x0_25_msmt17",
        "tracker_type": "botsort",
        "params": {
            # 遮挡后保留轨迹更久
            "track_buffer": 90,

            # 高置信检测阈值
            "track_high_thresh": 0.45,

            # 低置信检测也参与二阶段匹配
            "track_low_thresh": 0.05,

            # 新建轨迹阈值稍微高一点，避免频繁产生新 ID
            "new_track_thresh": 0.65,

            "match_thresh": 0.85,
            "proximity_thresh": 0.6,
            "appearance_thresh": 0.35,
            "cmc_method": "ecc",

            # BoT-SORT + ReID
            "with_reid": True,
        }
    }
]


# =========================
# 3. 自然排序函数
# =========================

def natural_sort_key(path):
    name = Path(path).name
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", name)
    ]


# =========================
# 4. 读取图片列表
# =========================

def load_image_paths(frames_dir):
    image_paths = sorted(
        [
            str(p)
            for p in Path(frames_dir).glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ],
        key=natural_sort_key
    )

    if len(image_paths) == 0:
        raise RuntimeError(f"没有找到图片，请检查路径：{frames_dir}")

    return image_paths


# =========================
# 5. 颜色函数
# =========================

def color_for_id(track_id):
    np.random.seed(int(track_id) + 12345)
    color = np.random.randint(50, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


# =========================
# 6. 绘制 BoxMOT 结果
# =========================

def draw_tracks(frame, tracks, masks, class_names):
    vis = frame.copy()

    for track in tracks:
        if len(track) < 7:
            continue

        x1, y1, x2, y2 = track[:4]
        track_id = int(track[4])
        conf_score = float(track[5])
        class_id = int(track[6])

        # BoxMOT 当前输出通常为：
        # x1, y1, x2, y2, track_id, conf, cls, det_ind
        det_index = int(track[7]) if len(track) >= 8 else -1

        color = color_for_id(track_id)

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # 画 mask
        if 0 <= det_index < len(masks):
            polygon = masks[det_index].astype(np.int32)

            if len(polygon) >= 3:
                overlay = vis.copy()
                cv2.fillPoly(overlay, [polygon], color)
                vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)

                cv2.polylines(
                    vis,
                    [polygon],
                    isClosed=True,
                    color=color,
                    thickness=2
                )

        # 画 bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        class_name = class_names.get(class_id, str(class_id))
        label = f"ID {track_id} | {class_name} {conf_score:.2f}"

        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

    return vis


# =========================
# 7. YOLO result 转 BoxMOT dets
# =========================

def yolo_result_to_dets(result):
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy().reshape(-1, 1)
    cls = result.boxes.cls.cpu().numpy().reshape(-1, 1)

    dets = np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)

    return dets


# =========================
# 8. 创建 BoxMOT tracker
# =========================

def build_tracker(tracker_type, params):
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=None,
        reid_weights=REID_WEIGHTS,
        device=BOXMOT_DEVICE,
        half=HALF,
        per_class=False,

        # 用这个传入自定义参数
        evolve_param_dict=params
    )

    return tracker


# =========================
# 9. 单个 tracker 实验
# =========================

def run_one_tracker(model, image_paths, exp_cfg, batch_output_dir):
    tracker_name = exp_cfg["name"]
    tracker_type = exp_cfg["tracker_type"]
    tracker_params = exp_cfg["params"]

    print("=" * 80)
    print(f"开始测试：{tracker_name}")
    print(f"tracker_type: {tracker_type}")
    print("=" * 80)

    tracker_output_dir = batch_output_dir / tracker_name
    output_video = tracker_output_dir / "tracking_result.mp4"
    output_csv = tracker_output_dir / "tracking_result.csv"
    output_frames_dir = tracker_output_dir / "tracking_frames"
    output_summary_json = tracker_output_dir / "run_summary.json"

    tracker_output_dir.mkdir(parents=True, exist_ok=True)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    # 读取第一帧，初始化视频写入器
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        raise RuntimeError(f"无法读取第一张图片：{image_paths[0]}")

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        FPS,
        (width, height)
    )

    if not video_writer.isOpened():
        raise RuntimeError(f"视频写入器初始化失败：{output_video}")

    # 创建 CSV
    csv_file = open(output_csv, mode="w", newline="", encoding="utf-8")
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
        "det_index",
        "mask_points"
    ])

    # 创建 tracker
    tracker = build_tracker(
        tracker_type=tracker_type,
        params=tracker_params
    )

    total_tracks_written = 0
    frames_with_tracks = 0
    unique_track_ids = set()

    start_time = time.time()

    for frame_index, image_path in enumerate(image_paths):
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"跳过无法读取的图片：{image_path}")
            continue

        # 1. YOLO segment 预测
        result = model.predict(
            source=frame,
            task=TASK,
            imgsz=IMG_SIZE,
            conf=CONF,
            iou=IOU,
            device=YOLO_DEVICE,
            verbose=False
        )[0]

        # 2. YOLO detections 转 BoxMOT 输入
        dets = yolo_result_to_dets(result)

        # 3. BoxMOT 更新轨迹
        tracks = tracker.update(dets, frame)

        if tracks is None:
            tracks = np.empty((0, 8), dtype=np.float32)

        tracks = np.asarray(tracks)

        # 4. 获取 mask
        if result.masks is not None and result.masks.xy is not None:
            masks = result.masks.xy
        else:
            masks = []

        # 5. 绘制结果
        annotated_frame = draw_tracks(
            frame=frame,
            tracks=tracks,
            masks=masks,
            class_names=model.names
        )

        if annotated_frame.shape[:2] != (height, width):
            annotated_frame = cv2.resize(annotated_frame, (width, height))

        # 6. 保存帧图
        output_image_path = output_frames_dir / Path(image_path).name
        cv2.imwrite(str(output_image_path), annotated_frame)

        # 7. 写入视频
        video_writer.write(annotated_frame)

        # 8. 保存 CSV
        if tracks.size > 0:
            frames_with_tracks += 1

        for track in tracks:
            if len(track) < 7:
                continue

            x1, y1, x2, y2 = track[:4]
            track_id = int(track[4])
            conf_score = float(track[5])
            class_id = int(track[6])
            det_index = int(track[7]) if len(track) >= 8 else -1

            class_name = model.names[class_id] if class_id in model.names else str(class_id)

            mask_points = ""
            if 0 <= det_index < len(masks):
                mask_points = json.dumps(
                    masks[det_index].astype(int).tolist(),
                    ensure_ascii=False
                )

            csv_writer.writerow([
                frame_index,
                Path(image_path).name,
                track_id,
                class_id,
                class_name,
                conf_score,
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                det_index,
                mask_points
            ])

            total_tracks_written += 1
            unique_track_ids.add(track_id)

        print(
            f"[{tracker_name}] 已处理 {frame_index + 1}/{len(image_paths)} "
            f"{Path(image_path).name} | tracks: {len(tracks)}"
        )

    elapsed = time.time() - start_time

    csv_file.close()
    video_writer.release()

    summary = {
        "tracker_name": tracker_name,
        "tracker_type": tracker_type,
        "model_path": MODEL_PATH,
        "reid_weights": str(REID_WEIGHTS),
        "frames_total": len(image_paths),
        "frames_with_tracks": frames_with_tracks,
        "total_tracks_written": total_tracks_written,
        "unique_track_id_count": len(unique_track_ids),
        "unique_track_ids": sorted(list(unique_track_ids)),
        "elapsed_seconds": elapsed,
        "output_video": str(output_video),
        "output_csv": str(output_csv),
        "output_frames_dir": str(output_frames_dir),
        "tracker_params": tracker_params
    }

    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"完成：{tracker_name}")
    print(f"视频：{output_video}")
    print(f"CSV：{output_csv}")
    print(f"帧图：{output_frames_dir}")
    print()

    return summary


# =========================
# 10. 主函数：批量运行
# =========================

def main():
    image_paths = load_image_paths(FRAMES_DIR)

    model_stem = Path(MODEL_PATH).stem
    batch_output_dir = Path(OUTPUT_ROOT) / f"{EXPERIMENT_NAME}_{model_stem}"
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"共找到 {len(image_paths)} 帧图片")
    print(f"模型路径：{MODEL_PATH}")
    print(f"批量输出目录：{batch_output_dir}")
    print(f"BoxMOT device：{BOXMOT_DEVICE}")
    print(f"ReID weights：{REID_WEIGHTS}")

    model = YOLO(MODEL_PATH)

    all_summaries = []

    for exp_cfg in TRACKER_EXPERIMENTS:
        try:
            summary = run_one_tracker(
                model=model,
                image_paths=image_paths,
                exp_cfg=exp_cfg,
                batch_output_dir=batch_output_dir
            )
            all_summaries.append(summary)

        except Exception as e:
            print(f"方法 {exp_cfg['name']} 运行失败：{e}")

            fail_summary = {
                "tracker_name": exp_cfg["name"],
                "tracker_type": exp_cfg["tracker_type"],
                "error": str(e)
            }
            all_summaries.append(fail_summary)

    # 写总 summary CSV
    batch_summary_csv = batch_output_dir / "batch_summary.csv"

    with open(batch_summary_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "tracker_name",
            "tracker_type",
            "frames_total",
            "frames_with_tracks",
            "total_tracks_written",
            "unique_track_id_count",
            "elapsed_seconds",
            "error",
            "output_video",
            "output_csv"
        ])

        for s in all_summaries:
            writer.writerow([
                s.get("tracker_name", ""),
                s.get("tracker_type", ""),
                s.get("frames_total", ""),
                s.get("frames_with_tracks", ""),
                s.get("total_tracks_written", ""),
                s.get("unique_track_id_count", ""),
                s.get("elapsed_seconds", ""),
                s.get("error", ""),
                s.get("output_video", ""),
                s.get("output_csv", "")
            ])

    # 写总 summary JSON
    batch_summary_json = batch_output_dir / "batch_summary.json"

    with open(batch_summary_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("全部方法测试完成！")
    print(f"总结果目录：{batch_output_dir}")
    print(f"总CSV汇总：{batch_summary_csv}")
    print(f"总JSON汇总：{batch_summary_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()