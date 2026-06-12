from ultralytics import YOLO

# 加载官方预训练模型
model = YOLO("yolo11n.pt")

# 对视频做检测，并保存结果
results = model.predict(
    source="demo.mp4",   # 你的视频路径
    save=True,           # 保存结果视频
    show=True
    conf=0.25            # 置信度阈值
)

print("检测完成")
