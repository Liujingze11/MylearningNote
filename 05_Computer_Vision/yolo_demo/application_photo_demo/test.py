from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("test.jpg", save=True)

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"类别ID: {cls_id}, 置信度: {conf:.2f}")