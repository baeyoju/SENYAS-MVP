from ultralytics import YOLO

# Pretrained YOLO model
model = YOLO("yolo26n.pt")


def detect_objects(frame):
    results = model(frame, verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": result.names[class_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

    return detections