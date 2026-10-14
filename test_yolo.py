import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()