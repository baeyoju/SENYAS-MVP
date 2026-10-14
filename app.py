import cv2
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np

from object_detector import detect_objects

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded."
        }), 400

    file = request.files["image"]

    # Convert uploaded image into an OpenCV frame
    image_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({
            "success": False,
            "error": "Invalid image."
        }), 400

    # Call your existing YOLO detector
    detections = detect_objects(frame)

    return jsonify({
        "success": True,
        "count": len(detections),
        "detections": detections
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)