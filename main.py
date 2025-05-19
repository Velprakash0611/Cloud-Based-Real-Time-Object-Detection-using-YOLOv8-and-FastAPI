from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = FastAPI()

# Use yolov8n.pt (nano model) for faster inference
MODEL_PATH = "yolov8n.pt"
if not os.path.exists(MODEL_PATH):
    from ultralytics.utils.downloads import attempt_download_asset
    attempt_download_asset(MODEL_PATH)

model = YOLO(MODEL_PATH)

class ImageRequest(BaseModel):
    image: str  # base64 encoded image string

@app.get("/")
def root():
    return {"message": "YOLOv8 FastAPI server is running!"}

@app.post("/predict")
def predict(data: ImageRequest):
    try:
        # Step 1: Decode base64 image safely
        try:
            img_bytes = base64.b64decode(data.image)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as decode_error:
            return {"success": False, "error": f"Image decoding failed: {str(decode_error)}"}

        # Step 2: Validate decoded image
        if img is None or not isinstance(img, np.ndarray):
            return {"success": False, "error": "Decoded image is invalid or None."}

        # Step 3: Resize for faster inference
        img = cv2.resize(img, (640, 480))

        # Step 4: Run YOLO inference
        try:
            results = model(img)
        except Exception as model_error:
            return {"success": False, "error": f"YOLO model inference failed: {str(model_error)}"}

        # Step 5: Extract predictions
        predictions = []
        for box in results[0].boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results[0].names[int(box.cls[0])]
                confidence = float(box.conf[0])
                predictions.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "xyxy": [x1, y1, x2, y2]
                })

        return {"success": True, "predictions": predictions}

    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {str(e)}"}
