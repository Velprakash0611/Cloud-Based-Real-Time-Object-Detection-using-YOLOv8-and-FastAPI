from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8s.pt")  # or yolov8n.pt for speed

class ImageRequest(BaseModel):
    image: str  # base64-encoded image

@app.get("/")
def root():
    return {"message": "YOLOv8 FastAPI server is running!"}

@app.post("/predict")
def predict(data: ImageRequest):
    try:
        # Decode the image
        img_bytes = base64.b64decode(data.image)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(img)
        predictions = []

        for box in results[0].boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results[0].names[int(box.cls[0])]
                confidence = float(box.conf[0])
                predictions.append({
                    "label": label,
                    "confidence": confidence,
                    "xyxy": [x1, y1, x2, y2]
                })

        return {"success": True, "predictions": predictions}
    
    except Exception as e:
        return {"success": False, "error": str(e)}
