from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Make sure this model is available in the same directory or give the full path

# Pydantic model for request body
class ImageRequest(BaseModel):
    image: str  # base64-encoded image string

@app.get("/")
def root():
    return {"message": "YOLOv8 Object Detection API is running!"}

@app.post("/predict")
def predict(data: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(data.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)

        # Run YOLOv8 model
        results = model(image_np)[0]

        predictions = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            predictions.append({
                "xyxy": [x1, y1, x2, y2],
                "label": label,
                "confidence": confidence
            })

        return {"success": True, "predictions": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
