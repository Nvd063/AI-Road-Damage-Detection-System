
from pathlib import Path
from io import BytesIO
import base64
import os

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "model" / "best.pt"

app = FastAPI(
    title="Road Damage Detection API",
    version="1.0.0"
)

origins = [
    x.strip()
    for x in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

model = YOLO(str(WEIGHTS))

@app.get("/")
def home():
    return {
        "status": "online",
        "model": "YOLO11s",
        "classes": model.names
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WEBP allowed"
        )

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum image size is 10 MB"
        )

    try:
        image = Image.open(BytesIO(content))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    if image.width * image.height > 20_000_000:
        raise HTTPException(
            status_code=413,
            detail="Image dimensions too large"
        )

    result = model.predict(
        source=np.array(image),
        imgsz=640,
        conf=0.25,
        verbose=False
    )[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        detections.append({
            "class_id": class_id,
            "class_name": model.names[class_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [
                round(float(v), 2)
                for v in box.xyxy[0].tolist()
            ]
        })

    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(
        annotated, cv2.COLOR_BGR2RGB
    )

    output = BytesIO()
    Image.fromarray(annotated_rgb).save(
        output, format="JPEG", quality=85
    )

    encoded = base64.b64encode(
        output.getvalue()
    ).decode("utf-8")

    return {
        "success": True,
        "filename": file.filename,
        "total_detections": len(detections),
        "detections": detections,
        "annotated_image": (
            "data:image/jpeg;base64," + encoded
        )
    }
