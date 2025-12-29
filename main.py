from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import base64
import io
from PIL import Image
import numpy as np
from typing import List
import os

app = FastAPI(title="AgriScan API", description="Tomato Leaf Disease Detection API")

# CORS - Frontend'in API'ye erisebilmesi icin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model yukle
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = None

def get_model():
    global model
    if model is None:
        model = YOLO(MODEL_PATH)
    return model

# Hastalik sinif isimleri (Turkce)
DISEASE_LABELS = {
    "Bacterial_spot": {"tr": "Bakteriyel Leke", "severity": "medium"},
    "Early_blight": {"tr": "Erken Yaniklik", "severity": "medium"},
    "Late_blight": {"tr": "Gec Yaniklik", "severity": "high"},
    "Leaf_Mold": {"tr": "Yaprak Kufu", "severity": "medium"},
    "Septoria_leaf_spot": {"tr": "Septoria Yaprak Lekesi", "severity": "medium"},
    "Spider_mites": {"tr": "Kirmizi Orumcek", "severity": "low"},
    "Target_Spot": {"tr": "Hedef Leke", "severity": "medium"},
    "Tomato_Yellow_Leaf_Curl_Virus": {"tr": "Sari Yaprak Kivircikligi", "severity": "high"},
    "Tomato_mosaic_virus": {"tr": "Mozaik Virusu", "severity": "high"},
    "healthy": {"tr": "Saglikli", "severity": "healthy"},
    "Healthy": {"tr": "Saglikli", "severity": "healthy"},
}

@app.get("/")
def root():
    return {"message": "AgriScan API - Tomato Leaf Disease Detection", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Goruntu analizi yap ve hastalik tespit et"""
    try:
        # Dosyayi oku
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # RGB'ye cevir
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Model ile tahmin yap
        yolo_model = get_model()
        results = yolo_model(image)

        predictions = []

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]

                    # Bbox koordinatlari
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    predictions.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2 - x1, y2 - y1]  # x, y, width, height
                    })

        return {
            "success": True,
            "predictions": predictions,
            "image_size": {"width": image.width, "height": image.height}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-base64")
async def analyze_base64(data: dict):
    """Base64 formatinda goruntu analizi"""
    try:
        base64_string = data.get("image", "")
        print(f"Received image data length: {len(base64_string)}")

        # data:image/jpeg;base64, kismini kaldir
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        # Base64'u decode et
        print("Decoding base64...")
        image_bytes = base64.b64decode(base64_string)
        print(f"Decoded bytes length: {len(image_bytes)}")

        image = Image.open(io.BytesIO(image_bytes))
        print(f"Image opened: {image.size}, mode: {image.mode}")

        # RGB'ye cevir
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Model ile tahmin yap
        print("Loading model...")
        yolo_model = get_model()
        print("Running inference...")
        results = yolo_model(image)
        print(f"Got {len(results)} results")

        predictions = []

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    predictions.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2 - x1, y2 - y1]
                    })

        print(f"Found {len(predictions)} predictions")

        # Modelin tum sinif isimlerini al
        all_classes = list(result.names.values()) if results and len(results) > 0 else []

        return {
            "success": True,
            "predictions": predictions,
            "image_size": {"width": image.width, "height": image.height},
            "all_classes": all_classes
        }

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
