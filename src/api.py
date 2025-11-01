
"""
api.py — FastAPI para servir un modelo YOLOv8 (detect u OBB)

POST /predict
  - Body: multipart/form-data con campo 'file' (imagen)
  - Respuesta: JSON con lista de detecciones
    [{"class":"house","score":0.92,"bbox":[x1,y1,x2,y2]}, ...]

Para ejecutar:
  uvicorn api:app --host 0.0.0.0 --port 8000
Variables opcionales:
  MODEL_PATH (default: ./model/best.pt)
  CONF (default: 0.25)
"""
import os
from typing import List, Dict, Any

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "../model/best.pt")
CONF = float(os.getenv("CONF", "0.25"))

_model = None
def get_model():
    global _model
    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}. "
                                    "Configura MODEL_PATH o coloca tus pesos en ../model/best.pt")
        _model = YOLO(MODEL_PATH)
    return _model

app = FastAPI(title="Casas YOLOv8 API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"status": "ok", "message": "YOLOv8(OBB) Casas API", "model_path": MODEL_PATH}

def _read_image_to_bgr(data: bytes):
    file_bytes = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")
    return img

def _bbox_from_poly(poly: np.ndarray):
    if poly.ndim == 1 and poly.size == 8:
        xs = poly[0::2]; ys = poly[1::2]
    elif poly.ndim == 2 and poly.shape == (4, 2):
        xs = poly[:,0]; ys = poly[:,1]
    else:
        xs = poly[:,0] if poly.ndim == 2 else poly[0::2]
        ys = poly[:,1] if poly.ndim == 2 else poly[1::2]
    x1, y1 = float(np.min(xs)), float(np.min(ys))
    x2, y2 = float(np.max(xs)), float(np.max(ys))
    return [x1, y1, x2, y2]

def _postprocess(res) -> List[Dict[str, Any]]:
    out = []
    r = res[0]
    names = r.names if hasattr(r, "names") else {}
    # Preferir OBB si existe
    try:
        obb = getattr(r, "obb", None)
        if obb is not None and hasattr(obb, "xyxyxyxy"):
            polys = obb.xyxyxyxy.cpu().numpy()
            confs = obb.conf.cpu().numpy() if hasattr(obb, "conf") else None
            cls_ids = obb.cls.cpu().numpy().astype(int) if hasattr(obb, "cls") else None
            for i, poly in enumerate(polys):
                out.append({
                    "class": names.get(int(cls_ids[i]), str(int(cls_ids[i]))) if cls_ids is not None else "0",
                    "score": float(confs[i]) if confs is not None else 0.0,
                    "bbox": _bbox_from_poly(poly),
                })
            if out:
                return out
    except Exception:
        pass
    # Fallback a 'detect'
    try:
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = map(float, xyxy[i].tolist())
            out.append({
                "class": names.get(int(cls_ids[i]), str(int(cls_ids[i]))),
                "score": float(conf[i]),
                "bbox": [x1, y1, x2, y2],
            })
        return out
    except Exception:
        return out

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = None):
    try:
        data = await file.read()
        img = _read_image_to_bgr(data)
        model = get_model()
        threshold = CONF if conf is None else float(conf)
        res = model.predict(source=img, conf=threshold, verbose=False)
        return _postprocess(res)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando la imagen: {e}")
