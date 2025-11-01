
"""
inferencia.py — predicción con un modelo YOLOv8 (detect u OBB) sobre UNA imagen.
Devuelve por stdout un JSON con este formato:
[{"class": "house", "score": 0.92, "bbox": [x1,y1,x2,y2]}, ...]

Ejemplo:
  python inferencia.py --weights ../model/best.pt --image ../data/valid/images/ejemplo.jpg --conf 0.25 --device 0
"""
import argparse, os, json
import numpy as np
import cv2
from ultralytics import YOLO

def _bbox_from_poly(poly: np.ndarray):
    # poly: [8] or shape (4,2). Devuelve [x1,y1,x2,y2]
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

def _postprocess(results_obj):
    """
    Soporta modelos 'detect' y 'obb' de Ultralytics.
    Retorna lista de dicts con keys: class, score, bbox [x1,y1,x2,y2].
    """
    out = []
    r = results_obj[0]
    names = r.names if hasattr(r, "names") else {}
    # 1) Intentar OBB (polígonos)
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
    # 2) Cajas axis-aligned (detect)
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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="../model/best.pt", help="Ruta a pesos .pt")
    ap.add_argument("--image", type=str, required=True, help="Ruta a la imagen a evaluar")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="")
    return ap.parse_args()

def main():
    args = parse_args()
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"No existe la imagen: {args.image}")
    model = YOLO(args.weights)
    # Leer imagen como array BGR (más robusto si hay rutas con caracteres especiales)
    data = np.fromfile(args.image, dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError("No se pudo abrir/decodificar la imagen")
    res = model.predict(source=im, conf=args.conf, device=args.device, verbose=False)
    detections = _postprocess(res)
    print(json.dumps(detections, ensure_ascii=False))

if __name__ == "__main__":
    main()
