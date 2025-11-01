# Detección de Casas – YOLOv8 (OBB)

Proyecto mínimo para reentrenar e inferir detección de **casas** usando **YOLOv8‑OBB** con dataset de casas colombianas extraídas de Google Maps, dataset propio exportado desde **Roboflow**.

## Estructura para el funcionamiento del modelo
```
.
├─ data/                 # dataset Roboflow (train/valid + data.yaml, no subido)
├─ model/                # best.pt (mejor peso)
└─ src/
   ├─ train_yolo.py
   ├─ inferencia.py      # UNA imagen → JSON
   ├─ api.py             # FastAPI /predict (imagen → JSON)
   └─ utils.py
```

## Requisitos
```bash
python -m venv venv
# Win: .\venv\Scripts\activate  |  Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Preparar dataset
Coloca el export de Roboflow en `data/`:
```
data/
  train/{images,labels}
  valid/{images,labels}
  data.yaml
```
Opcional, chequear que todo esté bien:
```bash
cd src
python utils.py --check --dataset ../data
python utils.py --fix-yaml --yaml ../data/data.yaml --dataset ../data
```

## Entrenamiento
```bash
python src/train_yolo.py --data data --yaml data/data.yaml \
                         --model yolov8n-obb.pt --epochs 50 --batch 16 --imgsz 640 --device 0
# Mejores pesos: model/<timestamp>/weights/best.pt  (copiar a models/)
```

## Inferencia (CLI, una imagen → JSON)
```bash
python src/inferencia.py --weights models/house-yolo.pt \
                         --image data/valid/images/ejemplo.jpg --conf 0.25
# Salida:
# [ {"class":"house","score":0.92,"bbox":[x1,y1,x2,y2]}, ... ]
```

## API (FastAPI)
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
# MODEL_PATH=models/house-yolo.pt CONF=0.25 uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Probar:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@data/valid/images/ejemplo.jpg"
```

## Resultados (ejemplo; reemplaza con los tuyos)
- mAP50-95 (OBB): 0.42 | mAP50: 0.71 | Precision/Recall: 0.78/0.63
- Ejemplos: `docs/examples/*.jpg` (predicciones vs. GT)

## Limitaciones y próximos pasos
- Mejorar el entrenamiento con más datos/diversidad y modelos mayores (`yolov8s/m-obb.pt`).
- Considerar exportar OBB completo considerando train, valid, test.

