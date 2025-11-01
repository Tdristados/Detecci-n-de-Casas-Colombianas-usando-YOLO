
"""
train_yolo.py — YOLOv8-OBB (oriented bounding boxes)

Guarda runs y pesos en la carpeta ../model

Uso:
  python train_yolo.py --data ../data --yaml ../data/data.yaml --epochs 50 --device 0
"""
import argparse, os, shutil, time
from ultralytics import YOLO
from utils import fix_yaml

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="../data")
    ap.add_argument("--yaml", type=str, default="../data/data.yaml")
    ap.add_argument("--model", type=str, default="yolov8n-obb.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", type=str, default="../model")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--resume", type=str, default="")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.project, exist_ok=True)
    exp_name = args.name or time.strftime("%Y%m%d-%H%M%S")

    yaml_path = os.path.abspath(args.yaml)
    fix_yaml(yaml_path, args.data)

    model = YOLO(args.resume or args.model)
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=exp_name,
        pretrained=not bool(args.resume),
        verbose=True,
    )
    run_dir = os.path.join(args.project, exp_name)
    best_src = os.path.join(run_dir, "weights", "best.pt")
    best_dst = os.path.join(args.project, "best.pt")
    if os.path.isfile(best_src):
        shutil.copy2(best_src, best_dst)
        print("Copié mejores pesos en:", best_dst)
    print("Carpeta del experimento:", run_dir)

if __name__ == "__main__":
    main()
