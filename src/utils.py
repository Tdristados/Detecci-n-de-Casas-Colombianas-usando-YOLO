
"""
utils.py — utilidades para datasets YOLOv8-OBB (Roboflow)

CLI:
  python utils.py --check --dataset ../data
  python utils.py --fix-yaml --yaml ../data/data.yaml --dataset ../data
  python utils.py --visualize --image ../data/train/images/xxx.jpg --label ../data/train/labels/xxx.txt --names house
"""
import argparse, os, yaml, cv2

IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")

def _as_list(names_or_csv):
    if isinstance(names_or_csv, list):
        return names_or_csv
    if names_or_csv is None:
        return []
    if "," in names_or_csv:
        return [n.strip() for n in names_or_csv.split(",") if n.strip()]
    return [names_or_csv.strip()]

def check_dataset(dataset_root: str):
    dataset_root = os.path.abspath(dataset_root)
    problems = []
    for split in ("train", "valid"):
        img_dir = os.path.join(dataset_root, split, "images")
        lbl_dir = os.path.join(dataset_root, split, "labels")
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)] if os.path.isdir(img_dir) else []
        lbls = [f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")] if os.path.isdir(lbl_dir) else []
        img_b = {os.path.splitext(f)[0] for f in imgs}
        lbl_b = {os.path.splitext(f)[0] for f in lbls}
        missing_lbl = img_b - lbl_b
        missing_img = lbl_b - img_b
        if missing_lbl:
            problems.append(f"{split}: faltan labels para {len(missing_lbl)} imágenes (ej: {next(iter(missing_lbl))}.txt)")
        if missing_img:
            problems.append(f"{split}: faltan imágenes para {len(missing_img)} labels (ej: {next(iter(missing_img))}.jpg)")
    return problems

def fix_yaml(yaml_path: str, dataset_root: str):
    dataset_root = os.path.abspath(dataset_root)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = dataset_root
    data["train"] = "train/images"
    data["val"] = "valid/images"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return yaml_path

def visualize_one(image_path: str, label_path: str, class_names):
    import cv2, numpy as np
    class_names = _as_list(class_names)
    im = cv2.imread(image_path)
    if im is None:
        raise FileNotFoundError(f"No pude abrir {image_path}")
    H, W = im.shape[:2]
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        cid = int(float(parts[0]))
        xc, yc, w, h, angle_deg = map(float, parts[1:6])  # OBB formato Ultralytics
        cx, cy = xc*W, yc*H
        bw, bh = w*W, h*H
        rect = ((cx, cy), (bw, bh), angle_deg)
        box = cv2.boxPoints(rect).astype(int)
        cv2.polylines(im, [box], True, (0,255,0), 2)
        label = class_names[cid] if cid < len(class_names) else str(cid)
        cv2.putText(im, label, (box[0][0], max(0, box[0][1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

    out = os.path.splitext(image_path)[0] + "_viz.jpg"
    cv2.imwrite(out, im)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fix-yaml", action="store_true")
    ap.add_argument("--dataset", type=str, default="../data")
    ap.add_argument("--yaml", type=str, default="../data/data.yaml")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--image", type=str)
    ap.add_argument("--label", type=str)
    ap.add_argument("--names", type=str, default="house")
    args = ap.parse_args()

    if args.check:
        probs = check_dataset(args.dataset)
        if probs:
            print("Problemas:")
            for p in probs:
                print(" -", p)
        else:
            print("Dataset OK ✅")

    if args.fix_yaml:
        out = fix_yaml(args.yaml, args.dataset)
        print("YAML actualizado:", out)

    if args.visualize:
        if not args.image or not args.label:
            raise SystemExit("--visualize requiere --image y --label")
        out = visualize_one(args.image, args.label, args.names)
        print("Guardado:", out)

if __name__ == "__main__":
    main()
