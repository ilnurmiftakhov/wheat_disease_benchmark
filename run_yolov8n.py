from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from benchmark_utils import (
    benchmark_inference,
    evaluate_detections,
    get_ground_truth,
    load_manifest,
    select_conf_threshold,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_records(model: YOLO, records, imgsz: int):
    preds = []
    for rec in records:
        results = model.predict(source=rec["tile_path"], imgsz=imgsz, conf=0.001, iou=0.7, device="cpu", verbose=False)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
        scores = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
        preds.append({"image_id": rec["image_id"], "boxes": boxes, "scores": scores})
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--data", default="data/nwrd_detection_tiles/dataset.yaml")
    parser.add_argument("--manifest", default="data/nwrd_detection_tiles/manifest.json")
    parser.add_argument("--outdir", default="runs/yolov8n")
    args = parser.parse_args()

    set_seed(42)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")
    train_res = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=0,
        project=str(outdir.parent),
        name=outdir.name,
        exist_ok=True,
        pretrained=True,
        seed=42,
        verbose=True,
        cache=False,
    )

    best_path = Path(train_res.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_path))
    manifest = load_manifest(args.manifest)
    val_records = manifest["splits"]["val"]
    test_records = manifest["splits"]["test"]
    val_preds = predict_records(best_model, val_records, args.imgsz)
    selected_thr = select_conf_threshold(get_ground_truth(val_records), val_preds, iou_thr=0.5)
    test_preds = predict_records(best_model, test_records, args.imgsz)
    test_metrics = evaluate_detections(get_ground_truth(test_records), test_preds, conf_thr=selected_thr)

    test_image_paths = [Path(r["tile_path"]) for r in test_records]

    def infer_fn(path: Path):
        res = best_model.predict(source=str(path), imgsz=args.imgsz, conf=0.25, iou=0.7, device="cpu", verbose=False)[0]
        return {"boxes": res.boxes.xyxy if res.boxes is not None else [], "scores": res.boxes.conf if res.boxes is not None else []}

    speed = benchmark_inference(test_image_paths, infer_fn, warmup=10, runs=50)
    num_params = sum(p.numel() for p in best_model.model.parameters())

    result = {
        "model": "yolov8n",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "num_params": int(num_params),
        "best_weights": str(best_path.as_posix()),
        "selected_conf_threshold_from_val": selected_thr,
        "test_metrics": test_metrics.to_dict(),
        "speed": speed,
    }
    with open(outdir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
