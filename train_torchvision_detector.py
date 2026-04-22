from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, ssdlite320_mobilenet_v3_large
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from benchmark_utils import (
    YOLOTileDataset,
    benchmark_inference,
    collate_fn,
    evaluate_detections,
    get_ground_truth,
    load_manifest,
    model_num_params,
    select_conf_threshold,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(name: str):
    if name == "ssdlite320_mobilenet_v3_large":
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=MobileNet_V3_Large_Weights.DEFAULT, num_classes=2)
        return model
    if name == "fasterrcnn_mobilenet_v3_large_320_fpn":
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=MobileNet_V3_Large_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        return model
    raise ValueError(name)


def predict_records(model, records, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for rec in records:
            image = torch.from_numpy(np.array(Image.open(rec["tile_path"]).convert("RGB"))).permute(2, 0, 1).float() / 255.0
            outputs = model([image.to(device)])[0]
            preds.append(
                {
                    "image_id": rec["image_id"],
                    "boxes": outputs["boxes"].detach().cpu().numpy().tolist(),
                    "scores": outputs["scores"].detach().cpu().numpy().tolist(),
                }
            )
    return preds


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ssdlite320_mobilenet_v3_large", "fasterrcnn_mobilenet_v3_large_320_fpn"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--manifest", default="data/nwrd_detection_tiles/manifest.json")
    parser.add_argument("--outdir", default="runs")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cpu")
    manifest = load_manifest(args.manifest)
    train_records = manifest["splits"]["train"]
    val_records = manifest["splits"]["val"]
    test_records = manifest["splits"]["test"]

    train_ds = YOLOTileDataset(train_records)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    run_dir = Path(args.outdir) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_score = -1.0
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_preds = predict_records(model, val_records, device)
        val_metrics = evaluate_detections(get_ground_truth(val_records), val_preds)
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics.to_dict()}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics.map50_95 > best_score:
            best_score = val_metrics.map50_95
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    val_preds = predict_records(model, val_records, device)
    selected_thr = select_conf_threshold(get_ground_truth(val_records), val_preds, iou_thr=0.5)
    test_preds = predict_records(model, test_records, device)
    test_metrics = evaluate_detections(get_ground_truth(test_records), test_preds, conf_thr=selected_thr)

    test_image_paths = [Path(r["tile_path"]) for r in test_records]

    def infer_fn(path: Path):
        image = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = model([image.to(device)])[0]
        return {"boxes": out["boxes"], "scores": out["scores"]}

    speed = benchmark_inference(test_image_paths, infer_fn, warmup=10, runs=50)

    result = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_params": model_num_params(model),
        "history": history,
        "selected_conf_threshold_from_val": selected_thr,
        "test_metrics": test_metrics.to_dict(),
        "speed": speed,
        "weights": str(best_path.as_posix()),
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result["test_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(speed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
