from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_manifest(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class YOLOTileDataset(Dataset):
    def __init__(self, records: Sequence[dict]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec["tile_path"]).convert("RGB")
        img_np = np.array(img)
        image = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(rec["boxes"], dtype=torch.float32) if rec["boxes"] else torch.zeros((0, 4), dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": rec["image_id"],
            "orig_size": (rec["height"], rec["width"]),
        }
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


@dataclass
class DetectionMetrics:
    map50: float
    map50_95: float
    precision: float
    recall: float
    f1: float
    threshold: float

    def to_dict(self):
        return {
            "mAP@0.5": self.map50,
            "mAP@0.5:0.95": self.map50_95,
            "precision@selected_conf_iou0.5": self.precision,
            "recall@selected_conf_iou0.5": self.recall,
            "f1@selected_conf_iou0.5": self.f1,
            "selected_conf_threshold": self.threshold,
        }


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def ap_for_threshold(gts: List[dict], preds: List[dict], iou_thr: float) -> float:
    gt_by_image: Dict[str, np.ndarray] = {g["image_id"]: np.array(g["boxes"], dtype=np.float32) for g in gts}
    matched: Dict[str, np.ndarray] = {img_id: np.zeros((len(boxes),), dtype=bool) for img_id, boxes in gt_by_image.items()}
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0:
        return float("nan")

    pred_rows = []
    for p in preds:
        img_id = p["image_id"]
        boxes = np.array(p["boxes"], dtype=np.float32)
        scores = np.array(p["scores"], dtype=np.float32)
        for box, score in zip(boxes, scores):
            pred_rows.append((float(score), img_id, box))
    pred_rows.sort(key=lambda x: x[0], reverse=True)

    tp = np.zeros((len(pred_rows),), dtype=np.float32)
    fp = np.zeros((len(pred_rows),), dtype=np.float32)
    for i, (_, img_id, box) in enumerate(pred_rows):
        gt_boxes = gt_by_image.get(img_id, np.zeros((0, 4), dtype=np.float32))
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue
        ious = box_iou_np(np.array([box], dtype=np.float32), gt_boxes)[0]
        best_idx = int(np.argmax(ious)) if len(ious) else -1
        best_iou = float(ious[best_idx]) if len(ious) else 0.0
        if best_iou >= iou_thr and not matched[img_id][best_idx]:
            tp[i] = 1
            matched[img_id][best_idx] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def precision_recall_f1(gts: List[dict], preds: List[dict], conf_thr: float = 0.25, iou_thr: float = 0.5):
    gt_by_image: Dict[str, np.ndarray] = {g["image_id"]: np.array(g["boxes"], dtype=np.float32) for g in gts}
    total_gt = sum(len(v) for v in gt_by_image.values())
    tp = 0
    fp = 0
    for p in preds:
        img_id = p["image_id"]
        boxes = np.array(p["boxes"], dtype=np.float32)
        scores = np.array(p["scores"], dtype=np.float32)
        keep = scores >= conf_thr
        boxes = boxes[keep]
        scores = scores[keep]
        gt_boxes = gt_by_image.get(img_id, np.zeros((0, 4), dtype=np.float32))
        matched = np.zeros((len(gt_boxes),), dtype=bool)
        order = np.argsort(scores)[::-1]
        for j in order:
            box = boxes[j]
            if len(gt_boxes) == 0:
                fp += 1
                continue
            ious = box_iou_np(np.array([box], dtype=np.float32), gt_boxes)[0]
            best_idx = int(np.argmax(ious)) if len(ious) else -1
            best_iou = float(ious[best_idx]) if len(ious) else 0.0
            if best_iou >= iou_thr and not matched[best_idx]:
                tp += 1
                matched[best_idx] = True
            else:
                fp += 1
    fn = total_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def select_conf_threshold(gts: List[dict], preds: List[dict], iou_thr: float = 0.5) -> float:
    candidates = np.linspace(0.01, 0.50, 25)
    best_thr = 0.25
    best_f1 = -1.0
    best_p = -1.0
    best_r = -1.0
    for thr in candidates:
        p, r, f1 = precision_recall_f1(gts, preds, conf_thr=float(thr), iou_thr=iou_thr)
        if (f1, p, r) > (best_f1, best_p, best_r):
            best_thr = float(thr)
            best_f1, best_p, best_r = f1, p, r
    return best_thr


def evaluate_detections(gts: List[dict], preds: List[dict], conf_thr: float | None = None) -> DetectionMetrics:
    thresholds = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    aps = [ap_for_threshold(gts, preds, t) for t in thresholds]
    if conf_thr is None:
        conf_thr = select_conf_threshold(gts, preds, iou_thr=0.5)
    p, r, f1 = precision_recall_f1(gts, preds, conf_thr=conf_thr, iou_thr=0.5)
    return DetectionMetrics(map50=aps[0], map50_95=float(np.mean(aps)), precision=p, recall=r, f1=f1, threshold=float(conf_thr))


def get_ground_truth(records: Sequence[dict]) -> List[dict]:
    return [{"image_id": r["image_id"], "boxes": r["boxes"]} for r in records]


def benchmark_inference(images: Sequence[Path], infer_fn: Callable[[Path], dict], warmup: int = 10, runs: int = 50) -> dict:
    images = list(images)
    if not images:
        return {"latency_ms": math.nan, "fps": math.nan}
    warmup_imgs = images[: min(warmup, len(images))]
    eval_imgs = images[min(warmup, len(images)) : min(warmup + runs, len(images))]
    if not eval_imgs:
        eval_imgs = images[: min(runs, len(images))]

    for img in warmup_imgs:
        _ = infer_fn(img)

    start = time.perf_counter()
    for img in eval_imgs:
        _ = infer_fn(img)
    elapsed = time.perf_counter() - start
    latency = elapsed / max(len(eval_imgs), 1) * 1000.0
    fps = len(eval_imgs) / max(elapsed, 1e-9)
    return {"latency_ms": latency, "fps": fps, "timed_images": len(eval_imgs)}


def model_num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
