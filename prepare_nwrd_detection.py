from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

RAW_ROOT = Path("data/NWRD")
OUT_ROOT = Path("data/nwrd_detection_tiles")
SEED = 42
TILE_SIZE = 1024
STRIDE = 1024
MIN_COMPONENT_AREA = 512
NEG_POS_RATIO = 1.0  # keep as many negative train tiles as positive tiles


@dataclass
class TileRecord:
    split: str
    image_id: str
    orig_image: str
    tile_path: str
    label_path: str
    width: int
    height: int
    boxes: List[List[float]]
    positive: bool


def image_boxes_from_mask(mask: np.ndarray, min_area: int) -> List[List[int]]:
    """Convert binary mask to detection boxes via connected components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes: List[List[int]] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
    return boxes


def tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def to_yolo_line(box: List[int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    stems = sorted(p.stem for p in (RAW_ROOT / "train" / "images").glob("*.jpg"))
    val_stems = set(stems[-10:])
    split_map = {stem: ("val" if stem in val_stems else "train") for stem in stems}
    for p in (RAW_ROOT / "test" / "images").glob("*.jpg"):
        split_map[p.stem] = "test"

    for split in ["train", "val", "test"]:
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

    staged: dict[str, List[TileRecord]] = {"train_pos": [], "train_neg": [], "val": [], "test": []}

    for raw_split in ["train", "test"]:
        for img_path in sorted((RAW_ROOT / raw_split / "images").glob("*.jpg")):
            stem = img_path.stem
            split = split_map[stem]
            mask_path = RAW_ROOT / raw_split / "masks" / f"{stem}.png"
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 0).astype(np.uint8)

            height, width = mask.shape
            xs = tile_starts(width, TILE_SIZE, STRIDE)
            ys = tile_starts(height, TILE_SIZE, STRIDE)

            tile_idx = 0
            for y in ys:
                for x in xs:
                    tile_img = image[y : y + TILE_SIZE, x : x + TILE_SIZE]
                    tile_mask = mask[y : y + TILE_SIZE, x : x + TILE_SIZE]
                    boxes = image_boxes_from_mask(tile_mask, MIN_COMPONENT_AREA)
                    tile_name = f"{stem}_x{x}_y{y}_{tile_idx}"
                    tile_idx += 1

                    split_key = split
                    if split == "train":
                        split_key = "train_pos" if boxes else "train_neg"

                    staged[split_key].append(
                        TileRecord(
                            split=split,
                            image_id=tile_name,
                            orig_image=str(img_path),
                            tile_path=str((OUT_ROOT / split / "images" / f"{tile_name}.jpg").as_posix()),
                            label_path=str((OUT_ROOT / split / "labels" / f"{tile_name}.txt").as_posix()),
                            width=int(tile_img.shape[1]),
                            height=int(tile_img.shape[0]),
                            boxes=boxes,
                            positive=bool(boxes),
                        )
                    )

                    # save immediately for non-train splits
                    if split in {"val", "test"}:
                        Image.fromarray(tile_img).save(OUT_ROOT / split / "images" / f"{tile_name}.jpg", quality=95)
                        with open(OUT_ROOT / split / "labels" / f"{tile_name}.txt", "w", encoding="utf-8") as f:
                            if boxes:
                                f.write("\n".join(to_yolo_line(box, tile_img.shape[1], tile_img.shape[0]) for box in boxes))

    train_pos = staged["train_pos"]
    train_neg = staged["train_neg"]
    keep_neg = min(len(train_neg), int(round(len(train_pos) * NEG_POS_RATIO)))
    train_neg = random.sample(train_neg, keep_neg)

    for record in train_pos + train_neg:
        img = np.array(Image.open(record.orig_image).convert("RGB"))
        # parse x/y back from tile id
        parts = record.image_id.split("_")
        x = int(parts[1][1:])
        y = int(parts[2][1:])
        tile_img = img[y : y + TILE_SIZE, x : x + TILE_SIZE]
        Image.fromarray(tile_img).save(record.tile_path, quality=95)
        with open(record.label_path, "w", encoding="utf-8") as f:
            if record.boxes:
                f.write("\n".join(to_yolo_line(box, record.width, record.height) for box in record.boxes))

    manifest = {
        "config": {
            "seed": SEED,
            "tile_size": TILE_SIZE,
            "stride": STRIDE,
            "min_component_area": MIN_COMPONENT_AREA,
            "neg_pos_ratio": NEG_POS_RATIO,
            "val_source": "last 10 lexicographically sorted original train images",
        },
        "splits": {
            "train": [asdict(r) for r in train_pos + train_neg],
            "val": [asdict(r) for r in staged["val"]],
            "test": [asdict(r) for r in staged["test"]],
        },
    }
    with open(OUT_ROOT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    yaml_text = (
        f"path: {OUT_ROOT.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: rust\n"
    )
    (OUT_ROOT / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    summary = {
        "train_pos_tiles": len(train_pos),
        "train_neg_tiles": len(train_neg),
        "val_tiles": len(staged["val"]),
        "test_tiles": len(staged["test"]),
        "train_boxes": sum(len(r.boxes) for r in train_pos + train_neg),
        "val_boxes": sum(len(r.boxes) for r in staged["val"]),
        "test_boxes": sum(len(r.boxes) for r in staged["test"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
