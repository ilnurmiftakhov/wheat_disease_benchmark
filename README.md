# Wheat Disease Benchmark

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/status-pilot%20benchmark-informational.svg)](#status)

Reproducible pilot benchmark for comparing lightweight wheat disease detectors on field imagery in UAV/edge-style conditions.

## Summary

This project compares three compact object detectors for **wheat rust region detection** on **NWRD (NUST Wheat Rust Disease Dataset)**:

- **YOLOv8n**
- **SSDLite320-MobileNetV3-Large**
- **Faster R-CNN MobileNetV3-Large 320 FPN**

Because NWRD is published as a segmentation dataset, this repository converts disease masks into detection labels by:

- tiling images into `1024x1024` crops;
- extracting connected components from binary disease masks;
- turning components with area `>= 512 px` into bounding boxes;
- keeping disease-free tiles as negative examples.

## Main result

| Model | Params | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Latency, ms/img | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 3.01M | **0.1301** | **0.0457** | **0.4545** | 0.0294 | 0.0552 | **53.95** | **18.54** |
| Faster R-CNN MobileNetV3 320 FPN | 18.93M | 0.0142 | 0.0033 | 0.0483 | **0.0794** | **0.0601** | 110.32 | 9.06 |
| SSDLite320-MobileNetV3-Large | 3.71M | 0.0008 | 0.00015 | 0.0051 | 0.0294 | 0.0087 | 86.19 | 11.60 |

### Practical takeaway

In this pilot benchmark, **YOLOv8n** gives the best overall balance of localization quality and inference speed.

Important caveat: absolute metrics are low for all tested models, so this repository should be treated as a **baseline and reproducible engineering benchmark**, not as a production-ready disease detection system.

## Quickstart

### 1. Clone the repository

```bash
git clone git@github.com:ilnurmiftakhov/wheat_disease_benchmark.git
cd wheat_disease_benchmark
```

### 2. Create an environment

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate wheat-disease-benchmark
```

### 3. Download NWRD

Sources:

- Dataset page: https://robustreading.com/datasets/NUST-Wheat-Rust-Disease-Dataset/
- Code repository: https://github.com/dll-ncai/NUST-Wheat-Rust-Disease-NWRD

Place the raw dataset under `data/NWRD` so that the structure is:

```text
data/NWRD/
├─ train/
│  ├─ images/
│  └─ masks/
└─ test/
   ├─ images/
   └─ masks/
```

If you download `NWRD.zip`, extract it so the final path is exactly `data/NWRD/...`.

### 4. Prepare the detection dataset

```bash
python prepare_nwrd_detection.py
```

This will:
- tile raw images;
- extract boxes from masks;
- create `train / val / test` splits;
- write detection artifacts to `data/nwrd_detection_tiles/`.

### 5. Reproduce the benchmark

YOLOv8n:

```bash
python run_yolov8n.py \
  --epochs 3 \
  --imgsz 320 \
  --batch 16 \
  --data data/nwrd_detection_tiles/dataset.yaml \
  --manifest data/nwrd_detection_tiles/manifest.json \
  --outdir runs/yolov8n
```

SSDLite320-MobileNetV3-Large:

```bash
python train_torchvision_detector.py \
  --model ssdlite320_mobilenet_v3_large \
  --epochs 3 \
  --batch-size 8 \
  --lr 1e-4 \
  --manifest data/nwrd_detection_tiles/manifest.json \
  --outdir runs
```

Faster R-CNN MobileNetV3 Large 320 FPN:

```bash
python train_torchvision_detector.py \
  --model fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 3 \
  --batch-size 2 \
  --lr 1e-4 \
  --manifest data/nwrd_detection_tiles/manifest.json \
  --outdir runs
```

### 6. Inspect results

Final metrics are saved to:

- `runs/yolov8n/result.json`
- `runs/ssdlite320_mobilenet_v3_large/result.json`
- `runs/fasterrcnn_mobilenet_v3_large_320_fpn/result.json`

## Repository contents

```text
.
├─ benchmark_utils.py
├─ prepare_nwrd_detection.py
├─ run_yolov8n.py
├─ train_torchvision_detector.py
├─ requirements.txt
├─ environment.yml
├─ wheat_uav_disease_detector_benchmark.md
├─ data/
│  └─ nwrd_detection_tiles/
│     ├─ dataset.yaml
│     ├─ manifest.json
│     └─ manifest_benchmark.json
└─ runs/
   ├─ yolov8n/result.json
   ├─ ssdlite320_mobilenet_v3_large/result.json
   └─ fasterrcnn_mobilenet_v3_large_320_fpn/result.json
```

## Scripts

### `prepare_nwrd_detection.py`
Builds a detection version of NWRD from segmentation masks.

Outputs include:
- `data/nwrd_detection_tiles/dataset.yaml`
- `data/nwrd_detection_tiles/manifest.json`
- `data/nwrd_detection_tiles/manifest_benchmark.json`

### `run_yolov8n.py`
Trains and evaluates YOLOv8n on the prepared detection benchmark.

### `train_torchvision_detector.py`
Trains and evaluates torchvision-based detectors:
- `ssdlite320_mobilenet_v3_large`
- `fasterrcnn_mobilenet_v3_large_320_fpn`

### `benchmark_utils.py`
Shared code for:
- loading manifests;
- COCO-style AP approximation;
- precision/recall/F1 evaluation;
- confidence threshold selection on validation data;
- CPU speed benchmarking.

## Evaluation protocol

All models are compared under one shared minimal protocol:

1. same `test split`;
2. same segmentation-to-detection conversion scheme;
3. same metrics;
4. CPU speed benchmark with `batch=1` on 50 test tiles;
5. confidence threshold selected on the validation split rather than tuned manually on test.

Reported metrics:
- `mAP@0.5`
- `mAP@0.5:0.95`
- `precision`
- `recall`
- `F1`
- latency / FPS

## What is stored in GitHub

Included in the repository:
- source code;
- manifests and YAML configs;
- final `result.json` files;
- benchmark report;
- environment setup files.

Intentionally excluded from the repository:
- raw dataset;
- extracted image tiles;
- model weights;
- large training artifacts.

This keeps the repository small and makes cloning practical.

## Reproducibility notes

This repository is designed as a **reproducible pilot benchmark**, with some important limitations:

- training was run in CPU-only mode;
- detection boxes are generated automatically from segmentation masks;
- absolute metrics are low, so this is a baseline rather than a deployment-ready pipeline;
- exact reproduction depends on library versions and on matching the expected raw dataset layout.

## Detailed report

Full benchmark write-up:

- [`wheat_uav_disease_detector_benchmark.md`](./wheat_uav_disease_detector_benchmark.md)

## Sources

- NWRD dataset page: https://robustreading.com/datasets/NUST-Wheat-Rust-Disease-Dataset/
- NWRD code repository: https://github.com/dll-ncai/NUST-Wheat-Rust-Disease-NWRD
- NWRD paper (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC10422341/
- NWRD paper (publisher): https://www.mdpi.com/1424-8220/23/15/6942
- YOLOv8 docs: https://docs.ultralytics.com/models/yolov8/
- Torchvision SSDLite320 MobileNetV3 Large: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
- Torchvision Faster R-CNN MobileNetV3 Large 320 FPN: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html

## License

This project is released under the [MIT License](./LICENSE).

## Status

Current status: **pilot benchmark**.

Natural next steps:
- longer training;
- GPU runs;
- `imgsz=512/640` comparison;
- hard-negative mining;
- lightweight segmentation baselines.
