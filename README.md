# Wheat Disease Benchmark

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/status-pilot%20benchmark-informational.svg)](#status)

Воспроизводимый пилотный benchmark-проект по сравнению легких детекторов болезней пшеницы на полевых изображениях в UAV/edge-сценарии.

## Кратко о проекте

В проекте сравниваются три компактные модели object detection для задачи **обнаружения очагов ржавчины пшеницы** на датасете **NWRD (NUST Wheat Rust Disease Dataset)**:

- **YOLOv8n**
- **SSDLite320-MobileNetV3-Large**
- **Faster R-CNN MobileNetV3-Large 320 FPN**

Так как NWRD опубликован как segmentation-dataset, в этом репозитории маски болезни переводятся в detection-разметку следующим образом:

- изображения режутся на тайлы `1024x1024`;
- из бинарных disease masks извлекаются connected components;
- компоненты площадью `>= 512 px` превращаются в bounding boxes;
- тайлы без болезни сохраняются как negative examples.

## Основной результат

| Модель | Параметры | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Latency, ms/img | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 3.01M | **0.1301** | **0.0457** | **0.4545** | 0.0294 | 0.0552 | **53.95** | **18.54** |
| Faster R-CNN MobileNetV3 320 FPN | 18.93M | 0.0142 | 0.0033 | 0.0483 | **0.0794** | **0.0601** | 110.32 | 9.06 |
| SSDLite320-MobileNetV3-Large | 3.71M | 0.0008 | 0.00015 | 0.0051 | 0.0294 | 0.0087 | 86.19 | 11.60 |

### Практический вывод

В этом pilot benchmark **YOLOv8n** показывает лучший общий баланс между качеством локализации и скоростью инференса.

Важная оговорка: абсолютные метрики у всех моделей низкие, поэтому репозиторий стоит рассматривать как **baseline и воспроизводимый инженерный benchmark**, а не как production-ready систему детекции болезни.

## Quickstart

### 1. Клонировать репозиторий

```bash
git clone git@github.com:ilnurmiftakhov/wheat_disease_benchmark.git
cd wheat_disease_benchmark
```

### 2. Создать окружение

Через `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Через `conda`:

```bash
conda env create -f environment.yml
conda activate wheat-disease-benchmark
```

### 3. Скачать NWRD

Источники:

- страница датасета: https://robustreading.com/datasets/NUST-Wheat-Rust-Disease-Dataset/
- репозиторий датасета: https://github.com/dll-ncai/NUST-Wheat-Rust-Disease-NWRD

Сырые данные должны лежать в `data/NWRD` со следующей структурой:

```text
data/NWRD/
├─ train/
│  ├─ images/
│  └─ masks/
└─ test/
   ├─ images/
   └─ masks/
```

Если вы скачали архив `NWRD.zip`, распакуйте его так, чтобы итоговый путь был именно `data/NWRD/...`.

### 4. Подготовить detection-датасет

```bash
python prepare_nwrd_detection.py
```

Скрипт:
- нарежет исходные изображения на тайлы;
- извлечет боксы из масок;
- создаст `train / val / test` split;
- сохранит detection-артефакты в `data/nwrd_detection_tiles/`.

### 5. Воспроизвести benchmark

#### YOLOv8n

```bash
python run_yolov8n.py \
  --epochs 3 \
  --imgsz 320 \
  --batch 16 \
  --data data/nwrd_detection_tiles/dataset.yaml \
  --manifest data/nwrd_detection_tiles/manifest.json \
  --outdir runs/yolov8n
```

#### SSDLite320-MobileNetV3-Large

```bash
python train_torchvision_detector.py \
  --model ssdlite320_mobilenet_v3_large \
  --epochs 3 \
  --batch-size 8 \
  --lr 1e-4 \
  --manifest data/nwrd_detection_tiles/manifest.json \
  --outdir runs
```

#### Faster R-CNN MobileNetV3 Large 320 FPN

```bash
python train_torchvision_detector.py \
  --model fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 3 \
  --batch-size 2 \
  --lr 1e-4 \
  --manifest data/nwrd_detection_tiles/manifest.json \
  --outdir runs
```

### 6. Посмотреть результаты

Итоговые метрики сохраняются в:

- `runs/yolov8n/result.json`
- `runs/ssdlite320_mobilenet_v3_large/result.json`
- `runs/fasterrcnn_mobilenet_v3_large_320_fpn/result.json`

## Содержимое репозитория

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

## Скрипты

### `prepare_nwrd_detection.py`
Строит detection-версию NWRD из segmentation masks.

Основные выходные файлы:
- `data/nwrd_detection_tiles/dataset.yaml`
- `data/nwrd_detection_tiles/manifest.json`
- `data/nwrd_detection_tiles/manifest_benchmark.json`

### `run_yolov8n.py`
Обучает и оценивает YOLOv8n на подготовленном detection benchmark.

### `train_torchvision_detector.py`
Обучает и оценивает детекторы из torchvision:
- `ssdlite320_mobilenet_v3_large`
- `fasterrcnn_mobilenet_v3_large_320_fpn`

### `benchmark_utils.py`
Общий код для:
- загрузки manifests;
- приближенного расчета AP в COCO-style;
- оценки precision/recall/F1;
- выбора confidence threshold по validation split;
- CPU speed benchmark.

## Протокол оценки

Все модели сравниваются по единому минимальному протоколу:

1. один и тот же `test split`;
2. одна и та же схема преобразования segmentation → detection;
3. один и тот же набор метрик;
4. speed benchmark на CPU с `batch=1` на 50 тестовых тайлах;
5. confidence threshold выбирается по validation split, а не подбирается вручную на тесте.

Используемые метрики:
- `mAP@0.5`
- `mAP@0.5:0.95`
- `precision`
- `recall`
- `F1`
- latency / FPS

## Что хранится в GitHub

В репозиторий включены:
- исходный код;
- manifests и YAML-конфиги;
- итоговые `result.json`;
- benchmark-отчет;
- файлы для настройки окружения.

В репозиторий **не включены**:
- raw dataset;
- extracted tiles;
- веса моделей;
- крупные training artifacts.

Это сделано специально, чтобы репозиторий оставался компактным и нормально клонировался.

## Заметки по воспроизводимости

Репозиторий задуман как **воспроизводимый pilot benchmark**, но есть важные ограничения:

- обучение выполнялось в CPU-only режиме;
- detection-boxes генерируются автоматически из segmentation masks;
- абсолютные метрики низкие, поэтому это baseline, а не deployment-ready pipeline;
- для точного воспроизведения важны версии библиотек и совпадение структуры исходного датасета.

## Подробный отчет

Полный текст benchmark-отчета:

- [`wheat_uav_disease_detector_benchmark.md`](./wheat_uav_disease_detector_benchmark.md)

## Источники

- NWRD dataset page: https://robustreading.com/datasets/NUST-Wheat-Rust-Disease-Dataset/
- NWRD code repository: https://github.com/dll-ncai/NUST-Wheat-Rust-Disease-NWRD
- NWRD paper (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC10422341/
- NWRD paper (publisher): https://www.mdpi.com/1424-8220/23/15/6942
- YOLOv8 docs: https://docs.ultralytics.com/models/yolov8/
- Torchvision SSDLite320 MobileNetV3 Large: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
- Torchvision Faster R-CNN MobileNetV3 Large 320 FPN: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html

## Лицензия

Проект распространяется под лицензией [MIT](./LICENSE).

## Статус

Текущий статус: **pilot benchmark**.

Естественные следующие шаги:
- более длинное обучение;
- GPU-запуски;
- сравнение `imgsz=512/640`;
- hard-negative mining;
- lightweight segmentation baselines.
