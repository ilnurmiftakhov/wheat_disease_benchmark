"# Wheat Disease Benchmark

Репозиторий с воспроизводимым пилотным benchmark-проектом по сравнению легких детекторов болезней пшеницы на полевых изображениях для UAV/edge-сценария.

## Что это за проект

В проекте сравниваются три компактные модели object detection для задачи обнаружения очагов ржавчины пшеницы на датасете **NWRD (NUST Wheat Rust Disease Dataset)**:

- **YOLOv8n**
- **SSDLite320-MobileNetV3-Large**
- **Faster R-CNN MobileNetV3-Large 320 FPN**

Так как NWRD публикуется как segmentation-dataset, в проекте реализована конвертация масок болезни в detection-разметку:

- изображения режутся на тайлы `1024x1024`
- по бинарной маске строятся connected components
- для компонент площадью от `512 px` строятся bounding boxes
- тайлы без болезни сохраняются как negative examples

## Краткий результат benchmark

| Модель | Параметры | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Latency, ms/img | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 3.01M | **0.1301** | **0.0457** | **0.4545** | 0.0294 | 0.0552 | **53.95** | **18.54** |
| Faster R-CNN MobileNetV3 320 FPN | 18.93M | 0.0142 | 0.0033 | 0.0483 | **0.0794** | **0.0601** | 110.32 | 9.06 |
| SSDLite320-MobileNetV3-Large | 3.71M | 0.0008 | 0.00015 | 0.0051 | 0.0294 | 0.0087 | 86.19 | 11.60 |

### Практический вывод

Лучший общий баланс качества и скорости в этом pilot benchmark показал **YOLOv8n**.  
Но абсолютные метрики у всех моделей низкие, поэтому проект стоит рассматривать как **baseline и инженерный benchmark**, а не как production-ready решение.

## Структура репозитория

```text
.
├─ benchmark_utils.py                  # общие утилиты для метрик и speed benchmark
├─ prepare_nwrd_detection.py           # подготовка detection-датасета из NWRD masks
├─ run_yolov8n.py                      # обучение и оценка YOLOv8n
├─ train_torchvision_detector.py       # обучение и оценка SSDLite / Faster R-CNN
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

## Что делает каждый скрипт

### `prepare_nwrd_detection.py`
Готовит detection-версию датасета:
- читает исходные изображения и маски NWRD
- режет изображения на тайлы
- извлекает bounding boxes из connected components
- формирует train / val / test split
- сохраняет `manifest.json` и `dataset.yaml`

Запуск:

```bash
python prepare_nwrd_detection.py
```

### `run_yolov8n.py`
Обучает и оценивает YOLOv8n.

Пример:

```bash
python run_yolov8n.py \
  --epochs 3 \
  --imgsz 320 \
  --batch 16 \
  --data data/nwrd_detection_tiles/dataset.yaml \
  --manifest data/nwrd_detection_tiles/manifest.json \
  --outdir runs/yolov8n
```

### `train_torchvision_detector.py`
Обучает и оценивает детекторы из torchvision.

SSDLite:

```bash
python train_torchvision_detector.py \
  --model ssdlite320_mobilenet_v3_large \
  --epochs 3 \
  --batch-size 8 \
  --lr 1e-4 \
  --manifest data/nwrd_detection_tiles/manifest.json \
  --outdir runs
```

Faster R-CNN:

```bash
python train_torchvision_detector.py \
  --model fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 3 \
  --batch-size 2 \
  --lr 1e-4 \
  --manifest data/nwrd_detection_tiles/manifest.json \
  --outdir runs
```

## Метрики и протокол

Для всех моделей использовался общий минимальный протокол:

1. один и тот же `test split`
2. одинаковая схема конвертации segmentation → detection
3. одинаковый набор метрик
4. speed benchmark на CPU, `batch=1`, на 50 тестовых тайлах
5. confidence threshold выбирается по `validation set`, а не вручную на тесте

Считаются:
- `mAP@0.5`
- `mAP@0.5:0.95`
- `precision`
- `recall`
- `F1`
- latency / FPS

## Зависимости

Минимально нужны:

- Python 3.10+
- `torch`
- `torchvision`
- `ultralytics`
- `numpy`
- `Pillow`
- `opencv-python`

В репозитории есть готовый файл зависимостей:

```bash
pip install -r requirements.txt
```

Если нужен ручной вариант:

```bash
pip install torch torchvision ultralytics numpy pillow opencv-python
```

## Как получить NWRD

Источник данных:

- NWRD dataset page: https://robustreading.com/datasets/NUST-Wheat-Rust-Disease-Dataset/
- NWRD repository: https://github.com/dll-ncai/NUST-Wheat-Rust-Disease-NWRD

Скрипт `prepare_nwrd_detection.py` ожидает исходные данные в каталоге `data/NWRD` со структурой вида:

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

Пример:

```bash
mkdir -p data
# распакуйте архив так, чтобы внутри появился каталог data/NWRD
```

## Как подготовить detection-датасет

После того как исходный NWRD лежит в `data/NWRD`, выполните:

```bash
python prepare_nwrd_detection.py
```

Скрипт:
- нарежет исходные изображения на тайлы,
- извлечет боксы из disease masks,
- создаст `train / val / test` split,
- сохранит detection-артефакты в `data/nwrd_detection_tiles/`.

В результате появятся, в частности:
- `data/nwrd_detection_tiles/dataset.yaml`
- `data/nwrd_detection_tiles/manifest.json`
- `data/nwrd_detection_tiles/manifest_benchmark.json`

## Как воспроизвести benchmark

### 1. Подготовить данные

```bash
python prepare_nwrd_detection.py
```

### 2. Обучить и оценить YOLOv8n

```bash
python run_yolov8n.py \
  --epochs 3 \
  --imgsz 320 \
  --batch 16 \
  --data data/nwrd_detection_tiles/dataset.yaml \
  --manifest data/nwrd_detection_tiles/manifest.json \
  --outdir runs/yolov8n
```

### 3. Обучить и оценить SSDLite320-MobileNetV3-Large

```bash
python train_torchvision_detector.py \
  --model ssdlite320_mobilenet_v3_large \
  --epochs 3 \
  --batch-size 8 \
  --lr 1e-4 \
  --manifest data/nwrd_detection_tiles/manifest.json \
  --outdir runs
```

### 4. Обучить и оценить Faster R-CNN MobileNetV3 Large 320 FPN

```bash
python train_torchvision_detector.py \
  --model fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 3 \
  --batch-size 2 \
  --lr 1e-4 \
  --manifest data/nwrd_detection_tiles/manifest.json \
  --outdir runs
```

### 5. Сравнить результаты

После запуска итоговые метрики будут лежать в:

- `runs/yolov8n/result.json`
- `runs/ssdlite320_mobilenet_v3_large/result.json`
- `runs/fasterrcnn_mobilenet_v3_large_320_fpn/result.json`

## Что хранится в GitHub, а что нет

В GitHub-репозиторий включены:
- код
- manifests и YAML-конфиги
- `result.json`
- benchmark-отчет
- `requirements.txt`

В GitHub **не включены** тяжелые артефакты:
- raw dataset
- extracted tiles
- model weights
- большие training outputs

Это сделано специально, чтобы репозиторий оставался компактным и нормально клонировался.

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

## Reproducibility notes

Этот репозиторий задуман как **воспроизводимый pilot benchmark**, но важно учитывать ограничения:
- обучение проводилось в CPU-only режиме;
- detection-boxes получены автоматически из segmentation masks;
- абсолютные метрики низкие, поэтому это baseline, а не production-ready pipeline;
- для полного совпадения результатов важны версии библиотек и структура исходного датасета.

## Статус

Сейчас это **воспроизводимый pilot benchmark** для дальнейшего расширения:
- можно увеличить число эпох
- перейти на GPU
- сравнить `imgsz=512/640`
- добавить hard-negative mining
- проверить lightweight segmentation-подходы
" 
