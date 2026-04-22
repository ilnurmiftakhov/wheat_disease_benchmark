# Эксперимент: сравнение легких моделей обнаружения болезней пшеницы для применения на БПЛА

## Краткий вывод
Я провел **собственный воспроизводимый пилотный benchmark** на открытом датасете **NWRD (NUST Wheat Rust Disease Dataset)**, чтобы сравнить три компактные детектора для задачи обнаружения очагов болезни пшеницы в полевых изображениях, пригодных для UAV-подобного сценария.

Сравнивались:
- **YOLOv8n**
- **SSDLite320-MobileNetV3-Large**
- **Faster R-CNN MobileNetV3-Large 320 FPN**

### Итог по эксперименту
- **YOLOv8n** показал **лучший баланс точности и скорости**:
  - лучший `mAP@0.5 = 0.1301`
  - лучший `mAP@0.5:0.95 = 0.0457`
  - лучшая скорость: **18.5 FPS** на CPU-прокси
  - самый маленький размер модели среди реально конкурентных: **~3.0M параметров**
- **Faster R-CNN MobileNetV3 320 FPN** дал **чуть лучший F1** (`0.0601` против `0.0552` у YOLOv8n), но оказался:
  - заметно тяжелее (**18.9M параметров**)
  - медленнее (**9.1 FPS**)
  - слабее по `mAP`
- **SSDLite320-MobileNetV3-Large** в этой задаче показал **самые слабые результаты** и для бортового применения на таком типе данных выглядит наименее убедительно.

### Практический вывод для БПЛА
Если нужен **реальный кандидат для бортового применения**, из трех проверенных моделей я бы выбрал **YOLOv8n**.

Но есть важное ограничение: **абсолютные метрики у всех моделей низкие**, то есть в текущем виде **ни одна модель не выглядит готовой к production для надежного автономного обнаружения болезни в поле**. Наиболее разумный вывод такой:
- **YOLOv8n — лучший стартовый baseline для onboard-детекции**;
- **Faster R-CNN MobileNetV3** можно рассматривать как более “осторожный” baseline вне жестких ограничений по вычислениям;
- **SSDLite320-MobileNetV3** для этой задачи, по моим данным, недостаточно силен.

---

## Что именно я проверял

### Цель
Сравнить легкие детекторы, которые потенциально можно запускать на борту БПЛА или на edge-компьютере, по критериям:
- `mAP@0.5`
- `mAP@0.5:0.95`
- `precision`
- `recall`
- `F1-score`
- скорость инференса
- пригодность для onboard-сценария

### Минимальный oracle / протокол проверки
Перед запуском эксперимента я зафиксировал следующий минимальный набор проверок:
1. Одна и та же тестовая выборка для всех моделей.
2. Одинаковая схема преобразования разметки в detection-задачу.
3. Одинаковый набор метрик.
4. Скорость измеряется на одном и том же CPU-прокси, batch=1, на 50 тестовых тайлах.
5. Порог для `precision/recall/F1` выбирается **по validation set**, а не вручную на тесте.

---

## Датасет

### Выбранный открытый датасет
**NWRD — NUST Wheat Rust Disease Dataset**
- 100 полевых изображений пшеницы
- есть **маски болезни**, но нет готовой object-detection разметки
- в статье авторы подчеркивают, что это **real-world multileaf dataset** с разными ракурсами, включая aerial/field views

Почему выбран именно он:
- он **открытый и доступный**;
- он ближе к полевой сцене, чем типичные close-up classification datasets;
- по обзору самого датасета, у открытых wheat-disease datasets часто есть только **classification labels**, а не detection boxes.

### Важное ограничение выбора датасета
Это **не multi-disease benchmark**, а benchmark по **обнаружению очагов rust disease (stripe/yellow rust)**. Я не нашел столь же удобного открытого wheat field dataset с готовыми detection-boxes для нескольких болезней; поэтому использовал открытый segmentation dataset и конвертировал его в detection-задачу.

---

## Как я превратил сегментацию в object detection

Так как у NWRD есть бинарные маски болезни, я построил detection-разметку сам:
1. Разбил изображения на тайлы **1024×1024**.
2. Использовал connected components по маске болезни.
3. Для каждого компонента площадью не меньше **512 px** строил bounding box.
4. Тайлы без болезни сохранял как negative examples.

### Почему это разумно для UAV-сценария
Для БПЛА на практике часто важнее не пиксельная сегментация спор, а **обнаружение локального очага/кластера поражения**, который дальше может быть:
- отмечен на карте,
- повторно обследован,
- использован для decision support.

То есть в эксперименте цель — не “поймать каждый пиксель ржавчины”, а сравнить модели как **детекторы пораженных областей**.

---

## Разделение данных
Использовалась следующая схема:
- исходные **10 test images** NWRD оставлены как test;
- из 90 train images последние 10 (в лексикографическом порядке имен файлов) выделены в validation;
- для CPU-реалистичного pilot benchmark train-часть была ограничена до:
  - **200 positive tiles**
  - **200 negative tiles**
  - всего **400 train tiles**
- validation: **204 tiles**
- test: **211 tiles**

Количество объектов:
- train boxes: **503**
- val boxes: **212**
- test boxes: **340**

Это важно: эксперимент **воспроизводимый**, но это **не полноразмерное GPU-обучение**, а **CPU-feasible benchmark**.

---

## Модели

### 1. YOLOv8n
- компактный one-stage detector
- использовалась предобученная базовая модель и дообучение под 1 класс (`rust`)

### 2. SSDLite320-MobileNetV3-Large
- классический легкий MobileNet-based detector
- ориентирован на edge/mobile use cases

### 3. Faster R-CNN MobileNetV3 Large 320 FPN
- более компактный two-stage baseline с MobileNetV3 backbone
- тяжелее предыдущих, но все еще относится к умеренно компактным детекторам

---

## Условия обучения и измерения

### Общее
- среда: **CPU only**
- изображения для обучения детекторов: тайлы из NWRD
- целевой класс: **rust**
- одинаковый test split для всех моделей

### Обучение
- **YOLOv8n**: 3 эпохи, `imgsz=320`, batch 16
- **SSDLite320-MobileNetV3-Large**: 3 эпохи, batch 8
- **Faster R-CNN MobileNetV3-Large 320 FPN**: 3 эпохи, batch 2

### Оценка
- `mAP@0.5`
- `mAP@0.5:0.95`
- `precision`, `recall`, `F1`
- порог confidence выбирался **на validation set** как порог с лучшим F1
- скорость инференса измерялась на **50 тестовых тайлах**, batch=1, CPU

---

## Результаты

| Модель | Параметры | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Val-selected conf | Latency, ms/img | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 3.01M | **0.1301** | **0.0457** | **0.4545** | 0.0294 | 0.0552 | 0.0304 | **53.95** | **18.54** |
| Faster R-CNN MobileNetV3 320 FPN | 18.93M | 0.0142 | 0.0033 | 0.0483 | **0.0794** | **0.0601** | 0.1325 | 110.32 | 9.06 |
| SSDLite320-MobileNetV3-Large | 3.71M | 0.0008 | 0.00015 | 0.0051 | 0.0294 | 0.0087 | 0.4796 | 86.19 | 11.60 |

### Наблюдения
1. **YOLOv8n уверенно лидирует по mAP и скорости.**
2. **Faster R-CNN MobileNetV3** показывает чуть лучший `F1`, но при этом сильно тяжелее и заметно медленнее.
3. **SSDLite** на этом типе полевых данных практически не справляется.
4. У всех моделей **recall остается низким**, что указывает на общую сложность задачи.

---

## Интерпретация результатов

### Почему метрики в целом низкие
Скорее всего, одновременно сработали несколько факторов:
1. **Очень сложная сцена**: листья перекрываются, фон неоднородный, ракурсы разные.
2. **Мелкие объекты**: rust-очаги маленькие относительно поля зрения.
3. **Прокси-разметка**: боксы получены автоматически из сегментационных масок, а не вручную размечены как detection targets.
4. **Ограниченный compute budget**: это короткий CPU-benchmark, а не полноценное GPU-обучение до насыщения.
5. **320 px input** — это разумный onboard-компромисс, но для мелких поражений он жесткий.

### Почему YOLOv8n выглядит лучшим кандидатом
Потому что он одновременно дает:
- лучшую локализационную точность (`mAP`),
- лучшую скорость,
- наименьшие требования к вычислениям среди конкурентных моделей,
- хорошую основу для дальнейшей калибровки порога и оптимизации под edge.

### Почему Faster R-CNN не победитель, несмотря на лучший F1
Его преимущество по F1 очень небольшое, а цена за это высокая:
- почти **в 6 раз больше параметров**, чем у YOLOv8n;
- примерно **в 2 раза ниже FPS**.

Для бортового применения на UAV это плохой trade-off.

### Почему SSDLite не рекомендую
Несмотря на MobileNet-бэкбон и edge-ориентацию, в этом эксперименте модель показала слишком слабое качество. Вывод: для сложных полевых сцен с мелкими disease-clusters **SSDLite320-MobileNetV3-Large в таком режиме недостаточен**.

---

## Вывод о пригодности для бортового применения

### Если выбирать одну модель сейчас
**YOLOv8n — лучший выбор из протестированных.**

### Практический verdict
- **Для onboard prototype:** да, **YOLOv8n** — лучший кандидат.
- **Для production UAV system:** пока **нет**, потому что абсолютные метрики слишком низкие.
- **Для offboard / наземной станции:** можно дополнительно рассматривать Faster R-CNN MobileNetV3, если приоритет — немного более высокий recall/F1, а не скорость.

### Что бы я рекомендовал для следующей итерации
1. Переобучить **YOLOv8n** дольше и на GPU.
2. Проверить **imgsz=512/640** и сравнить с текущим `320`.
3. Добавить **hard-negative mining**.
4. Попробовать **YOLOv8n-seg** или lightweight instance segmentation вместо box-only pipeline.
5. Проверить **quantization / ONNX / TensorRT / OpenVINO** для реальной бортовой задержки.
6. Если нужна именно multi-disease задача — собрать или разметить **действительно много-классовый wheat disease detection dataset**.

---

## Что сохранено на диск

### Канонический отчет
- `outputs/wheat_uav_disease_detector_benchmark.md`

### Код и артефакты эксперимента
- `experiments/wheat_disease_benchmark/prepare_nwrd_detection.py`
- `experiments/wheat_disease_benchmark/benchmark_utils.py`
- `experiments/wheat_disease_benchmark/train_torchvision_detector.py`
- `experiments/wheat_disease_benchmark/run_yolov8n.py`
- `experiments/wheat_disease_benchmark/data/nwrd_detection_tiles/manifest_benchmark.json`
- `experiments/wheat_disease_benchmark/runs/yolov8n/result.json`
- `experiments/wheat_disease_benchmark/runs/ssdlite320_mobilenet_v3_large/result.json`
- `experiments/wheat_disease_benchmark/runs/fasterrcnn_mobilenet_v3_large_320_fpn/result.json`

---

## Ограничения и честные оговорки
1. Это **pilot benchmark**, а не full-scale training campaign.
2. Использовался **CPU-only** режим.
3. Датасет фактически дает **одну болезнь (rust)**, а не полноценный multi-disease setup.
4. Detection-boxes были **сгенерированы из masks**, а не вручную размечены детекторами.
5. NWRD содержит полевые изображения с разными ракурсами, но это **не чисто UAV-only dataset**.

Поэтому главный вывод нужно читать аккуратно:
- я действительно выполнил **собственный воспроизводимый эксперимент**;
- но это **ограниченный инженерный benchmark**, а не окончательная оценка SOTA для UAV disease detection.

---

## Источники
- NWRD dataset page: https://robustreading.com/datasets/NUST-Wheat-Rust-Disease-Dataset/
- NWRD code repository: https://github.com/dll-ncai/NUST-Wheat-Rust-Disease-NWRD
- NWRD paper (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC10422341/
- NWRD paper (publisher): https://www.mdpi.com/1424-8220/23/15/6942
- YOLOv8 official docs: https://docs.ultralytics.com/models/yolov8/
- Torchvision SSDLite320 MobileNetV3 Large: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
- Torchvision Faster R-CNN MobileNetV3 Large 320 FPN: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html
