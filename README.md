<p align="center">
  <h1 align="center">🧩 Layout Ensembler</h1>
  <p align="center">
    <strong>Multi-model ensemble framework for document layout detection</strong>
  </p>
  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#extending">Extending</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/badge/torch-2.0+-red.svg" alt="PyTorch">
  </p>
</p>

---

## Motivation

No single document layout detection model excels at **all** region types. In practice:

| Model | Strength | Weakness |
|-------|----------|----------|
| **Surya** | General layout (paragraphs, tables, headers) | May miss signatures, Indic-script regions |
| **YOLO (DocLayNet)** | Dense layouts, tables | Noisy on handwriting/formulas |
| **IndicDLP** | Indic-script documents | Sparse on Latin-script content |
| **Signature detector** | Handwritten signatures | No general layout |

**Layout Ensembler** combines these heterogeneous detectors through a **class-aware, priority-based NMS** strategy that keeps the best detections from each model — producing layout annotations that are better than any individual model.

## Key Features

- **🔗 Class-aware ensemble** — Each model contributes only the region types it excels at, controlled by per-model class allowlists and confidence thresholds
- **🛡️ Protected classes** — Critical region types (e.g., signatures) are never suppressed by overlap, even if another model detected something nearby
- **🌳 Hierarchy construction** — Automatically builds parent-child containment trees (e.g., `caption` inside `figure`) using IoA-based nesting
- **⚡ Multi-GPU parallelism** — Surya replicas round-robin across GPUs; multiple YOLO checkpoints run concurrently on separate GPUs
- **🔌 Pluggable detectors** — Add any new model by implementing a simple `BaseModel` interface
- **📦 Production-ready** — Async I/O, batch processing, skip-existing, YAML-driven configuration
- **📊 Built-in evaluation** — Per-class precision/recall/F1 and COCO-style mAP metrics

## Architecture

```
                           ┌──────────────────┐
                           │  Document Pages   │
                           └────────┬─────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌──────────┐   ┌──────────┐   ┌──────────────┐
             │  Surya    │   │  YOLO    │   │  IndicDLP    │
             │ (GPU 0,1) │   │ (GPU 2)  │   │  (GPU 3)     │
             └─────┬─────┘   └─────┬────┘   └──────┬───────┘
                   │               │               │
                   ▼               ▼               ▼
             ┌─────────────────────────────────────────┐
             │      Class-Aware Priority NMS            │
             │                                          │
             │  1. Accept all base model detections     │
             │  2. Add supplementary if no overlap      │
             │  3. Protected classes bypass NMS         │
             │  4. Per-model confidence thresholds      │
             └──────────────────┬──────────────────────┘
                                │
                                ▼
             ┌──────────────────────────────────────────┐
             │          Hierarchy Builder                │
             │                                          │
             │  IoA-based parent → child nesting        │
             │  Greedy smallest-parent selection         │
             └──────────────────┬──────────────────────┘
                                │
                     ┌──────────┼──────────┐
                     ▼                     ▼
              ┌─────────────┐      ┌──────────────┐
              │ Layout JSON │      │  Page Crops   │
              └─────────────┘      └──────────────┘
```

## Installation

```bash
# Core package (no GPU dependencies)
pip install -e .

# With Surya detector
pip install -e ".[surya]"

# With YOLO detectors
pip install -e ".[yolo]"

# Everything (Surya + YOLO + entity detection)
pip install -e ".[all]"

# Development (adds pytest, ruff)
pip install -e ".[dev]"
```

## Quick Start

### 1. Single Image

```python
from layout_ensembler import Region, Ensembler, build_hierarchy
from layout_ensembler.config import load_cfg
from layout_ensembler.detectors.surya import SuryaModel
from layout_ensembler.detectors.yolo import YOLOModel

# Load config
cfg = load_cfg("configs/default.yaml")

# Run detectors
with SuryaModel(cfg) as surya:
    surya_regions = surya.predict([pil_image])[0]

with YOLOModel(cfg) as yolo:
    yolo_regions = yolo.predict([cv2_image])[0]

# Ensemble
ensembler = Ensembler(cfg.ensemble_config.model_dump())
merged = ensembler({
    "surya": surya_regions,
    "yolov11x_best": [r for r in yolo_regions if r.source == "yolov11x_best"],
    "IndicDLP": [r for r in yolo_regions if r.source == "IndicDLP"],
})

# Build hierarchy
layout = build_hierarchy(merged, ioa_threshold=0.9)
```

### 2. Batch Processing (CLI)

```bash
# Run layout detection on an entire document repository
layout-ensemble \
    --cfg configs/default.yaml \
    --root /data/documents \
    --batch_size 64 \
    --read_threads 8

# Crop detected regions
layout-crop \
    --root /data/documents \
    --pad_pct 0.02 \
    --workers 4
```

### 3. Example Script

```bash
python examples/run_single_image.py \
    --image /path/to/page.png \
    --cfg configs/default.yaml \
    --output_dir ./output \
    --visualize
```

## Configuration

All behavior is driven by a single YAML file. See [`configs/default.yaml`](configs/default.yaml) for a fully-commented example.

### Key Configuration Sections

| Section | Purpose |
|---------|---------|
| `surya_to_unified_map` | Map Surya labels → unified taxonomy |
| `yolo_to_unified_map` | Map YOLO labels → unified taxonomy |
| `device_map` | GPU assignment per model family |
| `ensemble_config` | Merge strategy, IoU thresholds, protected classes |
| `yolo_model_paths` | Paths to YOLO checkpoints |

### Ensemble Config Deep Dive

```yaml
ensemble_config:
  base_model_name: surya                    # All its detections accepted
  model_priority_order: [surya, yolo, IndicDLP]  # Processing order
  iou_threshold: 0.1                        # Overlap rejection threshold

  protected_classes_from_overlap_removal:   # Never suppressed
    - signature

  conf_thresholds_for_supplementary_models: # Per-model confidence floors
    IndicDLP: 0.4

  filtered_classes_to_keep:                 # Per-model class allowlists
    IndicDLP: [paragraph, table, figure]
```

## Project Structure

```
layout-ensembler/
├── layout_ensembler/           # Core package
│   ├── config.py               # Pydantic YAML configuration
│   ├── ensemble.py             # ★ Class-aware priority NMS
│   ├── hierarchy.py            # IoA-based containment tree
│   ├── cropper.py              # Region image extraction
│   ├── datamodels/
│   │   ├── regions.py          # Region, RegionType, TextLine
│   │   └── layout.py           # LayoutCrop, Detection, Line
│   ├── detectors/
│   │   ├── base_model.py       # Abstract detector interface
│   │   ├── surya.py            # Multi-GPU Surya adapter
│   │   └── yolo.py             # Multi-checkpoint YOLO adapter
│   ├── entity_detection/
│   │   ├── face_detector.py    # YOLO face detection
│   │   └── signature_detector.py
│   ├── line_detection/
│   │   └── surya_ocr.py        # Batch OCR pipeline
│   ├── evaluation/
│   │   └── metrics.py          # mAP, precision, recall, F1
│   ├── utils/
│   │   ├── geometry.py         # IoU, IoA, bbox utilities
│   │   ├── io_utils.py         # JSON serialization
│   │   └── viz.py              # Visualization with legends
│   └── drivers/
│       ├── run_layout_batch.py # Batch layout pipeline
│       └── run_crop_batch.py   # Batch region cropping
├── configs/
│   └── default.yaml            # Example configuration
├── tests/                      # Unit tests
├── examples/                   # Usage examples
├── docs/                       # Architecture documentation
├── pyproject.toml              # Package metadata
└── README.md
```

## Extending

### Adding a New Detector

1. Implement the `BaseModel` interface:

```python
from layout_ensembler.detectors import BaseModel
from layout_ensembler.datamodels.regions import Region, RegionType

class MyDetector(BaseModel):
    def load(self):
        self._model = load_my_model(self.device)

    def predict(self, batch_inputs):
        # Return list[list[Region]], one inner list per image
        ...

    def unload(self):
        self._model = None
```

2. Add label mappings to your config YAML
3. Include the model name in `ensemble_config.model_priority_order`

### Adding a New Region Type

Add the type to the `RegionType` enum in `layout_ensembler/datamodels/regions.py` and update the label maps in your config.

## Evaluation

```python
from layout_ensembler.evaluation.metrics import compute_precision_recall, compute_map

metrics = compute_precision_recall(gt_regions, pred_regions, iou_threshold=0.5)
map_scores = compute_map(gt_regions, pred_regions)
print(f"mAP@0.50: {map_scores['mAP@0.50']:.3f}")
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=layout_ensembler --cov-report=term-missing
```

## Acknowledgments

This project builds upon the following excellent open-source tools:

- [Surya](https://github.com/VikParuchuri/surya) — Document OCR and layout detection
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — Real-time object detection
- [IndicDLP](https://huggingface.co/IndicDLP) — Indic document layout parser
- [YOLOv8 Face Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)
- [Signature Detector](https://huggingface.co/tech4humans/yolov8s-signature-detector)

## License

MIT License — see [LICENSE](LICENSE) for details.
