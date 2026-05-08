# Architecture

This document describes the design philosophy and technical architecture of
Layout Ensembler.

## Problem Statement

No single document layout detection model excels across all region types.
For example:

- **Surya** provides excellent general coverage (paragraphs, tables, headers)
  but may miss specialized elements like signatures or Indic-script regions.
- **YOLO-based models** trained on specific datasets (IndicDLP, DocLayNet)
  may detect region types that Surya misses, but introduce false positives
  in categories where they were not specifically trained.
- **Specialized detectors** (face detection, signature detection) are highly
  accurate for their target class but provide no general layout information.

**Layout Ensembler** solves this by combining the strengths of multiple
heterogeneous detectors through a configurable, class-aware merge strategy.

## Design Principles

1. **Model-agnostic** — Any detector that produces bounding boxes can be
   integrated by implementing the `BaseModel` interface.
2. **Class-aware** — The ensemble strategy understands that different models
   are authoritative for different region types.
3. **Configurable** — All behavior is driven by YAML configuration, not code
   changes.
4. **Production-ready** — Multi-GPU, batch processing, async I/O, and
   fault tolerance throughout.

## Pipeline Architecture

```
                           ┌──────────────────┐
                           │   Document Pages  │
                           └────────┬─────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────────┐
              │  Surya    │  │  YOLO    │  │  IndicDLP    │
              │ (GPU 0,1) │  │ (GPU 2)  │  │  (GPU 3)     │
              └─────┬─────┘  └─────┬────┘  └──────┬───────┘
                    │              │               │
                    ▼              ▼               ▼
              ┌─────────────────────────────────────────┐
              │         Class-Aware Priority NMS         │
              │                                          │
              │  1. Accept all base model detections     │
              │  2. Add supplementary (if no overlap)    │
              │  3. Protected classes bypass NMS         │
              │  4. Per-model confidence thresholds      │
              └──────────────────┬──────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────────┐
              │         Hierarchy Builder                 │
              │                                          │
              │  IoA-based parent/child assignment        │
              │  Greedy smallest-parent selection         │
              └──────────────────┬──────────────────────┘
                                 │
                      ┌──────────┼──────────┐
                      ▼                     ▼
               ┌─────────────┐      ┌──────────────┐
               │ Layout JSON │      │  Page Crops   │
               │  (per page) │      │  (per region) │
               └─────────────┘      └──────────────┘
```

## Key Components

### Detectors (`layout_ensembler/detectors/`)

All detectors implement the `BaseModel` abstract class:

```python
class BaseModel(ABC):
    def load(self) -> None: ...
    def predict(self, batch) -> list[list[Region]]: ...
    def unload(self) -> None: ...
```

This interface provides:
- **Lifecycle management** via `load()`/`unload()` or context managers
- **Batch processing** — all detectors accept batches for GPU efficiency
- **Uniform output** — all detectors emit `list[Region]`

#### Multi-GPU Parallelism

**Surya**: Creates one predictor per GPU, distributes images round-robin,
and collects results via `ThreadPoolExecutor`.

**YOLO**: Each checkpoint runs on its own GPU. All checkpoints see the full
batch, and their results are merged per image. This allows combining
a general model with specialized models.

### Ensemble (`layout_ensembler/ensemble.py`)

The `Ensembler` implements a five-phase merge:

1. **Base acceptance** — All detections from the base model pass through
2. **Class filtering** — Per-model allowlists prune irrelevant classes
3. **Confidence thresholding** — Per-model minimum confidence floors
4. **Protected bypass** — Certain classes are never suppressed
5. **IoU-based suppression** — Only non-overlapping supplementary detections are accepted

### Hierarchy (`layout_ensembler/hierarchy.py`)

Converts a flat list of regions into a tree:

- **Criterion**: Region A is a child of B if `IoA(A, B) ≥ threshold` and
  `area(A) < area(B)`
- **Tie-breaking**: Among valid parents, pick the smallest one (tightest fit)
- **Complexity**: O(n²) — acceptable for typical page layouts (< 200 regions)

### Data Models (`layout_ensembler/datamodels/`)

- **`Region`** — Universal detection primitive with bbox, label, score,
  source tracking, and hierarchy pointers (parent_id, children_ids)
- **`RegionType`** — Exhaustive enum of 50+ document region types
- **`TextLine`** — OCR-level primitive for text detection results
- **`LayoutCrop`** — Cropped region with OCR text for downstream tasks

### Configuration (`layout_ensembler/config.py`)

Pydantic-validated YAML configuration covering:
- Model selection and device placement
- Label taxonomy mappings
- Ensemble strategy parameters
- Numeric hyper-parameters

## Adding a New Detector

1. Create a new file in `layout_ensembler/detectors/`
2. Implement `BaseModel`:

```python
from layout_ensembler.detectors.base_model import BaseModel
from layout_ensembler.datamodels.regions import Region, RegionType

class MyDetector(BaseModel):
    def load(self):
        self._model = load_my_model(self.device)

    def predict(self, batch_inputs):
        results = []
        for img in batch_inputs:
            detections = self._model(img)
            regions = [
                Region(
                    xyxy=det.bbox,
                    label=RegionType(det.class_name),
                    score=det.confidence,
                    source="my_detector",
                )
                for det in detections
            ]
            results.append(regions)
        return results

    def unload(self):
        self._model = None
```

3. Add its label mapping to the config YAML
4. Include it in the ensemble priority order
