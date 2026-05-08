"""
YOLO layout detection adapter.

Supports loading **multiple YOLO checkpoints** (e.g. a general layout
detector alongside specialized models like IndicDLP or signature
detectors), each on its own GPU. All models see the full batch, and
their detections are merged per image.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import torch
from ultralytics import YOLO

from layout_ensembler.config import LayoutCfg
from layout_ensembler.datamodels.regions import Region, RegionType
from layout_ensembler.detectors.base_model import BaseModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker function (module-level for thread safety)
# ---------------------------------------------------------------------------
def _run_net(
    net: YOLO,
    batch_cv2: List,
    img_indices: List[int],
    cfg: LayoutCfg,
    source_name: str,
) -> Dict[int, List[Region]]:
    """
    Run a single YOLO network on the batch and return results keyed by
    global image index.
    """
    out: Dict[int, List[Region]] = {}
    try:
        raw = net.predict(source=batch_cv2, conf=0.2, iou=0.7, verbose=False)
        for i, res in enumerate(raw):
            gidx = img_indices[i]
            regs: List[Region] = []
            for b in res.boxes:
                label_raw = net.names[int(b.cls)]
                unified = cfg.yolo_to_unified_map.get(label_raw, label_raw)
                regs.append(
                    Region(
                        xyxy=tuple(int(v) for v in b.xyxy[0].tolist()),
                        label=RegionType(unified),
                        score=float(b.conf),
                        source=source_name,
                    )
                )
            out[gidx] = regs
    except Exception as e:
        log.error("YOLO net '%s' failed: %s", source_name, e)
    return out


# ---------------------------------------------------------------------------
# YOLOModel
# ---------------------------------------------------------------------------
class YOLOModel(BaseModel):
    """
    Multi-checkpoint YOLO detector.

    Loads one YOLO model per entry in ``cfg.yolo_model_paths``, assigning
    each to a different GPU via ``cfg.device_map["yolo"]``. At inference
    time every model sees the full batch and their detections are merged.

    This design allows combining:
      - A general layout detector (e.g. YOLOv11x trained on DocLayNet)
      - A specialized detector (e.g. IndicDLP for Indic scripts)
      - A signature detector

    … into a single unified output per image.
    """

    def load(self) -> None:
        self._source_map: Dict[str, str] = {}  # "cuda:N" → checkpoint stem
        self._nets: List[YOLO] = []

        devices = self.cfg.device_map.get("yolo") or list(range(torch.cuda.device_count()))

        if len(devices) < len(self.cfg.yolo_model_paths):
            raise RuntimeError(
                f"Not enough GPUs ({len(devices)}) for "
                f"{len(self.cfg.yolo_model_paths)} YOLO models."
            )

        for weight_path, dev_id in zip(self.cfg.yolo_model_paths, devices):
            net = YOLO(str(weight_path))
            net.to(f"cuda:{dev_id}")
            self._source_map[f"cuda:{dev_id}"] = Path(weight_path).stem
            self._nets.append(net)
            log.info("  ✓ YOLO '%s' on cuda:%d", Path(weight_path).stem, dev_id)

        self._model = self._nets
        log.info("Loaded %d YOLO checkpoint(s)", len(self._nets))

    def unload(self) -> None:
        self._nets.clear()
        self._source_map.clear()
        self._model = None

    @torch.no_grad()
    def predict(self, batch_cv2: List) -> List[List[Region]]:
        """
        Run all YOLO checkpoints on the batch and merge detections.

        Parameters
        ----------
        batch_cv2 : list
            List of cv2 BGR images (``np.ndarray``).

        Returns
        -------
        list[list[Region]]
            Merged detections from all checkpoints, parallel to input.
        """
        n_imgs = len(batch_cv2)
        if not self.is_available() or n_imgs == 0:
            return [[] for _ in batch_cv2]

        merged: List[List[Region]] = [[] for _ in batch_cv2]

        # Each net processes the full batch; results are merged
        with ThreadPoolExecutor(max_workers=len(self._nets)) as pool:
            futures = [
                pool.submit(
                    _run_net,
                    net,
                    batch_cv2,
                    list(range(n_imgs)),
                    self.cfg,
                    self._source_map.get(f"cuda:{net.device.index}", Path("unknown").stem),
                )
                for net in self._nets
            ]
            for fut in as_completed(futures):
                piece = fut.result()
                for idx, regs in piece.items():
                    merged[idx].extend(regs)

        return merged
