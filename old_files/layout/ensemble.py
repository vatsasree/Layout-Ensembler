from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json
import math

from doc2html.layout.datamodels.datamodels import Region

def _iou(boxA, boxB) -> float:
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

class Ensembler:
    """
    Hierarchical, class-aware NMS that follows the rules in `ensemble_config`
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.base_model = cfg["base_model_name"]
        self.priority_models = cfg.get("model_priority_order", [])
        self.iou_thr = cfg.get("iou_threshold", 0.1)
        self.class_filters = cfg.get("filtered_classes_to_keep", {})
        self.protected = set(cfg.get("protected_classes_from_overlap_removal", []))

        # Confidence overrides per supplementary model
        self.conf_thr_overrides  = cfg.get("conf_thresholds_for_supplementary_models", {})

    def __call__(self, detections_by_model: Dict[str, list[Region]]) -> List[Region]:
        """
        Parameters
        ----------
        detections_by_model  key = model_name, value = list[Region]

        Returns
        -------
        List[Region] - the final set after ensembling
        """
        final: list[Region] = []

        # ---- 1. Base Model -----------------------------------------------
        base_regs = detections_by_model.get(self.base_model, [])
        final += self._filter_by_class(base_regs, self.base_model)

        # ---- 2. supplementary models in priority order -------------------
        for m in self.priority_models:
            if m == self.base_model or m not in detections_by_model:
                continue
            cand = self._filter_by_class(detections_by_model[m], m)
            # Confidence override
            if thr := self.conf_thr_overrides.get(m): # TBD
                cand = [r for r in cand if (r.score or 0) >= thr]
            for r in cand:
                if r.label in self.protected:
                    final.append(r)
                    continue
                if all(_iou(list(r.xyxy), list(f.xyxy)) < self.iou_thr for f in final):
                    final.append(r)
        return final
    
    # .......................................................................
    def _filter_by_class(self, regs: List[Region], model_name: str) -> List[Region]:
        keep = set(self.class_filters.get(model_name, []))
        if not keep:                       # empty list => keep everything
            return regs
        return [r for r in regs if r.label in keep or r.label in self.protected]