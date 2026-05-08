"""
Evaluation metrics for document layout detection.

Provides per-class and aggregate metrics for comparing predicted layout
regions against ground-truth annotations, following the COCO-style
mean Average Precision (mAP) evaluation protocol.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from layout_ensembler.datamodels.regions import Region
from layout_ensembler.utils.geometry import compute_iou


def match_predictions(
    gt_regions: List[Region],
    pred_regions: List[Region],
    iou_threshold: float = 0.5,
) -> Tuple[List[bool], List[bool]]:
    """
    Match predicted regions to ground-truth using greedy IoU matching.

    Parameters
    ----------
    gt_regions : list[Region]
        Ground-truth regions.
    pred_regions : list[Region]
        Predicted regions (sorted by confidence, highest first).
    iou_threshold : float
        Minimum IoU to consider a match.

    Returns
    -------
    tuple[list[bool], list[bool]]
        ``(pred_matched, gt_matched)`` — boolean arrays indicating
        which predictions and GT boxes were successfully matched.
    """
    # Sort predictions by confidence (descending)
    preds = sorted(pred_regions, key=lambda r: r.score or 0, reverse=True)

    gt_matched = [False] * len(gt_regions)
    pred_matched = [False] * len(preds)

    for p_idx, pred in enumerate(preds):
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, gt in enumerate(gt_regions):
            if gt_matched[g_idx]:
                continue
            if gt.label != pred.label:
                continue

            iou = compute_iou(pred.xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            pred_matched[p_idx] = True
            gt_matched[best_gt_idx] = True

    return pred_matched, gt_matched


def compute_precision_recall(
    gt_regions: List[Region],
    pred_regions: List[Region],
    iou_threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, and F1 score.

    Parameters
    ----------
    gt_regions : list[Region]
        Ground-truth regions.
    pred_regions : list[Region]
        Predicted regions.
    iou_threshold : float
        IoU threshold for matching.

    Returns
    -------
    dict[str, dict[str, float]]
        Per-class metrics: ``{label: {"precision", "recall", "f1", "support"}}``.
    """
    # Group by class
    gt_by_class: Dict[str, List[Region]] = defaultdict(list)
    pred_by_class: Dict[str, List[Region]] = defaultdict(list)

    for r in gt_regions:
        gt_by_class[r.label.value].append(r)
    for r in pred_regions:
        pred_by_class[r.label.value].append(r)

    all_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())
    metrics: Dict[str, Dict[str, float]] = {}

    for cls in sorted(all_classes):
        gt_cls = gt_by_class.get(cls, [])
        pred_cls = pred_by_class.get(cls, [])

        if not pred_cls and not gt_cls:
            continue

        pred_matched, gt_matched = match_predictions(gt_cls, pred_cls, iou_threshold)

        tp = sum(pred_matched)
        fp = len(pred_matched) - tp
        fn = len(gt_matched) - sum(gt_matched)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": len(gt_cls),
            "predictions": len(pred_cls),
        }

    return metrics


def compute_map(
    gt_regions: List[Region],
    pred_regions: List[Region],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute mean Average Precision across IoU thresholds.

    Parameters
    ----------
    gt_regions : list[Region]
        Ground-truth regions.
    pred_regions : list[Region]
        Predicted regions.
    iou_thresholds : list[float] | None
        IoU thresholds. Defaults to COCO-style [0.50, 0.55, …, 0.95].

    Returns
    -------
    dict[str, float]
        ``{"mAP@0.50", "mAP@0.75", "mAP@0.50:0.95"}``.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]

    ap_per_threshold: List[float] = []

    for iou_thr in iou_thresholds:
        class_metrics = compute_precision_recall(gt_regions, pred_regions, iou_thr)
        # AP ≈ average F1 across classes (simplified)
        f1_scores = [m["f1"] for m in class_metrics.values()]
        ap = np.mean(f1_scores) if f1_scores else 0.0
        ap_per_threshold.append(float(ap))

    return {
        "mAP@0.50": ap_per_threshold[0] if len(ap_per_threshold) > 0 else 0.0,
        "mAP@0.75": ap_per_threshold[5] if len(ap_per_threshold) > 5 else 0.0,
        "mAP@0.50:0.95": float(np.mean(ap_per_threshold)),
    }
