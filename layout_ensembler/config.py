"""
Configuration management for Layout Ensembler.

Provides strongly-typed Pydantic models for YAML-based configuration of the
layout detection pipeline, including detector selection, label mapping,
ensemble strategy, and multi-GPU device allocation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
RGB = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Ensemble configuration
# ---------------------------------------------------------------------------
class EnsembleCfg(BaseModel):
    """
    Controls how detections from multiple models are merged.

    The ensembler operates in two phases:
      1. Accept *all* detections from the ``base_model_name``.
      2. Walk ``model_priority_order`` and add each supplementary detection
         only if it does not overlap (IoU < ``iou_threshold``) with any
         detection already accepted — unless its class is in
         ``protected_classes_from_overlap_removal``.

    Attributes
    ----------
    base_model_name : str
        The primary detector whose output forms the backbone of the final
        layout. All its detections are accepted unconditionally.
    model_priority_order : list[str]
        Ordered list of model names. Supplementary models are processed in
        this order; earlier models take precedence.
    iou_threshold : float
        Overlap threshold for suppression. A supplementary detection is
        rejected if its IoU with *any* already-accepted box ≥ this value.
    protected_classes_from_overlap_removal : list[str]
        Region types that are *never* suppressed, regardless of overlap.
    conf_thresholds_for_supplementary_models : dict[str, float]
        Per-model minimum confidence. Detections below the threshold are
        discarded before the overlap check.
    filtered_classes_to_keep : dict[str, list[str]]
        Per-model allowlist. If a model appears here, only the listed
        classes are retained from its output; everything else is dropped.
    """

    base_model_name: str
    model_priority_order: List[str] = Field(min_length=1)
    iou_threshold: float = Field(default=0.1, ge=0.0, le=1.0)

    protected_classes_from_overlap_removal: List[str] = Field(default_factory=list)
    conf_thresholds_for_supplementary_models: Dict[str, float] = Field(default_factory=dict)
    filtered_classes_to_keep: Dict[str, List[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_base_in_priority(self) -> "EnsembleCfg":
        """Warn (but don't fail) if base model is missing from priority list."""
        if self.base_model_name not in self.model_priority_order:
            import warnings

            warnings.warn(
                f"base_model_name '{self.base_model_name}' is not listed in "
                f"model_priority_order — it will still be used as the base.",
                stacklevel=2,
            )
        return self


# ---------------------------------------------------------------------------
# Main layout pipeline configuration
# ---------------------------------------------------------------------------
class LayoutCfg(BaseModel):
    """
    Top-level configuration for the layout detection pipeline.

    This is loaded from a YAML file via :func:`load_cfg` and validated at
    parse time. It controls every aspect of the pipeline: which detectors to
    run, how labels are unified, device placement, ensemble strategy, and
    numeric hyper-parameters.
    """

    # ── Paths ──────────────────────────────────────────────────────────
    input_base_path: Path = Field(description="Root directory containing input documents.")
    ocr_results_sub_path: Path = Field(default=Path("ocr"))
    layout_results_sub_path: Path = Field(default=Path("layouts"))
    text_attributes_sub_path: Path = Field(default=Path("text_attributes"))

    # ── Model identifiers ──────────────────────────────────────────────
    table_model_name: Path = Field(default=Path(""), description="Path to table-specific model.")
    generic_model_name: Path = Field(default=Path(""), description="Path to generic model.")

    # ── Label maps ─────────────────────────────────────────────────────
    multi_prompt_map: Dict[str, Path] = Field(default_factory=dict)
    yolo_to_unified_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from YOLO class names to the unified taxonomy.",
    )
    surya_to_unified_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from Surya class names to the unified taxonomy.",
    )
    direct_add_labels: List[str] = Field(
        default_factory=list,
        description="Labels added directly without mapping.",
    )

    # ── Visualization ──────────────────────────────────────────────────
    unified_label_viz: Dict[str, RGB] = Field(
        default_factory=dict,
        description="Per-label RGB color for debug visualizations.",
    )

    # ── Detector weights ───────────────────────────────────────────────
    yolo_model_paths: List[Path] = Field(
        default_factory=list,
        description="Ordered list of YOLO checkpoint paths.",
    )

    # ── Device mapping ─────────────────────────────────────────────────
    device_map: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="GPU assignment per model family, e.g. {'surya': [0,1], 'yolo': [2,3]}.",
    )

    # ── Model toggles ──────────────────────────────────────────────────
    yolo: Dict[str, bool] = Field(
        default_factory=lambda: {"enabled": True},
    )
    surya: Dict[str, bool] = Field(
        default_factory=lambda: {"enabled": True},
    )

    # ── Numeric hyper-parameters ───────────────────────────────────────
    box_padding: int = Field(default=10, gt=0)
    nms_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    text_enclosure_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    indicdlp_add_iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    batch_size: int = Field(default=32, gt=0)

    # ── Ensemble ───────────────────────────────────────────────────────
    ensemble_config: EnsembleCfg


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_cfg(path: Path) -> LayoutCfg:
    """Load a YAML file into a strongly-typed :class:`LayoutCfg`."""
    data = yaml.safe_load(path.read_text())
    return LayoutCfg.model_validate(data)
