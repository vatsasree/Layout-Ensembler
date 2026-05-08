"""
Class-aware, priority-based detection ensemble.

This is the core algorithmic contribution of the project. Rather than
simple NMS across all detections, the ensembler recognizes that different
models excel at different region types and implements a **hierarchical
merge strategy**:

1. **Base model acceptance** — All detections from the designated base
   model (typically Surya) are accepted unconditionally, since it provides
   the most complete general coverage.

2. **Priority-ordered supplementary merge** — Detections from
   supplementary models (e.g. IndicDLP for Indic-script regions, a
   signature detector, etc.) are added in priority order, *only if* they
   don't overlap (IoU < threshold) with any already-accepted detection.

3. **Protected classes** — Certain region types (e.g. ``"signature"``)
   are *never* suppressed, even if they overlap, because they are
   semantically important and typically produced by a single specialized
   model.

4. **Per-model class filtering** — Each supplementary model can be
   restricted to only contribute specific classes, preventing a model
   strong on tables from introducing noisy paragraph detections.

5. **Per-model confidence thresholds** — Supplementary models can have
   higher confidence floors than the base model to reduce false positives.
"""

from __future__ import annotations

from typing import Any, Dict, List

from layout_ensembler.datamodels.regions import Region
from layout_ensembler.utils.geometry import compute_iou


class Ensembler:
    """
    Hierarchical, class-aware NMS for multi-model layout detection.

    Parameters
    ----------
    cfg : dict[str, Any]
        Ensemble configuration dictionary (see :class:`~layout_ensembler.config.EnsembleCfg`).
        Typically obtained via ``cfg.ensemble_config.model_dump()``.

    Example
    -------
    >>> ensembler = Ensembler(cfg.ensemble_config.model_dump())
    >>> merged = ensembler({
    ...     "surya": surya_regions,
    ...     "IndicDLP": indicdlp_regions,
    ...     "Signature_detection": sig_regions,
    ... })
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.base_model = cfg["base_model_name"]
        self.priority_models = cfg.get("model_priority_order", [])
        self.iou_thr = cfg.get("iou_threshold", 0.1)
        self.class_filters = cfg.get("filtered_classes_to_keep", {})
        self.protected = set(cfg.get("protected_classes_from_overlap_removal", []))
        self.conf_thr_overrides = cfg.get("conf_thresholds_for_supplementary_models", {})

    def __call__(self, detections_by_model: Dict[str, List[Region]]) -> List[Region]:
        """
        Merge detections from multiple models into a single, de-duplicated set.

        Parameters
        ----------
        detections_by_model : dict[str, list[Region]]
            Key = model name, value = list of regions detected by that model.

        Returns
        -------
        list[Region]
            The merged layout after class-aware priority NMS.
        """
        final: List[Region] = []

        # ── Phase 1: Accept all base model detections ─────────────
        base_regs = detections_by_model.get(self.base_model, [])
        final.extend(self._filter_by_class(base_regs, self.base_model))

        # ── Phase 2: Supplementary models in priority order ───────
        for model_name in self.priority_models:
            if model_name == self.base_model or model_name not in detections_by_model:
                continue

            candidates = self._filter_by_class(
                detections_by_model[model_name], model_name
            )

            # Apply per-model confidence threshold
            conf_thr = self.conf_thr_overrides.get(model_name)
            if conf_thr is not None:
                candidates = [r for r in candidates if (r.score or 0) >= conf_thr]

            for region in candidates:
                # Protected classes bypass overlap suppression
                if region.label.value in self.protected:
                    final.append(region)
                    continue

                # Add only if no significant overlap with existing detections
                if all(
                    compute_iou(region.xyxy, existing.xyxy) < self.iou_thr
                    for existing in final
                ):
                    final.append(region)

        return final

    def _filter_by_class(
        self, regions: List[Region], model_name: str
    ) -> List[Region]:
        """
        Apply per-model class allowlist.

        If the model has no entry in ``filtered_classes_to_keep``, all its
        detections pass through. Otherwise, only classes in the allowlist
        (plus protected classes) are retained.
        """
        keep = set(self.class_filters.get(model_name, []))
        if not keep:
            return regions
        return [
            r for r in regions
            if r.label.value in keep or r.label.value in self.protected
        ]
