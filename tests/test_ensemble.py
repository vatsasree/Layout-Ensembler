"""Tests for the Ensembler — class-aware priority-based NMS."""

import pytest

from layout_ensembler.datamodels.regions import Region, RegionType
from layout_ensembler.ensemble import Ensembler


def _make_region(xyxy, label, score=0.9, source="test"):
    return Region(xyxy=xyxy, label=RegionType(label), score=score, source=source)


@pytest.fixture
def base_cfg():
    """Minimal ensemble configuration."""
    return {
        "base_model_name": "surya",
        "model_priority_order": ["surya", "yolo", "indicdlp"],
        "iou_threshold": 0.3,
        "protected_classes_from_overlap_removal": ["signature"],
        "conf_thresholds_for_supplementary_models": {"indicdlp": 0.5},
        "filtered_classes_to_keep": {},
    }


class TestEnsembler:

    def test_base_model_all_accepted(self, base_cfg):
        """All base model detections should be accepted."""
        ensembler = Ensembler(base_cfg)
        regions = {
            "surya": [
                _make_region((10, 10, 100, 100), "paragraph", source="surya"),
                _make_region((200, 200, 400, 400), "table", source="surya"),
            ]
        }
        result = ensembler(regions)
        assert len(result) == 2

    def test_non_overlapping_supplementary_accepted(self, base_cfg):
        """Supplementary detections that don't overlap should be accepted."""
        ensembler = Ensembler(base_cfg)
        regions = {
            "surya": [_make_region((10, 10, 100, 100), "paragraph", source="surya")],
            "yolo": [_make_region((500, 500, 700, 700), "table", source="yolo")],
        }
        result = ensembler(regions)
        assert len(result) == 2

    def test_overlapping_supplementary_rejected(self, base_cfg):
        """Supplementary detections overlapping with base should be rejected."""
        ensembler = Ensembler(base_cfg)
        regions = {
            "surya": [_make_region((10, 10, 100, 100), "paragraph", source="surya")],
            "yolo": [_make_region((20, 20, 90, 90), "paragraph", source="yolo")],
        }
        result = ensembler(regions)
        assert len(result) == 1
        assert result[0].source == "surya"

    def test_protected_class_always_accepted(self, base_cfg):
        """Protected classes should be accepted even with overlap."""
        ensembler = Ensembler(base_cfg)
        regions = {
            "surya": [_make_region((10, 10, 100, 100), "paragraph", source="surya")],
            "yolo": [_make_region((20, 20, 90, 90), "signature", source="yolo")],
        }
        result = ensembler(regions)
        assert len(result) == 2
        labels = {r.label.value for r in result}
        assert "signature" in labels

    def test_confidence_threshold_filtering(self, base_cfg):
        """Low-confidence supplementary detections should be filtered."""
        ensembler = Ensembler(base_cfg)
        regions = {
            "surya": [],
            "indicdlp": [
                _make_region((10, 10, 100, 100), "paragraph", score=0.3, source="indicdlp"),
                _make_region((200, 200, 400, 400), "table", score=0.8, source="indicdlp"),
            ],
        }
        result = ensembler(regions)
        # Only the high-confidence detection should pass (threshold is 0.5)
        assert len(result) == 1
        assert result[0].score == 0.8

    def test_class_filtering(self, base_cfg):
        """Per-model class allowlists should filter detections."""
        base_cfg["filtered_classes_to_keep"] = {"yolo": ["table"]}
        ensembler = Ensembler(base_cfg)
        regions = {
            "surya": [],
            "yolo": [
                _make_region((10, 10, 100, 100), "paragraph", source="yolo"),
                _make_region((200, 200, 400, 400), "table", source="yolo"),
            ],
        }
        result = ensembler(regions)
        assert len(result) == 1
        assert result[0].label == RegionType.table

    def test_empty_input(self, base_cfg):
        """Empty input should produce empty output."""
        ensembler = Ensembler(base_cfg)
        result = ensembler({})
        assert result == []

    def test_priority_order_matters(self, base_cfg):
        """Higher-priority supplementary models should take precedence."""
        base_cfg["iou_threshold"] = 0.3
        ensembler = Ensembler(base_cfg)

        # yolo comes before indicdlp in priority order
        # Both detect the same area, but yolo should win
        regions = {
            "surya": [],
            "yolo": [_make_region((10, 10, 100, 100), "table", score=0.7, source="yolo")],
            "indicdlp": [_make_region((15, 15, 95, 95), "table", score=0.9, source="indicdlp")],
        }
        result = ensembler(regions)
        assert len(result) == 1
        assert result[0].source == "yolo"

    def test_missing_base_model_graceful(self, base_cfg):
        """Missing base model in input should not crash."""
        ensembler = Ensembler(base_cfg)
        regions = {
            "yolo": [_make_region((10, 10, 100, 100), "table", source="yolo")],
        }
        result = ensembler(regions)
        assert len(result) == 1
