"""Tests for Region and TextLine data models."""

import json

import pytest

from layout_ensembler.datamodels.regions import Region, RegionType, TextLine


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------
class TestRegion:
    """Tests for the Region dataclass."""

    def test_create_valid_region(self):
        r = Region(
            xyxy=(10, 20, 100, 200),
            label=RegionType.paragraph,
            score=0.95,
            source="surya",
        )
        assert r.xyxy == (10, 20, 100, 200)
        assert r.label == RegionType.paragraph
        assert r.score == 0.95
        assert r.source == "surya"
        assert r.parent_id is None
        assert r.children_ids == []

    def test_invalid_bbox_raises(self):
        with pytest.raises(ValueError, match="Invalid bounding box"):
            Region(xyxy=(100, 20, 10, 200), label=RegionType.paragraph)

    def test_invalid_bbox_zero_height(self):
        with pytest.raises(ValueError, match="Invalid bounding box"):
            Region(xyxy=(10, 200, 100, 200), label=RegionType.paragraph)

    def test_invalid_score_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            Region(xyxy=(10, 20, 100, 200), label=RegionType.paragraph, score=1.5)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            Region(xyxy=(10, 20, 100, 200), label=RegionType.paragraph, score=-0.1)

    def test_none_score_is_valid(self):
        r = Region(xyxy=(10, 20, 100, 200), label=RegionType.paragraph, score=None)
        assert r.score is None

    def test_area_property(self):
        r = Region(xyxy=(0, 0, 100, 200), label=RegionType.paragraph)
        assert r.area == 20000

    def test_width_height(self):
        r = Region(xyxy=(10, 20, 110, 220), label=RegionType.paragraph)
        assert r.width == 100
        assert r.height == 200

    def test_to_dict(self):
        r = Region(
            xyxy=(10, 20, 100, 200),
            label=RegionType.table,
            score=0.8,
            source="yolo",
        )
        d = r.to_dict()
        assert d["label"] == "table"
        assert d["xyxy"] == (10, 20, 100, 200)
        assert d["score"] == 0.8

    def test_roundtrip_json(self):
        r = Region(
            xyxy=(10, 20, 100, 200),
            label=RegionType.figure,
            score=0.7,
            source="test",
        )
        json_str = r.to_json()
        r2 = Region.from_json(json_str)
        assert r2.xyxy == r.xyxy
        assert r2.label == r.label
        assert r2.score == r.score
        assert r2.source == r.source
        assert r2.id == r.id

    def test_from_dict(self):
        d = {
            "xyxy": [10, 20, 100, 200],
            "label": "paragraph",
            "score": 0.9,
            "source": "surya",
            "id": "abc123",
        }
        r = Region.from_dict(d)
        assert r.label == RegionType.paragraph
        assert r.id == "abc123"

    def test_unique_ids(self):
        r1 = Region(xyxy=(10, 20, 100, 200), label=RegionType.paragraph)
        r2 = Region(xyxy=(10, 20, 100, 200), label=RegionType.paragraph)
        assert r1.id != r2.id


# ---------------------------------------------------------------------------
# TextLine
# ---------------------------------------------------------------------------
class TestTextLine:
    """Tests for the TextLine dataclass."""

    def test_create_valid(self):
        tl = TextLine(text="Hello world", bbox=(10, 20, 200, 40), score=0.95)
        assert tl.text == "Hello world"
        assert tl.source == "surya"  # default

    def test_invalid_bbox_raises(self):
        with pytest.raises(ValueError, match="Invalid text line bbox"):
            TextLine(text="bad", bbox=(200, 20, 10, 40), score=0.5)

    def test_roundtrip_json(self):
        tl = TextLine(
            text="Sample text",
            bbox=(10, 20, 300, 50),
            score=0.88,
            source="ensemble",
        )
        json_str = tl.to_json()
        tl2 = TextLine.from_json(json_str)
        assert tl2.text == tl.text
        assert tl2.bbox == tl.bbox
        assert tl2.score == tl.score
        assert tl2.source == tl.source

    def test_frozen(self):
        tl = TextLine(text="immutable", bbox=(10, 20, 200, 40), score=0.5)
        with pytest.raises(AttributeError):
            tl.text = "modified"


# ---------------------------------------------------------------------------
# RegionType
# ---------------------------------------------------------------------------
class TestRegionType:
    """Tests for the RegionType enum."""

    def test_string_value(self):
        assert RegionType.paragraph.value == "paragraph"
        assert RegionType.section_title.value == "section-title"

    def test_from_string(self):
        rt = RegionType("table")
        assert rt == RegionType.table

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            RegionType("nonexistent_type")
