"""
Layout crop and OCR line data models.

These types represent the *output* of the layout pipeline: cropped page
regions enriched with OCR text and word-level attributes, ready for
downstream document understanding tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
BBox = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# OCR primitives (from GR-OCR schema)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    """Line-level bounding box from a text detection model."""

    bbox: BBox
    score: float
    source: str  # "surya", "hisam", "ensemble"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_schema": "gr-ocr/detection/v1",
            "bbox": list(self.bbox),
            "score": self.score,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Detection":
        return cls(bbox=tuple(d["bbox"]), score=float(d["score"]), source=d["source"])


@dataclass(frozen=True)
class Recognition:
    """Text recognition result for a single detected line."""

    text: str
    score: float
    source_rec: str  # "surya", "google_ocr"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "source_rec": self.source_rec,
        }


@dataclass(frozen=True)
class Line:
    """
    End-to-end OCR result combining detection and recognition.

    Preserves both the spatial information (bounding box) and the decoded
    text so that downstream consumers can choose how to use each.
    """

    detection: Detection
    recognition: Recognition

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_schema": "gr-ocr/line/v1",
            "bbox": list(self.detection.bbox),
            "det_score": self.detection.score,
            "rec_text": self.recognition.text,
            "rec_score": self.recognition.score,
            "source_det": self.detection.source,
            "source_rec": self.recognition.source_rec,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Line":
        det = Detection(
            bbox=tuple(d["bbox"]),
            score=float(d["det_score"]),
            source=d["source_det"],
        )
        rec = Recognition(
            text=d["rec_text"],
            score=float(d["rec_score"]),
            source_rec=d["source_rec"],
        )
        return cls(detection=det, recognition=rec)


# ---------------------------------------------------------------------------
# Word-level attributes (for rich text rendering)
# ---------------------------------------------------------------------------
class WordAttr(BaseModel):
    """Word with typographic attributes for HTML / PDF generation."""

    text: str
    bbox: BBox
    bold: bool = False
    italic: bool = False
    underline: bool = False


# ---------------------------------------------------------------------------
# LayoutCrop — a cropped region ready for downstream tasks
# ---------------------------------------------------------------------------
class LayoutCrop(BaseModel):
    """
    A single region cropped from a page image, enriched with OCR text.

    This is the final output type: it ties together the spatial detection
    (bounding box, parent/child hierarchy) with the textual content
    extracted from the cropped region.
    """

    id: str
    label: str
    bbox: BBox
    page_id: str
    image_path: str = ""
    ocr_text: List[Dict[str, Any]] = []  # serialized Line dicts
    word_attr: List[WordAttr] = []

    parent_id: Optional[str] = None
    children_ids: Optional[List[str]] = None

    def prompt_context(self, max_chars: int = 1_000_000) -> str:
        """
        Return concatenated OCR text for use as LLM prompt context.

        Parameters
        ----------
        max_chars : int
            Truncate output to this many characters.
        """
        lines = []
        for entry in self.ocr_text:
            if isinstance(entry, dict):
                lines.append(entry.get("rec_text", ""))
            elif hasattr(entry, "recognition"):
                lines.append(entry.recognition.text)
        text = " ".join(lines)
        return text[:max_chars]
