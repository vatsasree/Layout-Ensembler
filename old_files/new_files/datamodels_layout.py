from __future__ import annotations
from typing import List, Literal
from . import _Base, BBox

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Any, Dict, List, Iterable
import json

# Picked up from -- GR_OCR directly --
@dataclass(frozen=True) 
class Detection: # Line level box
    bbox: BBox
    score: float
    source: str # hisam, surya, ensemble

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_schema": "gr-ocr/detection/v1",
            "bbox": list(self.bbox),
            "score": self.score,
            "source": self.source
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Detection":
        return Detection(tuple(d["bbox"]), float(d["score"]), d["source"])

@dataclass(frozen=True)
class Recognition: # Line level string + box
    """Result of your OCR (text-recognition) model on a single detection."""
    text: str
    score: float
    source_rec: str  # e.g. 'surya', 'google_ocr'

@dataclass(frozen=True)
class Line:
    """
    The combined, end-to-end result of detection + recognition.
    Keeps both the raw Detection (for any downstream use)
    and the decoded text.
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
            "source_rec": self.recognition.source_rec
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Line":
        det = Detection(tuple(d["bbox"]), float(d["det_score"]), d["source_det"])
        rec = Recognition(d["rec_text"], float(d["rec_score"]), d["source_rec"])
        return Line(det, rec)

class WordAttr(_Base):
    text: str
    bbox: BBox
    bold: bool = False
    italic: bool = False
    underline: bool = False

class LayoutCrop(_Base):
    """
    Single region cropped from the page.
    """
    label: str # paragraph, header, table
    bbox: BBox 
    page_id: str # To know, what image it belonged to 
    image_path: str
    ocr_text: List[Line] = [] # Lines -> plain text
    word_attr: List[WordAttr] = [] # Maybe empty

    # children, and parents
    id: str 
    parent_id: str | None = None
    children_ids: List[str] | None

    def prompt_context(self, max_chars=1e6) -> str:
        """
        Return truncated OCR + attribute markers for the prompt
        """
        # For now, just return the text
        return " ".join(e.recognition.text for e in self.ocr_text)