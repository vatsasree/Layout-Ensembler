"""
Core data models for document layout regions and text lines.

Every detector adapter *must* emit :class:`Region` instances so that
downstream stages (ensemble, hierarchy, cropping, visualization) can
operate uniformly regardless of which model produced the detection.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Unified label taxonomy
# ---------------------------------------------------------------------------
class RegionType(str, Enum):
    """
    Exhaustive set of document region types used across the pipeline.

    Every detector maps its native labels into this taxonomy via the
    ``*_to_unified_map`` dictionaries in :class:`~layout_ensembler.config.LayoutCfg`.
    """

    advertisement = "advertisement"
    answer = "answer"
    author = "author"
    chapter_title = "chapter-title"
    caption = "caption"
    code = "code"
    contact_info = "contact-info"
    dateline = "dateline"
    figure = "figure"
    figure_caption = "figure-caption"
    first_level_question = "first-level-question"
    flag = "flag"
    folio = "folio"
    footer = "footer"
    footnote = "footnote"
    form = "form"
    formula = "formula"
    header = "header"
    page_number = "page-number"
    handwriting = "handwriting"
    headline = "headline"
    index = "index"  # type: ignore[assignment]
    jumpline = "jumpline"
    options = "options"
    ordered_list = "ordered-list"
    list_item = "list-item"
    paragraph = "paragraph"
    picture = "picture"
    placeholder_text = "placeholder-text"
    quote = "quote"
    reference = "reference"
    second_level_question = "second-level-question"
    section_title = "section-title"
    sidebar = "sidebar"
    signature = "signature"
    sub_headline = "sub-headline"
    sub_ordered_list = "sub-ordered-list"
    sub_section_title = "sub-section-title"
    subsub_ordered_list = "subsub-ordered-list"
    subsub_section_title = "subsub-section-title"
    sub_unordered_list = "sub-unordered-list"
    subsub_headline = "subsub-headline"
    subsub_unordered_list = "subsub-unordered-list"
    table = "table"
    table_caption = "table-caption"
    table_of_contents = "table-of-contents"
    text_inline_math = "text-inline-math"
    third_level_question = "third-level-question"
    title = "title"
    unordered_list = "unordered-list"
    website_link = "website-link"


# ---------------------------------------------------------------------------
# Region — the universal detection primitive
# ---------------------------------------------------------------------------
@dataclass(order=True)
class Region:
    """
    A single detected region on a document page.

    Parameters
    ----------
    xyxy : tuple[int, int, int, int]
        Bounding box in ``(x1, y1, x2, y2)`` pixel coordinates.
    label : RegionType
        Unified region type.
    score : float | None
        Detector confidence in ``[0, 1]``, or ``None`` if unavailable.
    source : str
        Name of the model that produced this detection (e.g. ``"surya"``).
    id : str
        UUID hex string — unique across the page.
    parent_id : str | None
        ID of the enclosing region (set by :func:`~layout_ensembler.hierarchy.build_hierarchy`).
    children_ids : list[str]
        IDs of regions enclosed by this one.
    payload : dict
        Arbitrary metadata attached by detectors or post-processors.
    """

    xyxy: Tuple[int, int, int, int]
    label: RegionType
    score: float | None = None
    source: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    children_ids: List[str] = field(default_factory=list, compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.xyxy
        if not (x1 < x2 and y1 < y2):
            raise ValueError(
                f"Invalid bounding box {self.xyxy}: requires x1 < x2 and y1 < y2."
            )
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"Confidence score {self.score} is out of range [0, 1]."
            )

    @property
    def area(self) -> int:
        """Area of the bounding box in pixels²."""
        x1, y1, x2, y2 = self.xyxy
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> int:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> int:
        return self.xyxy[3] - self.xyxy[1]

    # ── Serialization ──────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary (JSON-safe)."""
        d = asdict(self)
        d["label"] = self.label.value
        return d

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Region":
        """Construct a Region from a plain dictionary."""
        d = dict(d)  # shallow copy
        d["label"] = RegionType(d["label"])
        d["xyxy"] = tuple(d["xyxy"])
        return cls(**d)

    @classmethod
    def from_json(cls, txt: str) -> "Region":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(txt))


# ---------------------------------------------------------------------------
# TextLine — OCR-level primitive
# ---------------------------------------------------------------------------
@dataclass(frozen=True, order=True)
class TextLine:
    """
    A single detected text line with recognized content.

    Parameters
    ----------
    text : str
        Recognized text content.
    bbox : tuple[int, int, int, int]
        Bounding box in ``(x1, y1, x2, y2)`` pixel coordinates.
    score : float
        Recognition confidence.
    source : str
        Model that produced this line (e.g. ``"surya"``, ``"ensemble"``).
    """

    text: str
    bbox: Tuple[int, int, int, int]
    score: float
    source: str = "surya"

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        if not (x1 < x2 and y1 < y2):
            raise ValueError(f"Invalid text line bbox {self.bbox}.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, txt: str) -> "TextLine":
        d = json.loads(txt)
        d["bbox"] = tuple(d["bbox"])
        return cls(**d)
