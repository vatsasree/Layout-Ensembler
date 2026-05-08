from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Tuple, Dict, Any, List 
import json, uuid

# Anything after this must accept or emit list[Region]
from enum import Enum

class RegionType(str, Enum):
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
    index = "index" # type: ignore
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

@dataclass(order=True)
class Region: 
    xyxy: Tuple[int, int, int, int]
    label: RegionType
    score: float | None = None
    source: str = ""
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)

    # ----- NEW FIELDS # ---------------------------
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    children_ids: List[str] = field(default_factory=list, compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post__init__(self):
        x1, y1, x2, y2 = self.xyxy
        if not (x1 < x2 and y1 < y2):
            raise ValueError(self.xyxy)
        if self.score is not None and not 0 <= self.score <= 1:
            raise ValueError("Confidence out of Range")
        
    # -- (De Serialization) -- #
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, txt: str) -> "Region":
        d = json.loads(txt) 
        d["label"] = RegionType(d["label"])
        return cls(**d)
    
@dataclass(frozen=True, order=True)
class TextLine:
    text: str
    bbox: Tuple[int, int, int, int] #x1, y1, x2, y2
    score: float
    source: str = "surya|hisam|ensemble"

    # -- (Post Initialization) -- #
    def __post__init__(self):
        x1, y1, x2, y2 = self.bbox
        if not (x1 < x2 and y1 < y2):
            raise ValueError(self.bbox)
    
    # -- (De Serialization) -- #
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    # -- (Load Serialization) -- #
    @classmethod
    def from_json(cls, txt: "str") -> "TextLine":
        d = json.loads(txt)
        return cls(**d)