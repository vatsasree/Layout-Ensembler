"""
Layout Ensembler — Multi-model ensemble framework for document layout detection.

Combines the class-specific strengths of heterogeneous detectors (Surya, YOLO,
IndicDLP, etc.) through a configurable, priority-based NMS strategy to produce
a single, high-quality set of layout regions per document page.
"""

__version__ = "0.1.0"

from layout_ensembler.datamodels.regions import Region, RegionType, TextLine
from layout_ensembler.ensemble import Ensembler
from layout_ensembler.hierarchy import build_hierarchy

__all__ = [
    "Region",
    "RegionType",
    "TextLine",
    "Ensembler",
    "build_hierarchy",
    "__version__",
]
