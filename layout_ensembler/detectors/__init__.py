"""
Detector adapters — pluggable backends for layout detection.

Every concrete detector implements :class:`BaseModel` and emits
:class:`~layout_ensembler.datamodels.regions.Region` instances.
"""

from layout_ensembler.detectors.base_model import BaseModel

__all__ = ["BaseModel"]
