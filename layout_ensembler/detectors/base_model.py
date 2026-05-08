"""
Abstract base class for layout detection models.

All heavy resources (GPU memory, model weights, file handles) are managed
through the explicit :meth:`load` / :meth:`unload` lifecycle.  Subclasses
can also be used as context managers for automatic cleanup.

Example
-------
>>> class MyDetector(BaseModel):
...     def load(self): ...
...     def predict(self, imgs): ...
...     def unload(self): ...
>>>
>>> with MyDetector(cfg, device="cuda:0") as det:
...     results = det(batch_of_images)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List

from layout_ensembler.datamodels.regions import Region

log = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for any model that returns ``list[Region]`` for a
    *batch* of inputs.

    Parameters
    ----------
    cfg : Any
        Configuration object (typically :class:`~layout_ensembler.config.LayoutCfg`).
    device : str
        Target device string, e.g. ``"cpu"``, ``"cuda:0"``.
    """

    def __init__(self, cfg: Any, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self._model = None  # subclass sets this in load()

    # ── Lifecycle ──────────────────────────────────────────────────────
    @abstractmethod
    def load(self) -> None:
        """Instantiate heavy objects (weights, tokenizers, …)."""

    @abstractmethod
    def predict(self, batch_inputs: List[Any]) -> List[List[Region]]:
        """
        Run inference on a batch.

        Parameters
        ----------
        batch_inputs : list
            Homogeneous list of inputs (e.g. PIL Images or cv2 arrays).

        Returns
        -------
        list[list[Region]]
            One inner list per input. Empty list if no detections.
        """

    @abstractmethod
    def unload(self) -> None:
        """Free GPU / CPU memory and close file handles."""

    # ── Convenience ────────────────────────────────────────────────────
    def __call__(self, batch_inputs: List[Any]) -> List[List[Region]]:
        """Alias for :meth:`predict`."""
        return self.predict(batch_inputs)

    def is_available(self) -> bool:
        """``True`` if the model has been loaded successfully."""
        return self._model is not None

    # ── Context manager ────────────────────────────────────────────────
    def __enter__(self) -> "BaseModel":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device!r}, loaded={self.is_available()})"
