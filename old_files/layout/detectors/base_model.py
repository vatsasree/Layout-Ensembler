from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List
import logging

from doc2html.layout.datamodels.datamodels import Region

log = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    An abstract base class for any model that returns `list[Region]`
    for a *batch* of inputs. All heavy resources (GPU, RAM, file_handles) must be 
    acquired in `load()` and released in `unload()`
    """
    def __init__(self, cfg: Any, device: str = "cpu") -> None:
        self.cfg = cfg # Keep the original cfg
        self.device = device
        self._model = None # A subclass should set this

    #Load the model -> (Weights, Tokenizer etc.)
    @abstractmethod
    def load(self) -> None:
        """
        Instantiate heavy objects (weights, tokenizer, ...)
        """
        pass

    @abstractmethod
    def predict(self, batch_inputs: list[Any]) -> list[list[Region]]:
        """
        Args: batch_inputs: list[Any]: A Homogeneous list (ex. cv2 Images, PIL Images).
        Returns: list[list[Region]]: One outer element per input, each inner list may be empty
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Free GPU/CPU memory, and close files.
        """
        pass

    # Let people do `yolo(batch)` instead of `yolo.predict(batch)`
    def __call__(self, batch_inputs: list[Any]) -> list[list[Region]]:
        return self.predict(batch_inputs)
    
    def is_available(self) -> bool:
        return self._model is not None