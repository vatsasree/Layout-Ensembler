"""
Signature detection using YOLO for document images.

Detects handwritten signatures in document pages — critical for
processing legal documents, contracts, and identity documents.
"""

from __future__ import annotations

import logging
from typing import Dict, List

log = logging.getLogger(__name__)


class SignatureDetector:
    """
    YOLO-based signature detector for document images.

    Uses a pre-trained YOLOv8 signature detection model from HuggingFace Hub.

    Parameters
    ----------
    device : str
        GPU device for inference.
    repo_id : str
        HuggingFace Hub model repository.
    filename : str
        Model weight filename.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        repo_id: str = "tech4humans/yolov8s-signature-detector",
        filename: str = "yolov8s.pt",
    ) -> None:
        self.device = device
        self.repo_id = repo_id
        self.filename = filename
        self._model = None

    def load(self) -> None:
        """Download (if needed) and load the signature detection model."""
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        weight_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
        self._model = YOLO(weight_path)
        log.info("SignatureDetector loaded from %s", self.repo_id)

    def unload(self) -> None:
        self._model = None

    def detect_batch(self, images: list) -> List[List[Dict]]:
        """
        Detect signatures in a batch of images.

        Parameters
        ----------
        images : list
            PIL Images or cv2 arrays.

        Returns
        -------
        list[list[dict]]
            Per-image list of signature detections with
            ``{"label": "Signature", "bbox": [x1,y1,x2,y2]}``.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        outputs = self._model(images, device=self.device, verbose=False)
        batch_results: List[List[Dict]] = []

        for result in outputs:
            sigs = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for det in boxes:
                    sigs.append({
                        "label": "Signature",
                        "bbox": det.tolist(),
                    })
            batch_results.append(sigs)

        return batch_results
