"""
Face detection using YOLO for document images.

Detects faces within document pages — primarily used to validate
``Picture`` regions as identity photos (e.g. in ID cards, passports)
by checking if a detected face is contained within a picture region.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from layout_ensembler.utils.geometry import check_ioa

log = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


class FaceDetector:
    """
    YOLO-based face detector for document images.

    Uses a pre-trained YOLOv8 face detection model from HuggingFace Hub.
    Detections can be filtered against layout regions to identify which
    ``Picture`` regions contain faces (useful for ID document processing).

    Parameters
    ----------
    device : str
        GPU device for inference.
    repo_id : str
        HuggingFace Hub model repository.
    filename : str
        Model weight filename within the repository.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        repo_id: str = "arnabdhar/YOLOv8-Face-Detection",
        filename: str = "model.pt",
    ) -> None:
        self.device = device
        self.repo_id = repo_id
        self.filename = filename
        self._model = None

    def load(self) -> None:
        """Download (if needed) and load the face detection model."""
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        weight_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
        self._model = YOLO(weight_path)
        log.info("FaceDetector loaded from %s", self.repo_id)

    def unload(self) -> None:
        self._model = None

    def detect_batch(self, images: list) -> List[List[Dict]]:
        """
        Detect faces in a batch of images.

        Parameters
        ----------
        images : list
            PIL Images or cv2 arrays.

        Returns
        -------
        list[list[dict]]
            Per-image list of face detections with ``{"label": "Face", "bbox": [x1,y1,x2,y2]}``.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        outputs = self._model(images, device=self.device, verbose=False)
        batch_results: List[List[Dict]] = []

        for result in outputs:
            faces = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for det in boxes:
                    faces.append({
                        "label": "Face",
                        "bbox": det.tolist(),
                    })
            batch_results.append(faces)

        return batch_results

    @staticmethod
    def filter_photos_with_faces(
        layouts: List[Dict],
        ioa_threshold: float = 0.3,
    ) -> List[Dict]:
        """
        Filter ``Picture`` regions to keep only those containing a face.

        This is useful for ID document processing where you want to
        distinguish identity photos from decorative images.

        Parameters
        ----------
        layouts : list[dict]
            Layout detections including Picture and Face labels.
        ioa_threshold : float
            Minimum IoA(face, picture) to consider the face inside.

        Returns
        -------
        list[dict]
            Filtered list: only photos with faces + all signatures.
        """
        photos = [l for l in layouts if l["label"] == "Picture"]
        faces = [l for l in layouts if l["label"] == "Face"]
        signatures = [l for l in layouts if l["label"] == "Signature"]

        valid_photos = []
        for photo in photos:
            p_bbox = tuple(int(v) for v in photo["bbox"])
            for face in faces:
                f_bbox = tuple(int(v) for v in face["bbox"])
                if check_ioa(f_bbox, p_bbox, ioa_threshold):
                    valid_photos.append(photo)
                    break

        return valid_photos + signatures
