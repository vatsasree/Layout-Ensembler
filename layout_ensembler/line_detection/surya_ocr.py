"""
Surya-based OCR pipeline for batch text line detection and recognition.

Combines Surya's detection and recognition models to extract text lines
with bounding boxes, recognized text, and per-line language detection
from document page images.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


def _chunked(lst: list, n: int):
    """Yield successive *n*-sized chunks from *lst*."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj


class SuryaOCRBatch:
    """
    Batch OCR pipeline using Surya detection + recognition.

    Loads the Surya foundation model, detection predictor, and recognition
    predictor onto the specified device for GPU-accelerated batch inference.

    Parameters
    ----------
    ocr_device : str
        GPU device for the OCR models (e.g. ``"cuda:0"``).

    Example
    -------
    >>> pipeline = SuryaOCRBatch(ocr_device="cuda:0")
    >>> images = [Image.open(p).convert("RGB") for p in paths]
    >>> results = pipeline.batch_ocr(images)
    """

    def __init__(self, ocr_device: str = "cuda:0") -> None:
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        self.foundation = FoundationPredictor(device=ocr_device)
        self.recognition_predictor = RecognitionPredictor(self.foundation)
        self.detection_predictor = DetectionPredictor(device=ocr_device)
        self.ocr_device = ocr_device

        log.info("SuryaOCRBatch initialized on %s", ocr_device)

    def batch_ocr(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        """
        Run text detection + recognition on a batch of images.

        Parameters
        ----------
        images : list[PIL.Image.Image]
            RGB images.

        Returns
        -------
        list[list[dict]]
            Per-image list of OCR results, each containing:
            ``{"text", "bbox", "polygon"}``.
        """
        predictions = self.recognition_predictor(
            images, det_predictor=self.detection_predictor
        )

        batch_results: List[List[Dict[str, Any]]] = []

        for pred in predictions:
            ocr_blocks: List[Dict[str, Any]] = []
            for text_line in pred.text_lines:
                x1, y1, x2, y2 = text_line.bbox
                ocr_blocks.append(
                    {
                        "text": text_line.text,
                        "bbox": [x1, y1, x2, y2],
                        "polygon": text_line.polygon,
                    }
                )
            batch_results.append(ocr_blocks)

        return batch_results


def run_ocr_pipeline(
    root_dir: str,
    output_json: str,
    batch_size: int = 32,
    ocr_device: str = "cuda:0",
) -> None:
    """
    End-to-end OCR pipeline: scan a directory, batch-process all images,
    and write results to a single JSON file.

    Parameters
    ----------
    root_dir : str
        Directory to recursively scan for images.
    output_json : str
        Output JSON file path.
    batch_size : int
        Number of images per batch (adjust based on VRAM).
    ocr_device : str
        GPU device for OCR inference.
    """
    # Collect images
    all_paths: List[str] = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(IMAGE_EXTS):
                all_paths.append(os.path.join(root, fname))

    log.info("Found %d images in %s", len(all_paths), root_dir)

    if not all_paths:
        log.warning("No images found. Exiting.")
        return

    pipeline = SuryaOCRBatch(ocr_device=ocr_device)
    results: Dict[str, Any] = {}

    for batch_paths in tqdm(
        list(_chunked(all_paths, batch_size)),
        desc="OCR batches",
        unit="batch",
    ):
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        ocr_results = pipeline.batch_ocr(images)

        for i, img_path in enumerate(batch_paths):
            rel_path = os.path.relpath(img_path, root_dir)
            results[rel_path] = {"ocr": ocr_results[i]}

    # Save
    json_safe = _to_json_safe(results)
    with open(output_json, "w") as f:
        json.dump(json_safe, f, indent=2)

    log.info("✅ OCR complete — saved to %s", output_json)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch OCR with Surya")
    parser.add_argument("--root_dir", required=True, help="Image directory")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_ocr_pipeline(args.root_dir, args.output, args.batch_size, args.device)
