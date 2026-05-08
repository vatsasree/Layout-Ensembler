import os
import json
import numpy as np
from PIL import Image
import cv2

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import supervision as sv

from tqdm import tqdm

from utils.lang import detect_language

# ==============================
# Utils
# ==============================

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj

# ==============================
# Batch Surya Wrapper
# ==============================

class SuryaDetectionsBatch:
    def __init__(self, ocr_device="cuda:0", layout_device="cuda:1"):
        # -------- Shared foundation --------
        self.foundation = FoundationPredictor(device=ocr_device)

        # -------- OCR --------
        self.recognition_predictor = RecognitionPredictor(self.foundation)
        self.detection_predictor = DetectionPredictor(device=ocr_device)

        # -------- Layout --------
        self.layout_foundation = FoundationPredictor(
            checkpoint=settings.LAYOUT_MODEL_CHECKPOINT,
            device=layout_device
        )
        self.layout_predictor = LayoutPredictor(self.layout_foundation)

        # -------- YOLO models --------
        self.face_model = YOLO(
            hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        )
        self.signature_model = YOLO(
            hf_hub_download(repo_id="tech4humans/yolov8s-signature-detector", filename="yolov8s.pt")
        )
        self.indicdlp_model = YOLO(hf_hub_download(repo_id="IndicDLP/IndicDLP-model", filename="best.pt"))

        self.ocr_device = ocr_device
        self.layout_device = layout_device

    # ---------------------------
    # Geometry utils
    # ---------------------------
    def is_inside(self, inner, outer, min_iou=0.3):
        ix1 = max(inner[0], outer[0])
        iy1 = max(inner[1], outer[1])
        ix2 = min(inner[2], outer[2])
        iy2 = min(inner[3], outer[3])

        if ix2 <= ix1 or iy2 <= iy1:
            return False

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])

        return (inter_area / inner_area) >= min_iou

    def filter_only_photo_signature(self, layouts):
        photos = [l for l in layouts if l['label'] == 'Picture']
        faces = [l for l in layouts if l['label'] == 'Face']
        signatures = [l for l in layouts if l['label'] == 'Signature']

        valid_photos = []
        valid_faces = []

        for photo in photos:
            photo_bbox = photo['bbox']
            faces_in_photo = [
                face for face in faces
                if self.is_inside(face['bbox'], photo_bbox)
            ]
            if len(faces_in_photo) > 0:
                valid_photos.append(photo)
                valid_faces.extend(faces_in_photo)

        return valid_photos + signatures

    # ===========================
    # Batch OCR
    # ===========================
    def batch_ocr(self, images):
        predictions = self.recognition_predictor(
            images,
            det_predictor=self.detection_predictor
        )

        batch_results = []

        for pred in predictions:
            ocr_blocks = []
            for text_line in pred.text_lines:
                x1, y1, x2, y2 = text_line.bbox
                text = text_line.text
                language_detected = detect_language(text)

                ocr_blocks.append({
                    "text": text,
                    "language": language_detected,
                    "bbox": [x1, y1, x2, y2],
                    "polygon": text_line.polygon
                })
            batch_results.append(ocr_blocks)

        return batch_results

    # ===========================
    # Batch Layout + Face + Sign
    # ===========================
    def batch_layout(self, images):
        # Layout
        layout_preds = self.layout_predictor(images)

        # YOLO batch
        face_outputs = self.face_model(images, device=self.layout_device)
        sign_outputs = self.signature_model(images, device=self.layout_device)
        indicdlp_outputs = self.indicdlp_model(images, device=self.layout_device)

        batch_results = []

        for i in range(len(images)):
            layouts = []

            # ---- Layout boxes ----
            for box in layout_preds[i].bboxes:
                x1, y1, x2, y2 = box.bbox
                layouts.append({
                    "label": box.label,
                    "bbox": [x1, y1, x2, y2]
                })

            # ---- Faces ----
            if face_outputs[i].boxes is not None:
                boxes = face_outputs[i].boxes.xyxy.cpu().numpy()
                for det in boxes:
                    layouts.append({
                        "label": "Face",
                        "bbox": det.tolist()
                    })

            # ---- Signatures ----
            if sign_outputs[i].boxes is not None:
                boxes = sign_outputs[i].boxes.xyxy.cpu().numpy()
                for det in boxes:
                    layouts.append({
                        "label": "Signature",
                        "bbox": det.tolist()
                    })

            filtered = self.filter_only_photo_signature(layouts)
            batch_results.append(filtered)

        return batch_results

# ==============================
# Main Pipeline
# ==============================

# root_dir = "/fsxvision/sreevatsa.s/RawDownloadsImages_deskewed"
root_dir = '/home/sreevatsa.s/SynthDocGeneration/bgeneration/Clean/CleanAadhar'
output_json = "results_faces_batch_aadhar.json"

BATCH_SIZE = 32   # adjust based on VRAM

# -------- Collect images --------
all_paths = []
for root, _, files in os.walk(root_dir):
    for fname in files:
        if fname.lower().endswith(IMAGE_EXTS):
            all_paths.append(os.path.join(root, fname))

print(f"Total images: {len(all_paths)}")

surya = SuryaDetectionsBatch()

results = {}

# -------- Batch processing --------
# for batch_paths in chunked(all_paths, BATCH_SIZE):
for batch_paths in tqdm(list(chunked(all_paths, BATCH_SIZE)), desc="Batch inference", unit="batch"):
    print(f"Processing batch {batch_paths[0]} -> {batch_paths[-1]}")

    images = [Image.open(p).convert("RGB") for p in batch_paths]

    # ---- OCR batch ----
    ocr_results = surya.batch_ocr(images)

    # ---- Layout batch ----
    layout_results = surya.batch_layout(images)

    # ---- Store ----
    for i, img_path in enumerate(batch_paths):
        rel_path = os.path.relpath(img_path, root_dir)
        results[rel_path] = {
            "ocr": ocr_results[i],
            "layout": layout_results[i]
        }

# -------- Save JSON --------
json_safe_results = to_json_safe(results)

with open(output_json, "w") as f:
    json.dump(json_safe_results, f, indent=2)

print(f"\n✅ Batch inference complete")
print(f"Saved to: {output_json}")