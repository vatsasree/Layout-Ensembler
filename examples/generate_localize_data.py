import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
import cv2
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPECIALIST_MAP = {
    "paragraph": "text",
    "header": "text",
    "footer": "text",
    "title": "text",
    "section-title": "text",
    "caption": "text",
    "page-number": "text",
    "footnote": "text",
    "signature": "text",
    "handwriting": "text",
    "table": "table",
    "figure": "image",
    "picture": "image",
    "list-item": "list",
    "formula": "equation"
}

def generate_subhtml(label, text, specialist):
    text = text.replace('\n', '<br>')
    if specialist == "table":
        return f"<table><tr><td>{text}</td></tr></table>"
    elif specialist == "image":
        return f"<img alt='{label}' />"
    elif specialist == "list":
        return f"<ul><li>{text}</li></ul>"
    elif specialist == "equation":
        return f"<div class='equation'>$$ {text} $$</div>"
    else:
        if label == "title":
            return f"<h1>{text}</h1>"
        elif label in ["header", "section-title"]:
            return f"<h2>{text}</h2>"
        else:
            return f"<p>{text}</p>"

def main():
    parser = argparse.ArgumentParser(description="Run layout ensemble + OCR on a single image")
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument("--cfg", required=True, type=Path, help="Config YAML path")
    parser.add_argument("--output_file", type=Path, default=Path("output.js"), help="Output JS file")
    args = parser.parse_args()

    # Load config
    from layout_ensembler.config import load_cfg
    cfg = load_cfg(args.cfg)

    # Load image
    cv2_img = cv2.imread(str(args.image))
    if cv2_img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    img_h, img_w = cv2_img.shape[:2]
    
    # Run detectors
    from layout_ensembler.detectors.surya import SuryaModel
    from layout_ensembler.detectors.yolo import YOLOModel

    detections = {}

    if cfg.surya.get("enabled", False):
        with SuryaModel(cfg) as surya:
            surya_out = surya.predict([pil_img])
            detections["surya"] = surya_out[0]

    if cfg.yolo.get("enabled", False) and getattr(cfg, 'yolo_model_paths', []):
        with YOLOModel(cfg) as yolo:
            yolo_out = yolo.predict([cv2_img])
            for r in yolo_out[0]:
                detections.setdefault(r.source, []).append(r)

    # Ensemble
    from layout_ensembler.ensemble import Ensembler
    ensembler = Ensembler(cfg.ensemble_config.model_dump())
    merged = ensembler(detections)

    # Hierarchy
    from layout_ensembler.hierarchy import build_hierarchy
    merged = build_hierarchy(merged, ioa_threshold=0.9)

    # Run Full-Page Text Detection
    from surya.detection import DetectionPredictor
    det_predictor = DetectionPredictor()
    det_results = det_predictor([pil_img])[0]
    line_polys = [b.polygon for b in det_results.bboxes]

    # Run Full-Page OCR
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    
    rec_foundation = FoundationPredictor()
    rec_predictor = RecognitionPredictor(rec_foundation)
    
    ocr_results = rec_predictor([pil_img], polygons=[line_polys])[0]

    def get_center(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    crops = []
    for r in merged:
        label = r.label.value
        specialist = SPECIALIST_MAP.get(label, "text")
        
        # Calculate fractions x, y, w, h
        lx1, ly1, lx2, ly2 = r.xyxy
        w = lx2 - lx1
        h = ly2 - ly1
        frac_x = round(lx1 / img_w, 4)
        frac_y = round(ly1 / img_h, 4)
        frac_w = round(w / img_w, 4)
        frac_h = round(h / img_h, 4)
        
        # Find all OCR lines whose center falls within this layout bbox
        region_texts = []
        for poly, rec_line in zip(line_polys, ocr_results.text_lines):
            cx, cy = get_center(poly)
            if lx1 <= cx <= lx2 and ly1 <= cy <= ly2:
                region_texts.append(rec_line.text)
                
        ocr_text = "\n".join(region_texts)
        subhtml = generate_subhtml(label, ocr_text, specialist)
        
        crops.append({
            "label": label.capitalize(),
            "bbox": [frac_x, frac_y, frac_w, frac_h],
            "ocr": ocr_text,
            "specialist": specialist,
            "subhtml": subhtml
        })
        
    page_name = args.image.stem
    result_obj = {
        "page": page_name,
        "crops": crops
    }
    
    json_str = json.dumps(result_obj, indent=2)
    # The user wants LOCALIZE_DATA["gov"] = ...
    # We can just output exactly what they asked
    final_output = f'LOCALIZE_DATA["{page_name}"] = {json_str}'
    
    with args.output_file.open("a") as f:
        f.write(final_output + "\n")
    log.info(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
