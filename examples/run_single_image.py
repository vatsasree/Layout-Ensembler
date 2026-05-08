"""
Minimal example: run the layout ensemble on a single image.

Usage::

    python examples/run_single_image.py \\
        --image /path/to/document.png \\
        --cfg configs/default.yaml \\
        --output_dir /tmp/layout_output
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run layout ensemble on a single image")
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument("--cfg", required=True, type=Path, help="Config YAML path")
    parser.add_argument("--output_dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Save annotated image")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────
    from layout_ensembler.config import load_cfg

    cfg = load_cfg(args.cfg)

    # ── Load image ────────────────────────────────────────────────
    cv2_img = cv2.imread(str(args.image))
    if cv2_img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    log.info("Loaded image: %s (%dx%d)", args.image.name, pil_img.width, pil_img.height)

    # ── Run detectors ─────────────────────────────────────────────
    from layout_ensembler.detectors.surya import SuryaModel
    from layout_ensembler.detectors.yolo import YOLOModel

    detections = {}

    if cfg.surya.get("enabled", False):
        with SuryaModel(cfg) as surya:
            surya_out = surya.predict([pil_img])
            detections["surya"] = surya_out[0]
            log.info("Surya detected %d regions", len(surya_out[0]))

    if cfg.yolo.get("enabled", False):
        with YOLOModel(cfg) as yolo:
            yolo_out = yolo.predict([cv2_img])
            for r in yolo_out[0]:
                detections.setdefault(r.source, []).append(r)
            log.info("YOLO detected %d regions", len(yolo_out[0]))

    # ── Ensemble ──────────────────────────────────────────────────
    from layout_ensembler.ensemble import Ensembler

    ensembler = Ensembler(cfg.ensemble_config.model_dump())
    merged = ensembler(detections)
    log.info("Ensemble produced %d regions", len(merged))

    # ── Hierarchy ─────────────────────────────────────────────────
    from layout_ensembler.hierarchy import build_hierarchy

    merged = build_hierarchy(merged, ioa_threshold=0.9)
    roots = sum(1 for r in merged if r.parent_id is None)
    log.info("Hierarchy: %d roots, %d total regions", roots, len(merged))

    # ── Save results ──────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = args.output_dir / f"{args.image.stem}_layout.json"
    data = [r.to_dict() for r in merged]
    json_path.write_text(json.dumps(data, indent=2))
    log.info("Saved layout JSON: %s", json_path)

    # Visualization
    if args.visualize:
        from layout_ensembler.utils.viz import draw_legend, draw_regions

        vis = draw_regions(cv2_img, merged, show_score=True, show_hierarchy=True)
        labels = list({r.label.value for r in merged})
        vis = draw_legend(vis, sorted(labels))
        vis_path = args.output_dir / f"{args.image.stem}_layout_viz.png"
        cv2.imwrite(str(vis_path), vis)
        log.info("Saved visualization: %s", vis_path)

    # Summary
    from collections import Counter

    counts = Counter(r.label.value for r in merged)
    log.info("Region breakdown: %s", dict(counts))


if __name__ == "__main__":
    main()
