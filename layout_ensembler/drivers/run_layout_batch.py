"""
Batch layout detection driver.

Scans a document repository, runs Surya + YOLO detectors, merges results
through the ensemble, builds the region hierarchy, and writes per-page
layout JSON files.

Directory contract
------------------
**Input**::

    {root}/{org_id}/{doc_id}/pages/{doc_id}_page_{n}.png

**Output**::

    {root}/{org_id}/{doc_id}/layouts/pages/{doc_id}_page_{n}.json
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import cv2
from PIL import Image
from tqdm import tqdm

from layout_ensembler.config import LayoutCfg, load_cfg
from layout_ensembler.datamodels.regions import Region
from layout_ensembler.detectors.surya import SuryaModel
from layout_ensembler.detectors.yolo import YOLOModel
from layout_ensembler.ensemble import Ensembler
from layout_ensembler.hierarchy import build_hierarchy

log = logging.getLogger("layout_ensembler.drivers.layout_batch")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff", "*.webp")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _iter_pages(root: Path):
    """Yield page image paths two levels deep: ``org/doc/pages/*.ext``."""
    for org_dir in root.iterdir():
        if not org_dir.is_dir():
            continue
        for doc_dir in org_dir.iterdir():
            pages_dir = doc_dir / "pages"
            if not pages_dir.is_dir():
                continue
            for ext in IMAGE_EXTS:
                yield from pages_dir.glob(ext)


def _load_cv2_pil(path: Path):
    """Load an image as both cv2 BGR and PIL RGB."""
    cv2_img = cv2.imread(str(path))
    if cv2_img is None:
        raise RuntimeError(f"Unreadable image: {path}")
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    return cv2_img, pil_img


def _write_json(out_path: Path, regions: List[Region]) -> None:
    """Write regions as JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in regions]
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)


def _to_output_path(in_path: Path) -> Path:
    """Map an input page path to its layout JSON output path."""
    return (
        in_path.parent.parent
        / "layouts"
        / "pages"
        / in_path.with_suffix(".json").name
    )


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------
def _process_batch(
    cfg: LayoutCfg,
    batch_paths: List[Path],
    cv2_batch: list,
    pil_batch: list,
    surya: SuryaModel,
    yolo: YOLOModel,
    writer_pool: ThreadPoolExecutor,
) -> None:
    """
    Run detectors → ensemble → hierarchy → async write for one batch.
    """
    ensembler = Ensembler(cfg.ensemble_config.model_dump())

    surya_out = surya.predict(pil_batch) if cfg.surya["enabled"] else [[] for _ in batch_paths]
    yolo_out = yolo.predict(cv2_batch) if cfg.yolo["enabled"] else [[] for _ in batch_paths]

    for in_path, regs_s, regs_y in zip(batch_paths, surya_out, yolo_out):
        # Group YOLO detections by source model
        by_model = {"surya": regs_s}
        for r in regs_y:
            by_model.setdefault(r.source, []).append(r)

        merged = ensembler(by_model)
        merged = build_hierarchy(merged, ioa_threshold=0.9)

        out_path = _to_output_path(in_path)
        writer_pool.submit(_write_json, out_path, merged)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point for batch layout detection."""
    ap = argparse.ArgumentParser(
        description="Batch layout detection: Surya + YOLO → Ensemble → Hierarchy → JSON"
    )
    ap.add_argument("--cfg", required=True, help="Path to layout YAML config")
    ap.add_argument("--root", required=True, type=Path, help="Data root directory")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--read_threads", type=int, default=8)
    ap.add_argument("--write_threads", type=int, default=8)
    args = ap.parse_args()

    cfg = load_cfg(Path(args.cfg))
    if args.batch_size:
        cfg.batch_size = args.batch_size

    # ── Discover pages ────────────────────────────────────────────
    log.info("Scanning for page images …")
    all_pages = list(_iter_pages(args.root))

    # Skip pages that already have layout JSON
    pages = [p for p in all_pages if not _to_output_path(p).exists()]
    log.info("Found %d total pages, %d new (skipping existing)", len(all_pages), len(pages))

    if not pages:
        log.info("Nothing to process.")
        return

    # ── Load models ───────────────────────────────────────────────
    surya = SuryaModel(cfg) if cfg.surya["enabled"] else None
    yolo = YOLOModel(cfg) if cfg.yolo["enabled"] else None

    if surya:
        surya.load()
    if yolo:
        yolo.load()

    # ── Process ───────────────────────────────────────────────────
    reader_pool = ThreadPoolExecutor(max_workers=args.read_threads)
    writer_pool = ThreadPoolExecutor(max_workers=args.write_threads)

    try:
        for i in tqdm(range(0, len(pages), cfg.batch_size), desc="Batches"):
            batch_paths = pages[i : i + cfg.batch_size]

            # Async read
            read_futs = {reader_pool.submit(_load_cv2_pil, p): p for p in batch_paths}
            cv2_batch, pil_batch, ok_paths = [], [], []

            for fut in as_completed(read_futs):
                path = read_futs[fut]
                try:
                    cv2_img, pil_img = fut.result()
                    cv2_batch.append(cv2_img)
                    pil_batch.append(pil_img)
                    ok_paths.append(path)
                except Exception as e:
                    log.error("Read failed %s: %s", path.name, e)

            if ok_paths:
                _process_batch(cfg, ok_paths, cv2_batch, pil_batch, surya, yolo, writer_pool)

    finally:
        reader_pool.shutdown(wait=True)
        writer_pool.shutdown(wait=True)
        if surya:
            surya.unload()
        if yolo:
            yolo.unload()
        log.info("✅ All batches complete.")


if __name__ == "__main__":
    main()
