"""
Single-node batch driver: Surya + YOLO -> Merged Layout JSONs

Directory contract
------------------
INPUT
    /data_mount/{org_id}/{doc_id}/pages/{doc_id}_page_{n}.png
OUTPUT
    /data_mount/{org_id}/{doc_id}/layouts/{doc_id}_page_{n}.json
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import argparse
import json
import logging

# import torch
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)

import cv2
from PIL import Image
from tqdm import tqdm

from doc2html.cfg import load_cfg, LayoutCfg
from doc2html.layout.detectors.surya import SuryaModel
from doc2html.layout.detectors.yolo import YOLOModel
from doc2html.layout.datamodels.datamodels import Region
from doc2html.utils.io import load_regions
from concurrent.futures import ThreadPoolExecutor, as_completed
from doc2html.layout.cropper import PageCropper
from doc2html.layout.ensemble import Ensembler
from doc2html.layout.hierarchy import build_in_memory_hierarchy

from doc2html.layout.utils.viz import draw_regions

log = logging.getLogger("batch layout processing")
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff', '*.webp']

### UTILS
def pages_generator(root: Path):
    """
    Yields page PNG paths two levels deep: org/doc/pages/*.png
    """
    for org_dir in root.iterdir():
        if not org_dir.is_dir():
            continue
        for doc_dir in org_dir.iterdir():
            pages_dir = doc_dir / "pages"
            if not pages_dir.is_dir():
                continue
            for ext in image_extensions:
                yield from pages_dir.glob(ext)

def load_cv2_pil(path: Path):
    """
    Returns (cv2_img, pil_img) or raises RuntimeError.
    """
    cv2_img = cv2.imread(str(path))
    if cv2_img is None:
        raise RuntimeError(f"Unreadable {path}")
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    return cv2_img, pil_img

def write_json(out_path: Path, regions: List[Region]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([r.to_json() for idx, r in enumerate(regions)], f)

def to_output_path(in_path: Path, root: Path) -> Path:
    """
    /root/org/doc/pages/abc.png -> /root/org/doc/layouts/abc.json
    """
    rel = in_path.relative_to(root)
    out_rel = in_path.parent.parent / "layouts" / "pages" / rel.name.replace(".png", ".json").replace(".jpg", ".json").replace(".jpeg", ".json")
    return out_rel

# ------- WORKER ------------------------------------------------------------
def process_batch(
    cfg: LayoutCfg,
    batch_paths: List[Path],
    cv2_batch,
    pil_batch, 
    surya: SuryaModel,
    yolo: YOLOModel,
    writer_pool: ThreadPoolExecutor,
    root: Path,
):
    """
    1. Run detectors
    2. Submit async writes to writer_pool
    """
    ensembler = Ensembler(cfg.ensemble_config.model_dump())

    surya_out = surya.predict(pil_batch) if cfg.surya["enabled"] else [[] for _ in batch_paths]
    yolo_out  = yolo.predict(cv2_batch) if cfg.yolo["enabled"] else [[] for _ in batch_paths]

    for in_path, regs_s, regs_y, img in zip(batch_paths, surya_out, yolo_out, cv2_batch):
        by_model = {
            "surya" : regs_s,
            "IndicDLP": [r for r in regs_y if r.source == "IndicDLP"],
            "Signature_detection": [r for r in regs_y if r.source == "Signature_detection"],
            "yolov11x_best": [r for r in regs_y if r.source == "yolov11x_best"]
        }
        
        merged = ensembler(by_model)
        # Build heirarchy in memory -> adds parent ids, and children ids
        merged = build_in_memory_hierarchy(merged, ioa_thr=0.9)
        
        # # # When debugging write to some test site
        # # Construct the output path
        # out_path = Path("/code") / "tests" / "imgs" / "layouts" / f"{in_path.stem}.png"
        # # Make sure parent directories exist
        # out_path.parent.mkdir(parents=True, exist_ok=True)
        # out = draw_regions(img, merged, cfg.unified_label_viz)
        # import pdb; pdb.set_trace()
        # cv2.imwrite(str(Path("/code") / "tests" / "imgs" / "layouts" / f"{in_path.stem}.png"), out)

        out_path = to_output_path(in_path, root)
        writer_pool.submit(write_json, out_path, merged)

# --------- MAIN ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("stream-batch layout extractor") 
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--root", required=True, type=Path, help="/data_mount")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--read_threads", type=int, default=8)
    ap.add_argument("--write_threads", type=int, default=8)
    args = ap.parse_args()

    cfg: LayoutCfg = load_cfg(Path(args.cfg))
    if args.batch_size:
        cfg.batch_size = args.batch_size

    log.info("Scanning page files ...")
    all_pages = list(pages_generator(args.root))
    log.info("Discoverd %d pages before skip-check", len(all_pages))

    # Fast skip: Filter all pages whose JSON already exists
    pages = [
        p 
        for p in all_pages
        # if not to_output_path(p, args.root).exists()
    ]

    import pdb; pdb.set_trace()

    pages = pages
    log.info("Will process %d new pages", len(pages))

    if not pages:
        log.info("Nothing to do.")
        return

    # ---------- initialise models once -------------------------------
    if cfg.surya["enabled"]:
        surya = SuryaModel(cfg); surya.load()
    else:
        surya = None
        log.warning("Surya disabled (cfg or CLI)")

    run_yolo = cfg.yolo["enabled"]
    if run_yolo:
        yolo = YOLOModel(cfg); yolo.load()
    else:
        yolo = None
        log.warning("YOLO disabled (cfg or CLI)")

    # ---------- executors -------------------------------------------
    reader_pool = ThreadPoolExecutor(max_workers=args.read_threads)
    writer_pool = ThreadPoolExecutor(max_workers=args.write_threads)

    try:
        for i in tqdm(range(0, len(pages), cfg.batch_size), desc="Batches"):
            batch_paths = pages[i : i + cfg.batch_size]
            # async read
            read_futs = {
                reader_pool.submit(load_cv2_pil, p): p for p in batch_paths
            }

            cv2_batch, pil_batch, ok_paths = [], [], []
            for fut in as_completed(read_futs):
                path = read_futs[fut] # Turns out this is a dict with key path and value ()
                try: 
                    cv2_img, pil_img = fut.result()
                    cv2_batch.append(cv2_img)
                    pil_batch.append(pil_img)
                    ok_paths.append(path)
                except Exception as e:
                    log.error("Read fail %s: %s", path.name, e)
            if ok_paths:
                process_batch(
                    cfg,
                    ok_paths,
                    cv2_batch,
                    pil_batch,
                    surya, # type: ignore
                    yolo, # type: ignore
                    writer_pool,
                    args.root,
                )
    finally:
        reader_pool.shutdown(wait=True)
        writer_pool.shutdown(wait=True)
        surya.unload() # type: ignore
        yolo.unload() # type: ignore
        log.info("Completed all batches.")

if __name__ == "__main__":
    main()