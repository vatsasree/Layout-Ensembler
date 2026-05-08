"""
Batch region cropping driver.

Reads layout JSON files produced by the layout detection pipeline and
crops the corresponding regions from page images, saving them as
individual PNG files with metadata.
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from layout_ensembler.cropper import PageCropper
from layout_ensembler.utils.io_utils import load_regions

log = logging.getLogger("layout_ensembler.drivers.crop_batch")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")


def _iter_pages(root: Path):
    """Yield page images recursively."""
    for img in root.rglob("*"):
        if img.suffix.lower() in IMG_EXT and img.parent.name == "pages":
            yield img


def _crop_page_task(
    page_path: Path,
    cropper: PageCropper,
    apply_padding: bool = True,
) -> int:
    """Crop one page. Returns number of crops written, 0 if layouts missing."""
    layouts_json = (
        page_path.parent.parent
        / "layouts"
        / "pages"
        / page_path.name.replace(page_path.suffix, ".json")
    )
    if not layouts_json.exists():
        return 0

    regions = load_regions(layouts_json)
    cropper.crop_page(page_path, page_path.stem, regions, apply_padding=apply_padding)
    return len(regions)


def main() -> None:
    """CLI entry point for batch cropping."""
    ap = argparse.ArgumentParser(
        description="Crop page regions using existing layout JSONs"
    )
    ap.add_argument("--root", required=True, type=Path, help="Data root directory")
    ap.add_argument("--no_pad", dest="pad", action="store_false", help="Disable padding")
    ap.add_argument("--pad_pct", type=float, default=0.02, help="Padding percentage")
    ap.add_argument("--workers", type=int, default=4, help="Thread pool size")
    args = ap.parse_args()

    cropper = PageCropper(pad_pct=args.pad_pct)
    pages = list(_iter_pages(args.root))
    log.info("Found %d page images", len(pages))

    t0 = time.time()
    written = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(
        total=len(pages), desc="Cropping"
    ) as pbar:
        fut_map = {
            pool.submit(_crop_page_task, p, cropper, args.pad): p for p in pages
        }
        for fut in as_completed(fut_map):
            try:
                written += fut.result()
            except Exception as e:
                log.error("Crop failed %s: %s", fut_map[fut].name, e)
            pbar.update(1)

    log.info("✅ Wrote %d crops in %.1fs", written, time.time() - t0)


if __name__ == "__main__":
    main()
