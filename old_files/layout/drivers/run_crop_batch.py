from __future__ import annotations
from pathlib import Path
import argparse, logging, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from doc2html.utils.io import load_regions
from doc2html.layout.cropper import PageCropper
from doc2html.layout.datamodels.datamodels import Region

log = logging.getLogger("cropper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")

def iter_pages(root: Path):
    for img in root.rglob("*"):
        if img.suffix.lower() in IMG_EXT and img.parent.name == "pages":
            yield img

def crop_page_task(
    page_path: Path,
    root: Path,
    cropper: PageCropper,
    crop_bool: bool = True,
    crops_dir_name: str = "crops"
) -> int:
    """
    Returns number of crops written, 0 if layouts missing.
    """
    layouts_json = (
        page_path.parent.parent / "layouts" / "pages" / page_path.name.replace(page_path.suffix, ".json")
    )
    if not layouts_json.exists():
        return 0
    regions = load_regions(layouts_json)
    merged  = regions     #   <- public .run() returns List[Region]
    out_dir = page_path.parent.parent / crops_dir_name
    cropper.crop_page(page_path, page_path.stem, merged, crop_bool)

    return len(merged)

def main():
    ap = argparse.ArgumentParser("Crops pages once layouts exist")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--no_pad", dest="pad_bool", action="store_false")
    ap.add_argument("--pad_pct", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    cropper   = PageCropper(pad_pct=args.pad_pct)

    pages = list(iter_pages(args.root))
    log.info("Found %d page images", len(pages))

    t0 = time.time()
    written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
        tqdm(total=len(pages), desc="threads") as pbar:
        fut2page = {pool.submit(crop_page_task, p, args.root, cropper, args.pad_bool): p for p in pages}
        for fut in as_completed(fut2page):
            written += fut.result()
            pbar.update(1)
    log.info("Wrote %d crops in %.1fs", written, time.time() - t0)

if __name__ == "__main__":
    main()