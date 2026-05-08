"""
Page cropping — extract individual region images from full pages.

After layout detection and hierarchy construction, this module crops each
detected region from the page image, optionally adds padding, and persists
the crops alongside their metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageOps

from layout_ensembler.datamodels.layout import LayoutCrop
from layout_ensembler.datamodels.regions import Region
from layout_ensembler.utils.io_utils import write_crop_meta

log = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


class PageCropper:
    """
    Crop detected regions from a page image.

    Supports both **pixel-based** and **percentage-based** padding to
    provide context around each region for downstream OCR or classification.

    Parameters
    ----------
    pad_px : int | None
        Fixed padding in pixels (applied to all four sides).
    pad_pct : float | None
        Relative padding as a fraction of ``max(H, W)`` of the page.
        Default is 2%. Only used if ``pad_px`` is ``None``.
    """

    def __init__(
        self,
        pad_px: Optional[int] = None,
        pad_pct: Optional[float] = 0.02,
    ) -> None:
        if pad_px is not None and pad_pct is not None:
            raise ValueError("Specify only pad_px or pad_pct, not both.")
        self.pad_px = pad_px
        self.pad_pct = pad_pct if pad_px is None else None

    def crop_page(
        self,
        page_img_path: Path,
        page_id: str,
        regions: List[Region],
        apply_padding: bool = True,
        save_to_disk: bool = True,
    ) -> List[LayoutCrop]:
        """
        Crop all regions from a page image.

        Parameters
        ----------
        page_img_path : Path
            Path to the full page image.
        page_id : str
            Unique identifier for this page.
        regions : list[Region]
            Regions to crop.
        apply_padding : bool
            Whether to add padding around each crop.
        save_to_disk : bool
            Whether to persist crop images as PNG files.

        Returns
        -------
        list[LayoutCrop]
            Crop metadata objects (with image paths if saved).
        """
        img = Image.open(page_img_path).convert("RGB")
        W, H = img.size
        pad_px = self.pad_px or int(max(H, W) * (self.pad_pct or 0))

        crops_base = page_img_path.parent.parent / "crops" / "pages" / page_img_path.stem

        crops: List[LayoutCrop] = []
        for reg in regions:
            x1, y1, x2, y2 = reg.xyxy
            crop_img = img.crop((x1, y1, x2, y2))

            if apply_padding and pad_px > 0:
                crop_img = ImageOps.expand(crop_img, border=pad_px, fill="white")

            rel_path = f"{reg.id}.png"

            image_path = ""
            if save_to_disk:
                out_dir = crops_base
                out_dir.mkdir(parents=True, exist_ok=True)
                crop_img.save(out_dir / rel_path, "PNG")
                image_path = str(out_dir / rel_path)

            layout_crop = LayoutCrop(
                id=reg.id,
                label=reg.label.value,
                bbox=(x1, y1, x2, y2),
                page_id=page_id,
                image_path=image_path,
                ocr_text=[],
                parent_id=reg.parent_id,
                children_ids=reg.children_ids,
            )
            crops.append(layout_crop)

        # Persist metadata
        if save_to_disk and crops:
            meta_root = page_img_path.parent.parent / "crops"
            write_crop_meta(crops, meta_root)

        return crops
