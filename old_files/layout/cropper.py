from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageOps
import math

from doc2html.datamodels.layout import LayoutCrop, WordAttr
from doc2html.layout.datamodels.datamodels import Region
from doc2html.utils.io import write_crop_meta

BBox = Tuple[int, int, int, int]

class PageCropper:
    """
    Crop a page into `LayoutCrop` objects.

    Parameters
    -----------
    pad_px : int | None
        Constant padding in *pixels* (added on all sides).
    pad_pct: float | None
        Relative padding (0.02 = 2 % of max(H, W)) if `pad_px` is None.

    Either pad_px or **or** pad_pct may be given not both.
    """
    def __init__(self, pad_px: int | None = None, pad_pct: float | None = 0.02):
        if pad_px is not None and pad_pct is not None:
            raise ValueError("Specify only pad_px or pad_pct, not both")
        self.pad_px = pad_px
        self.pad_pct = pad_pct

    # ---------------------------------------------------------
    def crop_page(
        self,
        page_img_path: Path,
        page_id: str,
        regions: List[Region],
        pad_bool: True, # type: ignore
        save_to_disk=True
    ) -> List[LayoutCrop]: # type: ignore
        img = Image.open(page_img_path).convert("RGB")
        W, H = img.size
        pad_px = self.pad_px or int(max(H, W) * self.pad_pct) # type: ignore

        crops: List[LayoutCrop] = []
        for reg in regions:
            x1, y1, x2, y2 = reg.xyxy
            crop_img = img.crop((x1, y1, x2, y2))
            if pad_px and pad_bool: 
                crop_img = ImageOps.expand(
                    crop_img,
                    border=pad_px,
                    fill="white"
                )
            crop_id  = reg.id
            rel_path = f"{crop_id}.png"
            layout_crop = LayoutCrop(
                id=crop_id,
                label=reg.label,
                bbox=(x1, y1, x2, y2),
                page_id=page_id,
                image_path=str(page_img_path.parent.parent / "crops" / "pages" / 
                               page_img_path.stem / rel_path) if save_to_disk else "",
                ocr_text=[],          # to be filled later 
                parent_id=reg.parent_id, 
                children_ids=reg.children_ids
            )
            if save_to_disk:
                # By default save to the crops folder of the parent
                out_dir = Path( page_img_path.parent.parent / "crops" / 
                               "pages" / page_img_path.stem ).mkdir(parents=True, exist_ok=True)
                crop_img.save( page_img_path.parent.parent / "crops" /  
                              "pages" / page_img_path.stem / rel_path , "PNG")

            crops.append(layout_crop)
        
        # Write the crops to this place
        write_crop_meta(crops, page_img_path.parent.parent / "crops")