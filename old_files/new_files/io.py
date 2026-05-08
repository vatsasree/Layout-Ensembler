from __future__ import annotations
from pathlib import Path
from typing import List, Union
import json
import logging

from doc2html.layout.datamodels.datamodels import Region, RegionType
from doc2html.datamodels.layout import LayoutCrop 

log = logging.getLogger("io")

def write_json_safe(path: Path, data):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.error("Failed writing %s: %s", path, e)

def load_regions(json_path: Path) -> List[Region]:
    """
    Robustly load a list[Region] that was written either as

        [ {"xyxy": [...], "label": "paragraph", ...}, ... ]

    or as

        [ "{\"xyxy\": [...], \"label\": \"paragraph\", ...}", ... ]
    """
    with json_path.open() as f:
        raw = json.load(f)

    regions: list[Region] = []
    for item in raw:
        if isinstance(item, str):
            # Case 2: string-encoded Region
            regions.append(Region.from_json(item))
        elif isinstance(item, dict):
            # Case 1: dict (label still needs enum conversion)
            item["label"] = RegionType(item["label"])
            regions.append(Region(**item))
        else:
            raise TypeError(f"Unsupported element in {json_path}: {type(item)}")
    return regions

def load_layout_crops(crops_dir: Path) -> List[LayoutCrop]:
    crops: List[LayoutCrop] = []
    for j in crops_dir.glob("*.json"):
        with j.open() as f:
            d = json.load(f)
            # one crop per file - adjust if you choose a different format
            d["label"] = RegionType(d["label"])
            crops.append(LayoutCrop(**d))
    return crops

def write_crop_meta(
    crops: Union[LayoutCrop, List[LayoutCrop]],
    root: Path,
    one_file_per_crop: bool = True,
    indent: int | None = 2,
):
    """
    Persist LayoutCrop metadata to disk
    """
    if isinstance(crops, LayoutCrop):
        crops = [crops]

    if one_file_per_crop:
        root.mkdir(parents=True, exist_ok=True)
        for crop in crops:
            out = root / "pages" / Path(crop.page_id).stem / f"{crop.id}.json"
            out.write_text(crop.model_dump_json(indent=indent))
        return root