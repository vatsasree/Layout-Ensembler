"""
I/O utilities for reading and writing layout detection results.

Handles both dict-based and string-encoded JSON formats for robustness
against different serialization strategies used during development.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Union

from layout_ensembler.datamodels.regions import Region, RegionType
from layout_ensembler.datamodels.layout import LayoutCrop

log = logging.getLogger(__name__)


def write_json_safe(path: Path, data) -> None:
    """
    Atomically write JSON data to *path*, creating parent directories.

    Logs errors instead of raising so that a single write failure does not
    crash a large batch run.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.error("Failed writing %s: %s", path, e)


def load_regions(json_path: Path) -> List[Region]:
    """
    Load a ``list[Region]`` from a JSON file.

    Supports two on-disk formats:

    1. **Dict list** — ``[{"xyxy": [...], "label": "paragraph", ...}, ...]``
    2. **String list** — ``['{"xyxy": [...], ...}', ...]``

    Parameters
    ----------
    json_path : Path
        Path to the JSON file.

    Returns
    -------
    list[Region]
    """
    with json_path.open() as f:
        raw = json.load(f)

    regions: list[Region] = []
    for item in raw:
        if isinstance(item, str):
            regions.append(Region.from_json(item))
        elif isinstance(item, dict):
            regions.append(Region.from_dict(item))
        else:
            raise TypeError(f"Unsupported element type in {json_path}: {type(item)}")
    return regions


def save_regions(regions: List[Region], json_path: Path) -> None:
    """
    Save a ``list[Region]`` to a JSON file.

    Uses the dict format for human readability.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in regions]
    json_path.write_text(json.dumps(data, indent=2))


def write_crop_meta(
    crops: Union[LayoutCrop, List[LayoutCrop]],
    root: Path,
    one_file_per_crop: bool = True,
    indent: int | None = 2,
) -> Path:
    """
    Persist :class:`LayoutCrop` metadata to disk.

    Parameters
    ----------
    crops : LayoutCrop | list[LayoutCrop]
        One or more crops to write.
    root : Path
        Base directory for crop metadata.
    one_file_per_crop : bool
        If ``True``, writes one JSON file per crop. Otherwise writes a
        single ``crops.json`` containing all crops.
    indent : int | None
        JSON indentation level.

    Returns
    -------
    Path
        The root directory where files were written.
    """
    if isinstance(crops, LayoutCrop):
        crops = [crops]

    root.mkdir(parents=True, exist_ok=True)

    if one_file_per_crop:
        for crop in crops:
            out = root / "pages" / Path(crop.page_id).stem / f"{crop.id}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(crop.model_dump_json(indent=indent))
    else:
        out = root / "crops.json"
        data = [json.loads(c.model_dump_json()) for c in crops]
        out.write_text(json.dumps(data, indent=indent))

    return root
