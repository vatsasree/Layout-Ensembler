from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import copy

from doc2html.layout.datamodels.datamodels import Region

RGB = Tuple[int, int, int]

def draw_regions(
    cv2_img: np.ndarray,
    region: List[Region],
    color_map: dict[str, RGB] | None = None,
    thickness: int | None = None
) -> np.ndarray:
    """
    Args:
        cv2_img: A single cv2_img
        regions: List of Regions Covered
        color_map: Dictionary of str -> RGB
        thickness: Thickness of the brush
    """
    out = copy.deepcopy(cv2_img)
    H, W = out.shape[:2]
    thickness = thickness or max(1, round(0.002 * (H + W))/2) # type: ignore
    color_map = color_map or {}

    for r in region:
        x1, y1, x2, y2 = map(int, r.xyxy)
        color = color_map.get(r.label.value, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, int(thickness)) # type: ignore
        label = f"{r.label.value}" if r.score else r.label.value
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        y_text = max(y1 - 4, th + 2)
        cv2.rectangle(out, (x1, y_text - th - 2), (x1 + tw + 4, y_text), color, -1)
        if not r.parent_id:
            cv2.putText(out, f"{label}_parent_{r.source}", (x1 + 4, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        else:
            cv2.putText(out, f"{label}_child_{r.source}", (x1 + 4, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return out