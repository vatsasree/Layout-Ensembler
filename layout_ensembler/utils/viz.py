"""
Visualization utilities for layout detection results.

Draws bounding boxes, labels, and hierarchy information onto document
page images for debugging and qualitative evaluation.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from layout_ensembler.datamodels.regions import Region

RGB = Tuple[int, int, int]

# ---------------------------------------------------------------------------
# Default color palette — visually distinct, muted tones
# ---------------------------------------------------------------------------
_DEFAULT_PALETTE: Dict[str, RGB] = {
    "paragraph": (70, 180, 120),
    "header": (220, 100, 60),
    "footer": (200, 130, 80),
    "title": (50, 100, 220),
    "section-title": (80, 130, 200),
    "table": (180, 60, 180),
    "figure": (60, 180, 180),
    "picture": (60, 180, 180),
    "caption": (180, 180, 60),
    "figure-caption": (180, 180, 60),
    "table-caption": (180, 160, 80),
    "formula": (140, 80, 200),
    "list-item": (100, 200, 100),
    "ordered-list": (90, 190, 110),
    "unordered-list": (110, 190, 90),
    "page-number": (160, 160, 160),
    "footnote": (130, 130, 200),
    "headline": (40, 80, 200),
    "signature": (200, 60, 60),
    "handwriting": (200, 120, 60),
    "code": (100, 100, 100),
    "form": (160, 100, 160),
    "sidebar": (100, 160, 160),
}


def draw_regions(
    cv2_img: np.ndarray,
    regions: List[Region],
    color_map: Optional[Dict[str, RGB]] = None,
    thickness: Optional[int] = None,
    show_score: bool = True,
    show_hierarchy: bool = True,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes and labels for all regions onto an image.

    Parameters
    ----------
    cv2_img : np.ndarray
        BGR image (will not be modified in-place).
    regions : list[Region]
        Detected regions to draw.
    color_map : dict[str, RGB] | None
        Per-label colors. Falls back to ``_DEFAULT_PALETTE`` and finally
        to green ``(0, 255, 0)`` for unknown labels.
    thickness : int | None
        Box line thickness. Auto-scaled from image size if ``None``.
    show_score : bool
        Whether to include confidence scores in the label text.
    show_hierarchy : bool
        Whether to annotate parent/child relationship in the label.
    font_scale : float
        OpenCV font scale for label text.

    Returns
    -------
    np.ndarray
        Annotated BGR image (deep copy of input).
    """
    out = copy.deepcopy(cv2_img)
    H, W = out.shape[:2]
    thickness = thickness or max(1, int(0.002 * (H + W) / 2))
    palette = {**_DEFAULT_PALETTE, **(color_map or {})}

    for r in regions:
        x1, y1, x2, y2 = map(int, r.xyxy)
        label_str = r.label.value
        color = palette.get(label_str, (0, 255, 0))

        # ── Bounding box ──────────────────────────────────────────
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # ── Label text ────────────────────────────────────────────
        parts = [label_str]
        if show_score and r.score is not None:
            parts.append(f"{r.score:.2f}")
        if show_hierarchy:
            if r.parent_id:
                parts.append(f"child|{r.source}")
            else:
                parts.append(f"root|{r.source}")
        text = " ".join(parts)

        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        y_text = max(y1 - 4, th + 2)

        # Background rectangle for readability
        cv2.rectangle(
            out,
            (x1, y_text - th - 4),
            (x1 + tw + 6, y_text + 2),
            color,
            -1,
        )
        cv2.putText(
            out,
            text,
            (x1 + 3, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def draw_legend(
    cv2_img: np.ndarray,
    labels: List[str],
    color_map: Optional[Dict[str, RGB]] = None,
    position: str = "top-right",
) -> np.ndarray:
    """
    Draw a color legend on the image for the given labels.

    Parameters
    ----------
    cv2_img : np.ndarray
        Image to annotate (will be deep-copied).
    labels : list[str]
        Label names to include in the legend.
    color_map : dict | None
        Color mapping; falls back to ``_DEFAULT_PALETTE``.
    position : str
        One of ``"top-right"``, ``"top-left"``.

    Returns
    -------
    np.ndarray
    """
    out = copy.deepcopy(cv2_img)
    H, W = out.shape[:2]
    palette = {**_DEFAULT_PALETTE, **(color_map or {})}

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    line_h = 22
    pad = 8
    swatch_w = 14

    # Measure max text width
    max_tw = 0
    for lbl in labels:
        (tw, _), _ = cv2.getTextSize(lbl, font, font_scale, 1)
        max_tw = max(max_tw, tw)

    legend_w = pad + swatch_w + pad + max_tw + pad
    legend_h = pad + line_h * len(labels) + pad

    if position == "top-right":
        x0 = W - legend_w - 10
        y0 = 10
    else:
        x0 = 10
        y0 = 10

    # Semi-transparent background
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + legend_w, y0 + legend_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    for i, lbl in enumerate(labels):
        color = palette.get(lbl, (0, 255, 0))
        cy = y0 + pad + i * line_h + line_h // 2
        # Color swatch
        cv2.rectangle(
            out,
            (x0 + pad, cy - swatch_w // 2),
            (x0 + pad + swatch_w, cy + swatch_w // 2),
            color,
            -1,
        )
        # Text
        cv2.putText(
            out,
            lbl,
            (x0 + pad + swatch_w + pad, cy + 4),
            font,
            font_scale,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    return out
