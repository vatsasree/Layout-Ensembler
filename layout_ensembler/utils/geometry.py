"""
Bounding-box geometry primitives.

All functions operate on ``(x1, y1, x2, y2)`` integer tuples where
``(x1, y1)`` is the top-left corner and ``(x2, y2)`` is the bottom-right.
"""

from __future__ import annotations

from typing import Tuple

BBox = Tuple[int, int, int, int]


def bbox_area(b: BBox) -> float:
    """
    Compute the area of a bounding box.

    Returns 0 for degenerate (zero-width or zero-height) boxes.
    """
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_intersection(a: BBox, b: BBox) -> float:
    """
    Compute the area of the intersection rectangle between two boxes.

    Returns 0 if the boxes do not overlap.
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return bbox_area((x1, y1, x2, y2))


def compute_iou(a: BBox, b: BBox) -> float:
    """
    Intersection-over-Union between two bounding boxes.

    .. math::

        \\text{IoU}(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}

    Returns 0.0 if either box has zero area.
    """
    inter = bbox_intersection(a, b)
    if inter == 0:
        return 0.0
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def check_ioa(inner: BBox, outer: BBox, threshold: float) -> bool:
    """
    Check whether ``inner`` is *mostly contained* in ``outer``.

    Intersection-over-Area of the *inner* box:

    .. math::

        \\text{IoA}(A, B) = \\frac{|A \\cap B|}{|A|}

    Parameters
    ----------
    inner : BBox
        The potentially enclosed box.
    outer : BBox
        The candidate parent box.
    threshold : float
        Minimum IoA ratio to consider ``inner`` enclosed.

    Returns
    -------
    bool
    """
    inter = bbox_intersection(inner, outer)
    area_inner = bbox_area(inner)
    return area_inner > 0 and (inter / area_inner) >= threshold


def clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    """Clip a bounding box to image dimensions."""
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )
