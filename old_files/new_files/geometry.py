from typing import Tuple, List

BBox = Tuple[int, int, int, int]

def bbox_area(b: BBox) -> float:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def bbox_intersection(a: BBox, b: BBox) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    return bbox_area((x1, y1, x2, y2))

def check_ioa(a: BBox, b: BBox, thr: float) -> bool:
    inter = bbox_intersection(a, b)
    area_a = bbox_area(a)
    return area_a > 0 and inter / area_a > thr