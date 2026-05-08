"""
Hierarchy builder for layout JSON produced by doc2html.

Input: List[Region]    (already loaded with utils.io.load_regions)
Output: Same list, but every Region now has
    - parent_id (or None)
    - children_ids (List[int])

Criterion: A is a child of B -> IoA(A int B)/ Area (A) >= IoA_Threshold
           and Area(A) < Area(B)
           For ties pick the smallest B.
"""
from __future__ import annotations
from typing import List
import math

from doc2html.layout.datamodels.datamodels import Region
from doc2html.utils.geometry import (bbox_area, bbox_intersection, check_ioa)

def build_in_memory_hierarchy(regs: List[Region], ioa_thr: float = 0.95) -> List[Region]:
    """
    Return *new* Regions with parent / children filled.
    """

    regs = list(regs) # do not mutate caller's list
    areas = {r.id: bbox_area(r.xyxy) for r in regs} # pre-compute areas

    # candidate parents for each region
    cand: dict[str, list[Region]] = {r.id: [] for r in regs}
    for a in regs:
        for b in regs:
            if a.id == b.id:
                continue
            if areas[a.id] >= areas[b.id]:
                continue
            if check_ioa(a.xyxy, b.xyxy, ioa_thr):
                cand[a.id].append(b) # Candidate parents -> b

    # Choose smallest valid parent
    by_id = {r.id: r for r in regs}
    updates: dict[str, Region] = {}
    for child in regs:
        best = min(cand[child.id], key=lambda r: areas[r.id], default=None)
        if best:
            up_child = child.__dict__.copy()
            up_child["parent_id"] = best.id
            updates[child.id] = Region(**up_child)

    # Update parent's children id's
    for c_id, updated in list(updates.items()):
        p_id = updated.parent_id
        par = updates.get(p_id, by_id[p_id])
        tmp = par.children_ids + [c_id]
        updates[p_id] = Region(**par.__dict__ | {"children_ids": tmp})

    # Return merged view
    for k, v in updates.items():
        by_id[k] = v
    return list(by_id.values())