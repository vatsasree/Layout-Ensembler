"""
Hierarchical region nesting for document layout trees.

After the ensemble produces a flat list of regions, this module
reconstructs the **containment hierarchy** — e.g. a ``paragraph`` inside
a ``sidebar``, or a ``figure-caption`` inside a ``figure``.

Algorithm
---------
1. Compute areas for all regions.
2. For each region **A**, find all regions **B** such that:
   - ``area(A) < area(B)``   (child must be smaller)
   - ``IoA(A, B) ≥ threshold``  (A is mostly inside B)
3. Among all valid parents, pick the **smallest** one (tightest fit).
4. Populate ``parent_id`` and ``children_ids`` fields accordingly.

This greedy approach runs in ``O(n²)`` which is acceptable for typical
page layouts (< 200 regions). For extremely dense layouts, spatial
indexing could be added.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from layout_ensembler.datamodels.regions import Region
from layout_ensembler.utils.geometry import bbox_area, check_ioa


def build_hierarchy(
    regions: List[Region],
    ioa_threshold: float = 0.95,
) -> List[Region]:
    """
    Build a containment tree over a flat list of regions.

    Parameters
    ----------
    regions : list[Region]
        Flat list of layout regions (typically from :class:`Ensembler`).
    ioa_threshold : float
        Minimum Intersection-over-Area(child, parent) to consider the
        child enclosed. Higher values require near-complete containment.

    Returns
    -------
    list[Region]
        New Region objects with ``parent_id`` and ``children_ids`` populated.
        The input list is *not* modified.
    """
    if not regions:
        return []

    # Work on copies so we don't mutate the caller's data
    regions = list(regions)
    areas: Dict[str, float] = {r.id: bbox_area(r.xyxy) for r in regions}

    # ── Step 1: Find candidate parents for each region ────────────
    candidates: Dict[str, List[Region]] = {r.id: [] for r in regions}

    for child in regions:
        for parent in regions:
            if child.id == parent.id:
                continue
            if areas[child.id] >= areas[parent.id]:
                continue  # parent must be strictly larger
            if check_ioa(child.xyxy, parent.xyxy, ioa_threshold):
                candidates[child.id].append(parent)

    # ── Step 2: Pick the tightest (smallest) parent ───────────────
    by_id: Dict[str, Region] = {r.id: r for r in regions}
    updates: Dict[str, Region] = {}

    for child in regions:
        best_parent: Optional[Region] = min(
            candidates[child.id],
            key=lambda r: areas[r.id],
            default=None,
        )
        if best_parent is not None:
            child_dict = child.__dict__.copy()
            child_dict["parent_id"] = best_parent.id
            updates[child.id] = Region(**child_dict)

    # ── Step 3: Backfill children_ids on parents ──────────────────
    for child_id, updated_child in list(updates.items()):
        parent_id = updated_child.parent_id
        parent = updates.get(parent_id, by_id[parent_id])
        parent_dict = parent.__dict__.copy()
        parent_dict["children_ids"] = parent.children_ids + [child_id]
        updates[parent_id] = Region(**parent_dict)

    # ── Merge updates back ────────────────────────────────────────
    for k, v in updates.items():
        by_id[k] = v

    return list(by_id.values())
