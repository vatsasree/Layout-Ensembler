"""Tests for hierarchical region nesting."""

import pytest

from layout_ensembler.datamodels.regions import Region, RegionType
from layout_ensembler.hierarchy import build_hierarchy


def _make_region(xyxy, label="paragraph", rid=None):
    return Region(
        xyxy=xyxy,
        label=RegionType(label),
        score=0.9,
        source="test",
        id=rid or None,  # let it auto-generate if None
    )


class TestBuildHierarchy:

    def test_no_nesting(self):
        """Non-overlapping regions should have no parent/child links."""
        regions = [
            _make_region((0, 0, 100, 100)),
            _make_region((200, 200, 300, 300)),
        ]
        result = build_hierarchy(regions, ioa_threshold=0.9)
        for r in result:
            assert r.parent_id is None
            assert r.children_ids == []

    def test_child_inside_parent(self):
        """A small region fully inside a large one should become its child."""
        parent = _make_region((0, 0, 500, 500), "figure", rid="parent")
        child = _make_region((50, 50, 200, 200), "caption", rid="child")

        result = build_hierarchy([parent, child], ioa_threshold=0.9)
        by_id = {r.id: r for r in result}

        assert by_id["child"].parent_id == "parent"
        assert "child" in by_id["parent"].children_ids

    def test_tightest_parent_selected(self):
        """When nested, the smallest valid parent should be chosen."""
        outer = _make_region((0, 0, 1000, 1000), "figure", rid="outer")
        middle = _make_region((10, 10, 500, 500), "sidebar", rid="middle")
        inner = _make_region((20, 20, 100, 100), "paragraph", rid="inner")

        result = build_hierarchy([outer, middle, inner], ioa_threshold=0.9)
        by_id = {r.id: r for r in result}

        # inner should pick middle (tighter), not outer
        assert by_id["inner"].parent_id == "middle"

    def test_empty_input(self):
        assert build_hierarchy([]) == []

    def test_single_region(self):
        regions = [_make_region((10, 10, 100, 100))]
        result = build_hierarchy(regions)
        assert len(result) == 1
        assert result[0].parent_id is None

    def test_does_not_mutate_input(self):
        """Input list should not be modified."""
        regions = [
            _make_region((0, 0, 500, 500), rid="parent"),
            _make_region((50, 50, 200, 200), rid="child"),
        ]
        original_ids = [r.id for r in regions]
        original_parents = [r.parent_id for r in regions]

        build_hierarchy(regions, ioa_threshold=0.9)

        assert [r.id for r in regions] == original_ids
        assert [r.parent_id for r in regions] == original_parents

    def test_equal_area_no_parent(self):
        """Regions of equal area should not nest into each other."""
        r1 = _make_region((0, 0, 100, 100), rid="a")
        r2 = _make_region((0, 0, 100, 100), rid="b")

        result = build_hierarchy([r1, r2], ioa_threshold=0.9)
        for r in result:
            assert r.parent_id is None

    def test_partial_overlap_below_threshold(self):
        """Partial overlap below the IoA threshold should not create nesting."""
        outer = _make_region((0, 0, 200, 200), rid="outer")
        partial = _make_region((150, 150, 300, 300), rid="partial")

        result = build_hierarchy([outer, partial], ioa_threshold=0.9)
        by_id = {r.id: r for r in result}

        assert by_id["partial"].parent_id is None
