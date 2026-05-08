"""Tests for bounding-box geometry utilities."""

import pytest

from layout_ensembler.utils.geometry import (
    bbox_area,
    bbox_intersection,
    check_ioa,
    clip_bbox,
    compute_iou,
)


class TestBBoxArea:

    def test_normal_box(self):
        assert bbox_area((0, 0, 100, 200)) == 20000

    def test_unit_box(self):
        assert bbox_area((0, 0, 1, 1)) == 1

    def test_degenerate_zero_width(self):
        assert bbox_area((50, 50, 50, 100)) == 0

    def test_degenerate_inverted(self):
        assert bbox_area((100, 100, 50, 50)) == 0


class TestBBoxIntersection:

    def test_full_overlap(self):
        assert bbox_intersection((0, 0, 100, 100), (0, 0, 100, 100)) == 10000

    def test_partial_overlap(self):
        assert bbox_intersection((0, 0, 100, 100), (50, 50, 150, 150)) == 2500

    def test_no_overlap(self):
        assert bbox_intersection((0, 0, 50, 50), (100, 100, 200, 200)) == 0

    def test_touching_edge(self):
        assert bbox_intersection((0, 0, 50, 50), (50, 0, 100, 50)) == 0

    def test_containment(self):
        assert bbox_intersection((25, 25, 75, 75), (0, 0, 100, 100)) == 2500


class TestComputeIoU:

    def test_identical_boxes(self):
        iou = compute_iou((0, 0, 100, 100), (0, 0, 100, 100))
        assert iou == pytest.approx(1.0)

    def test_no_overlap(self):
        iou = compute_iou((0, 0, 50, 50), (100, 100, 200, 200))
        assert iou == 0.0

    def test_partial_overlap(self):
        iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        # Intersection = 50*50 = 2500, Union = 10000 + 10000 - 2500 = 17500
        assert iou == pytest.approx(2500 / 17500)

    def test_containment(self):
        iou = compute_iou((25, 25, 75, 75), (0, 0, 100, 100))
        # Intersection = 2500, Union = 10000
        assert iou == pytest.approx(0.25)


class TestCheckIoA:

    def test_fully_inside(self):
        assert check_ioa((25, 25, 75, 75), (0, 0, 100, 100), 0.9) is True

    def test_partially_inside(self):
        # Inner is 100x100, 50x50 overlap → IoA = 0.25
        assert check_ioa((50, 50, 150, 150), (0, 0, 100, 100), 0.3) is False
        assert check_ioa((50, 50, 150, 150), (0, 0, 100, 100), 0.2) is True

    def test_no_overlap(self):
        assert check_ioa((200, 200, 300, 300), (0, 0, 100, 100), 0.1) is False

    def test_threshold_boundary(self):
        # IoA = 5000/10000 = 0.5 exactly. check_ioa uses >=, so 0.5 >= 0.5 is True
        assert check_ioa((0, 0, 100, 100), (50, 0, 200, 100), 0.5) is True
        assert check_ioa((0, 0, 100, 100), (50, 0, 200, 100), 0.51) is False


class TestClipBBox:

    def test_already_inside(self):
        assert clip_bbox((10, 20, 90, 80), 100, 100) == (10, 20, 90, 80)

    def test_clip_negative(self):
        assert clip_bbox((-10, -20, 50, 50), 100, 100) == (0, 0, 50, 50)

    def test_clip_overflow(self):
        assert clip_bbox((10, 20, 200, 300), 100, 100) == (10, 20, 100, 100)
