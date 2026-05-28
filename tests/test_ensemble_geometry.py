"""Tests for oriented box geometry utilities."""

import numpy as np
import pytest

from rotary_target.ensemble.geometry import obb_to_corners, rotated_iou


class TestObbToCorners:
    def test_axis_aligned(self):
        corners = obb_to_corners(0, 0, 4, 2, 0)
        assert corners.shape == (4, 2)
        xs = sorted(corners[:, 0])
        ys = sorted(corners[:, 1])
        assert xs[0] == pytest.approx(-2)
        assert xs[-1] == pytest.approx(2)
        assert ys[0] == pytest.approx(-1)
        assert ys[-1] == pytest.approx(1)

    def test_center_offset(self):
        corners = obb_to_corners(10, 20, 4, 2, 0)
        center = corners.mean(axis=0)
        np.testing.assert_array_almost_equal(center, [10, 20])

    def test_rotation_preserves_area(self):
        c1 = obb_to_corners(0, 0, 6, 3, 0)
        c2 = obb_to_corners(0, 0, 6, 3, np.pi / 4)
        area1 = _shoelace(c1)
        area2 = _shoelace(c2)
        assert area1 == pytest.approx(18.0, rel=1e-6)
        assert area2 == pytest.approx(18.0, rel=1e-6)


class TestRotatedIoU:
    def test_identical_boxes(self):
        box = (5, 5, 10, 4, 0.3)
        assert rotated_iou(box, box) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        box1 = (0, 0, 2, 2, 0)
        box2 = (100, 100, 2, 2, 0)
        assert rotated_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap_axis_aligned(self):
        box1 = (0, 0, 4, 4, 0)
        box2 = (2, 0, 4, 4, 0)
        iou = rotated_iou(box1, box2)
        # overlap = 2x4=8, union = 16+16-8=24, iou=8/24=1/3
        assert iou == pytest.approx(1.0 / 3.0, abs=0.02)

    def test_symmetric(self):
        box1 = (1, 2, 6, 3, 0.5)
        box2 = (3, 4, 5, 4, -0.3)
        assert rotated_iou(box1, box2) == pytest.approx(
            rotated_iou(box2, box1), abs=1e-6
        )


def _shoelace(corners: np.ndarray) -> float:
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
