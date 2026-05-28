"""Oriented bounding box geometry utilities (pure numpy, no external deps)."""

from __future__ import annotations

import numpy as np


def obb_to_corners(cx: float, cy: float, w: float, h: float, theta: float) -> np.ndarray:
    """Convert OBB (cx, cy, w, h, theta) to 4 corner points.

    Returns ndarray of shape (4, 2) in order: top-left, top-right, bottom-right, bottom-left
    relative to the box's local frame before rotation.
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = w / 2.0
    dy = h / 2.0
    corners_local = np.array([
        [-dx, -dy],
        [dx, -dy],
        [dx, dy],
        [-dx, dy],
    ])
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners = corners_local @ R.T + np.array([cx, cy])
    return corners


def _polygon_area(vertices: np.ndarray) -> float:
    """Shoelace formula for polygon area. Vertices shape (N, 2)."""
    n = len(vertices)
    if n < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _clip_polygon_by_edge(
    polygon: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    """Sutherland-Hodgman: clip polygon by one edge (p1->p2, keeps left side)."""
    if len(polygon) == 0:
        return polygon
    edge = p2 - p1
    output = []
    for i in range(len(polygon)):
        current = polygon[i]
        prev = polygon[i - 1]
        curr_side = float(np.cross(edge, current - p1))
        prev_side = float(np.cross(edge, prev - p1))
        if curr_side >= 0:
            if prev_side < 0:
                t = prev_side / (prev_side - curr_side)
                output.append(prev + t * (current - prev))
            output.append(current)
        elif prev_side >= 0:
            t = prev_side / (prev_side - curr_side)
            output.append(prev + t * (current - prev))
    return np.array(output) if output else np.empty((0, 2))


def polygon_intersection_area(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """Compute intersection area of two convex polygons via Sutherland-Hodgman."""
    clipped = corners1.copy()
    n2 = len(corners2)
    for i in range(n2):
        if len(clipped) == 0:
            return 0.0
        p1 = corners2[i]
        p2 = corners2[(i + 1) % n2]
        clipped = _clip_polygon_by_edge(clipped, p1, p2)
    return _polygon_area(clipped)


def rotated_iou(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two OBBs. Each box is (cx, cy, w, h, theta)."""
    corners1 = obb_to_corners(*box1)
    corners2 = obb_to_corners(*box2)
    inter = polygon_intersection_area(corners1, corners2)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def rotated_iou_matrix(boxes1: list[tuple], boxes2: list[tuple]) -> np.ndarray:
    """Pairwise rotated IoU matrix. Returns shape (len(boxes1), len(boxes2))."""
    m, n = len(boxes1), len(boxes2)
    iou_mat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            iou_mat[i, j] = rotated_iou(boxes1[i], boxes2[j])
    return iou_mat
