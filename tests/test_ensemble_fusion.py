"""Tests for oriented box fusion logic."""

import numpy as np
import pytest

from rotary_target.ensemble.fusion import FusionConfig, FusedPrediction, OrientedBoxFusion
from rotary_target.ensemble.io import Prediction


def _pred(cx, cy, w, h, theta, score, cls=0):
    return Prediction(cx=cx, cy=cy, w=w, h=h, theta=theta, score=score, class_id=cls)


class TestOrientedBoxFusion:
    def test_empty_input(self):
        fusion = OrientedBoxFusion()
        assert fusion.fuse([]) == []

    def test_single_model_passthrough(self):
        preds = [_pred(0, 0, 4, 4, 0, 0.9)]
        result = OrientedBoxFusion().fuse([preds])
        assert len(result) == 1
        assert result[0].cx == pytest.approx(0)
        assert result[0].score == pytest.approx(0.9, abs=0.01)

    def test_identical_predictions_fuse(self):
        box = _pred(5, 5, 10, 4, 0.3, 0.8)
        result = OrientedBoxFusion().fuse([[box], [box], [box]])
        assert len(result) == 1
        assert result[0].cx == pytest.approx(5)
        assert result[0].cy == pytest.approx(5)
        assert result[0].source_count == 3

    def test_non_overlapping_stay_separate(self):
        box1 = _pred(0, 0, 2, 2, 0, 0.9)
        box2 = _pred(100, 100, 2, 2, 0, 0.8)
        result = OrientedBoxFusion().fuse([[box1, box2]])
        assert len(result) == 2

    def test_score_penalty_single_model(self):
        box = _pred(0, 0, 4, 4, 0, 0.9)
        result = OrientedBoxFusion().fuse([[box], [], []])
        assert len(result) == 1
        assert result[0].score < 0.9 * 0.5

    def test_circular_mean_angles(self):
        # 10 degrees and 350 degrees (-10 degrees) should average to ~0
        box1 = _pred(0, 0, 4, 4, np.radians(5), 0.9)
        box2 = _pred(0, 0, 4, 4, np.radians(-5), 0.9)
        result = OrientedBoxFusion().fuse([[box1], [box2]])
        assert len(result) == 1
        assert abs(result[0].theta) < np.radians(2)

    def test_model_weights_affect_center(self):
        box1 = _pred(0, 0, 4, 4, 0, 0.9)
        box2 = _pred(1, 0, 4, 4, 0, 0.9)
        config = FusionConfig(iou_threshold=0.3)
        result = OrientedBoxFusion(config).fuse(
            [[box1], [box2]], model_weights=[3.0, 1.0]
        )
        assert len(result) == 1
        assert result[0].cx < 0.5

    def test_different_classes_separate(self):
        box1 = _pred(0, 0, 4, 4, 0, 0.9, cls=0)
        box2 = _pred(0, 0, 4, 4, 0, 0.9, cls=1)
        result = OrientedBoxFusion().fuse([[box1, box2]])
        assert len(result) == 2
