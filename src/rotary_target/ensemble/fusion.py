"""Oriented Weighted Box Fusion for multi-model ensemble inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .geometry import rotated_iou
from .io import Prediction


@dataclass(frozen=True)
class FusionConfig:
    iou_threshold: float = 0.5
    score_threshold: float = 0.05
    model_weights: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class FusedPrediction:
    cx: float
    cy: float
    w: float
    h: float
    theta: float
    score: float
    class_id: int
    source_count: int


class OrientedBoxFusion:
    """Fuse oriented box predictions from multiple models."""

    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()

    def fuse(
        self,
        model_predictions: list[list[Prediction]],
        model_weights: list[float] | None = None,
    ) -> list[FusedPrediction]:
        n_models = len(model_predictions)
        if n_models == 0:
            return []

        weights = model_weights or self.config.model_weights
        if not weights:
            weights = [1.0] * n_models

        all_classes: set[int] = set()
        for preds in model_predictions:
            for p in preds:
                all_classes.add(p.class_id)

        results: list[FusedPrediction] = []
        for cls_id in sorted(all_classes):
            cls_results = self._fuse_class(
                model_predictions, cls_id, weights, n_models
            )
            results.extend(cls_results)
        return results

    def _fuse_class(
        self,
        model_predictions: list[list[Prediction]],
        cls_id: int,
        weights: list[float],
        n_models: int,
    ) -> list[FusedPrediction]:
        tagged: list[tuple[Prediction, float]] = []
        for model_idx, preds in enumerate(model_predictions):
            for p in preds:
                if p.class_id == cls_id and p.score >= self.config.score_threshold:
                    tagged.append((p, weights[model_idx]))

        if not tagged:
            return []

        tagged.sort(key=lambda x: x[0].score, reverse=True)
        used = [False] * len(tagged)
        clusters: list[list[int]] = []

        for i in range(len(tagged)):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            box_i = tagged[i][0].as_tuple()
            for j in range(i + 1, len(tagged)):
                if used[j]:
                    continue
                box_j = tagged[j][0].as_tuple()
                if rotated_iou(box_i, box_j) >= self.config.iou_threshold:
                    cluster.append(j)
                    used[j] = True
            clusters.append(cluster)

        results: list[FusedPrediction] = []
        for cluster in clusters:
            fused = self._fuse_cluster(tagged, cluster, n_models, cls_id)
            results.append(fused)
        return results

    def _fuse_cluster(
        self,
        tagged: list[tuple[Prediction, float]],
        indices: list[int],
        n_models: int,
        cls_id: int,
    ) -> FusedPrediction:
        total_w = 0.0
        cx_sum = cy_sum = w_sum = h_sum = 0.0
        sin_sum = cos_sum = 0.0
        score_sum = 0.0

        for idx in indices:
            pred, model_w = tagged[idx]
            sw = model_w * pred.score
            total_w += sw
            cx_sum += sw * pred.cx
            cy_sum += sw * pred.cy
            w_sum += sw * pred.w
            h_sum += sw * pred.h
            sin_sum += sw * np.sin(2.0 * pred.theta)
            cos_sum += sw * np.cos(2.0 * pred.theta)
            score_sum += model_w * pred.score

        inv_w = 1.0 / max(total_w, 1e-10)
        theta_fused = 0.5 * np.arctan2(sin_sum, cos_sum)
        n_matched = len(indices)
        score_penalty = min(n_matched, n_models) / n_models
        fused_score = (score_sum / sum(tagged[i][1] for i in indices)) * score_penalty

        return FusedPrediction(
            cx=cx_sum * inv_w,
            cy=cy_sum * inv_w,
            w=w_sum * inv_w,
            h=h_sum * inv_w,
            theta=float(theta_fused),
            score=float(fused_score),
            class_id=cls_id,
            source_count=n_matched,
        )
