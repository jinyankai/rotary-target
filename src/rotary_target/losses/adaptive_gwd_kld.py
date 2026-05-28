"""Adaptive GWD-KLD hybrid loss for rotated object detection.

A plug-and-play loss module that blends Gaussian Wasserstein Distance and
Kullback-Leibler Divergence based on predicted aspect ratio. Works with any
OBB detector that regresses (cx, cy, w, h, theta).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def obb_to_gaussian(
    cx: float, cy: float, w: float, h: float, theta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert oriented bounding box to 2D Gaussian (mu, Sigma)."""
    mu = np.array([cx, cy])
    R = rotation_matrix(theta)
    D = np.diag([w**2 / 12.0, h**2 / 12.0])
    sigma = R @ D @ R.T
    return mu, sigma


def _matrix_sqrt_2x2(M: np.ndarray) -> np.ndarray:
    """Closed-form square root for a 2x2 positive-definite matrix."""
    tr = M[0, 0] + M[1, 1]
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    s = np.sqrt(max(det, 1e-10))
    denom = np.sqrt(max(tr + 2.0 * s, 1e-10))
    return (M + s * np.eye(2)) / denom


def gwd_loss(
    mu_p: np.ndarray, sigma_p: np.ndarray,
    mu_t: np.ndarray, sigma_t: np.ndarray,
) -> float:
    """Squared Wasserstein-2 distance between two 2D Gaussians."""
    diff = mu_p - mu_t
    center_term = float(diff @ diff)

    sqrt_p = _matrix_sqrt_2x2(sigma_p)
    inner = sqrt_p @ sigma_t @ sqrt_p
    sqrt_inner = _matrix_sqrt_2x2(inner)

    shape_term = float(
        np.trace(sigma_p) + np.trace(sigma_t) - 2.0 * np.trace(sqrt_inner)
    )
    return center_term + max(shape_term, 0.0)


def kld_loss_symmetric(
    mu_p: np.ndarray, sigma_p: np.ndarray,
    mu_t: np.ndarray, sigma_t: np.ndarray,
) -> float:
    """Symmetrized KL divergence between two 2D Gaussians."""
    return 0.5 * (_kld_one_way(mu_p, sigma_p, mu_t, sigma_t)
                  + _kld_one_way(mu_t, sigma_t, mu_p, sigma_p))


def _kld_one_way(
    mu_p: np.ndarray, sigma_p: np.ndarray,
    mu_t: np.ndarray, sigma_t: np.ndarray,
) -> float:
    """KL(P || T) for 2D Gaussians."""
    sigma_t_inv = np.linalg.inv(sigma_t + 1e-8 * np.eye(2))
    diff = mu_t - mu_p
    trace_term = float(np.trace(sigma_t_inv @ sigma_p))
    quad_term = float(diff @ sigma_t_inv @ diff)
    det_p = max(np.linalg.det(sigma_p), 1e-10)
    det_t = max(np.linalg.det(sigma_t), 1e-10)
    log_term = np.log(det_t / det_p)
    return 0.5 * (trace_term + quad_term - 2.0 + log_term)


def adaptive_weight(w: float, h: float, k: float = 2.0, r0: float = 3.0) -> float:
    """Compute blending weight alpha from box dimensions.

    Returns value in (0, 1). High aspect ratio -> alpha close to 1 (favor GWD).
    """
    r = max(w, h) / max(min(w, h), 1e-6)
    return 1.0 / (1.0 + np.exp(-k * (r - r0)))


@dataclass
class AdaptiveGWDKLDLoss:
    """Plug-and-play hybrid loss combining GWD and KLD with adaptive weighting."""

    k: float = 2.0
    r0: float = 3.0
    tau: float = 1.0

    def __call__(
        self,
        pred: Tuple[float, float, float, float, float],
        target: Tuple[float, float, float, float, float],
    ) -> float:
        """Compute loss for a single pred-target OBB pair.

        Args:
            pred: (cx, cy, w, h, theta)
            target: (cx, cy, w, h, theta)
        """
        mu_p, sigma_p = obb_to_gaussian(*pred)
        mu_t, sigma_t = obb_to_gaussian(*target)

        alpha = adaptive_weight(pred[2], pred[3], self.k, self.r0)
        loss_gwd = gwd_loss(mu_p, sigma_p, mu_t, sigma_t)
        loss_kld = kld_loss_symmetric(mu_p, sigma_p, mu_t, sigma_t)

        raw = alpha * loss_gwd + (1.0 - alpha) * loss_kld
        return float(np.log(raw / self.tau + 1.0))

    def batch(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Mean loss over a batch of OBBs. Each row is (cx,cy,w,h,theta)."""
        n = preds.shape[0]
        if n == 0:
            return 0.0
        total = sum(
            self(tuple(preds[i]), tuple(targets[i])) for i in range(n)
        )
        return total / n
