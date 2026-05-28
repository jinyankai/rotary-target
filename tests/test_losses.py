"""Tests for the adaptive GWD-KLD hybrid loss module."""

import numpy as np
import pytest

from rotary_target.losses import (
    AdaptiveGWDKLDLoss,
    adaptive_weight,
    gwd_loss,
    kld_loss_symmetric,
    obb_to_gaussian,
)


class TestObbToGaussian:
    def test_axis_aligned_box(self):
        mu, sigma = obb_to_gaussian(0, 0, 6, 4, 0)
        np.testing.assert_array_almost_equal(mu, [0, 0])
        assert sigma[0, 1] == pytest.approx(0, abs=1e-10)
        assert sigma[1, 0] == pytest.approx(0, abs=1e-10)
        assert sigma[0, 0] == pytest.approx(6**2 / 12)
        assert sigma[1, 1] == pytest.approx(4**2 / 12)

    def test_90_degree_rotation_swaps_eigenvalues(self):
        _, sigma_0 = obb_to_gaussian(1, 2, 6, 4, 0)
        _, sigma_90 = obb_to_gaussian(1, 2, 6, 4, np.pi / 2)
        assert sigma_90[0, 0] == pytest.approx(sigma_0[1, 1], abs=1e-10)
        assert sigma_90[1, 1] == pytest.approx(sigma_0[0, 0], abs=1e-10)


class TestGWDLoss:
    def test_identical_boxes_give_zero(self):
        mu, sigma = obb_to_gaussian(5, 5, 10, 4, 0.3)
        assert gwd_loss(mu, sigma, mu, sigma) == pytest.approx(0, abs=1e-8)

    def test_different_centers_positive(self):
        mu_p, sig_p = obb_to_gaussian(0, 0, 4, 4, 0)
        mu_t, sig_t = obb_to_gaussian(10, 0, 4, 4, 0)
        assert gwd_loss(mu_p, sig_p, mu_t, sig_t) > 0


class TestKLDLoss:
    def test_identical_boxes_give_zero(self):
        mu, sigma = obb_to_gaussian(3, 3, 8, 2, 0.5)
        assert kld_loss_symmetric(mu, sigma, mu, sigma) == pytest.approx(0, abs=1e-6)

    def test_symmetric(self):
        mu_p, sig_p = obb_to_gaussian(0, 0, 4, 2, 0)
        mu_t, sig_t = obb_to_gaussian(1, 1, 6, 3, 0.2)
        kl_ab = kld_loss_symmetric(mu_p, sig_p, mu_t, sig_t)
        kl_ba = kld_loss_symmetric(mu_t, sig_t, mu_p, sig_p)
        assert kl_ab == pytest.approx(kl_ba, rel=1e-6)


class TestAdaptiveWeight:
    def test_high_aspect_ratio_favors_gwd(self):
        alpha = adaptive_weight(20, 2, k=2.0, r0=3.0)
        assert alpha > 0.9

    def test_square_favors_kld(self):
        alpha = adaptive_weight(4, 4, k=2.0, r0=3.0)
        assert alpha < 0.1

    def test_at_crossover(self):
        alpha = adaptive_weight(6, 2, k=2.0, r0=3.0)
        assert 0.4 < alpha < 0.6


class TestAdaptiveGWDKLDLoss:
    def test_identical_boxes_near_zero(self):
        loss_fn = AdaptiveGWDKLDLoss()
        box = (5.0, 5.0, 10.0, 4.0, 0.3)
        result = loss_fn(box, box)
        assert result == pytest.approx(0, abs=1e-6)

    def test_different_boxes_positive(self):
        loss_fn = AdaptiveGWDKLDLoss()
        pred = (0.0, 0.0, 8.0, 2.0, 0.0)
        target = (2.0, 1.0, 8.0, 2.0, 0.1)
        assert loss_fn(pred, target) > 0

    def test_batch_mean(self):
        loss_fn = AdaptiveGWDKLDLoss()
        preds = np.array([[0, 0, 4, 4, 0], [5, 5, 10, 2, 0.5]])
        targets = np.array([[0, 0, 4, 4, 0], [5, 5, 10, 2, 0.5]])
        assert loss_fn.batch(preds, targets) == pytest.approx(0, abs=1e-6)

    def test_batch_empty(self):
        loss_fn = AdaptiveGWDKLDLoss()
        preds = np.empty((0, 5))
        targets = np.empty((0, 5))
        assert loss_fn.batch(preds, targets) == 0.0
