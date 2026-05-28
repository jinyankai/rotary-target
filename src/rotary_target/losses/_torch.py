"""PyTorch plug-and-play wrapper for AdaptiveGWDKLDLoss.

Usage in MMRotate config:
    custom_imports = dict(imports=['rotary_target.losses._torch'], allow_failed_imports=False)
    model = dict(
        bbox_head=dict(
            loss_bbox=dict(type='AdaptiveGWDKLDLoss', k=2.0, r0=3.0, tau=1.0)
        )
    )
"""

from __future__ import annotations


def _register():
    """Register the loss with MMRotate's registry. Call at import time."""
    try:
        import torch
        import torch.nn as nn
        from mmrotate.models.builder import ROTATED_LOSSES  # type: ignore
    except ImportError:
        return

    from rotary_target.losses.adaptive_gwd_kld import (
        adaptive_weight,
        gwd_loss,
        kld_loss_symmetric,
        obb_to_gaussian,
    )
    import numpy as np

    @ROTATED_LOSSES.register_module()
    class AdaptiveGWDKLDLoss(nn.Module):
        def __init__(self, k: float = 2.0, r0: float = 3.0, tau: float = 1.0,
                     reduction: str = "mean", loss_weight: float = 1.0):
            super().__init__()
            self.k = k
            self.r0 = r0
            self.tau = tau
            self.reduction = reduction
            self.loss_weight = loss_weight

        def forward(self, pred, target, weight=None, **kwargs):
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            n = pred_np.shape[0]
            losses = np.zeros(n)
            for i in range(n):
                p = tuple(pred_np[i])
                t = tuple(target_np[i])
                mu_p, sig_p = obb_to_gaussian(*p)
                mu_t, sig_t = obb_to_gaussian(*t)
                alpha = adaptive_weight(p[2], p[3], self.k, self.r0)
                raw = (alpha * gwd_loss(mu_p, sig_p, mu_t, sig_t)
                       + (1 - alpha) * kld_loss_symmetric(mu_p, sig_p, mu_t, sig_t))
                losses[i] = np.log(raw / self.tau + 1.0)

            loss_tensor = torch.tensor(losses, dtype=pred.dtype, device=pred.device)
            if weight is not None:
                loss_tensor = loss_tensor * weight
            if self.reduction == "mean":
                return loss_tensor.mean() * self.loss_weight
            return loss_tensor.sum() * self.loss_weight


_register()
