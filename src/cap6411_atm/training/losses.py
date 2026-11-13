"""Losses for overlapping WT, TC, and ET segmentation targets."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MultilabelBCEDiceLoss(nn.Module):
    """Weighted BCE-with-logits plus channel-balanced soft Dice loss.

    Dice is computed independently for WT, TC, and ET over the batch and spatial
    dimensions, then averaged. Targets are overlapping binary channels, so this
    loss intentionally uses sigmoid semantics rather than softmax.
    """

    def __init__(
        self,
        *,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("loss weights must be non-negative with a positive sum")
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def components(self, logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
        if logits.shape != targets.shape or logits.ndim != 4:
            raise ValueError(
                f"logits and targets must have identical [B,C,H,W] shapes; "
                f"received {tuple(logits.shape)} and {tuple(targets.shape)}"
            )
        if logits.shape[1] != 3:
            raise ValueError("expected three WT/TC/ET channels")
        if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
            raise ValueError("loss inputs must be finite")
        probabilities = torch.sigmoid(logits.float())
        targets_float = targets.float()
        reduction_dims = (0, 2, 3)
        intersection = (probabilities * targets_float).sum(dim=reduction_dims)
        denominator = probabilities.sum(dim=reduction_dims) + targets_float.sum(
            dim=reduction_dims
        )
        dice_per_channel = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        bce = F.binary_cross_entropy_with_logits(logits.float(), targets_float)
        dice = 1.0 - dice_per_channel.mean()
        total = self.bce_weight * bce + self.dice_weight * dice
        return {"loss": total, "bce": bce, "dice": dice}

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.components(logits, targets)["loss"]
