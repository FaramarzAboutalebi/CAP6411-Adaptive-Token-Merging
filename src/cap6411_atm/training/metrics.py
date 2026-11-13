"""Streaming validation metrics for overlapping BraTS tumor regions."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


REGION_NAMES = ("WT", "TC", "ET")


@dataclass
class StreamingDice:
    """Accumulate cohort-level thresholded Dice without empty-slice inflation.

    Counts are aggregated over all batches before computing Dice. Empty target
    slices therefore do not receive automatic perfect scores. If an entire
    evaluated channel contains neither target nor prediction voxels, its Dice is
    defined as 1.0 and reported through ``all_empty``.
    """

    threshold: float = 0.5
    intersection: Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float64)
    )
    prediction_count: Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float64)
    )
    target_count: Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float64)
    )

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between zero and one")

    def update(self, logits: Tensor, targets: Tensor) -> None:
        if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 3:
            raise ValueError("expected matching logits/targets shaped [B,3,H,W]")
        predictions = torch.sigmoid(logits.detach().float()) >= self.threshold
        target_masks = targets.detach() >= 0.5
        dims = (0, 2, 3)
        self.intersection += (predictions & target_masks).sum(dim=dims).cpu().double()
        self.prediction_count += predictions.sum(dim=dims).cpu().double()
        self.target_count += target_masks.sum(dim=dims).cpu().double()

    def compute(self) -> dict[str, object]:
        denominator = self.prediction_count + self.target_count
        all_empty = denominator == 0
        dice = torch.where(
            all_empty,
            torch.ones_like(denominator),
            2.0 * self.intersection / denominator.clamp_min(1.0),
        )
        values = {name: float(dice[index]) for index, name in enumerate(REGION_NAMES)}
        return {
            "dice": values,
            "macro_dice": float(dice.mean()),
            "all_empty": {
                name: bool(all_empty[index]) for index, name in enumerate(REGION_NAMES)
            },
            "target_voxels": {
                name: int(self.target_count[index]) for index, name in enumerate(REGION_NAMES)
            },
            "prediction_voxels": {
                name: int(self.prediction_count[index])
                for index, name in enumerate(REGION_NAMES)
            },
        }
