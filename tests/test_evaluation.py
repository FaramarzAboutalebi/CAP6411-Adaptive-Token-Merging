"""Tests for end-to-end BraTS segmentation evaluation."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from cap6411_atm.evaluation import evaluate_segmentation


class FirstThreeChannels(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :3]


class EvaluationTests(unittest.TestCase):
    def test_evaluation_computes_global_dice_and_restores_mode(self) -> None:
        model = FirstThreeChannels().train()
        images = torch.full((1, 4, 2, 2), -10.0)
        targets = torch.zeros(1, 3, 2, 2)
        for channel, pixel in enumerate((0, 1, 2)):
            images[0, channel].view(-1)[pixel] = 10.0
            targets[0, channel].view(-1)[pixel] = 1.0
        batches = [{"image": images, "target": targets}]

        result = evaluate_segmentation(model, batches, threshold=0.5)

        self.assertEqual(result["batches"], 1)
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["dice"], {"WT": 1.0, "TC": 1.0, "ET": 1.0})
        self.assertEqual(result["macro_dice"], 1.0)
        self.assertTrue(model.training)

    def test_evaluation_rejects_empty_iterable(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty batch iterable"):
            evaluate_segmentation(FirstThreeChannels(), [])

    def test_evaluation_rejects_non_finite_model_output(self) -> None:
        class NonFiniteModel(nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.full(
                    (images.shape[0], 3, images.shape[2], images.shape[3]),
                    float("nan"),
                    device=images.device,
                )

        images = torch.zeros(1, 4, 2, 2)
        targets = torch.zeros(1, 3, 2, 2)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            evaluate_segmentation(NonFiniteModel(), [(images, targets)])


if __name__ == "__main__":
    unittest.main()
