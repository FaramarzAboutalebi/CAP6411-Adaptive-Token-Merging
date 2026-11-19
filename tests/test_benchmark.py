"""Tests for repeatable segmentation inference measurements."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from cap6411_atm.benchmark import benchmark_inference


class FirstThreeChannels(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :3]


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_reports_latency_shape_and_restores_mode(self) -> None:
        model = FirstThreeChannels().train()
        images = torch.randn(2, 4, 8, 8)

        result = benchmark_inference(model, images, warmup=1, iterations=3)

        self.assertEqual(result["input_shape"], [2, 4, 8, 8])
        self.assertEqual(result["output_shape"], [2, 3, 8, 8])
        self.assertEqual(result["measured_iterations"], 3)
        self.assertEqual(len(result["latency_ms"]), 3)
        self.assertGreater(result["median_latency_ms"], 0)
        self.assertGreater(result["samples_per_second"], 0)
        self.assertIsNone(result["peak_cuda_bytes"])
        self.assertTrue(model.training)

    def test_benchmark_rejects_non_segmentation_output(self) -> None:
        model = nn.Identity()
        with self.assertRaisesRegex(RuntimeError, "expected segmentation logits"):
            benchmark_inference(model, torch.randn(1, 4, 8, 8), warmup=0, iterations=1)


if __name__ == "__main__":
    unittest.main()
