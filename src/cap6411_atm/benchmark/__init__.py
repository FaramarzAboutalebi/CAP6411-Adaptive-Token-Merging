"""Reproducible inference benchmarking utilities for segmentation models."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any

import torch
from torch import Tensor, nn


def _model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    tensor = next(iter(model.parameters()), None)
    if tensor is None:
        tensor = next(iter(model.buffers()), None)
    return tensor.device if tensor is not None else fallback


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_inference(
    model: nn.Module,
    images: Tensor,
    *,
    warmup: int = 10,
    iterations: int = 50,
) -> dict[str, Any]:
    """Measure model-only inference latency and peak allocated CUDA memory.

    Inputs must already be on the model's device. The result reports individual
    latency samples plus summary statistics so callers can retain reproducible
    benchmark records instead of relying on a single timing.
    """
    if images.ndim != 4 or images.shape[0] == 0:
        raise ValueError(f"expected non-empty image batch [B,C,H,W], got {tuple(images.shape)}")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    device = _model_device(model, images.device)
    if images.device != device:
        raise ValueError(f"images are on {images.device}, but model is on {device}")

    was_training = model.training
    model.eval()
    latencies_ms: list[float] = []
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                logits = model(images)
                if not isinstance(logits, Tensor):
                    raise TypeError("segmentation model must return a logits tensor")
                del logits
            _synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            output_shape: tuple[int, ...] | None = None
            for _ in range(iterations):
                _synchronize(device)
                started = time.perf_counter()
                logits = model(images)
                _synchronize(device)
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
                if not isinstance(logits, Tensor):
                    raise TypeError("segmentation model must return a logits tensor")
                expected_shape = (
                    images.shape[0],
                    3,
                    images.shape[2],
                    images.shape[3],
                )
                if tuple(logits.shape) != expected_shape:
                    raise RuntimeError(
                        f"expected segmentation logits {expected_shape}, got {tuple(logits.shape)}"
                    )
                output_shape = tuple(logits.shape)
                del logits
    finally:
        model.train(was_training)

    sorted_latencies = sorted(latencies_ms)
    p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
    median_ms = statistics.median(latencies_ms)
    return {
        "device": str(device),
        "input_shape": list(images.shape),
        "output_shape": list(output_shape) if output_shape is not None else None,
        "batch_size": int(images.shape[0]),
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "latency_ms": latencies_ms,
        "mean_latency_ms": statistics.fmean(latencies_ms),
        "median_latency_ms": median_ms,
        "p95_latency_ms": sorted_latencies[p95_index],
        "samples_per_second": float(images.shape[0] * 1000.0 / median_ms),
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }


__all__ = ["benchmark_inference"]
