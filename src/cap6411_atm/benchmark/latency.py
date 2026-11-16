"""Warm-up-adjusted inference latency and peak-memory measurement.

All numbers produced here are *measured* wall-clock values on the device of
the input tensor.  Compare them only across runs on the same GPU with the
same dtype and batch size (see ``env.collect_env``).
"""

from __future__ import annotations

import math
import statistics
import time
from contextlib import nullcontext
from typing import Any, Callable

import torch
from torch import Tensor, nn

from cap6411_atm.models import BraTSDINOv3Segmenter

from .resolution import ReducedResolutionSegmenter


def autocast(device: torch.device, amp: bool):
    """Match Workstream 1 inference: fp16 autocast on CUDA, fp32 elsewhere."""

    if amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def precision_label(device: torch.device, amp: bool) -> str:
    return "fp16-autocast" if amp and device.type == "cuda" else "fp32"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(times_ms: list[float], batch_size: int) -> dict[str, float]:
    median = statistics.median(times_ms)
    return {
        "latency_median_ms": median,
        "latency_mean_ms": statistics.fmean(times_ms),
        "latency_std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "latency_p10_ms": _percentile(times_ms, 0.10),
        "latency_p90_ms": _percentile(times_ms, 0.90),
        "latency_min_ms": min(times_ms),
        "throughput_img_per_s": batch_size / (median / 1000.0) if median > 0 else float("inf"),
    }


def benchmark(
    model: nn.Module | Callable[[Any], Any],
    inputs: Any,
    *,
    warmup: int = 25,
    iters: int = 200,
    amp: bool = True,
    device: torch.device | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Time ``model(inputs)`` and record peak GPU memory.

    ``inputs`` may be a tensor or any object accepted by ``model``; pass
    ``device`` and ``batch_size`` explicitly when it is not a tensor.  Each iteration is timed
    individually (CUDA events on GPU, ``perf_counter`` on CPU) after
    ``warmup`` untimed calls.  Peak memory covers the timed iterations and
    includes resident weights; ``activation_memory_mb`` subtracts what was
    already allocated before the timed loop.
    """

    if warmup < 0 or iters < 1:
        raise ValueError("warmup must be >= 0 and iters >= 1")
    if device is None:
        if not isinstance(inputs, Tensor):
            raise ValueError("pass device= when inputs is not a tensor")
        device = inputs.device
    if batch_size is None:
        if not isinstance(inputs, Tensor):
            raise ValueError("pass batch_size= when inputs is not a tensor")
        batch_size = int(inputs.shape[0])
    if isinstance(model, nn.Module):
        model.eval()

    is_cuda = device.type == "cuda"
    times_ms: list[float] = []
    # Make the requested GPU current for the whole measurement, so the model's
    # kernels, the timing events, and the synchronization all use the same
    # device and stream even when it is not the default (e.g. --device cuda:1).
    device_context = torch.cuda.device(device) if is_cuda else nullcontext()
    with torch.inference_mode(), autocast(device, amp), device_context:
        stream = torch.cuda.current_stream(device) if is_cuda else None
        for _ in range(warmup):
            model(inputs)
        synchronize(device)
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            resident = torch.cuda.memory_allocated(device)
        for _ in range(iters):
            if is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                model(inputs)
                end.record(stream)
                end.synchronize()
                times_ms.append(start.elapsed_time(end))
            else:
                started = time.perf_counter()
                model(inputs)
                times_ms.append((time.perf_counter() - started) * 1000.0)

    result: dict[str, Any] = {
        "device": str(device),
        "precision": precision_label(device, amp),
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        **summarize(times_ms, batch_size),
    }
    if is_cuda:
        peak = torch.cuda.max_memory_allocated(device)
        result["peak_memory_mb"] = peak / 1024**2
        result["activation_memory_mb"] = (peak - resident) / 1024**2
    else:
        result["peak_memory_mb"] = float("nan")
        result["activation_memory_mb"] = float("nan")
    return result


def _stage_functions(model: nn.Module):
    """Stage callables for models whose forward path is known exactly.

    Exact type checks (not ``isinstance``/``hasattr``) are deliberate: a
    subclass or wrapper that merges tokens inside its own forward would
    otherwise be timed on the unmerged baseline path.  Other variants can opt
    in by defining ``benchmark_stage_functions()`` returning
    ``((name, callable), ...)`` whose chained calls equal ``forward``.
    """

    custom = getattr(model, "benchmark_stage_functions", None)
    if callable(custom):
        return tuple(custom())
    if type(model) is ReducedResolutionSegmenter and type(model.base) is BraTSDINOv3Segmenter:
        return (
            ("adapter", lambda x: model.base.adapter(model.downsample(x))),
            ("encoder", model.encode_adapted),
            ("decoder", model.base.decoder),
        )
    if type(model) is BraTSDINOv3Segmenter:
        return (
            ("adapter", model.adapter),
            ("encoder", model.encoder),
            ("decoder", model.decoder),
        )
    raise TypeError(
        f"no known stage split for {type(model).__name__}; "
        "define benchmark_stage_functions() to enable stage timing"
    )


def supports_stage_timing(model: nn.Module) -> bool:
    try:
        _stage_functions(model)
    except TypeError:
        return False
    return True


def benchmark_stages(
    model: nn.Module,
    images: Tensor,
    *,
    warmup: int = 25,
    iters: int = 200,
    amp: bool = True,
) -> dict[str, dict[str, Any]]:
    """Time adapter, encoder, and decoder separately on precomputed inputs.

    This shows where inference time is spent.  A merger applied after the
    encoder can only change the decoder share; encoder savings require
    merging inside the transformer blocks.
    """

    model.eval()
    results: dict[str, dict[str, Any]] = {}
    stage_input: Any = images
    for name, function in _stage_functions(model):
        results[name] = benchmark(
            function,
            stage_input,
            warmup=warmup,
            iters=iters,
            amp=amp,
            device=images.device,
            batch_size=int(images.shape[0]),
        )
        with torch.inference_mode(), autocast(images.device, amp):
            stage_input = function(stage_input)
    return results
