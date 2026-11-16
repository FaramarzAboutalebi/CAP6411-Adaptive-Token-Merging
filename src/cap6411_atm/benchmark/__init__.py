"""Efficiency benchmarking and the reduced-input-resolution baseline (Workstream 3)."""

from .compare import (
    attach_quality,
    describe,
    evaluate_quality,
    evaluate_quality_many,
    load_timing_only_dinov3_vitb16,
    match_latency,
    plot_tradeoff,
    read_rows,
    run_suite,
    validation_loader,
    write_rows,
)
from .env import collect_env, write_env
from .latency import benchmark, benchmark_stages, supports_stage_timing
from .resolution import DEFAULT_RESOLUTIONS, ReducedResolutionSegmenter, resolution_sweep

__all__ = [
    "DEFAULT_RESOLUTIONS",
    "ReducedResolutionSegmenter",
    "attach_quality",
    "benchmark",
    "benchmark_stages",
    "collect_env",
    "describe",
    "evaluate_quality",
    "evaluate_quality_many",
    "load_timing_only_dinov3_vitb16",
    "match_latency",
    "plot_tradeoff",
    "read_rows",
    "resolution_sweep",
    "run_suite",
    "supports_stage_timing",
    "validation_loader",
    "write_env",
    "write_rows",
]
