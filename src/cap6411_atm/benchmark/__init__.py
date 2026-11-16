"""Efficiency benchmarking and the reduced-input-resolution baseline (Workstream 3)."""

from .compare import (
    attach_quality,
    checkpoint_fingerprint,
    describe,
    evaluate_quality,
    evaluate_quality_many,
    load_benchmark_config,
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
from .tables import matched_table, results_table
from .resolution import DEFAULT_RESOLUTIONS, ReducedResolutionSegmenter, resolution_sweep

__all__ = [
    "DEFAULT_RESOLUTIONS",
    "ReducedResolutionSegmenter",
    "attach_quality",
    "benchmark",
    "benchmark_stages",
    "checkpoint_fingerprint",
    "collect_env",
    "describe",
    "evaluate_quality",
    "evaluate_quality_many",
    "load_benchmark_config",
    "load_timing_only_dinov3_vitb16",
    "match_latency",
    "matched_table",
    "plot_tradeoff",
    "read_rows",
    "resolution_sweep",
    "results_table",
    "run_suite",
    "supports_stage_timing",
    "validation_loader",
    "write_env",
    "write_rows",
]
