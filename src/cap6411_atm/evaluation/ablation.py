"""Resumable ablation runner over model variants and decision rules.

A variant is one *model* configuration (baseline, uniform/adaptive merging at
some ratio, reduced resolution, ...). Each variant is evaluated once per
forward pass with all of its decision ``rules``. Results land in
``<output_dir>/<variant>/per_patient.csv``; finished variants are skipped on
re-run, so a Colab disconnect only loses the variant in progress.

Workstream 2 plugs in through ``build_model(spec) -> logits callable``; the
merging parameters live in ``spec.params`` and are recorded with the results.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from .evaluator import evaluate_model, save_evaluation
from .metrics import MetricConfig
from .postprocess import DEFAULT_RULES
from .statistics import load_results


@dataclass(frozen=True)
class VariantSpec:
    """One evaluated model configuration."""

    name: str
    method: str = "baseline"  # baseline | uniform | adaptive | lowres | ...
    merge_ratio: float | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    rules: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: dict(DEFAULT_RULES))

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", self.name):
            raise ValueError(f"variant name {self.name!r} must be filesystem-safe")
        if self.merge_ratio is not None and not 0 <= self.merge_ratio < 1:
            raise ValueError("merge_ratio must lie in [0, 1)")

    def info(self) -> dict[str, Any]:
        values = asdict(self)
        values["params"] = dict(self.params)
        values["rules"] = {key: dict(value) for key, value in self.rules.items()}
        return values


def merge_ratio_grid(
    method: str,
    ratios: Sequence[float],
    *,
    params: Mapping[str, Any] | None = None,
    rules: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[VariantSpec]:
    """Variants for a merge-ratio sweep, e.g. ``merge_ratio_grid("adaptive", [0.1, 0.25])``."""

    return [
        VariantSpec(
            name=f"{method}_r{ratio:.2f}",
            method=method,
            merge_ratio=float(ratio),
            params=dict(params or {}),
            rules=dict(rules or DEFAULT_RULES),
        )
        for ratio in ratios
    ]


def run_ablation(
    variants: Sequence[VariantSpec],
    build_model: Callable[[VariantSpec], Any],
    loader: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    metric_config: MetricConfig = MetricConfig(),
    overwrite: bool = False,
    progress: bool = True,
    **evaluate_kwargs: Any,
) -> pd.DataFrame:
    """Evaluate every variant (skipping finished ones) and return the combined table."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = [variant.name for variant in variants]
    if len(names) != len(set(names)):
        raise ValueError("variant names must be unique")
    for variant in variants:
        destination = output / variant.name
        if (destination / "per_patient.csv").is_file() and not overwrite:
            if progress:
                print(f"[skip] {variant.name}: results exist", flush=True)
            continue
        model = build_model(variant)
        result = evaluate_model(
            model,
            loader,
            rules=variant.rules,
            metric_config=metric_config,
            variant=variant.name,
            variant_info=variant.info(),
            progress=False,
            **evaluate_kwargs,
        )
        save_evaluation(result, destination)
        if progress:
            macro = result.metadata["global_dice"]
            print(f"[done] {variant.name}: " + json.dumps({k: round(v["macro"], 4) for k, v in macro.items()}), flush=True)
        del model
    return collect_results(output, names)


def collect_results(output_dir: str | Path, names: Sequence[str] | None = None) -> pd.DataFrame:
    """Load finished variants and attach their method / merge-ratio metadata."""

    output = Path(output_dir)
    directories = (
        [output / name for name in names]
        if names is not None
        else sorted(path.parent for path in output.glob("*/per_patient.csv"))
    )
    directories = [path for path in directories if (path / "per_patient.csv").is_file()]
    if not directories:
        raise FileNotFoundError(f"no finished variants under {output}")
    frame = load_results(directories)
    info = {}
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        variant_info = summary.get("variant_info", {})
        info[summary["variant"]] = {
            "family": variant_info.get("method", "baseline"),
            "merge_ratio": variant_info.get("merge_ratio"),
        }
    frame["family"] = frame["variant"].map(lambda v: info.get(v, {}).get("family"))
    frame["merge_ratio"] = frame["variant"].map(lambda v: info.get(v, {}).get("merge_ratio"))
    return frame
