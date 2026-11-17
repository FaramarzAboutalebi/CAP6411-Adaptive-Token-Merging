"""Resumable ablation runner over model variants and decision rules.

A variant is one *model* configuration (baseline, uniform/adaptive merging at
some ratio, reduced resolution, ...). Each variant is evaluated once per
forward pass with all of its decision ``rules``. Results land in
``<output_dir>/<variant>/per_patient.csv``; finished variants are skipped on
re-run, so a Colab disconnect only loses the variant in progress. A finished
variant is reused only if its recorded ``run_key`` (checkpoint, dataset,
manifest, patients, rules and metric settings) still matches.

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
import torch

from .evaluator import _dataset_attribute, dataset_patient_ids, evaluate_model, resolve_split, save_evaluation
from .metrics import MetricConfig
from .postprocess import DEFAULT_RULES
from .provenance import SHARED_PROVENANCE, digest, fingerprint_file, run_key, validate_provenance
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
    provenance: Mapping[str, Any] | Callable[[VariantSpec], Mapping[str, Any]],
    metric_config: MetricConfig = MetricConfig(),
    overwrite: bool = False,
    progress: bool = True,
    **evaluate_kwargs: Any,
) -> pd.DataFrame:
    """Evaluate every variant (skipping finished ones) and return the combined table.

    ``provenance`` (a mapping, or a callable giving one per variant) must
    fingerprint at least the checkpoint, dataset archive and split manifest
    (see :func:`~cap6411_atm.evaluation.provenance.build_provenance`).
    Together with the variant spec, rules, metric settings, split and patient
    list it forms the variant's ``run_key``. Existing results are reused only
    when their ``run_key`` and CSV checksum match; otherwise this raises
    instead of silently mixing stale numbers (pass ``overwrite=True`` or a new
    ``output_dir``).
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = [variant.name for variant in variants]
    if len(names) != len(set(names)):
        raise ValueError("variant names must be unique")
    context = _evaluation_context(loader, metric_config, evaluate_kwargs)
    for variant in variants:
        destination = output / variant.name
        variant_provenance = validate_provenance(provenance(variant) if callable(provenance) else provenance)
        identity = {**context, "variant": variant.info(), "provenance": variant_provenance}
        key = run_key(identity)
        if (destination / "per_patient.csv").is_file() and not overwrite:
            _check_reusable(destination, key)
            if progress:
                print(f"[skip] {variant.name}: results exist and match", flush=True)
            continue
        model = build_model(variant)
        result = evaluate_model(
            model,
            loader,
            rules=variant.rules,
            metric_config=metric_config,
            variant=variant.name,
            variant_info=variant.info(),
            provenance=variant_provenance,
            progress=False,
            **evaluate_kwargs,
        )
        result.metadata["run_identity"] = identity
        result.metadata["run_key"] = key
        save_evaluation(result, destination)
        if progress:
            macro = result.metadata["global_dice"]
            print(f"[done] {variant.name}: " + json.dumps({k: round(v["macro"], 4) for k, v in macro.items()}), flush=True)
        del model
    return collect_results(output, names)


def _evaluation_context(
    loader: Iterable[Mapping[str, Any]], metric_config: MetricConfig, evaluate_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Run settings shared by every variant of one ablation call."""

    dataset = getattr(loader, "dataset", None)
    split = resolve_split(dataset, evaluate_kwargs.get("split"))
    patient_ids = dataset_patient_ids(dataset) if dataset is not None else None
    if not patient_ids:
        raise ValueError("run_ablation needs a loader whose dataset lists its patients (patient_ids or samples)")
    device = evaluate_kwargs.get("device")
    device_type = torch.device(device).type if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    expected_depth = evaluate_kwargs.get("expected_depth")
    if expected_depth is None:
        expected_depth = _dataset_attribute(dataset, "axial_depth")
    return {
        "split": split,
        "patient_ids_sha256": digest(sorted(patient_ids)),
        "n_patients": len(patient_ids),
        "expected_depth": expected_depth,
        "complete_volumes": not evaluate_kwargs.get("allow_partial_volumes", False),
        "metric_config": asdict(metric_config),
        "amp": bool(evaluate_kwargs.get("amp", True)) and device_type == "cuda",
    }


def _check_reusable(directory: Path, key: str) -> None:
    summary_path = directory / "summary.json"
    hint = "pass overwrite=True to recompute, or use a new output directory"
    if not summary_path.is_file():
        raise RuntimeError(f"{directory}: per_patient.csv has no summary.json; {hint}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("run_key") is None:
        raise RuntimeError(f"{directory}: results predate provenance tracking; {hint}")
    if summary["run_key"] != key:
        raise RuntimeError(
            f"{directory}: stale results (model, data or evaluation settings changed since they "
            f"were computed); {hint}"
        )
    _check_csv(directory, summary)


def _check_csv(directory: Path, summary: Mapping[str, Any]) -> None:
    recorded = summary.get("per_patient_sha256")
    actual = fingerprint_file(directory / "per_patient.csv")["sha256"]
    if recorded != actual:
        raise RuntimeError(f"{directory}: per_patient.csv does not match its summary.json checksum")


# Run settings that must be identical for variants to be compared.
_SHARED_CONTEXT = ("split", "patient_ids_sha256", "expected_depth", "complete_volumes", "metric_config")


def _check_comparable(summaries: Mapping[Path, Mapping[str, Any]]) -> None:
    reference: dict[str, Any] | None = None
    reference_dir: Path | None = None
    for directory, summary in summaries.items():
        identity = summary.get("run_identity")
        if identity is None:
            raise RuntimeError(f"{directory}: results predate provenance tracking; rerun or pass strict=False")
        _check_csv(directory, summary)
        if summary.get("variant") != directory.name:
            raise RuntimeError(f"{directory}: summary.json belongs to variant {summary.get('variant')!r}")
        shared = {key: identity.get(key) for key in _SHARED_CONTEXT}
        shared["provenance"] = {key: identity.get("provenance", {}).get(key) for key in SHARED_PROVENANCE}
        if reference is None:
            reference, reference_dir = shared, directory
        elif shared != reference:
            differing = sorted(key for key in shared if shared[key] != reference[key])
            raise RuntimeError(
                f"{directory.name} and {reference_dir.name} come from different experiments "
                f"(they differ in {differing}) and cannot be compared"
            )


def collect_results(
    output_dir: str | Path, names: Sequence[str] | None = None, *, strict: bool = True
) -> pd.DataFrame:
    """Load finished variants and attach their method / merge-ratio metadata.

    With ``strict`` (default) every variant must carry provenance, its CSV must
    match the recorded checksum, and all variants must share the same split,
    patients, volume depth, metric settings, dataset, manifest and inspection
    config, so results from different experiments cannot be mixed.
    """

    output = Path(output_dir)
    directories = (
        [output / name for name in names]
        if names is not None
        else sorted(path.parent for path in output.glob("*/per_patient.csv"))
    )
    directories = [path for path in directories if (path / "per_patient.csv").is_file()]
    if not directories:
        raise FileNotFoundError(f"no finished variants under {output}")
    summaries = {d: json.loads((d / "summary.json").read_text(encoding="utf-8")) for d in directories}
    if strict:
        _check_comparable(summaries)
    frame = load_results(directories)
    info = {}
    for directory, summary in summaries.items():
        variant_info = summary.get("variant_info", {})
        info[summary["variant"]] = {
            "family": variant_info.get("method", "baseline"),
            "merge_ratio": variant_info.get("merge_ratio"),
        }
    frame["family"] = frame["variant"].map(lambda v: info.get(v, {}).get("family"))
    frame["merge_ratio"] = frame["variant"].map(lambda v: info.get(v, {}).get("merge_ratio"))
    return frame
