"""Run a model over patient-local slice batches and score whole patient volumes.

The evaluator is model-agnostic: anything mapping ``[B,4,240,240]`` images to
``[B,3,240,240]`` raw logits can be evaluated (baseline, reduced-resolution
models that upsample their logits, or token-merging variants).

One forward pass feeds many decision rules (``rules``), so threshold and
post-processing ablations are free. Slices are reassembled into
``[3,H,W,D]`` patient volumes before metrics are computed, which is the BraTS
convention, and the global voxel-count Dice of Workstream 1's ``StreamingDice``
is reported alongside for continuity.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .metrics import REGION_NAMES, MetricConfig, patient_metrics
from .postprocess import DEFAULT_RULES, build_postprocessor


LogitsFn = Callable[[Tensor], Tensor]


@dataclass
class EvaluationResult:
    """Per-patient, per-region metric rows plus run metadata."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    saved_predictions: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    def global_dice(self) -> dict[str, dict[str, float]]:
        """Cohort voxel-count Dice per rule (equals StreamingDice for full volumes)."""

        totals: dict[str, dict[str, list[int]]] = {}
        for row in self.rows:
            entry = totals.setdefault(row["rule"], {}).setdefault(row["region"], [0, 0])
            entry[0] += 2 * row["tp"]
            entry[1] += row["pred_volume"] + row["gt_volume"]
        output: dict[str, dict[str, float]] = {}
        for rule, regions in totals.items():
            values = {
                region: (numerator / denominator if denominator else 1.0)
                for region, (numerator, denominator) in regions.items()
            }
            values["macro"] = float(np.mean([values[name] for name in REGION_NAMES]))
            output[rule] = values
        return output


def evaluate(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
    patch_size: int = 16,
) -> dict[str, Any]:
    """Quick batch-level check matching the README's ``evaluate(logits, targets)``.

    Treats the batch as independent 2D slices and reports pooled overlap metrics
    and token-patch error rates per region. Surface-distance metrics need whole
    patient volumes; use :func:`evaluate_model` for reported results.
    """

    from .metrics import overlap_metrics, patch_class_errors

    if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != len(REGION_NAMES):
        raise ValueError("expected matching [B,3,H,W] logits and targets")
    predictions = (torch.sigmoid(logits.detach().float()) >= threshold).cpu().numpy()
    target_masks = (targets.detach() >= 0.5).cpu().numpy()
    output: dict[str, Any] = {}
    for index, region in enumerate(REGION_NAMES):
        # [B,H,W] -> [H,W,B] so slices act as the depth axis of the patch tiler.
        pred = np.moveaxis(predictions[:, index], 0, -1)
        gt = np.moveaxis(target_masks[:, index], 0, -1)
        output[region] = {**overlap_metrics(pred, gt), **patch_class_errors(pred, gt, patch_size)}
    output["macro_dice"] = float(np.mean([output[name]["dice"] for name in REGION_NAMES]))
    return output


def _split_of(dataset: Any) -> str | None:
    while dataset is not None:
        split = getattr(dataset, "split", None)
        if split is not None:
            return split
        dataset = getattr(dataset, "dataset", None)
    return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class _PatientBuffer:
    """Collect contiguous slices of one patient and emit a sorted volume."""

    def __init__(self) -> None:
        self.patient_id: str | None = None
        self.slices: list[int] = []
        self.probabilities: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.images: list[np.ndarray] = []

    def add(self, slice_index: int, probabilities: np.ndarray, target: np.ndarray, image: np.ndarray | None) -> None:
        self.slices.append(slice_index)
        self.probabilities.append(probabilities)
        self.targets.append(target)
        if image is not None:
            self.images.append(image)

    def volumes(self) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray | None]:
        order = np.argsort(self.slices)
        if len(set(self.slices)) != len(self.slices):
            raise RuntimeError(f"duplicate slices for patient {self.patient_id}")
        probabilities = np.stack([self.probabilities[i] for i in order], axis=-1)
        targets = np.stack([self.targets[i] for i in order], axis=-1)
        images = np.stack([self.images[i] for i in order], axis=-1) if self.images else None
        return [self.slices[i] for i in order], probabilities, targets, images


def evaluate_model(
    model: torch.nn.Module | LogitsFn,
    loader: Iterable[Mapping[str, Any]],
    *,
    rules: Mapping[str, Mapping[str, Any]] | None = None,
    metric_config: MetricConfig = MetricConfig(),
    device: torch.device | str | None = None,
    amp: bool = True,
    variant: str = "baseline",
    variant_info: Mapping[str, Any] | None = None,
    save_patients: Sequence[str] = (),
    allow_test: bool = False,
    progress: bool = False,
) -> EvaluationResult:
    """Evaluate ``model`` on every patient produced by ``loader``.

    Args:
        model: module or callable returning raw logits ``[B,3,H,W]``.
        loader: yields dataset batches with ``image``, ``target``,
            ``patient_id`` and ``slice_index``. Batches must be patient-local
            and each patient's slices contiguous (use ``build_eval_loader``).
        rules: named post-processing specs; default is the 0.5 threshold.
        variant: label stored in every row (e.g. ``"uniform_r0.30"``).
        variant_info: JSON-compatible metadata (method, merge ratio, ...).
        save_patients: patient IDs whose image/target/masks are kept in memory
            for figures.
        allow_test: must be ``True`` to evaluate the isolated test split.
    """

    dataset = getattr(loader, "dataset", None)
    split = _split_of(dataset)
    if split == "test" and not allow_test:
        raise PermissionError(
            "refusing to evaluate the isolated test split; pass allow_test=True only "
            "for the final, pre-registered evaluation"
        )
    rules = dict(rules or DEFAULT_RULES)
    processors = {name: build_postprocessor(spec) for name, spec in rules.items()}
    save_set = set(save_patients)
    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp and device.type == "cuda"
    if isinstance(model, torch.nn.Module):
        model.eval()

    result = EvaluationResult()
    finished: set[str] = set()
    buffer = _PatientBuffer()
    slice_count = 0
    started = time.perf_counter()
    forward_seconds = 0.0
    metric_seconds = 0.0

    def finalize() -> None:
        nonlocal metric_seconds
        if buffer.patient_id is None:
            return
        metric_start = time.perf_counter()
        slices, probabilities, targets, images = buffer.volumes()
        target_masks = targets >= 0.5
        keep = buffer.patient_id in save_set
        if keep:
            result.saved_predictions[buffer.patient_id] = {
                "slices": np.asarray(slices),
                "target": target_masks,
                "image": images,
            }
        for rule_name, processor in processors.items():
            masks = processor(probabilities)
            if keep:
                result.saved_predictions[buffer.patient_id][rule_name] = masks
            for region, values in patient_metrics(masks, target_masks, metric_config).items():
                result.rows.append(
                    {
                        "variant": variant,
                        "rule": rule_name,
                        "patient_id": buffer.patient_id,
                        "region": region,
                        "n_slices": len(slices),
                        **values,
                    }
                )
        finished.add(buffer.patient_id)
        metric_seconds += time.perf_counter() - metric_start
        if progress:
            print(f"[{variant}] {len(finished)} patients evaluated", flush=True)

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"]
            patient_ids = list(batch["patient_id"])
            slice_indices = [int(value) for value in batch["slice_index"]]
            forward_start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images.to(device, non_blocking=True))
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            forward_seconds += time.perf_counter() - forward_start
            if probabilities.shape[1:] != tuple(batch["target"].shape[1:]):
                raise ValueError(
                    f"logits {tuple(logits.shape)} do not match targets {tuple(batch['target'].shape)}"
                )
            targets = batch["target"].numpy()
            image_array = images.numpy()
            for offset, patient_id in enumerate(patient_ids):
                if patient_id != buffer.patient_id:
                    finalize()
                    if patient_id in finished:
                        raise RuntimeError(
                            f"patient {patient_id} slices are not contiguous; use a "
                            "non-shuffled PatientLocalBatchSampler"
                        )
                    buffer = _PatientBuffer()
                    buffer.patient_id = patient_id
                keep_image = patient_id in save_set
                buffer.add(
                    slice_indices[offset],
                    probabilities[offset].astype(np.float32, copy=False),
                    targets[offset],
                    image_array[offset] if keep_image else None,
                )
                slice_count += 1
        finalize()

    if not finished:
        raise ValueError("loader produced no samples")
    result.metadata = {
        "variant": variant,
        "variant_info": dict(variant_info or {}),
        "rules": rules,
        "split": split,
        "patients": len(finished),
        "slices": slice_count,
        "metric_config": asdict(metric_config),
        "amp": use_amp,
        "device": str(device),
        "git_commit": _git_commit(),
        "elapsed_seconds": time.perf_counter() - started,
        "forward_seconds": forward_seconds,
        "metric_seconds": metric_seconds,
        "note": "forward_seconds is not a latency benchmark; use Workstream 3 tools for latency",
    }
    result.metadata["global_dice"] = result.global_dice()
    return result


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def save_evaluation(result: EvaluationResult, output_dir: str | Path) -> Path:
    """Write ``per_patient.csv`` and ``summary.json`` (no images or patient data)."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in result.rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = output / "per_patient.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            writer.writerow({key: _clean(value) for key, value in row.items()})
    temporary.replace(output / "per_patient.csv")
    summary = output / "summary.json.tmp"
    summary.write_text(json.dumps(result.metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    summary.replace(output / "summary.json")
    return output


def build_eval_loader(
    archive_path: str | Path,
    manifest_path: str | Path,
    inspection_config: str | Path,
    *,
    split: str = "validation",
    patients: Sequence[str] | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
    cache_dir: str | Path | None = None,
) -> torch.utils.data.DataLoader:
    """Deterministic, patient-contiguous loader over a manifest split."""

    from cap6411_atm.data.dataset import BraTSAxialSliceDataset
    from cap6411_atm.training.sampling import PatientLocalBatchSampler, PatientSubset

    if split == "test":
        raise PermissionError("build the test loader explicitly only for the final evaluation")
    dataset = BraTSAxialSliceDataset(
        archive_path, manifest_path, split, inspection_config, cache_dir=cache_dir
    )
    if patients is not None:
        dataset = PatientSubset(dataset, list(patients))
    sampler = PatientLocalBatchSampler(
        dataset, batch_size=batch_size, seed=0, shuffle_patients=False, shuffle_slices=False
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
