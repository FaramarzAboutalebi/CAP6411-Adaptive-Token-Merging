"""Reproducible patient-level splits for BraTS-style archives."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .inspection import InspectionConfig, structural_inventory


DEFAULT_SPLIT_PERCENTAGES = {"train": 0.70, "validation": 0.15, "test": 0.15}


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic patient-level partitioning."""

    seed: int = 42
    percentages: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SPLIT_PERCENTAGES)
    )

    def __post_init__(self) -> None:
        if not self.percentages:
            raise ValueError("percentages must not be empty")
        if any(value < 0 for value in self.percentages.values()):
            raise ValueError("split percentages must be non-negative")
        if abs(sum(self.percentages.values()) - 1.0) > 1e-12:
            raise ValueError("split percentages must sum to 1.0")


def allocate_split_counts(total: int, percentages: Mapping[str, float]) -> dict[str, int]:
    """Allocate counts using cumulative half-up boundaries.

    Cumulative boundaries avoid independently rounded counts that do not sum to
    the cohort size. With 1,251 patients and 70/15/15, boundaries are 876 and
    1,063, producing 876 train, 187 validation, and 188 test patients.
    """

    if total < 0:
        raise ValueError("total must be non-negative")
    if not percentages or abs(sum(percentages.values()) - 1.0) > 1e-12:
        raise ValueError("split percentages must sum to 1.0")

    counts: dict[str, int] = {}
    previous_boundary = 0
    cumulative = 0.0
    names = list(percentages)
    for index, name in enumerate(names):
        cumulative += percentages[name]
        boundary = total if index == len(names) - 1 else int(total * cumulative + 0.5)
        counts[name] = boundary - previous_boundary
        previous_boundary = boundary
    return counts


def split_patient_ids(
    patient_ids: Iterable[str],
    config: SplitConfig = SplitConfig(),
) -> dict[str, list[str]]:
    """Return deterministic, disjoint patient lists.

    IDs are sorted before shuffling so results do not depend on archive member or
    caller iteration order.
    """

    ids = list(patient_ids)
    if not ids:
        raise ValueError("patient_ids must not be empty")
    if any(not patient_id for patient_id in ids):
        raise ValueError("patient IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("patient IDs must be unique")

    shuffled = sorted(ids)
    random.Random(config.seed).shuffle(shuffled)
    counts = allocate_split_counts(len(shuffled), config.percentages)

    splits: dict[str, list[str]] = {}
    start = 0
    for name, count in counts.items():
        splits[name] = shuffled[start : start + count]
        start += count
    validate_splits(splits, expected_patient_ids=set(ids))
    return splits


def validate_splits(
    splits: Mapping[str, Sequence[str]],
    *,
    expected_patient_ids: set[str] | None = None,
) -> dict[str, bool]:
    """Validate uniqueness, disjointness, and complete patient coverage."""

    flattened = [patient_id for patients in splits.values() for patient_id in patients]
    unique_within_splits = all(len(patients) == len(set(patients)) for patients in splits.values())
    globally_unique = len(flattened) == len(set(flattened))
    observed = set(flattened)
    complete = expected_patient_ids is None or observed == expected_patient_ids
    validation = {
        "unique_within_splits": unique_within_splits,
        "sets_disjoint": globally_unique,
        "complete_patient_coverage": complete,
    }
    if not all(validation.values()):
        raise ValueError(f"invalid patient split: {validation}")
    return validation


def patient_to_split(splits: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Build the patient-level lookup future slice datasets must use."""

    validate_splits(splits)
    return {
        patient_id: split_name
        for split_name, patient_ids in splits.items()
        for patient_id in patient_ids
    }


def build_split_manifest(
    patient_ids: Iterable[str],
    *,
    split_config: SplitConfig = SplitConfig(),
    dataset_name: str,
    dataset_source: str,
    archive: str | Path | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a self-describing JSON-compatible split manifest."""

    original_ids = list(patient_ids)
    splits = split_patient_ids(original_ids, split_config)
    validation = validate_splits(splits, expected_patient_ids=set(original_ids))
    generated_at = generated_at_utc or datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "dataset": {
            "name": dataset_name,
            "source": dataset_source,
            "archive": str(archive) if archive is not None else None,
        },
        "generation": {
            "generated_at_utc": generated_at,
            "seed": split_config.seed,
            "algorithm": "sort IDs; Python random.Random(seed).shuffle; cumulative half-up boundaries",
            "python_version": platform.python_version(),
        },
        "split_percentages": dict(split_config.percentages),
        "patient_counts": {name: len(ids) for name, ids in splits.items()},
        "total_patients": len(original_ids),
        "validation": validation,
        "splits": splits,
    }


def build_manifest_from_archive(
    archive: str | Path,
    inspection_config: InspectionConfig,
    *,
    split_config: SplitConfig = SplitConfig(),
    dataset_name: str,
    dataset_source: str,
    expected_patient_count: int | None = None,
) -> dict[str, Any]:
    """Inventory an archive structurally and split only complete patients."""

    inventory = structural_inventory(archive, inspection_config)
    incomplete = [report.patient_id for report in inventory if report.status != "complete"]
    if incomplete:
        preview = ", ".join(incomplete[:5])
        raise ValueError(
            f"cannot split archive with {len(incomplete)} incomplete patients; first IDs: {preview}"
        )
    if expected_patient_count is not None and len(inventory) != expected_patient_count:
        raise ValueError(
            f"discovered {len(inventory)} patients; expected {expected_patient_count}"
        )
    return build_split_manifest(
        (report.patient_id for report in inventory),
        split_config=split_config,
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        archive=archive,
    )


def write_json_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    """Write one row per patient with sufficient metadata to reconstruct the split."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "patient_id",
        "split",
        "dataset_name",
        "dataset_source",
        "archive",
        "seed",
        "split_percentage",
        "split_patient_count",
        "total_patients",
        "generated_at_utc",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split_name, patient_ids in manifest["splits"].items():
            for patient_id in patient_ids:
                writer.writerow(
                    {
                        "patient_id": patient_id,
                        "split": split_name,
                        "dataset_name": manifest["dataset"]["name"],
                        "dataset_source": manifest["dataset"]["source"],
                        "archive": manifest["dataset"]["archive"],
                        "seed": manifest["generation"]["seed"],
                        "split_percentage": manifest["split_percentages"][split_name],
                        "split_patient_count": manifest["patient_counts"][split_name],
                        "total_patients": manifest["total_patients"],
                        "generated_at_utc": manifest["generation"]["generated_at_utc"],
                    }
                )


def load_json_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validation = validate_splits(manifest["splits"])
    if sum(manifest["patient_counts"].values()) != manifest["total_patients"]:
        raise ValueError("manifest patient counts do not sum to total_patients")
    if {name: len(ids) for name, ids in manifest["splits"].items()} != manifest["patient_counts"]:
        raise ValueError("manifest patient_counts do not match split contents")
    manifest["validation"] = validation
    return manifest


def load_csv_manifest(path: str | Path) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            splits.setdefault(row["split"], []).append(row["patient_id"])
    validate_splits(splits)
    return splits


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="BraTS training ZIP archive")
    parser.add_argument("--inspection-config", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True, dest="json_path")
    parser.add_argument("--csv", type=Path, required=True, dest="csv_path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-patient-count", type=int, default=1251)
    parser.add_argument("--dataset-name", default="BraTS 2023 Adult Glioma (BraTS-GLI)")
    parser.add_argument("--dataset-source", default="Synapse syn51514132")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = build_manifest_from_archive(
            args.archive,
            InspectionConfig.from_json(args.inspection_config),
            split_config=SplitConfig(seed=args.seed),
            dataset_name=args.dataset_name,
            dataset_source=args.dataset_source,
            expected_patient_count=args.expected_patient_count,
        )
    except (FileNotFoundError, ValueError) as exc:
        _build_parser().error(str(exc))
    write_json_manifest(manifest, args.json_path)
    write_csv_manifest(manifest, args.csv_path)
    print(
        json.dumps(
            {
                "total_patients": manifest["total_patients"],
                "patient_counts": manifest["patient_counts"],
                "validation": manifest["validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
