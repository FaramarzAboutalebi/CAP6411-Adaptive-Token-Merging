"""Streaming structural and NIfTI inspection for BraTS-style ZIP archives.

The fast inventory path reads ZIP member names only. NIfTI inspection extracts one
patient at a time to a temporary directory so the complete archive is never
expanded and the cohort is never held in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np


DEFAULT_SUFFIXES = {
    "t1": "-t1n.nii.gz",
    "t1ce": "-t1c.nii.gz",
    "t2": "-t2w.nii.gz",
    "flair": "-t2f.nii.gz",
    "segmentation": "-seg.nii.gz",
}


@dataclass(frozen=True)
class InspectionConfig:
    """Explicit dataset conventions and optional validation expectations."""

    file_suffixes: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_SUFFIXES))
    segmentation_key: str = "segmentation"
    expected_shape: tuple[int, ...] | None = None
    expected_spacing: tuple[float, ...] | None = None
    allowed_segmentation_labels: tuple[int, ...] | None = None
    affine_tolerance: float = 1e-5
    spacing_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if not self.file_suffixes:
            raise ValueError("file_suffixes must not be empty")
        if self.segmentation_key not in self.file_suffixes:
            raise ValueError("segmentation_key must be present in file_suffixes")
        suffixes = list(self.file_suffixes.values())
        if len(suffixes) != len(set(suffixes)):
            raise ValueError("file suffixes must be unique")

    @classmethod
    def from_json(cls, path: str | Path) -> "InspectionConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("expected_shape", "expected_spacing", "allowed_segmentation_labels"):
            if payload.get(key) is not None:
                payload[key] = tuple(payload[key])
        return cls(**payload)


@dataclass
class VolumeReport:
    member: str
    status: str = "uninspected"
    shape: list[int] | None = None
    spacing: list[float] | None = None
    affine: list[list[float]] | None = None
    data_type: str | None = None
    non_finite_count: int | None = None
    label_counts: dict[str, int] | None = None
    unexpected_labels: list[float | int] | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class PatientReport:
    patient_id: str
    status: str
    present_files: dict[str, str]
    missing_files: list[str]
    duplicate_files: dict[str, list[str]]
    volumes: dict[str, VolumeReport] = field(default_factory=dict)
    spatial_alignment: str = "uninspected"
    errors: list[str] = field(default_factory=list)


def _matched_key(filename: str, suffixes: Mapping[str, str]) -> str | None:
    matches = [key for key, suffix in suffixes.items() if filename.endswith(suffix)]
    if len(matches) > 1:
        raise ValueError(f"ambiguous configured suffix match for {filename!r}: {matches}")
    return matches[0] if matches else None


def discover_patients(
    archive: str | Path | zipfile.ZipFile,
    config: InspectionConfig,
) -> dict[str, dict[str, list[str]]]:
    """Map patient IDs to configured file roles and ZIP member names.

    A patient ID is the immediate parent directory of a matching file. Unrelated
    archive contents are ignored. Lists preserve duplicate candidates so the
    caller can report malformed archives instead of silently choosing one.
    """

    close_archive = not isinstance(archive, zipfile.ZipFile)
    zf = zipfile.ZipFile(archive) if close_archive else archive
    try:
        patients: dict[str, dict[str, list[str]]] = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = PurePosixPath(info.filename)
            key = _matched_key(member.name, config.file_suffixes)
            if key is None or not member.parent.name:
                continue
            patient_id = member.parent.name
            patients.setdefault(patient_id, {}).setdefault(key, []).append(info.filename)
        return dict(sorted(patients.items()))
    finally:
        if close_archive:
            zf.close()


def structural_inventory(
    archive: str | Path,
    config: InspectionConfig,
) -> list[PatientReport]:
    """Perform a fast filename-only inventory without reading NIfTI data."""

    with zipfile.ZipFile(archive) as zf:
        discovered = discover_patients(zf, config)
    expected = tuple(config.file_suffixes)
    reports: list[PatientReport] = []
    for patient_id, files in discovered.items():
        missing = [key for key in expected if key not in files]
        duplicates = {key: members for key, members in files.items() if len(members) > 1}
        present = {key: members[0] for key, members in files.items() if len(members) == 1}
        status = "complete" if not missing and not duplicates else "incomplete"
        reports.append(
            PatientReport(
                patient_id=patient_id,
                status=status,
                present_files=present,
                missing_files=missing,
                duplicate_files=duplicates,
                volumes={key: VolumeReport(member=member) for key, member in present.items()},
            )
        )
    return reports


def _safe_extract_member(zf: zipfile.ZipFile, member: str, destination: Path) -> Path:
    """Extract one member while rejecting paths outside the temporary directory."""

    target = (destination / PurePosixPath(member)).resolve()
    root = destination.resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"unsafe ZIP member path: {member}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as source, target.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)
    return target


def _as_json_number(value: np.generic | float | int) -> float | int:
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    numeric = float(scalar)
    return int(numeric) if numeric.is_integer() else numeric


def _scan_finite_values(image: nib.spatialimages.SpatialImage) -> int:
    """Count non-finite voxels without requesting float64 conversion."""

    data = np.asanyarray(image.dataobj)
    return int(data.size - np.count_nonzero(np.isfinite(data)))


def _scan_segmentation(
    image: nib.spatialimages.SpatialImage,
    allowed_labels: Sequence[int] | None,
) -> tuple[dict[str, int], list[float | int]]:
    data = np.asanyarray(image.dataobj)
    values, counts = np.unique(data, return_counts=True)
    label_counts = {str(_as_json_number(value)): int(count) for value, count in zip(values, counts)}
    allowed = set(allowed_labels) if allowed_labels is not None else None
    unexpected = (
        [_as_json_number(value) for value in values if _as_json_number(value) not in allowed]
        if allowed is not None
        else []
    )
    return label_counts, unexpected


def inspect_patient(
    archive: str | Path,
    patient: PatientReport,
    config: InspectionConfig,
    *,
    inspect_headers: bool = True,
    scan_values: bool = False,
    scan_labels: bool = False,
) -> PatientReport:
    """Inspect one patient, updating and returning its structural report."""

    if not inspect_headers and (scan_values or scan_labels):
        inspect_headers = True
    if not patient.present_files:
        return patient

    images: dict[str, nib.spatialimages.SpatialImage] = {}
    with zipfile.ZipFile(archive) as zf, tempfile.TemporaryDirectory(
        prefix=f"brats-inspect-{patient.patient_id}-"
    ) as temp_dir:
        temp_root = Path(temp_dir)
        for key, member in patient.present_files.items():
            volume = patient.volumes[key]
            if not inspect_headers:
                continue
            try:
                path = _safe_extract_member(zf, member, temp_root)
                image = nib.load(path)
                images[key] = image
                volume.status = "header_inspected"
                volume.shape = [int(value) for value in image.shape]
                volume.spacing = [float(value) for value in image.header.get_zooms()[: len(image.shape)]]
                volume.affine = np.asarray(image.affine, dtype=float).tolist()
                volume.data_type = str(image.get_data_dtype())

                if config.expected_shape is not None and tuple(image.shape) != config.expected_shape:
                    volume.errors.append(
                        f"shape {tuple(image.shape)} does not match expected {config.expected_shape}"
                    )
                if config.expected_spacing is not None and not np.allclose(
                    volume.spacing,
                    config.expected_spacing,
                    atol=config.spacing_tolerance,
                    rtol=0,
                ):
                    volume.errors.append(
                        f"spacing {tuple(volume.spacing)} does not match expected "
                        f"{config.expected_spacing}"
                    )
                if scan_values:
                    volume.non_finite_count = _scan_finite_values(image)
                    volume.status = "values_scanned"
                    if volume.non_finite_count:
                        volume.errors.append(f"found {volume.non_finite_count} non-finite voxels")
                if scan_labels and key == config.segmentation_key:
                    volume.label_counts, volume.unexpected_labels = _scan_segmentation(
                        image, config.allowed_segmentation_labels
                    )
                    volume.status = "labels_scanned"
                    if volume.unexpected_labels:
                        volume.errors.append(
                            f"unexpected segmentation labels: {volume.unexpected_labels}"
                        )
            except Exception as exc:  # report malformed members without aborting the cohort
                volume.status = "error"
                volume.errors.append(f"{type(exc).__name__}: {exc}")

        if not inspect_headers:
            patient.spatial_alignment = "uninspected"
        elif len(images) != len(patient.present_files) or patient.missing_files or patient.duplicate_files:
            patient.spatial_alignment = "not_verifiable"
        elif images:
            reference_key, reference = next(iter(images.items()))
            mismatches: list[str] = []
            reference_spacing = reference.header.get_zooms()[: len(reference.shape)]
            for key, image in list(images.items())[1:]:
                if image.shape != reference.shape:
                    mismatches.append(f"{key} shape differs from {reference_key}")
                if not np.allclose(
                    image.header.get_zooms()[: len(image.shape)],
                    reference_spacing,
                    atol=config.spacing_tolerance,
                    rtol=0,
                ):
                    mismatches.append(f"{key} spacing differs from {reference_key}")
                if not np.allclose(
                    image.affine,
                    reference.affine,
                    atol=config.affine_tolerance,
                    rtol=0,
                ):
                    mismatches.append(f"{key} affine differs from {reference_key}")
            patient.spatial_alignment = "aligned" if not mismatches else "mismatch"
            patient.errors.extend(mismatches)
        else:
            patient.spatial_alignment = "not_verifiable"

    if patient.errors or any(volume.errors for volume in patient.volumes.values()):
        patient.status = "issues_found"
    return patient


def inspect_archive(
    archive: str | Path,
    config: InspectionConfig,
    *,
    patient_ids: Iterable[str] | None = None,
    inspect_headers: bool = False,
    scan_values: bool = False,
    scan_labels: bool = False,
) -> dict[str, Any]:
    """Inspect an archive with bounded memory, optionally selecting patients."""

    archive_path = Path(archive)
    reports = structural_inventory(archive_path, config)
    discovered_patient_count = len(reports)
    selected = set(patient_ids) if patient_ids is not None else None
    if selected is not None:
        known = {report.patient_id for report in reports}
        missing_requested = sorted(selected - known)
        reports = [report for report in reports if report.patient_id in selected]
    else:
        missing_requested = []

    if inspect_headers or scan_values or scan_labels:
        for report in reports:
            inspect_patient(
                archive_path,
                report,
                config,
                inspect_headers=inspect_headers,
                scan_values=scan_values,
                scan_labels=scan_labels,
            )

    status_counts = Counter(report.status for report in reports)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive_path),
        "config": asdict(config),
        "inspection": {
            "headers": bool(inspect_headers or scan_values or scan_labels),
            "finite_values": scan_values,
            "segmentation_labels": scan_labels,
        },
        "summary": {
            "discovered_patient_count": discovered_patient_count,
            "patient_count": len(reports),
            "complete_count": sum(not report.missing_files and not report.duplicate_files for report in reports),
            "incomplete_count": sum(bool(report.missing_files or report.duplicate_files) for report in reports),
            "status_counts": dict(sorted(status_counts.items())),
            "requested_patients_not_found": missing_requested,
        },
        "patients": [asdict(report) for report in reports],
    }


def write_json_report(report: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_report(report: Mapping[str, Any], path: str | Path) -> None:
    """Write one deterministic summary row per patient."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "status",
        "missing_files",
        "duplicate_files",
        "spatial_alignment",
        "errors",
        "segmentation_label_counts",
        "unexpected_segmentation_labels",
    ]
    segmentation_key = report["config"]["segmentation_key"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for patient in report["patients"]:
            segmentation = patient["volumes"].get(segmentation_key, {})
            all_errors = list(patient["errors"])
            for key, volume in patient["volumes"].items():
                all_errors.extend(f"{key}: {error}" for error in volume["errors"])
            writer.writerow(
                {
                    "patient_id": patient["patient_id"],
                    "status": patient["status"],
                    "missing_files": ";".join(patient["missing_files"]),
                    "duplicate_files": json.dumps(patient["duplicate_files"], sort_keys=True),
                    "spatial_alignment": patient["spatial_alignment"],
                    "errors": ";".join(all_errors),
                    "segmentation_label_counts": json.dumps(
                        segmentation.get("label_counts"), sort_keys=True
                    ),
                    "unexpected_segmentation_labels": json.dumps(
                        segmentation.get("unexpected_labels")
                    ),
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="BraTS-style ZIP archive")
    parser.add_argument("--config", type=Path, required=True, help="inspection JSON config")
    parser.add_argument("--patient", action="append", dest="patients", help="patient ID; repeatable")
    parser.add_argument("--inspect-headers", action="store_true", help="read NIfTI headers")
    parser.add_argument("--scan-values", action="store_true", help="scan all selected volumes for NaN/Inf")
    parser.add_argument("--scan-labels", action="store_true", help="scan selected segmentation volumes")
    parser.add_argument("--json", type=Path, dest="json_path", help="write full JSON report")
    parser.add_argument("--csv", type=Path, dest="csv_path", help="write patient summary CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = InspectionConfig.from_json(args.config)
    try:
        report = inspect_archive(
            args.archive,
            config,
            patient_ids=args.patients,
            inspect_headers=args.inspect_headers,
            scan_values=args.scan_values,
            scan_labels=args.scan_labels,
        )
    except (FileNotFoundError, zipfile.BadZipFile, ValueError) as exc:
        _build_parser().error(str(exc))
    if args.json_path:
        write_json_report(report, args.json_path)
    if args.csv_path:
        write_csv_report(report, args.csv_path)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    summary = report["summary"]
    has_failures = bool(
        not summary["patient_count"]
        or summary["incomplete_count"]
        or summary["status_counts"].get("issues_found")
        or summary["requested_patients_not_found"]
    )
    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
