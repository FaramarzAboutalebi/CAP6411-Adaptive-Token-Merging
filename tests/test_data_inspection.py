"""Synthetic tests for BraTS ZIP inspection; no real MRI data is required."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

from cap6411_atm.data.inspection import (
    InspectionConfig,
    inspect_archive,
    write_csv_report,
    write_json_report,
)


SUFFIXES = {
    "t1": "-t1n.nii.gz",
    "t1ce": "-t1c.nii.gz",
    "t2": "-t2w.nii.gz",
    "flair": "-t2f.nii.gz",
    "segmentation": "-seg.nii.gz",
}


class InspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = InspectionConfig(
            file_suffixes=SUFFIXES,
            expected_shape=(4, 5, 6),
            expected_spacing=(1.0, 1.0, 1.0),
            allowed_segmentation_labels=(0, 1, 2, 3),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _nifti(self, path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), path)

    def _patient_files(
        self,
        patient_id: str,
        *,
        missing: str | None = None,
        shape: tuple[int, int, int] = (4, 5, 6),
        affine_overrides: dict[str, np.ndarray] | None = None,
        labels: np.ndarray | None = None,
    ) -> Path:
        folder = self.root / "source" / patient_id
        affine_overrides = affine_overrides or {}
        for key, suffix in SUFFIXES.items():
            if key == missing:
                continue
            data = labels if key == "segmentation" and labels is not None else np.zeros(shape, dtype=np.float32)
            self._nifti(folder / f"{patient_id}{suffix}", data, affine_overrides.get(key))
        return folder

    def _zip(self, *folders: Path) -> Path:
        archive = self.root / "dataset.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for folder in folders:
                for path in folder.iterdir():
                    zf.write(path, f"TrainingData/{folder.name}/{path.name}")
        return archive

    def test_patient_discovery_and_missing_modality(self) -> None:
        complete = self._patient_files("BraTS-GLI-00001-000")
        incomplete = self._patient_files("BraTS-GLI-00002-000", missing="flair")
        report = inspect_archive(self._zip(complete, incomplete), self.config)

        self.assertEqual(report["summary"]["patient_count"], 2)
        self.assertEqual(report["summary"]["complete_count"], 1)
        self.assertEqual(report["summary"]["incomplete_count"], 1)
        patients = {item["patient_id"]: item for item in report["patients"]}
        self.assertEqual(patients["BraTS-GLI-00002-000"]["missing_files"], ["flair"])
        self.assertEqual(
            patients["BraTS-GLI-00001-000"]["volumes"]["t1"]["status"], "uninspected"
        )

    def test_incorrect_dimensions_are_reported(self) -> None:
        folder = self._patient_files("BraTS-GLI-00003-000", shape=(3, 5, 6))
        report = inspect_archive(self._zip(folder), self.config, inspect_headers=True)
        patient = report["patients"][0]

        self.assertEqual(patient["status"], "issues_found")
        self.assertIn("does not match expected", patient["volumes"]["t1"]["errors"][0])

    def test_affine_mismatch_is_reported(self) -> None:
        shifted = np.eye(4)
        shifted[0, 3] = 2.0
        folder = self._patient_files(
            "BraTS-GLI-00004-000", affine_overrides={"segmentation": shifted}
        )
        report = inspect_archive(self._zip(folder), self.config, inspect_headers=True)
        patient = report["patients"][0]

        self.assertEqual(patient["spatial_alignment"], "mismatch")
        self.assertTrue(any("affine differs" in item for item in patient["errors"]))

    def test_unexpected_labels_and_counts(self) -> None:
        labels = np.zeros((4, 5, 6), dtype=np.uint8)
        labels.flat[:4] = [1, 2, 3, 9]
        folder = self._patient_files("BraTS-GLI-00005-000", labels=labels)
        report = inspect_archive(self._zip(folder), self.config, scan_labels=True)
        segmentation = report["patients"][0]["volumes"]["segmentation"]

        self.assertEqual(segmentation["label_counts"], {"0": 116, "1": 1, "2": 1, "3": 1, "9": 1})
        self.assertEqual(segmentation["unexpected_labels"], [9])
        self.assertEqual(report["patients"][0]["status"], "issues_found")

    def test_non_finite_values_are_reported(self) -> None:
        folder = self._patient_files("BraTS-GLI-00006-000")
        data = np.zeros((4, 5, 6), dtype=np.float32)
        data.flat[:2] = [np.nan, np.inf]
        self._nifti(folder / f"{folder.name}{SUFFIXES['t1']}", data)
        report = inspect_archive(self._zip(folder), self.config, scan_values=True)
        t1 = report["patients"][0]["volumes"]["t1"]

        self.assertEqual(t1["non_finite_count"], 2)
        self.assertEqual(report["patients"][0]["status"], "issues_found")

    def test_json_and_csv_reports(self) -> None:
        folder = self._patient_files("BraTS-GLI-00007-000")
        report = inspect_archive(self._zip(folder), self.config, scan_labels=True)
        json_path = self.root / "reports" / "inventory.json"
        csv_path = self.root / "reports" / "inventory.csv"

        write_json_report(report, json_path)
        write_csv_report(report, csv_path)

        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["summary"]["patient_count"], 1)
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["patient_id"], "BraTS-GLI-00007-000")
        self.assertEqual(json.loads(rows[0]["segmentation_label_counts"]), {"0": 120})

    def test_malformed_nifti_does_not_abort_archive(self) -> None:
        folder = self._patient_files("BraTS-GLI-00008-000")
        (folder / f"{folder.name}{SUFFIXES['t2']}").write_bytes(b"not a nifti")
        report = inspect_archive(self._zip(folder), self.config, inspect_headers=True)

        self.assertEqual(report["patients"][0]["volumes"]["t2"]["status"], "error")
        self.assertEqual(report["patients"][0]["spatial_alignment"], "not_verifiable")

    def test_requested_patient_selection_reports_unknown_ids(self) -> None:
        folder = self._patient_files("BraTS-GLI-00009-000")
        report = inspect_archive(
            self._zip(folder),
            self.config,
            patient_ids=["BraTS-GLI-00009-000", "BraTS-GLI-99999-000"],
        )

        self.assertEqual(report["summary"]["discovered_patient_count"], 1)
        self.assertEqual(report["summary"]["patient_count"], 1)
        self.assertEqual(
            report["summary"]["requested_patients_not_found"], ["BraTS-GLI-99999-000"]
        )


if __name__ == "__main__":
    unittest.main()
