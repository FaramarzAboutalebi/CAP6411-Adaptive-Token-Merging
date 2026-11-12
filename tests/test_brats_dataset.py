"""Synthetic tests for the BraTS axial-slice dataset."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from cap6411_atm.data.dataset import (
    BraTSAxialSliceDataset,
    normalize_nonzero_zscore,
    segmentation_to_regions,
)
from cap6411_atm.data.inspection import InspectionConfig
from cap6411_atm.data.splitting import build_split_manifest, write_json_manifest


SUFFIXES = {
    "t1": "-t1n.nii.gz",
    "t1ce": "-t1c.nii.gz",
    "t2": "-t2w.nii.gz",
    "flair": "-t2f.nii.gz",
    "segmentation": "-seg.nii.gz",
}
SHAPE = (4, 5, 6)


class BraTSDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patient_ids = ["BraTS-GLI-00001-000", "BraTS-GLI-00002-000"]
        self.config = InspectionConfig(
            file_suffixes=SUFFIXES,
            expected_shape=SHAPE,
            expected_spacing=(1.0, 1.0, 1.0),
            allowed_segmentation_labels=(0, 1, 2, 3),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_patient(
        self,
        patient_id: str,
        *,
        missing: str | None = None,
        wrong_shape_role: str | None = None,
        non_finite: bool = False,
    ) -> Path:
        folder = self.root / patient_id
        folder.mkdir()
        base = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
        base[0, :, :] = 0
        if non_finite:
            base[1, 0, 0] = np.nan
            base[1, 0, 1] = np.inf
        labels = np.zeros(SHAPE, dtype=np.float32)
        labels[1, 1, 2] = 1
        labels[1, 2, 2] = 2
        labels[1, 3, 2] = 3
        for role, suffix in SUFFIXES.items():
            if role == missing:
                continue
            data = labels if role == "segmentation" else base + MODALITY_OFFSETS[role]
            if role == wrong_shape_role:
                data = np.zeros((3, 5, 6), dtype=np.float32)
            nib.save(nib.Nifti1Image(data, np.eye(4)), folder / f"{patient_id}{suffix}")
        return folder

    def _zip(self, folders: list[Path]) -> Path:
        archive = self.root / "dataset.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for folder in folders:
                for path in folder.iterdir():
                    zf.write(path, f"TrainingData/{folder.name}/{path.name}")
        return archive

    def _manifest(self) -> tuple[Path, dict]:
        manifest = build_split_manifest(
            self.patient_ids,
            dataset_name="synthetic",
            dataset_source="unit test",
            archive="dataset.zip",
            generated_at_utc="2026-11-12T00:00:00+00:00",
        )
        path = self.root / "split.json"
        write_json_manifest(manifest, path)
        return path, manifest

    def _dataset(
        self,
        archive: Path,
        manifest_path: Path,
        split: str,
        **kwargs,
    ) -> BraTSAxialSliceDataset:
        return BraTSAxialSliceDataset(
            archive,
            manifest_path,
            split,
            self.config,
            **kwargs,
        )

    def test_four_channel_input_three_channel_target_and_axial_shape(self) -> None:
        folders = [self._write_patient(patient_id) for patient_id in self.patient_ids]
        archive = self._zip(folders)
        manifest_path, manifest = self._manifest()
        split = next(name for name, ids in manifest["splits"].items() if ids)
        dataset = self._dataset(archive, manifest_path, split, slice_indices=[2])
        sample = dataset[0]

        self.assertEqual(sample["image"].shape, (4, 4, 5))
        self.assertEqual(sample["target"].shape, (3, 4, 5))
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertEqual(sample["target"].dtype, torch.float32)
        batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
        self.assertEqual(batch["image"].shape, (1, 4, 4, 5))
        self.assertEqual(batch["target"].shape, (1, 3, 4, 5))

    def test_wt_tc_et_conversion(self) -> None:
        labels = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        targets = segmentation_to_regions(labels)
        np.testing.assert_array_equal(targets[0], [[0, 1, 1, 1]])
        np.testing.assert_array_equal(targets[1], [[0, 1, 0, 1]])
        np.testing.assert_array_equal(targets[2], [[0, 0, 0, 1]])

    def test_normalization_preserves_background_and_handles_constant_data(self) -> None:
        volume = np.array([0, 1, 2, 3], dtype=np.float32)
        normalized = normalize_nonzero_zscore(volume)
        self.assertEqual(normalized[0], 0)
        self.assertAlmostEqual(float(normalized[1:].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[1:].std()), 1.0, places=6)
        np.testing.assert_array_equal(
            normalize_nonzero_zscore(np.array([0, 5, 5], dtype=np.float32)),
            np.zeros(3, dtype=np.float32),
        )

    def test_non_finite_image_values_become_finite_zero_background(self) -> None:
        folders = [
            self._write_patient(self.patient_ids[0], non_finite=True),
            self._write_patient(self.patient_ids[1]),
        ]
        archive = self._zip(folders)
        manifest_path, manifest = self._manifest()
        split = next(name for name, ids in manifest["splits"].items() if self.patient_ids[0] in ids)
        dataset = self._dataset(archive, manifest_path, split, slice_indices=[0])
        sample = next(item for item in (dataset[i] for i in range(len(dataset))) if item["patient_id"] == self.patient_ids[0])
        self.assertTrue(torch.isfinite(sample["image"]).all())

    def test_missing_file_is_rejected(self) -> None:
        folders = [
            self._write_patient(self.patient_ids[0], missing="flair"),
            self._write_patient(self.patient_ids[1]),
        ]
        archive = self._zip(folders)
        manifest_path, manifest = self._manifest()
        split = next(name for name, ids in manifest["splits"].items() if self.patient_ids[0] in ids)
        with self.assertRaisesRegex(ValueError, "incomplete patient"):
            self._dataset(archive, manifest_path, split)

    def test_invalid_dimensions_are_rejected_when_patient_loads(self) -> None:
        folders = [
            self._write_patient(self.patient_ids[0], wrong_shape_role="t2"),
            self._write_patient(self.patient_ids[1]),
        ]
        archive = self._zip(folders)
        manifest_path, manifest = self._manifest()
        split = next(name for name, ids in manifest["splits"].items() if self.patient_ids[0] in ids)
        dataset = self._dataset(archive, manifest_path, split, slice_indices=[0])
        index = next(i for i, (patient_id, _) in enumerate(dataset.samples) if patient_id == self.patient_ids[0])
        with self.assertRaisesRegex(ValueError, "shape"):
            dataset[index]

    def test_patient_split_enforcement_and_reproducible_index(self) -> None:
        folders = [self._write_patient(patient_id) for patient_id in self.patient_ids]
        archive = self._zip(folders)
        manifest_path, manifest = self._manifest()
        split = next(name for name, ids in manifest["splits"].items() if ids)
        first = self._dataset(archive, manifest_path, split, slice_indices=[1, 3])
        second = self._dataset(archive, manifest_path, split, slice_indices=[1, 3])
        self.assertEqual(first.samples, second.samples)
        self.assertEqual({patient_id for patient_id, _ in first.samples}, set(manifest["splits"][split]))
        other_patients = set(self.patient_ids) - set(manifest["splits"][split])
        self.assertTrue(other_patients.isdisjoint(patient_id for patient_id, _ in first.samples))

    def test_disk_cache_reuses_preprocessed_patient(self) -> None:
        folders = [self._write_patient(patient_id) for patient_id in self.patient_ids]
        archive = self._zip(folders)
        manifest_path, manifest = self._manifest()
        split = next(name for name, ids in manifest["splits"].items() if ids)
        cache_dir = self.root / "cache"
        first = self._dataset(archive, manifest_path, split, slice_indices=[2], cache_dir=cache_dir)
        first_sample = first[0]
        second = self._dataset(archive, manifest_path, split, slice_indices=[2], cache_dir=cache_dir)
        second_sample = second[0]
        self.assertTrue(torch.equal(first_sample["image"], second_sample["image"]))
        self.assertTrue((cache_dir / first_sample["patient_id"] / "metadata.json").is_file())


MODALITY_OFFSETS = {"t1": 0, "t1ce": 10, "t2": 20, "flair": 30}


if __name__ == "__main__":
    unittest.main()
