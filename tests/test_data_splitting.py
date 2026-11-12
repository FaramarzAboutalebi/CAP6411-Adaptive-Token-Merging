"""Tests for deterministic patient-level dataset splitting."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cap6411_atm.data.splitting import (
    SplitConfig,
    build_split_manifest,
    load_csv_manifest,
    load_json_manifest,
    patient_to_split,
    split_patient_ids,
    validate_splits,
    write_csv_manifest,
    write_json_manifest,
)


class SplittingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.patient_ids = [f"BraTS-GLI-{index:05d}-000" for index in range(1251)]
        self.config = SplitConfig(seed=42)

    def test_reproducible_and_independent_of_input_order(self) -> None:
        first = split_patient_ids(self.patient_ids, self.config)
        second = split_patient_ids(reversed(self.patient_ids), self.config)
        self.assertEqual(first, second)

    def test_correct_counts(self) -> None:
        splits = split_patient_ids(self.patient_ids, self.config)
        self.assertEqual(
            {name: len(ids) for name, ids in splits.items()},
            {"train": 876, "validation": 187, "test": 188},
        )

    def test_disjoint_and_complete(self) -> None:
        splits = split_patient_ids(self.patient_ids, self.config)
        validation = validate_splits(splits, expected_patient_ids=set(self.patient_ids))
        self.assertTrue(all(validation.values()))
        self.assertEqual(sum(len(ids) for ids in splits.values()), 1251)

    def test_duplicate_patient_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            split_patient_ids([self.patient_ids[0], self.patient_ids[0]], self.config)

    def test_patient_lookup_keeps_all_slices_in_one_split(self) -> None:
        splits = split_patient_ids(self.patient_ids, self.config)
        lookup = patient_to_split(splits)
        synthetic_slices = [
            (patient_id, slice_index, lookup[patient_id])
            for patient_id in self.patient_ids[:10]
            for slice_index in range(155)
        ]
        for patient_id in self.patient_ids[:10]:
            assigned = {split for pid, _, split in synthetic_slices if pid == patient_id}
            self.assertEqual(assigned, {lookup[patient_id]})

    def test_json_and_csv_round_trip(self) -> None:
        manifest = build_split_manifest(
            self.patient_ids,
            split_config=self.config,
            dataset_name="BraTS 2023 Adult Glioma (BraTS-GLI)",
            dataset_source="Synapse syn51514132",
            archive="training.zip",
            generated_at_utc="2026-11-12T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "split.json"
            csv_path = Path(temp_dir) / "split.csv"
            write_json_manifest(manifest, json_path)
            write_csv_manifest(manifest, csv_path)

            loaded_json = load_json_manifest(json_path)
            loaded_csv = load_csv_manifest(csv_path)
            self.assertEqual(loaded_json["splits"], manifest["splits"])
            self.assertEqual(loaded_csv, manifest["splits"])

            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1251)
            self.assertEqual(rows[0]["dataset_source"], "Synapse syn51514132")
            self.assertEqual(rows[0]["seed"], "42")
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["patient_counts"]["train"], 876)


if __name__ == "__main__":
    unittest.main()
