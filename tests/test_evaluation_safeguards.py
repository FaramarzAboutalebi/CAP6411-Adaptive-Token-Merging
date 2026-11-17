"""Regression tests for the evaluation safeguards requested in the PR #2 review.

Covers: complete patient volumes, stale/mixed ablation results, provenance,
split guards, paired-comparison populations and duplicate rows.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

try:
    from test_evaluation_pipeline import DEPTH, PROVENANCE, OracleModel, SyntheticSliceDataset, make_loader
except ImportError:  # run as tests.test_evaluation_safeguards
    from tests.test_evaluation_pipeline import DEPTH, PROVENANCE, OracleModel, SyntheticSliceDataset, make_loader

from cap6411_atm.evaluation import (
    VariantSpec,
    build_eval_loader,
    build_provenance,
    collect_results,
    evaluate_model,
    fingerprint_file,
    paired_comparison,
    run_ablation,
)
from cap6411_atm.evaluation import provenance as provenance_module
from cap6411_atm.evaluation.statistics import failure_cases, load_results


class DroppedSliceDataset(SyntheticSliceDataset):
    """Synthetic dataset where patient P1 is missing one axial slice."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.samples = tuple(sample for sample in self.samples if sample != ("P1", 3))


class NoDepthDataset(SyntheticSliceDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        del self.axial_depth


class NoSplitDataset(SyntheticSliceDataset):
    """Items are labeled validation, but the dataset declares no split."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        del self.split

    def __getitem__(self, index):
        self.__dict__["split"] = "validation"
        try:
            return super().__getitem__(index)
        finally:
            del self.__dict__["split"]


class Wrapper(torch.utils.data.Dataset):
    """Dataset wrapper declaring its own split (e.g. a mislabeled subset)."""

    def __init__(self, dataset, split) -> None:
        self.dataset, self.split = dataset, split
        self.samples, self.patient_ids = dataset.samples, dataset.patient_ids

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class MislabeledItems(SyntheticSliceDataset):
    """Dataset says validation, but its items claim the test split."""

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["split"] = "test"
        return item


class CompleteVolumeTests(unittest.TestCase):
    def test_missing_slice_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "incomplete volume for patient P1"):
            evaluate_model(OracleModel(), make_loader(DroppedSliceDataset()), device="cpu")

    def test_wrong_depth_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "incomplete volume"):
            evaluate_model(OracleModel(), make_loader(SyntheticSliceDataset()), device="cpu", expected_depth=DEPTH + 1)

    def test_unknown_depth_requires_explicit_choice(self) -> None:
        loader = make_loader(NoDepthDataset())
        with self.assertRaisesRegex(ValueError, "expected_depth"):
            evaluate_model(OracleModel(), loader, device="cpu")
        result = evaluate_model(OracleModel(), loader, device="cpu", expected_depth=DEPTH)
        self.assertEqual(result.metadata["expected_depth"], DEPTH)

    def test_partial_volumes_only_when_requested(self) -> None:
        result = evaluate_model(
            OracleModel(), make_loader(DroppedSliceDataset()), device="cpu", allow_partial_volumes=True
        )
        self.assertFalse(result.metadata["complete_volumes"])

    def test_missing_patient_is_rejected(self) -> None:
        dataset = SyntheticSliceDataset()
        full = make_loader(dataset)

        class FirstPatientOnly:
            def __init__(self) -> None:
                self.dataset = dataset

            def __iter__(self):
                for batch in full:
                    if set(batch["patient_id"]) == {"P0"}:
                        yield batch

        with self.assertRaisesRegex(RuntimeError, "evaluated patients differ"):
            evaluate_model(OracleModel(), FirstPatientOnly(), device="cpu")

    def test_metadata_records_complete_volumes(self) -> None:
        result = evaluate_model(OracleModel(), make_loader(SyntheticSliceDataset()), device="cpu")
        self.assertTrue(result.metadata["complete_volumes"])
        self.assertEqual(result.metadata["expected_depth"], DEPTH)
        self.assertEqual(len(result.metadata["patient_ids_sha256"]), 64)


class SplitGuardTests(unittest.TestCase):
    def test_missing_split_is_rejected(self) -> None:
        loader = make_loader(NoSplitDataset())
        with self.assertRaisesRegex(ValueError, "cannot determine the evaluated split"):
            evaluate_model(OracleModel(), loader, device="cpu")
        evaluate_model(OracleModel(), loader, device="cpu", split="validation")

    def test_declared_split_must_match_dataset(self) -> None:
        loader = make_loader(SyntheticSliceDataset(split="test"))
        with self.assertRaisesRegex(ValueError, "conflicting split"):
            evaluate_model(OracleModel(), loader, device="cpu", split="validation", allow_test=True)

    def test_wrapper_cannot_relabel_test_split(self) -> None:
        loader = make_loader(Wrapper(SyntheticSliceDataset(split="test"), "validation"))
        with self.assertRaisesRegex(ValueError, "conflicting split"):
            evaluate_model(OracleModel(), loader, device="cpu")

    def test_batch_split_must_match_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts with evaluated split"):
            evaluate_model(OracleModel(), make_loader(MislabeledItems()), device="cpu")

    def test_unknown_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown split"):
            evaluate_model(OracleModel(), make_loader(SyntheticSliceDataset(split="val")), device="cpu")

    def test_test_split_still_needs_allow_test(self) -> None:
        with self.assertRaises(PermissionError):
            evaluate_model(OracleModel(), make_loader(SyntheticSliceDataset(split="test")), device="cpu", split="test")

    def test_eval_loader_never_builds_test(self) -> None:
        with self.assertRaises(PermissionError):
            build_eval_loader("missing.zip", "missing.json", "missing.json", split="test")


class AblationProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        self.loader = make_loader(SyntheticSliceDataset())
        self.variant = VariantSpec("baseline")
        self.built: list[str] = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build(self, spec):
        self.built.append(spec.name)
        return OracleModel()

    def run_once(self, variants=None, loader=None, provenance=PROVENANCE, **kwargs):
        return run_ablation(
            variants or [self.variant], self.build, loader or self.loader, self.output,
            provenance=provenance, device="cpu", progress=False, **kwargs,
        )

    def test_provenance_is_required(self) -> None:
        with self.assertRaises(TypeError):
            run_ablation([self.variant], self.build, self.loader, self.output, device="cpu", progress=False)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.run_once(provenance={"dataset": "x", "manifest": "y"})

    def test_summary_records_provenance_and_run_key(self) -> None:
        self.run_once()
        summary = json.loads((self.output / "baseline" / "summary.json").read_text())
        self.assertEqual(summary["provenance"]["checkpoint"]["name"], "oracle")
        self.assertEqual(len(summary["run_key"]), 64)
        self.assertEqual(summary["run_identity"]["split"], "validation")
        self.assertEqual(summary["run_identity"]["expected_depth"], DEPTH)

    def test_matching_results_are_reused(self) -> None:
        self.run_once()
        self.run_once()
        self.assertEqual(self.built, ["baseline"])

    def test_changed_checkpoint_makes_results_stale(self) -> None:
        self.run_once()
        changed = {**PROVENANCE, "checkpoint": {"name": "other", "sha256": "f" * 64}}
        with self.assertRaisesRegex(RuntimeError, "stale results"):
            self.run_once(provenance=changed)
        self.run_once(provenance=changed, overwrite=True)
        self.assertEqual(self.built, ["baseline", "baseline"])

    def test_changed_rules_make_results_stale(self) -> None:
        self.run_once()
        with self.assertRaisesRegex(RuntimeError, "stale results"):
            self.run_once(variants=[VariantSpec("baseline", rules={"raw": {"threshold": 0.4}})])

    def test_changed_patients_make_results_stale(self) -> None:
        self.run_once()
        smaller = make_loader(SyntheticSliceDataset(patients=("P0", "P1")))
        with self.assertRaisesRegex(RuntimeError, "stale results"):
            self.run_once(loader=smaller)

    def test_edited_csv_is_detected(self) -> None:
        self.run_once()
        csv = self.output / "baseline" / "per_patient.csv"
        csv.write_text(csv.read_text().replace("1.0", "0.9", 1))
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            self.run_once()

    def test_legacy_results_are_not_reused(self) -> None:
        self.run_once()
        summary_path = self.output / "baseline" / "summary.json"
        summary = json.loads(summary_path.read_text())
        del summary["run_key"]
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(RuntimeError, "predate provenance"):
            self.run_once()

    def test_results_from_different_experiments_are_not_mixed(self) -> None:
        self.run_once()
        other = {**PROVENANCE, "dataset": {"name": "other_archive", "sha256": "e" * 64}}
        self.run_once(variants=[VariantSpec("uniform_r0.25", method="uniform", merge_ratio=0.25)], provenance=other)
        with self.assertRaisesRegex(RuntimeError, "different experiments"):
            collect_results(self.output)
        self.assertEqual(len(collect_results(self.output, strict=False)["variant"].unique()), 2)

    def test_per_variant_checkpoints_are_allowed(self) -> None:
        variants = [VariantSpec("baseline"), VariantSpec("adaptive_r0.25", method="adaptive", merge_ratio=0.25)]

        def per_variant(spec):
            return {**PROVENANCE, "checkpoint": {"name": spec.name, "sha256": "a" * 64}}

        frame = self.run_once(variants=variants, provenance=per_variant)
        self.assertEqual(set(frame["variant"]), {"baseline", "adaptive_r0.25"})


class PairedComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        models = {"baseline": OracleModel(), "shifted": OracleModel(shift=1)}
        cls.frame = run_ablation(
            [VariantSpec("baseline"), VariantSpec("shifted")], lambda spec: models[spec.name],
            make_loader(SyntheticSliceDataset()), cls.tmp.name, provenance=PROVENANCE, device="cpu", progress=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_valid_comparison_uses_all_patients(self) -> None:
        table = paired_comparison(self.frame, "baseline", metrics=["dice"], regions=["WT"], n_resamples=200)
        self.assertEqual(int(table.iloc[0]["n_patients"]), 4)

    def test_duplicate_rows_are_rejected(self) -> None:
        doubled = pd.concat([self.frame, self.frame[self.frame["method"] == "shifted"]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_comparison(doubled, "baseline", metrics=["dice"], n_resamples=200)

    def test_different_patient_populations_are_rejected(self) -> None:
        partial = self.frame[~((self.frame["method"] == "shifted") & (self.frame["patient_id"] == "P2"))]
        with self.assertRaisesRegex(ValueError, "different patients"):
            paired_comparison(partial, "baseline", metrics=["dice"], n_resamples=200)
        with self.assertRaisesRegex(ValueError, "different patients"):
            failure_cases(partial, "baseline", "shifted")

    def test_unknown_method_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not in results"):
            paired_comparison(self.frame, "baseline", methods=["missing"], n_resamples=200)

    def test_loading_a_variant_twice_is_rejected(self) -> None:
        directory = Path(self.tmp.name) / "baseline"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_results([directory, directory])


class ProvenanceTests(unittest.TestCase):
    def test_small_file_is_fully_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            path.write_bytes(b"abc" * 1000)
            info = fingerprint_file(path)
            self.assertEqual(info["sha256"], hashlib.sha256(b"abc" * 1000).hexdigest())
            self.assertEqual(info["method"], "sha256")

    def test_large_file_uses_head_and_tail(self) -> None:
        limit, chunk = provenance_module.FULL_HASH_LIMIT, provenance_module.PARTIAL_CHUNK
        provenance_module.FULL_HASH_LIMIT, provenance_module.PARTIAL_CHUNK = 10, 4
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "archive.zip"
                path.write_bytes(b"0123456789abcdef")
                first = fingerprint_file(path)
                path.write_bytes(b"0123XXXXXXXXcdef")  # middle changed: same head/tail
                self.assertEqual(fingerprint_file(path)["sha256"], first["sha256"])
                path.write_bytes(b"0123456789abcdeX")  # tail changed
                self.assertNotEqual(fingerprint_file(path)["sha256"], first["sha256"])
                self.assertEqual(first["method"], "sha256_size_head_tail_64MiB")
        finally:
            provenance_module.FULL_HASH_LIMIT, provenance_module.PARTIAL_CHUNK = limit, chunk

    def test_build_provenance_hashes_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            files = {}
            for name in ("backbone.pth", "adapter.pt", "data.zip", "split.json"):
                files[name] = Path(directory) / name
                files[name].write_bytes(name.encode())
            info = build_provenance(
                checkpoint={"backbone": files["backbone.pth"], "adapter": files["adapter.pt"]},
                dataset=files["data.zip"],
                manifest=files["split.json"],
                note="unit test",
            )
            self.assertEqual(info["checkpoint"]["adapter"]["sha256"], hashlib.sha256(b"adapter.pt").hexdigest())
            self.assertEqual(info["dataset"]["name"], "data.zip")
            self.assertEqual(info["note"], "unit test")

    def test_missing_file_is_an_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_provenance(checkpoint="nope.pt", dataset="nope.zip", manifest="nope.json")


if __name__ == "__main__":
    unittest.main()
