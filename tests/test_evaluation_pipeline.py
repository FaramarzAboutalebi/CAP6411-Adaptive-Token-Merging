"""Evaluator, statistics, ablation and figure tests on a synthetic slice dataset."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from cap6411_atm.evaluation import (
    POSTPROCESSING_ABLATION,
    VariantSpec,
    evaluate,
    evaluate_model,
    global_dice,
    paired_comparison,
    patient_metrics,
    run_ablation,
    save_evaluation,
    summarize,
)
from cap6411_atm.evaluation.statistics import (
    boundary_profile,
    failure_cases,
    format_table,
    holm_adjust,
    lesion_detection,
    load_results,
    patch_profile,
    stratify_by_tumor_size,
)
from cap6411_atm.training.metrics import StreamingDice
from cap6411_atm.training.sampling import PatientLocalBatchSampler

H = W = 32
DEPTH = 6
PROVENANCE = {
    "checkpoint": {"name": "oracle", "sha256": "0" * 64},
    "dataset": {"name": "synthetic", "sha256": "1" * 64},
    "manifest": {"name": "synthetic_split", "sha256": "2" * 64},
}


class SyntheticSliceDataset(Dataset):
    """Patient volumes with a cubic tumor; images carry the target for the fake model."""

    def __init__(self, patients=("P0", "P1", "P2", "P3"), split="validation") -> None:
        self.split = split
        self.axial_depth = DEPTH
        self.patient_ids = tuple(patients)
        self.samples = tuple((p, z) for p in patients for z in range(DEPTH))
        rng = np.random.default_rng(0)
        self.targets = {}
        for index, patient in enumerate(patients):
            target = np.zeros((3, H, W, DEPTH), dtype=np.float32)
            size = 6 + 2 * index
            x, y = rng.integers(0, H - size, 2)
            target[0, x : x + size, y : y + size, 1:5] = 1
            target[1, x + 1 : x + size - 1, y + 1 : y + size - 1, 1:5] = 1
            if index != 0:  # patient P0 is ET-negative
                target[2, x + 2 : x + size - 2, y + 2 : y + size - 2, 2:4] = 1
            self.targets[patient] = target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        patient, z = self.samples[index]
        target = torch.from_numpy(self.targets[patient][..., z].copy())
        image = torch.cat([target, torch.zeros(1, H, W)], dim=0)  # 4 channels
        return {"image": image, "target": target, "patient_id": patient, "slice_index": z, "split": self.split}


class OracleModel(torch.nn.Module):
    """Returns confident logits for the target, optionally shifted or dropping ET."""

    def __init__(self, shift: int = 0, drop_et: bool = False) -> None:
        super().__init__()
        self.shift = shift
        self.drop_et = drop_et

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        masks = images[:, :3].clone()
        if self.shift:
            masks = torch.roll(masks, shifts=self.shift, dims=2)
        if self.drop_et:
            masks[:, 2] = 0
        return (masks * 2 - 1) * 8.0


def make_loader(dataset: Dataset, batch_size: int = 4, shuffle: bool = False) -> DataLoader:
    sampler = PatientLocalBatchSampler(
        dataset, batch_size=batch_size, seed=0, shuffle_patients=shuffle, shuffle_slices=shuffle
    )
    return DataLoader(dataset, batch_sampler=sampler)


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = SyntheticSliceDataset()
        self.loader = make_loader(self.dataset)

    def test_oracle_is_perfect_and_rows_are_complete(self) -> None:
        result = evaluate_model(OracleModel(), self.loader, device="cpu", rules=POSTPROCESSING_ABLATION)
        self.assertEqual(len(result.rows), 4 * 3 * len(POSTPROCESSING_ABLATION))
        self.assertTrue(all(row["dice"] == 1.0 for row in result.rows if row["rule"] in ("raw", "nested")))
        # Synthetic ET lesions are 32 voxels, so the 50-voxel component filter must remove them.
        removed = [r for r in result.rows if r["rule"] == "nested_cc50" and r["region"] == "ET" and r["patient_id"] == "P1"]
        self.assertEqual(removed[0]["status"], "pred_empty")
        self.assertEqual(result.metadata["patients"], 4)
        self.assertEqual(result.metadata["slices"], 4 * DEPTH)
        self.assertEqual(result.metadata["global_dice"]["raw"]["macro"], 1.0)
        statuses = {(row["patient_id"], row["region"]): row["status"] for row in result.rows}
        self.assertEqual(statuses[("P0", "ET")], "both_empty")

    def test_volume_metrics_match_direct_computation(self) -> None:
        result = evaluate_model(OracleModel(shift=1), self.loader, device="cpu")
        target = self.dataset.targets["P2"] >= 0.5
        expected = patient_metrics(np.roll(target, 1, axis=1), target)
        row = next(r for r in result.rows if r["patient_id"] == "P2" and r["region"] == "TC")
        self.assertAlmostEqual(row["dice"], expected["TC"]["dice"])
        self.assertAlmostEqual(row["hd95"], expected["TC"]["hd95"])

    def test_global_dice_matches_streaming_dice(self) -> None:
        model = OracleModel(shift=2)
        streaming = StreamingDice(0.5)
        with torch.inference_mode():
            for batch in self.loader:
                streaming.update(model(batch["image"]), batch["target"])
        expected = streaming.compute()
        result = evaluate_model(model, self.loader, device="cpu")
        computed = result.metadata["global_dice"]["raw"]
        for region in ("WT", "TC", "ET"):
            self.assertAlmostEqual(computed[region], expected["dice"][region])
        self.assertAlmostEqual(computed["macro"], expected["macro_dice"])

    def test_test_split_is_refused(self) -> None:
        loader = make_loader(SyntheticSliceDataset(split="test"))
        with self.assertRaises(PermissionError):
            evaluate_model(OracleModel(), loader, device="cpu")
        evaluate_model(OracleModel(), loader, device="cpu", allow_test=True)

    def test_non_contiguous_patients_are_rejected(self) -> None:
        class Interleaved:
            dataset = self.dataset

            def __iter__(inner):
                for z in range(DEPTH):
                    for patient in ("P0", "P1"):
                        item = self.dataset[self.dataset.samples.index((patient, z))]
                        yield {
                            "image": item["image"][None],
                            "target": item["target"][None],
                            "patient_id": [patient],
                            "slice_index": torch.tensor([z]),
                            "split": ["validation"],
                        }

        with self.assertRaises(RuntimeError):
            evaluate_model(OracleModel(), Interleaved(), device="cpu")

    def test_saved_predictions_and_csv(self) -> None:
        result = evaluate_model(OracleModel(), self.loader, device="cpu", save_patients=["P1"])
        saved = result.saved_predictions["P1"]
        self.assertEqual(saved["raw"].shape, (3, H, W, DEPTH))
        self.assertEqual(saved["image"].shape, (4, H, W, DEPTH))
        with tempfile.TemporaryDirectory() as directory:
            save_evaluation(result, directory)
            frame = pd.read_csv(Path(directory) / "per_patient.csv")
            self.assertEqual(len(frame), 12)
            self.assertTrue((Path(directory) / "summary.json").is_file())

    def test_batch_level_evaluate(self) -> None:
        batch = next(iter(self.loader))
        output = evaluate(OracleModel()(batch["image"]), batch["target"])
        self.assertEqual(output["WT"]["dice"], 1.0)
        self.assertIn("patch_boundary_error_rate", output["TC"])


class StatisticsAndAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = SyntheticSliceDataset(patients=tuple(f"P{i}" for i in range(8)))
        cls.loader = make_loader(cls.dataset)
        cls.tmp = tempfile.TemporaryDirectory()
        variants = [
            VariantSpec("baseline", rules={"raw": {"threshold": 0.5}, "nested": {"threshold": 0.5, "nesting": True}}),
            VariantSpec("uniform_r0.25", method="uniform", merge_ratio=0.25),
            VariantSpec("adaptive_r0.25", method="adaptive", merge_ratio=0.25),
        ]
        models = {"baseline": OracleModel(), "uniform_r0.25": OracleModel(shift=2), "adaptive_r0.25": OracleModel(shift=1)}
        cls.built: list[str] = []

        def build(spec: VariantSpec):
            cls.built.append(spec.name)
            return models[spec.name]

        cls.variants, cls.build = variants, build
        cls.frame = run_ablation(variants, build, cls.loader, cls.tmp.name, provenance=PROVENANCE, device="cpu", progress=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_ablation_resumes_without_rebuilding(self) -> None:
        before = list(self.built)
        run_ablation(self.variants, self.build, self.loader, self.tmp.name, provenance=PROVENANCE, device="cpu", progress=False)
        self.assertEqual(self.built, before)
        self.assertEqual(set(self.frame["method"]), {"baseline", "baseline/nested", "uniform_r0.25", "adaptive_r0.25"})
        self.assertEqual(set(self.frame["family"]), {"baseline", "uniform", "adaptive"})

    def test_summary_and_global_dice(self) -> None:
        summary = summarize(self.frame, metrics=["dice", "hd95"])
        baseline = summary[(summary["method"] == "baseline") & (summary["metric"] == "dice")]
        self.assertTrue(np.allclose(baseline["mean"], 1.0))
        table = global_dice(self.frame)
        self.assertAlmostEqual(table.loc["baseline", "macro"], 1.0)
        self.assertLess(table.loc["uniform_r0.25", "WT"], table.loc["adaptive_r0.25", "WT"])
        self.assertIsInstance(format_table(summary, metrics=["dice"]), str)
        self.assertIn(r"\pm", format_table(summary, metrics=["dice"], latex=True))

    def test_paired_comparison_direction(self) -> None:
        comparison = paired_comparison(self.frame, "baseline", metrics=["dice", "hd95"], regions=["WT"])
        dice = comparison[(comparison["metric"] == "dice") & (comparison["method"] == "uniform_r0.25")].iloc[0]
        self.assertLess(dice["mean_diff"], 0)
        self.assertEqual(dice["worse"], 8)
        hd = comparison[(comparison["metric"] == "hd95") & (comparison["method"] == "uniform_r0.25")].iloc[0]
        self.assertGreater(hd["mean_diff"], 0)
        self.assertEqual(hd["worse"], 8)
        self.assertTrue(0 <= hd["p_holm"] <= 1)

    def test_holm(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03, np.nan])
        np.testing.assert_allclose(adjusted[:3], [0.03, 0.06, 0.06])
        self.assertTrue(np.isnan(adjusted[3]))

    def test_profiles_and_failures(self) -> None:
        bands = boundary_profile(self.frame, region="WT")
        self.assertTrue(np.allclose(bands.loc["uniform_r0.25"].sum(), 1.0))
        patches = patch_profile(self.frame, region="WT")
        self.assertEqual(patches.loc["baseline", "boundary"], 0.0)
        lesions = lesion_detection(self.frame, region="ET")
        self.assertIn("fp_components_per_patient", lesions.columns)
        worst = failure_cases(self.frame, "baseline", "uniform_r0.25", region="WT", k=3)
        self.assertEqual(len(worst), 3)
        self.assertTrue((worst["delta"] <= 0).all())
        strata = stratify_by_tumor_size(self.frame, metric_region="WT")
        self.assertEqual(strata.shape[0], 4)

    def test_figures(self) -> None:
        from cap6411_atm.evaluation import visualization as viz

        target = self.dataset.targets["P3"] >= 0.5
        z = viz.pick_slice(target, "WT")
        image = np.concatenate([target, np.zeros((1, H, W, DEPTH))])[..., z]
        figures = [
            viz.plot_case(image, target[..., z], {"baseline": target[..., z], "shifted": np.roll(target[..., z], 2, axis=1)}),
            viz.plot_merge_map(image, np.array([0, 0, 1, 2]), target=target[..., z]),
            viz.plot_metric_vs_x(self.frame, metric="dice", region="WT"),
            viz.plot_paired_differences(self.frame, "baseline", region="WT"),
            viz.plot_profile(patch_profile(self.frame), ylabel="error rate"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = viz.save_figure(figures[0], Path(directory) / "case")
            self.assertTrue(all(path.is_file() for path in paths))
        for figure in figures:
            matplotlib.pyplot.close(figure)


if __name__ == "__main__":
    unittest.main()
