"""Synthetic integration coverage across token merging, WS3, and WS4."""

from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from cap6411_atm.benchmark import describe, run_suite
from cap6411_atm.evaluation import VariantSpec, build_model_variant, evaluate_model
from test_dinov3_segmentation import BlockwiseSyntheticDINOv3ViTB16


class _OneSliceValidationDataset(Dataset):
    split = "validation"
    axial_depth = 1
    patient_ids = ("synthetic-patient",)
    samples = (("synthetic-patient", 0),)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(index)
        return {
            "image": torch.randn(4, 240, 240, generator=generator),
            "target": torch.zeros(3, 240, 240),
            "patient_id": "synthetic-patient",
            "slice_index": 0,
            "split": "validation",
        }


class WorkstreamIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specs = (
            VariantSpec("baseline", method="baseline"),
            VariantSpec("uniform_r0.50", method="uniform", retention_ratio=0.5),
            VariantSpec("adaptive_r0.50", method="adaptive", retention_ratio=0.5),
        )

    def _models(self):
        return {
            spec.name: build_model_variant(spec, BlockwiseSyntheticDINOv3ViTB16)
            for spec in self.specs
        }

    def test_ws3_identifies_and_benchmarks_all_variants(self) -> None:
        models = self._models()
        descriptions = {name: describe(model) for name, model in models.items()}
        self.assertEqual(descriptions["baseline"]["family"], "baseline")
        self.assertEqual(descriptions["uniform_r0.50"]["family"], "uniform_merge")
        self.assertEqual(descriptions["adaptive_r0.50"]["family"], "adaptive_merge")
        self.assertEqual(descriptions["baseline"]["tokens_theoretical"], 225)
        self.assertEqual(descriptions["uniform_r0.50"]["tokens_theoretical"], 112)
        self.assertEqual(descriptions["adaptive_r0.50"]["tokens_theoretical"], 112)

        rows = run_suite(
            models,
            batch_sizes=(1,),
            device="cpu",
            warmup=0,
            iters=1,
            stages=True,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row["family"] for row in rows},
            {"baseline", "uniform_merge", "adaptive_merge"},
        )
        self.assertTrue(all("encoder_median_ms" in row for row in rows))

    def test_ws4_evaluates_distinct_variants(self) -> None:
        loader = DataLoader(_OneSliceValidationDataset(), batch_size=1)
        models = self._models()
        self.assertEqual(len({id(model) for model in models.values()}), 3)
        for spec in self.specs:
            result = evaluate_model(
                models[spec.name],
                loader,
                device="cpu",
                variant=spec.name,
                variant_info=spec.info(),
                progress=False,
            )
            self.assertEqual(result.metadata["patients"], 1)
            self.assertEqual(result.metadata["variant_info"]["method"], spec.method)
            self.assertEqual({row["variant"] for row in result.rows}, {spec.name})


if __name__ == "__main__":
    unittest.main()
