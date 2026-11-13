"""Focused synthetic tests for the reusable baseline training pipeline."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cap6411_atm.training.baseline import (
    BaselineTrainingConfig,
    TrainingComponents,
    _make_scaler,
    build_training_components,
    fit,
    resume_from_checkpoint,
    run_one_batch_smoke_test,
    trainable_parameters,
)
from cap6411_atm.training.losses import MultilabelBCEDiceLoss
from cap6411_atm.training.metrics import StreamingDice
from cap6411_atm.training.sampling import (
    PatientLocalBatchSampler,
    PatientSubset,
    select_pilot_patients,
)


class SyntheticSlices(Dataset):
    def __init__(self) -> None:
        self.patient_ids = ("patient-a", "patient-b", "patient-c")
        self.samples = tuple(
            (patient_id, slice_index)
            for patient_id in self.patient_ids
            for slice_index in range(3)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        patient_id, slice_index = self.samples[index]
        generator = torch.Generator().manual_seed(index)
        image = torch.randn(4, 8, 8, generator=generator)
        target = torch.zeros(3, 8, 8)
        target[0, 2:6, 2:6] = 1
        target[1, 3:5, 3:5] = 1
        if slice_index == 1:
            target[2, 3:4, 3:4] = 1
        return {
            "image": image,
            "target": target,
            "patient_id": patient_id,
            "slice_index": slice_index,
            "split": "train",
        }


class SyntheticSplitSlices(Dataset):
    """Tiny split fixture with the dataset attributes used by the trainer."""

    def __init__(self, split: str, *, cache_dir=None, **kwargs) -> None:
        del kwargs
        self.cache_dir = cache_dir
        prefix = "train" if split == "train" else "validation"
        count = 2 if split == "train" else 1
        self.patient_ids = tuple(f"{prefix}-{index}" for index in range(count))
        self.samples = tuple(
            (patient_id, index)
            for patient_id in self.patient_ids
            for index in range(2)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        patient_id, slice_index = self.samples[index]
        return {
            "image": torch.zeros(4, 8, 8),
            "target": torch.zeros(3, 8, 8),
            "patient_id": patient_id,
            "slice_index": slice_index,
            "split": patient_id.split("-", 1)[0],
        }


class TinyFrozenEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Conv2d(3, 3, 1)
        self.backbone.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


@dataclass(frozen=True)
class TinyModelConfig:
    name: str = "synthetic"


class TinySegmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyModelConfig()
        self.adapter = nn.Conv2d(4, 3, 1)
        self.encoder = TinyFrozenEncoder()
        self.decoder = nn.Conv2d(3, 3, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images=self.adapter(images)))


def make_config(root: Path, **overrides) -> BaselineTrainingConfig:
    values = {
        "archive_path": "synthetic.zip",
        "manifest_path": "synthetic-manifest.json",
        "inspection_config": "inspection.json",
        "model_config": "model.json",
        "dinov3_repository": "dinov3",
        "checkpoint_path": "weights.pth",
        "output_dir": str(root),
        "experiment_name": "synthetic-run",
        "epochs": 1,
        "batch_size": 2,
        "amp": False,
        "pilot_enabled": True,
        "pilot_train_patients": 2,
        "pilot_validation_patients": 1,
    }
    values.update(overrides)
    return BaselineTrainingConfig(**values)


def make_components() -> TrainingComponents:
    dataset = SyntheticSlices()
    train_subset = PatientSubset(dataset, ("patient-a", "patient-b"))
    validation_subset = PatientSubset(dataset, ("patient-c",))
    train_sampler = PatientLocalBatchSampler(
        train_subset, batch_size=2, seed=42, shuffle_patients=True, shuffle_slices=True
    )
    validation_sampler = PatientLocalBatchSampler(
        validation_subset,
        batch_size=2,
        seed=42,
        shuffle_patients=False,
        shuffle_slices=False,
    )
    model = TinySegmenter()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=1e-3)
    return TrainingComponents(
        model=model,  # type: ignore[arg-type]
        train_loader=DataLoader(train_subset, batch_sampler=train_sampler),
        validation_loader=DataLoader(validation_subset, batch_sampler=validation_sampler),
        train_sampler=train_sampler,
        validation_sampler=validation_sampler,
        optimizer=optimizer,
        criterion=MultilabelBCEDiceLoss(),
        device=torch.device("cpu"),
        amp_enabled=False,
        scaler=_make_scaler(torch.device("cpu"), False),
        dataset_metadata={
            "scope": "pilot",
            "pilot_metrics": True,
            "metrics_scope": "deterministic_patient_subset",
            "test_split_loaded": False,
            "full_train_samples": len(dataset),
            "full_validation_samples": len(dataset),
        },
    )


class BaselineTrainingTests(unittest.TestCase):
    def test_training_components_disable_disk_cache_without_creating_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(
                root,
                cache_dir=None,
                pilot_enabled=False,
                model_config=str(root / "model.json"),
            )
            created_datasets = []

            def dataset_factory(*args, **kwargs):
                dataset = SyntheticSplitSlices(args[2], **kwargs)
                created_datasets.append(dataset)
                return dataset

            with (
                mock.patch(
                    "cap6411_atm.training.baseline.DINOv3SegmentationConfig.from_json",
                    return_value=TinyModelConfig(),
                ),
                mock.patch(
                    "cap6411_atm.training.baseline.build_pretrained_brats_dinov3",
                    return_value=TinySegmenter(),
                ),
                mock.patch(
                    "cap6411_atm.training.baseline.BraTSAxialSliceDataset",
                    side_effect=dataset_factory,
                ),
            ):
                components = build_training_components(config)

            self.assertEqual(len(created_datasets), 2)
            self.assertTrue(all(dataset.cache_dir is None for dataset in created_datasets))
            self.assertEqual(list(root.iterdir()), [])
            self.assertEqual(components.dataset_metadata["train_patient_count"], 2)
            self.assertEqual(components.dataset_metadata["validation_patient_count"], 1)

    def test_full_cohort_configuration_is_storage_safe(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = BaselineTrainingConfig.from_json(
            root / "configs" / "brats2023_gli_full_cohort_training.json"
        )
        self.assertFalse(config.pilot_enabled)
        self.assertIsNone(config.cache_dir)
        self.assertEqual(config.batch_size, 2)
        self.assertTrue(config.amp)
        self.assertEqual(config.epochs, 1)
        self.assertEqual(config.max_cached_patients, 1)
        self.assertIn("full_cohort", config.output_dir)

    def test_loss_is_finite_and_propagates_gradients(self) -> None:
        logits = torch.zeros(2, 3, 8, 8, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, 0, 2:6, 2:6] = 1
        loss = MultilabelBCEDiceLoss()(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_streaming_dice_uses_global_counts_and_documents_empty_channels(self) -> None:
        metric = StreamingDice(0.5)
        targets = torch.zeros(1, 3, 2, 2)
        targets[:, 0, 0, 0] = 1
        logits = torch.full_like(targets, -20)
        logits[:, 0, 0, 0] = 20
        metric.update(logits, targets)
        result = metric.compute()
        self.assertEqual(result["dice"], {"WT": 1.0, "TC": 1.0, "ET": 1.0})
        self.assertEqual(result["all_empty"], {"WT": False, "TC": True, "ET": True})

    def test_pilot_selection_is_deterministic_disjoint_and_whole_patient(self) -> None:
        train = tuple(f"train-{index}" for index in range(10))
        validation = tuple(f"validation-{index}" for index in range(5))
        selected_a = select_pilot_patients(train, 4, seed=42)
        selected_b = select_pilot_patients(train, 4, seed=42)
        selected_validation = select_pilot_patients(validation, 2, seed=43)
        self.assertEqual(selected_a, selected_b)
        self.assertTrue(set(selected_a).isdisjoint(selected_validation))
        subset = PatientSubset(SyntheticSlices(), ("patient-b",))
        self.assertEqual({patient for patient, _ in subset.samples}, {"patient-b"})
        self.assertEqual(len(subset), 3)

    def test_patient_local_sampler_is_reproducible_and_never_mixes_patients(self) -> None:
        dataset = SyntheticSlices()
        sampler = PatientLocalBatchSampler(dataset, batch_size=2, seed=42)
        epoch_zero = list(sampler)
        self.assertEqual(epoch_zero, list(sampler))
        for batch in epoch_zero:
            self.assertEqual(len({dataset.samples[index][0] for index in batch}), 1)
        sampler.set_epoch(1)
        self.assertNotEqual(epoch_zero, list(sampler))

    def test_smoke_test_updates_trainable_layers_only_and_reports_throughput(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            components = make_components()
            config = make_config(Path(temporary))
            frozen_before = {
                name: value.detach().clone()
                for name, value in components.model.encoder.backbone.state_dict().items()
            }
            result = run_one_batch_smoke_test(components, config)
            self.assertEqual(result["status"], "passed")
            self.assertGreater(result["samples_per_second"], 0)
            self.assertEqual(result["image_shape"][1:], [4, 8, 8])
            for name, value in components.model.encoder.backbone.state_dict().items():
                self.assertTrue(torch.equal(value, frozen_before[name]))

    def test_fit_saves_complete_checkpoint_and_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(root)
            components = make_components()
            history = fit(components, config)
            run_dir = root / "synthetic-run"
            self.assertEqual(len(history), 1)
            self.assertTrue((run_dir / "best.pt").is_file())
            self.assertTrue((run_dir / "last.pt").is_file())
            self.assertTrue((run_dir / "history.json").is_file())
            restored = make_components()
            next_epoch, best, restored_history = resume_from_checkpoint(
                run_dir / "last.pt", restored, config
            )
            self.assertEqual(next_epoch, 1)
            self.assertEqual(len(restored_history), 1)
            self.assertEqual(best, history[0]["validation"]["macro_dice"])

    def test_resume_rejects_changed_patient_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(root)
            fit(make_components(), config)
            changed = make_config(root, pilot_train_patients=3)
            with self.assertRaisesRegex(ValueError, "pilot_train_patients"):
                resume_from_checkpoint(
                    root / "synthetic-run" / "last.pt", make_components(), changed
                )


if __name__ == "__main__":
    unittest.main()
