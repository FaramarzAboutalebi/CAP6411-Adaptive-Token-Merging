"""Deterministic pilot selection and patient-local slice batching."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

from torch.utils.data import Dataset, Sampler


def select_pilot_patients(
    patient_ids: Sequence[str],
    count: int,
    *,
    seed: int,
) -> tuple[str, ...]:
    """Select a deterministic subset from one already-isolated manifest split."""

    unique = sorted(patient_ids)
    if len(unique) != len(patient_ids):
        raise ValueError("patient IDs must be unique")
    if not 1 <= count <= len(unique):
        raise ValueError(f"pilot count must be in [1, {len(unique)}]")
    random.Random(seed).shuffle(unique)
    return tuple(unique[:count])


class PatientSubset(Dataset[dict[str, Any]]):
    """Restrict a BraTS slice dataset to complete selected patients."""

    def __init__(self, dataset: Dataset, patient_ids: Sequence[str]) -> None:
        if not hasattr(dataset, "samples"):
            raise TypeError("patient subset requires a dataset with a samples index")
        selected = set(patient_ids)
        if len(selected) != len(patient_ids):
            raise ValueError("selected patient IDs must be unique")
        self.dataset = dataset
        self.indices = tuple(
            index
            for index, (patient_id, _) in enumerate(dataset.samples)
            if patient_id in selected
        )
        self.samples = tuple(dataset.samples[index] for index in self.indices)
        observed = {patient_id for patient_id, _ in self.samples}
        if observed != selected:
            raise ValueError(f"selected patients not found: {sorted(selected - observed)}")
        self.patient_ids = tuple(patient_ids)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]


class PatientLocalBatchSampler(Sampler[list[int]]):
    """Shuffle patients reproducibly while keeping every batch patient-local."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        seed: int,
        shuffle_patients: bool = True,
        shuffle_slices: bool = True,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not hasattr(dataset, "samples"):
            raise TypeError("patient-local batching requires dataset.samples")
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle_patients = shuffle_patients
        self.shuffle_slices = shuffle_slices
        self.drop_last = drop_last
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, (patient_id, _) in enumerate(dataset.samples):
            grouped[patient_id].append(index)
        if not grouped:
            raise ValueError("dataset contains no samples")
        self.patient_indices = dict(grouped)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        patient_ids = sorted(self.patient_indices)
        if self.shuffle_patients:
            rng.shuffle(patient_ids)
        for patient_id in patient_ids:
            indices = list(self.patient_indices[patient_id])
            if self.shuffle_slices:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                len(indices) // self.batch_size for indices in self.patient_indices.values()
            )
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.patient_indices.values()
        )
