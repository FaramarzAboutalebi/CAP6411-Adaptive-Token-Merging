"""Reusable baseline training utilities.

Baseline entry points are loaded lazily so ``python -m ...training.baseline``
does not import its target module before execution.
"""

from typing import Any

from .losses import MultilabelBCEDiceLoss
from .metrics import StreamingDice
from .sampling import PatientLocalBatchSampler, PatientSubset, select_pilot_patients


def __getattr__(name: str) -> Any:
    if name in {
        "BaselineTrainingConfig",
        "build_training_components",
        "fit",
        "run_one_batch_smoke_test",
    }:
        from . import baseline

        return getattr(baseline, name)
    raise AttributeError(name)

__all__ = [
    "BaselineTrainingConfig",
    "MultilabelBCEDiceLoss",
    "PatientLocalBatchSampler",
    "PatientSubset",
    "StreamingDice",
    "build_training_components",
    "fit",
    "run_one_batch_smoke_test",
    "select_pilot_patients",
]
