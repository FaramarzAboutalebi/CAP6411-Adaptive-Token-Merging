"""Rules for when two benchmark rows may be compared.

Latency is only comparable between rows measured on the same GPU model with
the same driver, CUDA, and PyTorch build, at the same batch size, precision,
and input kind.  Dice is only comparable between rows evaluated with the same
trained weights (checkpoint fingerprint), the same validation slices (data
fingerprint), and the same threshold.  ``match_latency``, ``results_table``,
and ``plot_tradeoff`` all apply these rules (the plot additionally requires a
single set of latency conditions), so rows combined from different
CSVs (for example area51 and a Colab GPU) are never silently mixed.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

# Environment columns stamped on every row by ``run_suite``.
ENVIRONMENT_KEYS = ("gpu", "nvidia_driver", "cuda_runtime", "torch")
LATENCY_KEYS = ("batch_size", "device", "precision", "input", *ENVIRONMENT_KEYS)

# Columns stamped by ``attach_quality`` from ``evaluate_quality_many``.
DICE_BASIS_KEYS = ("dice_checkpoint", "dice_data", "dice_threshold")


def latency_conditions(row: Mapping[str, Any]) -> tuple:
    """Measurement conditions that must match for latencies to be compared."""

    return tuple(row.get(key) for key in LATENCY_KEYS)


def dice_basis(row: Mapping[str, Any]) -> tuple | None:
    """Weights/data/threshold identity of a row's Dice, or None without Dice."""

    if not isinstance(row.get("macro_dice"), (int, float)):
        return None
    return tuple(row.get(key) for key in DICE_BASIS_KEYS)


def same_dice_basis(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """True when both rows have Dice from identical weights, slices, and threshold.

    Rows lacking the fingerprint columns (e.g. hand-built rows or CSVs written
    before fingerprints existed) are never considered comparable.
    """

    a, b = dice_basis(first), dice_basis(second)
    return a is not None and a == b and all(value is not None for value in a)


def check_same_dice_basis(rows: Iterable[Mapping[str, Any]]) -> None:
    """Raise if rows with Dice were evaluated on different weights or slices."""

    bases: dict[tuple, list[str]] = {}
    for row in rows:
        basis = dice_basis(row)
        if basis is not None:
            bases.setdefault(basis, []).append(str(row.get("variant")))
    if len(bases) > 1:
        detail = "; ".join(
            f"{dict(zip(DICE_BASIS_KEYS, basis))}: {', '.join(names)}"
            for basis, names in bases.items()
        )
        raise ValueError(
            "Dice values come from different checkpoints, validation slices, or "
            f"thresholds and cannot be shown together: {detail}"
        )
