"""Decision rules that turn WT/TC/ET probability volumes into binary masks.

A *post-processor* is any callable ``probabilities [3,H,W,D] -> bool [3,H,W,D]``.
Specs are plain JSON-compatible dictionaries so that ablation configurations
can be recorded with their results:

```python
{"threshold": 0.5}                                    # baseline rule (matches StreamingDice)
{"threshold": 0.5, "nesting": True}                   # enforce ET <= TC <= WT
{"threshold": 0.5, "min_component": {"ET": 50}}       # drop small ET islands
{"threshold": 0.5, "min_total": {"ET": 200}}          # clear ET if its total is tiny
```

All rules here are evaluated from one forward pass, so threshold and
post-processing ablations cost no extra GPU time.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
from scipy import ndimage

from .metrics import REGION_NAMES

PostProcessor = Callable[[np.ndarray], np.ndarray]


def _region_index(name: str) -> int:
    try:
        return REGION_NAMES.index(name)
    except ValueError as error:
        raise ValueError(f"unknown region {name!r}; expected one of {REGION_NAMES}") from error


def apply_threshold(probabilities: np.ndarray, threshold: float | Mapping[str, float]) -> np.ndarray:
    """Threshold with ``>=`` (same rule as ``StreamingDice``), optionally per region."""

    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 4 or probabilities.shape[0] != len(REGION_NAMES):
        raise ValueError(f"expected [3,H,W,D] probabilities, got {probabilities.shape}")
    if isinstance(threshold, Mapping):
        values = [float(threshold.get(name, 0.5)) for name in REGION_NAMES]
    else:
        values = [float(threshold)] * len(REGION_NAMES)
    if not all(0 < value < 1 for value in values):
        raise ValueError("thresholds must lie strictly between 0 and 1")
    return np.stack([probabilities[i] >= values[i] for i in range(len(REGION_NAMES))])


def enforce_nesting(masks: np.ndarray) -> np.ndarray:
    """Constrain inner regions by outer ones: TC &= WT, then ET &= TC.

    BraTS regions are nested by definition, but the baseline predicts three
    independent sigmoid channels that can violate ET <= TC <= WT.
    """

    output = np.array(masks, dtype=bool, copy=True)
    output[1] &= output[0]
    output[2] &= output[1]
    return output


def remove_small_components(masks: np.ndarray, min_sizes: Mapping[str, int]) -> np.ndarray:
    """Remove 26-connected components smaller than ``min_sizes[region]`` voxels."""

    output = np.array(masks, dtype=bool, copy=True)
    structure = ndimage.generate_binary_structure(3, 3)
    for name, minimum in min_sizes.items():
        if minimum <= 0:
            continue
        index = _region_index(name)
        labels, count = ndimage.label(output[index], structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(labels.ravel())
        small = sizes < minimum
        small[0] = False
        output[index][small[labels]] = False
    return output


def clear_small_totals(masks: np.ndarray, min_totals: Mapping[str, int]) -> np.ndarray:
    """Clear a region entirely when its total predicted volume is below a minimum.

    Targets enhancing-tumor false positives on ET-negative patients.
    """

    output = np.array(masks, dtype=bool, copy=True)
    for name, minimum in min_totals.items():
        index = _region_index(name)
        if np.count_nonzero(output[index]) < minimum:
            output[index] = False
    return output


def build_postprocessor(spec: Mapping[str, Any] | None = None) -> PostProcessor:
    """Build a post-processor from a JSON-compatible spec (see module docstring)."""

    spec = dict(spec or {})
    allowed = {"threshold", "nesting", "min_component", "min_total"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"unknown post-processing keys: {sorted(unknown)}")
    threshold = spec.get("threshold", 0.5)
    nesting = bool(spec.get("nesting", False))
    min_component = dict(spec.get("min_component", {}))
    min_total = dict(spec.get("min_total", {}))
    for name in (*min_component, *min_total):
        _region_index(name)

    def postprocess(probabilities: np.ndarray) -> np.ndarray:
        masks = apply_threshold(probabilities, threshold)
        if nesting:
            masks = enforce_nesting(masks)
        if min_component:
            masks = remove_small_components(masks, min_component)
        if min_total:
            masks = clear_small_totals(masks, min_total)
        return masks

    postprocess.spec = spec  # type: ignore[attr-defined]
    return postprocess


def threshold_sweep(thresholds: list[float]) -> dict[str, dict[str, Any]]:
    """Named specs for a global threshold sweep, e.g. ``{"thr_0.40": {...}}``."""

    return {f"thr_{value:.2f}": {"threshold": float(value)} for value in thresholds}


DEFAULT_RULES: dict[str, dict[str, Any]] = {
    "raw": {"threshold": 0.5},
}

POSTPROCESSING_ABLATION: dict[str, dict[str, Any]] = {
    "raw": {"threshold": 0.5},
    "nested": {"threshold": 0.5, "nesting": True},
    "nested_cc50": {"threshold": 0.5, "nesting": True, "min_component": {"ET": 50, "TC": 50, "WT": 50}},
    "nested_et_total200": {"threshold": 0.5, "nesting": True, "min_total": {"ET": 200}},
    "nested_cc50_et_total200": {
        "threshold": 0.5,
        "nesting": True,
        "min_component": {"ET": 50, "TC": 50, "WT": 50},
        "min_total": {"ET": 200},
    },
}
