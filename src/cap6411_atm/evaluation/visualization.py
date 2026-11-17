"""Report figures: case overlays, error maps, token merge maps and metric plots.

Colors use the Okabe-Ito colorblind-safe palette. Every plotting function
returns a Matplotlib figure and never writes patient data; ``save_figure``
writes PNG and PDF for the report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .metrics import REGION_NAMES
from .statistics import HIGHER_IS_BETTER, summarize

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}
REGION_COLORS = {"WT": OKABE_ITO["yellow"], "TC": OKABE_ITO["sky"], "ET": OKABE_ITO["vermillion"]}
ERROR_COLORS = {"TP": OKABE_ITO["green"], "FP": OKABE_ITO["orange"], "FN": OKABE_ITO["blue"]}
MODALITIES = ("T1n", "T1c", "T2w", "T2f")
SERIES_COLORS = [OKABE_ITO[key] for key in ("blue", "vermillion", "green", "purple", "orange", "sky", "black")]


def pick_slice(target: np.ndarray, region: str = "WT", mode: str = "max_area") -> int:
    """Choose an axial slice from a ``[3,H,W,D]`` target (``max_area`` or ``median``)."""

    areas = np.asarray(target)[REGION_NAMES.index(region)].sum(axis=(0, 1))
    if not areas.any():
        return int(areas.size // 2)
    if mode == "max_area":
        return int(np.argmax(areas))
    if mode == "median":
        present = np.flatnonzero(areas)
        return int(present[present.size // 2])
    raise ValueError("mode must be 'max_area' or 'median'")


def _modality(image: np.ndarray, modality: str) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    return image[MODALITIES.index(modality)]


def _show_mri(ax: plt.Axes, slice_2d: np.ndarray) -> None:
    finite = slice_2d[slice_2d != 0]
    lo, hi = (np.percentile(finite, (1, 99)) if finite.size else (0, 1))
    ax.imshow(slice_2d.T, cmap="gray", vmin=lo, vmax=hi, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])


def _contours(ax: plt.Axes, masks: np.ndarray) -> None:
    for index, name in enumerate(REGION_NAMES):
        if masks[index].any():
            ax.contour(masks[index].T.astype(float), levels=[0.5], colors=[REGION_COLORS[name]], linewidths=1.0, origin="lower")


def error_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Integer map: 0 = TN, 1 = TP, 2 = FP, 3 = FN."""

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    output = np.zeros(target.shape, dtype=np.uint8)
    output[prediction & target] = 1
    output[prediction & ~target] = 2
    output[~prediction & target] = 3
    return output


def plot_case(
    image: np.ndarray,
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    modality: str = "T2f",
    region: str = "ET",
    title: str | None = None,
) -> plt.Figure:
    """One row per method: MRI + GT contours, MRI + predicted contours, error map.

    ``image`` is ``[4,H,W]``; ``target`` and each prediction are ``[3,H,W]``.
    """

    names = list(predictions)
    fig, axes = plt.subplots(len(names), 3, figsize=(9, 3 * len(names)), squeeze=False)
    mri = _modality(image, modality)
    region_index = REGION_NAMES.index(region)
    cmap = ListedColormap([(0, 0, 0, 0), ERROR_COLORS["TP"], ERROR_COLORS["FP"], ERROR_COLORS["FN"]])
    for row, name in enumerate(names):
        prediction = np.asarray(predictions[name], dtype=bool)
        _show_mri(axes[row, 0], mri)
        _contours(axes[row, 0], np.asarray(target, dtype=bool))
        _show_mri(axes[row, 1], mri)
        _contours(axes[row, 1], prediction)
        _show_mri(axes[row, 2], mri)
        errors = error_map(prediction[region_index], np.asarray(target)[region_index])
        axes[row, 2].imshow(errors.T, cmap=cmap, vmin=0, vmax=3, origin="lower", interpolation="nearest")
        axes[row, 0].set_ylabel(name)
    axes[0, 0].set_title(f"Ground truth ({modality})")
    axes[0, 1].set_title("Prediction")
    axes[0, 2].set_title(f"{region} errors")
    handles = [Line2D([], [], color=REGION_COLORS[n], label=n) for n in REGION_NAMES]
    handles += [Patch(color=ERROR_COLORS[k], label=k) for k in ("TP", "FP", "FN")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


def plot_merge_map(
    image: np.ndarray,
    assignment: np.ndarray,
    *,
    target: np.ndarray | None = None,
    modality: str = "T2f",
    patch_size: int = 16,
    title: str | None = None,
) -> plt.Figure:
    """Overlay token-merge groups (one group id per patch) on an MRI slice.

    ``assignment`` holds the merged-token id of each row-major patch, as
    ``[rows*cols]`` or ``[rows, cols]``. Borders are drawn between neighbouring
    patches that belong to different groups; optional ``target`` adds GT contours.
    """

    mri = _modality(image, modality)
    height, width = mri.shape
    rows, cols = height // patch_size, width // patch_size
    groups = np.asarray(assignment).reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(5, 5))
    _show_mri(ax, mri)
    pixel_groups = np.kron(groups, np.ones((patch_size, patch_size), dtype=groups.dtype))
    rng = np.random.default_rng(0)
    unique = np.unique(groups)
    sizes = {g: np.count_nonzero(groups == g) for g in unique}
    colors = np.zeros((*pixel_groups.shape, 4))
    for g in unique:
        if sizes[g] > 1:  # only shade tokens that were actually merged
            colors[pixel_groups == g] = (*rng.uniform(0.2, 1.0, 3), 0.35)
    ax.imshow(np.transpose(colors, (1, 0, 2)), origin="lower")
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows and groups[r, c] != groups[r + 1, c]:
                y = (r + 1) * patch_size - 0.5
                ax.plot([y, y], [c * patch_size - 0.5, (c + 1) * patch_size - 0.5], color="white", lw=0.6)
            if c + 1 < cols and groups[r, c] != groups[r, c + 1]:
                x = (c + 1) * patch_size - 0.5
                ax.plot([r * patch_size - 0.5, (r + 1) * patch_size - 0.5], [x, x], color="white", lw=0.6)
    if target is not None:
        _contours(ax, np.asarray(target, dtype=bool))
    ax.set_xlim(-0.5, height - 0.5)
    ax.set_ylim(-0.5, width - 0.5)
    ax.set_title(title or f"{len(unique)} tokens from {rows * cols} patches")
    fig.tight_layout()
    return fig


def plot_metric_vs_x(
    frame: pd.DataFrame,
    *,
    metric: str = "dice",
    region: str = "ET",
    x: str = "merge_ratio",
    x_values: Mapping[str, float] | None = None,
    reference: str | None = "baseline",
    xlabel: str | None = None,
) -> plt.Figure:
    """Mean metric (95% bootstrap CI) against merge ratio, latency or another x.

    One line per ``family``. ``x_values`` (method -> x) overrides the column,
    e.g. measured latencies from Workstream 3 for matched-latency plots.
    """

    summary = summarize(frame[frame["region"] == region], metrics=[metric])
    family = frame.drop_duplicates("method").set_index("method")["family"] if "family" in frame else None
    xs = frame.drop_duplicates("method").set_index("method")[x] if x in frame else None
    summary["family"] = summary["method"].map(family) if family is not None else summary["method"]
    summary["x"] = summary["method"].map(x_values) if x_values else (summary["method"].map(xs) if xs is not None else np.nan)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    plotted = summary.dropna(subset=["x"])
    for color, (name, group) in zip(SERIES_COLORS, plotted.groupby("family")):
        group = group.sort_values("x")
        yerr = np.vstack([group["mean"] - group["ci_low"], group["ci_high"] - group["mean"]])
        ax.errorbar(group["x"], group["mean"], yerr=yerr, marker="o", capsize=3, color=color, label=name)
    if reference is not None and reference in set(summary["method"]):
        ref = summary[summary["method"] == reference].iloc[0]
        ax.axhline(ref["mean"], color="gray", ls="--", lw=1, label=f"{reference}")
        ax.axhspan(ref["ci_low"], ref["ci_high"], color="gray", alpha=0.15, lw=0)
    arrow = "↑" if HIGHER_IS_BETTER.get(metric, True) else "↓"
    ax.set_ylabel(f"{region} {metric} {arrow}")
    ax.set_xlabel(xlabel or x)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_paired_differences(
    frame: pd.DataFrame,
    reference: str,
    *,
    metric: str = "dice",
    region: str = "ET",
    methods: Sequence[str] | None = None,
) -> plt.Figure:
    """Per-patient (method - reference) differences as box + jittered points."""

    subset = frame[frame["region"] == region]
    pivot = subset.pivot_table(index="patient_id", columns="method", values=metric, aggfunc="first")
    methods = [m for m in (methods or pivot.columns) if m != reference]
    data = [(pivot[m] - pivot[reference]).dropna().to_numpy() for m in methods]
    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(methods) + 2), 4))
    ax.boxplot(data, showfliers=False, widths=0.5)
    rng = np.random.default_rng(0)
    for i, values in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, values.size), values, s=6, alpha=0.5, color=OKABE_ITO["blue"])
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(range(1, len(methods) + 1), methods, rotation=30, ha="right")
    ax.set_ylabel(f"Δ {region} {metric} vs {reference}")
    fig.tight_layout()
    return fig


def plot_profile(profile: pd.DataFrame, *, ylabel: str, title: str | None = None) -> plt.Figure:
    """Grouped bars for ``boundary_profile`` / ``patch_profile`` tables (methods x bins)."""

    fig, ax = plt.subplots(figsize=(max(5, 1.0 * profile.shape[1] + 3), 4))
    width = 0.8 / max(len(profile.index), 1)
    positions = np.arange(profile.shape[1])
    for i, (color, (method, row)) in enumerate(zip(SERIES_COLORS * 4, profile.iterrows())):
        ax.bar(positions + i * width, row.to_numpy(dtype=float), width, label=method, color=color)
    ax.set_xticks(positions + width * (len(profile.index) - 1) / 2, profile.columns, rotation=20)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path_stem: str | Path, *, dpi: int = 200) -> list[Path]:
    """Save ``<stem>.png`` and ``<stem>.pdf``; returns the written paths."""

    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return paths
