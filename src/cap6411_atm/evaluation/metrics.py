"""Patient-level 3D segmentation metrics for overlapping WT/TC/ET regions.

Every function here works on **binary volumes** shaped ``[H, W, D]`` (one
region) or ``[3, H, W, D]`` (WT, TC, ET), with the axial axis last to match the
``slice_index`` convention of :class:`BraTSAxialSliceDataset`.

Metric families
---------------
* Overlap: Dice, sensitivity, precision, voxel counts (TP/FP/FN) and volumes.
* Surface distance: HD95, average symmetric surface distance (ASSD) and
  normalized surface Dice (NSD) at configurable tolerances.
* Boundary bands: where errors occur as a function of distance from the
  ground-truth tumor boundary.
* Token-aligned patches: error rates inside 16x16 axial patches that are
  background, fully tumor (interior) or straddle the tumor boundary. These are
  exactly the DINOv3 token footprints that a merger can combine.
* Lesions: detection rate of ground-truth connected components by size and
  the number of false-positive predicted components.

Empty-mask conventions (documented in ``docs/EVALUATION.md``)
-------------------------------------------------------------
* Both empty: Dice 1, NSD 1, HD95 0, ASSD 0.
* Exactly one empty: Dice 0, NSD 0, HD95/ASSD = ``empty_distance_mm``
  (default: the 240x240x155 mm volume diagonal, 373.13 mm, as used by BraTS).
* Sensitivity is NaN when the ground truth is empty; precision is NaN when the
  prediction is empty. Aggregation code ignores NaNs and reports counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage


REGION_NAMES = ("WT", "TC", "ET")
BRATS_VOLUME_DIAGONAL_MM = math.sqrt(240.0**2 + 240.0**2 + 155.0**2)  # 373.1287...
PATCH_CLASSES = ("background", "boundary", "interior")


@dataclass(frozen=True)
class MetricConfig:
    """Settings shared by every evaluated variant; keep fixed across comparisons."""

    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    nsd_tolerances_mm: tuple[float, ...] = (1.0, 2.0)
    boundary_band_edges_mm: tuple[float, ...] = (0.0, 2.0, 4.0, 8.0)
    lesion_size_edges: tuple[int, ...] = (0, 100, 1000, 10000)
    patch_size: int = 16
    empty_distance_mm: float = BRATS_VOLUME_DIAGONAL_MM
    hd_percentile: float = 95.0

    def __post_init__(self) -> None:
        if len(self.spacing) != 3 or any(value <= 0 for value in self.spacing):
            raise ValueError("spacing must contain three positive values")
        if any(value < 0 for value in self.nsd_tolerances_mm):
            raise ValueError("NSD tolerances must be non-negative")
        edges = self.boundary_band_edges_mm
        if len(edges) < 2 or edges[0] != 0 or list(edges) != sorted(set(edges)):
            raise ValueError("boundary band edges must start at 0 and strictly increase")
        sizes = self.lesion_size_edges
        if len(sizes) < 2 or sizes[0] != 0 or list(sizes) != sorted(set(sizes)):
            raise ValueError("lesion size edges must start at 0 and strictly increase")
        if self.patch_size < 1:
            raise ValueError("patch_size must be positive")
        if not 0 < self.hd_percentile <= 100:
            raise ValueError("hd_percentile must be in (0, 100]")

    @property
    def band_names(self) -> tuple[str, ...]:
        edges = self.boundary_band_edges_mm
        names = [f"band_{_fmt(lo)}_{_fmt(hi)}mm" for lo, hi in zip(edges[:-1], edges[1:])]
        names.append(f"band_ge{_fmt(edges[-1])}mm")
        return tuple(names)

    @property
    def lesion_bin_names(self) -> tuple[str, ...]:
        edges = self.lesion_size_edges
        names = [f"lesion_{lo}_{hi}" for lo, hi in zip(edges[:-1], edges[1:])]
        names.append(f"lesion_ge{edges[-1]}")
        return tuple(names)


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def _as_bool(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype != bool:
        array = array.astype(bool)
    return array


def _crop_box(union: np.ndarray, margin: int) -> tuple[slice, ...]:
    coords = np.argwhere(union)
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, union.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _surface(mask: np.ndarray) -> np.ndarray:
    """Voxels of ``mask`` with at least one 6-connected background neighbour."""

    if not mask.any():
        return mask.copy()
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded


def _distance_to(surface: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    """Euclidean distance (mm) from every voxel to the nearest ``surface`` voxel."""

    return ndimage.distance_transform_edt(~surface, sampling=spacing)


def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    pred = _as_bool(pred)
    gt = _as_bool(gt)
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    pred_volume = tp + fp
    gt_volume = tp + fn
    if gt_volume == 0 and pred_volume == 0:
        status, dice = "both_empty", 1.0
    elif gt_volume == 0:
        status, dice = "gt_empty", 0.0
    elif pred_volume == 0:
        status, dice = "pred_empty", 0.0
    else:
        status, dice = "both_present", 2.0 * tp / (pred_volume + gt_volume)
    return {
        "status": status,
        "dice": dice,
        "sensitivity": tp / gt_volume if gt_volume else math.nan,
        "precision": tp / pred_volume if pred_volume else math.nan,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_volume": gt_volume,
        "pred_volume": pred_volume,
        "volume_error": pred_volume - gt_volume,
    }


def region_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    config: MetricConfig = MetricConfig(),
) -> dict[str, Any]:
    """All metric families for one binary region volume ``[H, W, D]``."""

    pred = _as_bool(pred)
    gt = _as_bool(gt)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError(f"expected matching [H,W,D] masks, got {pred.shape} and {gt.shape}")

    result = overlap_metrics(pred, gt)
    result.update(patch_class_errors(pred, gt, config.patch_size))
    status = result["status"]
    tolerances = config.nsd_tolerances_mm
    band_names = config.band_names

    if status == "both_empty":
        result.update({"hd95": 0.0, "assd": 0.0})
        result.update({f"nsd_{_fmt(t)}mm": 1.0 for t in tolerances})
        result.update(_empty_bands(band_names, fp=0, fn=0))
        result.update(_lesion_stats(pred, gt, config))
        return result
    if status in ("gt_empty", "pred_empty"):
        result.update({"hd95": config.empty_distance_mm, "assd": config.empty_distance_mm})
        result.update({f"nsd_{_fmt(t)}mm": 0.0 for t in tolerances})
        # Without a ground-truth boundary there is no band distance: report the
        # errors as unbanded and leave every band NaN.
        result.update(_empty_bands(band_names, fp=result["fp"], fn=result["fn"]))
        result.update(_lesion_stats(pred, gt, config))
        return result

    finite_edges = [*config.boundary_band_edges_mm, *tolerances]
    margin = int(math.ceil(max(finite_edges) / min(config.spacing))) + 2
    box = _crop_box(pred | gt, margin)
    pred_c, gt_c = pred[box], gt[box]

    pred_surface = _surface(pred_c)
    gt_surface = _surface(gt_c)
    dist_to_gt = _distance_to(gt_surface, config.spacing)
    dist_to_pred = _distance_to(pred_surface, config.spacing)
    d_pred_to_gt = dist_to_gt[pred_surface]
    d_gt_to_pred = dist_to_pred[gt_surface]

    result["hd95"] = float(
        max(
            np.percentile(d_pred_to_gt, config.hd_percentile),
            np.percentile(d_gt_to_pred, config.hd_percentile),
        )
    )
    n_surface = d_pred_to_gt.size + d_gt_to_pred.size
    result["assd"] = float((d_pred_to_gt.sum() + d_gt_to_pred.sum()) / n_surface)
    for tolerance in tolerances:
        within = np.count_nonzero(d_pred_to_gt <= tolerance) + np.count_nonzero(
            d_gt_to_pred <= tolerance
        )
        result[f"nsd_{_fmt(tolerance)}mm"] = float(within / n_surface)

    result.update(_band_errors(pred_c, gt_c, dist_to_gt, config))
    result.update(_lesion_stats(pred_c, gt_c, config))
    return result


def _empty_bands(names: tuple[str, ...], *, fp: int, fn: int) -> dict[str, Any]:
    values: dict[str, Any] = {"unbanded_fp": fp, "unbanded_fn": fn}
    for name in names:
        values[f"{name}_fp"] = math.nan
        values[f"{name}_fn"] = math.nan
        values[f"{name}_voxels"] = math.nan
        values[f"{name}_error_rate"] = math.nan
    return values


def _band_errors(
    pred: np.ndarray,
    gt: np.ndarray,
    dist_to_gt_surface: np.ndarray,
    config: MetricConfig,
) -> dict[str, Any]:
    """FP/FN voxels binned by distance from the ground-truth boundary.

    Band voxel counts (and therefore error rates) are exact for the bounded
    bands because the crop margin exceeds the largest finite edge. The last,
    open-ended band only reports error counts: its size depends on the volume.
    """

    edges = config.boundary_band_edges_mm
    values: dict[str, Any] = {"unbanded_fp": 0, "unbanded_fn": 0}
    fp_mask = pred & ~gt
    fn_mask = ~pred & gt
    bounds = [*zip(edges[:-1], edges[1:]), (edges[-1], math.inf)]
    for name, (lo, hi) in zip(config.band_names, bounds):
        in_band = (dist_to_gt_surface >= lo) & (dist_to_gt_surface < hi)
        fp = int(np.count_nonzero(fp_mask & in_band))
        fn = int(np.count_nonzero(fn_mask & in_band))
        values[f"{name}_fp"] = fp
        values[f"{name}_fn"] = fn
        if math.isfinite(hi):
            voxels = int(np.count_nonzero(in_band))
            values[f"{name}_voxels"] = voxels
            values[f"{name}_error_rate"] = (fp + fn) / voxels if voxels else math.nan
        else:
            values[f"{name}_voxels"] = math.nan
            values[f"{name}_error_rate"] = math.nan
    return values


def patch_class_errors(pred: np.ndarray, gt: np.ndarray, patch_size: int = 16) -> dict[str, Any]:
    """Errors inside token-sized axial patches, grouped by ground-truth content.

    Each axial slice is tiled into ``patch_size`` x ``patch_size`` patches (the
    DINOv3 token footprint). A patch is *background* if it has no tumor voxels,
    *interior* if it is entirely tumor, and *boundary* otherwise. The error rate
    is (FP + FN) / pixels within that patch class.
    """

    pred = _as_bool(pred)
    gt = _as_bool(gt)
    height, width, depth = gt.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"spatial size {(height, width)} not divisible by patch {patch_size}")
    rows, cols = height // patch_size, width // patch_size
    shape = (rows, patch_size, cols, patch_size, depth)
    gt_fraction = gt.reshape(shape).mean(axis=(1, 3))  # [rows, cols, depth]
    errors = (pred ^ gt).reshape(shape).sum(axis=(1, 3))
    pixels_per_patch = patch_size * patch_size
    classes = {
        "background": gt_fraction == 0,
        "boundary": (gt_fraction > 0) & (gt_fraction < 1),
        "interior": gt_fraction == 1,
    }
    values: dict[str, Any] = {}
    for name, selected in classes.items():
        patches = int(np.count_nonzero(selected))
        error_count = int(errors[selected].sum())
        values[f"patch_{name}_count"] = patches
        values[f"patch_{name}_errors"] = error_count
        values[f"patch_{name}_error_rate"] = (
            error_count / (patches * pixels_per_patch) if patches else math.nan
        )
    return values


def _lesion_stats(pred: np.ndarray, gt: np.ndarray, config: MetricConfig) -> dict[str, Any]:
    """Size-binned detection of ground-truth components and false-positive components.

    Components use 26-connectivity. A ground-truth lesion is *detected* when any
    predicted voxel overlaps it; a predicted component is a *false positive*
    when it overlaps no ground-truth voxel.
    """

    structure = ndimage.generate_binary_structure(3, 3)
    values: dict[str, Any] = {}
    gt_labels, n_gt = ndimage.label(gt, structure=structure)
    sizes = np.bincount(gt_labels.ravel(), minlength=n_gt + 1)[1:]
    hit = np.bincount(gt_labels[pred], minlength=n_gt + 1)[1:] > 0
    edges = [*config.lesion_size_edges, math.inf]
    for name, lo, hi in zip(config.lesion_bin_names, edges[:-1], edges[1:]):
        in_bin = (sizes >= lo) & (sizes < hi)
        values[f"{name}_count"] = int(np.count_nonzero(in_bin))
        values[f"{name}_detected"] = int(np.count_nonzero(in_bin & hit))
    values["gt_components"] = int(n_gt)
    values["gt_components_detected"] = int(np.count_nonzero(hit))

    pred_labels, n_pred = ndimage.label(pred, structure=structure)
    overlaps = np.bincount(pred_labels[gt], minlength=n_pred + 1)[1:] > 0
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)[1:]
    values["pred_components"] = int(n_pred)
    values["fp_components"] = int(np.count_nonzero(~overlaps))
    values["fp_component_voxels"] = int(pred_sizes[~overlaps].sum())
    return values


def patient_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    config: MetricConfig = MetricConfig(),
) -> dict[str, dict[str, Any]]:
    """Evaluate a ``[3, H, W, D]`` WT/TC/ET prediction against its target."""

    pred = _as_bool(pred)
    gt = _as_bool(gt)
    if pred.shape != gt.shape or pred.ndim != 4 or pred.shape[0] != len(REGION_NAMES):
        raise ValueError(f"expected matching [3,H,W,D] volumes, got {pred.shape} and {gt.shape}")
    return {
        name: region_metrics(pred[index], gt[index], config)
        for index, name in enumerate(REGION_NAMES)
    }
