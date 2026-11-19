"""Cohort summaries, paired method comparisons and stratified analyses.

All inputs are the long-format per-patient tables written by
``save_evaluation`` (one row per variant x rule x patient x region). The
*method* of a row is ``"<variant>/<rule>"`` unless the rule is ``raw``.
Patients are the unit of statistical inference: bootstrap resampling and the
Wilcoxon signed-rank test both operate on paired per-patient values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import REGION_NAMES

# Direction of improvement for each metric (True = higher is better).
HIGHER_IS_BETTER = {
    "dice": True,
    "sensitivity": True,
    "precision": True,
    "nsd_1mm": True,
    "nsd_2mm": True,
    "hd95": False,
    "assd": False,
    "patch_boundary_error_rate": False,
    "patch_interior_error_rate": False,
    "patch_background_error_rate": False,
    "fp_components": False,
    "abs_volume_error": False,
}

DEFAULT_METRICS = ("dice", "hd95", "nsd_1mm", "nsd_2mm", "assd")


def load_results(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Concatenate ``per_patient.csv`` files (or directories containing one)."""

    frames = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / "per_patient.csv"
        frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError("no result files given")
    frame = add_derived_columns(pd.concat(frames, ignore_index=True))
    check_unique_rows(frame)
    return frame


ROW_KEY = ("method", "region", "patient_id")


def check_unique_rows(frame: pd.DataFrame) -> None:
    """Reject repeated (method, region, patient) rows, e.g. a variant loaded twice."""

    duplicated = frame.duplicated(list(ROW_KEY), keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, list(ROW_KEY)].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"duplicate per-patient rows ({int(duplicated.sum())} rows), e.g. {examples}")


def paired_table(frame: pd.DataFrame, region: str, metric: str, methods: Sequence[str]) -> pd.DataFrame:
    """Patient x method table for paired tests; methods must cover identical patients.

    Raises if any (method, region, patient) row is duplicated or if a method
    was evaluated on a different patient set than the others. A metric that is
    undefined (NaN) for a patient stays NaN; callers drop such pairs.
    """

    subset = frame[(frame["region"] == region) & frame["method"].isin(methods)]
    check_unique_rows(subset)
    missing = [m for m in methods if m not in set(subset["method"])]
    if missing:
        raise ValueError(f"methods without {region} rows: {missing}")
    populations = {m: frozenset(g["patient_id"]) for m, g in subset.groupby("method")}
    reference = populations[methods[0]]
    for method in methods[1:]:
        if populations[method] != reference:
            only_method = sorted(populations[method] - reference)[:5]
            only_reference = sorted(reference - populations[method])[:5]
            raise ValueError(
                f"{method!r} and {methods[0]!r} cover different patients for {region} "
                f"(only in {method!r}: {only_method}, only in {methods[0]!r}: {only_reference}); "
                "paired comparisons need identical patient populations"
            )
    return subset.pivot(index="patient_id", columns="method", values=metric)[list(methods)]


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["method"] = np.where(
        frame["rule"] == "raw", frame["variant"], frame["variant"] + "/" + frame["rule"]
    )
    if "volume_error" in frame:
        frame["abs_volume_error"] = frame["volume_error"].abs()
    return frame


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (NaNs dropped)."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (np.nan, np.nan)
    if array.size == 1:
        return (float(array[0]), float(array[0]))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def summarize(
    frame: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
    *,
    with_ci: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean / std / median / IQR (and bootstrap CI) per method x region x metric."""

    records = []
    for (method, region), group in frame.groupby(["method", "region"], sort=False):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            record = {
                "method": method,
                "region": region,
                "metric": metric,
                "n": int(finite.size),
                "n_missing": int(values.size - finite.size),
                "mean": float(finite.mean()) if finite.size else np.nan,
                "std": float(finite.std(ddof=1)) if finite.size > 1 else np.nan,
                "median": float(np.median(finite)) if finite.size else np.nan,
                "q25": float(np.quantile(finite, 0.25)) if finite.size else np.nan,
                "q75": float(np.quantile(finite, 0.75)) if finite.size else np.nan,
            }
            if with_ci:
                record["ci_low"], record["ci_high"] = bootstrap_ci(finite, seed=seed)
            records.append(record)
    return pd.DataFrame.from_records(records)


def global_dice(frame: pd.DataFrame) -> pd.DataFrame:
    """Voxel-count (cohort) Dice per method x region, comparable to StreamingDice."""

    grouped = frame.groupby(["method", "region"], sort=False)[["tp", "pred_volume", "gt_volume"]].sum()
    denominator = grouped["pred_volume"] + grouped["gt_volume"]
    dice = (2 * grouped["tp"] / denominator.where(denominator > 0)).fillna(1.0)
    table = dice.unstack("region")[list(REGION_NAMES)]
    table["macro"] = table.mean(axis=1)
    return table


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment (NaNs are left untouched)."""

    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(p[finite])]
    m = order.size
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_comparison(
    frame: pd.DataFrame,
    reference: str,
    *,
    metrics: Sequence[str] = ("dice", "hd95", "nsd_2mm"),
    regions: Sequence[str] = REGION_NAMES,
    methods: Sequence[str] | None = None,
    n_resamples: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired per-patient differences (method - reference) with CI and Wilcoxon p.

    ``better`` / ``worse`` count patients improved / degraded according to the
    metric's direction. Holm adjustment is applied within each metric x region
    family across the compared methods.
    """

    if reference not in set(frame["method"]):
        raise ValueError(f"reference method {reference!r} not in results")
    methods = [m for m in (methods or frame["method"].unique()) if m != reference]
    unknown = sorted(set(methods) - set(frame["method"]))
    if unknown:
        raise ValueError(f"methods not in results: {unknown}")
    records = []
    for metric in metrics:
        higher = HIGHER_IS_BETTER.get(metric, True)
        for region in regions:
            pivot = paired_table(frame, region, metric, [reference, *methods])
            family = []
            for method in methods:
                # Same patients by construction; only pairs with an undefined metric drop out.
                pair = pivot[[reference, method]].dropna()
                diff = (pair[method] - pair[reference]).to_numpy(dtype=float)
                improvement = diff if higher else -diff
                if diff.size and np.any(diff != 0):
                    p_value = float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
                else:
                    p_value = np.nan
                low, high = bootstrap_ci(diff, n_resamples=n_resamples, seed=seed)
                family.append(
                    {
                        "metric": metric,
                        "region": region,
                        "method": method,
                        "reference": reference,
                        "n_patients": int(diff.size),
                        "reference_mean": float(pair[reference].mean()) if diff.size else np.nan,
                        "method_mean": float(pair[method].mean()) if diff.size else np.nan,
                        "mean_diff": float(diff.mean()) if diff.size else np.nan,
                        "median_diff": float(np.median(diff)) if diff.size else np.nan,
                        "ci_low": low,
                        "ci_high": high,
                        "better": int(np.count_nonzero(improvement > 0)),
                        "worse": int(np.count_nonzero(improvement < 0)),
                        "p_wilcoxon": p_value,
                    }
                )
            adjusted = holm_adjust([row["p_wilcoxon"] for row in family])
            for row, value in zip(family, adjusted):
                row["p_holm"] = float(value)
            records.extend(family)
    return pd.DataFrame.from_records(records)


def stratify_by_tumor_size(
    frame: pd.DataFrame,
    *,
    size_region: str = "WT",
    metric: str = "dice",
    metric_region: str = "ET",
    n_bins: int = 3,
    labels: Sequence[str] = ("small", "medium", "large"),
) -> pd.DataFrame:
    """Mean ``metric`` per method within patient tumor-size quantile bins.

    Bins come from the ground-truth ``size_region`` volume, which is identical
    for every method, so all methods share the same patient strata.
    """

    sizes = (
        frame[frame["region"] == size_region]
        .drop_duplicates("patient_id")
        .set_index("patient_id")["gt_volume"]
    )
    present = sizes[sizes > 0]
    strata = pd.qcut(present, q=n_bins, labels=list(labels)[:n_bins])
    subset = frame[frame["region"] == metric_region].copy()
    subset["stratum"] = subset["patient_id"].map(strata)
    subset = subset.dropna(subset=["stratum"])
    table = subset.groupby(["method", "stratum"], observed=True)[metric].agg(["mean", "count"]).unstack("stratum")
    return table


def failure_cases(
    frame: pd.DataFrame,
    reference: str,
    method: str,
    *,
    metric: str = "dice",
    region: str = "ET",
    k: int = 10,
) -> pd.DataFrame:
    """Patients with the largest degradation of ``method`` relative to ``reference``."""

    higher = HIGHER_IS_BETTER.get(metric, True)
    subset = frame[frame["region"] == region]
    pivot = paired_table(frame, region, metric, [reference, method]).dropna()
    pivot["delta"] = pivot[method] - pivot[reference]
    gt = subset.drop_duplicates("patient_id").set_index("patient_id")["gt_volume"]
    pivot["gt_volume"] = gt.reindex(pivot.index)
    return pivot.sort_values("delta", ascending=higher).head(k)


def boundary_profile(frame: pd.DataFrame, region: str = "WT") -> pd.DataFrame:
    """Share of each method's errors per distance band from the GT boundary."""

    subset = frame[(frame["region"] == region) & (frame["status"] == "both_present")]
    band_columns = [c for c in subset.columns if c.startswith("band_") and c.endswith(("_fp", "_fn"))]
    sums = subset.groupby("method")[band_columns].sum()
    bands = list(dict.fromkeys(c.rsplit("_", 1)[0] for c in band_columns))
    totals = pd.DataFrame({band: sums[f"{band}_fp"] + sums[f"{band}_fn"] for band in bands})
    return totals.div(totals.sum(axis=1), axis=0)


def patch_profile(frame: pd.DataFrame, region: str = "WT", patch_size: int = 16) -> pd.DataFrame:
    """Pooled error rate inside background / boundary / interior token patches."""

    subset = frame[frame["region"] == region]
    records = {}
    for method, group in subset.groupby("method"):
        record = {}
        for name in ("background", "boundary", "interior"):
            errors = group[f"patch_{name}_errors"].sum()
            pixels = group[f"patch_{name}_count"].sum() * patch_size * patch_size
            record[name] = errors / pixels if pixels else np.nan
        records[method] = record
    return pd.DataFrame.from_dict(records, orient="index")


def lesion_detection(frame: pd.DataFrame, region: str = "ET") -> pd.DataFrame:
    """Pooled detection rate of GT components per size bin, plus FP components/patient."""

    subset = frame[frame["region"] == region]
    bins = [c[: -len("_count")] for c in subset.columns if c.startswith("lesion_") and c.endswith("_count")]
    records = {}
    for method, group in subset.groupby("method"):
        record = {}
        for name in bins:
            count = group[f"{name}_count"].sum()
            record[f"{name}_recall"] = group[f"{name}_detected"].sum() / count if count else np.nan
            record[f"{name}_n"] = int(count)
        record["fp_components_per_patient"] = group["fp_components"].mean()
        records[method] = record
    return pd.DataFrame.from_dict(records, orient="index")


def format_table(
    summary: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("dice", "hd95", "nsd_2mm"),
    regions: Sequence[str] = REGION_NAMES,
    digits: int = 3,
    latex: bool = False,
) -> str:
    """Report-ready ``mean ± std`` table with one row per method."""

    subset = summary[summary["metric"].isin(metrics) & summary["region"].isin(regions)].copy()
    subset["cell"] = subset.apply(
        lambda row: f"{row['mean']:.{digits}f} ± {row['std']:.{digits}f}", axis=1
    )
    table = subset.pivot_table(index="method", columns=["metric", "region"], values="cell", aggfunc="first")
    table = table.reindex(columns=[(m, r) for m in metrics for r in regions if (m, r) in table.columns])
    if latex:
        return table.to_latex(escape=True).replace("±", r"$\pm$")
    return table.to_markdown() if _has_tabulate() else table.to_string()


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True
