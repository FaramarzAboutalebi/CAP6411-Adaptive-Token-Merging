"""A1 (decision rules), A2 (threshold), A7 (size / small lesions / where errors live) analysis.

Usage: python scripts/ws4_analyze.py <run_dir>   (the --out/validation187 folder of ws4_run_baseline.py)

Writes CSV + LaTeX tables and quick-look PNG/PDF figures (aggregate only, no MRI, no patient IDs in
figures) to <run_dir>/analysis. Patient-level tables (worst cases) stay in the private run dir.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402,F401
import pandas as pd  # noqa: E402

import _ws4_paths  # noqa: E402,F401  (puts src/ on sys.path)
from cap6411_atm.evaluation import visualization as viz  # noqa: E402
from cap6411_atm.evaluation.statistics import (  # noqa: E402
    boundary_profile, format_table, global_dice, lesion_detection, load_results, paired_comparison,
    patch_profile, stratify_by_tumor_size, summarize,
)

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
run_dir = Path(sys.argv[1])
out = run_dir / "analysis"
out.mkdir(parents=True, exist_ok=True)
frame = load_results([run_dir / "baseline"])
n_patients = frame["patient_id"].nunique()
pd.set_option("display.width", 200)

A1 = ["baseline", "baseline/nested", "baseline/nested_cc50", "baseline/nested_et_total200",
      "baseline/nested_cc50_et_total200"]
THR = {0.3: "baseline/thr_0.30", 0.4: "baseline/thr_0.40", 0.5: "baseline", 0.6: "baseline/thr_0.60",
       0.7: "baseline/thr_0.70"}
LABEL = {"baseline": "raw (thr 0.5)", "baseline/nested": "+ nesting", "baseline/nested_cc50": "+ nesting + CC<50 removed",
         "baseline/nested_et_total200": "+ nesting + ET<200 cleared",
         "baseline/nested_cc50_et_total200": "+ nesting + CC<50 + ET<200"}
REG = ["WT", "TC", "ET"]


def save(df, name, latex=None):
    df.to_csv(out / f"{name}.csv")
    if latex is not None:
        (out / f"{name}.tex").write_text(latex)
    print(f"\n== {name}\n{df.to_string()}")


# ---------------- A1: decision rules ----------------
gd = global_dice(frame)
save(gd.loc[A1].rename(index=LABEL).round(4), "A1_global_dice")
sub = frame[frame["method"].isin(A1)]
summ = summarize(sub, metrics=["dice", "hd95", "nsd_2mm"])
summ["method"] = summ["method"].map(LABEL)
save(summ.round(4), "A1_patient_summary",
     format_table(summ, metrics=["dice", "hd95", "nsd_2mm"], latex=True))
comp = paired_comparison(sub, "baseline", metrics=["dice", "hd95", "nsd_2mm"])
comp["method"] = comp["method"].map(LABEL)
save(comp[["metric", "region", "method", "mean_diff", "ci_low", "ci_high", "better", "worse", "p_holm"]]
     .round(5).reset_index(drop=True), "A1_paired_vs_raw")
status = (sub[sub["status"] != "both_present"].groupby(["method", "region", "status"]).size()
          .unstack(["region", "status"]).fillna(0).astype(int).rename(index=LABEL))
save(status, "A1_empty_mask_cases")

# ---------------- A2: threshold ----------------
thr_rows = []
for t, m in THR.items():
    row = {"threshold": t, **{f"global_{r}": gd.loc[m, r] for r in REG}, "global_macro": gd.loc[m, "macro"]}
    msub = frame[frame["method"] == m]
    for r in REG:
        rs = msub[msub["region"] == r]
        row[f"patient_dice_{r}"] = rs["dice"].mean()
        row[f"hd95_{r}"] = rs["hd95"].median()
        row[f"vol_ratio_{r}"] = rs["pred_volume"].sum() / rs["gt_volume"].sum()
    thr_rows.append(row)
thr = pd.DataFrame(thr_rows).set_index("threshold")
save(thr.round(4), "A2_threshold_sweep")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
colors = dict(zip(REG + ["macro"], viz.SERIES_COLORS))
for key in REG + ["macro"]:
    axes[0].plot(thr.index, thr[f"global_{key}"], marker="o", label=key, color=colors[key])
axes[0].set(xlabel="threshold", ylabel="global Dice", title="A2: Dice vs threshold")
axes[0].legend(frameon=False, fontsize=8)
for key in REG:
    axes[1].plot(thr.index, thr[f"vol_ratio_{key}"], marker="o", label=key, color=colors[key])
axes[1].axhline(1.0, color="0.5", lw=0.8, ls="--")
axes[1].set(xlabel="threshold", ylabel="predicted / true volume", title="Over- vs under-segmentation")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
viz.save_figure(fig, out / "A2_threshold_sweep")

# ---------------- A7: tumor size, small lesions, where errors live (raw baseline) ----------------
base = frame[frame["method"] == "baseline"]
for metric_region in REG:
    strat = stratify_by_tumor_size(base, size_region="WT", metric="dice", metric_region=metric_region)
    save(strat.round(4), f"A7_size_strata_dice_{metric_region}")
lesions = pd.concat({r: lesion_detection(sub[sub["method"].isin(["baseline", "baseline/nested_cc50_et_total200"])],
                                         region=r) for r in REG}, names=["region", "method"])
save(lesions.round(4), "A7_lesion_detection")
bands = pd.concat({r: boundary_profile(base, region=r) for r in REG}, names=["region", "method"]).droplevel("method")
save(bands.round(4), "A7_error_share_by_boundary_distance")
patches = pd.concat({r: patch_profile(base, region=r) for r in REG}, names=["region", "method"]).droplevel("method")
save(patches.round(5), "A7_error_rate_by_token_patch_type")

viz.save_figure(viz.plot_profile(bands, ylabel="share of FP+FN voxels",
                                 title="Baseline: where errors are (distance to GT boundary)"),
                out / "A7_boundary_bands")
viz.save_figure(viz.plot_profile(patches, ylabel="error rate (FP+FN)/pixels",
                                 title="Baseline: error rate per 16x16 token-patch type"),
                out / "A7_patch_profile")

fig, ax = plt.subplots(figsize=(6, 3.6))
for i, r in enumerate(REG):
    rs = base[(base["region"] == r) & (base["gt_volume"] > 0)]
    ax.scatter(rs["gt_volume"], rs["dice"], s=10, alpha=0.7, label=r, color=viz.SERIES_COLORS[i])
ax.set_xscale("log")
ax.set(xlabel="ground-truth volume (voxels, log)", ylabel="patient Dice", title="A7: Dice vs tumor size")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
viz.save_figure(fig, out / "A7_dice_vs_volume")

fig, ax = plt.subplots(figsize=(6, 3.6))
data = [base[base["region"] == r]["dice"].dropna() for r in REG]
ax.boxplot(data, tick_labels=REG, showfliers=True)
ax.set(ylabel="patient Dice", title=f"Baseline per-patient Dice (n={n_patients})")
fig.tight_layout()
viz.save_figure(fig, out / "baseline_dice_distribution")

viz.save_figure(viz.plot_paired_differences(sub, "baseline", metric="dice", region="ET"), out / "A1_paired_dice_ET")
viz.save_figure(viz.plot_paired_differences(sub, "baseline", metric="hd95", region="ET"), out / "A1_paired_hd95_ET")

# Worst cases (patient IDs -> private dir only)
worst = (base[base["region"] == "ET"].nsmallest(10, "dice")
         [["patient_id", "dice", "hd95", "gt_volume", "pred_volume", "status", "fp_components"]])
worst.to_csv(run_dir / "worst_ET_cases_PRIVATE.csv", index=False)
print("\n== worst ET cases (private)\n", worst.drop(columns="patient_id").to_string(index=False))
print(f"\npatients: {n_patients}; outputs in {out}")
