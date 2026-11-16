"""Presentation-quality Workstream 4 figures from the 187-patient validation run (aggregate data only).

Usage: python scripts/ws4_figures.py <run_dir> [out_dir]   (out_dir defaults to <run_dir>/figures)

Requires seaborn. Figures contain no MRI and no patient IDs, so they are safe to commit
(docs/figures/workstream4 holds the reported versions).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

import _ws4_paths  # noqa: E402,F401  (puts src/ on sys.path)
from cap6411_atm.evaluation.statistics import global_dice, load_results, paired_comparison  # noqa: E402

if len(sys.argv) not in (2, 3):
    raise SystemExit(__doc__)
RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) == 3 else RUN / "figures"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(0)  # seaborn strip-plot jitter -> byte-reproducible figures

# ---------- style ----------
REG = ["WT", "TC", "ET"]
REG_NAME = {"WT": "Whole tumor (WT)", "TC": "Tumor core (TC)", "ET": "Enhancing tumor (ET)"}
PAL = {"WT": "#0072B2", "TC": "#009E73", "ET": "#D55E00"}  # Okabe-Ito, colorblind-safe
INK, MUTED, GRID = "#1F2933", "#616E7C", "#E4E7EB"
sns.set_theme(context="talk", style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "axes.edgecolor": "#9AA5B1", "axes.linewidth": 0.8,
    "axes.titleweight": "bold", "axes.titlesize": 17, "axes.labelsize": 14, "axes.labelcolor": INK,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "xtick.color": INK, "ytick.color": INK,
    "legend.fontsize": 12, "legend.title_fontsize": 12, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlepad": 12, "figure.titlesize": 20, "figure.titleweight": "bold", "text.color": INK,
})


def finish(fig, name, title, subtitle, header_in=1.45):
    """Reserve a fixed-height header (inches) so title, subtitle and panel titles never collide."""
    h = fig.get_size_inches()[1]
    fig.subplots_adjust(top=1 - header_in / h)
    fig.text(0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=21, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.55 / h, subtitle, ha="left", va="top", fontsize=13.5, color=MUTED)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


frame = load_results([RUN / "baseline"])
base = frame[frame["method"] == "baseline"].copy()
base["Region"] = pd.Categorical(base["region"], REG, ordered=True)
N = base["patient_id"].nunique()
gd = global_dice(frame)

# ---------- 1. Baseline overview ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), gridspec_kw={"wspace": 0.28})
specs = [("dice", "Dice ↑", (0, 1.02), False), ("hd95", "HD95 (mm, log scale) ↓", None, True),
         ("nsd_2mm", "Surface Dice @ 2 mm ↑", (0, 1.02), False)]
for ax, (col, label, ylim, log) in zip(axes, specs):
    data = base[base["status"] == "both_present"] if col == "hd95" else base
    common = dict(data=data, x="Region", y=col, hue="Region", palette=PAL, ax=ax, legend=False, dodge=False,
                  log_scale=log)
    sns.violinplot(**common, inner=None, cut=0, linewidth=0, saturation=0.9, alpha=0.25, density_norm="width",
                   width=0.85)
    sns.stripplot(**common, size=3.2, alpha=0.5, jitter=0.16, edgecolor="white", linewidth=0.3)
    sns.boxplot(**common, width=0.16, showfliers=False, boxprops={"alpha": 0.95, "edgecolor": INK, "zorder": 3},
                medianprops={"color": "white", "linewidth": 2.4, "zorder": 4}, whiskerprops={"color": INK},
                capprops={"color": INK})
    ax.set(xlabel="", ylabel=label)
    if ylim:
        ax.set_ylim(*ylim)
    if log:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    meds = [data[data["region"] == r][col].median() for r in REG]
    ax.set_xticks(range(3), [f"{r}\nmed. {m:.1f} mm" if log else f"{r}\nmedian {m:.2f}" for r, m in zip(REG, meds)])
axes[0].set_title("Overlap")
axes[1].set_title("Boundary distance")
axes[2].set_title("Boundary agreement")
for i, r in enumerate(REG):
    axes[0].text(i, 0.04, f"global\n{gd.loc['baseline', r]:.3f}", ha="center", fontsize=11, color=PAL[r],
                 fontweight="bold")
finish(fig, "01_baseline_overview", "Baseline segmentation quality per patient",
       f"Frozen DINOv3 ViT-B/16, BraTS 2023 GLI validation (n = {N} patients). Global (pooled) Macro Dice "
       f"{gd.loc['baseline', 'macro']:.4f} reproduces Workstream 1 (0.8192). HD95 shown where both masks are non-empty.")

# ---------- 2. Where errors happen: distance bands ----------
bands = ["band_0_2mm", "band_2_4mm", "band_4_8mm", "band_ge8mm"]
band_lbl = ["0–2 mm", "2–4 mm", "4–8 mm", "≥ 8 mm"]
present = base[base["status"] == "both_present"]
share = pd.DataFrame({b: present.groupby("region")[[f"{b}_fp", f"{b}_fn"]].sum().sum(axis=1) for b in bands}).loc[REG]
share = share.div(share.sum(axis=1), axis=0)
band_colors = sns.color_palette("rocket_r", 6)[1:5]
fig, ax = plt.subplots(figsize=(15, 5.2))
left = np.zeros(len(REG))
for b, lbl, c in zip(bands, band_lbl, band_colors):
    vals = share[b].to_numpy()
    ax.barh([REG_NAME[r] for r in REG], vals, left=left, color=c, label=lbl, edgecolor="white", height=0.62)
    for y, (v, l) in enumerate(zip(vals, left)):
        if v > 0.025:
            ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=13 if v > 0.05 else 10.5,
                    fontweight="bold",
                    color="white" if c != band_colors[0] else INK)
    left += vals
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xlabel("share of all misclassified voxels (FP + FN)")
ax.legend(title="distance from true boundary", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
ax.grid(axis="y", visible=False)
finish(fig, "02_errors_by_boundary_distance", "Most errors sit right on the tumor boundary",
       "Share of voxel errors by distance from the ground-truth surface (baseline, threshold 0.5).")

# ---------- 3. Token-patch error rates ----------
rows = []
for r in REG:
    g = base[base["region"] == r]
    for t, lbl in [("background", "Background\n(no tumor)"), ("boundary", "Boundary\n(partly tumor)"),
                   ("interior", "Interior\n(all tumor)")]:
        rows.append({"region": r, "patch": lbl, "rate": g[f"patch_{t}_errors"].sum() / (g[f"patch_{t}_count"].sum() * 256),
                     "count": int(g[f"patch_{t}_count"].sum())})
pp = pd.DataFrame(rows)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 6.2), gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.3})
sns.barplot(data=pp, x="patch", y="rate", hue="region", palette=PAL, ax=ax, edgecolor="white", saturation=0.95)
ax.set_yscale("log")
ax.set_ylim(5e-5, 3)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=2))
for cont in ax.containers:
    ax.bar_label(cont, labels=[f"{v.get_height():.2%}" for v in cont], fontsize=10.5, padding=3, color=INK)
ax.set(xlabel="16 × 16-pixel patch = one DINOv3 token", ylabel="error rate (log scale)")
ax.set_title("Error rate inside each kind of token")
ax.legend(title="", frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
wt = pp[pp["region"] == "WT"].set_index("patch")["rate"]
ratio = wt.iloc[1] / wt.iloc[0]
ax.annotate(f"WT boundary tokens:\n~{ratio:,.0f}× the error rate\nof background tokens",
            xy=(0.73, wt.iloc[1] * 0.6), xytext=(-0.4, 0.0012), fontsize=12.5, color=PAL["WT"], fontweight="bold",
            arrowprops={"arrowstyle": "-|>", "color": PAL["WT"], "lw": 1.6, "connectionstyle": "arc3,rad=-0.2"})
counts = pp[pp["region"] == "WT"].copy()
counts["share"] = counts["count"] / counts["count"].sum()
sns.barplot(data=counts, x="patch", y="share", color="#9AA5B1", ax=ax2, edgecolor="white")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
for cont in ax2.containers:
    ax2.bar_label(cont, labels=[f"{v.get_height():.1%}" for v in cont], fontsize=11, padding=3)
ax2.set(xlabel="", ylabel="share of all tokens", ylim=(0, 1.05))
ax2.set_title("How common each token is (WT)")
ax2.tick_params(axis="x", labelsize=10.5)
finish(fig, "03_token_patch_errors", "Errors concentrate in the few tokens that straddle the boundary",
       "Each axial slice tiled into DINOv3's 15 × 15 grid of 16 × 16 patches. Merging background tokens should be "
       "nearly free; boundary tokens must be protected.")

# ---------- 4. A1 decision rules: forest plot ----------
A1 = {"baseline/nested": "Nesting (ET ⊆ TC ⊆ WT)", "baseline/nested_cc50": "+ remove blobs < 50 vox",
      "baseline/nested_et_total200": "+ clear ET < 200 vox", "baseline/nested_cc50_et_total200": "+ both"}
sub = frame[frame["method"].isin(["baseline", *A1])]
comp = paired_comparison(sub, "baseline", metrics=["dice", "hd95", "nsd_2mm"])
comp["rule"] = comp["method"].map(A1)
fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), sharey=True, gridspec_kw={"wspace": 0.08})
metric_lbl = {"dice": "Δ Dice (higher = better)", "hd95": "Δ HD95, mm (lower = better)",
              "nsd_2mm": "Δ Surface Dice @2 mm (higher = better)"}
order = list(A1.values())
for ax, m in zip(axes, ["dice", "hd95", "nsd_2mm"]):
    d = comp[comp["metric"] == m]
    for j, r in enumerate(REG):
        dr = d[d["region"] == r].set_index("rule").reindex(order)
        y = np.arange(len(order)) + (j - 1) * 0.22
        ax.errorbar(dr["mean_diff"], y, xerr=[dr["mean_diff"] - dr["ci_low"], dr["ci_high"] - dr["mean_diff"]],
                    fmt="o", color=PAL[r], ms=8, capsize=3, lw=2, label=r)
    good = "right" if m != "hd95" else "left"
    ax.axvline(0, color=INK, lw=1, ls="--")
    ax.set_xlabel(metric_lbl[m])
    lo, hi = ax.get_xlim()
    span = max(abs(lo), abs(hi))
    ax.set_xlim(-span, span)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5, symmetric=True))
    if m != "hd95":
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.3f}" if v else "0"))
    ax.axvspan(0 if good == "right" else -span, span if good == "right" else 0, color="#E3F2E9", zorder=0)
    ax.text(0.98 if good == "right" else 0.02, 0.02, "better →" if good == "right" else "← better",
            transform=ax.transAxes, ha="right" if good == "right" else "left", fontsize=11, color="#2E7D4F")
axes[0].set_yticks(range(len(order)), order)
axes[0].invert_yaxis()
axes[0].legend(title="region", frameon=False, loc="lower left")
axes[0].set_title("Dice")
axes[1].set_title("HD95")
axes[2].set_title("Surface Dice")
macro = gd.loc[["baseline", *A1], "macro"]
finish(fig, "04_A1_postprocessing", "A1 · Post-processing: statistically real, practically tiny",
       f"Paired per-patient change vs raw output (mean, 95% bootstrap CI, n = {N}). Global Macro Dice "
       f"{macro.iloc[0]:.4f} → {macro.max():.4f} at best; mainly trims stray blobs (HD95).")

# ---------- 5. A2 threshold sweep ----------
THR = {0.3: "baseline/thr_0.30", 0.4: "baseline/thr_0.40", 0.5: "baseline", 0.6: "baseline/thr_0.60",
       0.7: "baseline/thr_0.70"}
rows = []
for t, m in THR.items():
    g = frame[frame["method"] == m]
    for r in REG:
        gr = g[g["region"] == r]
        rows.append({"threshold": t, "region": r, "dice": gd.loc[m, r],
                     "vol": gr["pred_volume"].sum() / gr["gt_volume"].sum()})
    rows.append({"threshold": t, "region": "Macro", "dice": gd.loc[m, "macro"], "vol": np.nan})
ts = pd.DataFrame(rows)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"wspace": 0.25})
pal2 = {**PAL, "Macro": INK}
sns.lineplot(data=ts, x="threshold", y="dice", hue="region", palette=pal2, marker="o", ms=9, lw=2.6, ax=a1,
             legend=False)
a1.set(ylabel="global Dice", xlabel="probability threshold", ylim=(0.70, 0.90))
a1.set_title("Dice is flat across thresholds")
for r in ["WT", "TC", "ET", "Macro"]:
    v = ts[(ts.region == r) & (ts.threshold == 0.5)]["dice"].iloc[0]
    a1.annotate(f"{v:.3f}", (0.5, v), xytext=(0, -20 if r == "TC" else 9), textcoords="offset points",
                ha="center", fontsize=11, color=pal2[r], fontweight="bold")
    end = ts[(ts.region == r) & (ts.threshold == 0.7)]["dice"].iloc[0]
    a1.text(0.715, end, r, va="center", fontsize=13, fontweight="bold", color=pal2[r])
sns.lineplot(data=ts[ts.region != "Macro"], x="threshold", y="vol", hue="region", palette=PAL, marker="o", ms=9,
             lw=2.6, ax=a2, legend=False)
for r in REG:
    end = ts[(ts.region == r) & (ts.threshold == 0.7)]["vol"].iloc[0]
    a2.text(0.715, end + (0.012 if r == "TC" else -0.012 if r == "WT" else 0), r, va="center", fontsize=13,
            fontweight="bold", color=PAL[r])
a2.axhline(1, color=INK, ls="--", lw=1)
a2.axhspan(0.95, 1.05, color="#E3F2E9", zorder=0)
a2.text(0.31, 0.965, "perfect volume ±5 %", va="center", fontsize=11, color="#2E7D4F")
a2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}×"))
a2.set(ylabel="predicted ÷ true volume", xlabel="probability threshold")
a2.set_title("ET is over-segmented at every threshold")
for a in (a1, a2):
    a.axvline(0.5, color=MUTED, lw=1, ls=":")
    a.set_xlim(0.27, 0.77)
    a.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
finish(fig, "05_A2_threshold", "A2 · Threshold sensitivity",
       "Moving the threshold from 0.3 to 0.7 changes Macro Dice by < 0.001; ET stays ~1.3–1.4× too large "
       "→ an error of the model, not of the cut-off. (Validation-only sensitivity check.)")

# ---------- 6. Tumor size ----------
wt_vol = base[base["region"] == "WT"].set_index("patient_id")["gt_volume"]
tert = pd.qcut(wt_vol, 3, labels=["Small", "Medium", "Large"])
edges = wt_vol.groupby(tert, observed=True).agg(["min", "max"])
base["size"] = base["patient_id"].map(tert)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(19, 6.4), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.22})
sns.boxplot(data=base, x="size", y="dice", hue="Region", palette=PAL, ax=a1, showfliers=False, width=0.7,
            boxprops={"edgecolor": INK, "alpha": 0.85}, medianprops={"color": "white", "lw": 2})
sns.stripplot(data=base, x="size", y="dice", hue="Region", palette=PAL, ax=a1, dodge=True, size=2.6, alpha=0.4,
              legend=False, jitter=0.12)
a1.set_xticks(range(3), [f"{s}\n{edges.loc[s, 'min'] / 1000:.0f}k–{edges.loc[s, 'max'] / 1000:.0f}k vox"
                         for s in ["Small", "Medium", "Large"]])
a1.set(xlabel="whole-tumor size tertile (62–63 patients each)", ylabel="patient Dice", ylim=(0, 1.02))
a1.set_title("Dice by tumor size")
a1.legend(title="region", frameon=False, loc="lower right")
log_edges = np.logspace(1, 5.7, 15)
for i, r in enumerate(REG):
    g = base[(base["region"] == r) & (base["gt_volume"] > 0)].copy()
    a2.scatter(g["gt_volume"], g["dice"], s=22, alpha=0.3, color=PAL[r], edgecolor="none")
    g["bin"] = pd.cut(g["gt_volume"], log_edges)
    st = g.groupby("bin", observed=True).agg(x=("gt_volume", "median"), med=("dice", "median"),
                                             q1=("dice", lambda s: s.quantile(0.25)),
                                             q3=("dice", lambda s: s.quantile(0.75)), n=("dice", "size"))
    st = st[st["n"] >= 8]
    a2.fill_between(st["x"], st["q1"], st["q3"], color=PAL[r], alpha=0.15, lw=0)
    a2.plot(st["x"], st["med"], color=PAL[r], lw=3, marker="o", ms=6, label=f"{r}: median & IQR")
a2.set_xscale("log")
a2.set_xlim(700, 4e5)
a2.set(xlabel="true volume of that region (voxels = mm³, log)", ylabel="patient Dice", ylim=(0, 1.02))
a2.set_title("Small regions are hardest")
a2.legend(frameon=False, loc="lower right")
finish(fig, "06_A7_tumor_size", "A7 · Performance by tumor size",
       "Smaller tumors and smaller sub-regions lose the most Dice — exactly where coarse tokens hurt. "
       "(Right: log-spaced bins with ≥ 8 patients.)")

# ---------- 7. Lesion detection ----------
bins = [("lesion_0_100", "< 100"), ("lesion_100_1000", "100–1k"), ("lesion_1000_10000", "1k–10k"),
        ("lesion_ge10000", "≥ 10k")]
rows = []
for r in REG:
    g = base[base["region"] == r]
    for key, lbl in bins:
        n = g[f"{key}_count"].sum()
        rows.append({"region": r, "bin": lbl, "recall": g[f"{key}_detected"].sum() / n if n else np.nan, "n": int(n)})
ld = pd.DataFrame(rows)
fp = base.groupby("region")["fp_components"].mean().reindex(REG)
fp_clean = frame[frame["method"] == "baseline/nested_cc50_et_total200"].groupby("region")["fp_components"].mean().reindex(REG)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 6.2), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.25})
sns.barplot(data=ld, x="bin", y="recall", hue="region", palette=PAL, ax=a1, edgecolor="white")
for cont, r in zip(a1.containers, REG):
    labels = [f"{v:.0%}\nn={n}" for v, n in zip(ld[ld.region == r]["recall"], ld[ld.region == r]["n"])]
    a1.bar_label(cont, labels=labels, fontsize=9.5, padding=3)
a1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
a1.set(ylim=(0, 1.22), xlabel="size of each separate ground-truth lesion (voxels)", ylabel="lesions detected")
a1.set_title("Small lesions are often missed")
a1.legend(title="region", frameon=False, loc="upper left", ncol=3)
x = np.arange(3)
a2.bar(x - 0.2, fp.values, 0.38, color=[PAL[r] for r in REG], label="raw output", edgecolor="white")
a2.bar(x + 0.2, fp_clean.values, 0.38, color=[PAL[r] for r in REG], alpha=0.4, hatch="//", edgecolor="white",
       label="after post-processing")
for xi, (a, b) in enumerate(zip(fp.values, fp_clean.values)):
    a2.text(xi - 0.2, a + 0.08, f"{a:.1f}", ha="center", fontsize=12)
    a2.text(xi + 0.2, b + 0.08, f"{b:.1f}", ha="center", fontsize=12)
a2.set_xticks(x, REG)
a2.set(ylabel="false-positive blobs per patient", ylim=(0, fp.max() * 1.25))
a2.set_title("Spurious blobs")
a2.legend(frameon=False)
finish(fig, "07_A7_lesion_detection", "A7 · Lesion-level detection",
       "A lesion counts as detected if any predicted voxel overlaps it (26-connected components).")

# ---------- 8. Volume agreement ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), gridspec_kw={"wspace": 0.25})
for ax, r in zip(axes, REG):
    g = base[(base["region"] == r) & ((base["gt_volume"] > 0) | (base["pred_volume"] > 0))]
    gv, pv = g["gt_volume"].clip(lower=10), g["pred_volume"].clip(lower=10)
    ax.scatter(gv, pv, s=28, alpha=0.6, color=PAL[r], edgecolor="white", linewidth=0.4)
    lo, hi = 10, max(gv.max(), pv.max()) * 1.6
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, ls="--", label="perfect")
    ax.fill_between([lo, hi], [lo / 1.25, hi / 1.25], [lo * 1.25, hi * 1.25], color="#E3F2E9", zorder=0,
                    label="±25 %")
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="true volume (voxels)",
           ylabel="predicted volume (voxels)" if r == "WT" else "")
    ratio = g["pred_volume"].sum() / g["gt_volume"].sum()
    ax.set_title(f"{REG_NAME[r]}\npredicted / true = {ratio:.2f}×", fontsize=15)
    ax.legend(frameon=False, loc="upper left", fontsize=11)
finish(fig, "08_volume_agreement", "Volume agreement per patient",
       "Points above the diagonal are over-segmented. ET is systematically too large (the model's main bias). "
       "Empty masks drawn at 10 voxels.", header_in=1.9)

print("all figures in", OUT)
