"""Qualitative figure: good / typical / hard validation cases with GT, prediction, errors and the token grid.

Usage: python scripts/ws4_qualitative.py <run_dir> [data/model args, see _ws4_paths.py]

The figure shows patient MRI (no IDs). Keep it in private storage / slides; do not commit it.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

from _ws4_paths import INSPECTION_CONFIG, MODEL_CONFIG, add_data_args, require  # noqa: E402

from cap6411_atm.evaluation import build_eval_loader, evaluate_model  # noqa: E402
from cap6411_atm.models import (  # noqa: E402
    DINOv3SegmentationConfig, build_pretrained_brats_dinov3, load_adapter_decoder_artifact,
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run_dir", type=Path, help="validation187 folder written by ws4_run_baseline.py")
parser.add_argument("--out", type=Path, default=None, help="defaults to <run_dir>/figures")
add_data_args(parser)
args = parser.parse_args()
require(args, "archive", "manifest", "artifact", "dinov3_repo", "dinov3_checkpoint")
RUN = args.run_dir
OUT = args.out or RUN / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PAL = {"WT": "#0072B2", "TC": "#009E73", "ET": "#D55E00"}
INK, MUTED = "#1F2933", "#616E7C"

frame = pd.read_csv(RUN / "baseline/per_patient.csv")
dice = frame[frame.rule == "raw"].pivot(index="patient_id", columns="region", values="dice")
dice["mean"] = dice[["WT", "TC", "ET"]].mean(axis=1)
dice = dice.sort_values("mean")
n = len(dice)
cases = [("Good case", dice.index[int(n * 0.9)]), ("Typical case", dice.index[n // 2]),
         ("Hard case", dice.index[int(n * 0.08)])]

device = "cuda" if torch.cuda.is_available() else "cpu"
config = DINOv3SegmentationConfig.from_json(MODEL_CONFIG)
model = build_pretrained_brats_dinov3(args.dinov3_repo, args.dinov3_checkpoint, config)
load_adapter_decoder_artifact(model, args.artifact)
model = model.to(device).eval()
ids = [pid for _, pid in cases]
loader = build_eval_loader(args.archive, args.manifest, INSPECTION_CONFIG, split="validation", patients=ids,
                           batch_size=32, cache_dir=args.cache_dir)
res = evaluate_model(model, loader, rules={"raw": {"threshold": 0.5}}, device=device, save_patients=ids)


def window(img, mask):
    lo, hi = np.percentile(img[mask], [1, 99.5])
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)


def labelmap(m):  # m: [3,H,W] -> 0 bg, 1 WT-only, 2 TC-only, 3 ET
    lab = np.zeros(m.shape[1:], np.uint8)
    lab[m[0]] = 1
    lab[m[1]] = 2
    lab[m[2]] = 3
    return lab


def overlay(ax, base, m):
    ax.imshow(base, cmap="gray", interpolation="lanczos")
    lab = labelmap(m)
    cmap = ListedColormap([(0, 0, 0, 0), PAL["WT"], PAL["TC"], PAL["ET"]])
    ax.imshow(np.ma.masked_where(lab == 0, lab), cmap=cmap, vmin=0, vmax=3, alpha=0.45, interpolation="nearest")
    for k, r in enumerate(["WT", "TC", "ET"]):
        if m[k].any():
            ax.contour(m[k], levels=[0.5], colors=[PAL[r]], linewidths=1.6)


fig, axes = plt.subplots(3, 4, figsize=(17, 13.6), gridspec_kw={"wspace": 0.03, "hspace": 0.12})
for row, (label, pid) in enumerate(cases):
    s = res.saved_predictions[pid]
    tgt, pred, img = s["target"].astype(bool), s["raw"].astype(bool), s["image"]
    z = int(np.argmax(tgt[0].sum(axis=(0, 1))))
    brain = np.any(np.abs(img[:, :, :, z]) > 1e-6, axis=0)  # z-scored volumes keep background at exactly 0
    # crop to the brain, snapped to the 16-px token grid, then rotate to radiological view
    rows_, cols_ = np.where(ndimage.binary_dilation(brain, iterations=4))
    side = min(240, -(-max(np.ptp(rows_), np.ptp(cols_)) // 16) * 16 + 16)  # square, token-aligned crop

    def span(lo, hi):
        start = int(np.clip(((lo + hi) // 2 - side // 2) // 16 * 16, 0, 240 - side))
        return start, start + side

    (r0, r1), (c0, c1) = span(rows_.min(), rows_.max()), span(cols_.min(), cols_.max())
    rot = lambda a: np.rot90(a[..., r0:r1, c0:c1], axes=(-2, -1))  # noqa: E731
    t1c = rot(window(img[1, :, :, z], brain) * brain)
    flair = rot(window(img[3, :, :, z], brain) * brain)
    g, p = rot(tgt[:, :, :, z]), rot(pred[:, :, :, z])

    axes[row, 0].imshow(t1c, cmap="gray", interpolation="lanczos")
    overlay(axes[row, 1], flair, g)
    overlay(axes[row, 2], flair, p)
    # errors for WT + token grid with boundary tokens highlighted
    ax = axes[row, 3]
    ax.imshow(flair * 0.55, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    tp, fp, fn = p[0] & g[0], p[0] & ~g[0], g[0] & ~p[0]
    err = np.zeros((*fp.shape, 4))
    err[tp] = matplotlib.colors.to_rgba("#9AA5B1", 0.55)
    err[fp] = matplotlib.colors.to_rgba("#FF4FA3", 1.0)
    err[fn] = matplotlib.colors.to_rgba("#00E5FF", 1.0)
    ax.imshow(err, interpolation="nearest")
    H, W = g.shape[1:]
    for i in range(0, H, 16):
        for j in range(0, W, 16):
            frac = g[0, i:i + 16, j:j + 16].mean()
            if 0 < frac < 1:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 16, 16, fill=False, ec="#F0E442", lw=1.1, alpha=0.95))
    for k in range(0, W + 1, 16):
        ax.axvline(k - 0.5, color="white", lw=0.35, alpha=0.35)
    for k in range(0, H + 1, 16):
        ax.axhline(k - 0.5, color="white", lw=0.35, alpha=0.35)
    d = dice.loc[pid]
    axes[row, 0].text(0.03, 0.97, label, transform=axes[row, 0].transAxes, va="top", fontsize=15,
                      fontweight="bold", color="white")
    axes[row, 0].text(0.03, 0.03, f"Dice  WT {d['WT']:.2f} · TC {d['TC']:.2f} · ET {d['ET']:.2f}",
                      transform=axes[row, 0].transAxes, va="bottom", fontsize=12, color="white")
    for ax in axes[row]:
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

for ax, t in zip(axes[0], ["MRI (T1 contrast-enhanced)", "Ground truth", "Baseline prediction",
                           "WT errors + DINOv3 token grid"]):
    ax.set_title(t, fontsize=15, fontweight="bold", color=INK, pad=10)

handles = [Patch(fc=PAL["WT"], alpha=0.6, label="WT (edema)"), Patch(fc=PAL["TC"], alpha=0.6, label="TC (core)"),
           Patch(fc=PAL["ET"], alpha=0.6, label="ET (enhancing)"),
           Patch(fc="#9AA5B1", label="correct (WT)"), Patch(fc="#FF4FA3", label="false positive"),
           Patch(fc="#00E5FF", label="false negative"),
           Patch(fc="none", ec="#F0E442", lw=2, label="boundary token (16×16 px)")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.03))
for ax in axes.flat:
    ax.set_facecolor("black")
fig.text(0.125, 0.965, "What the baseline gets right and wrong", fontsize=22, fontweight="bold", color=INK)
fig.text(0.125, 0.945, "Axial slice with the largest tumor area per case (FLAIR background). Cases picked at the "
         "90th / 50th / 8th percentile of mean Dice. Errors hug the boundary tokens (yellow).",
         fontsize=13.5, color=MUTED, va="top")
fig.savefig(OUT / "09_qualitative_cases.png", dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "09_qualitative_cases.pdf", bbox_inches="tight", facecolor="white")
print("saved 09_qualitative_cases", json.dumps({c: round(float(dice.loc[p, "mean"]), 3) for c, p in cases}))
