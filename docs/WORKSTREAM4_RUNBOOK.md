# Workstream 4 runbook: running the evaluation

How to reproduce the Workstream 4 baseline evaluation (ablations **A1, A2,
A7**) on your own machine or in Colab, and how the merging ablations
(**A3–A6**) will run once Workstream 2 is ready. Metric definitions and
results are in [`EVALUATION.md`](EVALUATION.md).

| Ablation | Question | Status |
|---|---|---|
| A1 | Do post-processing rules (nesting, small-blob removal) help? | **Done** (baseline) |
| A2 | Is the 0.5 threshold the problem? | **Done** (baseline) |
| A7 | How does accuracy depend on tumor size, small lesions and the boundary? | **Done** (baseline) |
| A3 | Dice vs merge ratio, uniform and adaptive | Waiting on Workstream 2 |
| A4 | Adaptive vs uniform at the same ratio (boundary bands, token patches) | Waiting on Workstream 2 |
| A5 | Adaptive merging vs lower resolution at the same measured latency | Waiting on Workstreams 2 and 3 |
| A6 | Merge design: where to merge, boundary protection, reconstruction | Waiting on Workstream 2 |

All results are on the **validation split (187 patients)**. The 188-patient
test split is never touched; the code refuses it (see "Guardrails" below).

## 1. What you need

| Input | File | Where to get it | Check |
|---|---|---|---|
| BraTS 2023 GLI training data | `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip` (13.2 GB, 1,251 patients) | Synapse / team Drive | Do not unzip; the loader reads the ZIP |
| Split manifest | `brats2023_gli_split_seed42.json` | Team Drive (Workstream 1 package) | 876 / 187 / 188 patients |
| Workstream 1 adapter + decoder | `dinov3_vitb16_brats_epoch2_adapter_decoder.pt` | Team Drive `workstream1_epoch2/` | SHA-256 `ed0d84ca…` ([`WORKSTREAM1_ARTIFACTS.md`](WORKSTREAM1_ARTIFACTS.md)) |
| DINOv3 backbone weights | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | Meta's DINOv3 access form (each person accepts the license) | Hash prefix `73cec8be` is in the filename |
| DINOv3 code | `facebookresearch/dinov3` at commit `6876159` | `git clone` + `git checkout 6876159` | |

No weights, data or patient IDs are in this repository; never commit them.

**Python environment:** Python 3.11+, a CUDA build of PyTorch, then

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `scipy`, `pandas` and `seaborn` (figures).
DINOv3 also needs `ftfy omegaconf regex scikit-learn submitit termcolor torchmetrics`.

## 2. Point the scripts at your files (once per shell)

The scripts never hard-code paths. Set these environment variables, or pass
the matching flags (`--archive`, `--manifest`, `--artifact`, `--dinov3-repo`,
`--dinov3-checkpoint`, `--cache-dir`).

Bash / Git Bash:

```bash
export CAP6411_BRATS_ZIP=/path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip
export CAP6411_MANIFEST=/path/to/brats2023_gli_split_seed42.json
export CAP6411_ARTIFACT=/path/to/dinov3_vitb16_brats_epoch2_adapter_decoder.pt
export CAP6411_DINOV3_REPO=/path/to/dinov3
export CAP6411_DINOV3_CKPT=/path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
export CAP6411_CACHE_DIR=/path/to/private/cache   # optional, speeds up reruns
```

PowerShell: `$env:CAP6411_BRATS_ZIP = "D:\path\to\...zip"` and so on.

Keep the output folder **outside the repo** (for example a private Drive
folder): the per-patient CSVs contain patient IDs.

## 3. Check the code first (no GPU, no data, ~5 s)

```bash
PYTHONPATH=src python -m unittest discover -s tests -t tests
```

Expect `OK` (106 tests). The evaluation tests use small synthetic volumes, so
they need no weights or data.

## 4. Run A1 + A2 (one GPU pass)

```bash
python scripts/ws4_run_baseline.py --out /path/to/private/results/workstream4
```

- One forward pass over all 187 validation patients, scored under the five A1
  rules and four extra thresholds for A2 (A1 and A2 cost no extra GPU time).
- Checks that the pooled Macro Dice reproduces Workstream 1's **0.8192**
  (within 0.002) and stops with an error if it doesn't.
- Time: about 22 min on an RTX 4080, mostly CPU metric time; about 11 min of
  forward passes plus metrics on a Colab T4.
- Quick checks: `--patients 2` scores the first 2 patients in under a minute;
  `--dry-run` uses a random backbone to test the pipeline without the Meta
  weights.

Output: `<out>/validation187/baseline/per_patient.csv` (one row per rule ×
patient × region; **contains patient IDs, keep private**) and `summary.json`
(settings, global Dice and the provenance fingerprints described below).

Rerunning the same command reuses the finished result
(`[skip] baseline: results exist and match`). If anything changed (weights,
data, manifest, rules, patients) it stops with "stale results" instead of
mixing old and new numbers.

## 5. Make the tables and figures

```bash
python scripts/ws4_analyze.py /path/to/private/results/workstream4/validation187
python scripts/ws4_figures.py /path/to/private/results/workstream4/validation187
```

`ws4_analyze.py` writes CSV + LaTeX tables to `validation187/analysis/`:

| File | Ablation | Content |
|---|---|---|
| `A1_global_dice` | A1 | Pooled Dice per rule |
| `A1_patient_summary` | A1 | Per-patient mean ± std, with a LaTeX table for the report |
| `A1_paired_vs_raw` | A1 | Paired differences vs raw output, bootstrap CI, Holm-corrected Wilcoxon p |
| `A1_empty_mask_cases` | A1 | How often a region is empty in the prediction and/or ground truth |
| `A2_threshold_sweep` | A2 | Dice and predicted/true volume for thresholds 0.3–0.7 |
| `A7_size_strata_dice_*` | A7 | Dice per tumor-size tertile |
| `A7_lesion_detection` | A7 | Lesion detection rate per size bin, false-positive blobs |
| `A7_error_share_by_boundary_distance` | A7 | Share of errors 0–2 / 2–4 / 4–8 / ≥8 mm from the boundary |
| `A7_error_rate_by_token_patch_type` | A7 | Error rate in background / boundary / interior 16×16 token patches |

It also writes `worst_ET_cases_PRIVATE.csv` (patient IDs; private).

`ws4_figures.py` writes the presentation figures `01`–`08` (PNG + PDF) to
`validation187/figures/`. These are aggregate only (no MRI, no patient IDs)
and are the ones in `docs/figures/workstream4/`.

Optional MRI case figure (needs the data and weights again):

```bash
python scripts/ws4_qualitative.py /path/to/private/results/workstream4/validation187
```

It writes `09_qualitative_cases.png`. **It shows patient MRI: fine for class
slides, never commit it or post it publicly.**

## 6. Colab instead of a local GPU

Open `notebooks/04_evaluation.ipynb` and run it top to bottom. It mounts
Drive, checks the artifact's SHA-256, builds the model, and runs the same
evaluation with `QUICK = True` (20 patients, ~2 min). Set `QUICK = False`
for the reported 187-patient numbers.

## 7. Expected results (check yours against these)

| | WT | TC | ET | Macro |
|---|---|---|---|---|
| Pooled Dice | 0.869 | 0.856 | 0.732 | **0.8192** |
| Patient Dice, mean | 0.843 | 0.800 | 0.668 | |
| HD95, median (mm) | 13.0 | 5.8 | 6.4 | |

- **A1:** best rule gives Macro Dice 0.8192 → 0.8196 (too small to claim);
  ET HD95 −2.6 mm; WT false-positive blobs 6.1 → 3.3 per patient.
- **A2:** Macro Dice 0.8184–0.8193 across thresholds 0.3–0.7; ET is
  over-segmented at every threshold (1.41× → 1.32×), so it's a model bias.
- **A7:** 74% of WT errors lie within 2 mm of the boundary; boundary token
  patches err ~480× more than background patches (11.2% vs 0.02%) but are
  only 2.2% of tokens; ET Dice 0.61 in the smallest tumors; 40% of ET lesions
  under 100 voxels are detected.

A Macro Dice difference in the 4th decimal (0.819172 vs 0.819176) comes from
fp16 autocast and is expected.

## 8. Guardrails (the code stops you if…)

| Situation | What happens |
|---|---|
| The test split is requested | `PermissionError` unless `allow_test=True` (final run only) |
| The split is missing, unknown, or disagrees between dataset, wrapper and batches | `ValueError` |
| A patient doesn't have all 155 axial slices, or a listed patient is never evaluated | `RuntimeError` ("incomplete volume" / "evaluated patients differ") |
| Old results were made with other weights, data, manifest, rules or patients | `RuntimeError` ("stale results"); rerun with `overwrite=True` or a new `--out` |
| A results CSV was edited after it was written | `RuntimeError` (checksum mismatch) |
| Variants from different experiments are combined | `RuntimeError` ("different experiments") |
| A paired comparison has different patients per method, or duplicate rows | `ValueError` |

Every `summary.json` records the SHA-256 of the checkpoint files, manifest
and inspection config, and a fingerprint of the data ZIP (size + first and
last 64 MiB). This is what the stale-result checks compare.

## 9. A3–A6: what's needed and how they will run

The pipeline is ready; it needs a model per variant.

**From Workstream 2 (merging):** a function

```python
def build_model(spec: VariantSpec) -> torch.nn.Module:
    """Return a model mapping normalized [B,4,240,240] images to raw [B,3,240,240] logits."""
```

- `spec.method` is `"uniform"`, `"adaptive"` or `"lowres"`; `spec.merge_ratio`
  is the fraction of tokens merged; extra options (merge blocks, boundary
  protection, reconstruction) go in `spec.params`.
- Reduced-resolution models must upsample logits back to 240×240.
- For merge-map figures: the per-sample `[225]` original→merged token
  assignment described in [`TOKEN_MERGING_INTEGRATION.md`](TOKEN_MERGING_INTEGRATION.md).

**From Workstream 3 (efficiency):** measured latency (ms/slice) and peak
memory per variant, on one stated GPU, dtype and batch size, for A5's
equal-latency curves.

**Then (Workstream 4):** in the notebook, fill in `build_model` and set
`RUN_MERGING = True`, or in Python:

```python
from cap6411_atm.evaluation import merge_ratio_grid, run_ablation, paired_comparison
variants = merge_ratio_grid("uniform", [0.1, 0.25, 0.4, 0.5]) + merge_ratio_grid("adaptive", [0.1, 0.25, 0.4, 0.5])
frame = run_ablation([baseline_spec, *variants], build_model, loader, run_dir, provenance=provenance, device="cuda")
paired_comparison(frame[frame["rule"] == "raw"], "baseline")
```

Each variant reuses the same 187 patients, threshold 0.5 and metrics, so all
comparisons are paired per patient. Expect about 10–25 min per variant.

| Ablation | Variants | Main outputs |
|---|---|---|
| A3 | uniform and adaptive at r ∈ {0.1, 0.25, 0.4, 0.5} | Dice / HD95 / surface Dice vs merge ratio |
| A4 | adaptive vs uniform at each r | Paired tests, boundary-band and token-patch error profiles |
| A5 | adaptive vs lowres at matched latency | Dice vs measured ms/slice curves |
| A6 | merge blocks, boundary protection, reconstruction | One variant per option via `spec.params` |
| A7 | all of the above | Size strata and small-lesion recall per variant |

## 10. Privacy rules

- Commit only aggregate tables and figures `01`–`08`.
- Keep `per_patient.csv`, `*_PRIVATE.csv` and figure `09` (MRI) in private
  storage.
- Never commit weights, the dataset, the manifest or patient IDs.
- Run the test split once, at the very end, with the protocol fixed.
