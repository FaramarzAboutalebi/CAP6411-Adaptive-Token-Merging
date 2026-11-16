# Workstream 4: segmentation evaluation and ablations

`src/cap6411_atm/evaluation/` scores any model that maps `[B,4,240,240]`
images to `[B,3,240,240]` WT/TC/ET logits. Slices are reassembled into
`[3,240,240,155]` patient volumes, and every metric is computed **per
patient**, following BraTS convention. The cohort voxel-count Dice that
Workstream 1's `StreamingDice` reports is computed as well, so the baseline
number (validation Macro Dice 0.8192) can be reproduced.

| Module | Contents |
|---|---|
| `metrics.py` | Overlap, surface-distance, boundary-band, token-patch and lesion metrics |
| `postprocess.py` | Decision rules: threshold, ET ⊆ TC ⊆ WT nesting, small-component and small-total removal |
| `evaluator.py` | `evaluate_model` (one forward pass, many rules), `save_evaluation`, `build_eval_loader`, batch-level `evaluate` |
| `statistics.py` | Summaries, bootstrap CIs, paired Wilcoxon + Holm comparisons, strata, failure cases, report tables |
| `ablation.py` | `VariantSpec`, `merge_ratio_grid`, resumable `run_ablation`, `collect_results` |
| `visualization.py` | Case overlays and error maps, token merge maps, metric-vs-ratio/latency plots, profiles |

The end-to-end Colab workflow is `notebooks/04_evaluation.ipynb`.

## Metrics

All metrics are reported per patient and per region (WT, TC, ET). Spacing is 1 mm isotropic.

| Metric | Definition |
|---|---|
| Dice, sensitivity, precision | Voxel overlap; TP/FP/FN counts and volumes are stored too |
| HD95 | max(95th percentile pred→GT, 95th percentile GT→pred) of surface distances (MONAI convention) |
| ASSD | Mean of all symmetric surface distances |
| NSD@τ | Share of surface voxels within τ mm of the other surface (τ = 1, 2 mm), from Nikolov et al. Uses surface-voxel counts, not surface area |
| Boundary bands | FP/FN voxels by distance from the GT surface: [0,2), [2,4), [4,8), ≥8 mm. Error rates are given for the bounded bands |
| Token patches | Each axial slice is tiled into 16×16 patches (exactly the DINOv3 token footprints). A patch is *background* (no tumor), *boundary* (partly tumor) or *interior* (all tumor). Reports (FP+FN)/pixels for each class |
| Lesions | 26-connected GT components binned by size (<100, 100–1k, 1k–10k, ≥10k voxels): count and detected (any overlap). Also predicted components that touch no GT (false-positive components) |

Surfaces are mask voxels with a 6-connected background neighbour.

**Empty masks.** When both masks are empty: Dice 1, NSD 1, HD95 0. When exactly
one is empty: Dice 0, NSD 0, HD95 = ASSD = 373.13 mm (the volume diagonal,
following BraTS). Sensitivity is NaN for an empty GT and precision is NaN for
an empty prediction. Summaries skip NaNs and report `n_missing`.
`status` records which case applies.

**Why the boundary and patch metrics.** Token merging pools 16×16 patches.
If adaptive merging works as intended, it should change errors in *interior*
and *background* patches but leave *boundary* patches and the 0–2 mm band
nearly unchanged. Uniform merging and lower resolution are expected to
degrade exactly those. These two analyses test the project's
boundary-preservation claim directly.

## Statistics

The patient is the unit of analysis.

- **Summaries:** `summarize` gives the mean, std, median, IQR and a 95%
  percentile-bootstrap CI (10,000 resamples) for each method, region and
  metric.
- **Paired comparisons:** `paired_comparison(frame, reference)` reports, per
  patient, the mean and median difference with a bootstrap CI, counts of
  patients improved and degraded (direction-aware: lower HD95 is better), and
  Wilcoxon signed-rank p-values, Holm-adjusted across the compared methods
  within each metric × region.
- **Tables:** `format_table(..., latex=True)` writes a CVPR-ready
  `mean ± std` table.

## Planned ablations

All variants share the same 187 validation patients, preprocessing, threshold and `MetricConfig`.

| # | Ablation | Variants | Needs |
|---|---|---|---|
| A1 | Decision rule | raw, nesting, small-component removal (50 vox), ET total < 200 vox cleared | Baseline only, available now |
| A2 | Threshold | 0.3 / 0.4 / 0.5 / 0.6 / 0.7 | Baseline only, available now |
| A3 | Merge ratio | uniform and adaptive at r ∈ {0.1, 0.25, 0.4, 0.5} | WS2 |
| A4 | Adaptive vs uniform at equal r | paired comparison, boundary bands, token patches | WS2 |
| A5 | Matched latency | adaptive vs reduced resolution at similar latency | WS2 + WS3 latencies (`plot_metric_vs_x(x_values=...)`) |
| A6 | Merge design | merge location (blocks), boundary-token protection, reconstruction method | WS2 options exposed via `VariantSpec.params` |
| A7 | Tumor size and small lesions | small/medium/large WT strata, ET lesion recall per size bin | Any of the above |

A1 and A2 cost no extra GPU time: `evaluate_model` applies every rule to the same forward pass.

## Running

```python
from cap6411_atm.evaluation import (
    VariantSpec, build_eval_loader, run_ablation, summarize, paired_comparison,
    POSTPROCESSING_ABLATION,
)

loader = build_eval_loader(archive, manifest, "configs/brats2023_gli_inspection.json",
                           split="validation", batch_size=16)
variants = [VariantSpec("baseline", rules=POSTPROCESSING_ABLATION)]
frame = run_ablation(variants, lambda spec: model, loader, "/drive/.../results/ws4", device="cuda")
summary = summarize(frame)
comparison = paired_comparison(frame, reference="baseline")
```

Output per variant goes to `<output>/<variant>/per_patient.csv` (one row per
rule × patient × region) and `summary.json` (metadata, rules, global Dice,
git commit). These files contain patient IDs, so keep them in restricted
storage. Commit only aggregate tables and figures that contain no MRI.

**Cost.** A full validation pass on a T4 takes about 11 minutes of forward
passes plus about 0.5 s per patient per rule for metrics on the CPU.
`forward_seconds` is not a latency benchmark; use Workstream 3's tools for
latency.

## Interface for Workstreams 2 and 3

`build_model(spec)` must return a module or callable mapping normalized
`[B,4,240,240]` images to raw `[B,3,240,240]` logits. Reduced-resolution
models must upsample their logits back to 240×240 so they are scored against
the original-resolution targets.

Merge-state visualization: `visualization.plot_merge_map(image, assignment)`
takes the per-sample, row-major `[225]` original-to-merged assignment from
`docs/TOKEN_MERGING_INTEGRATION.md`.

## Guardrails

- `evaluate_model` refuses the `test` split unless `allow_test=True` is
  passed, and `build_eval_loader` never builds it. Run the test split once, at
  the end, with the protocol fixed.
- The evaluator requires each patient's slices to be contiguous (a
  non-shuffled `PatientLocalBatchSampler`). It raises an error otherwise.
- Run the weight-free tests with
  `python -m unittest tests.test_evaluation_metrics tests.test_evaluation_pipeline`.
