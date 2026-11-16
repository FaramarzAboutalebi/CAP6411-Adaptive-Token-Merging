# Experiment Tracking

Copy this template for each planned or completed run. Use `Not run` until execution actually occurs.

```markdown
## Experiment: <short name>

- Status: Planned | Running | Completed | Failed
- Date:
- Owner:
- Git commit:
- Command or notebook:
- Objective / hypothesis:
- Dataset access source and approved version:
- Split definition:
- Configuration file:
- Random seed(s):
- Model / backbone:
- Token-merging strategy:
- Training setup:
- Evaluation metrics:
- Compute environment (GPU, OS, Python, key package versions):
- Artifact location (not committed):
- Results: Not run
- Observations:
- Follow-up:
```

## Experiment: WS3 reduced-resolution latency sweep (timing only)

- Status: Completed
- Date: 2026-11-16
- Owner: Jordan Becci (Workstream 3)
- Git commit: `eb199d2` (clean tree, `git_dirty: false`). This supersedes the earlier same-day run at `31265a1`. Latency is unchanged; this run adds environment stamping and isolated per-variant memory accounting after the PR #1 review.
- Command or notebook: `python -m cap6411_atm.benchmark.compare --dinov3-repo ~/dinov3 --tag timing_only_random_weights_official`
- Objective / hypothesis: Measure how latency and peak memory scale with encoder token count for the reduced-input-resolution baseline (240 → 128px, 225 → 64 tokens) at batch sizes 1, 8, 32 and 128.
- Dataset access source and approved version: None. Inputs are seeded random `[B,4,240,240]` tensors. Latency of these fixed-token models does not depend on input content.
- Split definition: Not applicable (no data).
- Configuration file: `configs/benchmark_resolution_sweep.json` (sizes 240/224/208/192/176/160/144/128; batch sizes 1/8/32/128; 25 warm-up + 200 timed iterations; fp16 autocast; stage timing on; seed 0)
- Random seed(s): 0
- Model / backbone: Official DINOv3 ViT-B/16 architecture at revision `6876159a`, built with `pretrained=False` (**random weights, valid for timing only**), plus the Workstream 1 adapter and decoder architecture (untrained). The model holds 331.9 MB of fp32 weights on the GPU.
- Token-merging strategy: None. This is the reduced-input-resolution baseline (`ReducedResolutionSegmenter`).
- Training setup: None (inference only).
- Evaluation metrics: Median and p90 latency (CUDA events, per iteration); per-stage median latency (adapter / encoder / decoder); peak GPU memory (`max_memory_allocated`). Peak memory is also reported excluding other allocations (`peak_memory_variant_mb`); here that is ~10 MB of cuBLAS workspace.
- Compute environment: NVIDIA GeForce RTX 5090 (32 GB), driver 595.99.02, CUDA 12.8, cuDNN 9.19, PyTorch 2.11.0+cu128, Python 3.12.14, Ubuntu 26.04.
- Artifacts (committed): `docs/results/ws3/20261116-173828_timing_only_random_weights_official.csv`, `..._env.json`, and `..._table.md`
- Results (median ms; speedup is measured against 240px at the same batch size):

  | Batch | 240px (225 tok) | 192px (144) | 160px (100) | 128px (64) | Speedup at 128px | Peak MB, 240 → 128 |
  |---|---|---|---|---|---|---|
  | 1 | 4.14 | 4.41 | 4.38 | 4.39 | 0.94× | 384 → 383 |
  | 8 | 5.95 | 5.39 | 5.34 | 5.21 | 1.14× | 679 → 669 |
  | 32 | 20.36 | 16.14 | 13.54 | 12.12 | 1.68× | 1694 → 1652 |
  | 128 | 96.44 | 71.49 | 57.83 | 48.51 | 1.99× | 5742 → 5574 |

  Encoder / decoder median at batch 128: 62.42 / 33.94 ms at 240px; 15.09 / 33.70 ms at 128px.
- Observations:
  - At batch 1 the model takes ~4 ms regardless of token count; the GPU is limited by per-step overhead, not compute. Token reduction gives no latency benefit in this setting.
  - At batch 32–128, encoder time scales with token count as expected (62.4 → 15.1 ms at batch 128).
  - The decoder runs at 240×240 and costs a fixed ~34 ms at batch 128 (~35% of the baseline). That caps end-to-end speedup at about 2.8× even if the encoder took no time.
  - Peak memory is dominated by the weights (332 MB) and the full-resolution decoder, so token reduction barely reduces it.
  - Results reproduce the earlier same-day runs to within ~1–2%.
  - Results are specific to the RTX 5090 and do not include token merging.
- Follow-up:
  - Measure validation Dice per resolution with official DINOv3 weights and the 8-epoch adapter/decoder (needs the BraTS ZIP).
  - Benchmark Workstream 2 merging variants on real MRI slices under the same config.
