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
- Git commit: `31265a1` (clean tree, `git_dirty: false`)
- Command or notebook: `python -m cap6411_atm.benchmark.compare --dinov3-repo ~/dinov3 --tag timing_only_random_weights_official`
- Objective / hypothesis: Measure how latency and peak memory scale with encoder token count for the reduced-input-resolution baseline (240 → 128px, 225 → 64 tokens) at batch sizes 1, 8, 32 and 128.
- Dataset access source and approved version: None. Inputs are seeded random `[B,4,240,240]` tensors. Latency of these fixed-token models does not depend on input content.
- Split definition: Not applicable (no data).
- Configuration file: `configs/benchmark_resolution_sweep.json` (sizes 240/224/208/192/176/160/144/128; batch sizes 1/8/32/128; 25 warm-up + 200 timed iterations; fp16 autocast; stage timing on; seed 0)
- Random seed(s): 0
- Model / backbone: Official DINOv3 ViT-B/16 architecture at revision `6876159a`, built with `pretrained=False` (**random weights, valid for timing only**), plus the Workstream 1 adapter and decoder architecture (untrained).
- Token-merging strategy: None. This is the reduced-input-resolution baseline (`ReducedResolutionSegmenter`).
- Training setup: None (inference only).
- Evaluation metrics: Median and p90 latency (CUDA events, per iteration), per-stage median latency (adapter / encoder / decoder), and peak GPU memory (`max_memory_allocated`).
- Compute environment: NVIDIA GeForce RTX 5090 (32 GB), driver 595.99.02, CUDA 12.8, cuDNN 9.19, PyTorch 2.11.0+cu128, Python 3.12.14, Ubuntu 26.04.
- Artifact location (not committed): `area51:~/CAP6411-Adaptive-Token-Merging/outputs/benchmarks/20261116-130815_timing_only_random_weights_official{.csv,_env.json,_table.md}`
- Results (median ms; speedup is measured against 240px at the same batch size):

  | Batch | 240px (225 tok) | 192px (144) | 160px (100) | 128px (64) | Speedup at 128px | Peak MB, 240 → 128 |
  |---|---|---|---|---|---|---|
  | 1 | 4.01 | 4.11 | 4.26 | 4.27 | 0.94× | 383 → 383 |
  | 8 | 5.91 | 5.06 | 5.21 | 5.23 | 1.13× | 679 → 669 |
  | 32 | 20.29 | 16.15 | 13.54 | 12.08 | 1.68× | 1695 → 1651 |
  | 128 | 96.26 | 71.45 | 57.87 | 48.44 | 1.99× | 5741 → 5573 |

  Encoder / decoder median at batch 128: 62.25 / 33.91 ms at 240px; 15.01 / 33.69 ms at 128px.
- Observations:
  - At batch 1 the model takes ~4 ms regardless of token count; the GPU is limited by per-step overhead, not compute. Token reduction gives no latency benefit in this setting.
  - At batch 32–128, encoder time scales with token count as expected (62.3 → 15.0 ms at batch 128).
  - The decoder runs at 240×240 and costs a fixed ~34 ms at batch 128 (~35% of the baseline). That caps end-to-end speedup at about 2.8× even if the encoder took no time.
  - Peak memory is dominated by the full-resolution decoder, so token reduction barely reduces it.
  - The numbers reproduce the earlier same-day runs to within about 1%.
  - Results are specific to the RTX 5090 and do not include token merging.
- Follow-up:
  - Measure validation Dice per resolution with official DINOv3 weights and the 8-epoch adapter/decoder (needs the BraTS ZIP).
  - Benchmark Workstream 2 merging variants on real MRI slices under the same config.
