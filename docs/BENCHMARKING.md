# Workstream 3: Benchmarking and the reduced-resolution baseline

Code: `src/cap6411_atm/benchmark/`. Driver notebook: `notebooks/03_efficiency.ipynb`.
Tests: `python -m unittest discover -s tests -p 'test_benchmark.py' -v` from the repo root,
after `pip install -e . --no-deps` (CPU, no weights or data needed). The tests reuse the
synthetic backbone from `tests/test_dinov3_segmentation.py`, so run them with `discover -s tests`.

## What is measured

- **Latency:** per-iteration wall clock after warm-up. CUDA events on GPU,
  `perf_counter` on CPU. Reports median, p10, p90, mean, std, min, and throughput.
- **Peak GPU memory:** `torch.cuda.max_memory_allocated` over the timed loop,
  including resident weights. `activation_memory_mb` subtracts memory that was
  already allocated before the loop.
- **Stage split** (`stages=True`): adapter / encoder / decoder timed separately.
  The decoder always upsamples to 240×240, so it is a fixed floor. Lower
  resolution and token merging can only shrink the encoder share.
- **Quality:** `evaluate_quality` uses Workstream 1's validation split, sampler,
  and `StreamingDice` (global voxel-count WT/TC/ET Dice, threshold 0.5).
  `evaluate_quality_many` scores every variant in one pass over the data. Each
  Dice value is saved with its scope (pilot vs full validation split) and its
  patient and slice counts.

`tokens_theoretical` is a token count, not a measurement. Never report it as a speedup.

## What makes two results comparable

These rules live in `benchmark/comparability.py`. `match_latency`, `results_table`
and `plot_tradeoff` all enforce them, so rows combined from different CSVs are
never mixed silently.

- **Latency:** every row is stamped with `gpu`, `nvidia_driver`, `cuda_runtime` and
  `torch`. Rows are only matched, and only given a speedup, against rows with the
  same GPU, driver, CUDA, PyTorch, batch size, device, precision and input kind.
  The plot refuses rows from different GPUs.
- **Dice:** each Dice result records `checkpoint_fingerprint` (a SHA-256 of all
  trained weights, so the backbone, adapter and decoder all count) and `data_fingerprint` (a SHA-256
  of the exact patient/slice list), plus the threshold. `attach_quality` copies these
  into the `dice_checkpoint`, `dice_data` and `dice_threshold` columns. A Dice delta is only reported when all
  three match. The tables and plot raise an error if their Dice values come from
  different checkpoints or validation slices.
- **Peak memory:** before each variant is measured, all other variants are moved
  off the GPU and the allocator cache is cleared. Any memory still held by something
  else is recorded as `other_gpu_memory_mb`, with a warning when it is non-trivial.
  `peak_memory_variant_mb` excludes it; `model_memory_mb` is the variant's own weights.

## Reduced-input-resolution baseline

`ReducedResolutionSegmenter(model, r)` keeps the shared
`[B,4,240,240] -> [B,3,240,240]` contract:

1. Antialiased bilinear resize of the input to r×r (r a multiple of 16).
2. Trained adapter, then frozen DINOv3 at r×r: (r/16)² patch tokens.
   Rotary positions handle any grid size.
3. Bilinear resize of the four feature maps to 15×15.
4. The trained Workstream 1 decoder, unchanged.

No retraining is needed. At r = 240 the wrapper is bit-identical to the baseline.
The default sweep is 240/224/208/192/176/160/144/128, giving 225 down to 64 tokens.

## Benchmark protocol

- Every compared variant runs on **one GPU** (the team benchmark machine, an RTX 5090).
  Colab runs are smoke tests only.
- Use fp16 autocast, matching Workstream 1 inference, at batch sizes 1, 8, 32 and 128 (team decision, Nov 16).
  Use 25 warm-up and 200 timed iterations.
- Each run writes `outputs/benchmarks/<stamp>_<tag>.csv` plus `<stamp>_<tag>_env.json`.
  The JSON records GPU, driver, CUDA, torch, git commit, and whether the tree had
  uncommitted changes, plus which weights were used.
- `random (timing only)` weights are valid for latency, never for Dice.
- Benchmark content-adaptive mergers on real MRI slices
  (`run_suite(..., inputs=slices)`), because their cost may depend on content.
- The isolated test split is never evaluated here.

## Settings and report tables

All sweep settings live in `configs/benchmark_resolution_sweep.json`:
resolutions, batch sizes, warm-up and timed iterations, fp16, stage timing, and seed.
The command-line tool and the notebook both read this file, and explicit command-line
options override it. Each run records the settings it actually used in the env JSON.

`results_table(rows, fmt="markdown"|"latex")` produces one line per variant:
theoretical tokens, measured median/p90 latency, speedup vs the full-resolution
baseline under the same conditions, peak memory, and (once attached) WT/TC/ET
and macro Dice. `matched_table(match_latency(rows))` produces the
latency-matched merger-vs-reference comparison. The command-line tool also writes
`<stamp>_<tag>_table.md`.

## Command line

```bash
pip install -r requirements.txt && pip install -e . --no-deps
pip install -e /path/to/dinov3 --no-deps
pip install ftfy omegaconf regex scikit-learn submitit termcolor torchmetrics   # DINOv3 hub imports

# Latency before gated access is granted (official architecture, random weights):
python -m cap6411_atm.benchmark.compare --dinov3-repo /path/to/dinov3

# With authorized weights and the restricted Workstream 1 artifact:
python -m cap6411_atm.benchmark.compare --dinov3-repo /path/to/dinov3 \
    --checkpoint /path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    --artifact /path/to/dinov3_vitb16_brats_epoch2_adapter_decoder.pt

# Quick check overriding the config: python -m ... --sizes 240 160 --iters 20 --no-stages
```

On an RTX 50-series (Blackwell) GPU, install a PyTorch build for CUDA 12.8 or
newer. Older wheels fail with "no kernel image is available".

## Adding Workstream 2 variants

Hand over an `nn.Module` with the same input/output contract and, optionally,
these attributes:

```python
merged.benchmark_family = "adaptive_merge"   # or "uniform_merge"
merged.benchmark_setting = "keep=0.5"
merged.token_count = 113                     # spatial tokens after merging (theoretical)
```

A variant without `benchmark_family` is reported as `unlabeled`. It is still
compared, never used as the baseline reference. Stage timing (`stages=True`) is
only recorded for models whose forward path is known exactly. A merger can opt
in by defining `benchmark_stage_functions()`, returning
`(("adapter", f1), ("encoder", f2), ("decoder", f3))`, where calling f1, then
f2, then f3 in sequence is equivalent to calling `forward`.

Then add it to the `variants` dict in the notebook. `match_latency(rows)` pairs
each merger with the closest-latency reduced-resolution (or full-resolution)
run, and `plot_tradeoff(rows, path)` draws Dice against measured latency.
A merger applied only after the encoder cannot reduce encoder time; the stage
split will show that.
