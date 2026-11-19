# Adaptive token-merging integration guide

## Workstream 1 contract

Workstream 1 has completed patient-level BraTS splitting, ZIP-backed loading,
MRI normalization, WT/TC/ET target conversion, a frozen DINOv3 ViT-B/16
segmentation baseline, training/checkpoint resume, and global validation Dice.

Use these source files:

- `src/cap6411_atm/data/dataset.py` — `[B,4,240,240]` MRI inputs and
  `[B,3,240,240]` WT/TC/ET targets.
- `src/cap6411_atm/models/dinov3_segmentation.py` — adapter, encoder wrapper,
  decoder, and stable feature boundary.
- `src/cap6411_atm/training/metrics.py` — global thresholded WT/TC/ET Dice.
- `configs/dinov3_vitb16_brats.json` — audited backbone/decoder dimensions.
- `tests/test_dinov3_segmentation.py` — weight-free synthetic backbone fixture.
- `docs/WORKSTREAM1_ARTIFACTS.md` — restricted trained-weight artifact,
  checksum, loading procedure, and access boundary.

The stable model calls are:

```python
features = model.extract_features(images)
logits = model.decode_features(features)
```

Contracts:

```text
images:   float [B,4,240,240] in T1n, T1c, T2w, T2f order
features: four float maps [B,768,15,15] from blocks 2, 5, 8, 11
logits:   float [B,3,240,240] in WT, TC, ET order, before sigmoid
```

Do not change preprocessing, patient splits, channel ordering, validation
threshold, or metric aggregation when comparing merging methods.

## Special tokens and spatial order

DINOv3 constructs the transformer sequence as one class token, four storage
tokens, then 225 row-major spatial patch tokens. Class and storage tokens are
protected non-spatial tokens and must not enter a spatial clustering algorithm.
Derive the protected-token count from the backbone rather than hard-coding it.

The current `extract_features` API returns patch-only dense maps, so teammates
can prototype post-encoder merging on:

```python
tokens = feature.flatten(2).transpose(1, 2)  # [B,225,768]
```

Before calling `decode_features`, reconstruct every original position and
reshape back to `[B,768,15,15]` in the original row-major order.

Minimum proposed standalone contract:

```python
merged, state = merger(tokens, grid_size=(15, 15))
dense_tokens = reconstructor(merged, state)  # [B,225,768]
```

`state` must be per sample and retain the original grid, original-to-merged
assignment, aggregation weights/token sizes, and coordinates used by merging.

The standalone API now has two methods:

- `method="uniform"` uses balanced groups in original row-major spatial order.
  If `merge_size` is omitted, `retention_ratio` is converted to an exact retained
  token count with `round(N * retention_ratio)`. The legacy `merge_ratio` name is
  accepted as an alias. When an explicit rectangular `merge_size` is used,
  achieved counts are still recorded but no ratio target is implied.
  When `merge_size` is provided explicitly, ratio-based requested counts are
  reported as `None`; the achieved count and ratio remain available.
- `method="adaptive"` clusters tokens independently per sample by descending
  pairwise cosine similarity, merging the most similar groups until the retained
  count is `round(N * retention_ratio)`. This can group non-adjacent positions.
  `merge_size` is not valid for this method.

The adaptive assignment is discrete and based on detached token content; merged
feature averages remain differentiable with respect to the input tokens. This is
a content-adaptive post-encoder prototype, not a tumor-boundary-aware algorithm.
`merge_coordinates` records original positions and `merged_coordinates` records
each merged group's mean grid coordinate.

The state reports an observed, serializable retention summary:

```python
merged, state = merge_tokens(tokens, grid_size=(15, 15), retention_ratio=0.75)
print(state.retention_metrics)
# Includes requested_token_count, achieved_token_count,
# requested_retention_ratio, achieved_retention_ratio, and ratio error.
```

## Opt-in inter-block DINOv3 integration

`BraTSDINOv3Segmenter` can opt in to blockwise merging without changing the
default baseline path:

```python
model = BraTSDINOv3Segmenter(
    backbone,
    merge_method="adaptive",
    retention_ratio=0.5,
    merge_after_block=2,
)
logits = model(images)
```

Block indices are zero-based. With `merge_after_block=2`, the model emits the
block-2 tap before merging, merges only patch tokens between blocks 2 and 3,
and runs blocks 3–11 with the shorter patch sequence. CLS and storage tokens
remain in their original prefix and are not passed to the merger. Each of the
four taps is returned as a dense `[B,768,15,15]` map; taps after the merge are
reconstructed with the retained assignment state before normalization and
decoding.

For compressed sequences, RoPE is evaluated at each merged group's mean
row/column coordinate using the pinned DINOv3 coordinate normalization and
periods. Batch-specific coordinates are represented in the RoPE tensor's batch
dimension. A zero merge ratio follows the blockwise path without shortening the
sequence and is tested for exact equivalence with the public dense-feature path.
This is a research prototype: its mean-coordinate RoPE treatment and clustering
quality still require evaluation with authorized official weights. It has not
yet demonstrated a latency, memory, or segmentation-quality improvement.

For a fixed image batch, use the shared inference benchmark:

```python
from cap6411_atm.benchmark import benchmark_inference

timing = benchmark_inference(model, images, warmup=10, iterations=50)
```

For validation DataLoader batches (`{"image": ..., "target": ...}`), use the
shared global WT/TC/ET Dice evaluator:

```python
from cap6411_atm.evaluation import evaluate_segmentation

metrics = evaluate_segmentation(model, validation_loader, device=device)
```

The benchmark reports latency samples, mean/median/p95 latency, throughput,
and peak allocated CUDA memory when run on CUDA. The evaluator preserves the
repository's sigmoid threshold and global voxel-count Dice protocol. For fair
comparisons, use the same hardware, input batches, precision, warm-up, patient
split, and threshold for every model variant.

Both helpers include an `experiment_metadata` entry for this model. It records
the configured merge method and ratio, the zero-based block after which merging
occurs and the first block receiving merged tokens, protected special-token
count, requested and achieved patch-token counts, full sequence-token counts,
and per-block sequence-token counts. Counts come from the merge state produced
by the measured/evaluated forward pass, so uniform merging reports its achieved
count rather than implying that the requested ratio was met. Baseline models
report `None` for merge settings and the unchanged sequence length for every
block. Models without `forward_with_metadata` report `experiment_metadata: None`.

## Required tests before baseline integration

1. Merge ratio zero is numerically equivalent to the unmerged path.
2. Reconstruction returns exactly 225 tokens in original row-major order.
3. Batch elements never share assignments or reconstruction state.
4. Device, dtype, embedding dimension, and gradients are preserved.
5. Class and storage tokens never enter spatial merging.
6. All four feature taps have explicit, consistent reconstruction semantics.
7. `decode_features(reconstructed)` returns `[B,3,240,240]` raw logits.
8. Baseline and merging experiments use identical patients and metrics.

Run the existing weight-free interface suite with:

```bash
python -m unittest tests.test_dinov3_segmentation -v
python -m unittest discover -s tests -v
```

`SyntheticDINOv3ViTB16` in `tests/test_dinov3_segmentation.py` implements the
expected intermediate-feature and special-token interface. It allows merger,
reconstructor, decoder, shape, and gradient development without the BraTS ZIP,
official gated weights, or a trained checkpoint.

When trained comparisons are needed, construct the official backbone from each
teammate's independently authorized checkpoint, then use
`load_adapter_decoder_artifact(...)` with the restricted Workstream 1 artifact.
The loader does not replace or modify encoder weights. Never add either the
official checkpoint or the trained artifact to Git.

## Acceleration claims boundary

The opt-in blockwise path reduces the sequence length for later transformer
blocks, unlike post-encoder merging. However, token-count reduction alone does
not prove an end-to-end speedup; measure warm-up-adjusted latency, peak GPU
memory, and segmentation quality on the same hardware and data protocol before
making efficiency claims. The 188-patient test split remains isolated and
unevaluated.
