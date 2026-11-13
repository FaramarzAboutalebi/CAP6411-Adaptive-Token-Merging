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

Post-encoder merging happens after all 12 transformer blocks. It may test
compression/reconstruction or reduce decoder work, but it cannot reduce DINOv3
encoder computation and must not be reported as encoder acceleration.

Actual encoder acceleration requires inter-block merging. That work needs a
block-level DINOv3 wrapper, protected special-token handling, and an explicit
treatment of rotary spatial coordinates for merged regions. The existing
`get_intermediate_layers` wrapper does not implement this.

`src/cap6411_atm/merging/` remains a placeholder. No adaptive merger,
reconstructor, speedup, latency improvement, or quality result is currently
claimed. The 188-patient test split also remains isolated and unevaluated.
