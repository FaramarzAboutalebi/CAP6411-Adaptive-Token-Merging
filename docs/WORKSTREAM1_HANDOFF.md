# Workstream 1 handoff: BraTS DINOv3 segmentation baseline

## Status and scope

Workstream 1 provides a reproducible 2D axial BraTS 2023 Adult Glioma
segmentation baseline built around a frozen official DINOv3 ViT-B/16 encoder.
The data inspection, patient splitting, preprocessing, dataset, model, training,
checkpointing, and validation-metric interfaces are implemented in this
repository. Two pilot experiments and two full-cohort epochs have been
completed in Google Colab; their validation results are recorded below.

This is a **validation handoff**, not an independent-test result. Validation
was used for checkpoint selection, and the isolated test split has not been
evaluated.

## 1. Dataset and target contract

- Dataset: BraTS 2023 Adult Glioma (BraTS-GLI) training archive,
  `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip`.
- Verified cohort size: 1,251 patients.
- Patient-level split seed: 42.
- Train: 876 patients (70%).
- Validation: 187 patients (15%).
- Test: 188 patients (15%).
- Expected volume geometry: 240 × 240 × 155 voxels at 1 × 1 × 1 mm.
- BraTS labels: background 0 and tumor labels 1, 2, and 3.

The loader emits one 2D axial slice at a time:

```text
image  : float32 [4, 240, 240] in T1n, T1c, T2w, T2f order
target : float32 [3, 240, 240] in WT, TC, ET order
```

The explicit archive-to-loader modality mapping must be preserved:

| Loader role | BraTS 2023 modality | Filename suffix |
|---|---|---|
| `t1` | T1n | `-t1n.nii.gz` |
| `t1ce` | T1c | `-t1c.nii.gz` |
| `t2` | T2w | `-t2w.nii.gz` |
| `flair` | T2f | `-t2f.nii.gz` |
| `segmentation` | Label mask | `-seg.nii.gz` |

Targets are overlapping binary regions, not mutually exclusive classes:

- Whole tumor (WT): labels 1, 2, or 3.
- Tumor core (TC): labels 1 or 3.
- Enhancing tumor (ET): label 3.

`BraTSAxialSliceDataset` applies the patient manifest before enumerating slices,
so slices from one patient cannot cross split boundaries. Training code creates
only train and validation datasets; it does not construct the test dataset.

## 2. Exact DINOv3 model

The official DINOv3 source checkout is pinned to:

```text
6876159a11b4df116f30f667f8c9888617df0751
```

The required official checkpoint is:

```text
dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

The implementation verifies the Git revision and checkpoint filename and does
not silently fall back to random weights.

| Component | Configuration |
|---|---|
| Backbone entry point | `dinov3_vitb16` |
| Input | `[B, 4, 240, 240]` |
| MRI adapter | Biased trainable `Conv2d(4, 3, kernel_size=1)` |
| Patch size | 16 × 16 |
| Patch grid | 15 × 15 = 225 spatial tokens |
| Embedding dimension | 768 |
| Transformer blocks | 12 |
| Special tokens | One class token and four storage tokens |
| Feature taps | Zero-based blocks `[2, 5, 8, 11]` |
| Dense features | Four normalized maps, each `[B, 768, 15, 15]` |
| Lateral width | 128 channels per feature map |
| Decoder widths | `[128, 96, 64, 32]` |
| Decoder scaling | Four 2× bilinear/refinement stages |
| Output | Raw logits `[B, 3, 240, 240]` for WT, TC, ET |

Backbone parameters have `requires_grad=False` and remain in evaluation mode.
The encoder is deliberately not wrapped in `no_grad()` during training because
the loss gradient must pass through its frozen operations to reach the MRI
adapter. Only the adapter and decoder are optimized.

## 3. Reproducible training pipeline

### Preprocessing and loading

Each MRI modality is converted to float32 and independently normalized with a
z-score computed over finite, nonzero voxels. Background remains zero.
Non-finite values are treated as invalid/background. Empty foreground or
near-zero foreground variance produces an all-zero modality rather than NaN or
Inf. The loader validates shape, spacing, affine alignment, file completeness,
and segmentation labels.

Patients are read individually from the ZIP. Only one patient's files are
temporarily extracted at a time. The patient-local batch sampler keeps every
batch within one patient. Pilot runs may use a bounded disk cache; full-cohort
training uses `cache_dir: null` to prevent normalized patient arrays from
exhausting Colab storage while retaining the one-patient in-memory LRU.

### Optimization defaults

The committed baseline configuration is
`configs/brats2023_gli_baseline_training.json`:

| Setting | Value |
|---|---:|
| Random seed | 42 |
| Physical batch size | 2 |
| Mixed precision | CUDA AMP enabled |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Gradient clipping | L2 norm 1.0 |
| Gradient accumulation | 1 step |
| BCE weight | 0.5 |
| Soft-Dice weight | 0.5 |
| Dice smoothing | 1e-6 |
| Validation threshold | Sigmoid probability 0.5 |

The loss is multi-label BCE-with-logits plus channel-balanced soft Dice. No
sigmoid or softmax is applied inside the model.

### Pilot selection and metrics

Pilot patients are selected deterministically and as complete patients from the
existing train and validation splits. Training uses seed 42; validation uses
seed 43. Pilot histories are explicitly marked `scope: pilot`,
`pilot_metrics: true`, and
`metrics_scope: deterministic_patient_subset`.

Validation Dice is accumulated globally over all evaluated slices before it is
computed separately for WT, TC, and ET. Empty slices therefore do not receive
automatic perfect scores. If an entire evaluated channel has neither predicted
nor target voxels, Dice is defined as 1.0 and its `all_empty` flag is true.
Macro Dice is the unweighted mean of WT, TC, and ET Dice.

### Checkpoints and resume

Every epoch atomically writes `last.pt`, conditionally updates `best.pt` using
validation Macro Dice, and writes `history.json`. A checkpoint contains:

- model state, including adapter, backbone, and decoder;
- AdamW optimizer state and AMP scaler state;
- training and model configurations;
- selected patient IDs and dataset metadata;
- epoch history and best validation Macro Dice;
- Python, NumPy, PyTorch, and CUDA RNG state; and
- environment metadata and Git commit when available.

Resume validates the seed, pilot mode/counts, archive, and manifest. It refuses
a checkpoint whose metadata says the isolated test split was loaded.

## 4. Completed pilot experiments

Two pilot models were trained with the Workstream 1 pipeline:

| Pilot | Training patients | Validation patients during pilot | Status |
|---|---:|---:|---|
| Initial pilot | 8 | 4 | Completed |
| Expanded pilot | 32 | 8 | Completed |

For the fair comparison below, both trained models were evaluated on the same
eight validation patients. These comparison numbers are validation results,
not independent test results. They are the completed-pilot results supplied for
this handoff; they were not recomputed locally because the Google Drive
checkpoints and experiment outputs are intentionally outside this repository.

| Metric | 8-patient model | 32-patient model |
|---|---:|---:|
| WT Dice | 0.8514 | 0.8679 |
| TC Dice | 0.7339 | 0.7529 |
| ET Dice | 0.5730 | 0.6953 |
| Macro Dice | 0.7194 | 0.7720 |

The expanded pilot improved all three regions on the common validation cohort,
with the largest absolute gain in ET Dice (+0.1223). On the ET-negative
validation patient, false-positive ET pixels decreased from 52,011 to 17,878:

```text
(52,011 - 17,878) / 52,011 = 0.6563 ≈ 65.6% reduction
```

## 5. Completed full-cohort baseline

The full-cohort run used the official frozen DINOv3 ViT-B/16 backbone, trainable
MRI adapter and decoder, batch size 2, AdamW, CUDA AMP, and persistent disk
caching disabled. It trained on all 876 training patients and validated on all
187 validation patients. Two epochs completed.

| Validation metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| WT Dice | 0.84847853 | 0.86898142 |
| TC Dice | 0.81595193 | 0.85647742 |
| ET Dice | 0.71793982 | 0.73206816 |
| Macro Dice | 0.79412342 | 0.81917567 |

These are global thresholded **validation** Dice values from the repository's
`StreamingDice` implementation. They are not test results. The 188-patient test
split remains isolated and unevaluated.

Trained checkpoints are not distributed through GitHub. An authorized teammate
may receive a checkpoint through the team's approved private storage, subject
to the DINOv3 and dataset access terms. The checkpoint's embedded
`training_config`, `model_config`, `dataset_metadata`, and `history` remain the
authoritative record for its run.

## 6. Reproducible setup, checkpoint loading, and validation

Create a Python 3.10+ environment, install a platform-appropriate PyTorch build,
then install this repository:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

Clone the official DINOv3 source and pin the audited revision:

```bash
git clone https://github.com/facebookresearch/dinov3.git /path/to/dinov3
git -C /path/to/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
python -m pip install -e /path/to/dinov3 --no-deps
python -m pip install ftfy omegaconf regex scikit-learn submitit termcolor torchmetrics
```

Request the gated ViT-B/16 LVD-1689M checkpoint through Meta's official DINOv3
access process. The required filename is
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`. Do not share download URLs,
tokens, or weight files. Obtain BraTS 2023 GLI through its authorized channel
and keep the ZIP and patient manifest outside the repository.

The sanitized notebook
`notebooks/CAP6411_Workstream1_Setup.ipynb` provides executable, guarded cells
for configuring local private paths, loading the dataset and authorized
weights, running validation, and resuming training. It contains no saved output
or private path.

To load an authorized trained checkpoint, construct the official backbone with
`build_pretrained_brats_dinov3(...)`, then restore the strict model state:

```python
import torch
from cap6411_atm.models import (
    DINOv3SegmentationConfig,
    build_pretrained_brats_dinov3,
)

config = DINOv3SegmentationConfig.from_json("configs/dinov3_vitb16_brats.json")
model = build_pretrained_brats_dinov3(
    repository="/path/to/dinov3",
    checkpoint="/private/path/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    config=config,
)
checkpoint = torch.load(
    "/private/path/trained_checkpoint.pt",
    map_location="cpu",
    weights_only=False,
)
assert checkpoint["dataset_metadata"]["test_split_loaded"] is False
model.load_state_dict(checkpoint["model_state"], strict=True)
model.eval()
```

Use `BraTSAxialSliceDataset` with `split="validation"`, the checkpoint's stored
`validation_patients`, `PatientSubset`, `PatientLocalBatchSampler`, and
`StreamingDice` to reproduce validation. Use `cache_dir=None` for full-cohort
work. Apply sigmoid only for independent WT/TC/ET probabilities or thresholded
masks; the model returns raw logits.

To resume without repeating completed epochs, set the runtime configuration's
`epochs` to the desired total and invoke:

```bash
python -m cap6411_atm.training.baseline \
  --config /private/runtime/full_cohort_resume.json \
  --resume /private/checkpoints/full_cohort/last.pt
```

`resume_from_checkpoint` returns `checkpoint["epoch"] + 1`, restores model,
optimizer, AMP scaler, history, and Python/NumPy/PyTorch RNG state, and therefore
starts after the last completed zero-based epoch. The runtime config must keep
the same seed, pilot settings, manifest, and archive. Never edit or recommit the
private checkpoint.

## 7. Adaptive token-merging integration guidance

There is currently no merger or reconstructor implementation:
`src/cap6411_atm/merging/__init__.py` is a placeholder. Do not invent an
existing callable interface or claim acceleration yet.

The baseline deliberately exposes two model boundaries:

```python
features = model.extract_features(images)
logits = model.decode_features(features)
```

`extract_features` returns four dense `[B, 768, 15, 15]` maps from blocks
2, 5, 8, and 11. `decode_features` requires four maps with exactly those dense
shapes. A future token-merging path must therefore reconstruct all 225 spatial
positions in original row-major order before calling the current decoder.

Recommended minimum future contract:

```text
merged, state = merger(patch_tokens [B, 225, 768], grid_size=(15, 15))
dense = reconstructor(merged [B, M, 768], state)  # [B, 225, 768]
```

The state must be per sample and retain the original grid, original-to-merged
assignments, aggregation weights/token sizes, and spatial coordinates. Protect
the one class token and four storage tokens from spatial merging; derive the
special-token offset from the backbone rather than hard-coding five.

Required integration checks include:

- zero merge ratio reproduces the unmerged baseline numerically;
- reconstruction restores 225 tokens in row-major order;
- batch elements, dtype, device, and embedding dimension remain independent;
- gradients follow the intended path;
- every feature tap has defined merge/reconstruction semantics; and
- comparisons use the same preprocessing, patient IDs, threshold, and metric.

Post-encoder merging occurs after all 12 transformer blocks and therefore
cannot accelerate the DINOv3 encoder. True encoder acceleration requires
inter-block merging, a block-level wrapper, and a defensible treatment of
DINOv3 rotary spatial positions for merged regions. The unmodified
`get_intermediate_layers` path does not provide that functionality.

## 8. Limitations and claims boundary

- Two full-cohort epochs completed, but this remains an early baseline rather
  than a converged training study.
- Validation data was used for checkpoint selection and model-development
  comparisons.
- The fair comparison reused the same eight validation patients; it is not an
  independent evaluation.
- The isolated 188-patient test split has not been evaluated and must remain
  untouched until the final model and evaluation protocol are fixed.
- The results do not establish independent-test or external-dataset
  generalization.
- The repository's `StreamingDice` implementation computes global thresholded
  Dice, not patient-averaged Dice. The pilot table should retain its supplied
  provenance unless it is recomputed from the saved checkpoints with the
  documented evaluation procedure.
- The model uses 2D axial slices and does not model through-plane context.
- Independent WT/TC/ET heads do not guarantee nested predictions.
- Patch-16 features may limit small-lesion and boundary accuracy, especially ET.
- No adaptive token-merging speedup or accuracy claim has yet been established.

Any presentation or report must label the pilot comparison as a
**same-eight-patient validation comparison** and the full-cohort table as
**187-patient validation results after two epochs**. Neither is a final
test-qualified result.
