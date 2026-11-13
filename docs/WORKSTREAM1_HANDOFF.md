# Workstream 1 handoff: BraTS DINOv3 segmentation baseline

## Status and scope

Workstream 1 provides a reproducible 2D axial BraTS 2023 Adult Glioma
segmentation baseline built around a frozen official DINOv3 ViT-B/16 encoder.
The data inspection, patient splitting, preprocessing, dataset, model, training,
checkpointing, and validation-metric interfaces are implemented in this
repository. Two small-data pilot experiments have been completed externally in
Google Colab; their fair-comparison results are recorded below.

This is a **pilot handoff**, not a full-cohort or independent-test result. The
validation split was used for checkpoint selection, and the isolated test split
has not been evaluated.

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
batch within one patient, and the bounded memory/disk cache avoids repeatedly
decompressing that patient. Colab's default cache is `/content/brats_cache`.

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

The recommended pilot checkpoint is:

```text
My Drive/CAP6411/checkpoints/dinov3_vitb16_brats_pilot32/best.pt
```

In a mounted Colab Drive, its path is:

```text
/content/drive/MyDrive/CAP6411/checkpoints/dinov3_vitb16_brats_pilot32/best.pt
```

The checkpoint's embedded `training_config`, `model_config`,
`dataset_metadata`, and `history` are the authoritative record for its exact
run configuration and selected patients.

## 5. Load and evaluate the recommended checkpoint

Run the following from the repository root in Colab. It uses only existing
repository APIs and evaluates exactly the validation patient IDs stored in the
checkpoint. Only load a checkpoint you trust because it is deserialized by
PyTorch.

First mount Drive, install the package, and pin the official source checkout:

```python
from google.colab import drive

drive.mount("/content/drive")
```

```bash
%%bash
cd /content/dinov3
git checkout 6876159a11b4df116f30f667f8c9888617df0751

cd /content/CAP6411-Adaptive-Token-Merging
pip install -e .
```

Then load the model and evaluate the checkpoint's stored validation cohort:

```python
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cap6411_atm.data.dataset import BraTSAxialSliceDataset
from cap6411_atm.models import (
    DINOv3SegmentationConfig,
    build_pretrained_brats_dinov3,
)
from cap6411_atm.training.metrics import StreamingDice
from cap6411_atm.training.sampling import PatientLocalBatchSampler, PatientSubset

REPO = Path("/content/CAP6411-Adaptive-Token-Merging")
ARCHIVE = Path(
    "/content/drive/MyDrive/CAP6411/datasets/"
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"
)
MANIFEST = Path("/content/drive/MyDrive/CAP6411/brats2023_gli_split_seed42.json")
PRETRAINED = Path(
    "/content/drive/MyDrive/CAP6411/checkpoints/"
    "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)
TRAINED = Path(
    "/content/drive/MyDrive/CAP6411/checkpoints/"
    "dinov3_vitb16_brats_pilot32/best.pt"
)
DINOV3_REPOSITORY = Path("/content/dinov3")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is a trusted project checkpoint. The fallback supports older PyTorch.
try:
    checkpoint = torch.load(TRAINED, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(TRAINED, map_location="cpu")

assert checkpoint["dataset_metadata"]["test_split_loaded"] is False
validation_ids = tuple(checkpoint["dataset_metadata"]["validation_patients"])
assert len(validation_ids) == 8

model_config = DINOv3SegmentationConfig.from_json(
    REPO / "configs/dinov3_vitb16_brats.json"
)
assert checkpoint["model_config"] == asdict(model_config)

model = build_pretrained_brats_dinov3(
    repository=DINOV3_REPOSITORY,
    checkpoint=PRETRAINED,
    config=model_config,
)
model.load_state_dict(checkpoint["model_state"], strict=True)
model.to(device).eval()

validation_full = BraTSAxialSliceDataset(
    archive_path=ARCHIVE,
    manifest_path=MANIFEST,
    split="validation",
    inspection_config=REPO / "configs/brats2023_gli_inspection.json",
    cache_dir="/content/brats_cache/validation",
    max_cached_patients=1,
)
assert set(validation_ids).issubset(set(validation_full.patient_ids))
validation = PatientSubset(validation_full, validation_ids)
sampler = PatientLocalBatchSampler(
    validation,
    batch_size=2,
    seed=42,
    shuffle_patients=False,
    shuffle_slices=False,
)
loader = DataLoader(
    validation,
    batch_sampler=sampler,
    num_workers=0,
    pin_memory=device.type == "cuda",
)

threshold = float(checkpoint["training_config"]["validation_threshold"])
metric = StreamingDice(threshold=threshold)

with torch.inference_mode():
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
        metric.update(logits, targets)

results = metric.compute()
print(results)
```

For segmentation masks from an individual batch, apply independent sigmoid
thresholds to the raw logits:

```python
probabilities = torch.sigmoid(logits.float())
predicted_wt_tc_et = probabilities >= threshold  # bool [B, 3, 240, 240]
```

The three heads are independent; the implementation does not post-process them
to force `ET ⊆ TC ⊆ WT`.

## 6. Adaptive token-merging integration guidance

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

## 7. Limitations and claims boundary

- Both experiments are small-data pilots, not full 876-patient training runs.
- Validation data was used for best-checkpoint selection.
- The fair comparison reused the same eight validation patients; it is not an
  independent evaluation.
- The isolated 188-patient test split has not been evaluated and must remain
  untouched until the final model and evaluation protocol are fixed.
- The results do not establish full-cohort or external-dataset generalization.
- The repository's `StreamingDice` implementation computes global thresholded
  Dice, not patient-averaged Dice. The pilot table should retain its supplied
  provenance unless it is recomputed from the saved checkpoints with the
  documented evaluation procedure.
- The model uses 2D axial slices and does not model through-plane context.
- Independent WT/TC/ET heads do not guarantee nested predictions.
- Patch-16 features may limit small-lesion and boundary accuracy, especially ET.
- No adaptive token-merging speedup or accuracy claim has yet been established.

Any presentation or report should label the table as a **same-eight-patient
validation comparison of two pilot models** and identify the 32-patient model's
`best.pt` as the recommended pilot checkpoint, not as a final test-qualified
model.
