# BraTS axial-slice dataset

`BraTSAxialSliceDataset` converts patient-restricted BraTS 2023 GLI volumes in
the training ZIP into 2D axial PyTorch samples without extracting the full
archive.

## Sample contract

Each item is a dictionary:

```text
image:       float32 [4, H, W]  — T1n, T1c, T2w, T2f
target:      float32 [3, H, W]  — overlapping WT, TC, ET masks
patient_id:  str
slice_index: int
split:       str
```

The target channels are `WT={1,2,3}`, `TC={1,3}`, and `ET={3}`. They overlap and
are intended for three-channel sigmoid/BCE-style segmentation, not a mutually
exclusive softmax target.

No resizing, resampling, cropping, or orientation change is performed. Axial
slices use the third NIfTI dimension and retain the verified 240×240 geometry.

## Normalization

Each MRI modality is converted to float32 and normalized independently using a
z-score over finite, nonzero voxels. Zero background remains zero. NaN and Inf
values are treated as invalid/background. Empty or constant foregrounds become
zeros, avoiding division by zero. This handles the verified mixture of int16 and
float32 stored volumes consistently.

## Usage

```python
from cap6411_atm.data.dataset import BraTSAxialSliceDataset

dataset = BraTSAxialSliceDataset(
    archive_path="/content/drive/MyDrive/BraTS/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip",
    manifest_path="outputs/brats2023_gli_split_seed42.json",
    split="train",
    inspection_config="configs/brats2023_gli_inspection.json",
    cache_dir="/content/brats_cache",
    max_cached_patients=1,
)

sample = dataset[0]
print(sample["image"].shape)   # [4, 240, 240]
print(sample["target"].shape)  # [3, 240, 240]
```

Pass `slice_indices=[40, 41, 42]` to use the same selected axial indices for all
patients, or pass `{patient_id: [indices...]}` for an explicit per-patient
selection. Selection is deterministic and validated for duplicates and bounds.

## ZIP access and caching

On a cache miss, the loader extracts only one patient's five `.nii.gz` members
into a temporary directory, validates shape/affine/labels, and normalizes the
volumes. `max_cached_patients` controls a per-process in-memory LRU cache.

When `cache_dir` is set, normalized images and targets are also stored as `.npy`
files and memory-mapped on reuse. In Colab, keep the source ZIP and manifests on
Google Drive but put the cache under fast ephemeral storage such as
`/content/brats_cache`. A normalized patient cache is substantially larger than
its compressed ZIP members, so populate it on demand rather than caching the
entire cohort. Multiple DataLoader workers have independent in-memory caches;
the disk-cache write uses a temporary directory and atomic rename.

For a full-cohort pass on a storage-constrained Colab runtime, set
`cache_dir=None` (JSON: `"cache_dir": null`). This disables persistent `.npy`
files entirely while preserving the bounded per-process in-memory LRU cache
controlled by `max_cached_patients`. Patient-local sampling still prevents a
patient from being repeatedly decompressed while its slices are consumed. The
tradeoff is that the same patient must be decompressed and normalized again in
a later epoch or a new runtime.

For best locality, avoid fully random slice ordering across the entire cohort.
Sampling nearby slices from the same patient reduces ZIP decompression and cache
thrashing. Start with `num_workers=0` while validating the pipeline, then tune
workers against the Colab runtime and cache location.

The manifest is applied before axial slices are enumerated. Never split a global
slice list: doing so would leak slices from the same patient across partitions.
