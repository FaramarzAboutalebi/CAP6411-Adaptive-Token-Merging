# BraTS ZIP inspection

`cap6411_atm.data.inspection` inventories a BraTS-style ZIP without extracting the
full dataset. Its fast default reads filenames only. Header inspection, full-volume
finite-value checks, and segmentation-label scans are separate opt-in operations.

## BraTS 2023 GLI configuration

The checked-in `configs/brats2023_gli_inspection.json` declares the five filename
suffixes, expected `240 × 240 × 155` shape, 1 mm isotropic spacing, and allowed
segmentation labels `0, 1, 2, 3`. These are explicit configuration values rather
than universal assumptions in the Python module.

## Commands

Fast structural inventory for all patients:

```bash
python -m cap6411_atm.data.inspection \
  /path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip \
  --config configs/brats2023_gli_inspection.json \
  --json outputs/brats2023_gli_structure.json \
  --csv outputs/brats2023_gli_structure.csv
```

Inspect headers and alignment for one patient without scanning voxel values:

```bash
python -m cap6411_atm.data.inspection \
  /path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip \
  --config configs/brats2023_gli_inspection.json \
  --patient BraTS-GLI-00324-000 \
  --inspect-headers \
  --json outputs/BraTS-GLI-00324-000-headers.json
```

Perform the expensive finite-value and segmentation-label scans for that patient:

```bash
python -m cap6411_atm.data.inspection \
  /path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip \
  --config configs/brats2023_gli_inspection.json \
  --patient BraTS-GLI-00324-000 \
  --scan-values --scan-labels \
  --json outputs/BraTS-GLI-00324-000-full.json
```

Omit `--patient` to scan every patient sequentially. This remains bounded to one
patient at a time, but it is I/O intensive. Run the structural inventory first and
use targeted patient scans before a cohort-wide full-volume scan.

## Report semantics

- `uninspected` means no NIfTI header or voxel data was read; it is not a passing result.
- `header_inspected`, `values_scanned`, and `labels_scanned` record the completed depth.
- Missing and duplicate files make a patient structurally incomplete.
- Header, expected-shape/spacing, affine-alignment, non-finite, and label problems set
  the patient status to `issues_found` while allowing the remaining patients to run.
- JSON contains complete per-volume metadata. CSV is a compact patient-level summary.

The command returns nonzero when structural incompleteness or inspected issues are
found. Malformed individual NIfTI files are captured in the report rather than
terminating the entire inventory.
