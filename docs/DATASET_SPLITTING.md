# Patient-level BraTS splits

`cap6411_atm.data.splitting` creates deterministic patient-level train,
validation, and test manifests from the ZIP archive's structural inventory. It
does not extract or read MRI volumes.

The Workstream 1 protocol uses:

- seed: `42`
- train: `70%` (`876` of 1,251 patients)
- validation: `15%` (`187` patients)
- test: `15%` (`188` patients)

IDs are sorted before a seed-controlled shuffle, so ZIP member ordering does not
change the result. Cumulative rounded boundaries produce counts that always sum
to the full cohort. Every patient is validated to appear exactly once.

## Generate manifests

From the repository root:

```bash
python -m cap6411_atm.data.splitting \
  ~/Downloads/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip \
  --inspection-config configs/brats2023_gli_inspection.json \
  --seed 42 \
  --expected-patient-count 1251 \
  --json outputs/brats2023_gli_split_seed42.json \
  --csv outputs/brats2023_gli_split_seed42.csv
```

The command refuses to generate a manifest if the archive contains incomplete
patients or the discovered count differs from 1,251.

## Load manifests

```python
from cap6411_atm.data.splitting import (
    load_csv_manifest,
    load_json_manifest,
    patient_to_split,
)

manifest = load_json_manifest("outputs/brats2023_gli_split_seed42.json")
splits = manifest["splits"]
lookup = patient_to_split(splits)

# Assign a patient before expanding it into axial slices.
split_name = lookup[patient_id]
```

CSV loading returns the split lists directly:

```python
splits = load_csv_manifest("outputs/brats2023_gli_split_seed42.csv")
```

Future slice datasets must choose patients from one manifest split first and only
then enumerate that patient's slices. They must never randomly partition a global
slice list, because that would allow slices from one patient to cross boundaries.
