# BraTS DINOv3 baseline training

The reusable baseline trains only the four-to-three-channel MRI adapter and the
segmentation decoder. The official DINOv3 ViT-B/16 backbone remains frozen, but
its forward pass is **not** wrapped in `no_grad`: gradients must pass through it
to train the input adapter.

## Data and target contract

`BraTSAxialSliceDataset` supplies float32 images in T1n, T1c, T2w, T2f order and
overlapping float32 WT, TC, ET masks. The inspection configuration retains the
explicit filename mapping `t1 -> t1n`, `t1ce -> t1c`, `t2 -> t2w`, and
`flair -> t2f`. Training never constructs a test dataset; the test split stays
isolated for a later final evaluation.

The loss is `0.5 * BCEWithLogits + 0.5 * soft Dice` by default. Validation Dice
is accumulated over the complete evaluated validation subset, separately for
WT, TC, and ET. Empty slices are not awarded individual perfect scores. If an
entire evaluated channel has no predicted or target voxels, its Dice is defined
as 1.0 and the corresponding `all_empty` flag is true.

## Colab setup and smoke test

Run these commands from the repository root after mounting Google Drive. The
DINOv3 checkout and checkpoint must already be available through authorized
access; the training code never downloads or substitutes weights.

```bash
pip install -e .
cd /content/dinov3
git checkout 6876159a11b4df116f30f667f8c9888617df0751
cd /content/CAP6411-Adaptive-Token-Merging
python -m cap6411_atm.training.baseline \
  --config configs/brats2023_gli_baseline_training.json \
  --smoke-test
```

The smoke test performs exactly one optimizer update and reports elapsed time,
samples/second, and peak CUDA allocation. Run training as a new command so the
model and optimizer are rebuilt from the official checkpoint.

## Pilot, full run, and resume

The committed configuration starts at batch size 2 with AMP and enables an
8-patient train / 4-patient validation pilot. Patients are selected
deterministically from their existing manifest splits (seed 42 for train and 43
for validation), so there is no leakage. Every history record is marked
`scope: pilot`, `pilot_metrics: true`, and
`metrics_scope: deterministic_patient_subset`; pilot metrics are not full-cohort
results.

```bash
# Deterministic pilot
python -m cap6411_atm.training.baseline \
  --config configs/brats2023_gli_baseline_training.json

# Resume the same run
python -m cap6411_atm.training.baseline \
  --config configs/brats2023_gli_baseline_training.json \
  --resume /content/drive/MyDrive/CAP6411/checkpoints/dinov3_vitb16_brats_baseline/last.pt
```

Before a full run, copy the JSON configuration to a new experiment config,
change `pilot_enabled` to `false`, choose a distinct `experiment_name`, and run
another smoke test. Do not point a full run at a pilot checkpoint. Batch size 2
is only the initial T4 setting; use smoke-test peak memory to decide whether it
must be reduced or can be increased. Actual T4 capacity is not claimed until
measured on a T4.

Each epoch records measured train/validation throughput and estimates the
corresponding full-split epoch duration. The sampler shuffles patient order and
slices deterministically while keeping every batch within one patient. With the
one-patient memory/disk cache, this avoids repeatedly decompressing a patient.
The default persistent cache is Colab-local (`/content/brats_cache`) so Drive is
not burdened with thousands of transient cache files.

Outputs are written atomically under
`<output_dir>/<experiment_name>/`: `last.pt`, `best.pt`, and `history.json`.
Checkpoints contain model, optimizer, AMP scaler, configuration, selected
patients, history, RNG state, and environment metadata.
