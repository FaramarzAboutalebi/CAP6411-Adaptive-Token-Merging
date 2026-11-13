# Workstream 1 model artifacts

## Recommended artifact

Use `dinov3_vitb16_brats_epoch2_adapter_decoder.pt` for teammate integration.
It is an inference-only export from the best checkpoint after two full-cohort
epochs. It contains 27 CPU tensors (the trained MRI adapter and segmentation
decoder only) plus sanitized model and validation metadata.

```text
Size:    5,276,889 bytes
SHA-256: ed0d84caf6bb6d4dcb4ee2dc29dc911c628406204e9683e7cde3b280b796314b
Source full-checkpoint SHA-256:
         fca82cd0e7c94c0e0baebf36b35dd8c3a78ca251ffa8e08b14a059489b3ffd7e
```

The artifact does **not** contain the DINOv3 backbone, optimizer, AMP scaler,
RNG state, patient identifiers, MRI data, credentials, or private filesystem
paths. The original 358,740,011-byte training checkpoint remains private and
unchanged.

The artifact records these full-validation results:

| Metric | Epoch 2 validation Dice |
|---|---:|
| WT | 0.86898142 |
| TC | 0.85647742 |
| ET | 0.73206816 |
| Macro | 0.81917567 |

These are global voxel-count Dice values on the 187-patient validation split,
not test results. The isolated test split remains unevaluated.

## Access and licensing boundary

The official DINOv3 code and weights use Meta's DINOv3 License. Authorized
users must accept the applicable terms and obtain
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` through Meta's official gated
access process. Do not copy that checkpoint into this repository or the team
artifact package. Redistribution of DINOv3 materials or derivatives is subject
to the license, including its requirement to provide the license terms.

The adapter/decoder artifact is kept in restricted team storage because public
redistribution has not been cleared against every applicable DINOv3 and BraTS
term. Grant access only to specifically authorized teammates; do not enable
public or anyone-with-the-link access. This document is an engineering access
boundary, not legal advice.

Official references:

- DINOv3 repository and license: <https://github.com/facebookresearch/dinov3>
- Official gated ViT-B/16 model card:
  <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
- BraTS data access must be obtained through the challenge's authorized
  distribution channel; the dataset and patient files are never part of this
  package.

## Load the artifact

First construct the exact official backbone using the audited revision and the
authorized local checkpoint. Then load the adapter and decoder strictly:

```python
import torch

from cap6411_atm.models import (
    DINOv3SegmentationConfig,
    build_pretrained_brats_dinov3,
    load_adapter_decoder_artifact,
)

config = DINOv3SegmentationConfig.from_json(
    "configs/dinov3_vitb16_brats.json"
)
model = build_pretrained_brats_dinov3(
    repository="/path/to/dinov3",
    checkpoint=(
        "/authorized/path/"
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    ),
    config=config,
)
metadata = load_adapter_decoder_artifact(
    model,
    "/restricted/team/path/dinov3_vitb16_brats_epoch2_adapter_decoder.pt",
)
assert metadata["validation"]["split"] == "validation"
model.eval()
```

The loader validates the schema, target order, and every model-configuration
field. It loads only `model.adapter` and `model.decoder`; the independently
obtained official backbone remains frozen and unchanged.

## Inference and feature extraction

`images` must be the existing dataset loader's normalized float tensor in
T1n/T1c/T2w/T2f order:

```python
with torch.inference_mode():
    images = batch["image"].to(device)       # [B, 4, 240, 240]
    logits = model(images)                   # [B, 3, 240, 240]
    probabilities = logits.sigmoid()         # WT, TC, ET
    predictions = probabilities >= 0.5

    features = model.extract_features(images)
    assert len(features) == 4
    assert all(x.shape[1:] == (768, 15, 15) for x in features)
    baseline_logits = model.decode_features(features)
```

Use `StreamingDice(threshold=0.5)` for the committed global validation
protocol. Keep the patient manifest, preprocessing, validation patients, and
channel order unchanged when comparing merged and unmerged paths.

## Export procedure for an authorized checkpoint custodian

The original checkpoint is read but never changed:

```bash
python - <<'PY'
from cap6411_atm.models import export_adapter_decoder_artifact

export_adapter_decoder_artifact(
    "/private/path/to/best.pt",
    "/restricted/team/path/dinov3_vitb16_brats_epoch2_adapter_decoder.pt",
)
PY
```

Verify the result:

```bash
python - <<'PY'
from cap6411_atm.models import sha256_file

path = "/restricted/team/path/dinov3_vitb16_brats_epoch2_adapter_decoder.pt"
print(sha256_file(path))
PY
```

Do not export or share the full `model_state`, because it embeds the official
DINOv3 backbone. Do not add any `.pt` artifact to Git.

## Required and optional teammate inputs

Required for trained inference:

- this repository at the documented commit;
- official DINOv3 source revision
  `6876159a11b4df116f30f667f8c9888617df0751`;
- each teammate's independently authorized official ViT-B/16 checkpoint;
- the restricted adapter/decoder artifact and matching SHA-256;
- BraTS data and split manifest only when running data-backed inference or
  validation.

Optional:

- the private full training checkpoint, needed only for exact optimizer/RNG
  resume and not needed for inference or token-merging integration;
- validation history JSON, because the sanitized artifact already records the
  selected epoch's metrics;
- the BraTS archive for weight-free interface development.

Token-merging and reconstruction development can run without any restricted
weights or MRI data using `SyntheticDINOv3ViTB16` in
`tests/test_dinov3_segmentation.py`. Zero-merging equivalence, spatial order,
shape, dtype, device, batch isolation, and gradient tests should pass before a
trained comparison is attempted.
