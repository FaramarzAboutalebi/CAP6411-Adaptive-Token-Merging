# DINOv3 BraTS segmentation baseline

The Phase 1 model implements the audited architecture without training or
adaptive token merging:

```text
[B,4,240,240] T1n/T1c/T2w/T2f
→ trainable balanced 1×1 adapter (4→3)
→ frozen DINOv3 ViT-B/16
→ normalized block [2,5,8,11] maps, each [B,768,15,15]
→ lightweight same-grid fusion and four 2× refinement stages
→ raw WT/TC/ET logits [B,3,240,240]
```

The model applies neither sigmoid nor softmax. WT, TC, and ET are overlapping
targets and require a sigmoid-compatible multi-label loss.

## Authorized backbone setup

The implementation does not download weights and never substitutes a random
DINOv3 model. Obtain access through Meta, then prepare:

1. An official DINOv3 checkout at revision
   `6876159a11b4df116f30f667f8c9888617df0751`.
2. The approved checkpoint
   `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` outside Git.
3. The dependencies required by that pinned DINOv3 checkout in the Colab
   runtime.

The pin and model dimensions are recorded in
`configs/dinov3_vitb16_brats.json`. If the source revision, checkpoint name, or
load operation differs, construction fails explicitly.

## Usage

```python
from cap6411_atm.models import (
    DINOv3SegmentationConfig,
    build_pretrained_brats_dinov3,
)

config = DINOv3SegmentationConfig.from_json(
    "configs/dinov3_vitb16_brats.json"
)
model = build_pretrained_brats_dinov3(
    repository="/content/dinov3",
    checkpoint="/content/drive/MyDrive/checkpoints/"
               "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    config=config,
).cuda()

batch = next(iter(train_loader))
images = batch["image"].cuda()       # float32 [B,4,240,240]
logits = model(images)                # raw [B,3,240,240]
```

Only the adapter and decoder belong in the optimizer:

```python
trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-3)
```

Although backbone parameters are frozen and remain in evaluation mode, do not
wrap the encoder in `torch.no_grad()` while the adapter is trainable. Backward
must traverse the frozen operations to update the adapter.

## Dense feature boundary

`model.extract_features(images)` returns the four dense maps, and
`model.decode_features(features)` consumes them. A future token-merging method
must reconstruct all 225 row-major patch positions before calling the decoder.
This boundary does not implement or imply token merging.

## Synthetic validation

The unit tests inject a lightweight synthetic backbone that implements the
official intermediate-feature contract, including one class token and four
storage tokens. It is explicitly not a pretrained model and provides no
evidence about pretrained accuracy or T4 performance.

```bash
python -m unittest tests.test_dinov3_segmentation -v
python -m unittest discover -s tests -v
```

## Remaining Colab validation

- Accept the DINOv3 license and obtain authorized checkpoint access.
- Recreate the exact pinned source revision and install its compatible packages.
- Load the real checkpoint and run strict shape/gradient smoke tests.
- Measure batches 1, 2, and 4 on an actual 16 GB T4 with CUDA AMP FP16.
- Record allocated and reserved peak VRAM and synchronized forward/backward time.
- Start with physical batch size 2 unless those measurements support a change.

Pretrained checkpoint loading and T4 compatibility remain unverified until
those steps are actually executed.
