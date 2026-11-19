# Integrated WS2/WS3/WS4 CUDA smoke test

This is a final integration gate, not a performance experiment. It uses one
seeded synthetic 4-channel image and an empty synthetic target. It does not
open the BraTS archive, retrain the model, or evaluate any dataset split.

## Colab setup

Select **Runtime → Change runtime type → T4 GPU** (or another CUDA GPU). The
integration branch must first be available to this authorized Colab session;
it is intentionally not pushed by the preparation task.

Run this cell, editing only the private Drive paths if necessary:

```python
from google.colab import drive
drive.mount("/content/drive")

import subprocess
from pathlib import Path

INTEGRATION_COMMIT = "34222fb1696c74bb2b346710b2edd270e62afd1b"
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
REPO = Path("/content/CAP6411-Adaptive-Token-Merging")
DINO = Path("/content/dinov3")

# This clone/fetch requires the integration branch to have been made available
# after approval. An uploaded private repository snapshot may be used instead.
if not REPO.exists():
    subprocess.run([
        "git", "clone", "--branch", "integration/ws2-ws3-ws4",
        "https://github.com/FaramarzAboutalebi/CAP6411-Adaptive-Token-Merging.git",
        str(REPO),
    ], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "integration/ws2-ws3-ws4"], check=True)
subprocess.run([
    "git", "-C", str(REPO), "merge-base", "--is-ancestor",
    INTEGRATION_COMMIT, "HEAD",
], check=True)

if not DINO.exists():
    subprocess.run([
        "git", "clone", "https://github.com/facebookresearch/dinov3.git", str(DINO)
    ], check=True)
subprocess.run(["git", "-C", str(DINO), "checkout", "-q", DINOV3_COMMIT], check=True)

subprocess.run(["python", "-m", "pip", "install", "-q", "-e", str(REPO)], check=True)
subprocess.run(["python", "-m", "pip", "install", "-q", "-e", str(DINO), "--no-deps"], check=True)
subprocess.run([
    "python", "-m", "pip", "install", "-q",
    "ftfy", "omegaconf", "regex", "scikit-learn", "submitit",
    "termcolor", "torchmetrics", "scipy", "pandas", "matplotlib", "nibabel",
], check=True)
```

Then run the smoke test:

```python
CHECKPOINT = Path(
    "/content/drive/MyDrive/CAP6411/checkpoints/"
    "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)
ARTIFACT = Path(
    "/content/drive/MyDrive/CAP6411/team_integration/workstream1_epoch2/"
    "dinov3_vitb16_brats_epoch2_adapter_decoder.pt"
)

assert CHECKPOINT.is_file(), f"Missing authorized DINOv3 checkpoint: {CHECKPOINT}"
assert ARTIFACT.is_file(), f"Missing WS1 adapter/decoder artifact: {ARTIFACT}"

subprocess.run([
    "python", str(REPO / "scripts/integration_cuda_smoke.py"),
    "--repo-root", str(REPO),
    "--dinov3-repo", str(DINO),
    "--checkpoint", str(CHECKPOINT),
    "--artifact", str(ARTIFACT),
    "--warmup", "3",
    "--iterations", "10",
], check=True)
```

## PASS criteria

The run passes only if it ends with `SMOKE TEST PASS` and all assertions hold:

1. CUDA is available and the integration commit is an ancestor of `HEAD`.
2. The WS1 artifact SHA-256 equals the documented value.
3. All variants strictly load identical adapter/decoder weights; their official
   DINOv3 backbones remain frozen and in evaluation mode.
4. Baseline, uniform, adaptive, and zero-merging forwards return finite
   `[1, 3, 240, 240]` logits.
5. The zero-merging logits match baseline within `rtol=1e-5, atol=1e-6`.
6. Uniform and adaptive methods each achieve exactly 112 retained patches at
   retention 0.50 and 169 at retention 0.75.
7. WS3 reports all three families, positive finite synchronized latency values,
   positive CUDA peak allocation, and encoder-stage timing after warmup.
8. WS4 emits WT/TC/ET rows for one complete synthetic validation volume for
   every variant.

Any exception, failed assertion, CUDA out-of-memory error, non-finite result,
missing output row, or nonzero subprocess exit is **FAIL**. Synthetic Dice
values and timing numbers are diagnostic only and must not be reported as
BraTS performance.
