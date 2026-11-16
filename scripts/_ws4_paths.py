"""Shared path handling for the Workstream 4 scripts.

Private inputs (BraTS ZIP, split manifest, Workstream 1 artifact, DINOv3 checkout and
checkpoint) are never hard-coded. Pass them as arguments or set the matching
environment variables once per shell:

    CAP6411_BRATS_ZIP      ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip
    CAP6411_MANIFEST       brats2023_gli_split_seed42.json
    CAP6411_ARTIFACT       dinov3_vitb16_brats_epoch2_adapter_decoder.pt
    CAP6411_DINOV3_REPO    official DINOv3 Git checkout (pinned revision)
    CAP6411_DINOV3_CKPT    dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
    CAP6411_CACHE_DIR      optional per-patient .npy cache directory
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

INSPECTION_CONFIG = REPO_ROOT / "configs" / "brats2023_gli_inspection.json"
MODEL_CONFIG = REPO_ROOT / "configs" / "dinov3_vitb16_brats.json"

_ENV = {
    "archive": "CAP6411_BRATS_ZIP",
    "manifest": "CAP6411_MANIFEST",
    "artifact": "CAP6411_ARTIFACT",
    "dinov3_repo": "CAP6411_DINOV3_REPO",
    "dinov3_checkpoint": "CAP6411_DINOV3_CKPT",
    "cache_dir": "CAP6411_CACHE_DIR",
}


def _default(name: str) -> Path | None:
    value = os.environ.get(_ENV[name])
    return Path(value) if value else None


def add_data_args(parser: argparse.ArgumentParser, *, model: bool = True) -> None:
    """Add the private-input arguments (defaults come from the environment variables above)."""

    parser.add_argument("--archive", type=Path, default=_default("archive"), help="BraTS 2023 GLI training ZIP")
    parser.add_argument("--manifest", type=Path, default=_default("manifest"), help="patient split manifest JSON")
    parser.add_argument("--cache-dir", type=Path, default=_default("cache_dir"), help="optional patient cache")
    if model:
        parser.add_argument("--artifact", type=Path, default=_default("artifact"),
                            help="Workstream 1 adapter/decoder artifact")
        parser.add_argument("--dinov3-repo", type=Path, default=_default("dinov3_repo"),
                            help="official DINOv3 Git checkout")
        parser.add_argument("--dinov3-checkpoint", type=Path, default=_default("dinov3_checkpoint"),
                            help="authorized dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")


def require(args: argparse.Namespace, *names: str) -> None:
    """Fail early with a clear message when a required private input is missing."""

    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        flags = ", ".join(f"--{n.replace('_', '-')} (or ${_ENV[n]})" for n in missing)
        raise SystemExit(f"missing required inputs: {flags}")
    for name in names:
        path = getattr(args, name)
        if not path.exists():
            raise SystemExit(f"--{name.replace('_', '-')} does not exist: {path}")
