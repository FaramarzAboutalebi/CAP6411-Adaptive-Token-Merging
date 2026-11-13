"""Sanitized Workstream 1 inference-artifact export and loading.

The shareable artifact intentionally contains only the trained MRI adapter and
segmentation decoder.  It never copies the gated DINOv3 backbone, optimizer,
AMP scaler, RNG state, patient identifiers, or private filesystem paths from a
training checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dinov3_segmentation import DINOv3SegmentationConfig, TARGET_ORDER


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_TYPE = "cap6411_brats_adapter_decoder"
_SHAREABLE_PREFIXES = ("adapter.", "decoder.")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError("checkpoint must be a dictionary")
    return value


def _final_validation(history: Any, threshold: float) -> dict[str, Any]:
    if not isinstance(history, list) or not history:
        raise ValueError("checkpoint history is missing or empty")
    validation = history[-1].get("validation")
    if not isinstance(validation, dict) or "dice" not in validation:
        raise ValueError("checkpoint lacks final validation metrics")
    return {
        "dice": {name: float(validation["dice"][name]) for name in TARGET_ORDER},
        "macro_dice": float(validation["macro_dice"]),
        "threshold": float(threshold),
        "split": "validation",
        "aggregation": "global_voxel_counts",
    }


def export_adapter_decoder_artifact(
    training_checkpoint: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Export a sanitized adapter/decoder artifact from a training checkpoint."""

    source = Path(training_checkpoint).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    checkpoint = _load_checkpoint(source)
    required = {"epoch", "history", "model_config", "model_state"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"training checkpoint missing keys: {sorted(missing)}")

    state = checkpoint["model_state"]
    if not isinstance(state, dict):
        raise ValueError("model_state must be a dictionary")
    shareable_state = {
        key: tensor.detach().cpu()
        for key, tensor in state.items()
        if key.startswith(_SHAREABLE_PREFIXES)
    }
    if not shareable_state or any(key.startswith("encoder.") for key in shareable_state):
        raise ValueError("unable to isolate adapter and decoder weights")
    if not any(key.startswith("adapter.") for key in shareable_state):
        raise ValueError("adapter weights are missing")
    if not any(key.startswith("decoder.") for key in shareable_state):
        raise ValueError("decoder weights are missing")

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "state_dict": shareable_state,
        "model_config": dict(checkpoint["model_config"]),
        "target_order": list(TARGET_ORDER),
        "source": {
            "checkpoint_sha256": sha256_file(source),
            "epoch_index": int(checkpoint["epoch"]),
            "completed_epochs": int(checkpoint["epoch"]) + 1,
        },
        "validation": _final_validation(
            checkpoint["history"],
            checkpoint.get("training_config", {}).get("validation_threshold", 0.5),
        ),
        "requirements": {
            "official_backbone_supplied_separately": True,
            "contains_dinov3_weights": False,
            "contains_optimizer_state": False,
            "contains_patient_data": False,
        },
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, destination)
    return artifact


def load_adapter_decoder_artifact(
    model: nn.Module,
    artifact_path: str | Path,
) -> dict[str, Any]:
    """Load sanitized weights into an already constructed official baseline."""

    path = Path(artifact_path).expanduser().resolve()
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported inference artifact schema")
    if artifact.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("not a CAP6411 adapter/decoder artifact")
    if artifact.get("target_order") != list(TARGET_ORDER):
        raise ValueError("artifact target ordering is incompatible")

    expected_config = asdict(getattr(model, "config", DINOv3SegmentationConfig()))
    stored_config = artifact.get("model_config")
    for key, expected in expected_config.items():
        actual = stored_config.get(key) if isinstance(stored_config, dict) else None
        if isinstance(expected, tuple) and isinstance(actual, list):
            actual = tuple(actual)
        if actual != expected:
            raise ValueError(f"artifact model_config mismatch for {key!r}")

    state = artifact.get("state_dict")
    if not isinstance(state, dict) or any(key.startswith("encoder.") for key in state):
        raise ValueError("artifact state must contain only adapter/decoder weights")
    adapter = {key.removeprefix("adapter."): value for key, value in state.items() if key.startswith("adapter.")}
    decoder = {key.removeprefix("decoder."): value for key, value in state.items() if key.startswith("decoder.")}
    model.adapter.load_state_dict(adapter, strict=True)
    model.decoder.load_state_dict(decoder, strict=True)
    return {key: value for key, value in artifact.items() if key != "state_dict"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_checkpoint", help="Private full training checkpoint")
    parser.add_argument("output", help="Destination for sanitized adapter/decoder artifact")
    args = parser.parse_args()
    artifact = export_adapter_decoder_artifact(args.training_checkpoint, args.output)
    print(f"wrote {args.output}")
    print(f"source SHA-256: {artifact['source']['checkpoint_sha256']}")


if __name__ == "__main__":
    main()
