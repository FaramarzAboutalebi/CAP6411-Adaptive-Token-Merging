"""Tests for sanitized Workstream 1 inference artifacts."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from cap6411_atm.models import (
    ARTIFACT_TYPE,
    BraTSDINOv3Segmenter,
    DINOv3SegmentationConfig,
    export_adapter_decoder_artifact,
    load_adapter_decoder_artifact,
    sha256_file,
)


class SyntheticBackbone(nn.Module):
    embed_dim = 768
    n_blocks = 12
    n_storage_tokens = 4
    patch_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 768, 16, stride=16, bias=False)

    def get_intermediate_layers(self, images, *, n, reshape, **_):
        feature = self.patch_embed(images)
        return tuple(feature for _ in n)


class ModelArtifactTests(unittest.TestCase):
    def _checkpoint(self, model: nn.Module) -> dict:
        return {
            "epoch": 1,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": {"private": "must not be exported"},
            "rng_state": {"private": "must not be exported"},
            "training_config": {"archive_path": "/private/patient-data.zip"},
            "dataset_metadata": {"train_patients": ["private-patient-id"]},
            "history": [
                {
                    "epoch": 1,
                    "validation": {
                        "dice": {"WT": 0.86, "TC": 0.85, "ET": 0.73},
                        "macro_dice": 0.8133333333,
                    },
                }
            ],
        }

    def test_export_is_sanitized_and_loads_adapter_decoder_strictly(self) -> None:
        torch.manual_seed(42)
        source_model = BraTSDINOv3Segmenter(SyntheticBackbone())
        with torch.no_grad():
            source_model.adapter.projection.bias.fill_(0.25)
            source_model.decoder.prediction.bias.fill_(-0.5)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "private-full-checkpoint.pt"
            output = root / "adapter-decoder.pt"
            torch.save(self._checkpoint(source_model), source)
            artifact = export_adapter_decoder_artifact(source, output)

            self.assertEqual(artifact["artifact_type"], ARTIFACT_TYPE)
            self.assertEqual(artifact["source"]["checkpoint_sha256"], sha256_file(source))
            self.assertEqual(artifact["source"]["completed_epochs"], 2)
            self.assertTrue(artifact["requirements"]["official_backbone_supplied_separately"])
            self.assertFalse(artifact["requirements"]["contains_dinov3_weights"])
            self.assertTrue(all(key.startswith(("adapter.", "decoder.")) for key in artifact["state_dict"]))
            self.assertFalse(any(key.startswith("encoder.") for key in artifact["state_dict"]))
            self.assertNotIn("training_config", artifact)
            self.assertNotIn("dataset_metadata", artifact)
            self.assertNotIn("optimizer_state", artifact)
            self.assertNotIn("rng_state", artifact)

            torch.manual_seed(99)
            target_model = BraTSDINOv3Segmenter(SyntheticBackbone())
            original_backbone = {
                key: value.clone() for key, value in target_model.encoder.state_dict().items()
            }
            metadata = load_adapter_decoder_artifact(target_model, output)
            self.assertNotIn("state_dict", metadata)
            for source_parameter, target_parameter in zip(
                source_model.adapter.parameters(), target_model.adapter.parameters()
            ):
                self.assertTrue(torch.equal(source_parameter, target_parameter))
            for source_parameter, target_parameter in zip(
                source_model.decoder.parameters(), target_model.decoder.parameters()
            ):
                self.assertTrue(torch.equal(source_parameter, target_parameter))
            for key, value in target_model.encoder.state_dict().items():
                self.assertTrue(torch.equal(value, original_backbone[key]))

    def test_loader_rejects_incompatible_config(self) -> None:
        model = BraTSDINOv3Segmenter(SyntheticBackbone())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            output = root / "artifact.pt"
            torch.save(self._checkpoint(model), source)
            export_adapter_decoder_artifact(source, output)
            artifact = torch.load(output, map_location="cpu", weights_only=True)
            artifact["model_config"]["embedding_dim"] = 123
            torch.save(artifact, output)
            with self.assertRaisesRegex(ValueError, "embedding_dim"):
                load_adapter_decoder_artifact(model, output)


if __name__ == "__main__":
    unittest.main()
