"""Synthetic tests for the DINOv3 BraTS model; no restricted weights required."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from cap6411_atm.models import (
    DINOV3_VITB16_CHECKPOINT,
    BraTSDINOv3Segmenter,
    DINOv3Encoder,
    DINOv3SegmentationConfig,
    MRIInputAdapter,
    load_official_dinov3_vitb16,
)


class SyntheticDINOv3ViTB16(nn.Module):
    """Small-compute fixture matching the official dense-feature interface."""

    embed_dim = 768
    n_blocks = 12
    n_storage_tokens = 4
    patch_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, self.embed_dim, 16, stride=16, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, 4, self.embed_dim))
        self.scale = nn.Parameter(torch.linspace(1.0, 2.1, self.n_blocks))
        self.last_call: dict | None = None

    def get_intermediate_layers(
        self,
        images: torch.Tensor,
        *,
        n,
        reshape: bool,
        return_class_token: bool,
        return_extra_tokens: bool,
        norm: bool,
    ):
        self.last_call = {
            "n": n,
            "reshape": reshape,
            "return_class_token": return_class_token,
            "return_extra_tokens": return_extra_tokens,
            "norm": norm,
        }
        patch_map = self.patch_embed(images)
        batch, channels, height, width = patch_map.shape
        patches = patch_map.flatten(2).transpose(1, 2)
        full_tokens = torch.cat(
            (
                self.cls_token.expand(batch, -1, -1),
                self.storage_tokens.expand(batch, -1, -1),
                patches,
            ),
            dim=1,
        )
        outputs = []
        for block in n:
            block_tokens = full_tokens * self.scale[block]
            if norm:
                block_tokens = nn.functional.layer_norm(block_tokens, (channels,))
            spatial = block_tokens[:, 1 + self.n_storage_tokens :]
            if reshape:
                spatial = spatial.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
            outputs.append(spatial.contiguous())
        return tuple(outputs)


class DINOv3SegmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.backbone = SyntheticDINOv3ViTB16()
        self.model = BraTSDINOv3Segmenter(self.backbone)

    def test_adapter_shape_dtype_and_balanced_initialization(self) -> None:
        adapter = MRIInputAdapter()
        inputs = torch.randn(2, 4, 240, 240, dtype=torch.float32)
        outputs = adapter(inputs)

        self.assertEqual(outputs.shape, (2, 3, 240, 240))
        self.assertEqual(outputs.dtype, torch.float32)
        expected = 0.5 * torch.tensor(
            [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1]], dtype=torch.float32
        )
        self.assertTrue(torch.equal(adapter.projection.weight[:, :, 0, 0], expected))
        self.assertTrue(torch.equal(adapter.projection.bias, torch.zeros(3)))

    def test_invalid_input_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[B,4"):
            self.model(torch.randn(1, 3, 240, 240))
        with self.assertRaisesRegex(ValueError, "240"):
            self.model(torch.randn(1, 4, 224, 224))

    def test_intermediate_features_and_special_token_options(self) -> None:
        features = self.model.extract_features(torch.randn(1, 4, 240, 240))

        self.assertEqual(len(features), 4)
        self.assertTrue(all(feature.shape == (1, 768, 15, 15) for feature in features))
        self.assertEqual(self.model.encoder.special_token_count, 5)
        self.assertEqual(
            self.backbone.last_call,
            {
                "n": [2, 5, 8, 11],
                "reshape": True,
                "return_class_token": False,
                "return_extra_tokens": False,
                "norm": True,
            },
        )

    def test_output_shape_and_raw_logits(self) -> None:
        logits = self.model(torch.randn(1, 4, 240, 240))

        self.assertEqual(logits.shape, (1, 3, 240, 240))
        self.assertTrue(logits.is_floating_point())
        self.assertTrue(torch.isfinite(logits).all())

    def test_backbone_frozen_adapter_decoder_trainable_and_eval_locked(self) -> None:
        self.assertTrue(all(not parameter.requires_grad for parameter in self.backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.adapter.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.decoder.parameters()))

        self.model.train()
        self.assertTrue(self.model.training)
        self.assertFalse(self.backbone.training)

    def test_gradients_reach_adapter_and_decoder_not_backbone(self) -> None:
        inputs = torch.randn(1, 4, 240, 240)
        loss = self.model(inputs).square().mean()
        loss.backward()

        self.assertTrue(
            all(parameter.grad is not None for parameter in self.model.adapter.parameters())
        )
        self.assertTrue(
            all(parameter.grad is not None for parameter in self.model.decoder.parameters())
        )
        self.assertTrue(all(parameter.grad is None for parameter in self.backbone.parameters()))

    def test_incompatible_storage_tokens_are_rejected(self) -> None:
        backbone = SyntheticDINOv3ViTB16()
        backbone.n_storage_tokens = 0
        with self.assertRaisesRegex(ValueError, "n_storage_tokens"):
            DINOv3Encoder(backbone)

    def test_dataset_batch_dictionary_is_compatible(self) -> None:
        dataset_batch = {
            "image": torch.randn(1, 4, 240, 240, dtype=torch.float32),
            "target": torch.zeros(1, 3, 240, 240, dtype=torch.float32),
            "patient_id": ["synthetic-patient"],
            "slice_index": torch.tensor([77]),
            "split": ["train"],
        }
        logits = self.model(dataset_batch["image"])
        self.assertEqual(logits.shape, dataset_batch["target"].shape)

    def test_json_configuration_matches_defaults(self) -> None:
        root = Path(__file__).resolve().parents[1]
        loaded = DINOv3SegmentationConfig.from_json(
            root / "configs" / "dinov3_vitb16_brats.json"
        )
        self.assertEqual(loaded, DINOv3SegmentationConfig())

    def test_checkpoint_loader_rejects_missing_and_never_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "Git checkout"):
                load_official_dinov3_vitb16(root / "missing", root / "missing.pth")

            repository = root / "dinov3"
            (repository / ".git").mkdir(parents=True)
            checkpoint = root / DINOV3_VITB16_CHECKPOINT
            checkpoint.write_bytes(b"synthetic-not-a-checkpoint")
            with (
                mock.patch(
                    "cap6411_atm.models.dinov3_segmentation._repository_head",
                    return_value=DINOv3SegmentationConfig().dinov3_revision,
                ),
                mock.patch("torch.hub.load", side_effect=ValueError("synthetic failure")),
            ):
                with self.assertRaisesRegex(RuntimeError, "no random-weight fallback"):
                    load_official_dinov3_vitb16(repository, checkpoint)


if __name__ == "__main__":
    unittest.main()
