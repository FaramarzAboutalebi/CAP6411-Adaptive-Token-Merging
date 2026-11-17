"""Synthetic tests for the DINOv3 BraTS model; no restricted weights required."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from cap6411_atm.benchmark import benchmark_inference
from cap6411_atm.evaluation import evaluate_segmentation
from cap6411_atm.models import (
    DINOV3_VITB16_CHECKPOINT,
    BraTSDINOv3Segmenter,
    DINOv3Encoder,
    DINOv3SegmentationConfig,
    MRIInputAdapter,
    build_pretrained_brats_dinov3,
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


class SyntheticRopePositionEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("periods", torch.linspace(1.0, 100.0, 16))
        self.dtype = torch.float32
        self.normalize_coords = "separate"

    def forward(self, *, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = (torch.arange(H, dtype=self.dtype, device=self.periods.device) + 0.5) / H
        cols = (torch.arange(W, dtype=self.dtype, device=self.periods.device) + 0.5) / W
        coords = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = (2.0 * torch.pi * coords[:, :, None] / self.periods[None, None, :])
        angles = angles.flatten(1, 2).tile(2)
        return angles.sin(), angles.cos()


class SyntheticTransformerBlock(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.input_token_count = 0
        self.last_input: torch.Tensor | None = None
        self.last_output: torch.Tensor | None = None
        self.rope_shape: tuple[int, ...] | None = None

    def forward(
        self,
        tokens: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        self.input_token_count = tokens.shape[1]
        self.last_input = tokens.detach().clone()
        output = tokens + (self.index + 1) * 0.001
        if rope is not None:
            sin, _ = rope
            self.rope_shape = tuple(sin.shape)
            positions = sin.mean(dim=-1)
            if positions.ndim == 3:
                positions = positions.squeeze(1)
            patch_count = sin.shape[-2]
            prefix_count = output.shape[1] - patch_count
            patch_tokens = output[:, prefix_count:] + positions.unsqueeze(-1) * 0.001
            output = torch.cat((output[:, :prefix_count], patch_tokens), dim=1)
        else:
            self.rope_shape = None
        self.last_output = output.detach().clone()
        return output


class BlockwiseSyntheticDINOv3ViTB16(SyntheticDINOv3ViTB16):
    """Fixture exposing the pinned official block, token-prep, and RoPE APIs."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SyntheticTransformerBlock(index) for index in range(self.n_blocks)
        )
        self.rope_embed = SyntheticRopePositionEmbedding()
        self.norm = nn.LayerNorm(self.embed_dim)
        self.chunked_blocks = False

    def prepare_tokens_with_masks(
        self, images: torch.Tensor, masks: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        patch_map = self.patch_embed(images)
        batch, channels, height, width = patch_map.shape
        patches = patch_map.flatten(2).transpose(1, 2)
        tokens = torch.cat(
            (
                self.cls_token.expand(batch, -1, -1),
                self.storage_tokens.expand(batch, -1, -1),
                patches,
            ),
            dim=1,
        )
        return tokens, (height, width)

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
        tokens, (height, width) = self.prepare_tokens_with_masks(images)
        rope = self.rope_embed(H=height, W=width)
        outputs = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens, rope)
            if index in n:
                output = self.norm(tokens) if norm else tokens
                patches = output[:, 1 + self.n_storage_tokens :]
                if reshape:
                    patches = patches.reshape(
                        images.shape[0], height, width, self.embed_dim
                    ).permute(0, 3, 1, 2)
                outputs.append(patches.contiguous())
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

    def test_inter_block_zero_ratio_matches_official_dense_path(self) -> None:
        backbone = BlockwiseSyntheticDINOv3ViTB16()
        images = torch.randn(1, 3, 240, 240)
        baseline = DINOv3Encoder(backbone).forward_features(images)
        merged_encoder = DINOv3Encoder(
            backbone,
            merge_method="adaptive",
            merge_ratio=0.0,
            merge_after_block=2,
        )
        merged, state = merged_encoder.forward_features_with_merge_state(images)

        self.assertIsNotNone(state)
        for expected, actual in zip(baseline, merged):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(all(block.input_token_count == 230 for block in backbone.blocks))
        self.assertTrue(all(block.rope_shape == (225, 64) for block in backbone.blocks))

    def test_inter_block_merge_protects_special_tokens_and_reconstructs_all_taps(self) -> None:
        backbone = BlockwiseSyntheticDINOv3ViTB16()
        encoder = DINOv3Encoder(
            backbone,
            merge_method="adaptive",
            merge_ratio=0.5,
            merge_after_block=2,
        )
        features, state = encoder.forward_features_with_merge_state(
            torch.randn(1, 3, 240, 240)
        )

        self.assertIsNotNone(state)
        self.assertEqual(len(features), 4)
        self.assertTrue(all(feature.shape == (1, 768, 15, 15) for feature in features))
        processed_token_counts = [block.input_token_count for block in backbone.blocks]
        self.assertEqual(processed_token_counts[:3], [230, 230, 230])
        self.assertEqual(processed_token_counts[3:], [117] * 9)
        self.assertTrue(all(count < 230 for count in processed_token_counts[3:]))
        self.assertTrue(
            torch.equal(
                backbone.blocks[3].last_input[:, : encoder.special_token_count],
                backbone.blocks[2].last_output[:, : encoder.special_token_count],
            )
        )
        self.assertEqual(backbone.blocks[3].rope_shape, (1, 1, 112, 64))

        for feature, block_index in zip(features, self.model.config.feature_blocks):
            block_output = backbone.blocks[block_index].last_output
            patch_tokens = block_output[:, encoder.special_token_count :]
            if block_index > 2:
                from cap6411_atm.merging import reconstruct_tokens

                patch_tokens = reconstruct_tokens(patch_tokens, state)
            expected = backbone.norm(patch_tokens).reshape(1, 15, 15, 768)
            expected = expected.permute(0, 3, 1, 2).contiguous()
            self.assertTrue(torch.equal(feature, expected))

    def test_segmenter_decodes_inter_block_merged_features(self) -> None:
        model = BraTSDINOv3Segmenter(
            BlockwiseSyntheticDINOv3ViTB16(),
            merge_method="adaptive",
            merge_ratio=0.5,
            merge_after_block=2,
        )
        logits = model(torch.randn(1, 4, 240, 240))
        self.assertEqual(logits.shape, (1, 3, 240, 240))
        self.assertTrue(torch.isfinite(logits).all())
        probabilities = torch.sigmoid(logits)
        self.assertEqual(probabilities.shape, (1, 3, 240, 240))
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(torch.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        logits.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.adapter.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.decoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.backbone.parameters()))

    def test_pretrained_factory_exposes_opt_in_merge_configuration(self) -> None:
        backbone = BlockwiseSyntheticDINOv3ViTB16()
        with mock.patch(
            "cap6411_atm.models.dinov3_segmentation.load_official_dinov3_vitb16",
            return_value=backbone,
        ):
            model = build_pretrained_brats_dinov3(
                "authorized-repository",
                "authorized-checkpoint",
                merge_method="adaptive",
                merge_ratio=0.4,
                merge_after_block=5,
            )
        self.assertEqual(model.encoder.merge_method, "adaptive")
        self.assertEqual(model.encoder.merge_ratio, 0.4)
        self.assertEqual(model.encoder.merge_after_block, 5)

    def test_inter_block_model_runs_through_benchmark_and_evaluation_apis(self) -> None:
        model = BraTSDINOv3Segmenter(
            BlockwiseSyntheticDINOv3ViTB16(),
            merge_method="adaptive",
            merge_ratio=0.5,
            merge_after_block=2,
        )
        images = torch.randn(1, 4, 240, 240)
        targets = torch.zeros(1, 3, 240, 240)

        timing = benchmark_inference(model, images, warmup=0, iterations=1)
        metrics = evaluate_segmentation(
            model, [{"image": images, "target": targets}]
        )

        self.assertEqual(timing["output_shape"], [1, 3, 240, 240])
        self.assertEqual(metrics["samples"], 1)
        self.assertEqual(set(metrics["dice"]), {"WT", "TC", "ET"})

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
