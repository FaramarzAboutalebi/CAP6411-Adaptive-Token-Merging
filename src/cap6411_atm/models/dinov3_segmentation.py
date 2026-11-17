"""Frozen DINOv3 ViT-B/16 baseline for 2D BraTS segmentation.

The production factory requires an authorized local DINOv3 checkout and
checkpoint.  Tests may inject a synthetic backbone with the same documented
``get_intermediate_layers`` interface; the module never silently substitutes
random weights for the requested pretrained model.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..merging import TokenMergeState, merge_tokens, reconstruct_tokens


DINOV3_VITB16_CHECKPOINT = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DINOV3_VITB16_HUB_ENTRY = "dinov3_vitb16"
MODALITY_ORDER = ("T1n", "T1c", "T2w", "T2f")
TARGET_ORDER = ("WT", "TC", "ET")


@dataclass(frozen=True)
class DINOv3SegmentationConfig:
    """Shape and decoder contract for the audited BraTS baseline."""

    input_channels: int = 4
    encoder_channels: int = 3
    input_size: tuple[int, int] = (240, 240)
    patch_size: int = 16
    embedding_dim: int = 768
    transformer_blocks: int = 12
    storage_tokens: int = 4
    feature_blocks: tuple[int, ...] = (2, 5, 8, 11)
    lateral_channels: int = 128
    decoder_channels: tuple[int, ...] = (128, 96, 64, 32)
    output_channels: int = 3
    checkpoint_filename: str = DINOV3_VITB16_CHECKPOINT
    dinov3_revision: str = "6876159a11b4df116f30f667f8c9888617df0751"

    @classmethod
    def from_json(cls, path: str | Path) -> "DINOv3SegmentationConfig":
        """Load configuration while converting JSON arrays to tuples."""

        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("input_size", "feature_blocks", "decoder_channels"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def __post_init__(self) -> None:
        if self.input_channels != 4 or self.encoder_channels != 3:
            raise ValueError("the audited baseline requires a 4-to-3 input adapter")
        if self.input_size != (240, 240):
            raise ValueError("the audited baseline requires 240x240 inputs")
        if self.patch_size != 16 or any(size % self.patch_size for size in self.input_size):
            raise ValueError("input dimensions must be divisible by patch size 16")
        if self.embedding_dim != 768 or self.transformer_blocks != 12:
            raise ValueError("the audited backbone is DINOv3 ViT-B/16")
        if self.storage_tokens != 4:
            raise ValueError("the official ViT-B/16 checkpoint uses four storage tokens")
        if self.feature_blocks != (2, 5, 8, 11):
            raise ValueError("the audited feature taps are blocks (2, 5, 8, 11)")
        if len(self.decoder_channels) != 4:
            raise ValueError("four decoder stages are required to upsample 15 to 240")
        if self.output_channels != 3:
            raise ValueError("the BraTS target contract requires WT, TC, and ET logits")

    @property
    def feature_grid(self) -> tuple[int, int]:
        return tuple(size // self.patch_size for size in self.input_size)


class MRIInputAdapter(nn.Module):
    """Learn a per-pixel projection from T1n/T1c/T2w/T2f to three channels.

    The biased 1x1 convolution is initialized with three balanced orthonormal
    combinations of the four independently normalized modalities.  This keeps
    initial row norms at one, but it does not make MRI channels equivalent to
    RGB and cannot preserve all four linear degrees of freedom.
    """

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(4, 3, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        balanced = 0.5 * torch.tensor(
            (
                (1.0, 1.0, 1.0, 1.0),
                (1.0, -1.0, 1.0, -1.0),
                (1.0, 1.0, -1.0, -1.0),
            ),
            dtype=self.projection.weight.dtype,
            device=self.projection.weight.device,
        )
        with torch.no_grad():
            self.projection.weight.copy_(balanced[:, :, None, None])
            self.projection.bias.zero_()

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 4:
            raise ValueError(f"expected MRI images [B,4,H,W], received {tuple(images.shape)}")
        if not images.is_floating_point():
            raise TypeError("MRI images must use a floating-point dtype")
        return self.projection(images)


def _rope_for_merged_coordinates(
    rope_embedding: nn.Module,
    coordinates: Tensor,
    grid_size: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Evaluate DINOv3 axial RoPE at per-sample merged-token coordinates."""
    periods = getattr(rope_embedding, "periods", None)
    if not isinstance(periods, Tensor):
        raise TypeError("DINOv3 RoPE module must expose its periods tensor")
    normalize = getattr(rope_embedding, "normalize_coords", None)
    if normalize == "max":
        height_denominator = width_denominator = max(grid_size)
    elif normalize == "min":
        height_denominator = width_denominator = min(grid_size)
    elif normalize == "separate":
        height_denominator, width_denominator = grid_size
    else:
        raise ValueError(f"unsupported DINOv3 RoPE coordinate normalization: {normalize!r}")

    rope_dtype = getattr(rope_embedding, "dtype", None) or periods.dtype
    coords = coordinates.to(device=periods.device, dtype=rope_dtype)
    row = 2.0 * (coords[..., 0] + 0.5) / height_denominator - 1.0
    col = 2.0 * (coords[..., 1] + 0.5) / width_denominator - 1.0
    normalized_coords = torch.stack((row, col), dim=-1)
    angles = 2.0 * math.pi * normalized_coords.unsqueeze(-1) / periods[None, None, None, :]
    angles = angles.flatten(-2).tile(2)
    sin = angles.sin().unsqueeze(1)
    cos = angles.cos().unsqueeze(1)
    return sin, cos


class DINOv3Encoder(nn.Module):
    """Freeze DINOv3 parameters while preserving input-adapter autograd.

    By default the wrapper calls the official patch-only intermediate-feature
    API. Optional blockwise merging instead retains protected special tokens,
    propagates coordinate-aware RoPE, and reconstructs dense features at taps.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: DINOv3SegmentationConfig = DINOv3SegmentationConfig(),
        *,
        merge_method: str | None = None,
        merge_ratio: float = 0.5,
        merge_after_block: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.merge_method = merge_method
        self.merge_ratio = merge_ratio
        self.merge_after_block = merge_after_block
        self._validate_backbone_contract()
        if self.merge_method is not None:
            self._validate_blockwise_merge_contract()
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @property
    def special_token_count(self) -> int:
        return 1 + self.config.storage_tokens

    def _validate_backbone_contract(self) -> None:
        if not callable(getattr(self.backbone, "get_intermediate_layers", None)):
            raise TypeError("DINOv3 backbone must provide get_intermediate_layers")
        checks = {
            "embed_dim": self.config.embedding_dim,
            "n_blocks": self.config.transformer_blocks,
            "n_storage_tokens": self.config.storage_tokens,
        }
        for attribute, expected in checks.items():
            actual = getattr(self.backbone, attribute, None)
            if actual != expected:
                raise ValueError(
                    f"incompatible DINOv3 {attribute}: expected {expected}, received {actual}"
                )
        patch_size = getattr(self.backbone, "patch_size", None)
        if patch_size not in (self.config.patch_size, (self.config.patch_size,) * 2):
            raise ValueError(
                f"incompatible DINOv3 patch_size: expected 16, received {patch_size}"
            )

    def _validate_blockwise_merge_contract(self) -> None:
        if self.merge_method not in ("uniform", "adaptive"):
            raise ValueError(f"unsupported merge method: {self.merge_method!r}")
        if not 0.0 <= self.merge_ratio <= 1.0:
            raise ValueError(f"merge_ratio must be in [0,1], received {self.merge_ratio}")
        blocks = getattr(self.backbone, "blocks", None)
        if not isinstance(blocks, (nn.ModuleList, list, tuple)) or len(blocks) != self.config.transformer_blocks:
            raise TypeError("blockwise merging requires the official unchunked DINOv3 blocks")
        if bool(getattr(self.backbone, "chunked_blocks", False)):
            raise ValueError("blockwise merging does not support chunked DINOv3 blocks")
        if not callable(getattr(self.backbone, "prepare_tokens_with_masks", None)):
            raise TypeError("blockwise merging requires prepare_tokens_with_masks")
        if not callable(getattr(self.backbone, "rope_embed", None)):
            raise TypeError("blockwise merging requires DINOv3 RoPE embeddings")
        if not callable(getattr(self.backbone, "norm", None)):
            raise TypeError("blockwise merging requires the DINOv3 final norm")
        if not 0 <= self.merge_after_block < self.config.transformer_blocks - 1:
            raise ValueError(
            f"merge_after_block must be in [0,{self.config.transformer_blocks - 2}]"
            )

    def train(self, mode: bool = True) -> "DINOv3Encoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward_features(self, images: Tensor) -> tuple[Tensor, ...]:
        expected = (self.config.encoder_channels, *self.config.input_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"expected adapted images [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"received {tuple(images.shape)}"
            )
        if self.merge_method is not None:
            features, _ = self.forward_features_with_merge_state(images)
            return features
        # Deliberately not wrapped in no_grad: gradients must reach the adapter.
        outputs = self.backbone.get_intermediate_layers(
            images,
            n=list(self.config.feature_blocks),
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        features = tuple(outputs)
        if len(features) != len(self.config.feature_blocks):
            raise RuntimeError(
                f"DINOv3 returned {len(features)} features; expected "
                f"{len(self.config.feature_blocks)}"
            )
        expected_feature = (
            images.shape[0],
            self.config.embedding_dim,
            *self.config.feature_grid,
        )
        for block, feature in zip(self.config.feature_blocks, features):
            if tuple(feature.shape) != expected_feature:
                raise RuntimeError(
                    f"DINOv3 block {block} feature shape {tuple(feature.shape)} "
                    f"!= {expected_feature}"
                )
        return features

    def forward_features_with_merge_state(
        self, images: Tensor
    ) -> tuple[tuple[Tensor, ...], TokenMergeState | None]:
        """Return dense tap maps and the state used for any inter-block merge."""
        expected = (self.config.encoder_channels, *self.config.input_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"expected adapted images [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"received {tuple(images.shape)}"
            )
        if self.merge_method is None:
            return self.forward_features(images), None

        tokens, grid_size = self.backbone.prepare_tokens_with_masks(images)
        grid_size = tuple(grid_size)
        if len(grid_size) != 2 or grid_size != self.config.feature_grid:
            raise RuntimeError(
                f"DINOv3 patch grid {grid_size} != expected {self.config.feature_grid}"
            )
        special_count = 1 + int(self.backbone.n_storage_tokens)
        if int(self.backbone.n_storage_tokens) != self.config.storage_tokens:
            raise RuntimeError("DINOv3 storage-token count changed after contract validation")
        if tokens.shape[1] != special_count + grid_size[0] * grid_size[1]:
            raise RuntimeError("DINOv3 prepared sequence has an unexpected token count")

        rope_embedding = self.backbone.rope_embed
        rope = rope_embedding(H=grid_size[0], W=grid_size[1])
        merge_state: TokenMergeState | None = None
        feature_outputs: list[Tensor] = []
        tap_blocks = set(self.config.feature_blocks)

        for block_index, block in enumerate(self.backbone.blocks):
            if block_index == self.merge_after_block + 1:
                special_tokens = tokens[:, :special_count]
                patch_tokens = tokens[:, special_count:]
                merged_tokens, merge_state = merge_tokens(
                    patch_tokens,
                    grid_size=grid_size,
                    merge_ratio=self.merge_ratio,
                    method=self.merge_method,
                )
                tokens = torch.cat((special_tokens, merged_tokens), dim=1)
                if merged_tokens.shape[1] == grid_size[0] * grid_size[1]:
                    rope = rope_embedding(H=grid_size[0], W=grid_size[1])
                else:
                    if merge_state.merged_coordinates is None:
                        raise RuntimeError("merger did not provide merged spatial coordinates")
                    rope = _rope_for_merged_coordinates(
                        rope_embedding, merge_state.merged_coordinates, grid_size
                    )

            tokens = block(tokens, rope)
            if block_index in tap_blocks:
                patches = tokens[:, special_count:]
                if merge_state is not None:
                    patches = reconstruct_tokens(patches, merge_state)
                patches = self.backbone.norm(patches)
                feature = patches.reshape(
                    patches.shape[0], grid_size[0], grid_size[1], patches.shape[-1]
                ).permute(0, 3, 1, 2).contiguous()
                feature_outputs.append(feature)

        if len(feature_outputs) != len(self.config.feature_blocks):
            raise RuntimeError(
                f"blockwise encoder returned {len(feature_outputs)} taps; "
                f"expected {len(self.config.feature_blocks)}"
            )
        expected_feature = (
            images.shape[0],
            self.config.embedding_dim,
            *self.config.feature_grid,
        )
        for block, feature in zip(self.config.feature_blocks, feature_outputs):
            if tuple(feature.shape) != expected_feature:
                raise RuntimeError(
                    f"DINOv3 block {block} feature shape {tuple(feature.shape)} "
                    f"!= {expected_feature}"
                )
        return tuple(feature_outputs), merge_state

    def forward(self, images: Tensor) -> tuple[Tensor, ...]:
        return self.forward_features(images)


class RefinementBlock(nn.Module):
    """Bilinear 2x upsampling followed by convolution, GroupNorm, and GELU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.normalization = nn.GroupNorm(8, out_channels)
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = F.interpolate(inputs, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.activation(self.normalization(self.convolution(inputs)))


class DINOv3SegmentationDecoder(nn.Module):
    """Fuse four same-grid DINOv3 features and emit WT/TC/ET logits."""

    def __init__(
        self,
        config: DINOv3SegmentationConfig = DINOv3SegmentationConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.lateral = nn.ModuleList(
            nn.Conv2d(config.embedding_dim, config.lateral_channels, kernel_size=1)
            for _ in config.feature_blocks
        )
        fused_channels = config.lateral_channels * len(config.feature_blocks)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, config.lateral_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, config.lateral_channels),
            nn.GELU(),
        )
        stage_inputs = (config.lateral_channels, *config.decoder_channels[:-1])
        self.refinement = nn.ModuleList(
            RefinementBlock(in_channels, out_channels)
            for in_channels, out_channels in zip(stage_inputs, config.decoder_channels)
        )
        self.prediction = nn.Conv2d(config.decoder_channels[-1], config.output_channels, 1)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != len(self.lateral):
            raise ValueError(f"expected four DINOv3 feature maps, received {len(features)}")
        batch_size = features[0].shape[0]
        expected = (batch_size, self.config.embedding_dim, *self.config.feature_grid)
        for index, feature in enumerate(features):
            if tuple(feature.shape) != expected:
                raise ValueError(
                    f"feature {index} shape {tuple(feature.shape)} != expected {expected}"
                )
        fused = self.fusion(
            torch.cat(
                [projection(feature) for projection, feature in zip(self.lateral, features)],
                dim=1,
            )
        )
        for block in self.refinement:
            fused = block(fused)
        logits = self.prediction(fused)
        expected_logits = (batch_size, self.config.output_channels, *self.config.input_size)
        if tuple(logits.shape) != expected_logits:
            raise RuntimeError(f"decoder logits shape {tuple(logits.shape)} != {expected_logits}")
        return logits


class BraTSDINOv3Segmenter(nn.Module):
    """Compose the MRI adapter, frozen DINOv3 encoder, and dense decoder.

    Inter-block token merging is opt-in; the default preserves the audited
    baseline execution path.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: DINOv3SegmentationConfig = DINOv3SegmentationConfig(),
        *,
        merge_method: str | None = None,
        merge_ratio: float = 0.5,
        merge_after_block: int = 2,
    ) -> None:
        super().__init__()
        self.config = config
        self.adapter = MRIInputAdapter()
        self.encoder = DINOv3Encoder(
            backbone,
            config,
            merge_method=merge_method,
            merge_ratio=merge_ratio,
            merge_after_block=merge_after_block,
        )
        self.decoder = DINOv3SegmentationDecoder(config)

    def train(self, mode: bool = True) -> "BraTSDINOv3Segmenter":
        super().train(mode)
        self.encoder.backbone.eval()
        return self

    def extract_features(self, images: Tensor) -> tuple[Tensor, ...]:
        """Return four dense feature maps, reconstructed if inter-block merging is enabled."""

        expected = (self.config.input_channels, *self.config.input_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"expected images [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"received {tuple(images.shape)}"
            )
        return self.encoder.forward_features(self.adapter(images))

    def extract_features_with_merge_state(
        self, images: Tensor
    ) -> tuple[tuple[Tensor, ...], TokenMergeState | None]:
        """Return dense tap maps and this forward's realized merge state."""
        expected = (self.config.input_channels, *self.config.input_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"expected images [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"received {tuple(images.shape)}"
            )
        return self.encoder.forward_features_with_merge_state(self.adapter(images))

    def forward_with_metadata(self, images: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """Run segmentation and return realized token/merge metadata for this pass."""
        features, state = self.extract_features_with_merge_state(images)
        logits = self.decode_features(features)
        original_patch_tokens = self.config.feature_grid[0] * self.config.feature_grid[1]
        special_tokens = self.encoder.special_token_count
        achieved_patch_tokens = (
            state.merged_count if state is not None else original_patch_tokens
        )
        retained = state.retention_metrics if state is not None else {
            "original_token_count": original_patch_tokens,
            "requested_token_count": original_patch_tokens,
            "achieved_token_count": original_patch_tokens,
            "requested_retention_ratio": 1.0,
            "achieved_retention_ratio": 1.0,
            "retention_ratio_error": 0.0,
        }
        merge_enabled = self.encoder.merge_method is not None
        merge_after_block = self.encoder.merge_after_block if merge_enabled else None
        block_token_counts = [
            (
                original_patch_tokens + special_tokens
                if not merge_enabled or block_index <= self.encoder.merge_after_block
                else achieved_patch_tokens + special_tokens
            )
            for block_index in range(self.config.transformer_blocks)
        ]
        metadata: dict[str, Any] = {
            "merge_method": self.encoder.merge_method,
            "merge_ratio": self.encoder.merge_ratio if merge_enabled else None,
            "merge_after_block": merge_after_block,
            "first_merged_block": (
                merge_after_block + 1 if merge_after_block is not None else None
            ),
            "feature_blocks": list(self.config.feature_blocks),
            "special_token_count": special_tokens,
            "original_patch_token_count": original_patch_tokens,
            "requested_patch_token_count": retained["requested_token_count"],
            "achieved_patch_token_count": achieved_patch_tokens,
            "original_sequence_token_count": original_patch_tokens + special_tokens,
            "merged_sequence_token_count": achieved_patch_tokens + special_tokens,
            "block_token_counts": block_token_counts,
            "retention_metrics": retained,
        }
        return logits, metadata

    def decode_features(self, features: Sequence[Tensor]) -> Tensor:
        """Decode the four dense feature maps emitted by the encoder."""

        return self.decoder(features)

    def forward(self, images: Tensor) -> Tensor:
        return self.decode_features(self.extract_features(images))


def _repository_head(repository: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"unable to verify DINOv3 repository revision: {repository}") from error
    return result.stdout.strip()


def load_official_dinov3_vitb16(
    repository: str | Path,
    checkpoint: str | Path,
    config: DINOv3SegmentationConfig = DINOv3SegmentationConfig(),
) -> nn.Module:
    """Load the authorized official checkpoint with strict path/revision checks."""

    repository_path = Path(repository).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not repository_path.is_dir() or not (repository_path / ".git").exists():
        raise FileNotFoundError(f"official DINOv3 Git checkout not found: {repository_path}")
    actual_revision = _repository_head(repository_path)
    if actual_revision != config.dinov3_revision:
        raise RuntimeError(
            f"DINOv3 revision mismatch: expected {config.dinov3_revision}, "
            f"received {actual_revision}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"approved DINOv3 checkpoint not found: {checkpoint_path}")
    if checkpoint_path.name != config.checkpoint_filename:
        raise ValueError(
            f"expected checkpoint filename {config.checkpoint_filename!r}, "
            f"received {checkpoint_path.name!r}"
        )
    try:
        return torch.hub.load(
            str(repository_path),
            DINOV3_VITB16_HUB_ENTRY,
            source="local",
            weights=str(checkpoint_path),
            check_hash=True,
        )
    except Exception as error:
        raise RuntimeError(
            "failed to load the authorized DINOv3 ViT-B/16 checkpoint; "
            "no random-weight fallback was used"
        ) from error


def build_pretrained_brats_dinov3(
    repository: str | Path,
    checkpoint: str | Path,
    config: DINOv3SegmentationConfig = DINOv3SegmentationConfig(),
    *,
    merge_method: str | None = None,
    merge_ratio: float = 0.5,
    merge_after_block: int = 2,
) -> BraTSDINOv3Segmenter:
    """Build the baseline or opt-in merging model from an authorized checkpoint."""

    backbone = load_official_dinov3_vitb16(repository, checkpoint, config)
    return BraTSDINOv3Segmenter(
        backbone,
        config,
        merge_method=merge_method,
        merge_ratio=merge_ratio,
        merge_after_block=merge_after_block,
    )
