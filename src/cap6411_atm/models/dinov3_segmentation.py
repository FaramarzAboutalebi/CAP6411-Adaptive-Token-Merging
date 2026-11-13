"""Frozen DINOv3 ViT-B/16 baseline for 2D BraTS segmentation.

The production factory requires an authorized local DINOv3 checkout and
checkpoint.  Tests may inject a synthetic backbone with the same documented
``get_intermediate_layers`` interface; the module never silently substitutes
random weights for the requested pretrained model.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


class DINOv3Encoder(nn.Module):
    """Freeze DINOv3 parameters while preserving input-adapter autograd.

    The wrapper calls the official patch-only intermediate-feature API, so the
    class token and four storage tokens participate inside the transformer but
    are not returned to the dense decoder.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: DINOv3SegmentationConfig = DINOv3SegmentationConfig(),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self._validate_backbone_contract()
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
    """Compose the MRI adapter, frozen DINOv3 encoder, and dense decoder."""

    def __init__(
        self,
        backbone: nn.Module,
        config: DINOv3SegmentationConfig = DINOv3SegmentationConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.adapter = MRIInputAdapter()
        self.encoder = DINOv3Encoder(backbone, config)
        self.decoder = DINOv3SegmentationDecoder(config)

    def train(self, mode: bool = True) -> "BraTSDINOv3Segmenter":
        super().train(mode)
        self.encoder.backbone.eval()
        return self

    def extract_features(self, images: Tensor) -> tuple[Tensor, ...]:
        """Return four dense maps for decoding or future reconstruction hooks."""

        expected = (self.config.input_channels, *self.config.input_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"expected images [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"received {tuple(images.shape)}"
            )
        return self.encoder.forward_features(self.adapter(images))

    def decode_features(self, features: Sequence[Tensor]) -> Tensor:
        """Decode dense spatial maps; future mergers must reconstruct before this boundary."""

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
) -> BraTSDINOv3Segmenter:
    """Build the audited baseline from an authorized official checkpoint."""

    backbone = load_official_dinov3_vitb16(repository, checkpoint, config)
    return BraTSDINOv3Segmenter(backbone, config)
