"""Reduced-input-resolution baseline for the frozen DINOv3 BraTS segmenter.

The audited baseline only accepts 240x240 slices and its decoder expects a
15x15 feature grid, so lower resolutions are implemented as a wrapper instead
of a configuration change:

    images [B,4,240,240] -> resize to r x r -> trained MRI adapter
      -> frozen DINOv3 at r x r ((r/16)^2 patch tokens)
      -> four feature maps [B,768,r/16,r/16] -> bilinear resize to 15x15
      -> trained decoder (unchanged) -> logits [B,3,240,240]

The encoder genuinely processes fewer tokens (DINOv3's rotary position
embeddings accept any grid size), and the trained Workstream 1 decoder is
reused without retraining.  At r == 240 the wrapper is exactly the baseline.
"""

from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn
from torch.nn import functional as F

from cap6411_atm.models import BraTSDINOv3Segmenter


DEFAULT_RESOLUTIONS = (240, 224, 208, 192, 176, 160, 144, 128)


class ReducedResolutionSegmenter(nn.Module):
    """Run a trained ``BraTSDINOv3Segmenter`` at a lower encoder resolution."""

    def __init__(
        self,
        model: BraTSDINOv3Segmenter,
        size: int,
        *,
        antialias: bool = True,
    ) -> None:
        super().__init__()
        patch = model.config.patch_size
        full = model.config.input_size
        if size < patch or size % patch:
            raise ValueError(f"resolution must be a positive multiple of {patch}, received {size}")
        if size > min(full):
            raise ValueError(f"resolution {size} exceeds the {full[0]}x{full[1]} baseline input")
        self.base = model
        self.size = int(size)
        self.antialias = antialias

    @property
    def config(self):
        return self.base.config

    @property
    def is_identity(self) -> bool:
        return (self.size, self.size) == tuple(self.config.input_size)

    @property
    def token_count(self) -> int:
        """Spatial patch tokens seen by the encoder (theoretical, not measured)."""

        return (self.size // self.config.patch_size) ** 2

    def downsample(self, images: Tensor) -> Tensor:
        expected = (self.config.input_channels, *self.config.input_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"expected images [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"received {tuple(images.shape)}"
            )
        if self.is_identity:
            return images
        return F.interpolate(
            images,
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False,
            antialias=self.antialias,
        )

    def encode_adapted(self, adapted: Tensor) -> tuple[Tensor, ...]:
        """Encode adapter output at ``size`` and resample features to the decoder grid."""

        if self.is_identity:
            return self.base.encoder(adapted)
        # Same call as DINOv3Encoder.forward_features, which hard-codes a 240x240
        # input check.  Keep the arguments in sync with that method (a test
        # asserts they match); letting forward_features take an expected grid
        # would remove this copy, but that is a Workstream 1 interface change.
        features = tuple(
            self.base.encoder.backbone.get_intermediate_layers(
                adapted,
                n=list(self.config.feature_blocks),
                reshape=True,
                return_class_token=False,
                return_extra_tokens=False,
                norm=True,
            )
        )
        # Same checks as DINOv3Encoder.forward_features, at the reduced grid.
        if len(features) != len(self.config.feature_blocks):
            raise RuntimeError(
                f"DINOv3 returned {len(features)} features; expected "
                f"{len(self.config.feature_blocks)}"
            )
        grid = self.size // self.config.patch_size
        expected = (adapted.shape[0], self.config.embedding_dim, grid, grid)
        for feature in features:
            if tuple(feature.shape) != expected:
                raise RuntimeError(f"feature shape {tuple(feature.shape)} != {expected}")
        return tuple(
            F.interpolate(
                feature,
                size=self.config.feature_grid,
                mode="bilinear",
                align_corners=False,
            )
            for feature in features
        )

    def extract_features(self, images: Tensor) -> tuple[Tensor, ...]:
        return self.encode_adapted(self.base.adapter(self.downsample(images)))

    def decode_features(self, features: Sequence[Tensor]) -> Tensor:
        return self.base.decode_features(features)

    def forward(self, images: Tensor) -> Tensor:
        return self.decode_features(self.extract_features(images))


def resolution_sweep(
    model: BraTSDINOv3Segmenter,
    sizes: Sequence[int] = DEFAULT_RESOLUTIONS,
) -> dict[str, ReducedResolutionSegmenter]:
    """Return ``{"res240": ..., "res224": ...}`` wrappers sharing one model."""

    return {f"res{size}": ReducedResolutionSegmenter(model, size) for size in sizes}
