"""Reduced-input-resolution baseline for the frozen DINOv3 BraTS segmenter.

The audited baseline only accepts 240x240 slices and its decoder expects a
15x15 feature grid, so lower resolutions are implemented as a wrapper instead
of a configuration change:

    images [B,4,240,240] -> resize to r x r -> trained MRI adapter
      -> frozen DINOv3 at r x r ((r/16)^2 patch tokens)
      -> four feature maps [B,768,r/16,r/16] -> bilinear resize to 15x15 (as one matmul)
      -> trained decoder (unchanged) -> logits [B,3,240,240]

The encoder genuinely processes fewer tokens (DINOv3's rotary position
embeddings accept any grid size), and the trained Workstream 1 decoder is
reused without retraining.  At r == 240 the wrapper is exactly the baseline.

The feature resize is applied as a precomputed linear map instead of
``F.interpolate``: PyTorch's bilinear kernel parallelizes over output pixels, so
a tiny 15x15 output with 768 channels leaves the GPU idle (measured on an RTX
5090 at batch 32: 13.5 ms for the four resizes vs 5.9 ms for the whole 160px
encoder).  The matrix is built by running ``F.interpolate`` on a basis, so the
result is the same bilinear interpolation up to floating-point rounding.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cap6411_atm.models import BraTSDINOv3Segmenter


DEFAULT_RESOLUTIONS = (240, 224, 208, 192, 176, 160, 144, 128)

_RESIZE_MATRICES: dict[tuple, Tensor] = {}


def bilinear_resize_matrix(
    source: tuple[int, int],
    target: tuple[int, int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``[target_h*target_w, source_h*source_w]`` matrix equal to bilinear resize.

    Built once per (grid, device, dtype) by resizing the standard basis with
    ``F.interpolate(mode="bilinear", align_corners=False)``, then cached.
    """

    key = (tuple(source), tuple(target), torch.device(device), dtype)
    matrix = _RESIZE_MATRICES.get(key)
    if matrix is None:
        size = source[0] * source[1]
        with torch.no_grad():
            basis = torch.eye(size, dtype=torch.float64).reshape(size, 1, *source)
            resized = F.interpolate(basis, size=target, mode="bilinear", align_corners=False)
            matrix = resized.reshape(size, -1).T.contiguous().to(device=device, dtype=dtype)
        _RESIZE_MATRICES[key] = matrix
    return matrix


def resize_feature_map(feature: Tensor, target: tuple[int, int]) -> Tensor:
    """Bilinear resize of ``[B,C,H,W]`` to ``[B,C,*target]`` as a single matmul.

    The matmul always runs in at least fp32 with autocast disabled, matching
    ``F.interpolate`` (which autocast leaves in the input dtype).  Otherwise
    fp16 autocast would round reduced-resolution features that the 240px
    baseline never resizes, biasing the Dice comparison.  The result keeps
    the input dtype.
    """

    batch, channels, height, width = feature.shape
    if (height, width) == tuple(target):
        return feature
    compute_dtype = torch.promote_types(feature.dtype, torch.float32)
    matrix = bilinear_resize_matrix(
        (height, width), target, device=feature.device, dtype=compute_dtype
    )
    flat = feature.reshape(batch, channels, height * width).to(compute_dtype)
    with torch.autocast(device_type=feature.device.type, enabled=False):
        resized = flat @ matrix.T
    return resized.to(feature.dtype).reshape(batch, channels, *target)


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
        return tuple(resize_feature_map(feature, self.config.feature_grid) for feature in features)

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
