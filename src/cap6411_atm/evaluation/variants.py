"""Consistent construction of baseline and token-merging evaluation variants."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from torch import nn

from cap6411_atm.models import (
    BraTSDINOv3Segmenter,
    DINOv3SegmentationConfig,
    build_pretrained_brats_dinov3,
    load_adapter_decoder_artifact,
)

from .ablation import VariantSpec


def _variant_options(spec: VariantSpec) -> dict[str, object]:
    if spec.method == "baseline":
        if spec.retention_ratio not in (None, 0.0, 1.0):
            raise ValueError("baseline variants cannot request token retention")
        return {"merge_method": None}
    if spec.method not in ("uniform", "adaptive"):
        raise ValueError(f"unsupported segmentation variant method: {spec.method!r}")
    if spec.retention_ratio is None:
        raise ValueError(f"{spec.method} variants require retention_ratio")
    options: dict[str, object] = {
        "merge_method": spec.method,
        "retention_ratio": spec.retention_ratio,
    }
    options["merge_after_block"] = int(spec.params.get("merge_after_block", 2))
    unknown = set(spec.params) - {"merge_after_block"}
    if unknown:
        raise ValueError(f"unsupported variant parameters: {sorted(unknown)}")
    return options


def build_model_variant(
    spec: VariantSpec,
    backbone_factory: Callable[[], nn.Module],
    *,
    config: DINOv3SegmentationConfig = DINOv3SegmentationConfig(),
    artifact: str | Path | None = None,
) -> BraTSDINOv3Segmenter:
    """Build one independent synthetic/local-backbone variant.

    A fresh backbone is required for every call so variants cannot mutate or
    relocate one another during WS3 benchmarking. The same optional sanitized
    adapter/decoder artifact is loaded under the repository's existing checks.
    """
    model = BraTSDINOv3Segmenter(backbone_factory(), config, **_variant_options(spec))
    if artifact is not None:
        load_adapter_decoder_artifact(model, artifact)
    return model


def build_pretrained_model_variant(
    spec: VariantSpec,
    repository: str | Path,
    checkpoint: str | Path,
    artifact: str | Path,
    *,
    config: DINOv3SegmentationConfig = DINOv3SegmentationConfig(),
) -> BraTSDINOv3Segmenter:
    """Build one official-weight variant and load the common sanitized artifact."""
    model = build_pretrained_brats_dinov3(
        repository, checkpoint, config, **_variant_options(spec)
    )
    load_adapter_decoder_artifact(model, artifact)
    return model

