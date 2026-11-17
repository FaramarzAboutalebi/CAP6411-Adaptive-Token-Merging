"""Segmentation evaluation utilities using the shared BraTS Dice protocol."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from cap6411_atm.training.metrics import StreamingDice


def _model_device(model: nn.Module) -> torch.device:
    tensor = next(iter(model.parameters()), None)
    if tensor is None:
        tensor = next(iter(model.buffers()), None)
    return tensor.device if tensor is not None else torch.device("cpu")


def _batch_tensors(batch: Any) -> tuple[Tensor, Tensor]:
    if isinstance(batch, Mapping):
        try:
            images, targets = batch["image"], batch["target"]
        except KeyError as error:
            raise ValueError("mapping batches must contain 'image' and 'target'") from error
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        images, targets = batch
    else:
        raise TypeError("each evaluation batch must be an image/target pair or dataset mapping")
    if not isinstance(images, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("evaluation images and targets must be tensors")
    if images.ndim != 4 or targets.ndim != 4:
        raise ValueError("evaluation images and targets must have shape [B,C,H,W]")
    return images, targets


def evaluate_segmentation(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    device: torch.device | str | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate global WT/TC/ET Dice over batches with `StreamingDice`.

    Batches may be `(images, targets)` pairs or the BraTS dataset mapping with
    `image` and `target` keys. The function uses inference mode and restores the
    model's original training/evaluation mode after evaluation.
    """
    evaluation_device = torch.device(device) if device is not None else _model_device(model)
    metric = StreamingDice(threshold=threshold)
    was_training = model.training
    batch_count = 0
    sample_count = 0
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                images, targets = _batch_tensors(batch)
                images = images.to(evaluation_device, non_blocking=evaluation_device.type == "cuda")
                targets = targets.to(
                    evaluation_device, non_blocking=evaluation_device.type == "cuda"
                )
                logits = model(images)
                if not isinstance(logits, Tensor):
                    raise TypeError("segmentation model must return a logits tensor")
                if logits.shape != targets.shape:
                    raise ValueError(
                        f"logits shape {tuple(logits.shape)} does not match target shape "
                        f"{tuple(targets.shape)}"
                    )
                if not torch.isfinite(logits).all():
                    raise RuntimeError("model produced non-finite segmentation logits")
                metric.update(logits, targets)
                batch_count += 1
                sample_count += int(images.shape[0])
    finally:
        model.train(was_training)

    if batch_count == 0:
        raise ValueError("cannot evaluate an empty batch iterable")
    result = metric.compute()
    result["batches"] = batch_count
    result["samples"] = sample_count
    result["device"] = str(evaluation_device)
    result["threshold"] = threshold
    return result


__all__ = ["evaluate_segmentation"]
