"""Reusable Colab-oriented training pipeline for the frozen DINOv3 baseline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from cap6411_atm.data.dataset import BraTSAxialSliceDataset
from cap6411_atm.models import (
    BraTSDINOv3Segmenter,
    DINOv3SegmentationConfig,
    build_pretrained_brats_dinov3,
)

from .losses import MultilabelBCEDiceLoss
from .metrics import StreamingDice
from .sampling import PatientLocalBatchSampler, PatientSubset, select_pilot_patients


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Complete configuration for a reproducible baseline run."""

    archive_path: str
    manifest_path: str
    inspection_config: str
    model_config: str
    dinov3_repository: str
    checkpoint_path: str
    output_dir: str
    experiment_name: str = "dinov3_vitb16_brats_baseline"
    seed: int = 42
    epochs: int = 30
    batch_size: int = 2
    num_workers: int = 0
    max_cached_patients: int = 1
    cache_dir: str = "/content/brats_cache"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    dice_smooth: float = 1e-6
    validation_threshold: float = 0.5
    amp: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0
    pilot_enabled: bool = True
    pilot_train_patients: int = 8
    pilot_validation_patients: int = 4
    resume_checkpoint: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "BaselineTrainingConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("epochs/batch_size must be positive and workers non-negative")
        if self.max_cached_patients < 1:
            raise ValueError("max_cached_patients must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or null")
        if self.pilot_train_patients < 1 or self.pilot_validation_patients < 1:
            raise ValueError("pilot patient counts must be positive")

    @property
    def scope(self) -> str:
        return "pilot" if self.pilot_enabled else "full_cohort"


@dataclass
class TrainingComponents:
    model: BraTSDINOv3Segmenter
    train_loader: DataLoader
    validation_loader: DataLoader
    train_sampler: PatientLocalBatchSampler
    validation_sampler: PatientLocalBatchSampler
    optimizer: torch.optim.Optimizer
    criterion: MultilabelBCEDiceLoss
    device: torch.device
    amp_enabled: bool
    scaler: Any
    dataset_metadata: dict[str, Any]


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing slow deterministic kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return adapter/decoder parameters and reject accidental backbone unfreezing."""

    selected: list[nn.Parameter] = []
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if name.startswith(("adapter.", "decoder.")):
                selected.append(parameter)
            else:
                unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {unexpected}")
    if not selected:
        raise RuntimeError("model has no trainable adapter/decoder parameters")
    return selected


def _make_scaler(device: torch.device, enabled: bool) -> Any:
    use_amp = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)  # pragma: no cover - old PyTorch


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def _loader(
    dataset,
    sampler: PatientLocalBatchSampler,
    *,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def build_training_components(config: BaselineTrainingConfig) -> TrainingComponents:
    """Construct isolated train/validation datasets, model, loss, and optimizer."""

    set_deterministic_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = DINOv3SegmentationConfig.from_json(config.model_config)
    model = build_pretrained_brats_dinov3(
        config.dinov3_repository,
        config.checkpoint_path,
        model_config,
    ).to(device)

    cache_root = Path(config.cache_dir)
    train_full = BraTSAxialSliceDataset(
        config.archive_path,
        config.manifest_path,
        "train",
        config.inspection_config,
        cache_dir=cache_root / "train",
        max_cached_patients=config.max_cached_patients,
    )
    validation_full = BraTSAxialSliceDataset(
        config.archive_path,
        config.manifest_path,
        "validation",
        config.inspection_config,
        cache_dir=cache_root / "validation",
        max_cached_patients=config.max_cached_patients,
    )
    if set(train_full.patient_ids) & set(validation_full.patient_ids):
        raise RuntimeError("train and validation patient sets overlap")

    if config.pilot_enabled:
        train_ids = select_pilot_patients(
            train_full.patient_ids, config.pilot_train_patients, seed=config.seed
        )
        validation_ids = select_pilot_patients(
            validation_full.patient_ids,
            config.pilot_validation_patients,
            seed=config.seed + 1,
        )
        train_dataset = PatientSubset(train_full, train_ids)
        validation_dataset = PatientSubset(validation_full, validation_ids)
    else:
        train_ids = tuple(train_full.patient_ids)
        validation_ids = tuple(validation_full.patient_ids)
        train_dataset = train_full
        validation_dataset = validation_full

    train_sampler = PatientLocalBatchSampler(
        train_dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle_patients=True,
        shuffle_slices=True,
    )
    validation_sampler = PatientLocalBatchSampler(
        validation_dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle_patients=False,
        shuffle_slices=False,
    )
    train_loader = _loader(
        train_dataset,
        train_sampler,
        workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = _loader(
        validation_dataset,
        validation_sampler,
        workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = MultilabelBCEDiceLoss(
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
        smooth=config.dice_smooth,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.amp and device.type == "cuda"
    metadata = {
        "scope": config.scope,
        "pilot_metrics": config.pilot_enabled,
        "metrics_scope": (
            "deterministic_patient_subset" if config.pilot_enabled else "full_train_validation_splits"
        ),
        "train_patients": list(train_ids),
        "validation_patients": list(validation_ids),
        "train_patient_count": len(train_ids),
        "validation_patient_count": len(validation_ids),
        "full_train_patient_count": len(train_full.patient_ids),
        "full_validation_patient_count": len(validation_full.patient_ids),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "full_train_samples": len(train_full),
        "full_validation_samples": len(validation_full),
        "test_split_loaded": False,
        "modality_mapping": {
            "t1": "t1n",
            "t1ce": "t1c",
            "t2": "t2w",
            "flair": "t2f",
        },
    }
    return TrainingComponents(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_sampler=train_sampler,
        validation_sampler=validation_sampler,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
        scaler=_make_scaler(device, amp_enabled),
        dataset_metadata=metadata,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    return images, targets


def run_one_batch_smoke_test(
    components: TrainingComponents,
    config: BaselineTrainingConfig,
) -> dict[str, Any]:
    """Perform one real optimizer update and validate frozen-backbone invariants."""

    model = components.model
    model.train()
    batch = next(iter(components.train_loader))
    images, targets = _move_batch(batch, components.device)
    components.optimizer.zero_grad(set_to_none=True)
    if components.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(components.device)
    _synchronize(components.device)
    started = time.perf_counter()
    with _autocast(components.device, components.amp_enabled):
        logits = model(images)
        loss = components.criterion(logits, targets)
    components.scaler.scale(loss).backward()
    components.scaler.unscale_(components.optimizer)
    adapter_gradients = [p.grad for p in model.adapter.parameters()]
    decoder_gradients = [p.grad for p in model.decoder.parameters()]
    backbone_gradients = [p.grad for p in model.encoder.backbone.parameters()]
    if not all(gradient is not None and torch.isfinite(gradient).all() for gradient in adapter_gradients):
        raise RuntimeError("smoke test found missing/non-finite adapter gradients")
    if not all(gradient is not None and torch.isfinite(gradient).all() for gradient in decoder_gradients):
        raise RuntimeError("smoke test found missing/non-finite decoder gradients")
    if any(gradient is not None for gradient in backbone_gradients):
        raise RuntimeError("smoke test found gradients on the frozen backbone")
    if config.gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters(model), config.gradient_clip_norm
        )
    components.scaler.step(components.optimizer)
    components.scaler.update()
    _synchronize(components.device)
    elapsed = time.perf_counter() - started
    return {
        "status": "passed",
        "scope": config.scope,
        "warning": "smoke test performs one optimizer update; rebuild before full training",
        "loss": float(loss.detach()),
        "image_shape": list(images.shape),
        "target_shape": list(targets.shape),
        "logit_shape": list(logits.shape),
        "batch_samples": int(images.shape[0]),
        "elapsed_seconds": elapsed,
        "samples_per_second": float(images.shape[0] / elapsed),
        "device": str(components.device),
        "amp_enabled": components.amp_enabled,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(components.device))
            if components.device.type == "cuda"
            else None
        ),
    }


def train_one_epoch(
    components: TrainingComponents,
    config: BaselineTrainingConfig,
) -> dict[str, float]:
    model = components.model
    model.train()
    components.optimizer.zero_grad(set_to_none=True)
    total_loss = total_bce = total_dice = 0.0
    sample_count = 0
    batch_count = 0
    _synchronize(components.device)
    started = time.perf_counter()
    total_batches = len(components.train_loader)
    for batch_index, batch in enumerate(components.train_loader):
        images, targets = _move_batch(batch, components.device)
        with _autocast(components.device, components.amp_enabled):
            logits = model(images)
            losses = components.criterion.components(logits, targets)
            group_start = (
                batch_index // config.gradient_accumulation_steps
            ) * config.gradient_accumulation_steps
            group_size = min(
                config.gradient_accumulation_steps, total_batches - group_start
            )
            scaled_loss = losses["loss"] / group_size
        components.scaler.scale(scaled_loss).backward()
        should_step = (
            (batch_index + 1) % config.gradient_accumulation_steps == 0
            or batch_index + 1 == total_batches
        )
        if should_step:
            components.scaler.unscale_(components.optimizer)
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(model), config.gradient_clip_norm
                )
            components.scaler.step(components.optimizer)
            components.scaler.update()
            components.optimizer.zero_grad(set_to_none=True)
        count = int(images.shape[0])
        sample_count += count
        batch_count += 1
        total_loss += float(losses["loss"].detach()) * count
        total_bce += float(losses["bce"].detach()) * count
        total_dice += float(losses["dice"].detach()) * count
    _synchronize(components.device)
    elapsed = time.perf_counter() - started
    throughput = sample_count / elapsed
    return {
        "loss": total_loss / sample_count,
        "bce_loss": total_bce / sample_count,
        "dice_loss": total_dice / sample_count,
        "samples": sample_count,
        "batches": batch_count,
        "elapsed_seconds": elapsed,
        "samples_per_second": throughput,
        "estimated_full_epoch_seconds": (
            components.dataset_metadata["full_train_samples"] / throughput
        ),
    }


def validate_one_epoch(
    components: TrainingComponents,
    config: BaselineTrainingConfig,
) -> dict[str, Any]:
    model = components.model
    model.eval()
    metric = StreamingDice(config.validation_threshold)
    total_loss = 0.0
    sample_count = 0
    batch_count = 0
    _synchronize(components.device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in components.validation_loader:
            images, targets = _move_batch(batch, components.device)
            with _autocast(components.device, components.amp_enabled):
                logits = model(images)
                loss = components.criterion(logits, targets)
            metric.update(logits, targets)
            count = int(images.shape[0])
            sample_count += count
            batch_count += 1
            total_loss += float(loss.detach()) * count
    _synchronize(components.device)
    elapsed = time.perf_counter() - started
    throughput = sample_count / elapsed
    result = metric.compute()
    result.update(
        {
            "loss": total_loss / sample_count,
            "samples": sample_count,
            "batches": batch_count,
            "elapsed_seconds": elapsed,
            "samples_per_second": throughput,
            "estimated_full_epoch_seconds": (
                components.dataset_metadata["full_validation_samples"] / throughput
            ),
        }
    )
    return result


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_payload(
    components: TrainingComponents,
    config: BaselineTrainingConfig,
    *,
    epoch: int,
    best_macro_dice: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "epoch": epoch,
        "best_macro_dice": best_macro_dice,
        "model_state": components.model.state_dict(),
        "optimizer_state": components.optimizer.state_dict(),
        "scaler_state": components.scaler.state_dict(),
        "training_config": asdict(config),
        "model_config": asdict(components.model.config),
        "dataset_metadata": components.dataset_metadata,
        "history": history,
        "rng_state": _rng_state(),
        "environment": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(components.device),
        },
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_history(history: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resume_from_checkpoint(
    path: str | Path,
    components: TrainingComponents,
    config: BaselineTrainingConfig,
) -> tuple[int, float, list[dict[str, Any]]]:
    """Restore model, optimizer, scaler, history, and deterministic RNG state."""

    try:
        checkpoint = torch.load(path, map_location=components.device, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with PyTorch < 2.6
        checkpoint = torch.load(path, map_location=components.device)
    saved = checkpoint["training_config"]
    for key in (
        "seed",
        "pilot_enabled",
        "pilot_train_patients",
        "pilot_validation_patients",
        "manifest_path",
        "archive_path",
    ):
        if saved[key] != getattr(config, key):
            raise ValueError(f"resume configuration mismatch for {key}")
    if checkpoint["dataset_metadata"]["test_split_loaded"]:
        raise ValueError("refusing checkpoint that used the isolated test split")
    components.model.load_state_dict(checkpoint["model_state"])
    components.optimizer.load_state_dict(checkpoint["optimizer_state"])
    components.scaler.load_state_dict(checkpoint["scaler_state"])
    _restore_rng_state(checkpoint["rng_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_macro_dice"]),
        list(checkpoint["history"]),
    )


def fit(
    components: TrainingComponents,
    config: BaselineTrainingConfig,
) -> list[dict[str, Any]]:
    """Train and validate, saving atomic last/best checkpoints and JSON history."""

    output = Path(config.output_dir) / config.experiment_name
    output.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_macro_dice = float("-inf")
    history: list[dict[str, Any]] = []
    if config.resume_checkpoint:
        start_epoch, best_macro_dice, history = resume_from_checkpoint(
            config.resume_checkpoint, components, config
        )
    for epoch in range(start_epoch, config.epochs):
        components.train_sampler.set_epoch(epoch)
        train_result = train_one_epoch(components, config)
        validation_result = validate_one_epoch(components, config)
        record = {
            "epoch": epoch,
            "scope": config.scope,
            "pilot_metrics": config.pilot_enabled,
            "metrics_scope": components.dataset_metadata["metrics_scope"],
            "train": train_result,
            "validation": validation_result,
        }
        history.append(record)
        improved = validation_result["macro_dice"] > best_macro_dice
        if improved:
            best_macro_dice = float(validation_result["macro_dice"])
        payload = _checkpoint_payload(
            components,
            config,
            epoch=epoch,
            best_macro_dice=best_macro_dice,
            history=history,
        )
        _atomic_torch_save(payload, output / "last.pt")
        if improved:
            _atomic_torch_save(payload, output / "best.pt")
        _write_history(history, output / "history.json")
        print(json.dumps(record, sort_keys=True))
    return history


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = BaselineTrainingConfig.from_json(args.config)
    if args.resume is not None:
        values = asdict(config)
        values["resume_checkpoint"] = str(args.resume)
        config = BaselineTrainingConfig(**values)
    components = build_training_components(config)
    if args.smoke_test:
        print(json.dumps(run_one_batch_smoke_test(components, config), indent=2))
    else:
        fit(components, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
