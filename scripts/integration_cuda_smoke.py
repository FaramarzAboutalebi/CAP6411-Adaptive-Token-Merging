"""CUDA smoke test for integrated WS2/WS3/WS4 using authorized private weights.

This script uses seeded synthetic images and targets only. It does not load a
BraTS archive, retrain any component, or produce cohort performance results.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from cap6411_atm.benchmark import run_suite
from cap6411_atm.evaluation import (
    VariantSpec,
    build_pretrained_model_variant,
    evaluate_model,
)
from cap6411_atm.models import DINOv3SegmentationConfig


INTEGRATION_BASE_COMMIT = "34222fb1696c74bb2b346710b2edd270e62afd1b"
ARTIFACT_SHA256 = "ed0d84caf6bb6d4dcb4ee2dc29dc911c628406204e9683e7cde3b280b796314b"
EXPECTED_SHAPE = (1, 3, 240, 240)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_integration_revision(repository: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            INTEGRATION_BASE_COMMIT,
            head,
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"repository HEAD {head} does not contain integration commit "
            f"{INTEGRATION_BASE_COMMIT}"
        )
    return head


def _trainable_fingerprint(model: torch.nn.Module) -> str:
    """Hash the adapter/decoder weights loaded from the common WS1 artifact."""
    digest = hashlib.sha256()
    for prefix, module in (("adapter", model.adapter), ("decoder", model.decoder)):
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(f"{prefix}.{name}".encode())
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class _SyntheticValidationDataset(Dataset):
    """One complete one-slice synthetic validation volume for WS4 plumbing."""

    split = "validation"
    axial_depth = 1
    patient_ids = ("synthetic-smoke",)
    samples = (("synthetic-smoke", 0),)

    def __init__(self, image: torch.Tensor, target: torch.Tensor) -> None:
        self.image = image.squeeze(0).cpu()
        self.target = target.squeeze(0).cpu()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index != 0:
            raise IndexError(index)
        return {
            "image": self.image,
            "target": self.target,
            "patient_id": "synthetic-smoke",
            "slice_index": 0,
            "split": "validation",
        }


def _build(
    spec: VariantSpec,
    *,
    dinov3_repository: Path,
    checkpoint: Path,
    artifact: Path,
    config: DINOv3SegmentationConfig,
) -> torch.nn.Module:
    model = build_pretrained_model_variant(
        spec,
        dinov3_repository,
        checkpoint,
        artifact,
        config=config,
    )
    if model.encoder.backbone.training:
        raise AssertionError("frozen DINOv3 backbone is not in evaluation mode")
    if any(parameter.requires_grad for parameter in model.encoder.backbone.parameters()):
        raise AssertionError("DINOv3 backbone is not frozen")
    return model.eval()


def _check_forward(
    name: str,
    model: torch.nn.Module,
    image: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.inference_mode():
        logits, metadata = model.forward_with_metadata(image)
    if tuple(logits.shape) != EXPECTED_SHAPE:
        raise AssertionError(f"{name}: logits shape {tuple(logits.shape)} != {EXPECTED_SHAPE}")
    if not torch.isfinite(logits).all():
        raise AssertionError(f"{name}: non-finite logits")
    return logits.detach().cpu(), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dinov3-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("PASS requires a CUDA GPU; torch.cuda.is_available() is False")
    if args.warmup < 1 or args.iterations < 2:
        raise ValueError("use at least one warmup and two measured iterations")
    for path in (args.dinov3_repo, args.checkpoint, args.artifact):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.checkpoint.name != "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth":
        raise ValueError("unexpected DINOv3 checkpoint filename")

    head = _assert_integration_revision(args.repo_root.resolve())
    artifact_hash = _sha256(args.artifact)
    if artifact_hash != ARTIFACT_SHA256:
        raise RuntimeError(f"WS1 artifact checksum mismatch: {artifact_hash}")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(6411)
    torch.cuda.manual_seed_all(6411)
    image_cpu = torch.randn(1, 4, 240, 240)
    target_cpu = torch.zeros(1, 3, 240, 240)
    image = image_cpu.to(device)
    config = DINOv3SegmentationConfig.from_json(
        args.repo_root / "configs" / "dinov3_vitb16_brats.json"
    )

    build_args = {
        "dinov3_repository": args.dinov3_repo,
        "checkpoint": args.checkpoint,
        "artifact": args.artifact,
        "config": config,
    }
    specs = {
        "baseline": VariantSpec("baseline", method="baseline"),
        "uniform_r0.50": VariantSpec(
            "uniform_r0.50", method="uniform", retention_ratio=0.5
        ),
        "adaptive_r0.50": VariantSpec(
            "adaptive_r0.50", method="adaptive", retention_ratio=0.5
        ),
    }

    print("CUDA:", torch.cuda.get_device_name(device))
    print("PyTorch:", torch.__version__)
    print("Integration HEAD:", head)
    print("WS1 artifact SHA-256: OK")

    # Build the three experiment models independently from the same two weight files.
    models = {name: _build(spec, **build_args) for name, spec in specs.items()}
    fingerprints = {name: _trainable_fingerprint(model) for name, model in models.items()}
    if len(set(fingerprints.values())) != 1:
        raise AssertionError("variants did not load identical adapter/decoder weights")
    print("Strict common adapter/decoder loading: OK", next(iter(fingerprints.values()))[:16])

    # Baseline and positive-ratio forward contracts.
    forward_metadata: dict[str, dict[str, Any]] = {}
    baseline = models["baseline"].to(device)
    baseline_logits, forward_metadata["baseline"] = _check_forward(
        "baseline", baseline, image
    )
    baseline.to("cpu")
    torch.cuda.empty_cache()

    for name in ("uniform_r0.50", "adaptive_r0.50"):
        model = models[name].to(device)
        _, metadata = _check_forward(name, model, image)
        forward_metadata[name] = metadata
        expected = round(225 * 0.5)
        if metadata["requested_patch_token_count"] != expected:
            raise AssertionError(f"{name}: wrong requested token count")
        if metadata["achieved_patch_token_count"] != expected:
            raise AssertionError(f"{name}: wrong achieved token count")
        model.to("cpu")
        torch.cuda.empty_cache()

    # A second positive ratio for both methods, built and released sequentially.
    for method in ("uniform", "adaptive"):
        spec = VariantSpec(
            f"{method}_r0.75", method=method, retention_ratio=0.75
        )
        model = _build(spec, **build_args).to(device)
        _, metadata = _check_forward(spec.name, model, image)
        expected = round(225 * 0.75)
        if metadata["requested_patch_token_count"] != expected:
            raise AssertionError(f"{spec.name}: wrong requested token count")
        if metadata["achieved_patch_token_count"] != expected:
            raise AssertionError(f"{spec.name}: wrong achieved token count")
        forward_metadata[spec.name] = metadata
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # The compatibility zero-retention setting must reproduce the dense baseline.
    zero_spec = VariantSpec(
        "adaptive_zero", method="adaptive", retention_ratio=0.0
    )
    zero_model = _build(zero_spec, **build_args).to(device)
    zero_logits, zero_metadata = _check_forward("adaptive_zero", zero_model, image)
    torch.testing.assert_close(zero_logits, baseline_logits, rtol=1e-5, atol=1e-6)
    if zero_metadata["achieved_patch_token_count"] != 225:
        raise AssertionError("zero-retention compatibility path unexpectedly shortened tokens")
    del zero_model
    gc.collect()
    torch.cuda.empty_cache()
    print("Forward contracts and zero-merging equivalence: OK")

    # WS3 end-to-end latency, stage timing, and allocated-memory sanity check.
    rows = run_suite(
        models,
        batch_sizes=(1,),
        device=device,
        inputs=image_cpu,
        warmup=args.warmup,
        iters=args.iterations,
        amp=True,
        stages=True,
        seed=6411,
    )
    expected_families = {"baseline", "uniform_merge", "adaptive_merge"}
    if {row["family"] for row in rows} != expected_families:
        raise AssertionError("WS3 did not identify all three model families")
    for row in rows:
        for key in ("latency_median_ms", "latency_p90_ms", "peak_memory_mb"):
            if not math.isfinite(float(row[key])) or float(row[key]) <= 0:
                raise AssertionError(f"{row['variant']}: invalid WS3 {key}={row[key]}")
        if "encoder_median_ms" not in row:
            raise AssertionError(f"{row['variant']}: missing encoder stage timing")
    print("WS3 CUDA sanity rows:")
    print(json.dumps(rows, indent=2, default=str))

    # WS4 patient-level plumbing on one synthetic validation volume only.
    loader = DataLoader(
        _SyntheticValidationDataset(image_cpu, target_cpu), batch_size=1, shuffle=False
    )
    ws4_summary = {}
    for name, model in models.items():
        model.to(device).eval()
        result = evaluate_model(
            model,
            loader,
            rules={"raw": {"threshold": 0.5}},
            device=device,
            amp=True,
            variant=name,
            variant_info=specs[name].info(),
            progress=False,
        )
        if result.metadata["patients"] != 1 or result.metadata["slices"] != 1:
            raise AssertionError(f"{name}: WS4 synthetic-volume accounting failed")
        if len(result.rows) != 3 or {row["region"] for row in result.rows} != {
            "WT", "TC", "ET"
        }:
            raise AssertionError(f"{name}: WS4 did not emit all region rows")
        ws4_summary[name] = result.metadata["global_dice"]
        model.to("cpu")
        torch.cuda.empty_cache()
    print("WS4 synthetic interface: OK")
    print(json.dumps(ws4_summary, indent=2))

    print("Retention metadata:")
    print(json.dumps(forward_metadata, indent=2, default=str))
    print("SMOKE TEST PASS")


if __name__ == "__main__":
    main()
