"""Compare model variants on measured latency, peak memory, and validation Dice.

Every variant is an ``nn.Module`` with the shared contract
``images [B,4,240,240] -> logits [B,3,240,240]``.  Optional attributes let
other workstreams describe their variants without changing this module:

- ``benchmark_family``: e.g. ``"uniform_merge"`` or ``"adaptive_merge"``
- ``benchmark_setting``: e.g. a merge ratio
- ``token_count``: spatial tokens the encoder processes (theoretical)

Reduced-resolution wrappers are described automatically.  Token counts are
reported as theoretical; only ``latency_*`` and ``*_memory_mb`` columns are
measurements.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from cap6411_atm.models import BraTSDINOv3Segmenter, DINOv3SegmentationConfig
from cap6411_atm.project import PROJECT_ROOT
from cap6411_atm.training.metrics import StreamingDice

from .comparability import (
    LATENCY_KEYS,
    check_same_dice_basis,
    latency_conditions,
    same_dice_basis,
)
from .env import collect_env, default_device, write_env
from .latency import (
    autocast,
    benchmark,
    benchmark_stages,
    precision_label,
    supports_stage_timing,
)
from .resolution import DEFAULT_RESOLUTIONS, ReducedResolutionSegmenter, resolution_sweep
from .tables import results_table


_CONFIG = DINOv3SegmentationConfig()
INPUT_SHAPE = (_CONFIG.input_channels, *_CONFIG.input_size)  # (4, 240, 240)
BASELINE_TOKENS = _CONFIG.feature_grid[0] * _CONFIG.feature_grid[1]  # 225

# Identifier columns that must never be parsed as numbers when reading a CSV.
_TEXT_COLUMNS = frozenset(
    {"variant", "family", "setting", "input", "device", "precision", "dice_scope"}
)

# Memory already on the GPU (other than the variant and its input) above this
# threshold is reported as a warning: it would inflate peak-memory numbers.
_OTHER_MEMORY_WARN_MB = 1.0


def describe(model: nn.Module) -> dict[str, Any]:
    """Return family/setting/token metadata for a variant.

    Explicit ``benchmark_*`` attributes always win.  Automatic labels are given
    only to the exact baseline and reduced-resolution classes (wrapping the exact
    baseline), so a merger built on either is never mislabeled as a reference.
    Anything else is ``unlabeled`` and is still compared.
    """

    if hasattr(model, "benchmark_family"):
        return {
            "family": model.benchmark_family,
            "setting": str(getattr(model, "benchmark_setting", "")),
            "resolution": getattr(model, "benchmark_resolution", _CONFIG.input_size[0]),
            "tokens_theoretical": getattr(model, "token_count", None),
        }
    if type(model) is ReducedResolutionSegmenter and type(model.base) is BraTSDINOv3Segmenter:
        return {
            "family": "baseline" if model.is_identity else "reduced_resolution",
            "setting": f"{model.size}px",
            "resolution": model.size,
            "tokens_theoretical": model.token_count,
        }
    if type(model) is BraTSDINOv3Segmenter:
        return {"family": "baseline", "setting": f"{_CONFIG.input_size[0]}px",
                "resolution": _CONFIG.input_size[0], "tokens_theoretical": BASELINE_TOKENS}
    return {
        "family": "unlabeled",
        "setting": "",
        "resolution": _CONFIG.input_size[0],
        "tokens_theoretical": getattr(model, "token_count", None),
    }


def _batch(inputs: Tensor | None, batch_size: int, device: torch.device, seed: int) -> Tensor:
    if inputs is None:
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(batch_size, *INPUT_SHAPE, generator=generator).to(device)
    repeats = math.ceil(batch_size / inputs.shape[0])
    return inputs.repeat(repeats, 1, 1, 1)[:batch_size].to(device)


def run_suite(
    variants: Mapping[str, nn.Module],
    *,
    batch_sizes: Sequence[int] = (1, 2),
    device: torch.device | str | None = None,
    inputs: Tensor | None = None,
    warmup: int = 25,
    iters: int = 200,
    amp: bool = True,
    stages: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Benchmark every variant at every batch size; return one row per pair.

    ``inputs`` may hold real MRI slices ``[N,4,240,240]``.  Use real slices for
    content-adaptive variants, whose cost can depend on image content; the
    default is seeded random input, which is fine for fixed-token models.

    Every row is stamped with the GPU model, driver, CUDA, and PyTorch build,
    so rows from different machines are never treated as comparable.  Before
    each variant is measured, all *other* variants are moved off the GPU and
    the allocator cache is cleared.  Memory still held by anything else is
    recorded as ``other_gpu_memory_mb`` (with a warning when non-trivial), and
    ``peak_memory_variant_mb`` excludes it.
    """

    device = default_device(device)
    env = collect_env(device)
    stamp = {
        "gpu": env.get("gpu_name", env.get("cpu")),
        "nvidia_driver": env.get("nvidia_driver"),
        "cuda_runtime": env.get("cuda_runtime"),
        "torch": env["torch"],
    }
    rows: list[dict[str, Any]] = []
    for name, model in variants.items():
        _isolate_on_device(variants, name, device)
        for batch_size in batch_sizes:
            images = _batch(inputs, batch_size, device, seed)
            row: dict[str, Any] = {"variant": name, **describe(model), **stamp}
            row["input"] = "random" if inputs is None else "mri_slices"
            other_mb = _other_gpu_memory_mb(model, images)
            row.update(benchmark(model, images, warmup=warmup, iters=iters, amp=amp))
            if device.type == "cuda":
                row["model_memory_mb"] = _module_bytes_on(model, device) / 1024**2
                row["other_gpu_memory_mb"] = other_mb
                row["peak_memory_variant_mb"] = row["peak_memory_mb"] - other_mb
                if other_mb > _OTHER_MEMORY_WARN_MB:
                    warnings.warn(
                        f"{name}: {other_mb:.1f} MB of unrelated GPU memory was allocated "
                        "during measurement; peak_memory_mb includes it, "
                        "peak_memory_variant_mb excludes it"
                    )
            if stages and supports_stage_timing(model):
                # Variants without a known stage split get no stage columns,
                # rather than stage times measured on the wrong code path.
                for stage, result in benchmark_stages(
                    model, images, warmup=warmup, iters=iters, amp=amp
                ).items():
                    row[f"{stage}_median_ms"] = result["latency_median_ms"]
            rows.append(row)
    return rows


def _unique_tensors(module: nn.Module):
    seen: set[int] = set()
    for tensor in itertools.chain(module.parameters(), module.buffers()):
        if id(tensor) not in seen:
            seen.add(id(tensor))
            yield tensor


def _module_bytes_on(module: nn.Module, device: torch.device) -> int:
    return sum(
        t.numel() * t.element_size() for t in _unique_tensors(module) if t.device == device
    )


def _isolate_on_device(
    variants: Mapping[str, nn.Module], current: str, device: torch.device
) -> None:
    """Move every other variant off the GPU, then put ``current`` on it.

    Variants may share submodules (the resolution sweep shares one base
    model), so the others are moved to CPU first and ``current`` last.
    """

    if device.type == "cuda":
        for name, other in variants.items():
            if name != current:
                other.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
    variants[current].to(device).eval()


def _other_gpu_memory_mb(model: nn.Module, images: Tensor) -> float:
    """Allocated GPU memory not belonging to ``model`` or ``images``."""

    device = images.device
    if device.type != "cuda":
        return 0.0
    torch.cuda.synchronize(device)
    own = _module_bytes_on(model, device) + images.numel() * images.element_size()
    return max(torch.cuda.memory_allocated(device) - own, 0) / 1024**2


def validation_loader(
    archive_path: str | Path,
    manifest_path: str | Path,
    inspection_config: str | Path,
    *,
    batch_size: int = 2,
    num_workers: int = 0,
    patients: int | None = None,
    seed: int = 42,
    cache_dir: str | Path | None = None,
    max_cached_patients: int = 1,
    pin_memory: bool | None = None,
):
    """Validation loader with Workstream 1's split, patient order, and batch sampler.

    Mirrors the validation half of ``training.baseline.build_training_components``
    (which cannot be reused directly because it also builds the model and the
    training set).  ``cache_dir``/``max_cached_patients`` are passed to the
    dataset exactly as there; ``pin_memory`` defaults to on when CUDA is
    available.  ``patients`` limits evaluation to a deterministic pilot subset
    (same selection rule as Workstream 1); leave it ``None`` for report numbers.
    The isolated test split is never loaded here.
    """

    from torch.utils.data import DataLoader

    from cap6411_atm.data.dataset import BraTSAxialSliceDataset
    from cap6411_atm.training.sampling import (
        PatientLocalBatchSampler,
        PatientSubset,
        select_pilot_patients,
    )

    dataset = BraTSAxialSliceDataset(
        archive_path,
        manifest_path,
        "validation",
        inspection_config,
        cache_dir=cache_dir,
        max_cached_patients=max_cached_patients,
    )
    if patients is not None:
        dataset = PatientSubset(
            dataset, select_pilot_patients(dataset.patient_ids, patients, seed=seed + 1)
        )
    sampler = PatientLocalBatchSampler(
        dataset, batch_size=batch_size, seed=seed, shuffle_patients=False, shuffle_slices=False
    )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    loader.scope = (
        "validation (full split)" if patients is None else f"validation pilot ({patients} patients)"
    )
    return loader


def evaluate_quality(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str | None = None,
    amp: bool = True,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Global voxel-count WT/TC/ET Dice, identical to Workstream 1 validation."""

    return evaluate_quality_many(
        {"model": model}, loader, device=device, amp=amp,
        threshold=threshold, max_batches=max_batches,
    )["model"]


def evaluate_quality_many(
    variants: Mapping[str, nn.Module],
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str | None = None,
    amp: bool = True,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Dice for every variant in a single pass over the loader.

    Each batch is read and decoded once and scored by all variants, instead of
    re-reading the validation archive once per variant.  Each result records
    its scope (``loader.scope`` from ``validation_loader``), patient and slice
    counts, and whether ``max_batches`` truncated the pass.
    """

    if not variants:
        raise ValueError("no variants to evaluate")
    if device is None:
        device = next(next(iter(variants.values())).parameters()).device
    device = torch.device(device)
    for model in variants.values():
        model.to(device).eval()
    metrics = {name: StreamingDice(threshold) for name in variants}
    samples = 0
    patients: set[str] = set()
    slice_keys: list[str] = []
    truncated = False
    with torch.inference_mode():
        iterator = iter(loader)
        batches = iterator if max_batches is None else itertools.islice(iterator, max_batches)
        batch_count = 0
        for batch in batches:
            batch_count += 1
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            for name, model in variants.items():
                with autocast(device, amp):
                    logits = model(images)
                metrics[name].update(logits, targets)
            samples += int(images.shape[0])
            batch_patients = [str(p) for p in batch.get("patient_id", ())]
            patients.update(batch_patients)
            slice_indices = batch.get("slice_index")
            if slice_indices is not None and len(batch_patients) == images.shape[0]:
                for patient, index in zip(batch_patients, slice_indices):
                    slice_keys.append(f"{patient}:{int(index)}")
            else:
                slice_keys.append(f"batch{batch_count}:{tuple(images.shape)}")
        # Truncated if the cap was reached and the loader is known to hold more.
        if max_batches is not None and batch_count == max_batches:
            try:
                truncated = len(loader) > max_batches
            except TypeError:
                truncated = True
    scope = getattr(loader, "scope", "unknown")
    data_fingerprint = hashlib.sha256(
        "\n".join(sorted(slice_keys)).encode("utf-8")
    ).hexdigest()[:16]
    fingerprints: dict[int, str] = {}
    results = {}
    for name, metric in metrics.items():
        base = _trained_model(variants[name])
        if id(base) not in fingerprints:
            fingerprints[id(base)] = checkpoint_fingerprint(base)
        result = metric.compute()
        result.update(
            {
                "samples": samples,
                "patients": len(patients),
                "scope": f"{scope} (truncated at {max_batches} batches)" if truncated else scope,
                "precision": precision_label(device, amp),
                "threshold": threshold,
                "checkpoint_fingerprint": fingerprints[id(base)],
                "data_fingerprint": data_fingerprint,
            }
        )
        results[name] = result
    return results


def _trained_model(model: nn.Module) -> nn.Module:
    """The trained segmenter inside a variant (the wrapped base for wrappers)."""

    if isinstance(model, BraTSDINOv3Segmenter):
        return model
    for module in model.modules():
        if isinstance(module, BraTSDINOv3Segmenter):
            return module
    return model


def checkpoint_fingerprint(model: nn.Module) -> str:
    """Short SHA-256 of every parameter and buffer (names, dtypes, shapes, values).

    Two variants built on the same trained weights (backbone + adapter +
    decoder) get the same fingerprint, regardless of resolution wrapper or
    device; any differing weight changes it.
    """

    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        data = tensor.detach().to("cpu").contiguous()
        digest.update(f"{key}|{data.dtype}|{tuple(data.shape)}|".encode("utf-8"))
        digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()[:16]


def attach_quality(
    rows: list[dict[str, Any]], quality: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Add Dice columns from ``{variant: evaluate_quality(...)}`` to benchmark rows.

    The evaluation scope and patient/slice counts are copied too, so a pilot
    Dice can never be mistaken for a full-validation Dice in a saved CSV.
    """

    for row in rows:
        result = quality.get(row["variant"])
        if result is None:
            continue
        row["macro_dice"] = result["macro_dice"]
        for region, value in result["dice"].items():
            row[f"dice_{region}"] = value
        row["dice_scope"] = result.get("scope", "unknown")
        row["dice_patients"] = result.get("patients")
        row["dice_samples"] = result.get("samples")
        row["dice_checkpoint"] = result.get("checkpoint_fingerprint")
        row["dice_data"] = result.get("data_fingerprint")
        row["dice_threshold"] = result.get("threshold")
    return rows


def match_latency(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 1,
    reference_family: str = "reduced_resolution",
) -> list[dict[str, Any]]:
    """Pair each merging variant with the closest-latency reduced-resolution run.

    The full-resolution baseline is included in the reference pool, so a
    merger slower than every reduced resolution is matched to the baseline.
    A candidate is only matched against references measured under the same
    conditions (batch size, device, precision, input kind, GPU model, driver,
    CUDA, and PyTorch build); candidates with no comparable reference are
    skipped with a warning.  A Dice delta is only reported when both rows'
    Dice share the same checkpoint and data fingerprints and threshold.
    """

    references = ("baseline", reference_family)
    selected = [row for row in rows if row["batch_size"] == batch_size]
    pool = [row for row in selected if row["family"] in references]
    candidates = [row for row in selected if row["family"] not in references]
    if not pool:
        raise ValueError(f"no {reference_family}/baseline rows at batch size {batch_size}")

    pairs = []
    for row in candidates:
        latency = row["latency_median_ms"]
        comparable = [ref for ref in pool if latency_conditions(ref) == latency_conditions(row)]
        if not comparable:
            warnings.warn(
                f"{row['variant']}: no reference measured under the same conditions "
                f"{dict(zip(LATENCY_KEYS, latency_conditions(row)))}; not matched"
            )
            continue
        match = min(comparable, key=lambda ref: abs(ref["latency_median_ms"] - latency))
        pair = {
            "variant": row["variant"],
            "latency_median_ms": latency,
            "matched_variant": match["variant"],
            "matched_latency_median_ms": match["latency_median_ms"],
            "latency_gap_pct": 100.0 * (latency - match["latency_median_ms"]) / match["latency_median_ms"],
            "batch_size": batch_size,
        }
        if "macro_dice" in row and "macro_dice" in match and not same_dice_basis(row, match):
            warnings.warn(
                f"{row['variant']} vs {match['variant']}: Dice from different or unrecorded "
                "checkpoints/validation slices/thresholds; Dice delta omitted"
            )
        elif "macro_dice" in row and "macro_dice" in match:
            pair["macro_dice"] = row["macro_dice"]
            pair["matched_macro_dice"] = match["macro_dice"]
            pair["macro_dice_delta"] = row["macro_dice"] - match["macro_dice"]
        pairs.append(pair)
    return pairs


def write_rows(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write rows to CSV with the union of all keys as columns."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a results CSV, converting numeric fields back to numbers.

    Empty cells (columns a row never had, e.g. Dice for an unevaluated variant)
    are dropped, so ``"macro_dice" in row`` means the same before and after a
    write/read round trip.
    """

    def convert(key: str, value: str) -> Any:
        if key in _TEXT_COLUMNS:
            return value
        for kind in (int, float):
            try:
                return kind(value)
            except ValueError:
                pass
        if value in ("True", "False"):
            return value == "True"
        return value

    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [
            {key: convert(key, value) for key, value in row.items() if value != ""}
            for row in csv.DictReader(stream)
        ]


def plot_tradeoff(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    batch_size: int = 1,
    gpu_name: str | None = None,
) -> Path:
    """Macro Dice vs measured median latency, one series per family."""

    # An explicit Figure renders off-screen without changing the global
    # matplotlib backend, so inline notebook plots keep working afterwards.
    from matplotlib.figure import Figure

    selected = [
        r for r in rows
        if r["batch_size"] == batch_size and isinstance(r.get("macro_dice"), (int, float))
    ]
    if not selected:
        raise ValueError("no rows with macro_dice; run attach_quality first")
    check_same_dice_basis(selected)
    gpus = {r.get("gpu") for r in selected}
    if len(gpus) > 1:
        raise ValueError(f"rows were measured on different GPUs {sorted(map(str, gpus))}")
    figure = Figure(figsize=(6, 4))
    axis = figure.subplots()
    families: dict[str, list[Mapping[str, Any]]] = {}
    for row in selected:
        families.setdefault(row["family"], []).append(row)
    for family, members in families.items():
        members = sorted(members, key=lambda r: r["latency_median_ms"])
        x = [r["latency_median_ms"] for r in members]
        y = [r["macro_dice"] for r in members]
        style = "-o" if len(members) > 1 else "o"
        axis.plot(x, y, style, label=family.replace("_", " "))
        for row in members:
            axis.annotate(
                str(row.get("setting", "")),
                (row["latency_median_ms"], row["macro_dice"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
    axis.set_xlabel("Median latency (ms, measured)")
    axis.set_ylabel("Validation macro Dice")
    title = f"Quality vs latency (batch {batch_size})"
    axis.set_title(f"{title}, {gpu_name}" if gpu_name else title)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    return destination


def load_timing_only_dinov3_vitb16(repository: str | Path) -> nn.Module:
    """Official ViT-B/16 architecture with random weights, for latency only.

    Latency does not depend on weight values, so this allows benchmarking
    before gated checkpoint access is granted.  Never use it for Dice.
    """

    from cap6411_atm.models.dinov3_segmentation import (
        DINOV3_VITB16_HUB_ENTRY,
        DINOv3SegmentationConfig,
        _repository_head,
    )

    repository_path = Path(repository).expanduser().resolve()
    if not repository_path.is_dir() or not (repository_path / ".git").exists():
        raise FileNotFoundError(f"official DINOv3 Git checkout not found: {repository_path}")
    expected = DINOv3SegmentationConfig().dinov3_revision
    actual = _repository_head(repository_path)
    if actual != expected:
        raise RuntimeError(f"DINOv3 revision mismatch: expected {expected}, received {actual}")
    return torch.hub.load(
        str(repository_path), DINOV3_VITB16_HUB_ENTRY, source="local", pretrained=False
    )


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "benchmark_resolution_sweep.json"
_CONFIG_KEYS = {
    "sizes": list,
    "batch_sizes": list,
    "warmup": int,
    "iters": int,
    "amp": bool,
    "stages": bool,
    "seed": int,
}


def load_benchmark_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate a benchmark settings JSON (see ``configs/``)."""

    values = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(values) - set(_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown benchmark config keys: {sorted(unknown)}")
    for key, kind in _CONFIG_KEYS.items():
        value = values.get(key)
        if key in values and (
            not isinstance(value, kind) or (kind is int and isinstance(value, bool))
        ):
            raise ValueError(f"benchmark config {key!r} must be {kind.__name__}")
    for key in ("sizes", "batch_sizes"):
        if key in values and not all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in values[key]
        ):
            raise ValueError(f"benchmark config {key!r} must be positive integers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the DINOv3 BraTS baseline and reduced-resolution variants. "
        "Settings come from --config; explicit options override it."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Benchmark settings JSON (default: configs/benchmark_resolution_sweep.json)",
    )
    parser.add_argument("--dinov3-repo", required=True, help="Official DINOv3 checkout at the pinned revision")
    parser.add_argument("--checkpoint", help="Authorized DINOv3 ViT-B/16 checkpoint; omit for timing-only random weights")
    parser.add_argument("--artifact", help="Workstream 1 adapter/decoder artifact (requires --checkpoint)")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="fp16 autocast on CUDA (use --no-amp for fp32)")
    parser.add_argument("--stages", action=argparse.BooleanOptionalAction, default=False,
                        help="Also time adapter/encoder/decoder separately")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "outputs" / "benchmarks"),
        help="Result directory (default: <repo>/outputs/benchmarks, git-ignored)",
    )
    parser.add_argument("--tag", default="resolution_sweep")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options, using the --config file for defaults.

    Precedence: explicit command-line option > config file > built-in default.
    """

    parser = _parser()
    known, _ = parser.parse_known_args(argv)
    config_path = Path(known.config)
    if config_path.is_file():
        parser.set_defaults(**load_benchmark_config(config_path))
    elif known.config != str(DEFAULT_CONFIG):
        raise SystemExit(f"benchmark config not found: {config_path}")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from cap6411_atm.models import (
        BraTSDINOv3Segmenter,
        build_pretrained_brats_dinov3,
        load_adapter_decoder_artifact,
    )

    args = parse_args(argv)
    if args.artifact and not args.checkpoint:
        raise SystemExit("--artifact needs --checkpoint: trained heads on random backbones are meaningless")
    if args.checkpoint:
        model = build_pretrained_brats_dinov3(args.dinov3_repo, args.checkpoint)
        weights = "official"
    else:
        model = BraTSDINOv3Segmenter(load_timing_only_dinov3_vitb16(args.dinov3_repo))
        weights = "random (timing only)"
    if args.artifact:
        load_adapter_decoder_artifact(model, args.artifact)
        weights += " + workstream1 adapter/decoder"

    device = default_device(args.device)
    rows = run_suite(
        resolution_sweep(model, args.sizes),
        batch_sizes=args.batch_sizes,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        amp=args.amp,
        stages=args.stages,
        seed=args.seed,
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out)
    csv_path = write_rows(rows, out / f"{stamp}_{args.tag}.csv")
    env = collect_env(device, weights=weights, argv=vars(args))
    write_env(out / f"{stamp}_{args.tag}_env.json", env)
    caption = f"{weights}; {env.get('gpu_name', env['device'])}; {rows[0]['precision']}"
    tables = [
        f"## Batch size {batch_size}\n\n" + results_table(rows, batch_size=batch_size, caption=caption)
        for batch_size in args.batch_sizes
    ]
    table_path = out / f"{stamp}_{args.tag}_table.md"
    table_path.write_text(f"Weights: {weights}\n\n" + "\n".join(tables), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "table": str(table_path),
                      "gpu": env.get("gpu_name", env["device"])}))
    print(tables[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
