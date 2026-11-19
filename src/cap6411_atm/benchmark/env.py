"""Record the hardware/software environment next to every benchmark result.

Latency numbers are only comparable when they come from the same GPU, driver,
CUDA, and PyTorch build, so every result file is paired with this metadata.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from cap6411_atm.project import PROJECT_ROOT


def git_commit(repository: Path = PROJECT_ROOT) -> str | None:
    """Return the checked-out commit, or ``None`` outside a Git checkout."""

    try:
        return subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_is_dirty(repository: Path = PROJECT_ROOT) -> bool | None:
    """Return whether the checkout differs from ``HEAD``.

    Untracked files count: new, uncommitted source modules mean the recorded
    commit does not contain the code that ran.  Git-ignored outputs, data, and
    checkpoints are not reported by ``git status`` and do not count.
    """

    try:
        output = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(output.strip())


def _nvidia_driver() -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def default_device(device: torch.device | str | None = None) -> torch.device:
    """``device`` if given, else CUDA when available, else CPU."""

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_env(device: torch.device | str | None = None, **extra: Any) -> dict[str, Any]:
    """Collect the facts needed to reproduce or compare a latency measurement."""

    device = default_device(device)
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads": torch.get_num_threads(),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        info.update(
            {
                "gpu_name": properties.name,
                "gpu_memory_gb": round(properties.total_memory / 1024**3, 2),
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "nvidia_driver": _nvidia_driver(),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            }
        )
    info.update(extra)
    return info


def write_env(path: str | Path, env: dict[str, Any]) -> Path:
    """Write environment metadata as indented JSON and return the path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
