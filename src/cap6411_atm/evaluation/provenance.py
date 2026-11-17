"""Fingerprints that tie every result to the exact inputs that produced it.

A result directory is only reusable (and only comparable with other result
directories) when it was produced from the same checkpoint, dataset archive,
split manifest, patients and evaluation settings. ``build_provenance`` records
the files; :func:`run_key` hashes everything that determines the numbers, so
stale or mixed results can be detected instead of silently reused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

# Files above this size are fingerprinted from their size plus the first and
# last chunk instead of a full read (the BraTS archive is ~13 GB).
FULL_HASH_LIMIT = 512 * 1024 * 1024
PARTIAL_CHUNK = 64 * 1024 * 1024

REQUIRED_PROVENANCE = ("checkpoint", "dataset", "manifest")

# Provenance entries that must be identical for results to be compared.
SHARED_PROVENANCE = ("dataset", "manifest", "inspection_config")


def fingerprint_file(path: str | Path) -> dict[str, Any]:
    """SHA-256 fingerprint of a file (full for small files, head+tail for large ones)."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if size <= FULL_HASH_LIMIT:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
            method = "sha256"
        else:
            digest.update(str(size).encode())
            digest.update(handle.read(PARTIAL_CHUNK))
            handle.seek(size - PARTIAL_CHUNK)
            digest.update(handle.read(PARTIAL_CHUNK))
            method = "sha256_size_head_tail_64MiB"
    return {"name": path.name, "size": size, "method": method, "sha256": digest.hexdigest()}


def build_provenance(
    *,
    checkpoint: str | Path | Mapping[str, Any] | None,
    dataset: str | Path | Mapping[str, Any],
    manifest: str | Path | Mapping[str, Any],
    inspection_config: str | Path | Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Fingerprint the inputs of a run.

    Paths are hashed. A mapping is recorded as given, except that its
    ``Path`` values are hashed, so ``checkpoint={"backbone": Path(...),
    "adapter": Path(...)}`` fingerprints both files, and
    ``{"name": ..., "sha256": ...}`` records a checksum verified elsewhere.
    Extra keyword arguments (JSON-compatible) are recorded as-is.
    """

    def describe(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return {
                key: fingerprint_file(item) if isinstance(item, Path) else item
                for key, item in value.items()
            }
        return fingerprint_file(value)

    provenance = {
        "checkpoint": describe(checkpoint),
        "dataset": describe(dataset),
        "manifest": describe(manifest),
        "inspection_config": describe(inspection_config),
    }
    provenance.update(extra)
    return provenance


def validate_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Require checkpoint, dataset and manifest entries; return a plain dict."""

    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping (see build_provenance)")
    missing = [key for key in REQUIRED_PROVENANCE if not provenance.get(key)]
    if missing:
        raise ValueError(f"provenance is missing {missing}; record them with build_provenance()")
    return json.loads(canonical_json(provenance))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def run_key(identity: Mapping[str, Any]) -> str:
    """Hash of everything that determines a variant's numbers."""

    return digest(identity)
