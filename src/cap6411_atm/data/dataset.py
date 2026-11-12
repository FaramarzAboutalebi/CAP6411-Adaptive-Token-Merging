"""PyTorch dataset for 2D axial BraTS segmentation samples."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .inspection import InspectionConfig, discover_patients
from .splitting import load_json_manifest, patient_to_split


MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")
TARGET_ORDER = ("WT", "TC", "ET")
CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NormalizationConfig:
    """Per-modality z-score normalization over finite, nonzero voxels."""

    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("normalization epsilon must be positive")


def normalize_nonzero_zscore(
    volume: np.ndarray,
    config: NormalizationConfig = NormalizationConfig(),
) -> np.ndarray:
    """Return float32 data with normalized foreground and zero background.

    Non-finite values are treated as invalid/background. A modality with no valid
    nonzero voxels, or with near-zero foreground standard deviation, becomes all
    zeros instead of producing NaN or Inf values.
    """

    values = np.asarray(volume, dtype=np.float32)
    valid = np.isfinite(values) & (values != 0)
    output = np.zeros(values.shape, dtype=np.float32)
    if not np.any(valid):
        return output
    foreground = values[valid].astype(np.float64, copy=False)
    mean = float(foreground.mean())
    std = float(foreground.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= config.epsilon:
        return output
    output[valid] = ((foreground - mean) / std).astype(np.float32)
    return output


def segmentation_to_regions(labels: np.ndarray) -> np.ndarray:
    """Convert BraTS 2023 labels into overlapping WT, TC, and ET masks."""

    values = np.asarray(labels)
    if not np.all(np.isfinite(values)):
        raise ValueError("segmentation contains non-finite values")
    unique = set(np.unique(values).tolist())
    unexpected = sorted(unique - {0, 1, 2, 3})
    if unexpected:
        raise ValueError(f"unexpected segmentation labels: {unexpected}")
    wt = np.isin(values, (1, 2, 3))
    tc = np.isin(values, (1, 3))
    et = values == 3
    return np.stack((wt, tc, et), axis=0).astype(np.uint8)


class BraTSAxialSliceDataset(Dataset[dict[str, Any]]):
    """Expose patient-restricted BraTS volumes as 2D axial slices.

    Each sample is a dictionary with:

    - ``image``: float32 tensor ``[4, H, W]`` in T1n, T1c, T2w, T2f order.
    - ``target``: float32 overlapping binary tensor ``[3, H, W]`` in WT, TC, ET order.
    - ``patient_id``: source patient identifier.
    - ``slice_index``: axial index along the third NIfTI dimension.
    - ``split``: manifest split name.

    The manifest is applied before slice enumeration, so slices from a patient
    cannot cross split boundaries.
    """

    def __init__(
        self,
        archive_path: str | Path,
        manifest_path: str | Path,
        split: str,
        inspection_config: str | Path | InspectionConfig,
        *,
        slice_indices: Sequence[int] | Mapping[str, Sequence[int]] | None = None,
        normalization: NormalizationConfig = NormalizationConfig(),
        cache_dir: str | Path | None = None,
        max_cached_patients: int = 1,
    ) -> None:
        if max_cached_patients < 1:
            raise ValueError("max_cached_patients must be at least 1")
        self.archive_path = Path(archive_path)
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.config = (
            inspection_config
            if isinstance(inspection_config, InspectionConfig)
            else InspectionConfig.from_json(inspection_config)
        )
        self.normalization = normalization
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.max_cached_patients = max_cached_patients
        self._memory_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        if not self.archive_path.is_file():
            raise FileNotFoundError(self.archive_path)
        manifest = load_json_manifest(self.manifest_path)
        if split not in manifest["splits"]:
            raise ValueError(f"unknown split {split!r}; available: {list(manifest['splits'])}")
        self.patient_ids = tuple(manifest["splits"][split])
        if not self.patient_ids:
            raise ValueError(f"split {split!r} contains no patients")
        lookup = patient_to_split(manifest["splits"])
        if any(lookup[patient_id] != split for patient_id in self.patient_ids):
            raise ValueError("manifest patient-level split enforcement failed")

        if self.config.expected_shape is None or len(self.config.expected_shape) != 3:
            raise ValueError("inspection config must declare a three-dimensional expected_shape")
        self.volume_shape = tuple(self.config.expected_shape)
        self.axial_depth = self.volume_shape[2]

        with zipfile.ZipFile(self.archive_path) as zf:
            discovered = discover_patients(zf, self.config)
        self._members: dict[str, dict[str, str]] = {}
        expected_roles = set(self.config.file_suffixes)
        for patient_id in self.patient_ids:
            if patient_id not in discovered:
                raise ValueError(f"manifest patient not found in archive: {patient_id}")
            files = discovered[patient_id]
            missing = sorted(expected_roles - set(files))
            duplicates = sorted(key for key, members in files.items() if len(members) != 1)
            if missing or duplicates:
                raise ValueError(
                    f"incomplete patient {patient_id}: missing={missing}, duplicate_roles={duplicates}"
                )
            self._members[patient_id] = {key: members[0] for key, members in files.items()}

        self.samples = tuple(self._build_sample_index(slice_indices))
        if not self.samples:
            raise ValueError("slice selection produced an empty dataset")
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_sample_index(
        self,
        selection: Sequence[int] | Mapping[str, Sequence[int]] | None,
    ) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        for patient_id in self.patient_ids:
            if selection is None:
                indices = range(self.axial_depth)
            elif isinstance(selection, Mapping):
                if patient_id not in selection:
                    raise ValueError(f"slice selection missing manifest patient: {patient_id}")
                indices = selection[patient_id]
            else:
                indices = selection
            normalized_indices = [int(index) for index in indices]
            if len(normalized_indices) != len(set(normalized_indices)):
                raise ValueError(f"duplicate slice indices for patient {patient_id}")
            for index in normalized_indices:
                if not 0 <= index < self.axial_depth:
                    raise ValueError(
                        f"slice index {index} outside [0, {self.axial_depth}) for {patient_id}"
                    )
                samples.append((patient_id, index))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        patient_id, slice_index = self.samples[index]
        images, targets = self._get_patient(patient_id)
        # Copies avoid read-only memmap warnings and isolate augmentation callers.
        image_slice = np.array(images[:, :, :, slice_index], dtype=np.float32, copy=True)
        target_slice = np.array(targets[:, :, :, slice_index], dtype=np.float32, copy=True)
        return {
            "image": torch.from_numpy(image_slice),
            "target": torch.from_numpy(target_slice),
            "patient_id": patient_id,
            "slice_index": slice_index,
            "split": self.split,
        }

    def _get_patient(self, patient_id: str) -> tuple[np.ndarray, np.ndarray]:
        if patient_id in self._memory_cache:
            arrays = self._memory_cache.pop(patient_id)
            self._memory_cache[patient_id] = arrays
            return arrays
        arrays = self._load_disk_cache(patient_id)
        if arrays is None:
            arrays = self._read_patient_from_zip(patient_id)
            if self.cache_dir is not None:
                self._write_disk_cache(patient_id, *arrays)
                arrays = self._load_disk_cache(patient_id) or arrays
        self._memory_cache[patient_id] = arrays
        while len(self._memory_cache) > self.max_cached_patients:
            self._memory_cache.popitem(last=False)
        return arrays

    def _cache_signature(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "shape": list(self.volume_shape),
            "modalities": list(MODALITY_ORDER),
            "targets": list(TARGET_ORDER),
            "normalization": "per-modality finite nonzero z-score",
            "epsilon": self.normalization.epsilon,
        }

    def _cache_patient_dir(self, patient_id: str) -> Path:
        if self.cache_dir is None:
            raise RuntimeError("cache_dir is not configured")
        return self.cache_dir / patient_id

    def _load_disk_cache(self, patient_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        if self.cache_dir is None:
            return None
        patient_dir = self._cache_patient_dir(patient_id)
        metadata_path = patient_dir / "metadata.json"
        image_path = patient_dir / "images.npy"
        target_path = patient_dir / "targets.npy"
        if not (metadata_path.is_file() and image_path.is_file() and target_path.is_file()):
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata != self._cache_signature():
            raise ValueError(f"incompatible cached patient data: {patient_dir}")
        return np.load(image_path, mmap_mode="r"), np.load(target_path, mmap_mode="r")

    def _write_disk_cache(
        self,
        patient_id: str,
        images: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        patient_dir = self._cache_patient_dir(patient_id)
        if patient_dir.exists():
            return
        with tempfile.TemporaryDirectory(prefix=f".{patient_id}-", dir=self.cache_dir) as temp_dir:
            temporary = Path(temp_dir)
            np.save(temporary / "images.npy", images, allow_pickle=False)
            np.save(temporary / "targets.npy", targets, allow_pickle=False)
            (temporary / "metadata.json").write_text(
                json.dumps(self._cache_signature(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                temporary.rename(patient_dir)
            except FileExistsError:
                pass  # Another DataLoader worker completed the same patient first.

    def _read_patient_from_zip(self, patient_id: str) -> tuple[np.ndarray, np.ndarray]:
        roles = (*MODALITY_ORDER, self.config.segmentation_key)
        missing_roles = [role for role in roles if role not in self._members[patient_id]]
        if missing_roles:
            raise ValueError(f"patient {patient_id} missing required roles: {missing_roles}")

        loaded: dict[str, nib.spatialimages.SpatialImage] = {}
        with zipfile.ZipFile(self.archive_path) as zf, tempfile.TemporaryDirectory(
            prefix=f"brats-dataset-{patient_id}-"
        ) as temp_dir:
            root = Path(temp_dir)
            for role in roles:
                destination = root / f"{role}.nii.gz"
                with zf.open(self._members[patient_id][role]) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                loaded[role] = nib.load(destination)

            reference = loaded[MODALITY_ORDER[0]]
            for role, image in loaded.items():
                if tuple(image.shape) != self.volume_shape:
                    raise ValueError(
                        f"{patient_id} {role} shape {tuple(image.shape)} != {self.volume_shape}"
                    )
                if self.config.expected_spacing is not None and not np.allclose(
                    image.header.get_zooms()[:3],
                    self.config.expected_spacing,
                    atol=self.config.spacing_tolerance,
                    rtol=0,
                ):
                    raise ValueError(
                        f"{patient_id} {role} spacing "
                        f"{tuple(float(value) for value in image.header.get_zooms()[:3])} "
                        f"!= {self.config.expected_spacing}"
                    )
                if not np.allclose(
                    image.affine,
                    reference.affine,
                    atol=self.config.affine_tolerance,
                    rtol=0,
                ):
                    raise ValueError(f"{patient_id} {role} affine is not aligned")

            images = np.stack(
                [
                    normalize_nonzero_zscore(
                        np.asanyarray(loaded[role].dataobj), self.normalization
                    )
                    for role in MODALITY_ORDER
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            targets = segmentation_to_regions(
                np.asanyarray(loaded[self.config.segmentation_key].dataobj)
            )
        return images, targets

    def __getstate__(self) -> dict[str, Any]:
        """Avoid copying loaded patient arrays when DataLoader workers spawn."""

        state = self.__dict__.copy()
        state["_memory_cache"] = OrderedDict()
        return state
