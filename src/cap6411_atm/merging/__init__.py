"""Token-merging interfaces and deterministic patch-group reconstruction.

This module provides the minimal post-encoder contract described in the project
specification: a merger that reduces spatial patch tokens while retaining a
per-sample state sufficient to reconstruct the original row-major ordering.
The implementation is intentionally lightweight and numerically stable so it can
be validated with synthetic DINOv3 feature tensors before any real checkpoint is
used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass
class TokenMergeState:
    """Per-sample bookkeeping for a uniform spatial token merge.

    `original_to_merged` maps every original row-major patch position to the index
    of the representative token in the merged tensor. `merge_weights` records the
    averaging weight used for each original patch and `group_sizes` stores the
    number of original tokens represented by each merged token. `original_positions`
    and `merge_coordinates` preserve the original grid semantics.
    """

    grid_size: tuple[int, int]
    original_to_merged: Tensor
    merge_weights: Tensor
    group_sizes: Tensor
    original_positions: Tensor
    merge_coordinates: Tensor | None = None
    merge_size: int = 1
    method: str = "uniform"
    merge_ratio: float = 0.0

    @property
    def batch_size(self) -> int:
        return int(self.original_to_merged.shape[0])

    @property
    def merged_count(self) -> int:
        return int(self.group_sizes.shape[1])


def _validate_input_tokens(tokens: Tensor, grid_size: tuple[int, int]) -> tuple[int, int, int]:
    if tokens.ndim != 3:
        raise ValueError(f"expected patch tokens [B,N,D], received {tuple(tokens.shape)}")
    batch_size, num_tokens, embedding_dim = tokens.shape
    expected_tokens = grid_size[0] * grid_size[1]
    if num_tokens != expected_tokens:
        raise ValueError(
            f"grid_size {grid_size} implies {expected_tokens} tokens, but inputs contain {num_tokens}"
        )
    if embedding_dim <= 0:
        raise ValueError("embedding dimension must be positive")
    return batch_size, num_tokens, embedding_dim


def _build_group_map(grid_size: tuple[int, int], merge_size: int) -> Tensor:
    if merge_size < 1:
        raise ValueError(f"merge_size must be >= 1, received {merge_size}")
    height, width = grid_size
    group_indices = torch.empty(height * width, dtype=torch.long)
    next_group = 0
    for row_start in range(0, height, merge_size):
        row_end = min(height, row_start + merge_size)
        for col_start in range(0, width, merge_size):
            col_end = min(width, col_start + merge_size)
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    group_indices[row * width + col] = next_group
            next_group += 1
    return group_indices


def _build_coordinate_grid(grid_size: tuple[int, int], device: torch.device) -> Tensor:
    height, width = grid_size
    row_index = torch.arange(height, device=device).repeat_interleave(width)
    col_index = torch.arange(width, device=device).repeat(height)
    return torch.stack((row_index, col_index), dim=-1)


def merge_tokens(
    tokens: Tensor,
    *,
    grid_size: tuple[int, int] = (15, 15),
    merge_size: int | None = None,
    merge_ratio: float = 0.0,
    method: str = "uniform",
) -> tuple[Tensor, TokenMergeState]:
    """Merge spatial patch tokens while preserving exact reconstruction semantics.

    The default no-merge path is kept numerically identical to the baseline when
    `merge_size == 1` or `merge_ratio == 0.0`.
    """

    if tokens.ndim != 3:
        raise ValueError(f"expected patch tokens [B,N,D], received {tuple(tokens.shape)}")
    if not 0.0 <= float(merge_ratio) <= 1.0:
        raise ValueError(f"merge_ratio must be in [0,1], received {merge_ratio}")
    if method != "uniform":
        raise ValueError(f"unsupported merge method: {method!r}")

    batch_size, num_tokens, embedding_dim = _validate_input_tokens(tokens, grid_size)
    if merge_size is None:
        merge_size = 1 if merge_ratio == 0.0 else 2
    if merge_size <= 0:
        raise ValueError(f"merge_size must be >= 1, received {merge_size}")

    if merge_size == 1 or merge_ratio == 0.0:
        device = tokens.device
        positions = torch.arange(num_tokens, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        identity = positions.clone()
        group_sizes = torch.ones(batch_size, num_tokens, device=device, dtype=torch.long)
        weights = torch.ones(batch_size, num_tokens, device=device, dtype=tokens.dtype)
        coordinates = _build_coordinate_grid(grid_size, device).unsqueeze(0).expand(batch_size, -1, -1)
        state = TokenMergeState(
            grid_size=grid_size,
            original_to_merged=identity,
            merge_weights=weights,
            group_sizes=group_sizes,
            original_positions=positions,
            merge_coordinates=coordinates,
            merge_size=1,
            method=method,
            merge_ratio=float(merge_ratio),
        )
        return tokens.clone(), state

    group_map = _build_group_map(grid_size, merge_size)
    num_groups = int(group_map.max().item()) + 1
    merged = torch.empty(batch_size, num_groups, embedding_dim, device=tokens.device, dtype=tokens.dtype)
    original_to_merged = torch.empty(batch_size, num_tokens, device=tokens.device, dtype=torch.long)
    group_sizes = torch.empty(batch_size, num_groups, device=tokens.device, dtype=torch.long)
    merge_weights = torch.empty(batch_size, num_tokens, device=tokens.device, dtype=tokens.dtype)
    original_positions = torch.arange(num_tokens, device=tokens.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    coordinates = _build_coordinate_grid(grid_size, tokens.device).unsqueeze(0).expand(batch_size, -1, -1)

    for batch_index in range(batch_size):
        batch_tokens = tokens[batch_index]
        per_group = [batch_tokens[group_map == group_id].mean(dim=0) for group_id in range(num_groups)]
        merged[batch_index] = torch.stack(per_group, dim=0)
        original_to_merged[batch_index] = group_map
        group_sizes[batch_index] = torch.bincount(group_map, minlength=num_groups)
        per_group_sizes = group_sizes[batch_index, original_to_merged[batch_index]]
        merge_weights[batch_index] = 1.0 / per_group_sizes.to(tokens.dtype)

    state = TokenMergeState(
        grid_size=grid_size,
        original_to_merged=original_to_merged,
        merge_weights=merge_weights,
        group_sizes=group_sizes,
        original_positions=original_positions,
        merge_coordinates=coordinates,
        merge_size=merge_size,
        method=method,
        merge_ratio=float(merge_ratio),
    )
    return merged, state


def reconstruct_tokens(merged_tokens: Tensor, state: TokenMergeState) -> Tensor:
    """Reconstruct the original row-major token ordering from merged tokens."""

    if merged_tokens.ndim != 3:
        raise ValueError(f"expected merged tokens [B,M,D], received {tuple(merged_tokens.shape)}")
    if merged_tokens.shape[0] != state.batch_size:
        raise ValueError(
            f"merged batch size mismatch: state has batch {state.batch_size}, merged has {merged_tokens.shape[0]}"
        )
    if state.original_to_merged.device != merged_tokens.device:
        state = TokenMergeState(
            grid_size=state.grid_size,
            original_to_merged=state.original_to_merged.to(merged_tokens.device),
            merge_weights=state.merge_weights.to(merged_tokens.device),
            group_sizes=state.group_sizes.to(merged_tokens.device),
            original_positions=state.original_positions.to(merged_tokens.device),
            merge_coordinates=state.merge_coordinates.to(merged_tokens.device) if state.merge_coordinates is not None else None,
            merge_size=state.merge_size,
            method=state.method,
            merge_ratio=state.merge_ratio,
        )
    index = state.original_to_merged.unsqueeze(-1).expand(-1, -1, merged_tokens.shape[-1])
    return merged_tokens.gather(1, index)


class UniformTokenMerger:
    """Deterministic, trainable-free spatial merger for feature-tensor prototypes."""

    def __init__(
        self,
        *,
        grid_size: tuple[int, int] = (15, 15),
        merge_size: int = 2,
        merge_ratio: float = 0.0,
    ) -> None:
        self.grid_size = grid_size
        self.merge_size = merge_size
        self.merge_ratio = merge_ratio

    def __call__(self, tokens: Tensor) -> tuple[Tensor, TokenMergeState]:
        return merge_tokens(
            tokens,
            grid_size=self.grid_size,
            merge_size=self.merge_size,
            merge_ratio=self.merge_ratio,
            method="uniform",
        )


class TokenReconstructor:
    """Minimal reconstruction wrapper for merged patch-token states."""

    def __call__(self, merged_tokens: Tensor, state: TokenMergeState) -> Tensor:
        return reconstruct_tokens(merged_tokens, state)


merger = merge_tokens
reconstructor = reconstruct_tokens

__all__ = [
    "TokenMergeState",
    "TokenReconstructor",
    "UniformTokenMerger",
    "merge_tokens",
    "merger",
    "reconstruct_tokens",
    "reconstructor",
]
