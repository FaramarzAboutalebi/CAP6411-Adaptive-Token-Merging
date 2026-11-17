"""Token-merging interfaces and deterministic patch-group reconstruction.

This module provides the minimal post-encoder contract described in the project
specification: a merger that reduces spatial patch tokens while retaining a
per-sample state sufficient to reconstruct the original row-major ordering.
The implementation is intentionally lightweight and numerically stable so it can
be validated with synthetic DINOv3 feature tensors before any real checkpoint is
used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TokenMergeState:
    """Per-sample bookkeeping for spatial or content-adaptive token merging.

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
    merged_coordinates: Tensor | None = None
    merge_size: int = 1
    method: str = "uniform"
    merge_ratio: float = 0.0
    requested_token_count: int | None = None

    @property
    def batch_size(self) -> int:
        return int(self.original_to_merged.shape[0])

    @property
    def merged_count(self) -> int:
        return int(self.group_sizes.shape[1])

    @property
    def original_token_count(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    @property
    def achieved_retention_ratio(self) -> float:
        return self.merged_count / self.original_token_count

    @property
    def retention_metrics(self) -> dict[str, int | float | None]:
        """Report requested and actual token counts and retention ratios."""
        requested_ratio = (
            self.requested_token_count / self.original_token_count
            if self.requested_token_count is not None
            else None
        )
        achieved_ratio = self.achieved_retention_ratio
        return {
            "original_token_count": self.original_token_count,
            "requested_token_count": self.requested_token_count,
            "achieved_token_count": self.merged_count,
            "requested_retention_ratio": requested_ratio,
            "achieved_retention_ratio": achieved_ratio,
            "retention_ratio_error": (
                achieved_ratio - requested_ratio if requested_ratio is not None else None
            ),
        }


def _validate_input_tokens(tokens: Tensor, grid_size: tuple[int, int]) -> tuple[int, int, int]:
    if tokens.ndim != 3:
        raise ValueError(f"expected patch tokens [B,N,D], received {tuple(tokens.shape)}")
    if len(grid_size) != 2 or grid_size[0] <= 0 or grid_size[1] <= 0:
        raise ValueError(f"grid_size must contain two positive dimensions, received {grid_size}")
    batch_size, num_tokens, embedding_dim = tokens.shape
    if batch_size <= 0 or num_tokens <= 0:
        raise ValueError("patch tokens must have non-empty batch and token dimensions")
    if not tokens.is_floating_point():
        raise TypeError("patch tokens must use a floating-point dtype")
    if not torch.isfinite(tokens).all():
        raise ValueError("patch tokens must contain only finite values")
    expected_tokens = grid_size[0] * grid_size[1]
    if num_tokens != expected_tokens:
        raise ValueError(
            f"grid_size {grid_size} implies {expected_tokens} tokens, but inputs contain {num_tokens}"
        )
    if embedding_dim <= 0:
        raise ValueError("embedding dimension must be positive")
    return batch_size, num_tokens, embedding_dim


def _resolve_group_shape(
    grid_size: tuple[int, int],
    *,
    merge_size: int | tuple[int, int] | None,
    merge_ratio: float,
) -> tuple[int, int]:
    height, width = grid_size
    if merge_size is not None:
        if isinstance(merge_size, tuple):
            if len(merge_size) != 2:
                raise ValueError(f"merge_size tuple must have exactly two entries, received {merge_size}")
            merge_height, merge_width = merge_size
        else:
            merge_height = merge_width = int(merge_size)
        if merge_height < 1 or merge_width < 1:
            raise ValueError(f"merge_size entries must be >= 1, received {merge_size}")
        return merge_height, merge_width

    if merge_ratio <= 0.0:
        return 1, 1
    if merge_ratio >= 1.0:
        return 1, 1
    if not 0.0 < merge_ratio < 1.0:
        raise ValueError(f"merge_ratio must be in (0, 1), received {merge_ratio}")

    total_tokens = height * width
    target_retained = max(1, min(total_tokens, int(round(total_tokens * merge_ratio))))

    best_shape = (1, 1)
    best_error = total_tokens
    best_retained = total_tokens

    for merge_height in range(1, height + 1):
        for merge_width in range(1, width + 1):
            retained = math.ceil(height / merge_height) * math.ceil(width / merge_width)
            error = abs(retained - target_retained)
            if error < best_error:
                best_error = error
                best_shape = (merge_height, merge_width)
                best_retained = retained
            elif error == best_error and retained < best_retained:
                best_shape = (merge_height, merge_width)
                best_retained = retained

    return best_shape


def _build_group_map(
    grid_size: tuple[int, int],
    merge_size: int | tuple[int, int],
    *,
    device: torch.device,
) -> Tensor:
    if isinstance(merge_size, tuple):
        merge_height, merge_width = merge_size
    else:
        merge_height = merge_width = int(merge_size)
    if merge_height < 1 or merge_width < 1:
        raise ValueError(f"merge_size entries must be >= 1, received {merge_size}")
    height, width = grid_size
    row_blocks = torch.arange(height, device=device, dtype=torch.long) // merge_height
    col_blocks = torch.arange(width, device=device, dtype=torch.long) // merge_width
    num_col_blocks = math.ceil(width / merge_width)
    return (
        row_blocks[:, None] * num_col_blocks + col_blocks[None, :]
    ).reshape(-1)


def _build_coordinate_grid(grid_size: tuple[int, int], device: torch.device) -> Tensor:
    height, width = grid_size
    row_index = torch.arange(height, device=device).repeat_interleave(width)
    col_index = torch.arange(width, device=device).repeat(height)
    return torch.stack((row_index, col_index), dim=-1)


def _content_adaptive_group_map(tokens: Tensor, target_groups: int) -> Tensor:
    """Cluster each sample by cosine similarity, independently of grid adjacency."""
    batch_size, num_tokens, _ = tokens.shape
    group_maps = torch.empty(batch_size, num_tokens, device=tokens.device, dtype=torch.long)
    if target_groups >= num_tokens:
        return torch.arange(num_tokens, device=tokens.device).expand(batch_size, -1).clone()

    edge_rows, edge_cols = torch.triu_indices(
        num_tokens, num_tokens, offset=1, device=tokens.device
    )
    for batch_index in range(batch_size):
        normalized = torch.nn.functional.normalize(tokens[batch_index].detach(), dim=-1)
        similarities = normalized @ normalized.transpose(0, 1)
        edge_order = (
            similarities[edge_rows, edge_cols]
            .argsort(descending=True, stable=True)
            .cpu()
            .tolist()
        )
        rows = edge_rows.cpu().tolist()
        cols = edge_cols.cpu().tolist()

        parents = list(range(num_tokens))
        sizes = [1] * num_tokens
        cluster_count = num_tokens

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        for edge_index in edge_order:
            first = find(rows[edge_index])
            second = find(cols[edge_index])
            if first == second:
                continue
            if sizes[first] < sizes[second]:
                first, second = second, first
            parents[second] = first
            sizes[first] += sizes[second]
            cluster_count -= 1
            if cluster_count == target_groups:
                break

        compact_ids: dict[int, int] = {}
        assignments = []
        for token_index in range(num_tokens):
            root = find(token_index)
            if root not in compact_ids:
                compact_ids[root] = len(compact_ids)
            assignments.append(compact_ids[root])
        group_maps[batch_index] = torch.tensor(
            assignments, device=tokens.device, dtype=torch.long
        )
    return group_maps


def _validate_merge_state(
    merged_tokens: Tensor,
    state: TokenMergeState,
    *,
    expected_batch_size: int | None = None,
) -> None:
    if merged_tokens.ndim != 3:
        raise ValueError(f"expected merged tokens [B,M,D], received {tuple(merged_tokens.shape)}")
    if any(size <= 0 for size in merged_tokens.shape):
        raise ValueError("merged tokens must have non-empty batch, token, and embedding dimensions")
    if expected_batch_size is not None and merged_tokens.shape[0] != expected_batch_size:
        raise ValueError(
            f"merged batch mismatch: expected {expected_batch_size}, received {merged_tokens.shape[0]}"
        )
    if state.original_to_merged.ndim != 2:
        raise ValueError("original_to_merged must have shape [B,N]")
    if merged_tokens.shape[0] != state.batch_size:
        raise ValueError(
            f"merged batch size mismatch: state has batch {state.batch_size}, merged has {merged_tokens.shape[0]}"
        )
    if merged_tokens.shape[1] != state.merged_count:
        raise ValueError(
            f"merged token count {merged_tokens.shape[1]} does not match state count "
            f"{state.merged_count}"
        )
    if state.merged_count <= 0:
        raise ValueError("merge state must describe at least one merged token")
    if state.requested_token_count is not None and not (
        1 <= state.requested_token_count <= state.grid_size[0] * state.grid_size[1]
    ):
        raise ValueError("requested_token_count must be within the original token range")
    if state.original_to_merged.dtype not in (torch.int32, torch.int64, torch.long):
        raise TypeError("original_to_merged must be integer-valued")
    if state.original_to_merged.shape[1] == 0:
        raise ValueError("merge state must describe at least one original token")
    if state.original_to_merged.min().item() < 0:
        raise ValueError("original_to_merged contains negative merge assignments")
    if state.original_to_merged.max().item() >= state.merged_count:
        raise ValueError("original_to_merged contains an assignment outside the merged-token range")
    if state.group_sizes.ndim != 2 or state.group_sizes.shape != (
        state.batch_size,
        state.merged_count,
    ):
        raise ValueError("group_sizes must have shape [B,M] matching the merge state")
    if state.group_sizes.dtype not in (torch.int32, torch.int64, torch.long):
        raise TypeError("group_sizes must be integer-valued")
    if state.merge_weights.shape != state.original_to_merged.shape:
        raise ValueError("merge_weights must match original_to_merged shape")
    if not state.merge_weights.is_floating_point():
        raise TypeError("merge_weights must be floating-point")
    if state.grid_size[0] <= 0 or state.grid_size[1] <= 0:
        raise ValueError(f"merge state has invalid grid_size {state.grid_size}")
    if state.grid_size[0] * state.grid_size[1] != state.original_to_merged.shape[1]:
        raise ValueError("original_to_merged length does not match grid_size")
    if state.original_positions.shape != state.original_to_merged.shape:
        raise ValueError("original_positions must match original_to_merged shape")
    if state.original_positions.dtype not in (torch.int32, torch.int64, torch.long):
        raise TypeError("original_positions must be integer-valued")
    state_tensors = (
        state.original_to_merged,
        state.merge_weights,
        state.group_sizes,
        state.original_positions,
        state.merge_coordinates,
        state.merged_coordinates,
    )
    state_devices = {tensor.device for tensor in state_tensors if tensor is not None}
    if len(state_devices) != 1:
        raise ValueError("all tensors in merge state must reside on the same device")
    if state.merge_coordinates is not None and state.merge_coordinates.shape != (
        state.batch_size,
        state.original_to_merged.shape[1],
        2,
    ):
        raise ValueError("merge_coordinates must have shape [B,N,2]")
    if state.merged_coordinates is not None and state.merged_coordinates.shape != (
        state.batch_size,
        state.merged_count,
        2,
    ):
        raise ValueError("merged_coordinates must have shape [B,M,2]")
    if state.merged_coordinates is not None and not state.merged_coordinates.is_floating_point():
        raise TypeError("merged_coordinates must preserve fractional spatial positions")

    device = state.original_to_merged.device
    expected_positions = torch.arange(
        state.original_to_merged.shape[1], device=device, dtype=torch.long
    ).expand(state.batch_size, -1)
    if not torch.equal(state.original_positions.long(), expected_positions):
        raise ValueError("original_positions must preserve row-major token order")

    group_ids = torch.arange(state.merged_count, device=device).view(1, -1, 1)
    assignments = state.original_to_merged.unsqueeze(1)
    membership = group_ids == assignments
    actual_group_sizes = membership.sum(dim=-1)
    if torch.any(actual_group_sizes == 0):
        raise ValueError("every merged token must have at least one original assignment")
    if not torch.equal(actual_group_sizes.to(state.group_sizes.dtype), state.group_sizes):
        raise ValueError("group_sizes do not match original_to_merged assignments")

    expected_weights = actual_group_sizes.gather(
        1, state.original_to_merged.long()
    ).to(state.merge_weights.dtype).reciprocal()
    if not torch.isfinite(state.merge_weights).all() or not torch.allclose(
        state.merge_weights, expected_weights, rtol=1e-5, atol=1e-7
    ):
        raise ValueError("merge_weights do not match original_to_merged assignments")

    if state.merge_coordinates is not None:
        grid_coordinates = _build_coordinate_grid(state.grid_size, device)
        expected_coordinates = grid_coordinates.unsqueeze(0).expand(state.batch_size, -1, -1)
        if not torch.equal(state.merge_coordinates.long(), expected_coordinates):
            raise ValueError("merge_coordinates do not match the original row-major grid")
        if state.merged_coordinates is not None:
            expected_centroids = (
                membership.to(torch.float32)
                @ grid_coordinates.to(torch.float32).unsqueeze(0).expand(state.batch_size, -1, -1)
            ) / actual_group_sizes.to(torch.float32).unsqueeze(-1)
            if not torch.allclose(
                state.merged_coordinates.to(torch.float32),
                expected_centroids,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("merged_coordinates do not match merge assignments")


def merge_tokens(
    tokens: Tensor,
    *,
    grid_size: tuple[int, int] = (15, 15),
    merge_size: int | tuple[int, int] | None = None,
    merge_ratio: float = 0.0,
    method: str = "uniform",
) -> tuple[Tensor, TokenMergeState]:
    """Merge spatial patch tokens by fixed blocks or feature cosine similarity.

    The no-merge path is numerically identical to the baseline when
    `merge_ratio == 0.0` or `merge_size == (1, 1)`.

    When `merge_size` is omitted, `merge_ratio` acts as a retention target in the
    range `(0, 1)`. This keeps the API deterministic and makes positive ratios
    vary the achieved token count instead of collapsing into a fixed 2×2 merge.
    The adaptive method uses deterministic single-link clustering over pairwise
    cosine similarities and retains the nearest integer token budget.
    """

    if tokens.ndim != 3:
        raise ValueError(f"expected patch tokens [B,N,D], received {tuple(tokens.shape)}")
    if method not in ("uniform", "adaptive"):
        raise ValueError(f"unsupported merge method: {method!r}")

    batch_size, num_tokens, embedding_dim = _validate_input_tokens(tokens, grid_size)

    if not 0.0 <= float(merge_ratio) <= 1.0:
        raise ValueError(f"merge_ratio must be in [0,1], received {merge_ratio}")
    if method == "adaptive" and merge_size is not None:
        raise ValueError("merge_size is only supported by the uniform method")

    if merge_size is None and merge_ratio == 0.0:
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
            merged_coordinates=coordinates.to(dtype=torch.float32),
            merge_size=1,
            method=method,
            merge_ratio=float(merge_ratio),
            requested_token_count=num_tokens,
        )
        return tokens.clone(), state

    if method == "adaptive":
        num_groups = max(1, min(num_tokens, int(round(num_tokens * merge_ratio))))
        group_maps = _content_adaptive_group_map(tokens, num_groups)
        effective_merge_size = (1, 1)
    else:
        effective_merge_size = _resolve_group_shape(
            grid_size, merge_size=merge_size, merge_ratio=merge_ratio
        )
        if merge_ratio == 0.0 and merge_size is None:
            effective_merge_size = (1, 1)
        group_map = _build_group_map(
            grid_size, effective_merge_size, device=tokens.device
        )
        num_groups = (
            math.ceil(grid_size[0] / effective_merge_size[0])
            * math.ceil(grid_size[1] / effective_merge_size[1])
        )
        group_maps = group_map.unsqueeze(0).expand(batch_size, -1)

    merged = torch.empty(batch_size, num_groups, embedding_dim, device=tokens.device, dtype=tokens.dtype)
    original_to_merged = torch.empty(batch_size, num_tokens, device=tokens.device, dtype=torch.long)
    group_sizes = torch.empty(batch_size, num_groups, device=tokens.device, dtype=torch.long)
    merge_weights = torch.empty(batch_size, num_tokens, device=tokens.device, dtype=tokens.dtype)
    original_positions = torch.arange(num_tokens, device=tokens.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    coordinates = _build_coordinate_grid(grid_size, tokens.device).unsqueeze(0).expand(batch_size, -1, -1)
    merged_coordinates = torch.empty(
        batch_size, num_groups, 2, device=tokens.device, dtype=torch.float32
    )

    for batch_index in range(batch_size):
        batch_tokens = tokens[batch_index]
        batch_map = group_maps[batch_index]
        membership = (
            torch.arange(num_groups, device=tokens.device).unsqueeze(1)
            == batch_map.unsqueeze(0)
        )
        batch_sizes = membership.sum(dim=1)
        if torch.any(batch_sizes == 0):
            raise RuntimeError("merging produced an empty token group")
        merged[batch_index] = (
            membership.to(tokens.dtype) @ batch_tokens
        ) / batch_sizes.to(tokens.dtype).unsqueeze(1)
        merged_coordinates[batch_index] = (
            membership.to(merged_coordinates.dtype) @ coordinates[batch_index].to(torch.float32)
        ) / batch_sizes.to(merged_coordinates.dtype).unsqueeze(1)
        original_to_merged[batch_index] = batch_map
        group_sizes[batch_index] = batch_sizes
        per_group_sizes = group_sizes[batch_index, batch_map]
        merge_weights[batch_index] = 1.0 / per_group_sizes.to(tokens.dtype)

    state = TokenMergeState(
        grid_size=grid_size,
        original_to_merged=original_to_merged,
        merge_weights=merge_weights,
        group_sizes=group_sizes,
        original_positions=original_positions,
        merge_coordinates=coordinates,
        merged_coordinates=merged_coordinates,
        merge_size=effective_merge_size[0] * effective_merge_size[1],
        method=method,
        merge_ratio=float(merge_ratio),
        requested_token_count=(
            None
            if merge_size is not None
            else max(1, min(num_tokens, int(round(num_tokens * merge_ratio))))
        ),
    )

    _validate_merge_state(merged, state, expected_batch_size=batch_size)
    return merged, state


def reconstruct_tokens(merged_tokens: Tensor, state: TokenMergeState) -> Tensor:
    """Reconstruct the original row-major token ordering from merged tokens."""

    _validate_merge_state(merged_tokens, state)
    if state.original_to_merged.device != merged_tokens.device:
        state = TokenMergeState(
            grid_size=state.grid_size,
            original_to_merged=state.original_to_merged.to(merged_tokens.device),
            merge_weights=state.merge_weights.to(merged_tokens.device),
            group_sizes=state.group_sizes.to(merged_tokens.device),
            original_positions=state.original_positions.to(merged_tokens.device),
            merge_coordinates=state.merge_coordinates.to(merged_tokens.device) if state.merge_coordinates is not None else None,
            merged_coordinates=state.merged_coordinates.to(merged_tokens.device) if state.merged_coordinates is not None else None,
            merge_size=state.merge_size,
            method=state.method,
            merge_ratio=state.merge_ratio,
            requested_token_count=state.requested_token_count,
        )
    index = state.original_to_merged.unsqueeze(-1).expand(-1, -1, merged_tokens.shape[-1])
    reconstructed = merged_tokens.gather(1, index)
    if reconstructed.shape[1] != state.original_to_merged.shape[1]:
        raise ValueError(
            f"reconstruction token count mismatch: expected {state.original_to_merged.shape[1]}, got {reconstructed.shape[1]}"
        )
    return reconstructed


class UniformTokenMerger:
    """Deterministic, trainable-free spatial merger for feature-tensor prototypes."""

    def __init__(
        self,
        *,
        grid_size: tuple[int, int] = (15, 15),
        merge_size: int | tuple[int, int] | None = None,
        merge_ratio: float = 0.5,
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


class ContentAdaptiveTokenMerger:
    """Cluster tokens by feature cosine similarity with a requested retention ratio.

    Clustering is global within each sample, so similar tokens may be spatially
    non-adjacent. Grouping decisions are discrete; pooled token values retain
    gradients with respect to the input features.
    """

    def __init__(
        self,
        *,
        grid_size: tuple[int, int] = (15, 15),
        merge_ratio: float = 0.5,
    ) -> None:
        self.grid_size = grid_size
        self.merge_ratio = merge_ratio

    def __call__(self, tokens: Tensor) -> tuple[Tensor, TokenMergeState]:
        return merge_tokens(
            tokens,
            grid_size=self.grid_size,
            merge_ratio=self.merge_ratio,
            method="adaptive",
        )


class TokenReconstructor:
    """Minimal reconstruction wrapper for merged patch-token states."""

    def __call__(self, merged_tokens: Tensor, state: TokenMergeState) -> Tensor:
        return reconstruct_tokens(merged_tokens, state)


merger = merge_tokens
reconstructor = reconstruct_tokens

__all__ = [
    "TokenMergeState",
    "ContentAdaptiveTokenMerger",
    "TokenReconstructor",
    "UniformTokenMerger",
    "merge_tokens",
    "merger",
    "reconstruct_tokens",
    "reconstructor",
]
