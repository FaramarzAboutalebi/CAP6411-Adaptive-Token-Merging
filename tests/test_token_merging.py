"""Synthetic tests for the token-merging contract."""

from __future__ import annotations

import unittest

import torch

from cap6411_atm.merging import (
    ContentAdaptiveTokenMerger,
    TokenMergeState,
    merge_tokens,
    reconstruct_tokens,
)


class TokenMergingTests(unittest.TestCase):
    def test_merge_ratio_zero_matches_baseline(self) -> None:
        tokens = torch.randn(2, 225, 768)
        merged, state = merge_tokens(tokens, grid_size=(15, 15), merge_ratio=0.0)
        self.assertTrue(torch.equal(merged, tokens))
        self.assertEqual(state.merge_size, 1)
        self.assertTrue(torch.equal(state.original_to_merged, torch.arange(225).unsqueeze(0).expand(2, -1)))

    def test_active_merge_changes_token_budget_and_reconstruction_is_lossy(self) -> None:
        tokens = torch.arange(1 * 225 * 4, dtype=torch.float32).reshape(1, 225, 4)
        merged, state = merge_tokens(tokens, grid_size=(15, 15), merge_ratio=0.5)
        target_retained = round(225 * 0.5)
        self.assertLess(abs(merged.shape[1] - target_retained), 15)
        self.assertLess(merged.shape[1], tokens.shape[1])
        self.assertGreater(merged.shape[1], 1)
        reconstructed = reconstruct_tokens(merged, state)
        self.assertEqual(reconstructed.shape, tokens.shape)
        self.assertFalse(torch.allclose(reconstructed, tokens))

    def test_batch_elements_do_not_share_state(self) -> None:
        tokens = torch.arange(2 * 225 * 8, dtype=torch.float32).reshape(2, 225, 8)
        merged, state = merge_tokens(tokens, grid_size=(15, 15), merge_ratio=0.5)
        self.assertEqual(merged.shape[0], 2)
        self.assertEqual(state.original_to_merged.shape, (2, 225))
        self.assertEqual(state.group_sizes.shape[0], 2)
        self.assertEqual(state.original_positions.shape, (2, 225))
        self.assertEqual(state.merge_coordinates.shape, (2, 225, 2))

    def test_state_tracks_original_grid_and_merged_group_sizes(self) -> None:
        tokens = torch.randn(1, 225, 8)
        merged, state = merge_tokens(tokens, grid_size=(15, 15), merge_ratio=0.5)
        self.assertEqual(state.grid_size, (15, 15))
        self.assertGreaterEqual(int(state.group_sizes.max().item()), 1)
        self.assertEqual(state.original_positions.shape, (1, 225))
        self.assertEqual(state.merge_coordinates.shape, (1, 225, 2))
        self.assertLess(merged.shape[1], tokens.shape[1])

    def test_adaptive_merger_groups_content_similar_nonadjacent_tokens(self) -> None:
        first = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0],
              [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
        )
        merger = ContentAdaptiveTokenMerger(grid_size=(2, 3), merge_ratio=1 / 3)
        merged, state = merger(first)

        assignments = state.original_to_merged[0]
        self.assertEqual(merged.shape, (1, 2, 2))
        self.assertEqual(int(assignments[0]), int(assignments[2]))
        self.assertEqual(int(assignments[2]), int(assignments[4]))
        self.assertNotEqual(int(assignments[0]), int(assignments[1]))
        self.assertTrue(torch.equal(merged[0], torch.tensor([[1.0, 0.0], [0.0, 1.0]])))
        self.assertEqual(state.merged_coordinates.shape, (1, 2, 2))
        self.assertEqual(reconstruct_tokens(merged, state).shape, first.shape)

    def test_adaptive_assignments_change_when_token_content_changes(self) -> None:
        first = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0],
              [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
        )
        second = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
              [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]
        )
        _, first_state = merge_tokens(
            first, grid_size=(2, 3), merge_ratio=1 / 3, method="adaptive"
        )
        _, second_state = merge_tokens(
            second, grid_size=(2, 3), merge_ratio=1 / 3, method="adaptive"
        )

        first_assignments = first_state.original_to_merged[0]
        second_assignments = second_state.original_to_merged[0]
        self.assertEqual(int(first_assignments[0]), int(first_assignments[2]))
        self.assertNotEqual(int(first_assignments[0]), int(first_assignments[1]))
        self.assertEqual(int(second_assignments[0]), int(second_assignments[2]))
        self.assertEqual(int(second_assignments[0]), int(second_assignments[1]))
        self.assertNotEqual(int(second_assignments[0]), int(second_assignments[3]))

    def test_adaptive_merge_preserves_gradients_and_exact_token_budget(self) -> None:
        tokens = torch.randn(2, 12, 5, requires_grad=True)
        merged, state = merge_tokens(
            tokens, grid_size=(3, 4), merge_ratio=0.5, method="adaptive"
        )
        self.assertEqual(merged.shape, (2, 6, 5))
        self.assertEqual(state.group_sizes.sum(dim=1).tolist(), [12, 12])
        merged.square().sum().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_adaptive_method_rejects_spatial_merge_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supported by the uniform"):
            merge_tokens(
                torch.randn(1, 6, 2),
                grid_size=(2, 3),
                merge_size=2,
                merge_ratio=0.5,
                method="adaptive",
            )


if __name__ == "__main__":
    unittest.main()
