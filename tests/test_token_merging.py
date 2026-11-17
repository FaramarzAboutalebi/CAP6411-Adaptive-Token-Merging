"""Synthetic tests for the token-merging contract."""

from __future__ import annotations

import unittest

import torch

from cap6411_atm.merging import TokenMergeState, merge_tokens, reconstruct_tokens


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


if __name__ == "__main__":
    unittest.main()
