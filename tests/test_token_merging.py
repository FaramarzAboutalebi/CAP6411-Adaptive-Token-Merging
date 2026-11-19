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

    def test_merge_rejects_integer_token_inputs_for_all_methods(self) -> None:
        integer_tokens = torch.arange(4).reshape(1, 4, 1)
        for method in ("uniform", "adaptive"):
            for merge_ratio in (0.0, 0.5):
                with self.subTest(method=method, merge_ratio=merge_ratio):
                    with self.assertRaisesRegex(TypeError, "floating-point"):
                        merge_tokens(
                            integer_tokens,
                            grid_size=(2, 2),
                            merge_ratio=merge_ratio,
                            method=method,
                        )

    def test_merge_rejects_nan_and_infinite_token_values(self) -> None:
        for invalid_value in (float("nan"), float("inf"), float("-inf")):
            tokens = torch.tensor([[[1.0], [invalid_value], [3.0], [4.0]]])
            for method in ("uniform", "adaptive"):
                for merge_ratio in (0.0, 0.5):
                    with self.subTest(
                        invalid_value=invalid_value,
                        method=method,
                        merge_ratio=merge_ratio,
                    ):
                        with self.assertRaisesRegex(ValueError, "finite"):
                            merge_tokens(
                                tokens,
                                grid_size=(2, 2),
                                merge_ratio=merge_ratio,
                                method=method,
                            )

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

    def test_uniform_retention_metrics_report_requested_and_achieved_counts(self) -> None:
        tokens = torch.zeros(1, 225, 2)
        for requested_ratio in (0.25, 0.5, 0.75):
            with self.subTest(requested_ratio=requested_ratio):
                merged, state = merge_tokens(
                    tokens, grid_size=(15, 15), merge_ratio=requested_ratio
                )
                report = state.retention_metrics
                self.assertEqual(report["original_token_count"], 225)
                self.assertEqual(
                    report["requested_token_count"],
                    max(1, round(225 * requested_ratio)),
                )
                self.assertEqual(report["achieved_token_count"], merged.shape[1])
                self.assertAlmostEqual(
                    report["achieved_retention_ratio"], merged.shape[1] / 225
                )
                self.assertAlmostEqual(
                    report["retention_ratio_error"],
                    report["achieved_retention_ratio"]
                    - report["requested_retention_ratio"],
                )
                self.assertEqual(
                    report["achieved_token_count"], report["requested_token_count"]
                )
                self.assertEqual(report["retention_ratio_error"], 0.0)

    def test_retention_ratio_name_and_legacy_alias_are_equivalent(self) -> None:
        tokens = torch.randn(1, 12, 3)
        preferred, preferred_state = merge_tokens(
            tokens, grid_size=(3, 4), retention_ratio=0.5
        )
        legacy, legacy_state = merge_tokens(
            tokens, grid_size=(3, 4), merge_ratio=0.5
        )
        self.assertTrue(torch.equal(preferred, legacy))
        self.assertEqual(preferred_state.retention_metrics, legacy_state.retention_metrics)
        with self.assertRaisesRegex(ValueError, "aliases disagree"):
            merge_tokens(
                tokens,
                grid_size=(3, 4),
                retention_ratio=0.5,
                merge_ratio=0.25,
            )

    def test_explicit_uniform_merge_size_reports_no_ratio_target(self) -> None:
        _, state = merge_tokens(
            torch.randn(1, 225, 2),
            grid_size=(15, 15),
            merge_size=2,
            merge_ratio=0.5,
        )
        report = state.retention_metrics
        self.assertIsNone(report["requested_token_count"])
        self.assertIsNone(report["requested_retention_ratio"])
        self.assertEqual(report["achieved_token_count"], state.merged_count)
        self.assertAlmostEqual(
            report["achieved_retention_ratio"], state.merged_count / 225
        )

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

    def test_uniform_merge_runs_on_available_devices_and_preserves_gradients(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            devices.append(torch.device("mps"))

        for device in devices:
            with self.subTest(device=str(device)):
                tokens = torch.arange(
                    18, device=device, dtype=torch.float32
                ).reshape(2, 9, 1).requires_grad_()
                merged, state = merge_tokens(
                    tokens,
                    grid_size=(3, 3),
                    merge_size=2,
                    merge_ratio=1.0,
                    method="uniform",
                )
                self.assertEqual(merged.device.type, device.type)
                self.assertEqual(state.original_to_merged.device.type, device.type)
                self.assertEqual(state.group_sizes.device.type, device.type)
                self.assertEqual(merged.shape, (2, 4, 1))
                self.assertTrue(state.merged_coordinates.is_floating_point())
                self.assertTrue(
                    torch.equal(
                        state.merged_coordinates[0],
                        torch.tensor(
                            [[0.5, 0.5], [0.5, 2.0], [2.0, 0.5], [2.0, 2.0]],
                            device=device,
                        ),
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        merged[0, :, 0],
                        torch.tensor([2.0, 3.5, 6.5, 8.0], device=device),
                    )
                )
                reconstructed = reconstruct_tokens(merged, state)
                self.assertEqual(reconstructed.device.type, device.type)
                self.assertEqual(reconstructed.shape, tokens.shape)
                merged.sum().backward()
                self.assertIsNotNone(tokens.grad)
                self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_reconstruction_rejects_extra_merged_tokens(self) -> None:
        merged, state = merge_tokens(
            torch.randn(1, 9, 3),
            grid_size=(3, 3),
            merge_size=2,
            merge_ratio=1.0,
        )
        extra_token = torch.cat((merged, merged[:, :1]), dim=1)
        with self.assertRaisesRegex(ValueError, "merged token count"):
            reconstruct_tokens(extra_token, state)

    def test_reconstruction_rejects_out_of_range_assignment(self) -> None:
        merged, state = merge_tokens(
            torch.randn(1, 9, 3),
            grid_size=(3, 3),
            merge_size=2,
            merge_ratio=1.0,
        )
        state.original_to_merged[0, 0] = state.merged_count
        with self.assertRaisesRegex(ValueError, "outside"):
            reconstruct_tokens(merged, state)

    def test_reconstruction_rejects_group_sizes_that_disagree_with_assignments(self) -> None:
        merged, state = merge_tokens(
            torch.randn(1, 9, 3),
            grid_size=(3, 3),
            merge_size=2,
            merge_ratio=1.0,
        )
        state.group_sizes[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "group_sizes do not match"):
            reconstruct_tokens(merged, state)

    def test_reconstruction_rejects_inconsistent_merge_weights(self) -> None:
        merged, state = merge_tokens(
            torch.randn(1, 9, 3),
            grid_size=(3, 3),
            merge_size=2,
            merge_ratio=1.0,
        )
        state.merge_weights[0, 0] = 0.75
        with self.assertRaisesRegex(ValueError, "merge_weights do not match"):
            reconstruct_tokens(merged, state)

    def test_reconstruction_rejects_inconsistent_centroids(self) -> None:
        merged, state = merge_tokens(
            torch.randn(1, 9, 3),
            grid_size=(3, 3),
            merge_size=2,
            merge_ratio=1.0,
        )
        state.merged_coordinates[0, 0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "merged_coordinates do not match"):
            reconstruct_tokens(merged, state)

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
        self.assertTrue(state.merged_coordinates.is_floating_point())
        self.assertTrue(
            torch.allclose(
                state.merged_coordinates[0],
                torch.tensor([[1 / 3, 1.0], [2 / 3, 1.0]]),
            )
        )
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
