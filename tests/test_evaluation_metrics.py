"""Known-answer tests for Workstream 4 metrics and post-processing (no data needed)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from cap6411_atm.evaluation.metrics import (
    BRATS_VOLUME_DIAGONAL_MM,
    MetricConfig,
    patch_class_errors,
    patient_metrics,
    region_metrics,
)
from cap6411_atm.evaluation.postprocess import (
    apply_threshold,
    build_postprocessor,
    clear_small_totals,
    enforce_nesting,
    remove_small_components,
)


def cube(shape=(48, 48, 32), start=(10, 10, 10), size=10) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x, y, z = start
    mask[x : x + size, y : y + size, z : z + size] = True
    return mask


class OverlapAndSurfaceTests(unittest.TestCase):
    def test_identical_masks_are_perfect(self) -> None:
        gt = cube()
        result = region_metrics(gt, gt)
        self.assertEqual(result["status"], "both_present")
        self.assertEqual(result["dice"], 1.0)
        self.assertEqual(result["hd95"], 0.0)
        self.assertEqual(result["assd"], 0.0)
        self.assertEqual(result["nsd_1mm"], 1.0)
        self.assertEqual(result["fp"] + result["fn"], 0)

    def test_shifted_cube_dice_and_hd95(self) -> None:
        gt = cube()
        pred = cube(start=(13, 10, 10))
        result = region_metrics(pred, gt)
        self.assertAlmostEqual(result["dice"], 0.7)  # 7x10x10 overlap of two 1000-voxel cubes
        self.assertAlmostEqual(result["hd95"], 3.0)
        self.assertGreater(result["assd"], 0.0)
        self.assertLess(result["assd"], 3.0)
        self.assertEqual(result["tp"], 700)
        self.assertEqual(result["fp"], 300)
        self.assertEqual(result["fn"], 300)

    def test_nsd_tolerance(self) -> None:
        gt = cube()
        pred = cube(start=(11, 10, 10))
        config = MetricConfig(nsd_tolerances_mm=(0.5, 1.0))
        result = region_metrics(pred, gt, config)
        self.assertEqual(result["nsd_1mm"], 1.0)
        self.assertLess(result["nsd_0p5mm"], 1.0)

    def test_anisotropic_spacing_scales_distances(self) -> None:
        gt = cube()
        pred = cube(start=(13, 10, 10))
        result = region_metrics(pred, gt, MetricConfig(spacing=(2.0, 1.0, 1.0)))
        self.assertAlmostEqual(result["hd95"], 6.0)

    def test_empty_conventions(self) -> None:
        empty = np.zeros((32, 32, 8), dtype=bool)
        both = region_metrics(empty, empty)
        self.assertEqual((both["status"], both["dice"], both["hd95"], both["nsd_2mm"]), ("both_empty", 1.0, 0.0, 1.0))
        self.assertTrue(math.isnan(both["sensitivity"]))

        pred = empty.copy()
        pred[4:8, 4:8, 2:4] = True
        false_alarm = region_metrics(pred, empty)
        self.assertEqual(false_alarm["status"], "gt_empty")
        self.assertEqual(false_alarm["dice"], 0.0)
        self.assertAlmostEqual(false_alarm["hd95"], BRATS_VOLUME_DIAGONAL_MM)
        self.assertEqual(false_alarm["nsd_2mm"], 0.0)
        self.assertEqual(false_alarm["unbanded_fp"], 32)
        self.assertEqual(false_alarm["fp_components"], 1)

        missed = region_metrics(empty, pred)
        self.assertEqual(missed["status"], "pred_empty")
        self.assertEqual(missed["sensitivity"], 0.0)
        self.assertTrue(math.isnan(missed["precision"]))

    def test_shape_validation(self) -> None:
        with self.assertRaises(ValueError):
            region_metrics(np.zeros((4, 4, 4)), np.zeros((4, 4, 5)))
        with self.assertRaises(ValueError):
            patient_metrics(np.zeros((2, 16, 16, 2)), np.zeros((2, 16, 16, 2)))


class BoundaryAnalysisTests(unittest.TestCase):
    def test_one_voxel_shift_errors_lie_next_to_boundary(self) -> None:
        gt = cube()
        pred = cube(start=(11, 10, 10))
        result = region_metrics(pred, gt)
        self.assertEqual(result["band_0_2mm_fp"] + result["band_0_2mm_fn"], result["fp"] + result["fn"])
        self.assertEqual(result["band_ge8mm_fp"], 0)
        self.assertGreater(result["band_0_2mm_voxels"], 0)

    def test_distant_false_positive_lands_in_far_band(self) -> None:
        gt = cube(shape=(64, 64, 32))
        pred = gt.copy()
        pred[50:53, 50:53, 15:17] = True
        result = region_metrics(pred, gt)
        self.assertEqual(result["band_ge8mm_fp"], 18)
        self.assertEqual(result["band_0_2mm_fp"], 0)

    def test_patch_classes(self) -> None:
        gt = np.zeros((32, 32, 2), dtype=bool)
        gt[0:16, 0:16, 0] = True  # one interior patch
        gt[16:24, 0:16, 0] = True  # one boundary patch
        result = patch_class_errors(gt, gt)
        self.assertEqual(result["patch_interior_count"], 1)
        self.assertEqual(result["patch_boundary_count"], 1)
        self.assertEqual(result["patch_background_count"], 6)
        self.assertEqual(result["patch_boundary_errors"], 0)

        pred = gt.copy()
        pred[16:20, 0:16, 0] = False  # 64 FN pixels inside the boundary patch
        result = patch_class_errors(pred, gt)
        self.assertEqual(result["patch_boundary_errors"], 64)
        self.assertAlmostEqual(result["patch_boundary_error_rate"], 64 / 256)
        self.assertEqual(result["patch_interior_errors"], 0)

    def test_patch_size_must_divide_grid(self) -> None:
        with self.assertRaises(ValueError):
            patch_class_errors(np.zeros((30, 32, 1)), np.zeros((30, 32, 1)))

    def test_lesion_detection_by_size(self) -> None:
        gt = np.zeros((64, 64, 32), dtype=bool)
        gt[2:4, 2:4, 2:4] = True  # 8-voxel lesion, missed
        gt[20:30, 20:30, 10:20] = True  # 1000-voxel lesion, detected
        pred = np.zeros_like(gt)
        pred[22:28, 22:28, 12:18] = True
        pred[50:52, 50:52, 25:27] = True  # false-positive component
        result = region_metrics(pred, gt)
        self.assertEqual(result["lesion_0_100_count"], 1)
        self.assertEqual(result["lesion_0_100_detected"], 0)
        self.assertEqual(result["lesion_1000_10000_count"], 1)
        self.assertEqual(result["lesion_1000_10000_detected"], 1)
        self.assertEqual(result["gt_components"], 2)
        self.assertEqual(result["fp_components"], 1)
        self.assertEqual(result["fp_component_voxels"], 8)


class PostProcessingTests(unittest.TestCase):
    def test_threshold_is_inclusive_and_per_region(self) -> None:
        probabilities = np.full((3, 2, 2, 1), 0.5, dtype=np.float32)
        self.assertTrue(apply_threshold(probabilities, 0.5).all())
        masks = apply_threshold(probabilities, {"WT": 0.4, "TC": 0.6, "ET": 0.5})
        self.assertTrue(masks[0].all())
        self.assertFalse(masks[1].any())

    def test_nesting(self) -> None:
        masks = np.zeros((3, 4, 4, 1), dtype=bool)
        masks[0, :2] = True
        masks[1, :3] = True
        masks[2, :] = True
        nested = enforce_nesting(masks)
        self.assertTrue(np.array_equal(nested[1], masks[0]))
        self.assertTrue(np.array_equal(nested[2], masks[0]))
        self.assertTrue(np.all(nested[2] <= nested[1]) and np.all(nested[1] <= nested[0]))

    def test_small_component_and_total_removal(self) -> None:
        masks = np.zeros((3, 32, 32, 8), dtype=bool)
        masks[2, 0:2, 0:2, 0:2] = True  # 8 voxels
        masks[2, 10:20, 10:20, 2:4] = True  # 200 voxels
        cleaned = remove_small_components(masks, {"ET": 50})
        self.assertEqual(int(cleaned[2].sum()), 200)
        self.assertEqual(int(masks[2].sum()), 208, "input must not be modified")
        self.assertFalse(clear_small_totals(cleaned, {"ET": 201})[2].any())
        self.assertEqual(int(clear_small_totals(cleaned, {"ET": 200})[2].sum()), 200)

    def test_spec_validation(self) -> None:
        with self.assertRaises(ValueError):
            build_postprocessor({"thresh": 0.5})
        with self.assertRaises(ValueError):
            build_postprocessor({"min_total": {"NCR": 10}})
        rule = build_postprocessor({"threshold": 0.5, "nesting": True})
        self.assertEqual(rule(np.full((3, 2, 2, 1), 0.9)).shape, (3, 2, 2, 1))


if __name__ == "__main__":
    unittest.main()
