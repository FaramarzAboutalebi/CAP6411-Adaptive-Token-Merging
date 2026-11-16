"""Weight-free tests for Workstream 3 benchmarking and the reduced-resolution baseline."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cap6411_atm.benchmark import (
    DEFAULT_RESOLUTIONS,
    ReducedResolutionSegmenter,
    attach_quality,
    benchmark,
    benchmark_stages,
    collect_env,
    describe,
    evaluate_quality,
    evaluate_quality_many,
    load_timing_only_dinov3_vitb16,
    match_latency,
    plot_tradeoff,
    read_rows,
    resolution_sweep,
    run_suite,
    supports_stage_timing,
    write_env,
    write_rows,
)
from cap6411_atm.benchmark.env import git_is_dirty
from cap6411_atm.models import BraTSDINOv3Segmenter
from test_dinov3_segmentation import SyntheticDINOv3ViTB16


def _model() -> BraTSDINOv3Segmenter:
    torch.manual_seed(0)
    return BraTSDINOv3Segmenter(SyntheticDINOv3ViTB16()).eval()


class ReducedResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _model()
        self.images = torch.randn(2, 4, 240, 240)

    def test_every_resolution_keeps_the_output_contract(self) -> None:
        for size in DEFAULT_RESOLUTIONS:
            wrapper = ReducedResolutionSegmenter(self.model, size)
            with torch.no_grad():
                logits = wrapper(self.images)
            self.assertEqual(logits.shape, (2, 3, 240, 240), size)
            self.assertTrue(torch.isfinite(logits).all(), size)
            self.assertEqual(wrapper.token_count, (size // 16) ** 2)

    def test_full_resolution_is_exactly_the_baseline(self) -> None:
        wrapper = ReducedResolutionSegmenter(self.model, 240)
        with torch.no_grad():
            self.assertTrue(torch.equal(wrapper(self.images), self.model(self.images)))

    def test_encoder_really_sees_fewer_tokens(self) -> None:
        wrapper = ReducedResolutionSegmenter(self.model, 160)
        with torch.no_grad():
            features = wrapper.encode_adapted(
                self.model.adapter(wrapper.downsample(self.images))
            )
        self.assertEqual(wrapper.downsample(self.images).shape[-2:], (160, 160))
        self.assertTrue(all(f.shape == (2, 768, 15, 15) for f in features))
        backbone_call = self.model.encoder.backbone.last_call
        self.assertEqual(backbone_call["n"], [2, 5, 8, 11])
        self.assertTrue(backbone_call["norm"] and backbone_call["reshape"])
        self.assertFalse(backbone_call["return_class_token"] or backbone_call["return_extra_tokens"])

    def test_invalid_resolutions_are_rejected(self) -> None:
        for size in (0, 150, 256):
            with self.assertRaises(ValueError):
                ReducedResolutionSegmenter(self.model, size)
        with self.assertRaises(ValueError):
            ReducedResolutionSegmenter(self.model, 160)(torch.randn(1, 4, 224, 224))

    def test_wrapper_keeps_backbone_frozen_in_eval_mode(self) -> None:
        wrapper = ReducedResolutionSegmenter(self.model, 160)
        wrapper.train()
        self.assertFalse(self.model.encoder.backbone.training)
        self.assertTrue(self.model.decoder.training)

    def test_sweep_shares_one_model(self) -> None:
        sweep = resolution_sweep(self.model, (240, 160))
        self.assertEqual(list(sweep), ["res240", "res160"])
        self.assertIs(sweep["res240"].base, sweep["res160"].base)


class LatencyTests(unittest.TestCase):
    def test_benchmark_reports_measured_statistics(self) -> None:
        result = benchmark(nn.Conv2d(4, 3, 1), torch.randn(2, 4, 32, 32), warmup=2, iters=5)
        self.assertEqual(result["batch_size"], 2)
        self.assertEqual(result["precision"], "fp32")
        self.assertGreater(result["latency_median_ms"], 0)
        self.assertLessEqual(result["latency_min_ms"], result["latency_median_ms"])
        self.assertLessEqual(result["latency_median_ms"], result["latency_p90_ms"])
        self.assertTrue(math.isnan(result["peak_memory_mb"]))  # CPU: not measured

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            benchmark(nn.Identity(), torch.zeros(1), iters=0)
        with self.assertRaises(ValueError):
            benchmark(nn.Identity(), (torch.zeros(1),))

    def test_stage_timing_for_baseline_and_wrapper(self) -> None:
        model = _model()
        images = torch.randn(1, 4, 240, 240)
        for variant in (model, ReducedResolutionSegmenter(model, 160)):
            stages = benchmark_stages(variant, images, warmup=1, iters=2)
            self.assertEqual(list(stages), ["adapter", "encoder", "decoder"])
            self.assertTrue(all(s["batch_size"] == 1 for s in stages.values()))


class CompareTests(unittest.TestCase):
    def test_run_suite_rows(self) -> None:
        rows = run_suite(
            resolution_sweep(_model(), (240, 160)),
            batch_sizes=(1, 2),
            device="cpu",
            warmup=1,
            iters=2,
            stages=True,
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["family"] for r in rows}, {"baseline", "reduced_resolution"})
        self.assertEqual({r["tokens_theoretical"] for r in rows}, {225, 100})
        self.assertTrue(all("encoder_median_ms" in r for r in rows))

    def test_describe_uses_workstream2_attributes(self) -> None:
        merged = nn.Identity()
        merged.benchmark_family = "adaptive_merge"
        merged.benchmark_setting = "r=0.5"
        merged.token_count = 113
        info = describe(merged)
        self.assertEqual((info["family"], info["setting"], info["tokens_theoretical"]),
                         ("adaptive_merge", "r=0.5", 113))

    def test_match_latency_picks_nearest_reference(self) -> None:
        rows = [
            {"variant": "res240", "family": "baseline", "batch_size": 1, "latency_median_ms": 10.0, "macro_dice": 0.82},
            {"variant": "res192", "family": "reduced_resolution", "batch_size": 1, "latency_median_ms": 7.0, "macro_dice": 0.78},
            {"variant": "res160", "family": "reduced_resolution", "batch_size": 1, "latency_median_ms": 5.0, "macro_dice": 0.74},
            {"variant": "ada50", "family": "adaptive_merge", "batch_size": 1, "latency_median_ms": 6.6, "macro_dice": 0.80},
            {"variant": "ada50", "family": "adaptive_merge", "batch_size": 2, "latency_median_ms": 9.0},
        ]
        (pair,) = match_latency(rows)
        self.assertEqual(pair["matched_variant"], "res192")
        self.assertAlmostEqual(pair["macro_dice_delta"], 0.02)
        self.assertAlmostEqual(pair["latency_gap_pct"], 100 * (6.6 - 7.0) / 7.0)

    def test_evaluate_quality_matches_streaming_dice(self) -> None:
        targets = torch.zeros(2, 3, 240, 240)
        targets[:, :, 100:140, 100:140] = 1.0

        class Perfect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(()))

            def forward(self, images):
                return images[:, :3] * 20 - 10

        loader = [{"image": torch.cat([targets, targets[:, :1]], 1), "target": targets}]
        result = evaluate_quality(Perfect(), loader, device="cpu")
        self.assertAlmostEqual(result["macro_dice"], 1.0)
        self.assertEqual(result["samples"], 2)

    def test_results_round_trip_and_plot(self) -> None:
        rows = [
            {"variant": "res240", "family": "baseline", "setting": "240px", "batch_size": 1, "latency_median_ms": 10.0},
            {"variant": "res160", "family": "reduced_resolution", "setting": "160px", "batch_size": 1, "latency_median_ms": 5.0},
        ]
        attach_quality(rows, {"res240": {"macro_dice": 0.82, "dice": {"WT": 0.87, "TC": 0.86, "ET": 0.73}},
                              "res160": {"macro_dice": 0.74, "dice": {"WT": 0.8, "TC": 0.75, "ET": 0.67}}})
        with tempfile.TemporaryDirectory() as directory:
            path = write_rows(rows, Path(directory) / "results.csv")
            loaded = read_rows(path)
            self.assertEqual(loaded[1]["latency_median_ms"], 5.0)
            self.assertEqual(loaded[0]["dice_ET"], 0.73)
            figure = plot_tradeoff(loaded, Path(directory) / "tradeoff.png")
            self.assertGreater(figure.stat().st_size, 0)
            env_path = write_env(Path(directory) / "env.json", collect_env("cpu", weights="synthetic"))
            env = json.loads(env_path.read_text())
            self.assertEqual(env["weights"], "synthetic")
            self.assertIn("torch", env)


class ReviewRegressionTests(unittest.TestCase):
    """Guards for issues found in code review."""

    def test_round_trip_with_partial_dice_still_matches(self) -> None:
        rows = [
            {"variant": "res240", "family": "baseline", "setting": "240px", "batch_size": 1, "latency_median_ms": 10.0},
            {"variant": "res192", "family": "reduced_resolution", "setting": "192px", "batch_size": 1, "latency_median_ms": 7.0},
            {"variant": "ada50", "family": "adaptive_merge", "setting": "0.5", "batch_size": 1, "latency_median_ms": 6.8},
        ]
        attach_quality(rows, {"ada50": {"macro_dice": 0.8, "dice": {"WT": 0.8, "TC": 0.8, "ET": 0.8}}})
        with tempfile.TemporaryDirectory() as directory:
            loaded = read_rows(write_rows(rows, Path(directory) / "r.csv"))
            self.assertNotIn("macro_dice", loaded[1])
            (pair,) = match_latency(loaded)
            self.assertEqual(pair["matched_variant"], "res192")
            self.assertNotIn("macro_dice_delta", pair)
            plot_tradeoff(loaded, Path(directory) / "p.png")  # only the Dice row is plotted

    def test_unlabeled_variant_is_compared_not_used_as_reference(self) -> None:
        self.assertEqual(describe(nn.Identity())["family"], "unlabeled")
        self.assertEqual(describe(_model())["family"], "baseline")
        rows = [
            {"variant": "res240", "family": "baseline", "batch_size": 1, "latency_median_ms": 10.0},
            {"variant": "mystery", "family": "unlabeled", "batch_size": 1, "latency_median_ms": 9.0},
        ]
        (pair,) = match_latency(rows)
        self.assertEqual((pair["variant"], pair["matched_variant"]), ("mystery", "res240"))

    def test_quality_records_scope_and_reads_data_once(self) -> None:
        reads = []

        class CountingLoader:
            scope = "validation pilot (1 patients)"

            def __iter__(self):
                for index in range(3):
                    reads.append(index)
                    yield {
                        "image": torch.randn(2, 4, 240, 240),
                        "target": (torch.rand(2, 3, 240, 240) > 0.5).float(),
                        "patient_id": ["p1", "p1"],
                    }

        model = _model()
        results = evaluate_quality_many(resolution_sweep(model, (240, 160)), CountingLoader(), device="cpu")
        self.assertEqual(len(reads), 3)  # one pass for two variants
        self.assertEqual(results["res240"]["scope"], "validation pilot (1 patients)")
        self.assertEqual((results["res160"]["patients"], results["res160"]["samples"]), (1, 6))
        truncated = evaluate_quality(model, CountingLoader(), device="cpu", max_batches=1)
        self.assertIn("truncated", truncated["scope"])
        rows = attach_quality([{"variant": "res240"}], results)
        self.assertEqual(rows[0]["dice_scope"], "validation pilot (1 patients)")

    def test_untracked_files_mark_tree_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            run = lambda *a: subprocess.run(["git", "-C", directory, *a], check=True, capture_output=True)
            run("init", "-q")
            (repo / "a.py").write_text("x = 1\n")
            run("add", "a.py")
            run("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")
            self.assertFalse(git_is_dirty(repo))
            (repo / "new_module.py").write_text("y = 2\n")
            self.assertTrue(git_is_dirty(repo))

    def test_stage_timing_only_for_known_forward_paths(self) -> None:
        class MergingSubclass(BraTSDINOv3Segmenter):
            benchmark_family = "adaptive_merge"

        merged = MergingSubclass(SyntheticDINOv3ViTB16())
        self.assertFalse(supports_stage_timing(merged))
        rows = run_suite({"merged": merged}, batch_sizes=(1,), device="cpu", warmup=0, iters=1, stages=True)
        self.assertNotIn("encoder_median_ms", rows[0])

        merged.benchmark_stage_functions = lambda: (
            ("adapter", merged.adapter), ("encoder", merged.encoder), ("decoder", merged.decoder)
        )
        self.assertTrue(supports_stage_timing(merged))

    def test_plot_does_not_change_matplotlib_backend(self) -> None:
        import matplotlib

        before = matplotlib.get_backend()
        rows = [{"variant": "res240", "family": "baseline", "setting": "240px", "batch_size": 1,
                 "latency_median_ms": 1.0, "macro_dice": 0.8}]
        with tempfile.TemporaryDirectory() as directory:
            plot_tradeoff(rows, Path(directory) / "p.png")
        self.assertEqual(matplotlib.get_backend(), before)

    def test_timing_only_loader_requires_git_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_timing_only_dinov3_vitb16(directory)

    def test_cli_output_defaults_to_repo_outputs(self) -> None:
        from cap6411_atm.benchmark.compare import _parser
        from cap6411_atm.project import PROJECT_ROOT

        args = _parser().parse_args(["--dinov3-repo", "x"])
        self.assertEqual(Path(args.out), PROJECT_ROOT / "outputs" / "benchmarks")


class SecondReviewRegressionTests(unittest.TestCase):
    """Guards for issues found in the second code review."""

    def test_plot_tolerates_rows_without_setting(self) -> None:
        rows = [{"variant": "m", "family": "adaptive_merge", "setting": "", "batch_size": 1,
                 "latency_median_ms": 2.0, "macro_dice": 0.7}]
        with tempfile.TemporaryDirectory() as directory:
            loaded = read_rows(write_rows(rows, Path(directory) / "r.csv"))
            self.assertNotIn("setting", loaded[0])
            plot_tradeoff(loaded, Path(directory) / "p.png")

    def test_mergers_built_on_the_wrapper_are_not_references(self) -> None:
        class MergedLowRes(ReducedResolutionSegmenter):
            pass

        model = _model()
        self.assertEqual(describe(MergedLowRes(model, 160))["family"], "unlabeled")
        labeled = ReducedResolutionSegmenter(model, 160)
        labeled.benchmark_family = "adaptive_merge"
        self.assertEqual(describe(labeled)["family"], "adaptive_merge")
        self.assertEqual(describe(ReducedResolutionSegmenter(model, 160))["family"],
                         "reduced_resolution")

    def test_match_latency_requires_same_conditions(self) -> None:
        base = {"batch_size": 1, "device": "cuda", "input": "random"}
        rows = [
            {**base, "variant": "res240_fp32", "family": "baseline", "precision": "fp32", "latency_median_ms": 6.0},
            {**base, "variant": "res160", "family": "reduced_resolution", "precision": "fp16-autocast", "latency_median_ms": 4.0},
            {**base, "variant": "ada", "family": "adaptive_merge", "precision": "fp16-autocast", "latency_median_ms": 5.9},
            {**base, "variant": "ada_mri", "family": "adaptive_merge", "precision": "fp16-autocast",
             "input": "mri_slices", "latency_median_ms": 5.9},
        ]
        with self.assertWarns(UserWarning):
            pairs = match_latency(rows)
        self.assertEqual([(p["variant"], p["matched_variant"]) for p in pairs], [("ada", "res160")])

    def test_numeric_looking_identifiers_stay_text(self) -> None:
        rows = [{"variant": "0.5", "family": "adaptive_merge", "setting": "113", "batch_size": 1,
                 "latency_median_ms": 3.0}]
        with tempfile.TemporaryDirectory() as directory:
            loaded = read_rows(write_rows(rows, Path(directory) / "r.csv"))
        self.assertEqual((loaded[0]["variant"], loaded[0]["setting"]), ("0.5", "113"))
        self.assertEqual(loaded[0]["batch_size"], 1)
        attach_quality(loaded, {"0.5": {"macro_dice": 0.9, "dice": {"WT": 0.9}}})
        self.assertEqual(loaded[0]["macro_dice"], 0.9)

    def test_max_batches_reads_no_extra_batch(self) -> None:
        reads = []

        class Loader:
            scope = "validation pilot"

            def __len__(self):
                return 5

            def __iter__(self):
                for index in range(5):
                    reads.append(index)
                    yield {"image": torch.randn(1, 4, 240, 240),
                           "target": torch.zeros(1, 3, 240, 240), "patient_id": ["p"]}

        result = evaluate_quality(_model(), Loader(), device="cpu", max_batches=2)
        self.assertEqual(reads, [0, 1])
        self.assertIn("truncated", result["scope"])
        reads.clear()
        result = evaluate_quality(_model(), Loader(), device="cpu", max_batches=5)
        self.assertEqual(result["scope"], "validation pilot")  # cap not binding

    def test_wrapper_calls_backbone_exactly_like_the_encoder(self) -> None:
        model = _model()
        images = torch.randn(1, 4, 240, 240)
        with torch.no_grad():
            model.encoder(model.adapter(images))
            encoder_call = dict(model.encoder.backbone.last_call)
            ReducedResolutionSegmenter(model, 160)(images)
        self.assertEqual(model.encoder.backbone.last_call, encoder_call)

    def test_wrapper_checks_feature_count(self) -> None:
        model = _model()
        wrapper = ReducedResolutionSegmenter(model, 160)
        original = model.encoder.backbone.get_intermediate_layers
        model.encoder.backbone.get_intermediate_layers = lambda *a, **k: original(*a, **k)[:3]
        with self.assertRaises(RuntimeError):
            wrapper(torch.randn(1, 4, 240, 240))


class FeatureResizeTests(unittest.TestCase):
    def test_matmul_resize_equals_bilinear_interpolate(self) -> None:
        from torch.nn import functional as F

        from cap6411_atm.benchmark.resolution import resize_feature_map

        for grid in (8, 10, 12, 14):
            feature = torch.randn(3, 768, grid, grid)
            expected = F.interpolate(feature, size=(15, 15), mode="bilinear", align_corners=False)
            actual = resize_feature_map(feature, (15, 15))
            self.assertEqual(actual.shape, (3, 768, 15, 15))
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_identity_grid_is_untouched_and_dtype_kept(self) -> None:
        from cap6411_atm.benchmark.resolution import resize_feature_map

        feature = torch.randn(1, 4, 15, 15)
        self.assertIs(resize_feature_map(feature, (15, 15)), feature)
        half = torch.randn(1, 4, 10, 10, dtype=torch.float64)
        self.assertEqual(resize_feature_map(half, (15, 15)).dtype, torch.float64)


class TableAndConfigTests(unittest.TestCase):
    ROWS = [
        {"variant": "res240", "family": "baseline", "batch_size": 1, "device": "cuda",
         "precision": "fp16-autocast", "input": "random", "tokens_theoretical": 225,
         "latency_median_ms": 10.0, "latency_p90_ms": 10.5, "peak_memory_mb": 500.0,
         "macro_dice": 0.819, "dice_WT": 0.869, "dice_TC": 0.856, "dice_ET": 0.732},
        {"variant": "res160", "family": "reduced_resolution", "batch_size": 1, "device": "cuda",
         "precision": "fp16-autocast", "input": "random", "tokens_theoretical": 100,
         "latency_median_ms": 5.0, "latency_p90_ms": 5.2, "peak_memory_mb": 300.0},
    ]

    def test_results_table_markdown(self) -> None:
        from cap6411_atm.benchmark import results_table

        table = results_table(self.ROWS).splitlines()
        self.assertTrue(table[0].startswith("| Variant |") and "Macro Dice" in table[0])
        self.assertIn("| res240 | baseline | 225 | 10.00 | 10.50 | 1.00× | 500 | 0.869 | 0.856 | 0.732 | 0.819 |", table)
        self.assertIn("| res160 | reduced resolution | 100 | 5.00 | 5.20 | 2.00× | 300 | – | – | – | – |", table)

    def test_speedup_only_against_same_conditions(self) -> None:
        from cap6411_atm.benchmark import results_table

        other = dict(self.ROWS[1], precision="fp32")
        table = results_table([self.ROWS[0], other])
        self.assertIn("| res160 | reduced resolution | 100 | 5.00 | 5.20 | – |", table)

    def test_latex_tables_escape_and_render(self) -> None:
        from cap6411_atm.benchmark import matched_table, results_table

        latex = results_table(self.ROWS, fmt="latex", caption="RTX 5090; fp16_autocast")
        self.assertIn(r"\begin{tabular}{lrrrrrrrrrr}", latex)
        self.assertIn(r"reduced resolution", latex)
        self.assertIn(r"fp16\_autocast", latex)
        pair = {"variant": "ada_50", "latency_median_ms": 5.1, "matched_variant": "res160",
                "matched_latency_median_ms": 5.0, "latency_gap_pct": 2.0,
                "macro_dice": 0.80, "matched_macro_dice": 0.76, "macro_dice_delta": 0.04}
        self.assertIn(r"ada\_50 & 5.10 & res160 & 5.00 & +2.0 & 0.800 & 0.760 & +0.040 \\",
                      matched_table([pair], fmt="latex"))
        with self.assertRaises(ValueError):
            results_table(self.ROWS, fmt="html")

    def test_repo_config_is_valid_and_drives_cli_defaults(self) -> None:
        from cap6411_atm.benchmark import load_benchmark_config
        from cap6411_atm.benchmark.compare import DEFAULT_CONFIG, parse_args

        config = load_benchmark_config(DEFAULT_CONFIG)
        self.assertEqual(config["sizes"][0], 240)
        args = parse_args(["--dinov3-repo", "x"])
        self.assertEqual((args.sizes, args.iters, args.stages, args.amp),
                         (config["sizes"], config["iters"], config["stages"], config["amp"]))
        overridden = parse_args(["--dinov3-repo", "x", "--iters", "7", "--no-stages", "--no-amp"])
        self.assertEqual((overridden.iters, overridden.stages, overridden.amp), (7, False, False))

    def test_bad_config_is_rejected(self) -> None:
        from cap6411_atm.benchmark import load_benchmark_config
        from cap6411_atm.benchmark.compare import parse_args

        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.json"
            bad.write_text(json.dumps({"sizes": [240], "iterz": 5}))
            with self.assertRaises(ValueError):
                load_benchmark_config(bad)
            bad.write_text(json.dumps({"sizes": ["240"]}))
            with self.assertRaises(ValueError):
                load_benchmark_config(bad)
            bad.write_text(json.dumps({"warmup": True}))
            with self.assertRaises(ValueError):
                load_benchmark_config(bad)
            with self.assertRaises(SystemExit):
                parse_args(["--dinov3-repo", "x", "--config", str(Path(directory) / "missing.json")])


if __name__ == "__main__":
    unittest.main()
