"""Workstream 4: segmentation evaluation, ablations and failure analysis.

Quick reference::

    from cap6411_atm.evaluation import (
        build_eval_loader, evaluate_model, save_evaluation,   # run a model
        VariantSpec, run_ablation, collect_results,            # many variants
        summarize, paired_comparison, global_dice,             # statistics
    )

Plotting helpers live in ``cap6411_atm.evaluation.visualization`` (imported
lazily so metric code does not require a display backend).
"""

from .ablation import VariantSpec, collect_results, merge_ratio_grid, run_ablation
from .evaluator import EvaluationResult, build_eval_loader, evaluate, evaluate_model, save_evaluation
from .metrics import REGION_NAMES, MetricConfig, patch_class_errors, patient_metrics, region_metrics
from .postprocess import (
    DEFAULT_RULES,
    POSTPROCESSING_ABLATION,
    apply_threshold,
    build_postprocessor,
    enforce_nesting,
    threshold_sweep,
)
from .statistics import (
    boundary_profile,
    bootstrap_ci,
    failure_cases,
    format_table,
    global_dice,
    lesion_detection,
    load_results,
    paired_comparison,
    patch_profile,
    stratify_by_tumor_size,
    summarize,
)

__all__ = [
    "DEFAULT_RULES",
    "POSTPROCESSING_ABLATION",
    "REGION_NAMES",
    "EvaluationResult",
    "MetricConfig",
    "VariantSpec",
    "apply_threshold",
    "boundary_profile",
    "bootstrap_ci",
    "build_eval_loader",
    "build_postprocessor",
    "collect_results",
    "enforce_nesting",
    "evaluate",
    "evaluate_model",
    "failure_cases",
    "format_table",
    "global_dice",
    "lesion_detection",
    "load_results",
    "merge_ratio_grid",
    "paired_comparison",
    "patch_class_errors",
    "patch_profile",
    "patient_metrics",
    "region_metrics",
    "run_ablation",
    "save_evaluation",
    "stratify_by_tumor_size",
    "summarize",
    "threshold_sweep",
]
