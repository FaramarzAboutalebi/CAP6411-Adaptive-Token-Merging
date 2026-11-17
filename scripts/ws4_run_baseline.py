"""Workstream 4 baseline run on the validation split: A1 decision rules + A2 thresholds in one pass.

Example (paths may instead come from the CAP6411_* environment variables, see _ws4_paths.py):

    python scripts/ws4_run_baseline.py --archive BraTS.zip --manifest split.json \
        --artifact adapter_decoder.pt --dinov3-repo ../dinov3 \
        --dinov3-checkpoint dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --out results/ws4

--dry-run uses a RANDOM-weight backbone (numbers meaningless) to test the GPU path before the
official DINOv3 checkpoint is available. Outputs contain patient IDs: write them to private storage.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from _ws4_paths import INSPECTION_CONFIG, MODEL_CONFIG, add_data_args, require

from cap6411_atm.evaluation import (
    POSTPROCESSING_ABLATION, VariantSpec, build_eval_loader, build_provenance, global_dice, run_ablation, threshold_sweep,
)
from cap6411_atm.models import (
    BraTSDINOv3Segmenter, DINOv3SegmentationConfig, build_pretrained_brats_dinov3, load_adapter_decoder_artifact,
)

WORKSTREAM1_MACRO_DICE = 0.819176


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    parser.add_argument("--out", type=Path, required=True, help="output directory (private storage)")
    parser.add_argument("--patients", type=int, default=None, help="only the first N validation patients")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="random backbone; tests the pipeline only")
    args = parser.parse_args()
    require(args, "archive", "manifest", "artifact", "dinov3_repo")
    if not args.dry_run:
        require(args, "dinov3_checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DINOv3SegmentationConfig.from_json(MODEL_CONFIG)
    if args.dry_run:
        backbone = torch.hub.load(str(args.dinov3_repo), "dinov3_vitb16", source="local", pretrained=False)
        model = BraTSDINOv3Segmenter(backbone, config)
    else:
        model = build_pretrained_brats_dinov3(args.dinov3_repo, args.dinov3_checkpoint, config)
    meta = load_adapter_decoder_artifact(model, args.artifact)
    model = model.to(device).eval()
    print("artifact validation macro Dice:", meta["validation"]["macro_dice"], "| device:", device)

    validation = sorted(json.loads(args.manifest.read_text())["splits"]["validation"])
    patients = validation[: args.patients] if args.patients else None
    name = "dryrun" if args.dry_run else ("validation187" if patients is None else f"first{len(patients)}")
    run_dir = args.out / name
    loader = build_eval_loader(args.archive, args.manifest, INSPECTION_CONFIG, split="validation",
                               patients=patients, batch_size=args.batch_size, cache_dir=args.cache_dir)
    rules = {**POSTPROCESSING_ABLATION, **threshold_sweep([0.3, 0.4, 0.6, 0.7])}
    variants = [VariantSpec("baseline", method="baseline", merge_ratio=0.0, rules=rules)]

    started = time.perf_counter()
    provenance = build_provenance(
        checkpoint={
            "dinov3": "random init (dry run)" if args.dry_run else args.dinov3_checkpoint,
            "adapter_decoder": args.artifact,
        },
        dataset=args.archive,
        manifest=args.manifest,
        inspection_config=INSPECTION_CONFIG,
        model_config=MODEL_CONFIG.name,
    )
    frame = run_ablation(variants, lambda spec: model, loader, run_dir, provenance=provenance,
                         device=device, progress=True)
    print(f"done in {time.perf_counter() - started:.0f}s -> {run_dir}")
    table = global_dice(frame)
    print(table.round(4).to_string())
    if not args.dry_run and patients is None:
        reproduced = float(table.loc["baseline", "macro"])
        print(f"reproduced global macro Dice {reproduced:.6f} vs Workstream 1 {WORKSTREAM1_MACRO_DICE}")
        if abs(reproduced - WORKSTREAM1_MACRO_DICE) >= 2e-3:
            raise SystemExit("evaluation does not reproduce the Workstream 1 number")


if __name__ == "__main__":
    main()
