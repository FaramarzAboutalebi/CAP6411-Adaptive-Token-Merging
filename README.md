# CAP6411 Adaptive Token Merging

Content-adaptive token merging for brain MRI segmentation using DINOv3 and BraTS.

## Project goal

Study whether content-adaptive token merging can reduce vision-transformer computation while preserving brain MRI segmentation quality. The project will use reusable Python modules in `src/`; notebooks are lightweight entry points for setup, exploration, and evaluation.

> Project status: initial scaffold only. No dataset version, split, hardware target, model implementation, experiment, or result has been selected or completed.

## Workstreams

All roles remain open for team selection. Teammates can begin these workstreams in parallel after agreeing on shared interfaces and recording decisions in issues.

| Workstream | Scope | Deliverables | Deadline |
| --- | --- | --- | --- |
| 1. Data pipeline | Define approved BraTS access, preprocessing, and split configuration without committing data. | Reproducible data loader/preprocessing module; documented data configuration and validation checks. | TBD by team |
| 2. DINOv3 integration | Select and integrate an appropriate DINOv3 feature-extraction strategy. | Reusable backbone adapter; documented input/output interface and dependency updates. | TBD by team |
| 3. Adaptive token merging | Design content-aware token selection/merging around the shared feature interface. | Reusable merging module; configurable strategy and unit-level sanity checks. | TBD by team |
| 4. Segmentation and evaluation | Connect the segmentation path and define fair efficiency/quality comparisons. | Training/evaluation entry points; agreed metrics; experiment records and final analysis. | TBD by team |

## Repository layout

```text
configs/       Experiment configuration files
data/          Local-only data locations (contents ignored)
docs/          Contribution and experiment-tracking guides
notebooks/     One lightweight Colab notebook per workstream
outputs/       Generated local experiment artifacts (ignored)
src/           Reusable Python modules
tests/         Automated checks
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open a notebook in Colab or Jupyter from the repository root. Notebooks add the repository root to `sys.path` and import project code from `src/`.

BraTS data and DINOv3 weights must be obtained through their official access channels and kept outside version control. Add implementation-specific dependencies to `requirements.txt` only after the team selects the model approach; do not pin a GPU-specific PyTorch build here.

## Team workflow

Select workstream owners as a team, track decisions and deadlines in GitHub issues/milestones, and integrate changes through reviewed pull requests. Keep commits focused, preserve shared interfaces, and record every run with the template in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for the short workflow.

Do not commit datasets, checkpoints, model weights, credentials, or generated experiment outputs.
