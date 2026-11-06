# CAP 6411 — Content-Adaptive Token Merging for Brain MRI Segmentation

**Course:** CAP 6411 — Computer Vision Systems, Fall 2026  
**Topic:** #2 — Content-adaptive token merging for dense & geometric foundation models  
**Team:** Jordan Becci, Anirudh Narasimha Bharadwaj, Faramarz Aboutalebi, and Geno

## Project Goal

We will investigate whether content-adaptive token merging can accelerate brain MRI tumor segmentation while preserving important tumor boundaries.

**Planned pipeline:**

Brain MRI (BraTS) → DINOv3 encoder → Adaptive token merging → Dense feature reconstruction → Segmentation head → Tumor prediction

We will compare four approaches:

1. Original model without merging
2. Lower-resolution input
3. Uniform token merging
4. Content-adaptive token merging

**Main research question:** At similar inference latency, can adaptive token merging preserve segmentation quality better than simply lowering image resolution?

Our main metrics will include Dice score, boundary quality, inference latency, and peak GPU memory.

The exact BraTS release, MRI modalities, model configuration, and evaluation protocol will be agreed upon before experiments.

## Team — Choose Your Workstream

We have four technical workstreams. **Each member chooses one primary role.** No roles are assigned yet.

Each role includes implementation, testing, experiments, documentation, and contributing to the final report and presentation.

### Role 1 — Dataset & Segmentation Baseline

**Goal:** Build a working uncompressed segmentation model.

Tasks:
- Obtain and prepare the selected BraTS dataset.
- Implement loading, preprocessing, normalization, and patient-level data splitting.
- Load a pretrained DINOv3 encoder.
- Implement and train a lightweight segmentation head.
- Save checkpoints and generate segmentation predictions.
- Establish baseline Dice, latency, and memory measurements.

**Main files:**
- `src/data/`
- `src/models/baseline.py`
- `scripts/train_baseline.py`
- `notebooks/01_baseline.ipynb`

**Deliverables:** Reproducible baseline, model checkpoint, prediction outputs, and initial results.

### Role 2 — Adaptive Token Merging

**Goal:** Implement the project's main efficiency method.

Tasks:
- Review ToMe and StructSAM.
- Implement a standard/uniform merging baseline.
- Develop content-adaptive token merging.
- Preserve important spatial information and reconstruct dense features.
- Support configurable merging ratios.
- Verify that merging reduces actual model computation.
- Integrate merging into the baseline encoder.

**Main files:**
- `src/merging/`
- `src/models/adaptive_model.py`
- `notebooks/02_token_merging.ipynb`

**Deliverables:** Working uniform/adaptive merging implementations, integration tests, and configurable experiments.

### Role 3 — Efficiency & Baseline Comparisons

**Goal:** Measure whether merging actually improves efficiency.

Tasks:
- Implement the reduced-input-resolution baseline.
- Build standardized GPU benchmarking tools.
- Measure inference latency and peak GPU memory.
- Compare original, reduced-resolution, uniform-merging, and adaptive-merging models.
- Find approximately latency-matched configurations.
- Generate efficiency-versus-quality plots and tables.

**Main files:**
- `src/benchmark/`
- `scripts/benchmark.py`
- `notebooks/03_efficiency.ipynb`

**Deliverables:** Reproducible benchmarks, comparison tables, and efficiency plots.

### Role 4 — Segmentation Evaluation & Ablations

**Goal:** Analyze segmentation quality and identify where merging helps or fails.

Tasks:
- Implement Dice and boundary-focused evaluation metrics.
- Evaluate tumor regions and small lesions using clearly defined criteria.
- Build visualization tools for MRI images, masks, predictions, and errors.
- Test multiple merging ratios and method configurations.
- Compare boundary preservation across model variants.
- Analyze failure cases and summarize experimental findings.

**Main files:**
- `src/evaluation/`
- `scripts/evaluate.py`
- `notebooks/04_evaluation.ipynb`

**Deliverables:** Evaluation pipeline, ablation tables, qualitative visualizations, and failure analysis.

## How We Work in Parallel

Nobody needs to wait for the entire baseline to be completed.

| Role | Start immediately with | Integrate later with |
|---|---|---|
| 1 | BraTS preparation and baseline setup | Shared model interface |
| 2 | Synthetic token tensors and a standalone ViT | Role 1's encoder |
| 3 | Dummy model and benchmarking tools | All model variants |
| 4 | Synthetic predictions and sample masks | Real model predictions |

### Shared Interfaces

We will agree on tensor formats before implementation.

Proposed interfaces:

```python
# images: [B, C, H, W]
# logits: [B, K, H, W]
logits = model(images)

# labels: [B, H, W]
results = evaluate(logits, labels)
```

The final number of input channels and segmentation outputs will depend on our selected BraTS configuration.

All methods must use compatible preprocessing, splits, prediction formats, and evaluation settings.

**Integration checkpoints:**
- Role 1 shares a minimal model interface early.
- Role 2 tests encoder compatibility before implementing advanced merging.
- Roles 3 and 4 validate their tools independently with mock data.
- Everyone integrates through reviewed pull requests.

## Development Environment

**GitHub:** Shared source code and collaboration  
**Google Colab:** GPU training and experiments  
**Google Drive:** Private dataset and checkpoint storage  
**Overleaf:** Final report  
**UCF Newton:** Optional backup for longer GPU experiments

### Colab Workflow

1. Open your assigned notebook in Google Colab.
2. Enable a GPU runtime when needed.
3. Clone the GitHub repository into the Colab session.
4. Install dependencies from `requirements.txt`.
5. Mount Google Drive if your task requires datasets or checkpoints.
6. Import reusable modules from `src/`.
7. Run your experiments.
8. Save checkpoints and large outputs to Drive.
9. Commit source code, notebooks, configurations, and small result summaries to GitHub.

Example setup:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone https://github.com/OWNER/CAP6411-Adaptive-Token-Merging.git
%cd CAP6411-Adaptive-Token-Merging
!pip install -r requirements.txt
```

Replace `OWNER` with the repository owner. Private repositories require authorized GitHub access.

**Important:**
- Colab runtime storage is temporary.
- Do not commit datasets or checkpoints to GitHub.
- Do not put GitHub tokens or other secrets inside notebooks.
- Do not use different GPU types for direct latency comparisons.
- Record the exact GPU, software versions, and experiment settings.
- Keep substantial implementations in `src/`, not scattered across notebooks.

## Repository Structure

```text
CAP6411-Adaptive-Token-Merging/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_token_merging.ipynb
│   ├── 03_efficiency.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── merging/
│   ├── benchmark/
│   └── evaluation/
│
├── scripts/
│   ├── train_baseline.py
│   ├── benchmark.py
│   └── evaluate.py
│
├── configs/
├── tests/
├── results/
│
└── docs/
    ├── CONTRIBUTING.md
    └── EXPERIMENTS.md
```

Only source code, configurations, small result tables/plots, and documentation belong in GitHub.

Datasets and large model checkpoints remain in private storage.

## Timeline

| Dates | Target |
|---|---|
| Nov 6–10 | Set up repository, agree on interfaces, begin parallel implementation |
| Nov 11–15 | Working baseline and first merging integration |
| Nov 16–21 | Run comparisons, benchmarks, and ablations |
| Nov 22–26 | Analyze results and draft report |
| Nov 27–29 | Final testing, report review, and submission preparation |
| **Nov 30, 11:59 PM** | **Final report and code due** |

Presentation preparation must happen earlier if our assigned presentation date is before the report deadline.

## Final Report & Presentation

**Final report:** 6–8 pages, excluding references, using the CVPR LaTeX template.

| Workstream | Primary report content |
|---|---|
| Role 1 | Dataset, baseline architecture, training setup |
| Role 2 | Related work, merging algorithm, method |
| Role 3 | Efficiency methodology, benchmarks, tradeoff analysis |
| Role 4 | Segmentation metrics, ablations, boundary/failure analysis |

Everyone also reviews the complete report and contributes to the introduction, conclusions, figures, references, and presentation.

All members must speak during the presentation.

The report must include:
- Abstract, introduction, related work, method, experiments, discussion/limitations, conclusion
- Individual contribution statement
- Generative AI disclosure
- Reproducibility instructions and code repository link

## GitHub Collaboration Rules

1. Each member works on a separate branch.
2. Do not push directly to `main`.
3. Open a pull request when your code is ready.
4. Include a short description and how the changes were tested.
5. Request a teammate's review before merging.
6. Keep notebooks lightweight and reusable code in `src/`.
7. Do not commit private datasets, large checkpoints, credentials, or generated caches.
8. Record experiment configurations and results.
9. Communicate interface changes before merging.
10. Each member is responsible for understanding and explaining their own work.

Suggested branch names:
- `feature/baseline`
- `feature/token-merging`
- `feature/efficiency`
- `feature/evaluation`

## Success Criteria

We aim to deliver:

- [ ] Reproducible DINOv3 + BraTS segmentation baseline
- [ ] Working uniform token merging
- [ ] Working adaptive token merging
- [ ] Reduced-resolution comparison
- [ ] Dice and boundary evaluation
- [ ] Real latency and peak-memory measurements
- [ ] Matched-latency quality comparison
- [ ] Ablation experiments and failure analysis
- [ ] Final CVPR-format report
- [ ] Reproducible GitHub repository
- [ ] Presentation with contributions from all members

Cross-slice token reuse and advanced image-guided unmerging are optional extensions, not prerequisites for completing the core project.

## References

- Bolya et al., *Token Merging: Your ViT But Faster*, ICLR 2023. https://arxiv.org/abs/2210.09461
- Siméoni et al., *DINOv3*, 2025. https://arxiv.org/abs/2508.10104
- *StructSAM: Structure- and Spectrum-Preserving Token Merging for Segment Anything Models*, 2026. https://arxiv.org/abs/2603.07307
- Menze et al., *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)*, IEEE TMI, 2015.

## Team Coordination

**Next steps:**
1. Everyone chooses a workstream.
2. Confirm team membership and presentation arrangements with the instructor where needed.
3. Agree on BraTS version, model configuration, and shared interfaces.
4. Set up GitHub access and start implementation.
5. Post progress and blockers in Discord.

Our priority is a working, measurable, reproducible project—not implementing every possible extension.
