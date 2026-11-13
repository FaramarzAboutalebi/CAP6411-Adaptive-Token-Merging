# DINOv3 BraTS baseline: audited implementation specification

## Audit status

This specification audits the proposed baseline against the official Facebook Research DINOv3 repository and this CAP6411 checkout at commit `845a4d0`. It is not an implementation or training result. No weights were downloaded and no GPU run was performed.

Finding labels used throughout:

- **Verified fact** — directly established by official DINOv3 source, release documentation/license, or the current CAP6411 source.
- **Engineering recommendation** — the prescribed first-baseline design; it still requires implementation testing.
- **Unresolved issue** — cannot be established without approved weights, target-hardware measurement, an experiment, or future token-merging code.

## Final decision

| Item | Audited specification | Classification |
|---|---|---|
| Backbone | `dinov3_vitb16`, frozen parameters, evaluation mode | Engineering recommendation based on verified model definition |
| Pretraining | LVD-1689M web-image checkpoint | Engineering recommendation |
| Exact checkpoint | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | Verified fact from official hub naming/hash logic |
| Input | BraTS axial image `[B,4,240,240]`, channels T1n, T1c, T2w, T2f | Verified fact in this repository |
| Input adaptation | Trainable biased `Conv2d(4,3,kernel_size=1)`, deterministic balanced initialization | Engineering recommendation |
| Feature taps | Zero-based transformer blocks `[2,5,8,11]`, normalized patch tokens only | Engineering recommendation; API and shapes verified |
| Decoder | Four lateral projections, same-grid multi-depth fusion, four 2× bilinear/refinement stages | Engineering recommendation |
| Output | Raw logits `[B,3,240,240]` in WT, TC, ET order | Verified repository contract / engineering architecture |
| Initial batch | Batch size 2 with AMP; test 4 after measured smoke test | Engineering recommendation |
| Token merging | Not part of the first baseline; future contract specified below | Verified repository state / engineering recommendation |

## 1. Official DINOv3 ViT-B/16 specification

The authoritative sources are Meta's [hub backbone definitions](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py), [vision transformer implementation](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py), [patch embedding implementation](https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/patch_embed.py), [release README](https://github.com/facebookresearch/dinov3), and [license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).

### Backbone definition

| Property | Official value | Classification |
|---|---:|---|
| Hub entry point | `dinov3_vitb16` | Verified fact |
| Nominal constructor image size | 224 | Verified fact |
| Input channels | 3 | Verified fact |
| Patch size | 16×16 | Verified fact |
| Embedding dimension `D` | 768 | Verified fact |
| Transformer blocks | 12 | Verified fact |
| Attention heads | 12 | Verified fact |
| FFN ratio | 4 | Verified fact |
| Class tokens | 1 | Verified fact |
| Storage/register tokens | 4 (`n_storage_tokens=4`) | Verified fact |
| Released parameter count | 86M | Verified fact |

The official implementation calls the four non-class special tokens **storage tokens**. They serve the role often described as register tokens, but code should use the official `n_storage_tokens` and `x_storage_tokens` terminology.

### Token construction and class-token behavior

**Verified fact.** `prepare_tokens_with_masks` applies the patch projection, flattens its `[H_p,W_p]` grid, and concatenates tokens in this exact order:

```text
[class token, four storage tokens, row-major patch tokens]
```

At 240×240:

```text
H_p = floor((240 - 16) / 16) + 1 = 15
W_p = 15
N_patch = 225
N_special = 1 + 4 = 5
N_total = 230
```

The class token participates in all transformer blocks. The ordinary model `forward` ultimately applies the head to the normalized class token. Dense segmentation must not use that default classification-style return; it must request patch features. The class and storage tokens are excluded from the default patch output of `get_intermediate_layers`.

### Input resolution requirements

**Verified fact.** The patch embedding is a `Conv2d` with kernel and stride 16. The nominal `img_size=224` is used for constructor metadata and FLOP bookkeeping, but the forward path derives the actual patch grid from the input. The official divisibility assertions are commented out.

**Engineering recommendation.** Use exactly 240×240, which is divisible by 16 and therefore loses no border pixels. Do not infer that every arbitrary resolution is safe: nonmultiples of 16 are floor-cropped by the strided convolution, and extremely different resolutions may change pretrained behavior. Assert `H % 16 == W % 16 == 0` in the future wrapper even though upstream does not.

### Intermediate feature API

**Verified fact.** The official API is:

```python
backbone.get_intermediate_layers(
    x,
    n=[2, 5, 8, 11],
    reshape=True,
    return_class_token=False,
    return_extra_tokens=False,
    norm=True,
)
```

- Block indices are zero-based.
- With `reshape=False`, each returned patch tensor is `[B,225,768]`.
- With `reshape=True`, each is `[B,768,15,15]`.
- `norm=True` applies the backbone's final normalization to every selected intermediate output.
- `return_class_token=True` pairs each patch result with `[B,768]`.
- `return_extra_tokens=True` pairs it with the four storage tokens `[B,4,768]`.
- `forward_features` also exposes `x_norm_clstoken`, `x_storage_tokens`, `x_norm_patchtokens`, `x_prenorm`, and masks for the final output.

### Checkpoint availability and access

**Verified fact.** Meta lists a distilled ViT-B/16 LVD-1689M checkpoint. The official hub definition supplies hash prefix `73cec8be`, yielding:

```text
dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

The expected local loading path is:

```python
backbone = torch.hub.load(
    PINNED_DINOV3_REPOSITORY,
    "dinov3_vitb16",
    source="local",
    weights=APPROVED_LOCAL_CHECKPOINT,
    check_hash=True,
)
```

**Verified fact.** Model weights are gated. Meta's README directs users to the [official access request](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/); accepted users receive checkpoint URLs. The repository also identifies the gated Hugging Face model `facebook/dinov3-vitb16-pretrain-lvd1689m`.

**Engineering recommendation.** Pin an exact official DINOv3 Git revision, use an approved local checkpoint in private Drive storage, verify its hash, and never commit the checkpoint, download URL, or credentials.

### License audit

**Verified fact.** Code and weights are governed by the DINOv3 License, not a standard permissive OSS license. The official license, last updated August 19, 2025:

- binds use or distribution of DINO materials to the agreement;
- grants a limited, non-exclusive, worldwide, non-transferable, royalty-free license to use, modify, reproduce, and create derivative works;
- requires redistribution and derivatives to remain subject to the agreement and include it;
- requires research publications using the materials to acknowledge DINOv3;
- requires compliance with laws, privacy/data-protection rules, sanctions, and trade controls;
- prohibits specified trade-controlled and military/warfare, nuclear, espionage, gun, and illegal-weapons end uses;
- includes a restriction against use involving or encouraging reverse engineering/decompilation/discovery of underlying components;
- provides no warranty or support; and
- requires deletion and cessation of use after termination.

**Engineering recommendation.** The team must read and accept the current official terms before access. This summary is not legal advice.

## 2. Audited end-to-end tensor contract

The proposal is dimensionally consistent if implemented as follows:

| Stage | Operation | Output | Classification |
|---|---|---|---|
| Dataset | Four normalized modalities | `[B,4,240,240]` FP32 before AMP | Verified fact |
| Adapter | `Conv2d(4,3,1,1,0,bias=True)` | `[B,3,240,240]` | Verified dimensions / recommended layer |
| Patch embedding | 16×16 kernel, stride 16, `3→768` | `[B,15,15,768]`, then 225 patches | Verified fact |
| Special-token prepend | CLS + four storage + patches | `[B,230,768]` | Verified fact |
| Blocks 2,5,8,11 | normalized patch tokens | four × `[B,225,768]` | Verified fact |
| Reshape | row-major tokens to maps | four × `[B,768,15,15]` | Verified fact |
| Lateral projection | four independent `1×1`, `768→128` | four × `[B,128,15,15]` | Engineering recommendation |
| Concatenation | channel axis | `[B,512,15,15]` | Derived fact |
| Fusion | `3×3`, `512→128`, GroupNorm, GELU | `[B,128,15,15]` | Engineering recommendation |
| Upsample 1 | bilinear 2× + `3×3 128→128` | `[B,128,30,30]` | Engineering recommendation |
| Upsample 2 | bilinear 2× + `3×3 128→96` | `[B,96,60,60]` | Engineering recommendation |
| Upsample 3 | bilinear 2× + `3×3 96→64` | `[B,64,120,120]` | Engineering recommendation |
| Upsample 4 | bilinear 2× + `3×3 64→32` | `[B,32,240,240]` | Engineering recommendation |
| Prediction | `1×1`, `32→3` | logits `[B,3,240,240]` | Engineering recommendation |

Use `align_corners=False` for every bilinear resize. Each refinement convolution uses padding 1 and therefore preserves its listed spatial size. GroupNorm is recommended instead of BatchNorm because the T4 batch will be small. The four DINOv3 features all have the same 15×15 resolution; this is **multi-depth same-grid fusion**, not a true multiscale feature pyramid.

The output channels are independent WT, TC, and ET logits. Do not apply softmax. A sigmoid-compatible BCE-plus-soft-Dice objective is recommended; apply sigmoid only inside the loss as required or for inference/metrics.

## 3. Four-modality adapter audit

### Modality ordering

**Verified fact.** `BraTSAxialSliceDataset` emits channels in `MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")`, documented clinically as T1n, T1c, T2w, T2f. The model wrapper must assert or document this exact order. Reordering modalities after training would invalidate adapter semantics.

### Technical soundness

**Verified fact.** A 1×1 `4→3` convolution is shape-compatible with the unmodified pretrained patch embedding and keeps upstream checkpoint loading strict. It mixes the four co-registered modalities independently at every pixel.

**Unresolved issue.** No linear map from four channels to three can preserve all four degrees of freedom. The adapter is a learned rank-at-most-three bottleneck. It also cannot preserve pretrained “RGB semantics” because MRI modalities are not RGB and use per-modality nonzero z-score normalization rather than the official natural-image preprocessing.

**Engineering recommendation.** Still use the adapter for the first baseline because it preserves every pretrained DINOv3 parameter and isolates modality adaptation in a tiny, inspectable layer. Treat the domain mismatch as an experiment risk, not as solved by initialization.

### Initialization

Use zero bias and a deterministic balanced orthonormal projection:

```text
W_init = 0.5 × [[ 1,  1,  1,  1],
                [ 1, -1,  1, -1],
                [ 1,  1, -1, -1]]
```

Assign this `[3,4]` matrix to the center coefficients of the `1×1` weights. Each output initially uses every modality; each row has unit L2 norm, which limits scale inflation when normalized modalities have comparable variance. This initialization does **not** preserve all information or reproduce RGB statistics. It is an engineering recommendation that must be recorded and compared with standard Kaiming initialization if results are unstable.

Monitor adapter weights, per-output mean/std, and the first and last DINO feature norms. Do not blindly apply ImageNet mean/std after the adapter: those constants assume bounded natural-image RGB pixels, not signed MRI z-scores.

### Why not modify patch embedding first

Expanding the pretrained patch projection from `[768,3,16,16]` to `[768,4,16,16]` is technically possible. A common engineering initialization would copy the three pretrained planes, initialize the fourth from their mean, and rescale to control activation magnitude. However:

- modality-to-R/G/B assignment is arbitrary;
- strict checkpoint loading requires explicit surgery;
- the patch embedding must become trainable, so the encoder is no longer wholly frozen;
- the comparison combines channel adaptation with a changed pretrained layer.

**Engineering recommendation.** Keep patch-embedding modification as an ablation after the adapter baseline, not the first implementation.

## 4. Gradient flow and T4 feasibility

### Critical gradient distinction

**Verified fact.** Setting DINOv3 parameters to `requires_grad=False` eliminates their parameter gradients and optimizer state, but it does not eliminate the need to backpropagate through DINOv3 when the trainable adapter precedes it.

To train the adapter from the segmentation loss:

```text
loss → decoder → DINO features → all frozen DINO operations → adapter
```

Autograd must retain the encoder intermediates needed for that backward path. Wrapping feature extraction in `torch.no_grad()` or `torch.inference_mode()` would detach the adapter and prevent it from learning. This corrects any interpretation that “frozen encoder” alone guarantees inference-only activation memory.

If the adapter is frozen, then the backbone can run under `inference_mode`, and only decoder activations require gradients. That is a valid lower-memory control but is not the recommended trainable-adapter baseline.

### Memory estimates

The following are analytical estimates, not measured T4 peaks:

- **Backbone parameters:** 86M parameters are approximately 344 MB in FP32 or 172 MB in FP16, excluding allocator and module overhead.
- **Saved tapped maps:** one `[768,15,15]` FP16 feature is about 0.33 MiB per sample; four are about 1.32 MiB per sample (2.64 MiB in FP32).
- **Token sequence:** only 230 tokens per sample, so attention matrices are small compared with high-resolution ViT workloads.
- **Decoder:** approximately 1.3M convolution weights under the specified channels. AdamW states and gradients for the adapter/decoder are modest relative to the backbone.
- **Dominant uncertainty:** autograd-saved tensors inside all 12 frozen blocks, CUDA kernel workspaces, autocast behavior of upstream normalization/RoPE, and PyTorch allocator caching.

| Batch | Expected feasibility with AMP | Audit recommendation |
|---:|---|---|
| 1 | Very likely | Mandatory smoke test and debugging configuration |
| 2 | Likely with trainable adapter | **Initial training batch** |
| 4 | Plausible at 240×240, not verified | Attempt only after measured batch-2 run |

Batch 8 is intentionally not specified for the first implementation audit. It may fit, but there is no measured evidence and the trainable-adapter gradient path makes inference-only estimates misleading.

### T4 execution strategy

**Engineering recommendation.**

1. Keep backbone parameters FP32 initially but run the forward under CUDA AMP FP16; T4 supports FP16 Tensor Cores but not native BF16 training in the same way as newer GPUs.
2. Set all backbone parameters `requires_grad_(False)` and call `backbone.eval()` to disable training-time behavior, but do not use `no_grad` while the adapter is trainable.
3. Optimize only adapter and decoder parameters with AdamW.
4. Start at batch 2. Measure batch 1, 2, and 4 using `torch.cuda.reset_peak_memory_stats()`, synchronized warm-up, `max_memory_allocated`, and `max_memory_reserved`.
5. Use gradient accumulation if an effective batch larger than 2 is needed.
6. If batch 2 is unexpectedly tight, use activation checkpointing across transformer blocks before falling back to ViT-S/16. Checkpointing trades compute for memory and must preserve frozen-block gradient flow to the adapter.
7. Do not cache GPU feature maps across batches. The existing dataset's patient cache is CPU/disk-side and should remain bounded.

**Unresolved issue.** Actual peak memory, throughput, and the largest safe batch on a Colab T4 require implementation and measurement with the pinned upstream revision.

## 5. Decoder decision

**Engineering recommendation.** Use the exact decoder in the tensor table: independent lateral `1×1` projections, concatenate, one fusion block, then four bilinear/refinement stages. Each refinement is:

```text
bilinear resize(scale_factor=2, align_corners=False)
→ Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
→ GroupNorm(valid_group_count, out_channels)
→ GELU
```

Choose a deterministic group count dividing the channel count, for example 8 groups for 128/96/64/32. This decoder:

- consumes the verified four `[B,768,15,15]` maps;
- has low parameter and optimizer-state cost;
- reaches exactly 240 through `15×2×2×2×2`;
- produces exactly three logits; and
- exposes a clean dense-token reconstruction boundary for later merging.

**Unresolved issue.** Patch-16 tokenization may be too coarse for small enhancing tumors. The first baseline should quantify ET and boundary performance before adding a high-resolution input skip, because such a skip could obscure whether DINOv3 or the skip path drives segmentation.

## 6. Adaptive token-merging audit

### Current repository state

**Verified fact.** No merger, reconstructor, DINO wrapper, or token-merging test currently exists. `src/cap6411_atm/merging/__init__.py` contains only a module docstring. `src/cap6411_atm/models/__init__.py` is likewise only a placeholder. The token-merging notebook says implementation is a future step. Therefore this audit does not claim an existing callable interface.

### Required future contract

**Engineering recommendation.** Workstream 2 and the model owner should agree on an explicit contract equivalent to:

```python
merged, state = merger(
    patch_tokens,       # [B,N,D], baseline [B,225,768]
    grid_size=(15, 15),
)

dense = reconstructor(
    merged,             # [B,M,D], M <= 225
    state,
)                      # [B,225,768]
```

The opaque `state` must record at least the original grid, per-sample original-to-merged assignment, aggregation weights/token sizes, and spatial coordinates used after merging. Required tests:

- merge ratio zero is numerically equivalent to the unmerged path;
- reconstruction returns 225 tokens in original row-major order;
- batch elements remain independent;
- dtype/device/embedding dimension are preserved;
- gradients behave as intended for trainable merger components; and
- class/storage tokens never enter spatial clustering.

### Special tokens and positions

**Verified fact.** The full sequence begins with one class and four storage tokens. These are non-spatial protected tokens. Only `tokens[:,5:,:]` are spatial in the baseline configuration, but future code should derive the offset as `1 + backbone.n_storage_tokens`, not hard-code five.

**Verified fact.** DINOv3 computes axial rotary positional embeddings from the patch-grid `(H,W)` and passes them into every transformer block. Standard block execution assumes a positional representation compatible with its patch sequence.

**Unresolved issue.** The repository has no design for assigning rotary coordinates to an arbitrary merged region. Mean coordinates, representative coordinates, and structured local merging are different algorithms and may change model behavior. This must be designed and tested before inter-block merging.

### Post-encoder merging versus acceleration

- **Post-encoder merge/reconstruct — verified implication:** all 12 backbone blocks already processed 230 tokens. This can test reconstruction and possibly reduce decoder work, but cannot reduce DINOv3 encoder compute. It must not be reported as encoder acceleration.
- **Inter-block merging — engineering requirement for acceleration:** merge after block `k`, run later blocks on fewer spatial tokens, then reconstruct before the dense decoder. This requires a block-level wrapper and merging-aware rotary-position handling; the unmodified public `get_intermediate_layers` path is insufficient.

If multi-depth features `[2,5,8,11]` remain part of the merged model, define what happens at each tap. Features before a merge are dense; features after a merge must be reconstructed using the correct state before same-grid fusion. Alternatively, compare a last-layer-only decoder across merging methods to avoid inconsistent tap semantics. This is an unresolved experimental-interface choice.

## 7. Risks and unresolved questions

| Finding | Classification | Required resolution |
|---|---|---|
| Natural-image frozen features may transfer poorly to MRI | Unresolved issue | Train/evaluate against a simple segmentation control |
| 4→3 adapter discards one linear degree of freedom | Verified fact | Measure adapter behavior; compare patch-embedding ablation later |
| Balanced adapter initialization is not an RGB-preserving map | Verified fact | Record initialization and inspect activations |
| Training adapter retains encoder backward graph | Verified fact | Measure T4 memory; use AMP/checkpointing if needed |
| `[2,5,8,11]` is plausible but not BraTS-validated | Unresolved issue | Compare last-only and four-tap variants |
| Patch-16 may miss small ET/boundaries | Unresolved issue | Track ET Dice, lesion size, and boundary metrics |
| Exact DINOv3 upstream revision is not pinned in current dependencies | Verified repository fact | Pin revision during implementation review |
| Approved checkpoint access has not been demonstrated in this audit | Unresolved issue | Obtain team access before implementation smoke test |
| T4 batch 4 is unmeasured | Unresolved issue | Run measured batches 1/2/4 before training |
| Token-merging implementation/interface does not exist | Verified repository fact | Coordinate contract with Workstream 2 |
| Rotary coordinates for merged patches are undefined | Unresolved issue | Design merging-aware positional strategy |

## 8. Step-by-step implementation checklist

This is the handoff checklist for a later implementation task:

1. [ ] Confirm team acceptance of the current DINOv3 License and approved weight access.
2. [ ] Pin an exact official DINOv3 Git revision; record it in dependencies and experiment metadata.
3. [ ] Place the approved `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` outside Git and verify the hash.
4. [ ] Load `dinov3_vitb16` strictly and assert patch size 16, embedding 768, 12 blocks, and four storage tokens.
5. [ ] Implement the `4→3` 1×1 adapter with zero bias and the documented balanced initialization.
6. [ ] Assert input modality order T1n, T1c, T2w, T2f and input shape `[B,4,240,240]`.
7. [ ] Freeze backbone parameters and set it to evaluation mode; do **not** detach the encoder while the adapter is trainable.
8. [ ] Extract normalized patch maps from zero-based blocks `[2,5,8,11]`; assert four `[B,768,15,15]` tensors.
9. [ ] Implement the specified lateral projections, fusion block, and four upsampling/refinement stages.
10. [ ] Assert raw output logits `[B,3,240,240]` in WT, TC, ET order; use no output softmax.
11. [ ] Optimize only adapter and decoder; use AMP FP16 and begin at physical batch size 2.
12. [ ] Run forward/backward smoke tests at batches 1, 2, and 4 and record allocated/reserved peak VRAM and latency.
13. [ ] Add tests for shape contracts, deterministic adapter initialization, frozen backbone parameters, adapter gradients, decoder gradients, and finite outputs.
14. [ ] Verify that `no_grad` is used only for a fixed-adapter control, never for trainable-adapter training.
15. [ ] Train/evaluate the unmerged baseline before introducing token merging.
16. [ ] Agree on the merger/reconstructor state contract and protected-token rules with Workstream 2.
17. [ ] Prove no-merge equivalence and post-encoder reconstruction correctness.
18. [ ] Design coordinate-aware inter-block merging before making any encoder-acceleration claim.

## Official sources consulted

- [DINOv3 official repository and pretrained model access](https://github.com/facebookresearch/dinov3)
- [Official ViT-B/16 hub definition](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py)
- [Official transformer token and intermediate-feature implementation](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py)
- [Official patch embedding implementation](https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/patch_embed.py)
- [Official DINOv3 model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)
- [Official DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- [Official Meta weight-access form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
