# DINOv3 architecture design for BraTS segmentation

## Status and decision

This document is an architecture investigation, not an implementation or experiment result. No pretrained weights were downloaded and no GPU memory measurement or training run was performed.

**Recommended first baseline:** a frozen **DINOv3 ViT-B/16 (LVD-1689M)** encoder, a trainable `1×1` convolutional **4-to-3 modality adapter**, four normalized intermediate feature maps from transformer blocks `[2, 5, 8, 11]`, and a small convolutional fusion/upsampling decoder producing three independent WT/TC/ET logits at `240×240`.

ViT-S/16 should remain a low-cost fallback and efficiency ablation. ViT-L/16 and larger variants are not appropriate for the first Colab T4 baseline. The recommendation favors ViT-B/16 because it is still practical with a frozen encoder on a 16 GB T4 while providing substantially stronger published dense features than ViT-S/16. This T4 judgment is an engineering estimate and must be confirmed by a measured smoke test.

## Evidence labels

- **Verified** means stated in Meta's official DINOv3 repository, model card, source code, paper, or license.
- **Repository observation** means derived from this CAP6411 checkout at commit `845a4d0`.
- **Derived** means calculated directly from verified dimensions.
- **Recommendation/assumption** means a design choice requiring later validation.

## Available DINOv3 backbones

Meta publishes LVD-1689M web-image weights for six ViT and four ConvNeXt backbones. ViT-L/16 and ViT-7B/16 also have SAT-493M satellite weights. The web-image weights are the relevant starting point for MRI because the satellite domain offers no clear advantage.

| Backbone | Parameters | Patch | Embedding | Depth | Heads | Dense output at 240×240 | Baseline suitability |
|---|---:|---:|---:|---:|---:|---|---|
| ViT-S/16 | 21M | 16 | 384 | 12 | 6 | `[B,384,15,15]` | Best low-cost fallback |
| ViT-S+/16 | 29M | 16 | 384 | 12 | 6 | `[B,384,15,15]` | More expensive FFN; limited benefit for first baseline |
| **ViT-B/16** | **86M** | **16** | **768** | **12** | **12** | **`[B,768,15,15]`** | **Recommended** |
| ViT-L/16 | 300M | 16 | 1024 | 24 | 16 | `[B,1024,15,15]` | Avoid initially on T4 |
| ViT-H+/16 | 840M | 16 | 1280 | 32 | 20 | `[B,1280,15,15]` | Not practical for this baseline |
| ViT-7B/16 | 6,716M | 16 | 4096 | 40 | 32 | `[B,4096,15,15]` | Incompatible with T4 capacity |

Parameter counts and released variants are verified in the [official DINOv3 README](https://github.com/facebookresearch/dinov3#pretrained-models). ViT dimensions, depths, heads, three-channel inputs, patch size 16, and four storage tokens are verified in the [official backbone definitions](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py). Meta's [model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md) reports progressively stronger dense-task results from ViT-S/16 through ViT-L/16, although those natural-image benchmarks do not establish BraTS performance.

ConvNeXt-T/S/B/L weights also exist, but they do not expose the transformer token sequence required by this project's adaptive token-merging research question. They are therefore out of scope for the main baseline, though ConvNeXt-T could be a separate non-token efficiency control.

### Input resolution and feature outputs

The hub constructors use a nominal `img_size=224`, but the official ViT forward path derives `H` and `W` from the actual patch embedding and constructs axial rotary positions for that grid. `get_intermediate_layers(..., reshape=True)` reshapes patch tokens using the actual input dimensions. Thus a 240×240 input, which is exactly divisible by 16, produces a 15×15 grid without resizing. This is verified from the [official transformer implementation](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py), but must still receive a local shape-contract test when the dependency and weights are introduced.

For ViT-B/16:

```text
input image                 [B, 3, 240, 240]
patch grid                  15 × 15
spatial patch tokens        [B, 225, 768]
class token                 [B, 1, 768]
storage tokens              [B, 4, 768]
full internal sequence      [B, 230, 768]
reshaped spatial feature    [B, 768, 15, 15]
```

The sequence is class token, four storage tokens, then patch tokens. `get_intermediate_layers` removes the class and storage tokens before returning its default patch output. `reshape=True` converts row-major patch tokens to `[B,D,H/16,W/16]`. `forward_features` alternatively returns named fields including `x_norm_clstoken`, `x_storage_tokens`, and `x_norm_patchtokens`.

## Weight access, loading, and license

### Verified access path

Meta requires users to submit the [DINOv3 access request](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). After acceptance, Meta sends checkpoint URLs. The official repository supports loading a local checkpoint path or an authorized URL through `torch.hub.load`:

```python
backbone = torch.hub.load(
    DINOV3_REPO_DIR,
    "dinov3_vitb16",
    source="local",
    weights=DINOV3_CHECKPOINT_PATH,
)
```

The repository also lists gated Hugging Face model IDs such as `facebook/dinov3-vitb16-pretrain-lvd1689m`; official Transformers support begins at version 4.56.0. The project should choose one loading path, pin its version/commit, verify the checkpoint hash, and store credentials and weights outside Git. A local clone plus an approved local checkpoint is the most auditable Colab route.

### License obligations

DINOv3 is not distributed under a generic permissive software license. Code and weights are covered by Meta's [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), last updated August 19, 2025. It grants limited use, modification, derivative-work, and redistribution rights subject to the agreement. Redistribution must retain the agreement; publications using the materials must acknowledge DINOv3; use must comply with law and trade controls; and termination requires deletion and cessation of use. The team must accept and review the current license before downloading or using weights. This summary is not legal advice.

## Four-modality input adaptation

The pretrained patch embedding expects three input channels. The current dataset contract is `[B,4,240,240]` in T1n, T1c, T2w, T2f order, with each modality independently nonzero-z-score normalized.

| Strategy | Pretrained patch embedding preserved? | Trainable cost | Advantages | Risks | Decision |
|---|---|---:|---|---|---|
| Learned 4→3 convolution | Yes | 15 parameters for biased `1×1` conv | Clean checkpoint loading; modality mixing is learned per pixel; encoder remains exactly frozen; easy ablation | Three-channel bottleneck; frozen natural-image encoder sees a learned distribution | **Use first** |
| Replace patch embedding with 4-channel projection | No, one channel must be initialized | Adds one channel plane (`D×16×16`) | No 3-channel bottleneck; direct four-modality patches | Patch embedding can no longer be fully frozen; strict checkpoint loading needs surgery; initialization choice confounds baseline | Later ablation |
| Independent/shared encoder per modality then fuse | Yes per stream | Roughly 4× encoder compute/activations | Retains modality-specific representations | Too slow/heavy for T4; fusion and token-merging become much more complex | Reject for first baseline |
| Fixed pseudo-RGB channel selection/averaging | Yes | 0 | Simplest control | Discards or arbitrarily mixes clinically distinct modalities | Diagnostic ablation only |
| Modality tokens/cross-attention fusion | Usually | Nontrivial | Explicit modality reasoning | Adds an unvalidated architecture and ambiguous token-merging semantics | Future research, not baseline |

### Recommended adapter

Use `Conv2d(4, 3, kernel_size=1, bias=True)` directly before the frozen encoder. Train the adapter and decoder; keep all DINOv3 parameters in evaluation mode with `requires_grad=False`. Initialize the adapter deterministically near an information-preserving projection rather than random high-variance output, then let it learn. The exact initialization should be recorded and ablated.

Do **not** apply ImageNet mean/std blindly to the already z-scored MRI channels. The adapter's output distribution is not RGB intensity. Start with the existing modality normalization, a learned adapter, and optionally a small channel-wise normalization immediately after the adapter only if measured activation statistics show instability. This is a recommendation, not a verified DINOv3 MRI recipe.

The modified patch-embedding alternative can initialize the fourth-channel weights from the mean of the three pretrained channel planes and train only that embedding plus the decoder. It should be evaluated later because it changes the meaning of “frozen pretrained encoder” and complicates a clean baseline comparison.

## Proposed baseline architecture

### Token-flow diagram

```text
BraTS axial slice
[B,4,240,240]
       │
       ▼
trainable 1×1 modality adapter (4→3)
[B,3,240,240]
       │
       ▼
frozen DINOv3 ViT-B/16 patch embedding
225 row-major spatial tokens + CLS + 4 storage tokens
[B,230,768]
       │
       ├── frozen blocks; capture normalized patch tokens at [2,5,8,11]
       │
       ▼
four dense feature maps
4 × [B,768,15,15]
       │
       ▼
trainable lateral 1×1 projections (768→128), concatenate, 3×3 fuse
[B,128,15,15]
       │
       ▼
bilinear ×2 + 3×3 conv blocks: 15→30→60→120→240
[B,64,240,240]
       │
       ▼
1×1 prediction head
logits [B,3,240,240]  ── sigmoid at evaluation only ──► WT, TC, ET
```

The logits follow the repository's existing multi-hot target contract. Training should use a sigmoid-compatible multi-label loss, not softmax, because WT/TC/ET overlap.

### Intermediate features

For the 12-block ViT-B model, capture blocks `[2,5,8,11]` using the official API:

```python
features = backbone.get_intermediate_layers(
    adapted_images,
    n=[2, 5, 8, 11],
    reshape=True,
    return_class_token=False,
    return_extra_tokens=False,
    norm=True,
)
```

The layer selection is a balanced early/middle/late recommendation, not a DINOv3 requirement. A last-layer-only decoder is a useful lower-complexity ablation.

### Decoder recommendation

Use a lightweight same-grid multi-layer fusion decoder rather than a full U-Net or Mask2Former:

1. Project each `[B,768,15,15]` feature to 128 channels with a `1×1` convolution.
2. Concatenate and fuse with a `3×3` convolution, normalization, and GELU.
3. Upsample in four 2× stages with bilinear interpolation followed by `3×3` refinement convolutions.
4. Produce three logits with a final `1×1` convolution.

This remains lightweight, exposes a clear dense-feature boundary for token merging, and avoids transposed-convolution checkerboard artifacts. Patch-16 features are intrinsically coarse for small ET regions; the four-stage decoder can smooth but cannot recover information destroyed by tokenization. A shallow high-resolution skip from the adapted input may be evaluated later, but excluding it keeps the first baseline's representation attributable to DINOv3.

## Adaptive token-merging integration

### What exists in this repository

At commit `845a4d0`, there is **no adaptive token-merging implementation or callable interface**. `src/cap6411_atm/merging/__init__.py` contains no implementation, and `notebooks/02_token_merging.ipynb` says to begin with synthetic tensors after components are added. The README describes only the conceptual sequence:

```text
DINOv3 encoder → adaptive token merging → dense feature reconstruction → segmentation head
```

Therefore, no exact existing merge function signature can be derived. The only existing shared model interface is `logits = model(images)` with `[B,4,H,W] → [B,3,H,W]`. Any claim that the repository already accepts a particular token shape would be false.

### Exact DINOv3-side contract that the future merger must satisfy

The following contract is derived from the official ViT implementation and should be agreed with Workstream 2 before model code is written:

```python
merged_patch_tokens, merge_state = merger(
    patch_tokens,       # [B, N, D], ViT-B: [B,225,768]
    grid_size=(15, 15),
)

dense_patch_tokens = reconstructor(
    merged_patch_tokens,  # [B,M,D], M <= N
    merge_state,
)                         # [B,225,D]
```

`merge_state` must be sufficient to map every original row-major patch position to a merged representative or weighted combination. At minimum it needs original grid size, original-to-merged assignment, merge weights/sizes, and any spatial coordinates needed by later attention. Reconstruction must preserve dtype, device, batch separation, and embedding dimension.

Required invariants:

- Never merge the class token or four storage tokens with spatial patches.
- Patch position `r*15+c` corresponds to grid location `(r,c)` before merging.
- Reconstruction returns exactly 225 patch tokens in original row-major order before reshaping to `[B,D,15,15]`.
- Merging is independent per batch element; no patient or slice may interact with another.
- The unmerged ratio reproduces the baseline numerically within an agreed tolerance.
- Any intermediate feature used by the decoder must either remain dense or have its own valid reconstruction state.

### Where merging can occur

1. **After the final encoder block:** easiest and compatible with the public feature API, but it cannot reduce DINOv3 encoder computation. It only tests decoder/reconstruction behavior and must not be reported as encoder acceleration.
2. **Between transformer blocks:** necessary to reduce later self-attention and MLP compute, but it requires an explicit DINOv3 block-level wrapper.

The second option is required for the project's main efficiency claim. It has a critical complication: DINOv3 computes rotary positional embeddings from the full `(H,W)` grid for each block. Once arbitrary patches are merged, token count and position no longer match the standard full-grid RoPE tensor. A valid implementation must propagate representative/aggregated coordinates and apply compatible rotary positions, or use a merge scheme whose sequence and positional treatment the modified attention explicitly supports. Simply shortening the patch sequence while calling the unmodified next block is not a valid integration.

Recommended integration sequence:

```text
prepare tokens (CLS + 4 storage + 225 patches)
→ frozen blocks 0..k
→ split protected and patch tokens
→ merge patches + retain spatial state
→ concatenate protected + merged patches
→ merging-aware blocks k+1..11 with matching positional coordinates
→ separate patches
→ reconstruct 225 dense patch tokens
→ reshape 15×15
→ segmentation decoder
```

Because the backbone is frozen, gradients need not be stored through its standard blocks. If the merger has trainable parameters inside the encoder path, however, wrapping the entire encoder in `torch.no_grad()` would also suppress merger gradients. The wrapper must freeze backbone parameters while retaining autograd through trainable merge decisions and the decoder, or initially use a deterministic/non-trainable merger.

## T4 memory and runtime considerations

Google Colab T4 generally provides 16 GB VRAM. The following are estimates until measured in the target runtime:

- ViT-B/16 weights are roughly 344 MB in FP32 or 172 MB in FP16 before framework overhead.
- At 240×240 there are only 230 total internal tokens, much fewer than high-resolution natural-image dense-prediction settings.
- A truly frozen encoder run without activation-gradient storage is substantially cheaper than fine-tuning.
- The trainable adapter and proposed decoder should be small relative to the backbone.
- Mixed-precision inference through the frozen encoder and mixed-precision decoder training should make batch sizes 4–8 plausible, but dataset cache, DataLoader workers, PyTorch kernels, loss tensors, and optimizer state still require measurement.
- ViT-L/16 has 300M weights and twice the depth; even if it fits at a small batch, it reduces T4 iteration speed and leaves less operational headroom. H+/7B are inappropriate.

Before training, measure batch sizes 1, 4, and 8 with `torch.cuda.max_memory_allocated()`, wall-clock latency after warm-up, and the exact Colab software/GPU configuration. Use the same precision and batch size for baseline-versus-merging comparisons. Do not treat parameter-size arithmetic as a measured peak-memory result.

## Technical risks and mitigations

| Risk | Consequence | Mitigation/test |
|---|---|---|
| Natural-image to MRI domain shift | Frozen features may be weak despite good natural-image dense benchmarks | Compare ViT-S/B frozen probes; inspect feature maps; retain a simple CNN control |
| Learned 4→3 bottleneck | One modality contribution may collapse | Log adapter weights/activations; ablate modified patch embedding and fixed projections |
| Existing MRI z-score differs from DINO RGB preprocessing | Frozen activation scale may be poorly calibrated | Measure early/late activation statistics; avoid unprincipled ImageNet normalization |
| Patch size 16 | Small ET and boundaries can be lost | Multi-layer features, boundary metrics, small-lesion stratification; later consider high-resolution skip |
| Overlapping targets | Softmax would encode the task incorrectly | Three logits with sigmoid-compatible Dice/BCE-style loss |
| RoPE incompatible with shortened token sequence | Incorrect merging or runtime shape failure | Design coordinate-aware block wrapper; add shape and no-merge equivalence tests |
| Reconstruction smears boundaries | Dice/boundary degradation | Preserve assignment/weights; visualize reconstructed grids; measure boundary metrics by merge ratio |
| Public API/version drift | Loading or token offsets change | Pin DINOv3 commit or Transformers version; assert `n_storage_tokens`, patch size, and all tensor shapes |
| Gated weights/license | Colab or teammate cannot reproduce | Obtain approval early; document approved storage and license; never commit URLs/weights |
| T4 estimate differs from reality | OOM or unusable throughput | Run a no-training memory/latency smoke test before selecting batch size |

## Final recommendation and staged validation

1. Obtain team acceptance of the current DINOv3 license and approved access to ViT-B/16 LVD-1689M weights.
2. Pin the official DINOv3 source revision and use the local-checkpoint `torch.hub` path.
3. Implement the unmerged baseline as 4→3 adapter, frozen ViT-B/16, block `[2,5,8,11]` feature extraction, lightweight fusion decoder, and three sigmoid-style logits.
4. Add contract tests asserting 240→15×15, 225 patch tokens, five protected tokens, row-major reconstruction, and `[B,3,240,240]` output.
5. Run a T4 forward/backward memory smoke test before any full training. Fall back to ViT-S/16 if batch size or throughput is unacceptable.
6. Coordinate the merger/reconstructor contract with Workstream 2. First prove no-merge numerical equivalence, then final-layer merge/reconstruction correctness, then implement coordinate-aware inter-block merging for actual encoder speedup.

This design keeps the first baseline reproducible and lightweight while establishing the precise spatial-token contract required for later adaptive merging.

## Primary sources

- [DINOv3 paper](https://arxiv.org/abs/2508.10104)
- [Official DINOv3 repository and pretrained-model instructions](https://github.com/facebookresearch/dinov3)
- [Official DINOv3 model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)
- [Official backbone definitions](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py)
- [Official ViT token and feature APIs](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py)
- [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- [Meta weight-access form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
