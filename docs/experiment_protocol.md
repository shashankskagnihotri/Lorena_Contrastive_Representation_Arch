# Experiment protocol

This document specifies the experiments that produced the reported results. The primary scientific question is whether fixed, signed contrast representations can be sparsified so that removing empty image patches preserves classification quality while reducing **measured online inference cost**. The frontend is an image transform; the objective is ordinary supervised cross-entropy, not contrastive learning.

There are two studies. `training_sparse_testing_sparse` trains a separate model at each imposed sparsity. `training_dense_testing_sparse` freezes existing zero-imposed-sparsity checkpoints and changes inference sparsity without training. Their results and identities must remain separate. The earlier batch-8 campaign is a preserved development/history archive; the reported primary training results use the subsequently authorized larger-batch recipe below.

The machine-readable specifications are published under [experiments/training_sparse_testing_sparse](../experiments/training_sparse_testing_sparse/) and [experiments/training_dense_testing_sparse](../experiments/training_dense_testing_sparse/). The [results document](results.md) distinguishes complete accuracy coverage from the dated partial follow-up latency snapshot.

## 1. Active conditions and seeds

The active dataset scope follows the user's explicit revision: MNIST contains its native grayscale baseline and grayscale contrast only. CIFAR-10 contains its RGB baseline and all three contrast representations. Withdrawn MNIST single-color/opponent-color experiments are not part of the results.

For each dataset and architecture, the registry contains one raw dense baseline. For every active contrast representation it contains one independently trained dense 0% control and five independently trained compact models at imposed coefficient sparsities **0%, 20%, 40%, 60%, and 80%**. Every condition uses seeds **0, 1, and 2**.

| Dataset | Architectures | Contrast representations | Conditions per architecture | Conditions across architectures | Main training runs |
|---|---|---|---:|---:|---:|
| MNIST | ViT-Small, Swin-Tiny | Grayscale | 1 + 1 + 5 = 7 | 14 | 42 |
| CIFAR-10 | ViT-Small, Swin-Tiny | Single color, grayscale, color opponency | 1 + 3 × (1 + 5) = 19 | 38 | 114 |
| Total | | | | **52** | **156** |

Four full raw-model pilots, one per dataset/architecture group, were run for the prespecified training-budget and validation checks. They are additional execution evidence, not additional independent seeds or main-result rows. The final tables exclude pilots and implementation checks.

`dense`, `compact`, and `dense_masked` have distinct meanings:

| Execution | Patch support | Purpose |
|---|---|---|
| `dense` | Every native-grid patch is retained | Raw baseline and independently trained dense contrast controls |
| `compact` | Empty native patches are physically removed; implementation padding is masked | Main sparse computation |
| `dense_masked` | The same sparse support is represented on the dense grid, with absent features masked | Numerical and timing control using the **same checkpoint** as compact execution |

Dense-masked timing controls do not create additional trained models. A dense contrast model at 0% and a compact model trained at 0% are distinct conditions: natural zeros can already remove patches in compact execution.

## 2. Data, splits, and permitted information

| Dataset | Official training population | Training partition | Held-out validation | Official clean test | Input resolution |
|---|---:|---:|---:|---:|---|
| MNIST | 60,000 | 55,000 | 5,000 | 10,000 | 28 × 28, one channel |
| CIFAR-10 | 50,000 | 45,000 | 5,000 | 10,000 | 32 × 32, three channels |

The primary protocol is `heldout_val`. The split is deterministic and stratified, with `split_seed=2026`; exact sample IDs and split hashes are saved. No full-training-set refit was enabled. Corruption benchmarks and the clean test sets contribute neither normalization statistics nor training/validation decisions.

Original datasets are reused through a validated dataset manifest, rather than copied into each run. Preparation validates publisher archive checksums where available, array/image counts, labels, layout, range, corruption names, and severity indexing. MNIST-C and CIFAR-10-C arrays are memory-mapped for corruption loading. The public repository describes data acquisition and preserves integrity metadata; the datasets themselves are external artifacts.

Each official evaluation cell has 10,000 examples. A “cell” means the clean test set or one named corruption at one official severity. Every sample contributes to summed loss, correct/total counts, confusion matrices, and saved prediction identities. Unequal batches are not averaged with equal weights.

## 3. Fixed representation and normalization

The authoritative frontend is `semseg/utils/blur_preprocessing.py` from `shashankskagnihotri/semseg_using_VLMs_Lorena`, commit `73deca8c15b8d91fee29bac71b6f73c013a47d22`, original file SHA-256 `f7c4420a3ab2bf4036323b9b4842e66d46225a88ce6f4440101c7a190861dff8`. Its native transform was inspected and numerically matched, rather than replaced with another blur operator. Attribution and the original source context are preserved. No upstream license is invented where none was found.

Input uint8 images become float32 values in `[0,1]` by division by 255. The fixed study transform uses one 3 × 3 uniform box-blur iteration with zero padding, then signed center-minus-blur contrast. Let the three channel contrasts be `D_R`, `D_G`, and `D_B`:

| Representation | Native coefficients per spatial location |
|---|---|
| Single color | `D_R`, `D_G`, `D_B` |
| Grayscale contrast | `(D_R + D_G + D_B) / 3` |
| Color opponency | `(D_R + D_G + D_B) / 3`, `(D_R − D_G) / 2`, `D_B / 3 − D_R / 6 − D_G / 6` |

The positive-blur-depth source uses **equal RGB weights**, despite an upstream prose description mentioning luminance weights. The experiment preserves the actual source arithmetic. MNIST is replicated to three identical channels only at the frontend input and returns native one-channel grayscale contrast. The original downstream channel adapter is outside the native transform and is omitted. Signed values are neither clipped to positive magnitudes nor reconstructed into RGB.

Raw baselines use the chosen published `[0,1]` normalization recipe:

| Dataset | Raw-input mean | Raw-input standard deviation |
|---|---|---|
| MNIST | `(0.1307,)` | `(0.3081,)` |
| CIFAR-10 | `(0.4914, 0.4822, 0.4465)` | `(0.2023, 0.1994, 0.2010)` |

Each of the **four active dataset/contrast combinations** has its own frozen channel-wise population mean and standard deviation. They were estimated from every unaugmented clean training image exactly once at 0% imposed thresholding. Frontend arithmetic is float32; deterministic chunked moment accumulation is float64. Natural zeros remain observations. Per-channel pixel counts are 43,120,000 for MNIST and 46,080,000 for CIFAR-10.

Population variance uses correction zero. For a channel with measured standard deviation below `1e-6`, the saved effective standard deviation is 1.0; its channel is retained. Means, measured/effective standard deviations, guards, counts, source/split identities, and artifact hashes are stored. The same fit is reused for every sparsity, architecture, seed, corruption, and execution control. Diagnostic test-distribution moments never update these buffers.

## 4. Exact sparsification and support rule

For an image with `M` native contrast coefficients and imposed percentage `p`, remove

```text
n_remove = floor(p × M / 100)
```

coefficients using a stable ascending sort of their absolute **unnormalized native** magnitudes. Native channel/raster order breaks ties. Removed entries become exactly zero; all other values and signs are preserved. At 0%, sorting and thresholding are bypassed. Existing zeros count toward the requested fraction, so achieved zeros may exceed the nominal percentage and adjacent settings can produce identical tensors.

The input patch size is 2 × 2. For each patch, retain it if any native coefficient has nonzero absolute magnitude. The primary `token_epsilon` is exactly 0.0. There is no independent token quota, top-K patch rule, learned router, or minimum image-token budget.

Support is decided **before** affine normalization, projection bias, or positional embeddings. Retained coefficients use `(C_s − mean) / effective_std`. Zeroed coefficients inside a retained patch receive that same affine transform; they are not silently re-zeroed. An empty patch remains absent even if affine centering would turn its numerical zeros into nonzero offsets. Dense normalization followed by gathering through the original mask is an equivalent implementation.

## 5. Models and compact execution

Both architectures are trained from scratch, at the native image resolutions, with native representation channel counts in their patch stems. They are small-image adaptations, not pretrained ImageNet checkpoints or unmodified 224-pixel reference configurations.

| Component | ViT-Small | Swin-Tiny |
|---|---|---|
| Patch size | 2 × 2 | 2 × 2 |
| Initial grid | MNIST 14 × 14; CIFAR 16 × 16 | MNIST 14 × 14; CIFAR 16 × 16 |
| Embedding width | 384 | 96 initially; doubles at merging |
| Depth | 12 transformer blocks | Stage depths `(2,2,6,2)` |
| Heads | 6 | `(3,6,12,24)` |
| MLP expansion | 4 | 4 |
| Classifier | Class token, final normalization, 10 classes | Final normalization, supported-token pooling, 10 classes |

ViT gathers surviving patches before projection, retains their original positional coordinates, and packs sequences to the batch maximum with masked padding. The class token remains even for an empty image. Padding and the host/device synchronization needed to establish the dynamic length count as actual implementation cost.

Swin retains sparse coordinates through shifted-window attention and hierarchical merging. The windows are `(7,7,4,2)` for MNIST and `(4,4,4,2)` for CIFAR-10. Stage grids are `(14,7,4,2)` and `(16,8,4,2)`, respectively. A merged parent is present if any child is present; missing children are structural zero features, not newly normalized image pixels. Supported-token pooling uses the actual retained count. Entirely empty inputs follow the explicitly defined zero-pooled-feature classifier path.

Compact and dense-masked paths use the same learned parameters, native support, and normalization. Parameter counts vary with native input channel count and are recorded rather than assumed identical. Name-keyed initialization matches shared parameter tensors across comparable representation conditions for the same seed.

## 6. Actual larger-batch training recipe

The user authorized a new campaign with one verified physical batch size per dataset/architecture, held fixed across representations, sparsities, controls, and seeds. It replaced the primary batch-8 recipe and is **not claimed to be optimization-equivalent to batch-8 training**. The old campaign was preserved separately.

| Dataset / architecture | Train batch | Validation/test batch | Epochs | Warmup epochs | Base learning rate |
|---|---:|---:|---:|---:|---:|
| MNIST / ViT-Small | 256 | 256 | 50 | 5 | 0.0004242640687 |
| MNIST / Swin-Tiny | 1,024 | 1,024 | 50 | 5 | 0.0008485281374 |
| CIFAR-10 / ViT-Small | 128 | 128 | 200 | 10 | 0.0003000000000 |
| CIFAR-10 / Swin-Tiny | 512 | 512 | 200 | 10 | 0.0006000000000 |

Learning rates follow the recorded amendment `3e-4 × sqrt(batch_size / 128)`. Each row uses AdamW with betas `(0.9,0.999)`, weight decay 0.05, and no weight decay on biases, normalization parameters, positional/class-token parameters, or other one-dimensional parameters. Gradients are clipped to global norm 1.0. Warmup is linear per optimizer step, followed by cosine decay toward `1e-6` over the fixed epoch budget. Cross-entropy has no label smoothing, mixup, or auxiliary loss. Dropout is 0; stochastic depth increases through the blocks to 0.1. All reported computation uses FP32.

There is no gradient accumulation: physical and effective training batches are equal. Incomplete final batches are retained. DataLoader worker counts are zero, using resident clean data without multiplying worker memory. Training/evaluation batch sizes do not change the latency protocol's batch sizes 1 and 8.

CIFAR-10 training applies zero padding of four pixels, a uniformly selected 32 × 32 crop, and a random horizontal flip, all **before** contrast extraction. MNIST uses its unaugmented native image. Augmentation decisions and epoch sample orders are keyed independently by seed, epoch, and sample identity, so model RNG consumption cannot change the input sequence. Validation and official tests use no random augmentation.

The primary checkpoint is the **final fixed-budget `final.pt`**, not the maximum-test-accuracy epoch. `best_validation_diagnostic.pt` is retained only as a diagnostic. Full validation is logged every epoch. Corruption tests do not select epochs, sparsities, or representations. Four raw pilots passed the prespecified validation floor, 97% for MNIST and 65% for CIFAR-10, before the main sweep.

## 7. Evaluation metrics and corruption coverage

MNIST-C has 15 official fixed-severity corruptions: shot noise, impulse noise, glass blur, motion blur, shear, scale, rotate, brightness, translate, stripe, fog, spatter, dotted line, zigzag, and Canny edges. It has no separate contrast corruption and no separately named “lighting” cell.

CIFAR-10-C has 15 corruptions, each at severities 1 through 5: Gaussian noise, shot noise, impulse noise, defocus blur, glass blur, motion blur, zoom blur, snow, frost, fog, brightness, contrast, elastic transform, pixelate, and JPEG compression. Brightness and fog are reported directly; “lighting robustness” is not treated as an additional invented metric.

For each seed, per-corruption CIFAR error is the arithmetic mean of the five severity errors. Overall `mCE_raw` is the unnormalized arithmetic mean of corruption-cell error: 15 MNIST cells or 75 CIFAR cells. It excludes clean test error and is **not** an AlexNet-normalized corruption score. Reported percentages multiply error fractions by 100.

The primary 156 runs produce 9,336 official evaluation cells: 42 MNIST runs × 16 cells and 114 CIFAR runs × 76 cells. Per-class counts, confusion matrices, support statistics, loss, and stable sample IDs accompany the scalar scores. Means and sample SDs are computed over the three seed-level results, not over individual images as if they were independent training replicates. Paired differences match dataset, architecture, and seed.

## 8. Frozen-checkpoint inference follow-up

The follow-up uses all 60 eligible final checkpoints: 12 raw dense references, 24 dense-trained contrast checkpoints at zero imposed sparsity, and 24 contrast checkpoints trained with compact execution at zero imposed sparsity. These are 18 MNIST and 42 CIFAR-10 checkpoints. The two contrast training families remain distinguishable in every case identity.

Each source receives compact inference at `0,10,20,...,90%`, giving 600 interventions. Twelve unchanged raw dense reference cases bring the total to **612 cases**. The registered raw intervention ranks native raw pixel magnitudes before normalization and is labeled `raw_pixels`; it is not called contrast sparsification. Contrast intervention uses `native_contrast`. The requested three-PDF comparison set displays contrast curves and an unchanged raw baseline; it does not merge raw-pixel sweeps into the contrast curves.

No weights, normalization statistics, training recipes, or checkpoint selections are updated. The same checkpoint's original dense evaluation is retained alongside compact-at-zero inference, because those executions need not be equivalent on naturally empty patches. All checkpoint loads are strict and bind the original completion/checkpoint hashes.

The 612 accuracy cases comprise **35,352 cells**: 612 clean and 34,740 corrupted. This follow-up was designed after the original test results were inspected; it is an exploratory reuse of those benchmarks, not a newly untouched test set or an independent model-selection validation set.

## 9. Latency, throughput, memory, and hardware

Latency is measured on saved deterministic real-image panels, independently of predictions and observed speedups. Compared methods share sample memberships within each dataset/architecture/seed/batch/corruption cell. Every primary worker uses a fresh process. Measurement batches are **1 and 8**, for clean input and all official corruption cells.

| Scope | Measured work |
|---|---|
| `gpu_raw_to_logits` | GPU-resident raw image through conversion, frontend, sparsification, support/gather, normalization, embedding, backbone, and logits |
| `host_raw_to_logits` | Host raw input, host-to-device transfer, then the complete online pipeline to logits |
| `loader_to_cpu_prediction` | Loader/batch retrieval, host-to-device transfer, complete pipeline, prediction, and required CPU return |
| `cached_input_model_only_diagnostic` | Previously prepared inputs through the model; explicitly excludes preparation and is not the online headline |

Each scope uses **50 warmup batches and 200 serial measured repetitions**. Synchronization occurs at the documented scope boundaries. Separate 200-batch sustained-throughput measurements avoid per-batch synchronization beyond operations that the model or required CPU return already impose. CUDA elapsed time is retained alongside the scope's wall time. Saved statistics include mean, median, p95, milliseconds per batch/image, throughput, and maximum allocated/reserved memory.

Memory setup synchronizes, collects garbage, empties unreferenced CUDA cache, performs the 50 scope-specific warmups, synchronizes again, then resets peak counters. Cache clearing or peak resets do not occur inside timed calls. Profiling and token/support telemetry are collected separately. Inclusive profiler intervals are **nonadditive**; they cannot be stacked or subtracted from total latency to invent a frontend cost. Empty-input diagnostics are labeled separately from real-image timing.

The primary homogeneous cohort uses NVIDIA RTX A6000 GPUs, AMD EPYC 7413 hosts, FP32 explicit PyTorch attention, four PyTorch intra-op threads, and 48 inter-op threads. The recorded software contract includes PyTorch 2.9.1+cu126, CUDA 12.6, cuDNN 9.10.2, and driver 610.57.04. GPU UUIDs and measurement sessions are retained. Matching hardware class does not imply the same physical GPU. Earlier Ada measurements remain separate and are never pooled into A6000 coverage. Each timing allocation is isolated; independent workers may run concurrently on different GPUs. Authorized H100 capacity was unavailable to the campaign account during this deployment.

Primary timing covers **552 workers and 33,312 official cells**, including compact and same-checkpoint dense-masked controls. The follow-up plans **2,424 workers and 140,304 official timing cells**. A reported three-seed point requires all three matching completed worker receipts; missing measurements are not replaced with zeros or interpolated estimates. Across-seed latency summaries use the mean and sample SD of each seed's median. Paired speedup is computed per seed as `baseline_median / method_median`, then averaged; it is not generally the ratio of the two cross-seed means.

## 10. Numerical validation amendment and provenance

A bounded timing-validation repair on October 8 addressed 16 MNIST ViT raw-pixel, seed-1, batch-8 compact/dense-masked workers at imposed levels 0–70%. The original strict FP32 compact-versus-dense-masked check remains first. If it fails, a copied backbone with the same FP32-stored weights and prepared inputs promoted to FP64 must pass a tighter semantic comparison; both original FP32 outputs must also agree with their respective references. This is an untimed validation amendment, not a change to measured inference precision or kernels.

The real failing panel had maximum FP32 path disagreement `3.55e-5`, while FP64 disagreement was `4.62e-14`. The amended worker releases the copy, restores RNG state, verifies restored GPU allocations, then enters the original timing warmup protocol. Receipt JSON and fallback tensor archives are hash-bound. The amended controller verifies them when accepting recovered workers and rechecks all 16 immediately before campaign completion. The immutable worker and orchestration snapshots are distinct from the scientific source identity.

The original FP32 guard uses `rtol=1e-4, atol=2e-5`. The independent FP64 compact/dense-masked comparison uses `rtol=1e-10, atol=1e-11`; each original FP32 output must match its own FP64 reference with `rtol=1e-4, atol=1e-4`. Nonfinite outputs and semantic/reference failures still fail visibly. This bounded fallback is recorded explicitly rather than retroactively relabeling the scientific implementation as unchanged validation policy.

| Identity | Frozen source hash |
|---|---|
| Larger-batch training/evaluation | `9359e4f10c0c2184eef4af8885edc7076bded30188775cc14222f79819f648c4` |
| Primary computation/timing/report base | `7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0` |
| Follow-up scientific inference | `8cf4d0c1e0ce5443d5080ea397df9a8f085d1bf2c4292912676545d38533a0cb` |
| Untimed numerical-validation worker | `3c965bf1a2c9f3f11bb14c0d6ad3354a2266e7d0763b2146b5a8fbc69d314769` |
| Recovery controller with final receipt recheck | `fcfbdad09dd2aa67d5e62616a5ed77f19f205b324d0474ad15af15bef15466f7` |

The public [source snapshots](../provenance/sources/) preserve those identities; the organized [source tree](../src/) provides the implementation entry points. Exact environments, resolved configurations, split/statistic hashes, checkpoint hashes, and per-attempt hardware/command records complete the provenance chain.

Python/NumPy/PyTorch RNGs are seeded, `PYTHONHASHSEED` and deterministic cuBLAS requirements are set before launch, deterministic PyTorch algorithms are enabled, cuDNN benchmarking is disabled, and TF32 is disabled. Resume occurs at an epoch boundary with model, optimizer, schedule step, RNG, and sampler epoch restored. This policy supports reproducibility on the recorded stack; it does not promise bitwise equality across GPU types or library releases.

TensorBoard accompanies machine-readable logs: every epoch's full online training and held-out validation metrics; step metrics every 50 optimizer steps plus first/last steps; parameter/gradient and fixed-image diagnostics every 10 epochs and at completion; official evaluation metrics; timing distributions; and cross-seed summaries. Fixed class-balanced visualization IDs and signed display scales are saved separately from model inputs. Numeric JSON/CSV/prediction evidence remains the source of truth even if a TensorBoard view downsamples events.
