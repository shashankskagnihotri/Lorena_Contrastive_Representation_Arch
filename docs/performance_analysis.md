# Why sparse contrast often loses accuracy and does not reduce latency

This analysis separates three experimentally different effects: changing the image representation, changing the token graph seen by a trained model, and implementing that graph efficiently on a GPU. The experiments show failures in all three areas, but they do not establish that sparsity is inherently ineffective. They establish that these representations, training recipes, and eager PyTorch execution paths usually fail to improve the measured accuracy–latency tradeoff.

The main comparisons below use the completed **training-sparse/testing-sparse** study: three independently trained seeds, the prescribed final-epoch checkpoint, all official accuracy cells, and the homogeneous RTX A6000 timing cohort. The **training-dense/testing-sparse** follow-up supplies checkpoint interventions that help explain the accuracy results. Its accuracy results are complete; its timing campaign was still completing numerical-validation recovery at this document's 2026-10-08 evidence cutoff. No missing follow-up latency is inferred from accuracy, training duration, or a different hardware cohort.

The exact method, equations, architecture, normalization, checkpoint semantics, and scope are in [method.md](method.md). The full result tables are in [results.md](results.md). Selected profiles and support records used here are preserved in [results/diagnostics](../results/diagnostics/README.md), with original SHA256 identities and a CPU-only reproduction script. No new timing runs were performed to write this document.

## 1. What the main results actually say

### 1.1 Accuracy and corruption error

The following are means ± sample standard deviations over seeds 0, 1, and 2. Accuracy and error are expressed in percent. “Gray 60” means grayscale contrast, 60% imposed coefficient sparsity, trained and tested with compact execution. The raw model is trained and tested with dense execution.

| Dataset | Architecture | Raw clean accuracy | Gray 60 clean accuracy | Raw mean corruption error | Gray 60 mean corruption error |
|---|---|---:|---:|---:|---:|
| MNIST | ViT-Small | 98.300 ± 0.053 | 97.883 ± 0.117 | 27.104 ± 0.167 | 37.118 ± 0.451 |
| MNIST | Swin-Tiny | 99.233 ± 0.015 | 99.037 ± 0.064 | 24.035 ± 0.579 | 28.738 ± 2.215 |
| CIFAR-10 | ViT-Small | 78.103 ± 0.702 | 68.403 ± 0.491 | 35.560 ± 0.177 | 54.314 ± 0.744 |
| CIFAR-10 | Swin-Tiny | 87.787 ± 0.148 | 79.853 ± 0.385 | 31.020 ± 0.283 | 44.355 ± 0.043 |

These are official clean-test accuracies, not held-out training-validation accuracies. The repository's `mCE_raw` is the **unnormalized arithmetic mean of corruption error fractions**: 15 MNIST-C corruption cells, or 15 CIFAR-10-C corruptions × 5 severities. Multiplication by 100 gives the percentages above. It is not the AlexNet-normalized ImageNet-C mCE convention. Lower is better. [Source: condition table](../results/training_sparse_testing_sparse/conditions.csv).

MNIST gray-60 loses only 0.417 percentage points for ViT and 0.197 for Swin on clean images. Its much larger corruption-error deterioration is a different result and should not be hidden by the near-99% clean accuracy. Conversely, describing every grayscale result as unsuccessful would be wrong: **MNIST Swin with dense grayscale at 0% has mean corruption error 22.787%, versus raw 24.035%**, while its clean accuracy is 99.000% versus raw 99.233%. That is an observed robustness tradeoff for a particular control, not a general benefit of compact sparsity.

Three seeds quantify a limited source of variability. Their standard deviations are not confidence intervals over datasets, hardware conditions, all training recipes, or all implementations. The table supports the reported within-study comparisons; it does not justify claims about a universally optimal representation.

### 1.2 Latency on the same hardware class

These are **milliseconds per batch**, for decoded raw input already on the GPU through final GPU logits. Each seed contributes the median of 200 synchronized, uninstrumented calls after 50 warmups. The table reports the mean ± between-seed sample standard deviation of those three medians. Ratio means compact mean divided by raw mean; greater than one is slower.

| Dataset | Architecture | Batch | Raw ms | Gray 60 ms | Compact/raw ratio |
|---|---|---:|---:|---:|---:|
| MNIST | ViT-Small | 1 | 10.088 ± 0.033 | 10.915 ± 0.044 | 1.082 |
| MNIST | ViT-Small | 8 | 10.609 ± 0.085 | 11.561 ± 0.037 | 1.090 |
| MNIST | Swin-Tiny | 1 | 18.195 ± 0.047 | 26.047 ± 0.271 | 1.432 |
| MNIST | Swin-Tiny | 8 | 18.780 ± 0.083 | 67.446 ± 0.503 | 3.591 |
| CIFAR-10 | ViT-Small | 1 | 10.186 ± 0.065 | 10.888 ± 0.031 | 1.069 |
| CIFAR-10 | ViT-Small | 8 | 11.647 ± 0.134 | 11.567 ± 0.144 | 0.993 |
| CIFAR-10 | Swin-Tiny | 1 | 18.306 ± 0.066 | 34.919 ± 0.136 | 1.908 |
| CIFAR-10 | Swin-Tiny | 8 | 18.748 ± 0.221 | 46.826 ± 0.051 | 2.498 |

[Source: latency overview](../results/training_sparse_testing_sparse/latency_overview.csv), filtered to `hardware_id=4ba79ef9…`, `scope=gpu_raw_to_logits`, `input_group=clean`, and `statistic=median_ms`.

The CIFAR-10 ViT batch-8 result is important counterevidence to a blanket “compact is always slower” claim. Its approximately 0.7% advantage in the ratio of means is small and is **not present in all seeds**. Seed 0 is slower, 11.585 ms versus raw 11.495 ms. It should not be promoted into a reliable deployment speedup without additional controlled evidence.

Batch-8 latency is not single-image latency. Dividing it by eight gives an amortized cost at that batch size, not the response time of a batch-1 request. Nor is the reciprocal of a median interchangeable with the separately measured sustained-throughput statistic. The original batch-8 training campaign, validation-loop duration, GPU-resident inference, host-to-GPU inference, and loader-to-CPU prediction are separate measurements.

The main negative systems result is especially strong for Swin. It is not merely the extra grayscale frontend: a same-checkpoint control below is much faster while preserving exactly the compact model's absent-token semantics.

## 2. “60% sparsity” does not mean 60% fewer tokens

### 2.1 The quantity being ranked

For each image with native coefficient tensor $c\in\mathbb{R}^{C\times H\times W}$, the implementation ranks all $CHW$ absolute coefficient magnitudes jointly and zeros the first

$$
k=\left\lfloor \frac{p}{100}CHW\right\rfloor
$$

entries, using a stable ordering for ties. Existing exact zeros are part of this ranking. It does not remove $p$ percent of the coefficients that were previously nonzero. It does not rank each channel separately. At $p=0$, it returns the original coefficients without sorting. At $p=60$, the sort still executes even when most or all selected values are already zero. [Implementation: `sparsify`](../src/sparse_contrast/frontend.py#L122).

A 2×2 token is absent only if **all coefficients across all native channels in that patch are exactly zero before normalization**. A three-channel patch contains 12 scalar coefficients; a grayscale patch contains four. The scalar denominator therefore changes with the representation, and equal nominal sparsity does not imply equal useful-signal removal or equal token retention.

### 2.2 Actual full-training-partition diagnostic

The following are averages over the full clean training partition, before augmentation: 55,000 MNIST images or 45,000 CIFAR-10 images. They are preprocessing diagnostics, not test accuracy, GPU utilization, or latency measurements.

| Dataset / representation | Natural zero coefficients at 0% | Zero coefficients at 60% | Empty 2×2 patches at 0% | Empty 2×2 patches at 60% |
|---|---:|---:|---:|---:|
| MNIST grayscale | 68.1166% | 68.5100% | 61.9587% | 62.1394% |
| CIFAR-10 grayscale | 0.0462% | 59.9609% | 0.0358% | 29.8450% |
| CIFAR-10 single-color | 0.9650% | 59.9935% | 0.1566% | 23.7972% |
| CIFAR-10 color opponency | 1.9981% | 59.9935% | 0.1567% | 4.6352% |

[Exact CSVs and their source hashes](../results/diagnostics/preprocessing/).

On clean MNIST, moving from imposed 0% to imposed 60% adds only **0.3935 percentage points of scalar zeros and 0.1806 percentage points of empty patches** on average. Most of the compact support reduction already exists naturally at 0%. Paying for a full ranking pass at 60% can therefore buy almost no additional clean-image token saving.

The averages do not imply that every MNIST image already has at least 60% zeros. Some images are affected, and corrupted images can have very different support. The pruning threshold is image-dependent. For example, adding noise can create many nonzero coefficients in formerly flat areas, so the same nominal percentage can remove a very different set of signals. Support is recomputed from the actual corrupted input; the implementation does not reuse a clean image's support.

For intuition only, if every coefficient were independently zero with probability $q$, a $C$-channel 2×2 patch would be empty with probability $q^{4C}$. At $q=0.6$, that is 12.96% for one channel and about 0.218% for three channels. Real images violate independence strongly, as the empirical table demonstrates. This formula explains why scalar sparsity is an insufficient token statistic; it is not a model fitted to these images.

The color-opponency result is particularly revealing: nearly 60% zero coefficients leave more than 95% of the initial patches active. Low-amplitude color-difference channels can lose many coefficients while another channel keeps each spatial patch alive. Saved normalization scales and fixed panels are consistent with this explanation, but a complete channel-wise causal attribution would require a separate analysis. A panel with one entirely zero opponent channel does not imply that the joint multichannel token is absent.

### 2.3 Raw MNIST already has sparse pixels

In the selected clean seed-0 timing panel, raw MNIST has about 80% naturally zero pixels and roughly 51 nonempty 2×2 patches per image. Grayscale contrast has roughly 75 active patches at 60%. A local difference operator can spread a digit's spatial footprint into its neighboring background, so contrast need not make the support smaller than raw pixel support.

The primary raw baseline nevertheless executes **all 196 patches**, as prescribed by its dense execution. Its logged native-support diagnostic must not be mistaken for executed token count. Conversely, sparsifying and compacting the raw model is a different inference policy, evaluated separately in the follow-up, not a hidden property of the raw dense reference. [Selected support records](../results/diagnostics/support_summary.csv).

## 3. Why a zero-input patch is not a neutral token

### 3.1 Dense 0% and compact 0% are different functions

Let a native patch be exactly zero. Frozen normalization produces $-\mu/\sigma$, which need not be zero. A linear patch embedding can add a learned bias. ViT adds a learned positional embedding. Subsequent LayerNorm affine terms, attention projections, residuals, and MLP biases can also produce nonzero features. Even an exactly zero key/value pair is not generally neutral to attention: the key contributes a term to the softmax denominator unless it is excluded by the attention mask.

Thus the dense model can use background positions, their count, their positions relative to the digit, and their interactions with foreground tokens. Removing those nodes after training changes the learned computation. This remains true when the sparse representation values themselves are unchanged.

For ViT, compact execution keeps the original spatial position embeddings and an always-present class token, but removes absent image tokens from attention. For Swin, it removes absent keys/queries within correctly shifted original windows, and later merges depend on surviving support. Its final pooling also operates over active tokens. These are intentional sparse semantics, not a claim that deleting a dense model's zero-valued inputs leaves its prediction invariant. [Backbone source](../src/sparse_contrast/models.py), [method description](method.md).

### 3.2 The checkpoint intervention isolates this effect

The follow-up uses the original checkpoint and frozen statistics with strict state loading. The inference adapter adds no parameters or buffers and keeps training configuration separate from the inference policy. At imposed 0%, switching dense-trained grayscale models to compact execution gives:

| Dataset | Architecture | Original dense-0 clean accuracy | Same checkpoint, compact-0 accuracy | Change |
|---|---|---:|---:|---:|
| MNIST | ViT-Small | 97.660% | 51.420% | −46.240 pp |
| MNIST | Swin-Tiny | 99.000% | 90.320% | −8.680 pp |
| CIFAR-10 | ViT-Small | 76.340% | 76.340% | 0.000 pp in the mean |
| CIFAR-10 | Swin-Tiny | 83.277% | 83.263% | −0.013 pp |

[Source: cross-study condition table](../results/diagnostics/cross_study_accuracy_conditions.csv), families `original_dense_same_checkpoint` and `dense0_compact_test`; [strict-layout adapter](../src/dense_sparse/adapter.py).

This contrast is strong evidence for the natural-zero graph change on MNIST. Clean CIFAR-10 grayscale has almost no naturally empty patches, and its compact-0 intervention barely changes predictions. It is not evidence that strict checkpoint loading failed: a compatible state dictionary does not make two different execution graphs equivalent.

The imposed-60 follow-up accuracy is 51.227% for the dense-trained MNIST ViT and 90.297% for dense-trained MNIST Swin. Almost all of that clean-image damage was already present at compact-0. Calling this solely “damage from pruning 60% of the information” would misidentify the intervention.

### 3.3 Training with compact support prevents that particular mismatch

Models trained compactly at 0% already experience missing natural-zero tokens during training. On MNIST, their mean clean accuracy at compact-0 and compact-60 is unchanged to the displayed precision:

| Architecture | Compact-0-trained, test 0% | Same checkpoints, test 60% | Separate compact-60-trained models, test 60% |
|---|---:|---:|---:|
| ViT-Small | 98.037% | 98.037% | 97.883% |
| Swin-Tiny | 98.983% | 98.983% | 99.037% |

The last column is a separately trained family and must not be pooled with checkpoint interventions. Equality of average clean accuracy also does not mean every image has identical coefficients, logits, or prediction. Under corruption, the compact-0-trained MNIST ViT changes mean corruption error from 38.986% at test 0% to 36.734% at test 60%; Swin changes from 27.061% to 29.492%. The same clean-accuracy pattern can hide different robustness behavior.

CIFAR-10 demonstrates the larger mismatch from genuinely removing signal at test time. Gray-60 accuracy for ViT is 54.267% from dense-0-trained checkpoints and 56.933% from compact-0-trained checkpoints, versus 68.403% when trained at compact-60. For Swin, the corresponding values are 67.557%, 68.087%, and 79.853%. Training with the support distribution helps, but does not recover the raw baseline in these experiments.

## 4. Representation quality and generalization are separate from execution speed

### 4.1 Contrast changes what the model must learn

The center-minus-local-blur transform emphasizes local variation and attenuates smooth structure. Grayscale additionally projects away color distinctions. Magnitude pruning then removes weak coefficients without access to the label or task relevance. A small coefficient can still carry discriminative information; absolute magnitude is not an estimate of its effect on classification loss.

It would be mathematically incorrect to say that every unpruned contrast transform completely destroys all low-frequency or color information. With the actual finite, zero-padded operator, constant images can leave boundary residuals. The single-color spatial operator is invertible in exact arithmetic under the stated boundary condition, although poorly conditioned for smooth components; the opponency channel mixing is also full rank. Grayscale is a genuine color projection, and hard coefficient pruning is lossy. Recoverability in exact arithmetic does not guarantee that a finite-precision network trained from scratch can exploit the weak residual efficiently. [Exact algebra and boundary discussion](method.md).

For MNIST, color is not an explanation for the retained experiment's loss. Its input is one channel, replicated only to reproduce the original grayscale frontend calculation, which returns one contrast channel. There are no retained MNIST single-color or color-opponency results. The inherited panel label `mean_rgb_contrast` refers to that calculation, not a three-channel MNIST color experiment.

### 4.2 Brightness gains do not establish broad robustness

Selected mean accuracies on MNIST-C make the tradeoff concrete:

| Corruption | ViT raw | ViT gray 60 | Swin raw | Swin gray 60 |
|---|---:|---:|---:|---:|
| Brightness | 44.023% | 80.653% | 14.863% | 37.797% |
| Glass blur | 76.377% | 36.490% | 91.550% | 54.660% |
| Impulse noise | 50.240% | 25.977% | 82.890% | 38.963% |
| Motion blur | 87.537% | 61.667% | 96.070% | 88.257% |
| Fog | 42.113% | 37.280% | 41.177% | 31.610% |
| Stripe | 72.297% | 68.530% | 23.627% | 48.683% |

[Official per-cell table](../results/training_sparse_testing_sparse/cells.csv), averaged over the three seeds.

The gray-60 ViT gains 36.630 percentage points on brightness and loses 39.887 on glass blur. Swin gains on brightness, stripe, and canny edges while losing heavily on impulse noise and glass blur. A local contrast operator's response to illumination, blur, and added high-frequency noise provides a plausible mechanism, but these measurements alone do not prove which component of the learned model causes each failure. Clipping and finite image boundaries also prevent an unconditional claim of exact brightness invariance.

CIFAR-10's color-preserving variants show another useful counterexample. At 60%, opponency beats grayscale clean accuracy for both architectures and improves fog accuracy over raw, yet remains worse in overall corruption error:

| Architecture / representation | Clean accuracy | Mean corruption error | Fog accuracy |
|---|---:|---:|---:|
| ViT raw | 78.103% | 35.560% | 56.517% |
| ViT grayscale 60 | 68.403% | 54.314% | 47.767% |
| ViT single-color 60 | 72.867% | 50.967% | 52.834% |
| ViT opponency 60 | 76.110% | 49.604% | 59.797% |
| Swin raw | 87.787% | 31.020% | 73.523% |
| Swin grayscale 60 | 79.853% | 44.355% | 67.891% |
| Swin single-color 60 | 83.377% | 41.957% | 72.453% |
| Swin opponency 60 | 85.150% | 41.641% | 75.614% |

[Three-seed cross-study aggregates](../results/diagnostics/cross_study_accuracy_conditions.csv), `matched_sparse_training` family.

### 4.3 Training evidence is compatible with generalization problems

Final-epoch means over three seeds are:

| Dataset / model | Input | Online train accuracy | Online train loss | Held-out validation accuracy | Held-out validation loss |
|---|---|---:|---:|---:|---:|
| MNIST ViT | Raw | 99.9958% | 0.000228 | 98.2533% | 0.0983 |
| MNIST ViT | Gray 60 | 99.9927% | 0.000282 | 97.5000% | 0.1544 |
| MNIST Swin | Raw | 99.9903% | 0.000671 | 99.1200% | 0.0427 |
| MNIST Swin | Gray 60 | 99.9885% | 0.000668 | 98.8000% | 0.0547 |
| CIFAR-10 ViT | Raw | 99.9637% | 0.001237 | 77.7267% | 1.8436 |
| CIFAR-10 ViT | Gray 60 | 99.5933% | 0.012871 | 68.9733% | 2.2138 |
| CIFAR-10 Swin | Raw | 99.3667% | 0.019598 | 88.1400% | 0.5601 |
| CIFAR-10 Swin | Gray 60 | 95.3659% | 0.135633 | 80.2133% | 0.7827 |

[Final training records, full epoch budgets and source hashes](../results/diagnostics/selected_final_training_epochs.csv).

“Online train” means predictions collected in training mode while parameters change, with the training augmentation and stochastic depth enabled where configured. It is not an evaluation-mode pass over the training set using the final checkpoint. Its loss and accuracy therefore cannot be treated as exactly matched estimators of the validation generalization gap. Nevertheless, near-perfect online fitting with materially worse validation, especially for ViT, is consistent with a substantial generalization problem. It gives no basis for claiming the experiment simply stopped before learning its training data.

These are full-size ViT-Small and Swin-Tiny backbones, roughly 21.4 million and 27.5 million parameters, applied to 28×28 or 32×32 native images with 2×2 patches. They are trained from scratch. The protocol does not establish the best achievable MNIST or CIFAR-10 accuracy for these architectures; it compares representations under the specified shared recipe.

## 5. What is and is not inside the latency number

The primary GPU-resident scope includes integer-to-float conversion, frontend work, ranking, support discovery, frozen normalization, patch layout, compact packing, every transformer layer, and the final head. CUDA synchronization around the complete Python call makes CPU dispatch gaps and necessary host-device synchronization part of wall time. Calling it “GPU-resident” describes input location; it does not mean it measures only GPU kernel execution.

It excludes archive/disk decode, dataset construction, checkpoint load, CUDA context setup, warmups, normalization fitting, and CPU prediction return. Separate scopes add pinned-host transfer or loader/collation, model inference, argmax, and prediction transfer to CPU. The cached-input diagnostic explicitly excludes online preprocessing and must not replace the headline raw-input result. [Measurement implementation](../src/sparse_contrast/benchmark.py#L421).

The A6000 cohort fixes GPU model, CPU model, backend, precision, and PyTorch thread settings. Selected records show RTX A6000, AMD EPYC 7413 CPU, PyTorch 2.9.1+cu126, CUDA 12.6, cuDNN 9.10.2, four intra-op threads, and the explicit attention implementation. Session records retain physical GPU UUIDs and isolation checks. Matching a hardware class does not mean every seed ran on the same physical card at the same instant; clock/thermal/scheduling noise remains possible. Measurements from the earlier heterogeneous or incomplete cohort are not pooled here. [Selected session metadata](../results/diagnostics/selected_profiles/).

### 5.1 Preprocessing matters, but does not explain Swin's large regression

Selected **seed-0** median measurements illustrate this. These are 200-call medians from separate scopes, not single-call profiler durations:

| Dataset / architecture | Batch | Gray 60 online ms | Gray 60 cached-input backbone ms |
|---|---:|---:|---:|
| MNIST ViT | 1 | 10.878 | 10.414 |
| MNIST ViT | 8 | 11.534 | 11.076 |
| MNIST Swin | 1 | 26.015 | 25.322 |
| MNIST Swin | 8 | 67.410 | 67.027 |
| CIFAR-10 ViT | 1 | 10.924 | 10.506 |
| CIFAR-10 ViT | 8 | 11.585 | 11.123 |
| CIFAR-10 Swin | 1 | 35.055 | 34.145 |
| CIFAR-10 Swin | 8 | 46.852 | 46.273 |

[Exact selected cell summaries](../results/diagnostics/selection.json).

Scope differences are not an additive stage profiler. Separate warmups and allocator/clock variation can even make one nominally smaller scope slightly slower in some cells. The robust conclusion is that the compact Swin backbone remains expensive even with prepared inputs: removing its frontend cannot turn approximately 67 ms into the raw backbone's approximately 19 ms on MNIST batch 8.

The preprocessing implementation still creates a dense coefficient tensor, performs a full rank operation for nonzero imposed sparsity, discovers support, and normalizes/layouts the full native grid. Active patches are gathered **before projection**, but normalization and patch-layout work are not themselves sparse. Zero-valued dense arrays do not automatically cause PyTorch GEMMs to skip arithmetic.

## 6. ViT: fewer useful tokens, padded batches, and substantial fixed overhead

### 6.1 Actual executed sequence lengths

Compact ViT gathers only retained patches for the input projection, preserves their original position indices, and pads each batch to its largest retained count. It then prepends one class token per image. The 12 transformer blocks operate on that rectangular batch. Masked padding is semantically excluded, but QKV projections, output projections, MLPs, and attention arrays still have the batch-maximum shape. [Implementation](../src/sparse_contrast/models.py#L134).

The selected clean seed-0 panel gives:

| Dataset | Batch | Raw sequence length | Gray 60 active image patches, mean | Gray 60 executed sequence length, mean | Padding rows per image, mean |
|---|---:|---:|---:|---:|---:|
| MNIST | 1 | 197 | 75.395 | 76.395 | 0 |
| MNIST | 8 | 197 | 75.186 | 95.570 | 19.384 |
| CIFAR-10 | 1 | 257 | 179.840 | 180.840 | 0 |
| CIFAR-10 | 8 | 257 | 179.336 | 196.970 | 16.634 |

The class token is included in sequence length but excluded from “active image patches.” These are averages over the saved 200 timing-batch support records. The fixed panel need not have exactly the same distribution as the full training partition. [Support summary](../results/diagnostics/support_summary.csv).

Batch padding explains why the arithmetic reduction at batch 8 is smaller than one would predict from average retained image tokens alone. It does not explain every millisecond: compact gathering also performs dynamic shape discovery, indexing, scatter, and mask construction, while the same large weight matrices remain in use.

### 6.2 Most ViT multiply-add work here is linear in token count

For hidden width $D=384$, MLP width $4D$, and sequence length $L$ including the class token, the principal multiply-add count per block per image is

$$
\underbrace{4LD^2}_{QKV+\text{output projection}}
+\underbrace{8LD^2}_{\text{MLP}}
+\underbrace{2L^2D}_{QK^\top\text{ and }AV}.
$$

Here one multiply-add is one MAC, approximately two arithmetic FLOPs. This deliberately excludes LayerNorm, softmax, GELU, biases, indexing, memory traffic, stem, and head. It is an arithmetic proxy, not measured hardware FLOPs.

At dense MNIST length 197, the quadratic attention products are only about 7.9% of this count; at CIFAR-10 length 257, about 10.0%. Saying “attention is quadratic” therefore exaggerates the fraction of work that can disappear quadratically at these widths and resolutions. Token-dependent QKV and MLP work dominates the MAC total.

Using each saved batch's actual maximum length, rather than squaring a mean length, gives:

| Dataset | Batch | Raw block GMAC/image | Gray 60 block GMAC/image | Compact/raw arithmetic ratio |
|---|---:|---:|---:|---:|
| MNIST | 1 | 4.541 | 1.678 | 0.370 |
| MNIST | 8 | 4.541 | 2.114 | 0.466 |
| CIFAR-10 | 1 | 6.066 | 4.143 | 0.683 |
| CIFAR-10 | 8 | 6.066 | 4.540 | 0.749 |

[Reproducible arithmetic proxy](../results/diagnostics/arithmetic_proxy.csv), [derivation code](../results/diagnostics/reproduce_summary.py).

The model truly removes a considerable amount of arithmetic, especially on MNIST. Its failure to improve latency is therefore not evidence that it secretly runs the full dense token grid. It indicates that arithmetic reduction alone does not dominate the measured small-batch runtime of this implementation.

The approximately 10 ms raw-ViT latency changes relatively little between batches 1 and 8. Selected profiles show hundreds of CUDA runtime kernel-launch calls, with more calls after compaction. This is consistent with a substantial dispatch/synchronization component and small-operation inefficiency. It is not, by itself, a roofline or occupancy measurement. The relative contributions of memory bandwidth, kernel launch latency, GPU occupancy, and CPU scheduling have not been separately measured.

## 7. Swin: exact variable-length execution can be much slower than dense windows

### 7.1 This implementation does not pad compact attention windows

Swin compact execution preserves original positions and shifted-window restrictions. It sorts active tokens into original windows, computes each window's retained length, and batches together windows with exactly the same length. It computes QKV/MLP only for active rows. There is **no attention-token padding within those compact groups**; all saved compact `padding_rows` values are zero.

This is different from ViT's batch-maximum padding. Blaming Swin's slowdown on padded compact attention would contradict both source and telemetry. Patch merging does allocate the four original child feature slots for each active parent, with absent children represented as zero slots, which is required for the defined merge operation; it is not dense-window attention padding.

The expensive implementation choice is the number of small groups and how they are executed. Every Swin block constructs window IDs, sorts, computes counts and offsets, extracts distinct active lengths, transfers that length list to the host, and executes a Python loop over length groups. Each iteration discovers the matching windows, gathers their QKV/position/region tensors, runs attention with gathered relative bias and shift restrictions, then scatters output. The loop is per **distinct window length**, not per token, but can still be large. [Exact compact path](../src/sparse_contrast/models.py#L243).

PyTorch documents that CUDA `nonzero` requires host-device synchronization, and that `Tensor.tolist()` moves values to the CPU when needed. Both occur in this path. Dynamic support is scientifically necessary; repeatedly resolving it through these eager host-driven operations is an implementation cost, not a mathematical requirement of sparse attention. [PyTorch `nonzero`](https://docs.pytorch.org/docs/2.9/generated/torch.nonzero.html), [PyTorch `tolist`](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.tolist.html).

### 7.2 Batch 8 can create many more groups without much more useful work

The average sum of distinct length groups over all 12 blocks is:

| Dataset | Batch 1 | Batch 8 |
|---|---:|---:|
| MNIST gray 60 | 17.170 | 82.820 |
| CIFAR-10 gray 60 | 30.365 | 49.810 |

[Saved support summary](../results/diagnostics/support_summary.csv).

For the first profiled MNIST batch-8 call, per-block group counts are `[15,22,6,6,4,4,4,4,4,4,1,1]`, totaling 75. The first batch-1 call totals 15. Even when a stage fits in one window per image, images with different retained counts become different groups. The compact implementation therefore gives up some of the dense path's efficient batching across equal-size windows.

This explains an otherwise surprising observation: MNIST contains fewer active tokens than CIFAR-10, yet its compact Swin batch-8 latency is worse. The distribution of support across images and windows changes the number and sizes of launched operations; total active-token count is insufficient to predict runtime.

### 7.3 Direct profiler evidence

Each row below describes **one instrumented first call, seed 0**, not the headline 200-call median. “Launches” is the profiler's count of `cudaLaunchKernel` runtime calls; it is not a claim about an independently measured total GPU instruction count.

| Dataset | Batch | Raw launches | Compact launches | Compact `aten::nonzero` calls | Compact `cudaStreamSynchronize` calls | Raw profiled wall ms | Compact profiled wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 1 | 1,059 | 1,829 | 28 | 105 | 37.445 | 50.346 |
| MNIST | 8 | 1,119 | 5,101 | 88 | 165 | 39.054 | 130.544 |
| CIFAR-10 | 1 | 1,069 | 2,814 | 46 | 123 | 37.542 | 73.759 |
| CIFAR-10 | 8 | 1,115 | 3,758 | 61 | 138 | 38.464 | 95.369 |

[Profile summary and exact selected JSON paths](../results/diagnostics/profile_summary.csv).

For these first calls, the `nonzero` count is structurally consistent with one initial support gather, 12 block-level active-count selections, and one window selection for each length group. For MNIST batch 8, $1+12+75=88$. This connects an observed profile event count to the actual grouping algorithm, rather than speculating that “sparse overhead” exists somewhere.

In the MNIST batch-8 compact profile, the 12 transformer-block CPU-inclusive intervals sum to approximately 123.080 ms, and the nested attention intervals sum to 104.026 ms. The comparable raw values are 34.846 and 18.469 ms. The frontend, ranking, and support scopes are far smaller. The large expansion sits inside the backbone's execution of grouped attention.

These inclusive intervals overlap hierarchically. They must not be added to form a new end-to-end total, and CPU-inclusive time is not isolated GPU compute time. Profiling inflates these selected Swin calls by approximately 2.0–2.1× relative to their matched uninstrumented call. The primary 67.446 ms MNIST batch-8 result comes from uninstrumented timing, not the 130.544 ms profiled call. The profile is evidence about operation structure and relative locations of overhead, with explicitly limited quantitative timing attribution.

### 7.4 Patch merging quickly removes sparsity from the expensive later stages

An output parent exists if any of its four children exists. Therefore merging propagates support by logical OR. Distinct small regions often merge into nearly complete coarse grids. The native stage grids are 14², 7², 4², 2² for MNIST, and 16², 8², 4², 2² for CIFAR-10. Width doubles across stages: 96, 192, 384, 768. Block depths are 2, 2, 6, 2.

Selected gray-60 batch-8 mean active tokens per image are:

| Dataset | Stage 0 active/dense | Stage 1 active/dense | Stage 2 active/dense | Stage 3 active/dense |
|---|---:|---:|---:|---:|
| MNIST | 75.186 / 196 | 24.916 / 49 | 9.890 / 16 | 3.991 / 4 |
| CIFAR-10 | 179.336 / 256 | 58.975 / 64 | 15.951 / 16 | 4.000 / 4 |

[Support summary](../results/diagnostics/support_summary.csv).

Saving an early width-96 token does not save the same arithmetic as saving a width-768 token. In dense CIFAR-10 Swin, the product of token count and squared width is constant across the four stages; six of the twelve blocks sit in stage 2. Almost all tokens in that stage survive gray-60 pruning. In MNIST, stages 2 and 3 together account for roughly 72% of the dense linear-projection/MLP MACs, and the final stage is essentially fully occupied after pruning.

For a stage with width $D$, $K$ executed token rows and window lengths $n_w$, the block MAC proxy is

$$
12KD^2 + 2D\sum_w n_w^2.
$$

For dense execution, the attention sum uses the full executed window area, including masked pairs that are still computed. For compact execution it uses each retained window's actual length. A merge adds $8D^2$ MACs per active parent for the $4D\to2D$ projection. Applying these formulas to every saved support record gives:

| Dataset | Batch | Raw block+merge GMAC/image | Gray 60 block+merge GMAC/image | Compact/raw arithmetic ratio |
|---|---:|---:|---:|---:|
| MNIST | 1 | 0.3330 | 0.2118 | 0.636 |
| MNIST | 8 | 0.3330 | 0.2114 | 0.635 |
| CIFAR-10 | 1 | 0.3575 | 0.3344 | 0.935 |
| CIFAR-10 | 8 | 0.3575 | 0.3343 | 0.935 |

These exclude the same kinds of non-matmul work as the ViT calculation, plus stem and final head. They are not profiler-measured FLOPs. **CIFAR-10 gray-60 Swin saves only about 6.5% of this transformer-plus-merge arithmetic proxy**, despite zeroing roughly 60% of coefficients. It simultaneously performs thousands of additional small runtime launches in the selected profile. There is little room for the arithmetic saving to pay back those costs. [Arithmetic proxy and executable derivation](../results/diagnostics/reproduce_summary.py).

## 8. The same-checkpoint dense-masked control identifies an execution problem

The raw-versus-contrast comparison changes both the model's learned function and the input representation. A sharper systems comparison takes the compact-trained gray-60 checkpoint and runs its **same support semantics** through dense-masked execution. Missing keys remain excluded and missing residual rows are zeroed, but the physical operations use dense grid/window shapes.

| Dataset / architecture | Batch | Compact ms | Same-checkpoint dense-masked ms |
|---|---:|---:|---:|
| MNIST ViT | 1 | 10.915 | 10.514 |
| MNIST ViT | 8 | 11.561 | 10.944 |
| MNIST Swin | 1 | 26.047 | 18.521 |
| MNIST Swin | 8 | 67.446 | 19.031 |
| CIFAR-10 ViT | 1 | 10.888 | 10.675 |
| CIFAR-10 ViT | 8 | 11.567 | 12.026 |
| CIFAR-10 Swin | 1 | 34.919 | 18.618 |
| CIFAR-10 Swin | 8 | 46.826 | 19.033 |

All entries are the same three-seed mean-of-medians convention and hardware class. [Exact paired control rows](../results/diagnostics/selected_execution_controls.csv) preserve all 48 seed/execution rows; [extraction provenance](../results/diagnostics/selected_execution_controls.provenance.json) records the original timing-table hash and verifies that each compact/dense-masked pair uses the same checkpoint and normalization.

Swin computes more arithmetic with dense-masked execution and nevertheless runs much faster. This is direct evidence that the eager compact execution strategy is the bottleneck under the measured conditions. It does not prove that physical sparse execution must always lose. CIFAR-10 ViT batch 8 already provides a modest opposite case: compact is faster than its dense-masked reference by about 3.8% in the ratio of means.

Dense-masked is not the original dense-trained model's unmasked inference. Confusing these controls would erase the distinction between a correct implementation comparison and a change of learned token semantics.

## 9. A break-even model for useful sparsity

Write the dense time as $T_d=T_{fixed}+T_{removable}$. Let $f=T_{removable}/T_d$, let $r$ be the remaining fraction of that removable time under an efficient sparse implementation, and let $H$ be new support/ranking/packing/dispatch overhead. Then

$$
\frac{T_s}{T_d}=(1-f)+rf+h,\qquad h=\frac{H}{T_d}.
$$

Sparsity pays off only if

$$
h < f(1-r).
$$

The imposed coefficient percentage is neither $f$ nor $1-r$. It must first translate into absent complete patches, stage-wise token savings, executed padded or grouped shapes, and efficient kernels. Larger token savings help only in the part of runtime that can actually be removed. Weight matrices do not become smaller when input tokens disappear; some memory traffic and all model parameters remain.

The arithmetic tables estimate one ingredient relevant to $r$, but cannot identify $f$ or $H$ from existing timings alone. A causal decomposition would require additional controlled variants. The current evidence supports the direction: MNIST ViT has large arithmetic savings but a considerable runtime floor; CIFAR-10 Swin has small arithmetic savings; MNIST Swin additionally incurs a severe increase in length-group dispatch at batch 8. It does not justify assigning an invented exact percentage of runtime to “Python,” “memory-bound kernels,” or “GPU compute.”

## 10. Fairness, limitations, and checks against plausible bugs

### 10.1 The training comparison is controlled within each group

The rerun uses a uniform physical batch within each dataset/architecture group, selected from actual full-model capacity/throughput characterization across its representations. There is no gradient accumulation. The resolved recipes are:

| Dataset / architecture | Train batch | Initial learning rate | Epochs | Warmup epochs |
|---|---:|---:|---:|---:|
| MNIST ViT | 256 | 0.0004242641 | 50 | 5 |
| MNIST Swin | 1,024 | 0.0008485281 | 50 | 5 |
| CIFAR-10 ViT | 128 | 0.0003000000 | 200 | 10 |
| CIFAR-10 Swin | 512 | 0.0006000000 | 200 | 10 |

The learning-rate rule is square-root scaling from 0.0003 at batch 128. All use AdamW with betas (0.9, 0.999), weight decay 0.05, cosine decay, minimum learning rate 1e-6, gradient clipping at 1, stochastic depth 0.1, no label smoothing, and FP32. CIFAR-10 uses the specified crop/flip augmentation; MNIST has no corresponding augmentation. The primary checkpoint is the final epoch, not the best test result or an opportunistically selected validation checkpoint. [Resolved selected configs](../results/diagnostics/selected_training_configs.json), [full protocol](experiment_protocol.md).

This controls the representation comparison **within** a dataset/architecture group. Different architectures use different physical batches and update counts, so cross-architecture accuracy cannot be attributed solely to attention architecture. A recipe that is shared fairly is not automatically the best recipe for each representation. Independent representation-specific tuning would answer a different question and require a fresh validation-only selection protocol.

The raw branch uses its prescribed canonical channel normalization. Contrast uses measured clean-training, unaugmented, 0%-sparsity statistics, frozen across sparsities and corruptions. All eight retained contrast-channel fits have unguarded standard deviations above the guard threshold. There is no observed collapsed normalization channel in the retained MNIST grayscale study. [Normalization evidence](../results/diagnostics/normalization.csv).

Freezing statistics is intentional. After pruning, variance and distribution can change, so the standardized tensor need not remain mean-zero or unit-variance. Refitting per corruption or per test sparsity would change the protocol and could leak test-distribution information. The current experiment does not isolate representation choice from every possible normalization choice.

### 10.2 Source/evidence audit

| Possible explanation | What the checked implementation/evidence supports |
|---|---|
| “The compact model actually projects every patch.” | False for compact input projection: retained patches are gathered first. ViT subsequently pads to the batch maximum; Swin uses exact window-length groups. |
| “Zeros in a dense tensor automatically avoid GEMMs.” | False. Dense and dense-masked paths execute their full configured shapes. |
| “Swin's compact slowdown is attention padding.” | Contradicted by zero compact padding rows and exact length grouping. Many small groups and later-stage densification are observed instead. |
| “The support mask was computed after normalization.” | False. Support uses exact native pre-normalization zeros, and epsilon is zero. |
| “OOD support was copied from the clean image.” | False. The frontend and support are recomputed from the actual corrupted input. |
| “Removing natural zeros should preserve a dense model.” | False in general because normalized values, biases, positions, attention denominators, pooling, and merges make those tokens non-neutral. |
| “Dense-to-compact loss proves a wrong checkpoint was loaded.” | Unsupported. The adapter preserves state layout and loads strictly; the p0 intervention changes token semantics even with the correct weights. |
| “The frontend destroys all DC/color information even without pruning.” | Too strong for the actual zero-padded finite operator and invertible opponent mixing. Grayscale projection and hard pruning are lossy; low-frequency attenuation/conditioning still matters. |
| “Every recorded stage time can be added to reproduce latency.” | False. Profile intervals are inclusive and nested, and profiling itself changes execution time. |
| “Reported latency includes only device arithmetic.” | False. Complete synchronized call timing includes CPU dispatch and required synchronizations. |
| “Full-model accuracy was estimated from a short diagnostic run.” | False for the results used here: final records reach the prescribed 50/200 epochs and the official accuracy cells are separately evaluated. |

Architecture tests compare all-active execution against the pinned dense references, and compact against dense-masked logits and parameter gradients, including empty support, position handling, shifted boundaries, and odd-grid merges. These checks support the implemented semantics; they do not establish that a model trained with dense all-position semantics tolerates deleting positions at inference.

The follow-up also exposed a genuine **numerical validation issue**, which is distinct from an incorrect scientific function. A strict FP32 compact-versus-dense-masked guard rejected a shape-dependent accumulation difference of about 3.55e-5 in one of 80 logits. A separate double-precision comparison agreed to about 4.62e-14. The documented recovery uses an untimed FP64 semantic witness plus bounded FP32-reference checks, restores RNG/allocation state, and leaves the measured FP32 path unchanged. Recovered timing workers must still complete before their evidence is counted. This is not a reason to relax away a large accuracy collapse or replace model outputs. [Frozen validation-recovery source](../provenance/sources/3c965bf1a2c9f3f11bb14c0d6ad3354a2266e7d0763b2146b5a8fbc69d314769/).

### 10.3 Real implementation costs still worth correcting in future versions

The measured code is a correct reference implementation with expensive eager operations, not a claim of a production-optimal sparse kernel. Both dense and compact attention explicitly compute scores, softmax, and value multiplication rather than using a fused SDPA path. Dense Swin also repeats some fixed geometry/indexing work. Comparing to this dense reference does not prove competitiveness with an optimized vendor/library dense implementation.

Detailed frontend diagnostics are disabled inside primary timing and rerun afterward. However, backbone telemetry assignments still execute on ordinary forwards: support counts, window-count summaries, and other tensor reductions are part of the measured path. Some counts are necessary for execution; logging-only reductions are not. It would be incorrect to claim that all telemetry overhead is absent from headline timing. A future version should separate execution-critical metadata from optional reporting while preserving exact outputs and verify both paths on the same protocol.

No new optimized backend is substituted into these published numbers. Any such change needs a separately identified measurement cohort, matched dense controls, and regression checks for logits, support, relative positions, shifted restrictions, all-empty inputs, and training gradients where applicable.

## 11. What to test or optimize next, and what each experiment would answer

These are proposed next experiments, not explanations already established by the results.

1. **Separate physical compaction from representation quality.** Retain the existing raw, dense-contrast, dense-masked, and compact controls. For the same compact-trained checkpoint, compare current compact execution to optimized compact and dense-masked implementations. This isolates systems improvement without changing its learned function. Separately compare representations at matched execution semantics to study information and generalization.

2. **Remove repeated host-driven window discovery.** Window support remains constant across the blocks of a Swin stage; only the shifted/unshifted layout changes. Precompute the two grouping plans once per stage and reuse them where valid, including original positions and shift regions. Replace per-length `nonzero` and host loops with a GPU-resident segmented/grouped kernel. Validate exact masks and gradients. This targets the directly observed synchronization and group-launch structure.

3. **Consider bounded padding or occupancy buckets as a systems tradeoff.** Exact length grouping minimizes arithmetic but can maximize dispatch fragmentation. A small number of padded buckets might intentionally do more arithmetic and run faster. The current no-padding result shows why “fewest MACs” is not automatically the best design. Such padding must mask absent keys and residual rows correctly, preserve sparse semantics, and report executed padding honestly.

4. **Use a dense-masked fallback when compaction cannot repay its cost.** The current same-checkpoint Swin comparison already establishes that this fallback is faster for the selected conditions. A dynamic hybrid could use dense-masked later stages once occupancy becomes high, while retaining compact early stages. A policy should be fixed using separate performance characterization, not chosen from per-example prediction correctness or favorable test timings. Kernel/backend changes still require new measurements.

5. **Evaluate fused attention and fused preprocessing deliberately.** PyTorch SDPA can select more efficient eligible implementations, but mask formats, relative-position bias, finite shifted-window penalties, dtype, and all-empty cases must remain correct. A kernel that handles packed variable-length segments may be necessary to realize savings. Fusing conversion, blur/color mixing, normalization, support construction and packing could reduce small allocations/launches; it must preserve the specified coefficient values, stable ranking and exact-zero decisions. [PyTorch SDPA documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html).

6. **Do not replace stable sparsification with an approximate threshold silently.** Full sorting is expensive, especially when natural zeros already exceed the requested rank. A faster exact selection algorithm could help. Tie behavior matters: changing which equal-magnitude coefficients are removed can change support and outputs. A shortcut that skips sorting must establish when it preserves the original result, including floating-point and corruption cases.

7. **Measure the actual execution regime.** Use isolated kernel/CPU timelines, occupancy and bandwidth evidence, then vary batch size and image/token count without conflating scopes. Determine the break-even curve for useful support distributions, not just uniform random sparsity. Include the grouping-length histogram, number of launches, per-stage occupancy and ViT padding. Existing B1/B8 evidence does not establish behavior at large inference batches or high-resolution images.

8. **Train for the inference graph when test-time deletion is the objective.** The MNIST dense-0→compact-0 failure makes this necessary to investigate. Compare compact-support training, support-pattern augmentation, gradual sparsification, or distillation from the dense model, using a fresh preregistered validation policy. These change the training method and cannot be presented as mere execution optimizations. Compact-0 training already prevents the particular clean natural-zero mismatch in the current data, without guaranteeing robustness after new pruning.

9. **Test whether discarded weak coefficients are task-relevant.** Compare scalar magnitude pruning with patch-structured pruning or a learned importance rule, retaining proper baselines and computational accounting. Patch-level pruning can produce more absent tokens for the same number of deleted scalar entries, but may remove more important spatial evidence. It is a new method, not a free efficiency improvement to this one.

10. **Separate recipe adequacy from representation limitations.** Tune regularization, augmentation, learning rate and normalization on the held-out training-validation partition for a deliberately bounded comparison. Keep test and corruption sets out of model selection. Report both shared-recipe and tuned-recipe results. The current near-perfect training fit and modest generalization justify this question; they do not identify the winning hyperparameters.

Every proposed optimization should retain the negative results from the original implementation. A faster revised kernel would improve the systems implementation, while a new support-aware training recipe would change the learned method. Both are legitimate research directions, but neither retroactively changes the meaning of the existing accuracy or latency measurements.

## 12. Evidence map and reproduction

The source frozen for the main A6000 measurement cohort is `7ea1b6e8…`; the training lineage is `9359e4f1…`. The inference-study source is `8cf4d0c1…`, with the separately recorded numerical-validation recovery `3c965bf1…`. Public source copies are under [src](../src/) and complete frozen trees under [provenance/sources](../provenance/sources/). The source/config identity is part of each saved result; a live repository edit is not silently treated as the code of an existing run.

| Claim or analysis | Public evidence |
|---|---|
| Primary clean accuracy and mCE, all conditions/seeds | [Conditions](../results/training_sparse_testing_sparse/conditions.csv), [official cells](../results/training_sparse_testing_sparse/cells.csv) |
| Primary latency, full three-seed values | [Latency overview](../results/training_sparse_testing_sparse/latency_overview.csv) |
| Dense-0 and compact-0 checkpoint interventions | [Cross-study condition table](../results/diagnostics/cross_study_accuracy_conditions.csv), [seed table](../results/diagnostics/cross_study_accuracy_seeds.csv) |
| Full-partition natural zeros and empty-patch fractions | [Preprocessing CSVs](../results/diagnostics/preprocessing/) |
| Final online training and held-out validation metrics | [Selected final epochs](../results/diagnostics/selected_final_training_epochs.csv) |
| Actual executed rows, stage support, length groups | [Support summary](../results/diagnostics/support_summary.csv), selected `tokens.jsonl` files |
| CUDA launch/synchronization calls and nested profiles | [Profile summary](../results/diagnostics/profile_summary.csv), selected `profile.json` files |
| Arithmetic proxy computed from actual executed shapes | [MAC table](../results/diagnostics/arithmetic_proxy.csv) |
| Hardware, zero-input control, cell medians, panel membership | [Selection index](../results/diagnostics/selection.json), [selected evidence tree](../results/diagnostics/selected_profiles/) |
| Copy identities and original relative source locations | [SHA256 manifest](../results/diagnostics/source_sha256_manifest.json) |

The diagnostic archive deliberately omits large Chrome traces, raw timing streams and checkpoint tensors. It preserves the small exact summary/profile/support evidence needed for the calculations here. Some summaries retain hashes and original paths to omitted full-study artifacts; their presence is provenance, not a claim that the lightweight public subset can revalidate every original completion receipt without the full archive.

To regenerate the support, profile-count, and arithmetic tables from the selected public evidence, run:

```bash
python results/diagnostics/reproduce_summary.py
```

This uses only the Python standard library, reads saved evidence, and performs no model execution or timing measurement.
