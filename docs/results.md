# Results and evidence

These results answer two different questions: what happens when models are **trained at the same sparsity used for testing**, and what happens when an existing **dense-trained checkpoint is sparsified only at inference**. The latter is an exploratory follow-up on already inspected benchmarks. It must not be presented as a fresh held-out model-selection study.

The primary larger-batch study completed all **156 main training/evaluation runs**, representing **52 conditions × three seeds**, and all **552 isolated A6000 timing workers**. The follow-up completed all **612 accuracy cases and 35,352 official evaluation cells**. Follow-up timing remains incomplete. Its live controller reported 976 of 2,424 workers completed at 2026-10-08 12:29:29 UTC; the published comparison curves below deliberately retain the separately verified **11:24:55 UTC** measurement snapshot. A live completion counter does not retroactively add data to a saved plot.

## 1. The three requested PDFs

| Figure | Contents | Numerical companions |
|---|---|---|
| [MNIST error by corruption](../plots/MNIST_error_by_corruption.pdf) | ViT-Small and Swin-Tiny pages; clean plus all 15 fixed-severity MNIST-C corruptions | [Aggregate CSV](../plots/MNIST_error_by_corruption.csv), [seed CSV](../plots/requested_comparisons/MNIST_error_by_corruption_seeds.csv) |
| [CIFAR-10 error by corruption](../plots/CIFAR10_error_by_corruption.pdf) | ViT-Small and Swin-Tiny pages; clean plus all 15 CIFAR-10-C corruption means over severities 1–5 | [Aggregate CSV](../plots/CIFAR10_error_by_corruption.csv), [seed CSV](../plots/requested_comparisons/CIFAR10_error_by_corruption_seeds.csv) |
| [Latency versus sparsity](../plots/latency_vs_sparsity.pdf) | Batch-1 and batch-8 pages, each with both datasets and architectures | [Aggregate CSV](../plots/latency_vs_sparsity.csv), [seed CSV](../plots/latency_vs_sparsity_seeds.csv) |

The PDFs contain six pages in total. Black is the unmodified input baseline: grayscale for MNIST, RGB for CIFAR-10. Purple is color opponency, orange is single-color contrast, and blue is grayscale contrast. Solid lines use compact models trained and tested at the same sparsity, including all five levels 0/20/40/60/80%. Dashed lines use only **dense-trained** checkpoints tested with compact inference at 0/10/.../90%. The separately trained compact-at-zero follow-up family is not included in those dashed curves.

All error panels share a 0–100% vertical axis. Curves show the mean over seeds 0, 1, and 2, with one sample SD. CIFAR corruption severities are first averaged within each seed; severity variation is not confused with training-seed uncertainty. The [figure guide and verification receipts](../plots/requested_comparisons/README.md) record definitions, hashes, exported-page checks, and actual-PDF visual inspection. The final error and latency rendering jobs, 361101 and 361102, completed successfully. Rendering generated no new model evaluations or timing measurements.

## 2. Matched sparse training: grayscale at 60%

The following comparison fixes dataset, architecture, seed set, training budget, and evaluation protocol. “Gray 60%” means a grayscale-contrast model **trained from scratch at 60% imposed coefficient sparsity and tested with compact execution at the same level**. This is not a dense checkpoint pruned after training.

All rows use `heldout_val`: 55,000 training images for MNIST or 45,000 for CIFAR-10, with 5,000 validation images. Clean accuracy uses 10,000 official test images. `mCE_raw` is the unnormalized mean corruption error in percent; **lower is better**. Other entries below are accuracy percentages; **higher is better**. Values are mean ± sample SD over three seeds. CIFAR lighting-related scores average its five severities per seed.

| Dataset / model | Method | Clean accuracy ↑ | `mCE_raw` ↓ | Brightness accuracy ↑ | Contrast accuracy ↑ | Fog accuracy ↑ |
|---|---|---:|---:|---:|---:|---:|
| MNIST / ViT-Small | Grayscale input baseline | 98.300 ± 0.053 | 27.104 ± 0.167 | 44.023 ± 3.998 | N/A | 42.113 ± 4.293 |
| MNIST / ViT-Small | Gray 60% | 97.883 ± 0.117 | 37.118 ± 0.451 | 80.653 ± 6.873 | N/A | 37.280 ± 3.512 |
| MNIST / Swin-Tiny | Grayscale input baseline | 99.233 ± 0.015 | 24.035 ± 0.579 | 14.863 ± 3.567 | N/A | 41.177 ± 3.315 |
| MNIST / Swin-Tiny | Gray 60% | 99.037 ± 0.064 | 28.738 ± 2.215 | 37.797 ± 10.457 | N/A | 31.610 ± 10.667 |
| CIFAR-10 / ViT-Small | RGB baseline | 78.103 ± 0.702 | 35.560 ± 0.177 | 71.447 ± 0.439 | 42.957 ± 0.321 | 56.517 ± 0.931 |
| CIFAR-10 / ViT-Small | Gray 60% | 68.403 ± 0.491 | 54.314 ± 0.744 | 65.623 ± 0.750 | 36.205 ± 0.897 | 47.767 ± 1.205 |
| CIFAR-10 / Swin-Tiny | RGB baseline | 87.787 ± 0.148 | 31.020 ± 0.283 | 84.237 ± 0.177 | 59.641 ± 0.414 | 73.523 ± 0.363 |
| CIFAR-10 / Swin-Tiny | Gray 60% | 79.853 ± 0.385 | 44.355 ± 0.043 | 78.174 ± 0.068 | 53.937 ± 1.000 | 67.891 ± 0.460 |

MNIST-C has no official contrast corruption, so its entries are unavailable rather than zero. “Lighting” is not a separate synthetic score: brightness, contrast where available, and fog are reported individually.

The seed-paired changes, gray 60% minus the matched baseline, make the overall tradeoff explicit:

| Dataset / model | Clean-accuracy change, percentage points | `mCE_raw` change, percentage points |
|---|---:|---:|
| MNIST / ViT-Small | −0.417 ± 0.064 | +10.014 ± 0.594 |
| MNIST / Swin-Tiny | −0.197 ± 0.071 | +4.703 ± 2.162 |
| CIFAR-10 / ViT-Small | −9.700 ± 1.159 | +18.754 ± 0.917 |
| CIFAR-10 / Swin-Tiny | −7.933 ± 0.301 | +13.336 ± 0.298 |

Brightness is a meaningful exception on MNIST: its mean accuracy improves by 36.630 percentage points for ViT and 22.933 for Swin. That improvement does not extend to aggregate corruption robustness or clean accuracy in these four gray-60 comparisons. Fog performance is lower on average for all four groups, with substantial seed variation for MNIST. CIFAR gray-60 also lowers brightness and contrast accuracy.

Clean cross-entropy increases from 0.08947 to 0.12567 for MNIST ViT, 0.03581 to 0.04576 for MNIST Swin, 1.81380 to 2.31130 for CIFAR ViT, and 0.63321 to 0.84096 for CIFAR Swin. These are full-test-set sample-weighted losses, not epoch training losses. [Seed-level evidence](../plots/overview/accuracy_seeds.csv) and [condition summaries](../plots/overview/accuracy_conditions.csv) retain all representations and both follow-up training families, beyond this prespecified readable comparison.

## 3. Does gray 60% actually run faster?

This table uses clean **GPU-resident raw input through logits**, including the contrast frontend and compact execution. Latencies are means of the three seed medians, in milliseconds **per batch**. Each seed median contains 200 measured repetitions after 50 warmups. Speedup is the mean ± sample SD of the three paired baseline/method ratios; a value above 1 is faster.

| Dataset / model | Batch | Baseline ms | Gray 60% ms | Paired speedup | Faster seeds |
|---|---:|---:|---:|---:|---:|
| MNIST / ViT-Small | 1 | 10.088 | 10.915 | 0.924 ± 0.003× | 0/3 |
| MNIST / ViT-Small | 8 | 10.609 | 11.561 | 0.918 ± 0.009× | 0/3 |
| MNIST / Swin-Tiny | 1 | 18.195 | 26.047 | 0.699 ± 0.006× | 0/3 |
| MNIST / Swin-Tiny | 8 | 18.780 | 67.446 | 0.278 ± 0.003× | 0/3 |
| CIFAR-10 / ViT-Small | 1 | 10.186 | 10.888 | 0.935 ± 0.007× | 0/3 |
| CIFAR-10 / ViT-Small | 8 | 11.647 | 11.567 | 1.007 ± 0.017× | 2/3 |
| CIFAR-10 / Swin-Tiny | 1 | 18.306 | 34.919 | 0.524 ± 0.004× | 0/3 |
| CIFAR-10 / Swin-Tiny | 8 | 18.748 | 46.826 | 0.400 ± 0.005× | 0/3 |

The one mean ratio slightly above 1, CIFAR ViT at batch 8, is small relative to variation across the three paired measurements and lacks agreement in one seed. This does not establish a reliable general speed advantage. Swin compact execution is substantially slower in these comparisons. Removing mathematical input tokens does not guarantee reduced elapsed time in this implementation: gathering, coordinate/window bookkeeping, packing, padding, synchronization, and small-kernel dispatch still cost time. Their exact contributions require the separately recorded profiler evidence, not subtraction of unrelated timing scopes.

These measurements use one A6000/EPYC 7413 hardware class and the same panel identities. Physical GPUs and sessions can differ. The result is implementation- and hardware-specific; it is not a claim that sparse execution cannot become faster with another validated kernel implementation. The full [52-condition clean-latency table](../debugging/a6000_all_conditions_clean_latency_2026-10-05.csv) includes all three online scopes and the cached diagnostic, so the reader can inspect the complete comparison rather than a selected winning curve.

## 4. The strongest observed primary clean-latency setting

Ranking all **96 nonbaseline condition/batch comparisons** in the primary clean-latency table, the largest mean GPU-input-to-logits speedup is **CIFAR-10 ViT-Small, single-color contrast, trained and tested compact at 80%, batch 8**. This is a descriptive maximum across a sweep, not an independently confirmed optimum or a configuration selected for a new test-set claim.

| Scope | RGB baseline ms | Single-color 80% ms | Mean paired speedup |
|---|---:|---:|---:|
| GPU raw input → logits | 11.647 | 11.404 | 1.0214× |
| Host raw input → logits | 11.714 | 11.424 | 1.0254× |
| Loader → CPU prediction | 11.926 | 11.610 | 1.0272× |

For the GPU scope, the paired speedup sample SD is **0.0157×**, with all three seeds directionally faster; method latency is 11.404 ± 0.061 ms. Clean accuracy falls from **78.103 ± 0.702% to 65.973 ± 0.055%**, a paired change of −12.130 percentage points. `mCE_raw` rises from **35.560 ± 0.177% to 54.597 ± 0.157%**, a paired increase of 19.038 percentage points. The observed roughly 2% online latency gain therefore accompanies substantial accuracy and corruption-error losses.

Across all 96 nonbaseline condition/batch rows, seven have mean GPU speedup above 1 and only two are faster in all three seeds. For host-input timing those counts are six and four; for loader-to-CPU timing, six and three. These are descriptive counts, not multiple-comparison-corrected significance tests. Cached-input results must not replace the online claim because they omit preprocessing.

## 5. Dense-trained checkpoints sparsified only at inference

The dashed curves isolate a different intervention: retain the dense-trained weights and change execution/support at test time. The original same-checkpoint dense result must be shown beside compact inference at 0%; otherwise a natural-zero removal effect can be mistaken for the effect of a positive imposed sparsity.

The table below uses grayscale contrast and the **same dense-trained checkpoint family** in each dataset/architecture group. All percentages are three-seed mean ± sample SD. No row involves retraining or refitting normalization.

| Dataset / model | Execution / imposed sparsity | Clean accuracy ↑ | `mCE_raw` ↓ |
|---|---|---:|---:|
| MNIST / ViT-Small | Original dense / 0% | 97.660 ± 0.128 | 30.446 ± 0.986 |
| MNIST / ViT-Small | Compact / 0% | 51.420 ± 1.536 | 60.162 ± 1.716 |
| MNIST / ViT-Small | Compact / 60% | 51.227 ± 1.573 | 71.299 ± 0.814 |
| MNIST / Swin-Tiny | Original dense / 0% | 99.000 ± 0.053 | 22.787 ± 1.201 |
| MNIST / Swin-Tiny | Compact / 0% | 90.320 ± 2.915 | 36.028 ± 0.406 |
| MNIST / Swin-Tiny | Compact / 60% | 90.297 ± 2.949 | 43.213 ± 1.343 |
| CIFAR-10 / ViT-Small | Original dense / 0% | 76.340 ± 1.099 | 49.013 ± 1.017 |
| CIFAR-10 / ViT-Small | Compact / 0% | 76.340 ± 1.101 | 49.014 ± 1.016 |
| CIFAR-10 / ViT-Small | Compact / 60% | 54.267 ± 1.355 | 66.274 ± 0.527 |
| CIFAR-10 / Swin-Tiny | Original dense / 0% | 83.277 ± 0.119 | 43.449 ± 0.104 |
| CIFAR-10 / Swin-Tiny | Compact / 0% | 83.263 ± 0.111 | 43.456 ± 0.103 |
| CIFAR-10 / Swin-Tiny | Compact / 60% | 67.557 ± 0.320 | 57.577 ± 0.523 |

The MNIST drop already occurs at zero imposed pruning. Empty background patches in a dense model still pass through affine normalization, learned biases, position encodings, and attention. Removing those positions changes the computation even when no additional native coefficients were thresholded. The result is evidence against treating dense-to-compact switching as an invariant operation; it is not evidence that the same model lost 46 percentage points specifically because of “60% pruning.”

For CIFAR grayscale at 60%, dense-trained inference sparsification reaches 54.267% clean accuracy for ViT and 67.557% for Swin, compared with 68.403% and 79.853% when compact models were trained at that same sparsity. Training at the intended support regime therefore matters for this comparison. The compact-at-zero-trained follow-up family remains in the full tables but is not conflated with dense-trained dashed curves.

## 6. What the partial follow-up latency figure can establish

The published 11:24:55 UTC snapshot contains 961 sealed follow-up workers. The requested contrast-only comparison plot has all **80 original matched-sparsity points**, all **eight original raw baseline points**, and **62 of 160 planned dense-trained contrast points**, each with exactly three matching seeds. The other 98 dashed points remain absent.

| Follow-up contrast group | Available imposed levels in the published latency plot |
|---|---|
| MNIST / ViT-Small / grayscale | 0%, 10%, 20% |
| MNIST / Swin-Tiny / grayscale | 0%, 10%, 20%, 30% |
| CIFAR-10 / either architecture / all three contrast representations | 0%, 10%, 20%, 30% |

Both batch sizes are represented. No complete three-seed follow-up 60% point appears in this snapshot, so the figure cannot answer whether dense-trained 60% inference gives a latency gain. Missing tails are neither joined nor extrapolated. In particular, complete accuracy at 60–90% does not imply complete latency there.

The unchanged raw baseline was remeasured in the follow-up cohort. Its group means differed from the original baseline by −0.1544 to +0.0576 ms across the eight dataset/model/batch groups. All 24 matched seed comparisons are preserved in the [raw-reference comparison CSV](../plots/latency_vs_sparsity_raw_reference_comparison.csv). The plotted original baseline was not rescaled or pooled with those new measurements. Small percentage speed differences should be read in that session-variation context.

## 7. Evidence boundaries

The training recipe, population sizes, execution semantics, corruption definitions, timing scopes, and immutable source identities are detailed in [the protocol](experiment_protocol.md). The comparison PDFs were derived from complete saved accuracy tables and hashed timing receipts, with reproducible [error](../analysis/original_exporters/build_requested_error_plots_20261008.py) and [latency](../analysis/original_exporters/render_requested_latency_20261008.py) rendering scripts. Their provenance files distinguish the numerical snapshot from the later rendering time.

The original primary clean-latency extraction is explicitly labeled preliminary relative to the final campaign report, even though it covers all 52 conditions and all three seeds. It validates saved worker/clean-cell/hardware identities rather than rereading every checkpoint, prediction array, or raw timing row. The full final reporter performs broader raw-evidence checks. The published summaries do not replace those retained artifacts.

No inference timing, loss, or accuracy value was substituted with a theoretical FLOP estimate, a validation-loop proxy, or a fabricated missing point. Three seeds characterize this experiment's observed variability; they do not by themselves establish broad statistical significance or generalization to other models, larger images, accelerators, corruption distributions, or sparse-kernel implementations.
