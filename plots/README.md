# Requested comparison plots

The requested presentation consists of **three PDFs**, using Matplotlib/Seaborn with the existing talk/whitegrid/serif style and 300 dpi export. Each PDF has a companion CSV with its plotted values.

| PDF | Contents |
|---|---|
| [CIFAR-10 errors](CIFAR10_error_by_corruption.pdf) | One page per architecture; clean error and each corruption's error in separate subplots. |
| [MNIST errors](MNIST_error_by_corruption.pdf) | One page per architecture; clean error and each corruption's error in separate subplots. |
| [Latency](latency_vs_sparsity.pdf) | Both datasets and both architectures; batch 1 and batch 8 on separate pages. |

In every subplot, the representations are compared together. Colors remain fixed: **black** for the raw baseline, **purple** for color opponency, **orange** for single colour, and **blue** for grayscale contrast. MNIST includes only the raw grayscale baseline and grayscale contrast.

**Solid lines:** trained and tested at matched sparsity. **Dashed lines:** trained densely at 0% sparsity, then tested with sparse compact inference. The raw baseline is a horizontal reference. Dashed curves use the actual dense-trained checkpoints, not the separate compact-at-zero-trained family.

All classification plots report **error (%)**, where lower is better. CIFAR-10 corruption errors average all five severities within each seed, then summarize the three seeds. MNIST-C has one released severity per corruption. Shading shows one sample standard deviation across the three training seeds. Compact inference at imposed 0% can already remove naturally empty patches.

Latency is clean, online GPU-resident raw-input-to-logits time, including preprocessing, on the matched RTX A6000/AMD EPYC7413 hardware class. Values summarize three per-seed median times. Batch-8 milliseconds are per batch. Missing three-seed measurements stay missing; the current dashed timing curves cover only early sparsities and must not be extrapolated to higher sparsity.

[Exact selection, coverage and reproduction details](requested_comparisons/README.md).

## Project status

The original `training_sparse_testing_sparse` study is complete: 156 training/evaluation runs and 552 isolated timing workers verified. The `training_dense_testing_sparse` follow-up has all 612 clean/OOD cases and 35,352 official evaluation cells complete; its timing campaign is still running. The numerical-validation repair is deployed under controller361093, preserving the original timed computation.

The status and repair record is [debugging/scratchpad.md](../debugging/scratchpad.md); live timing counts are in the original cluster file `training_dense_testing_sparse/STATUS.json` (not part of the public snapshot). Final full follow-up latency plots remain dependent on outstanding measurements.

## Detailed report archives

The earlier report figures and diagnostic exports remain available for detailed inspection:

- [Original study report figures](training_sparse_testing_sparse/INDEX.md), including training/validation curves, per-corruption errors, direct timing, p95, support and profiler diagnostics.
- [Follow-up report figures](training_dense_testing_sparse/INDEX.md).
- [Fixed images, masks and normalization diagnostics](diagnostics/README.md).
- [Earlier per-representation overviews](overview/README.md).
- [Catalog of the earlier 1,143 verified PDFs](pdf_catalog.csv) and [its verification receipt](pdf_verification.json). This catalog predates the three corrected comparison PDFs; their own export receipts provide current verification.
