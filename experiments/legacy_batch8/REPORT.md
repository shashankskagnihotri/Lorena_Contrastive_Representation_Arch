# Sparse contrast classification benchmark

Both headline tables use protocol `heldout_val`: MNIST Train N=55,000 and CIFAR-10 Train N=45,000, each with 5,000 held-out validation images. Counts in completed-run metadata must match these exact partitions.

| Dataset | Architecture | Condition | Clean test accuracy (%) | Mean corruption error (%) |
|---|---|---|---|---|
| cifar10 | swin_tiny | Raw dense | not measured | not measured |
| cifar10 | swin_tiny | Color, dense, 0% | not measured | not measured |
| cifar10 | swin_tiny | Grayscale, dense, 0% | not measured | not measured |
| cifar10 | swin_tiny | Opponent, dense, 0% | not measured | not measured |
| cifar10 | swin_tiny | Color, compact, 0% | not measured | not measured |
| cifar10 | swin_tiny | Color, compact, 20% | not measured | not measured |
| cifar10 | swin_tiny | Color, compact, 40% | not measured | not measured |
| cifar10 | swin_tiny | Color, compact, 60% | not measured | not measured |
| cifar10 | swin_tiny | Color, compact, 80% | not measured | not measured |
| cifar10 | swin_tiny | Grayscale, compact, 0% | not measured | not measured |
| cifar10 | swin_tiny | Grayscale, compact, 20% | not measured | not measured |
| cifar10 | swin_tiny | Grayscale, compact, 40% | not measured | not measured |
| cifar10 | swin_tiny | Grayscale, compact, 60% | not measured | not measured |
| cifar10 | swin_tiny | Grayscale, compact, 80% | not measured | not measured |
| cifar10 | swin_tiny | Opponent, compact, 0% | not measured | not measured |
| cifar10 | swin_tiny | Opponent, compact, 20% | not measured | not measured |
| cifar10 | swin_tiny | Opponent, compact, 40% | not measured | not measured |
| cifar10 | swin_tiny | Opponent, compact, 60% | not measured | not measured |
| cifar10 | swin_tiny | Opponent, compact, 80% | not measured | not measured |
| cifar10 | vit_small | Raw dense | not measured | not measured |
| cifar10 | vit_small | Color, dense, 0% | not measured | not measured |
| cifar10 | vit_small | Grayscale, dense, 0% | not measured | not measured |
| cifar10 | vit_small | Opponent, dense, 0% | not measured | not measured |
| cifar10 | vit_small | Color, compact, 0% | not measured | not measured |
| cifar10 | vit_small | Color, compact, 20% | not measured | not measured |
| cifar10 | vit_small | Color, compact, 40% | not measured | not measured |
| cifar10 | vit_small | Color, compact, 60% | not measured | not measured |
| cifar10 | vit_small | Color, compact, 80% | not measured | not measured |
| cifar10 | vit_small | Grayscale, compact, 0% | not measured | not measured |
| cifar10 | vit_small | Grayscale, compact, 20% | not measured | not measured |
| cifar10 | vit_small | Grayscale, compact, 40% | not measured | not measured |
| cifar10 | vit_small | Grayscale, compact, 60% | not measured | not measured |
| cifar10 | vit_small | Grayscale, compact, 80% | not measured | not measured |
| cifar10 | vit_small | Opponent, compact, 0% | not measured | not measured |
| cifar10 | vit_small | Opponent, compact, 20% | not measured | not measured |
| cifar10 | vit_small | Opponent, compact, 40% | not measured | not measured |
| cifar10 | vit_small | Opponent, compact, 60% | not measured | not measured |
| cifar10 | vit_small | Opponent, compact, 80% | not measured | not measured |
| mnist | swin_tiny | Raw dense | not measured | not measured |
| mnist | swin_tiny | Grayscale, dense, 0% | not measured | not measured |
| mnist | swin_tiny | Grayscale, compact, 0% | not measured | not measured |
| mnist | swin_tiny | Grayscale, compact, 20% | not measured | not measured |
| mnist | swin_tiny | Grayscale, compact, 40% | not measured | not measured |
| mnist | swin_tiny | Grayscale, compact, 60% | not measured | not measured |
| mnist | swin_tiny | Grayscale, compact, 80% | not measured | not measured |
| mnist | vit_small | Raw dense | 98.04 (1/3 seeds) | 30.74 (1/3 seeds) |
| mnist | vit_small | Grayscale, dense, 0% | 97.21 (1/3 seeds) | 37.02 (1/3 seeds) |
| mnist | vit_small | Grayscale, compact, 0% | 97.84 +/- 0.05 (2/3 seeds) | 44.12 +/- 1.29 (2/3 seeds) |
| mnist | vit_small | Grayscale, compact, 20% | 97.81 (1/3 seeds) | 42.43 (1/3 seeds) |
| mnist | vit_small | Grayscale, compact, 40% | 97.84 +/- 0.05 (2/3 seeds) | 42.18 +/- 1.03 (2/3 seeds) |
| mnist | vit_small | Grayscale, compact, 60% | not measured | not measured |
| mnist | vit_small | Grayscale, compact, 80% | not measured | not measured |

Latency table training protocol: `heldout_val`; MNIST Train N=55,000; CIFAR-10 Train N=45,000.

| Dataset | Architecture | Batch | Scope | Input | Condition | Latency (ms/batch), mean seed median +/- sample SD | Seeds |
|---|---|---|---|---|---|---|---|
| mnist | Both | 1 | gpu_raw_to_logits | Clean and balanced corruption | All 7 configurations | not measured | 0/3 |
| cifar10 | Both | 1 | gpu_raw_to_logits | Clean and balanced corruption | All 19 configurations | not measured | 0/3 |
| mnist | Both | 1 | host_raw_to_logits | Clean and balanced corruption | All 7 configurations | not measured | 0/3 |
| cifar10 | Both | 1 | host_raw_to_logits | Clean and balanced corruption | All 19 configurations | not measured | 0/3 |
| mnist | Both | 8 | gpu_raw_to_logits | Clean and balanced corruption | All 7 configurations | not measured | 0/3 |
| cifar10 | Both | 8 | gpu_raw_to_logits | Clean and balanced corruption | All 19 configurations | not measured | 0/3 |
| mnist | Both | 8 | host_raw_to_logits | Clean and balanced corruption | All 7 configurations | not measured | 0/3 |
| cifar10 | Both | 8 | host_raw_to_logits | Clean and balanced corruption | All 19 configurations | not measured | 0/3 |

Primary campaign coverage: 25/156 final checkpoints completed; 7/156 fully evaluated. Manifest state: `frozen`. Missing runs, cells, and seed results remain in `outputs/tables/run_coverage.csv`; they are never removed from a denominator. Pilots and optional refits are excluded from these primary aggregates.

Clean evaluation requires all 10,000 official test images. Each MNIST-C checkpoint requires 15 fixed-severity cells and 150,000 predictions; each CIFAR-10-C checkpoint requires 75 cells and 750,000 predictions. The reported mean corruption error is the unnormalized macro-average across the 15 corruption means. CIFAR-10-C averages all five severities within each corruption first. Aggregate uncertainty is sample SD across independently trained seeds. Repeated batches and corruption cells are not training seeds.

The primary protocol uses 55,000 MNIST or 45,000 CIFAR-10 training examples and 5,000 genuinely held-out validation examples. Training curves distinguish online augmented training from unaugmented held-out validation. Primary results use the final epoch, not test-selected or validation-selected checkpoints.

Per-corruption and severity evidence: `outputs/tables/cells.csv`. Prediction identities, labels, and correctness are stored beside each run's evaluation JSON. Paired percentage-point changes against the matching raw seed are in `outputs/tables/paired_changes.csv`. A missing corruption/severity cell prevents a completed mean corruption result.

Latency records and stage evidence: `outputs/tables/timings.csv`, `outputs/tables/stages.csv`, and the originating run directories recorded there. Primary online scopes include the frontend, coefficient sorting when nonzero sparsity is requested, support, normalization, compaction, and the model. Host-input timing additionally includes H2D from pinned decoded input. Disk/decoding and CPU prediction return are excluded from those two scopes and reported only in the practical loader scope. Profiler intervals are shown separately and are never added into an invented total. Hardware types are never pooled. Within one matched hardware class, resumed cells may span physical GPUs or measurement sessions; per-cell device UUID, session ID, host, scheduler job, and immutable hardware-metadata hashes are retained in the timing/stage tables, with explicit session/device unions for balanced mixtures. Batch latency divided by batch size is an amortized cost, not a batch-one request measurement.

| Dataset | Representation | Channel | Mean | Measured std | Effective std | Guarded |
|---|---|---|---|---|---|---|
| cifar10 | color_opponency | 0 | 0.02046135 | 0.0836314 | 0.0836314 | False |
| cifar10 | color_opponency | 1 | -0.000114874 | 0.01046859 | 0.01046859 | False |
| cifar10 | color_opponency | 2 | -0.0004263698 | 0.008367761 | 0.008367761 | False |
| cifar10 | grayscale | 0 | 0.02046135 | 0.0836314 | 0.0836314 | False |
| cifar10 | single_color | 0 | 0.02077285 | 0.08556194 | 0.08556194 | False |
| cifar10 | single_color | 1 | 0.0210026 | 0.08547426 | 0.08547426 | False |
| cifar10 | single_color | 2 | 0.01960862 | 0.08363753 | 0.08363753 | False |
| mnist | grayscale | 0 | 8.502879e-06 | 0.1084623 | 0.1084623 | False |

Raw-input constants are the requested published dataset-specific recipes. Contrast means and population standard deviations are fitted at 0% sparsity to the permitted clean training partition only, then frozen for every sparsity, seed, architecture, and evaluation cell. Evaluation distributions never update those constants. The active dataset-specific representation scope is recorded in `experiment_scope.json`; withdrawn MNIST color encodings and their outputs are excluded. MNIST already has sparse raw backgrounds; absent a raw-empty-patch control, a MNIST speedup cannot be attributed uniquely to contrast. These are small-image ViT-Small and Swin-Tiny adaptations with native 28/32-pixel inputs and 2x2 patches, not ImageNet configuration reproductions or a state-of-the-art comparison.

Scientific figures and their CSV data are under `outputs/figures/`. Shared signed-input panels, fixed training-derived display scales, and exact NPZ tensors/masks/IDs are under `outputs/previews/`. Rendering never changes model tensors. Per-run raw/augmented prediction panels and released-corruption panels are under each run's `diagnostics/` directory.

TensorBoard event files: `outputs/tensorboard/`. Start with `tensorboard --logdir outputs/tensorboard --host 127.0.0.1 --port 6006` in the project environment; access the host through the site-approved tunnel. Study figures are mirrored under `outputs/tensorboard/study_summary/heldout_val/<dataset>/<architecture>/report/`. Per-run paths distinguish protocol, representation, compact/dense execution, sparsity, seed, and configuration hash.

[Training-only runtime estimate](outputs/runtime_estimate.json): 3681.4 reference GPU-hours for 156 main runs plus four pilots, projected from observed raw-pilot epochs on the recorded GPU classes. This is a provisional training reference, not a whole-campaign completion forecast; contrast-specific costs, checkpoint/setup, evaluation, latency benchmarking, and queue/resource availability are excluded.

**Contrast versus raw input and agreement across seeds.** Positive clean-accuracy changes and positive corruption-error reductions favor contrast. Ranges below include every fully observed contrast condition, including dense controls, with each dataset's active-condition denominator; no best condition is selected.

| Dataset / architecture | Clean gain, pp | Corruption-error reduction, pp |
|---|---|---|
| mnist / vit_small | Incomplete: 0/6 conditions with three paired seeds | Incomplete: 0/6 conditions with three paired seeds |
| mnist / swin_tiny | Incomplete: 0/6 conditions with three paired seeds | Incomplete: 0/6 conditions with three paired seeds |
| cifar10 / vit_small | Incomplete: 0/18 conditions with three paired seeds | Incomplete: 0/18 conditions with three paired seeds |
| cifar10 / swin_tiny | Incomplete: 0/18 conditions with three paired seeds | Incomplete: 0/18 conditions with three paired seeds |

**Effect of coefficient sparsity.** Endpoint changes below compare compact 80% against compact 0% within each seed; all five requested sparsity settings must be observed for all three seeds before a representation is summarized. Endpoint differences do not establish monotonicity.
The complete five-level, three-seed sparsity comparisons are not yet available.

**Tokens removed and Swin hierarchy.** The following are fractions of the original spatial grid, not FLOP or speedup claims. Corrupted-input values require every official corruption/severity cell and are averaged equally within each seed before seed averaging.
Complete three-seed clean/corrupted token-removal results are unavailable.
Swin stage-density evidence is incomplete, so the stage at which merging restores density cannot yet be identified.

**Measured online gains at batches 1 and 8.** Speedups below are raw median latency divided by method median latency, paired by seed, input group, batch, and matched hardware class. Ranges include all active contrast conditions for the dataset, separately for GPU-resident and pinned-host input.
GPU/host speedups at batches 1 and 8 are incomplete; no gain after online frontend overhead is established yet.

**Architecture differences and remaining uncertainty.** The separate ViT and Swin rows above preserve their observed clean/corruption effects, initial removal, hierarchical activity, and online latency. Where matched architecture/seed/cell evidence is missing, a general ranking is unsupported; even complete ranges can favor different methods for different conditions. Native coefficient zeros, removed image patches, executed padding, and latency are distinct measurements. All-three-seed sign agreement is descriptive evidence, not a significance test; differences within timing variability remain uncertain. MNIST grayscale results do not establish color robustness.

Generated figure index:
- [empty_patches_grayscale](outputs/figures/mnist/preprocessing/empty_patches_grayscale.png)
- [achieved_sparsity_grayscale](outputs/figures/mnist/preprocessing/achieved_sparsity_grayscale.png)
- [empty_patches_single_color](outputs/figures/cifar10/preprocessing/empty_patches_single_color.png)
- [achieved_sparsity_single_color](outputs/figures/cifar10/preprocessing/achieved_sparsity_single_color.png)
- [empty_patches_grayscale](outputs/figures/cifar10/preprocessing/empty_patches_grayscale.png)
- [achieved_sparsity_grayscale](outputs/figures/cifar10/preprocessing/achieved_sparsity_grayscale.png)
- [empty_patches_color_opponency](outputs/figures/cifar10/preprocessing/empty_patches_color_opponency.png)
- [achieved_sparsity_color_opponency](outputs/figures/cifar10/preprocessing/achieved_sparsity_color_opponency.png)
- [clean_accuracy](outputs/figures/mnist/vit_small/clean_accuracy.png)
- [mean_corruption_error](outputs/figures/mnist/vit_small/mean_corruption_error.png)
- [per_corruption_error](outputs/figures/mnist/vit_small/per_corruption_error.png)
- [learning_raw_dense_pNone_loss](outputs/figures/mnist/vit_small/learning_raw_dense_pNone_loss.png)
- [learning_raw_dense_pNone_accuracy](outputs/figures/mnist/vit_small/learning_raw_dense_pNone_accuracy.png)
- [learning_grayscale_dense_p0_loss](outputs/figures/mnist/vit_small/learning_grayscale_dense_p0_loss.png)
- [learning_grayscale_dense_p0_accuracy](outputs/figures/mnist/vit_small/learning_grayscale_dense_p0_accuracy.png)
- [learning_grayscale_compact_p0_loss](outputs/figures/mnist/vit_small/learning_grayscale_compact_p0_loss.png)
- [learning_grayscale_compact_p0_accuracy](outputs/figures/mnist/vit_small/learning_grayscale_compact_p0_accuracy.png)
- [learning_grayscale_compact_p20_loss](outputs/figures/mnist/vit_small/learning_grayscale_compact_p20_loss.png)
- [learning_grayscale_compact_p20_accuracy](outputs/figures/mnist/vit_small/learning_grayscale_compact_p20_accuracy.png)
- [learning_grayscale_compact_p40_loss](outputs/figures/mnist/vit_small/learning_grayscale_compact_p40_loss.png)
- [learning_grayscale_compact_p40_accuracy](outputs/figures/mnist/vit_small/learning_grayscale_compact_p40_accuracy.png)
- [learning_grayscale_compact_p60_loss](outputs/figures/mnist/vit_small/learning_grayscale_compact_p60_loss.png)
- [learning_grayscale_compact_p60_accuracy](outputs/figures/mnist/vit_small/learning_grayscale_compact_p60_accuracy.png)
- [learning_grayscale_compact_p80_loss](outputs/figures/mnist/vit_small/learning_grayscale_compact_p80_loss.png)
- [learning_grayscale_compact_p80_accuracy](outputs/figures/mnist/vit_small/learning_grayscale_compact_p80_accuracy.png)
- [support_achieved_zero_fraction_mean](outputs/figures/mnist/vit_small/support_achieved_zero_fraction_mean.png)
- [support_executed_image_tokens_mean](outputs/figures/mnist/vit_small/support_executed_image_tokens_mean.png)
- [support_executed_sequence_length_mean](outputs/figures/mnist/vit_small/support_executed_sequence_length_mean.png)
- [support_natural_zero_fraction_mean](outputs/figures/mnist/vit_small/support_natural_zero_fraction_mean.png)
- [support_padding_tokens_mean](outputs/figures/mnist/vit_small/support_padding_tokens_mean.png)
- [support_retained_patches_mean](outputs/figures/mnist/vit_small/support_retained_patches_mean.png)
- [support_retained_per_image_mean](outputs/figures/mnist/vit_small/support_retained_per_image_mean.png)
- [support_spatial_zero_fraction_mean](outputs/figures/mnist/vit_small/support_spatial_zero_fraction_mean.png)
- [learning_raw_dense_pNone_loss](outputs/figures/mnist/swin_tiny/learning_raw_dense_pNone_loss.png)
- [learning_raw_dense_pNone_accuracy](outputs/figures/mnist/swin_tiny/learning_raw_dense_pNone_accuracy.png)
- [learning_grayscale_dense_p0_loss](outputs/figures/mnist/swin_tiny/learning_grayscale_dense_p0_loss.png)
- [learning_grayscale_dense_p0_accuracy](outputs/figures/mnist/swin_tiny/learning_grayscale_dense_p0_accuracy.png)
- [learning_grayscale_compact_p0_loss](outputs/figures/mnist/swin_tiny/learning_grayscale_compact_p0_loss.png)
- [learning_grayscale_compact_p0_accuracy](outputs/figures/mnist/swin_tiny/learning_grayscale_compact_p0_accuracy.png)
- [learning_grayscale_compact_p20_loss](outputs/figures/mnist/swin_tiny/learning_grayscale_compact_p20_loss.png)
- [learning_grayscale_compact_p20_accuracy](outputs/figures/mnist/swin_tiny/learning_grayscale_compact_p20_accuracy.png)
- [learning_grayscale_compact_p40_loss](outputs/figures/mnist/swin_tiny/learning_grayscale_compact_p40_loss.png)
- [learning_grayscale_compact_p40_accuracy](outputs/figures/mnist/swin_tiny/learning_grayscale_compact_p40_accuracy.png)
- [learning_grayscale_compact_p60_loss](outputs/figures/mnist/swin_tiny/learning_grayscale_compact_p60_loss.png)
- [learning_grayscale_compact_p60_accuracy](outputs/figures/mnist/swin_tiny/learning_grayscale_compact_p60_accuracy.png)
- [learning_grayscale_compact_p80_loss](outputs/figures/mnist/swin_tiny/learning_grayscale_compact_p80_loss.png)
- [learning_grayscale_compact_p80_accuracy](outputs/figures/mnist/swin_tiny/learning_grayscale_compact_p80_accuracy.png)
- [learning_grayscale_dense_p0_loss](outputs/figures/cifar10/vit_small/learning_grayscale_dense_p0_loss.png)
- [learning_grayscale_dense_p0_accuracy](outputs/figures/cifar10/vit_small/learning_grayscale_dense_p0_accuracy.png)
- [learning_single_color_compact_p20_loss](outputs/figures/cifar10/vit_small/learning_single_color_compact_p20_loss.png)
- [learning_single_color_compact_p20_accuracy](outputs/figures/cifar10/vit_small/learning_single_color_compact_p20_accuracy.png)
- [learning_grayscale_compact_p0_loss](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p0_loss.png)
- [learning_grayscale_compact_p0_accuracy](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p0_accuracy.png)
- [learning_grayscale_compact_p80_loss](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p80_loss.png)
- [learning_grayscale_compact_p80_accuracy](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p80_accuracy.png)
- [learning_color_opponency_compact_p60_loss](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p60_loss.png)
- [learning_color_opponency_compact_p60_accuracy](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p60_accuracy.png)
- [learning_color_opponency_compact_p80_loss](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p80_loss.png)
- [learning_color_opponency_compact_p80_accuracy](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p80_accuracy.png)
