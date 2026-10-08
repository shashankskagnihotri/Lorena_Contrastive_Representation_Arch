# training_sparse_testing_sparse: Sparse contrast classification benchmark

Campaign: `sparse_contrast_benchmark_large_batch`. Evidence root: `/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch`. Only this campaign's manifest and run directories contribute results; prior batch-8 campaign results are not pooled into a larger-batch rerun.

Both headline tables use protocol `heldout_val`: MNIST Train N=55,000 and CIFAR-10 Train N=45,000, each with 5,000 held-out validation images. Counts in completed-run metadata must match these exact partitions.

| Dataset | Architecture | Condition | Clean test accuracy (%) | Mean corruption error (%) |
|---|---|---|---|---|
| cifar10 | swin_tiny | Raw dense | 87.79 +/- 0.15 (3/3 seeds) | 31.02 +/- 0.28 (3/3 seeds) |
| cifar10 | swin_tiny | Color, dense, 0% | 87.06 +/- 0.06 (3/3 seeds) | 39.15 +/- 0.12 (3/3 seeds) |
| cifar10 | swin_tiny | Grayscale, dense, 0% | 83.28 +/- 0.12 (3/3 seeds) | 43.45 +/- 0.10 (3/3 seeds) |
| cifar10 | swin_tiny | Opponent, dense, 0% | 86.88 +/- 0.10 (3/3 seeds) | 40.06 +/- 0.39 (3/3 seeds) |
| cifar10 | swin_tiny | Color, compact, 0% | 87.16 +/- 0.24 (3/3 seeds) | 39.14 +/- 0.26 (3/3 seeds) |
| cifar10 | swin_tiny | Color, compact, 20% | 86.80 +/- 0.16 (3/3 seeds) | 39.48 +/- 0.61 (3/3 seeds) |
| cifar10 | swin_tiny | Color, compact, 40% | 85.92 +/- 0.15 (3/3 seeds) | 40.02 +/- 0.36 (3/3 seeds) |
| cifar10 | swin_tiny | Color, compact, 60% | 83.38 +/- 0.06 (3/3 seeds) | 41.96 +/- 0.48 (3/3 seeds) |
| cifar10 | swin_tiny | Color, compact, 80% | 73.68 +/- 0.17 (3/3 seeds) | 49.06 +/- 0.11 (3/3 seeds) |
| cifar10 | swin_tiny | Grayscale, compact, 0% | 83.25 +/- 0.17 (3/3 seeds) | 43.52 +/- 0.23 (3/3 seeds) |
| cifar10 | swin_tiny | Grayscale, compact, 20% | 83.17 +/- 0.13 (3/3 seeds) | 43.66 +/- 0.11 (3/3 seeds) |
| cifar10 | swin_tiny | Grayscale, compact, 40% | 82.26 +/- 0.55 (3/3 seeds) | 43.80 +/- 0.46 (3/3 seeds) |
| cifar10 | swin_tiny | Grayscale, compact, 60% | 79.85 +/- 0.38 (3/3 seeds) | 44.36 +/- 0.04 (3/3 seeds) |
| cifar10 | swin_tiny | Grayscale, compact, 80% | 67.07 +/- 0.48 (3/3 seeds) | 53.94 +/- 0.23 (3/3 seeds) |
| cifar10 | swin_tiny | Opponent, compact, 0% | 86.76 +/- 0.25 (3/3 seeds) | 40.25 +/- 0.13 (3/3 seeds) |
| cifar10 | swin_tiny | Opponent, compact, 20% | 86.82 +/- 0.10 (3/3 seeds) | 39.91 +/- 0.14 (3/3 seeds) |
| cifar10 | swin_tiny | Opponent, compact, 40% | 86.43 +/- 0.18 (3/3 seeds) | 40.36 +/- 0.22 (3/3 seeds) |
| cifar10 | swin_tiny | Opponent, compact, 60% | 85.15 +/- 0.06 (3/3 seeds) | 41.64 +/- 0.34 (3/3 seeds) |
| cifar10 | swin_tiny | Opponent, compact, 80% | 82.85 +/- 0.14 (3/3 seeds) | 43.65 +/- 0.22 (3/3 seeds) |
| cifar10 | vit_small | Raw dense | 78.10 +/- 0.70 (3/3 seeds) | 35.56 +/- 0.18 (3/3 seeds) |
| cifar10 | vit_small | Color, dense, 0% | 74.08 +/- 1.44 (3/3 seeds) | 50.52 +/- 0.74 (3/3 seeds) |
| cifar10 | vit_small | Grayscale, dense, 0% | 76.34 +/- 1.10 (3/3 seeds) | 49.01 +/- 1.02 (3/3 seeds) |
| cifar10 | vit_small | Opponent, dense, 0% | 77.97 +/- 0.25 (3/3 seeds) | 48.68 +/- 0.44 (3/3 seeds) |
| cifar10 | vit_small | Color, compact, 0% | 76.16 +/- 0.84 (3/3 seeds) | 48.84 +/- 0.72 (3/3 seeds) |
| cifar10 | vit_small | Color, compact, 20% | 76.03 +/- 1.25 (3/3 seeds) | 49.13 +/- 0.56 (3/3 seeds) |
| cifar10 | vit_small | Color, compact, 40% | 75.08 +/- 1.88 (3/3 seeds) | 49.61 +/- 1.02 (3/3 seeds) |
| cifar10 | vit_small | Color, compact, 60% | 72.87 +/- 0.79 (3/3 seeds) | 50.97 +/- 1.11 (3/3 seeds) |
| cifar10 | vit_small | Color, compact, 80% | 65.97 +/- 0.06 (3/3 seeds) | 54.60 +/- 0.16 (3/3 seeds) |
| cifar10 | vit_small | Grayscale, compact, 0% | 76.90 +/- 0.55 (3/3 seeds) | 48.48 +/- 0.53 (3/3 seeds) |
| cifar10 | vit_small | Grayscale, compact, 20% | 77.08 +/- 1.11 (3/3 seeds) | 48.34 +/- 1.41 (3/3 seeds) |
| cifar10 | vit_small | Grayscale, compact, 40% | 74.98 +/- 2.34 (3/3 seeds) | 49.30 +/- 2.89 (3/3 seeds) |
| cifar10 | vit_small | Grayscale, compact, 60% | 68.40 +/- 0.49 (3/3 seeds) | 54.31 +/- 0.74 (3/3 seeds) |
| cifar10 | vit_small | Grayscale, compact, 80% | 56.81 +/- 0.55 (3/3 seeds) | 61.52 +/- 0.20 (3/3 seeds) |
| cifar10 | vit_small | Opponent, compact, 0% | 77.90 +/- 0.64 (3/3 seeds) | 48.16 +/- 0.65 (3/3 seeds) |
| cifar10 | vit_small | Opponent, compact, 20% | 77.95 +/- 0.49 (3/3 seeds) | 48.08 +/- 0.43 (3/3 seeds) |
| cifar10 | vit_small | Opponent, compact, 40% | 77.53 +/- 0.20 (3/3 seeds) | 48.40 +/- 0.33 (3/3 seeds) |
| cifar10 | vit_small | Opponent, compact, 60% | 76.11 +/- 0.29 (3/3 seeds) | 49.60 +/- 0.36 (3/3 seeds) |
| cifar10 | vit_small | Opponent, compact, 80% | 72.03 +/- 0.66 (3/3 seeds) | 51.63 +/- 0.70 (3/3 seeds) |
| mnist | swin_tiny | Raw dense | 99.23 +/- 0.02 (3/3 seeds) | 24.03 +/- 0.58 (3/3 seeds) |
| mnist | swin_tiny | Grayscale, dense, 0% | 99.00 +/- 0.05 (3/3 seeds) | 22.79 +/- 1.20 (3/3 seeds) |
| mnist | swin_tiny | Grayscale, compact, 0% | 98.98 +/- 0.16 (3/3 seeds) | 27.06 +/- 1.44 (3/3 seeds) |
| mnist | swin_tiny | Grayscale, compact, 20% | 98.95 +/- 0.10 (3/3 seeds) | 26.92 +/- 1.03 (3/3 seeds) |
| mnist | swin_tiny | Grayscale, compact, 40% | 98.93 +/- 0.06 (3/3 seeds) | 27.80 +/- 1.44 (3/3 seeds) |
| mnist | swin_tiny | Grayscale, compact, 60% | 99.04 +/- 0.06 (3/3 seeds) | 28.74 +/- 2.22 (3/3 seeds) |
| mnist | swin_tiny | Grayscale, compact, 80% | 98.85 +/- 0.05 (3/3 seeds) | 39.97 +/- 0.70 (3/3 seeds) |
| mnist | vit_small | Raw dense | 98.30 +/- 0.05 (3/3 seeds) | 27.10 +/- 0.17 (3/3 seeds) |
| mnist | vit_small | Grayscale, dense, 0% | 97.66 +/- 0.13 (3/3 seeds) | 30.45 +/- 0.99 (3/3 seeds) |
| mnist | vit_small | Grayscale, compact, 0% | 98.04 +/- 0.14 (3/3 seeds) | 38.99 +/- 0.91 (3/3 seeds) |
| mnist | vit_small | Grayscale, compact, 20% | 98.04 +/- 0.14 (3/3 seeds) | 38.25 +/- 0.72 (3/3 seeds) |
| mnist | vit_small | Grayscale, compact, 40% | 98.04 +/- 0.14 (3/3 seeds) | 37.83 +/- 0.80 (3/3 seeds) |
| mnist | vit_small | Grayscale, compact, 60% | 97.88 +/- 0.12 (3/3 seeds) | 37.12 +/- 0.45 (3/3 seeds) |
| mnist | vit_small | Grayscale, compact, 80% | 97.62 +/- 0.01 (3/3 seeds) | 45.58 +/- 0.60 (3/3 seeds) |

Latency table training protocol: `heldout_val`; MNIST Train N=55,000; CIFAR-10 Train N=45,000.

| Hardware ID | Dataset | Architecture | Batch | Scope | Input | Condition | Latency (ms/batch), mean seed median +/- sample SD | Seeds |
|---|---|---|---|---|---|---|---|---|
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | clean | Raw dense | 10.088 +/- 0.033 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | clean | Grayscale, dense, 0% | 10.260 +/- 0.014 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 0% | 10.770 +/- 0.011 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 20% | 10.883 +/- 0.043 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 40% | 10.907 +/- 0.045 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 60% | 10.915 +/- 0.044 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 80% | 10.898 +/- 0.047 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | clean | Raw dense | 10.139 +/- 0.019 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | clean | Grayscale, dense, 0% | 10.288 +/- 0.013 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | clean | Grayscale, compact, 0% | 10.788 +/- 0.022 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | clean | Grayscale, compact, 20% | 10.875 +/- 0.068 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | clean | Grayscale, compact, 40% | 10.920 +/- 0.052 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | clean | Grayscale, compact, 60% | 10.925 +/- 0.033 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | clean | Grayscale, compact, 80% | 10.934 +/- 0.039 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | clean | Raw dense | 10.609 +/- 0.085 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, dense, 0% | 10.845 +/- 0.222 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 0% | 11.330 +/- 0.148 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 20% | 11.443 +/- 0.039 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 40% | 11.466 +/- 0.029 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 60% | 11.561 +/- 0.037 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 80% | 11.425 +/- 0.044 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | clean | Raw dense | 10.646 +/- 0.089 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | clean | Grayscale, dense, 0% | 10.867 +/- 0.230 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 0% | 11.372 +/- 0.081 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 20% | 11.450 +/- 0.035 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 40% | 11.481 +/- 0.044 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 60% | 11.575 +/- 0.066 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 80% | 11.454 +/- 0.070 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Raw dense | 11.502 +/- 0.044 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 11.718 +/- 0.045 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 12.278 +/- 0.022 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 12.460 +/- 0.074 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 12.462 +/- 0.067 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 12.484 +/- 0.032 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 12.490 +/- 0.048 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | balanced_corruption | Raw dense | 11.539 +/- 0.045 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 11.748 +/- 0.044 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 12.308 +/- 0.020 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 12.489 +/- 0.075 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 12.491 +/- 0.064 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 12.515 +/- 0.034 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 12.520 +/- 0.048 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Raw dense | 12.187 +/- 0.106 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 12.454 +/- 0.248 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 13.068 +/- 0.137 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 13.176 +/- 0.052 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 13.240 +/- 0.013 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 13.278 +/- 0.069 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 13.139 +/- 0.051 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | balanced_corruption | Raw dense | 12.211 +/- 0.105 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 12.486 +/- 0.248 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 13.098 +/- 0.135 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 13.212 +/- 0.053 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 13.271 +/- 0.005 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 13.317 +/- 0.068 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 13.173 +/- 0.055 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | clean | Raw dense | 18.195 +/- 0.047 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | clean | Grayscale, dense, 0% | 18.498 +/- 0.140 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 0% | 25.819 +/- 0.078 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 20% | 26.201 +/- 0.113 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 40% | 26.290 +/- 0.290 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 60% | 26.047 +/- 0.271 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 80% | 26.100 +/- 0.176 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | clean | Raw dense | 18.267 +/- 0.111 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | clean | Grayscale, dense, 0% | 18.527 +/- 0.131 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | clean | Grayscale, compact, 0% | 25.837 +/- 0.050 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | clean | Grayscale, compact, 20% | 26.179 +/- 0.093 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | clean | Grayscale, compact, 40% | 26.311 +/- 0.272 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | clean | Grayscale, compact, 60% | 26.101 +/- 0.234 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | clean | Grayscale, compact, 80% | 26.043 +/- 0.127 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | clean | Raw dense | 18.780 +/- 0.083 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | clean | Grayscale, dense, 0% | 18.985 +/- 0.028 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 0% | 68.051 +/- 0.413 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 20% | 68.290 +/- 0.542 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 40% | 68.075 +/- 0.417 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 60% | 67.446 +/- 0.503 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 80% | 63.254 +/- 0.378 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | clean | Raw dense | 18.808 +/- 0.078 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | clean | Grayscale, dense, 0% | 18.992 +/- 0.040 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | clean | Grayscale, compact, 0% | 67.910 +/- 0.447 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | clean | Grayscale, compact, 20% | 68.281 +/- 0.491 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | clean | Grayscale, compact, 40% | 68.176 +/- 0.570 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | clean | Grayscale, compact, 60% | 67.531 +/- 0.463 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | clean | Grayscale, compact, 80% | 63.451 +/- 0.241 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Raw dense | 21.839 +/- 0.083 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 22.150 +/- 0.172 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 31.533 +/- 0.180 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 32.054 +/- 0.228 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 32.470 +/- 0.129 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 32.195 +/- 0.183 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 31.744 +/- 0.192 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Raw dense | 21.878 +/- 0.085 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 22.186 +/- 0.167 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 31.548 +/- 0.166 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 32.118 +/- 0.236 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 32.504 +/- 0.129 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 32.220 +/- 0.218 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 31.771 +/- 0.191 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Raw dense | 22.479 +/- 0.054 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 22.625 +/- 0.084 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 81.307 +/- 0.244 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 81.751 +/- 0.141 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 80.981 +/- 0.601 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 79.100 +/- 0.243 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 75.762 +/- 0.323 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Raw dense | 22.505 +/- 0.058 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 22.660 +/- 0.089 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 81.338 +/- 0.309 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 81.728 +/- 0.096 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 80.980 +/- 0.641 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 79.188 +/- 0.216 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | mnist | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 75.717 +/- 0.336 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Raw dense | 10.186 +/- 0.065 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Color, dense, 0% | 10.348 +/- 0.068 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Grayscale, dense, 0% | 10.363 +/- 0.011 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Opponent, dense, 0% | 10.335 +/- 0.016 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Color, compact, 0% | 10.885 +/- 0.072 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Color, compact, 20% | 11.010 +/- 0.043 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Color, compact, 40% | 11.080 +/- 0.040 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Color, compact, 60% | 10.801 +/- 0.024 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Color, compact, 80% | 10.905 +/- 0.031 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 0% | 10.834 +/- 0.041 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 20% | 11.111 +/- 0.150 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 40% | 11.057 +/- 0.057 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 60% | 10.888 +/- 0.031 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 80% | 10.882 +/- 0.049 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Opponent, compact, 0% | 10.849 +/- 0.024 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Opponent, compact, 20% | 10.977 +/- 0.142 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Opponent, compact, 40% | 11.047 +/- 0.074 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Opponent, compact, 60% | 11.078 +/- 0.165 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | clean | Opponent, compact, 80% | 10.899 +/- 0.077 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Raw dense | 10.216 +/- 0.069 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Color, dense, 0% | 10.353 +/- 0.078 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Grayscale, dense, 0% | 10.392 +/- 0.006 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Opponent, dense, 0% | 10.379 +/- 0.020 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Color, compact, 0% | 10.912 +/- 0.077 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Color, compact, 20% | 11.089 +/- 0.044 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Color, compact, 40% | 11.126 +/- 0.043 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Color, compact, 60% | 10.844 +/- 0.025 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Color, compact, 80% | 10.931 +/- 0.028 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Grayscale, compact, 0% | 10.858 +/- 0.048 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Grayscale, compact, 20% | 11.165 +/- 0.204 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Grayscale, compact, 40% | 11.104 +/- 0.076 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Grayscale, compact, 60% | 10.913 +/- 0.026 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Grayscale, compact, 80% | 10.901 +/- 0.039 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Opponent, compact, 0% | 10.880 +/- 0.023 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Opponent, compact, 20% | 10.999 +/- 0.138 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Opponent, compact, 40% | 11.080 +/- 0.063 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Opponent, compact, 60% | 11.135 +/- 0.117 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | clean | Opponent, compact, 80% | 10.889 +/- 0.034 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Raw dense | 11.647 +/- 0.134 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Color, dense, 0% | 11.814 +/- 0.098 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, dense, 0% | 11.720 +/- 0.039 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Opponent, dense, 0% | 11.847 +/- 0.039 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Color, compact, 0% | 12.104 +/- 0.130 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Color, compact, 20% | 12.313 +/- 0.120 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Color, compact, 40% | 12.187 +/- 0.399 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Color, compact, 60% | 11.441 +/- 0.057 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Color, compact, 80% | 11.404 +/- 0.061 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 0% | 12.037 +/- 0.048 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 20% | 11.934 +/- 0.075 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 40% | 11.576 +/- 0.048 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 60% | 11.567 +/- 0.144 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 80% | 11.419 +/- 0.088 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Opponent, compact, 0% | 12.109 +/- 0.094 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Opponent, compact, 20% | 12.262 +/- 0.130 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Opponent, compact, 40% | 12.295 +/- 0.163 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Opponent, compact, 60% | 12.179 +/- 0.023 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Opponent, compact, 80% | 11.562 +/- 0.136 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Raw dense | 11.714 +/- 0.100 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Color, dense, 0% | 11.839 +/- 0.106 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Grayscale, dense, 0% | 11.776 +/- 0.028 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Opponent, dense, 0% | 11.890 +/- 0.006 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Color, compact, 0% | 12.200 +/- 0.124 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Color, compact, 20% | 12.370 +/- 0.068 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Color, compact, 40% | 12.227 +/- 0.376 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Color, compact, 60% | 11.435 +/- 0.074 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Color, compact, 80% | 11.424 +/- 0.067 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 0% | 12.125 +/- 0.068 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 20% | 11.991 +/- 0.073 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 40% | 11.617 +/- 0.067 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 60% | 11.596 +/- 0.167 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 80% | 11.427 +/- 0.092 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Opponent, compact, 0% | 12.169 +/- 0.071 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Opponent, compact, 20% | 12.303 +/- 0.087 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Opponent, compact, 40% | 12.347 +/- 0.141 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Opponent, compact, 60% | 12.276 +/- 0.015 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Opponent, compact, 80% | 11.588 +/- 0.149 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Raw dense | 11.590 +/- 0.056 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Color, dense, 0% | 11.785 +/- 0.067 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 11.765 +/- 0.068 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Opponent, dense, 0% | 11.730 +/- 0.007 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Color, compact, 0% | 12.433 +/- 0.112 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Color, compact, 20% | 12.599 +/- 0.004 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Color, compact, 40% | 12.667 +/- 0.079 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Color, compact, 60% | 12.385 +/- 0.065 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Color, compact, 80% | 12.479 +/- 0.074 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 12.376 +/- 0.048 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 12.698 +/- 0.145 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 12.672 +/- 0.076 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 12.481 +/- 0.046 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 12.445 +/- 0.047 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 0% | 12.406 +/- 0.059 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 20% | 12.583 +/- 0.103 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 40% | 12.651 +/- 0.068 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 60% | 12.683 +/- 0.062 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 80% | 12.450 +/- 0.040 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Raw dense | 11.629 +/- 0.060 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Color, dense, 0% | 11.813 +/- 0.072 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 11.797 +/- 0.069 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Opponent, dense, 0% | 11.761 +/- 0.007 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Color, compact, 0% | 12.469 +/- 0.111 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Color, compact, 20% | 12.641 +/- 0.010 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Color, compact, 40% | 12.702 +/- 0.083 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Color, compact, 60% | 12.424 +/- 0.065 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Color, compact, 80% | 12.517 +/- 0.073 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 12.409 +/- 0.050 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 12.731 +/- 0.142 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 12.702 +/- 0.079 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 12.517 +/- 0.044 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 12.479 +/- 0.045 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Opponent, compact, 0% | 12.444 +/- 0.058 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Opponent, compact, 20% | 12.619 +/- 0.103 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Opponent, compact, 40% | 12.691 +/- 0.066 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Opponent, compact, 60% | 12.718 +/- 0.061 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 1 | host_raw_to_logits | balanced_corruption | Opponent, compact, 80% | 12.481 +/- 0.037 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Raw dense | 12.354 +/- 0.099 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Color, dense, 0% | 12.532 +/- 0.137 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 12.366 +/- 0.063 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, dense, 0% | 12.599 +/- 0.158 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 0% | 13.053 +/- 0.053 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 20% | 13.245 +/- 0.170 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 40% | 13.863 +/- 0.506 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 60% | 13.231 +/- 0.109 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 80% | 13.128 +/- 0.053 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 13.041 +/- 0.035 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 13.277 +/- 0.087 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 13.321 +/- 0.043 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 13.325 +/- 0.174 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 13.162 +/- 0.128 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 0% | 13.014 +/- 0.037 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 20% | 13.297 +/- 0.116 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 40% | 13.215 +/- 0.082 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 60% | 13.122 +/- 0.003 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 80% | 13.339 +/- 0.189 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Raw dense | 12.396 +/- 0.096 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Color, dense, 0% | 12.572 +/- 0.140 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 12.408 +/- 0.064 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Opponent, dense, 0% | 12.642 +/- 0.159 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 0% | 13.089 +/- 0.053 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 20% | 13.276 +/- 0.169 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 40% | 13.899 +/- 0.506 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 60% | 13.264 +/- 0.116 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 80% | 13.170 +/- 0.054 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 13.078 +/- 0.035 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 13.311 +/- 0.086 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 13.360 +/- 0.042 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 13.359 +/- 0.169 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 13.195 +/- 0.127 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 0% | 13.048 +/- 0.035 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 20% | 13.328 +/- 0.114 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 40% | 13.252 +/- 0.088 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 60% | 13.156 +/- 0.004 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 80% | 13.388 +/- 0.175 | 3/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Raw dense | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Color, dense, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, dense, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Opponent, dense, 0% | 42.651 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Color, compact, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Color, compact, 20% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Color, compact, 40% | 44.448 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Color, compact, 60% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Color, compact, 80% | 44.589 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 0% | 43.652 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 20% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 40% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 60% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 80% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Opponent, compact, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Opponent, compact, 20% | 43.987 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Opponent, compact, 40% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Opponent, compact, 60% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | clean | Opponent, compact, 80% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Raw dense | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Color, dense, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Grayscale, dense, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Opponent, dense, 0% | 42.689 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Color, compact, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Color, compact, 20% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Color, compact, 40% | 44.515 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Color, compact, 60% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Color, compact, 80% | 44.202 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 0% | 43.690 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 20% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 40% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 60% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Grayscale, compact, 80% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Opponent, compact, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Opponent, compact, 20% | 44.065 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Opponent, compact, 40% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Opponent, compact, 60% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | clean | Opponent, compact, 80% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Raw dense | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Color, dense, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, dense, 0% | 48.043 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 20% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 40% | 50.236 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 60% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 80% | 50.459 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 49.535 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 20% | 50.118 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 40% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 60% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 80% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Raw dense | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Color, dense, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Opponent, dense, 0% | 48.116 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 20% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 40% | 50.323 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 60% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 80% | 50.533 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 49.583 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 0% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 20% | 50.219 | 1/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 40% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 60% | not measured | 0/3 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | cifar10 | vit_small | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 80% | not measured | 0/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Raw dense | 18.306 +/- 0.066 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Color, dense, 0% | 18.442 +/- 0.071 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Grayscale, dense, 0% | 18.450 +/- 0.087 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Opponent, dense, 0% | 18.381 +/- 0.158 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Color, compact, 0% | 23.027 +/- 0.203 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Color, compact, 20% | 24.293 +/- 0.044 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Color, compact, 40% | 28.478 +/- 0.103 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Color, compact, 60% | 33.915 +/- 0.211 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Color, compact, 80% | 35.359 +/- 0.129 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 0% | 23.081 +/- 0.099 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 20% | 25.829 +/- 0.170 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 40% | 30.143 +/- 0.178 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 60% | 34.919 +/- 0.136 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Grayscale, compact, 80% | 35.350 +/- 0.168 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Opponent, compact, 0% | 23.118 +/- 0.199 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Opponent, compact, 20% | 23.057 +/- 0.118 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Opponent, compact, 40% | 23.191 +/- 0.051 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Opponent, compact, 60% | 25.499 +/- 0.114 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | clean | Opponent, compact, 80% | 31.552 +/- 0.161 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Raw dense | 18.324 +/- 0.061 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Color, dense, 0% | 18.433 +/- 0.103 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Grayscale, dense, 0% | 18.524 +/- 0.081 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Opponent, dense, 0% | 18.392 +/- 0.163 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Color, compact, 0% | 23.058 +/- 0.202 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Color, compact, 20% | 24.352 +/- 0.111 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Color, compact, 40% | 28.480 +/- 0.375 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Color, compact, 60% | 33.890 +/- 0.168 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Color, compact, 80% | 35.301 +/- 0.177 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Grayscale, compact, 0% | 23.079 +/- 0.066 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Grayscale, compact, 20% | 25.823 +/- 0.141 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Grayscale, compact, 40% | 30.158 +/- 0.189 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Grayscale, compact, 60% | 34.664 +/- 0.119 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Grayscale, compact, 80% | 35.193 +/- 0.279 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Opponent, compact, 0% | 23.029 +/- 0.350 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Opponent, compact, 20% | 23.076 +/- 0.114 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Opponent, compact, 40% | 23.248 +/- 0.027 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Opponent, compact, 60% | 25.584 +/- 0.171 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | clean | Opponent, compact, 80% | 31.488 +/- 0.160 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Raw dense | 18.748 +/- 0.221 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Color, dense, 0% | 18.804 +/- 0.210 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Grayscale, dense, 0% | 18.740 +/- 0.108 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Opponent, dense, 0% | 18.767 +/- 0.055 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Color, compact, 0% | 23.379 +/- 0.126 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Color, compact, 20% | 28.871 +/- 0.093 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Color, compact, 40% | 38.075 +/- 0.438 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Color, compact, 60% | 45.382 +/- 0.304 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Color, compact, 80% | 55.002 +/- 0.178 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 0% | 23.476 +/- 0.127 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 20% | 31.503 +/- 0.113 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 40% | 40.237 +/- 0.284 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 60% | 46.826 +/- 0.051 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Grayscale, compact, 80% | 55.817 +/- 0.305 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Opponent, compact, 0% | 23.334 +/- 0.048 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Opponent, compact, 20% | 24.954 +/- 0.283 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Opponent, compact, 40% | 28.216 +/- 0.163 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Opponent, compact, 60% | 34.982 +/- 0.185 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | clean | Opponent, compact, 80% | 43.082 +/- 0.212 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Raw dense | 18.749 +/- 0.285 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Color, dense, 0% | 18.826 +/- 0.180 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Grayscale, dense, 0% | 18.757 +/- 0.107 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Opponent, dense, 0% | 18.780 +/- 0.050 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Color, compact, 0% | 23.405 +/- 0.081 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Color, compact, 20% | 28.754 +/- 0.120 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Color, compact, 40% | 38.142 +/- 0.198 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Color, compact, 60% | 45.356 +/- 0.357 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Color, compact, 80% | 54.992 +/- 0.285 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Grayscale, compact, 0% | 23.521 +/- 0.116 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Grayscale, compact, 20% | 31.601 +/- 0.197 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Grayscale, compact, 40% | 40.124 +/- 0.261 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Grayscale, compact, 60% | 46.665 +/- 0.198 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Grayscale, compact, 80% | 55.882 +/- 0.357 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Opponent, compact, 0% | 23.368 +/- 0.060 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Opponent, compact, 20% | 24.985 +/- 0.278 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Opponent, compact, 40% | 28.188 +/- 0.057 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Opponent, compact, 60% | 34.934 +/- 0.118 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | clean | Opponent, compact, 80% | 42.995 +/- 0.217 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Raw dense | 21.992 +/- 0.137 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Color, dense, 0% | 22.213 +/- 0.133 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 22.162 +/- 0.108 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Opponent, dense, 0% | 21.953 +/- 0.092 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Color, compact, 0% | 28.038 +/- 0.261 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Color, compact, 20% | 29.335 +/- 0.132 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Color, compact, 40% | 33.466 +/- 0.383 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Color, compact, 60% | 40.077 +/- 0.205 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Color, compact, 80% | 42.931 +/- 0.188 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 27.896 +/- 0.068 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 30.827 +/- 0.197 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 35.908 +/- 0.039 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 41.870 +/- 0.066 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 43.279 +/- 0.273 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 0% | 27.949 +/- 0.151 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 20% | 27.994 +/- 0.133 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 40% | 28.219 +/- 0.143 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 60% | 30.401 +/- 0.279 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 80% | 38.246 +/- 0.500 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Raw dense | 22.024 +/- 0.142 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Color, dense, 0% | 22.235 +/- 0.141 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 22.192 +/- 0.114 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Opponent, dense, 0% | 21.984 +/- 0.092 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Color, compact, 0% | 28.071 +/- 0.257 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Color, compact, 20% | 29.354 +/- 0.134 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Color, compact, 40% | 33.501 +/- 0.360 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Color, compact, 60% | 40.118 +/- 0.188 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Color, compact, 80% | 42.948 +/- 0.154 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 27.932 +/- 0.053 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 30.845 +/- 0.198 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 35.949 +/- 0.030 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 41.918 +/- 0.063 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 43.323 +/- 0.276 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Opponent, compact, 0% | 27.989 +/- 0.152 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Opponent, compact, 20% | 28.028 +/- 0.131 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Opponent, compact, 40% | 28.261 +/- 0.146 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Opponent, compact, 60% | 30.429 +/- 0.283 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 1 | host_raw_to_logits | balanced_corruption | Opponent, compact, 80% | 38.306 +/- 0.478 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Raw dense | 22.413 +/- 0.268 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Color, dense, 0% | 22.534 +/- 0.239 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 22.530 +/- 0.188 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, dense, 0% | 22.468 +/- 0.053 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 0% | 28.520 +/- 0.150 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 20% | 33.995 +/- 0.218 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 40% | 44.121 +/- 0.288 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 60% | 53.525 +/- 0.424 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Color, compact, 80% | 66.194 +/- 0.449 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 28.494 +/- 0.240 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 36.797 +/- 0.371 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 47.407 +/- 0.376 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 56.132 +/- 0.405 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 66.844 +/- 0.334 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 0% | 28.403 +/- 0.021 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 20% | 29.805 +/- 0.351 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 40% | 33.329 +/- 0.128 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 60% | 40.318 +/- 0.185 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | gpu_raw_to_logits | balanced_corruption | Opponent, compact, 80% | 51.479 +/- 0.229 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Raw dense | 22.432 +/- 0.269 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Color, dense, 0% | 22.563 +/- 0.247 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Grayscale, dense, 0% | 22.562 +/- 0.186 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Opponent, dense, 0% | 22.499 +/- 0.055 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 0% | 28.551 +/- 0.146 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 20% | 33.998 +/- 0.216 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 40% | 44.154 +/- 0.302 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 60% | 53.573 +/- 0.427 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Color, compact, 80% | 66.174 +/- 0.474 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 0% | 28.543 +/- 0.235 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 20% | 36.817 +/- 0.387 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 40% | 47.460 +/- 0.383 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 60% | 56.158 +/- 0.452 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Grayscale, compact, 80% | 66.851 +/- 0.351 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 0% | 28.441 +/- 0.025 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 20% | 29.834 +/- 0.318 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 40% | 33.359 +/- 0.129 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 60% | 40.360 +/- 0.183 | 3/3 |
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | cifar10 | swin_tiny | 8 | host_raw_to_logits | balanced_corruption | Opponent, compact, 80% | 51.545 +/- 0.227 | 3/3 |

Hardware identities below keep GPU/CPU cohorts separate. Legacy measurements are supplementary; each latency row retains its original seed coverage. The CPU model was not recorded in the legacy hardware contract.

| Hardware ID | GPU | CPU | Cohort |
|---|---|---|---|
| 4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687 | NVIDIA RTX A6000 | AMD EPYC 7413 24-Core Processor | a6000-2026-10-04 |
| 7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102 | NVIDIA RTX 5000 Ada Generation | not recorded in legacy contract | legacy (supplementary measurements) |

**Training and evaluation recipe provenance.** Batch sizes below come from saved run configurations where available; otherwise the table explicitly marks planned manifest values. The learning rate is the configured base rate before warmup/cosine scheduling. Evaluation batch is for full validation/test/corruption passes; isolated latency remains measured separately at batches 1 and 8.

| Dataset | Architecture | Train batch | Effective batch | Evaluation batch | Base LR | Evidence |
|---|---|---|---|---|---|---|
| mnist | vit_small | 256 | 256 | 256 | 0.000424264 | saved run configs (21 runs) |
| mnist | swin_tiny | 1024 | 1024 | 1024 | 0.000848528 | saved run configs (21 runs) |
| cifar10 | vit_small | 128 | 128 | 128 | 0.0003 | saved run configs (57 runs) |
| cifar10 | swin_tiny | 512 | 512 | 512 | 0.0006 | saved run configs (57 runs) |

Primary campaign coverage: 156/156 final checkpoints completed; 156/156 fully evaluated. Manifest state: `frozen`. Missing runs, cells, and seed results remain in `outputs/tables/run_coverage.csv`; they are never removed from a denominator. Pilots and optional refits are excluded from these primary aggregates.

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

[Training-only runtime estimate](outputs/runtime_estimate.json): 589.2 reference GPU-hours for 156 main runs plus four pilots, projected from observed raw-pilot epochs on the recorded GPU classes. This is a provisional training reference, not a whole-campaign completion forecast; contrast-specific costs, checkpoint/setup, evaluation, latency benchmarking, and queue/resource availability are excluded.

**Contrast versus raw input and agreement across seeds.** Positive clean-accuracy changes and positive corruption-error reductions favor contrast. Ranges below include every fully observed contrast condition, including dense controls, with each dataset's active-condition denominator; no best condition is selected.

| Dataset / architecture | Clean gain, pp | Corruption-error reduction, pp |
|---|---|---|
| mnist / vit_small | -0.68 to -0.26; 6/6 complete; all-three-seed positive 0, negative 6 | -18.48 to -3.34; 6/6 complete; all-three-seed positive 0, negative 6 |
| mnist / swin_tiny | -0.38 to -0.20; 6/6 complete; all-three-seed positive 0, negative 6 | -15.93 to +1.25; 6/6 complete; all-three-seed positive 0, negative 5 |
| cifar10 / vit_small | -21.30 to -0.13; 18/18 complete; all-three-seed positive 0, negative 12 | -25.96 to -12.52; 18/18 complete; all-three-seed positive 0, negative 18 |
| cifar10 / swin_tiny | -20.71 to -0.62; 18/18 complete; all-three-seed positive 0, negative 18 | -22.92 to -8.12; 18/18 complete; all-three-seed positive 0, negative 18 |

**Effect of coefficient sparsity.** Endpoint changes below compare compact 80% against compact 0% within each seed; all five requested sparsity settings must be observed for all three seeds before a representation is summarized. Endpoint differences do not establish monotonicity.

| Dataset / architecture / representation | Clean accuracy, 80%-0% | Mean corruption error, 80%-0% |
|---|---|---|
| mnist / vit_small / grayscale | -0.41 +/- 0.14 pp; signs -/-/- | +6.60 +/- 1.51 pp; signs +/+/+ |
| mnist / swin_tiny / grayscale | -0.13 +/- 0.11 pp; signs -/-/- | +12.90 +/- 2.12 pp; signs +/+/+ |
| cifar10 / vit_small / single_color | -10.19 +/- 0.80 pp; signs -/-/- | +5.76 +/- 0.74 pp; signs +/+/+ |
| cifar10 / vit_small / grayscale | -20.10 +/- 1.09 pp; signs -/-/- | +13.04 +/- 0.73 pp; signs +/+/+ |
| cifar10 / vit_small / color_opponency | -5.86 +/- 0.13 pp; signs -/-/- | +3.47 +/- 0.05 pp; signs +/+/+ |
| cifar10 / swin_tiny / single_color | -13.48 +/- 0.35 pp; signs -/-/- | +9.93 +/- 0.20 pp; signs +/+/+ |
| cifar10 / swin_tiny / grayscale | -16.18 +/- 0.64 pp; signs -/-/- | +10.41 +/- 0.38 pp; signs +/+/+ |
| cifar10 / swin_tiny / color_opponency | -3.91 +/- 0.35 pp; signs -/-/- | +3.40 +/- 0.26 pp; signs +/+/+ |

**Tokens removed and Swin hierarchy.** The following are fractions of the original spatial grid, not FLOP or speedup claims. Corrupted-input values require every official corruption/severity cell and are averaged equally within each seed before seed averaging.

| Dataset / architecture / input | Initial image patches removed | Three-seed conditions |
|---|---|---|
| mnist / vit_small / clean | 61.89 to 69.03% | 5/5 |
| mnist / vit_small / corrupted | 41.46 to 65.25% | 5/5 |
| mnist / swin_tiny / clean | 61.89 to 69.03% | 5/5 |
| mnist / swin_tiny / corrupted | 41.46 to 65.25% | 5/5 |
| cifar10 / vit_small / clean | 0.04 to 56.82% | 15/15 |
| cifar10 / vit_small / corrupted | 0.02 to 56.02% | 15/15 |
| cifar10 / swin_tiny / clean | 0.04 to 56.82% | 15/15 |
| cifar10 / swin_tiny / corrupted | 0.02 to 56.02% | 15/15 |

Swin active-grid occupancy after successive merge boundaries (S1 is before the first merge):
- mnist, clean: S1: 31.0-38.1% (5/5); S2: 44.6-50.6% (5/5); S3: 56.0-61.5% (5/5); S4: 99.7-99.8% (5/5).
- mnist, corrupted: S1: 34.7-58.5% (5/5); S2: 55.7-69.1% (5/5); S3: 69.1-75.9% (5/5); S4: 99.4-99.6% (5/5).
- cifar10, clean: S1: 43.2-100.0% (15/15); S2: 75.0-100.0% (15/15); S3: 97.5-100.0% (15/15); S4: 100.0-100.0% (15/15).
- cifar10, corrupted: S1: 44.0-100.0% (15/15); S2: 77.0-100.0% (15/15); S3: 98.0-100.0% (15/15); S4: 100.0-100.0% (15/15).
Occupancy approaching 100% locates stages becoming dense; an observed range is retained rather than assigning one density threshold to every image.

**Measured online gains at batches 1 and 8.** Speedups below are raw median latency divided by method median latency, paired by seed, input group, batch, and matched hardware class. Ranges include all active contrast conditions for the dataset, separately for GPU-resident and pinned-host input.

| Dataset / architecture | Scope / batch / input / hardware | Mean paired speedup range | Coverage and seed agreement |
|---|---|---|---|
| mnist / vit_small | gpu_raw_to_logits / B1 / clean / 4ba79ef9f76f | 0.924-0.983x | 6/6; faster all three 0, slower all three 6 |
| mnist / vit_small | gpu_raw_to_logits / B1 / balanced_corruption / 4ba79ef9f76f | 0.921-0.982x | 6/6; faster all three 0, slower all three 6 |
| mnist / vit_small | gpu_raw_to_logits / B8 / clean / 4ba79ef9f76f | 0.918-0.979x | 6/6; faster all three 0, slower all three 6 |
| mnist / vit_small | gpu_raw_to_logits / B8 / balanced_corruption / 4ba79ef9f76f | 0.918-0.979x | 6/6; faster all three 0, slower all three 6 |
| mnist / vit_small | host_raw_to_logits / B1 / clean / 4ba79ef9f76f | 0.927-0.985x | 6/6; faster all three 0, slower all three 6 |
| mnist / vit_small | host_raw_to_logits / B1 / balanced_corruption / 4ba79ef9f76f | 0.922-0.982x | 6/6; faster all three 0, slower all three 6 |
| mnist / vit_small | host_raw_to_logits / B8 / clean / 4ba79ef9f76f | 0.920-0.980x | 6/6; faster all three 0, slower all three 6 |
| mnist / vit_small | host_raw_to_logits / B8 / balanced_corruption / 4ba79ef9f76f | 0.917-0.978x | 6/6; faster all three 0, slower all three 6 |
| mnist / swin_tiny | gpu_raw_to_logits / B1 / clean / 4ba79ef9f76f | 0.692-0.984x | 6/6; faster all three 0, slower all three 6 |
| mnist / swin_tiny | gpu_raw_to_logits / B1 / balanced_corruption / 4ba79ef9f76f | 0.673-0.986x | 6/6; faster all three 0, slower all three 6 |
| mnist / swin_tiny | gpu_raw_to_logits / B8 / clean / 4ba79ef9f76f | 0.275-0.989x | 6/6; faster all three 0, slower all three 6 |
| mnist / swin_tiny | gpu_raw_to_logits / B8 / balanced_corruption / 4ba79ef9f76f | 0.275-0.994x | 6/6; faster all three 0, slower all three 6 |
| mnist / swin_tiny | host_raw_to_logits / B1 / clean / 4ba79ef9f76f | 0.694-0.986x | 6/6; faster all three 0, slower all three 6 |
| mnist / swin_tiny | host_raw_to_logits / B1 / balanced_corruption / 4ba79ef9f76f | 0.673-0.986x | 6/6; faster all three 0, slower all three 6 |
| mnist / swin_tiny | host_raw_to_logits / B8 / clean / 4ba79ef9f76f | 0.275-0.990x | 6/6; faster all three 0, slower all three 6 |
| mnist / swin_tiny | host_raw_to_logits / B8 / balanced_corruption / 4ba79ef9f76f | 0.275-0.993x | 6/6; faster all three 0, slower all three 6 |
| cifar10 / vit_small | gpu_raw_to_logits / B1 / clean / 4ba79ef9f76f | 0.917-0.986x | 18/18; faster all three 0, slower all three 18 |
| cifar10 / vit_small | gpu_raw_to_logits / B1 / balanced_corruption / 4ba79ef9f76f | 0.913-0.988x | 18/18; faster all three 0, slower all three 18 |
| cifar10 / vit_small | gpu_raw_to_logits / B8 / clean / 4ba79ef9f76f | 0.946-1.021x | 18/18; faster all three 2, slower all three 12 |
| cifar10 / vit_small | gpu_raw_to_logits / B8 / balanced_corruption / 4ba79ef9f76f | 0.892-0.999x | 18/18; faster all three 0, slower all three 17 |
| cifar10 / vit_small | host_raw_to_logits / B1 / clean / 4ba79ef9f76f | 0.915-0.987x | 18/18; faster all three 0, slower all three 18 |
| cifar10 / vit_small | host_raw_to_logits / B1 / balanced_corruption / 4ba79ef9f76f | 0.914-0.989x | 18/18; faster all three 0, slower all three 18 |
| cifar10 / vit_small | host_raw_to_logits / B8 / clean / 4ba79ef9f76f | 0.947-1.025x | 18/18; faster all three 4, slower all three 10 |
| cifar10 / vit_small | host_raw_to_logits / B8 / balanced_corruption / 4ba79ef9f76f | 0.893-0.999x | 18/18; faster all three 0, slower all three 17 |
| cifar10 / swin_tiny | gpu_raw_to_logits / B1 / clean / 4ba79ef9f76f | 0.518-0.996x | 18/18; faster all three 0, slower all three 17 |
| cifar10 / swin_tiny | gpu_raw_to_logits / B1 / balanced_corruption / 4ba79ef9f76f | 0.508-1.002x | 18/18; faster all three 0, slower all three 17 |
| cifar10 / swin_tiny | gpu_raw_to_logits / B8 / clean / 4ba79ef9f76f | 0.336-1.000x | 18/18; faster all three 0, slower all three 15 |
| cifar10 / swin_tiny | gpu_raw_to_logits / B8 / balanced_corruption / 4ba79ef9f76f | 0.335-0.998x | 18/18; faster all three 0, slower all three 16 |
| cifar10 / swin_tiny | host_raw_to_logits / B1 / clean / 4ba79ef9f76f | 0.519-0.996x | 18/18; faster all three 0, slower all three 17 |
| cifar10 / swin_tiny | host_raw_to_logits / B1 / balanced_corruption / 4ba79ef9f76f | 0.508-1.002x | 18/18; faster all three 0, slower all three 16 |
| cifar10 / swin_tiny | host_raw_to_logits / B8 / clean / 4ba79ef9f76f | 0.336-1.000x | 18/18; faster all three 0, slower all three 15 |
| cifar10 / swin_tiny | host_raw_to_logits / B8 / balanced_corruption / 4ba79ef9f76f | 0.336-0.997x | 18/18; faster all three 0, slower all three 16 |

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
- [latency_7_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/vit_small/latency_7_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/vit_small/latency_7_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/vit_small/latency_7_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/vit_small/latency_7_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/vit_small/latency_7_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/vit_small/latency_7_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/vit_small/latency_7_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/vit_small/latency_7_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/vit_small/latency_7_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/vit_small/latency_7_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/vit_small/latency_7_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/vit_small/latency_7_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/vit_small/latency_7_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/vit_small/latency_7_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/vit_small/latency_7_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/vit_small/latency_7_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/vit_small/latency_7_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/vit_small/latency_7_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/vit_small/latency_7_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/vit_small/latency_7_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/vit_small/latency_7_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/vit_small/latency_7_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/vit_small/latency_7_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/vit_small/latency_7_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [median_ms_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/median_ms_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_clean_accuracy_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_mCE_raw_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/median_ms_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_clean_accuracy_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_mCE_raw_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_clean_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/median_ms_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_clean_accuracy_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_mCE_raw_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/median_ms_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_clean_accuracy_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_mCE_raw_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_clean_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/median_ms_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_clean_accuracy_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_mCE_raw_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/median_ms_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_clean_accuracy_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_mCE_raw_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/median_ms_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_clean_accuracy_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_mCE_raw_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/median_ms_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_clean_accuracy_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/tradeoff_mCE_raw_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b1_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/stages_profiler_diagnostic_b1_clean_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b8_clean_4ba79ef9f76f](outputs/figures/mnist/vit_small/stages_profiler_diagnostic_b8_clean_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/stages_profiler_diagnostic_b1_balanced_corruption_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/vit_small/stages_profiler_diagnostic_b8_balanced_corruption_4ba79ef9f76f.png)
- [clean_accuracy](outputs/figures/mnist/swin_tiny/clean_accuracy.png)
- [mean_corruption_error](outputs/figures/mnist/swin_tiny/mean_corruption_error.png)
- [per_corruption_error](outputs/figures/mnist/swin_tiny/per_corruption_error.png)
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
- [support_achieved_zero_fraction_mean](outputs/figures/mnist/swin_tiny/support_achieved_zero_fraction_mean.png)
- [support_final_retained_per_image_mean](outputs/figures/mnist/swin_tiny/support_final_retained_per_image_mean.png)
- [support_initial_retained_per_image_mean](outputs/figures/mnist/swin_tiny/support_initial_retained_per_image_mean.png)
- [support_natural_zero_fraction_mean](outputs/figures/mnist/swin_tiny/support_natural_zero_fraction_mean.png)
- [support_retained_patches_mean](outputs/figures/mnist/swin_tiny/support_retained_patches_mean.png)
- [support_spatial_zero_fraction_mean](outputs/figures/mnist/swin_tiny/support_spatial_zero_fraction_mean.png)
- [support_stages_0_blocks_0_active_per_image_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_0_active_per_image_mean.png)
- [support_stages_0_blocks_0_active_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_0_active_windows_mean.png)
- [support_stages_0_blocks_0_executed_attention_pairs_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_0_executed_attention_pairs_mean.png)
- [support_stages_0_blocks_0_executed_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_0_executed_rows_mean.png)
- [support_stages_0_blocks_0_padding_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_0_padding_rows_mean.png)
- [support_stages_0_blocks_0_total_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_0_total_windows_mean.png)
- [support_stages_0_blocks_0_window_length_groups_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_0_window_length_groups_mean.png)
- [support_stages_0_blocks_0_window_lengths_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_0_window_lengths_mean.png)
- [support_stages_0_blocks_1_active_per_image_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_1_active_per_image_mean.png)
- [support_stages_0_blocks_1_active_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_1_active_windows_mean.png)
- [support_stages_0_blocks_1_executed_attention_pairs_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_1_executed_attention_pairs_mean.png)
- [support_stages_0_blocks_1_executed_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_1_executed_rows_mean.png)
- [support_stages_0_blocks_1_padding_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_1_padding_rows_mean.png)
- [support_stages_0_blocks_1_total_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_1_total_windows_mean.png)
- [support_stages_0_blocks_1_window_length_groups_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_1_window_length_groups_mean.png)
- [support_stages_0_blocks_1_window_lengths_mean](outputs/figures/mnist/swin_tiny/support_stages_0_blocks_1_window_lengths_mean.png)
- [support_stages_1_blocks_0_active_per_image_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_0_active_per_image_mean.png)
- [support_stages_1_blocks_0_active_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_0_active_windows_mean.png)
- [support_stages_1_blocks_0_executed_attention_pairs_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_0_executed_attention_pairs_mean.png)
- [support_stages_1_blocks_0_executed_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_0_executed_rows_mean.png)
- [support_stages_1_blocks_0_padding_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_0_padding_rows_mean.png)
- [support_stages_1_blocks_0_total_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_0_total_windows_mean.png)
- [support_stages_1_blocks_0_window_length_groups_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_0_window_length_groups_mean.png)
- [support_stages_1_blocks_0_window_lengths_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_0_window_lengths_mean.png)
- [support_stages_1_blocks_1_active_per_image_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_1_active_per_image_mean.png)
- [support_stages_1_blocks_1_active_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_1_active_windows_mean.png)
- [support_stages_1_blocks_1_executed_attention_pairs_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_1_executed_attention_pairs_mean.png)
- [support_stages_1_blocks_1_executed_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_1_executed_rows_mean.png)
- [support_stages_1_blocks_1_padding_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_1_padding_rows_mean.png)
- [support_stages_1_blocks_1_total_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_1_total_windows_mean.png)
- [support_stages_1_blocks_1_window_length_groups_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_1_window_length_groups_mean.png)
- [support_stages_1_blocks_1_window_lengths_mean](outputs/figures/mnist/swin_tiny/support_stages_1_blocks_1_window_lengths_mean.png)
- [support_stages_2_blocks_0_active_per_image_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_0_active_per_image_mean.png)
- [support_stages_2_blocks_0_active_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_0_active_windows_mean.png)
- [support_stages_2_blocks_0_executed_attention_pairs_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_0_executed_attention_pairs_mean.png)
- [support_stages_2_blocks_0_executed_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_0_executed_rows_mean.png)
- [support_stages_2_blocks_0_padding_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_0_padding_rows_mean.png)
- [support_stages_2_blocks_0_total_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_0_total_windows_mean.png)
- [support_stages_2_blocks_0_window_length_groups_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_0_window_length_groups_mean.png)
- [support_stages_2_blocks_0_window_lengths_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_0_window_lengths_mean.png)
- [support_stages_2_blocks_1_active_per_image_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_1_active_per_image_mean.png)
- [support_stages_2_blocks_1_active_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_1_active_windows_mean.png)
- [support_stages_2_blocks_1_executed_attention_pairs_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_1_executed_attention_pairs_mean.png)
- [support_stages_2_blocks_1_executed_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_1_executed_rows_mean.png)
- [support_stages_2_blocks_1_padding_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_1_padding_rows_mean.png)
- [support_stages_2_blocks_1_total_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_1_total_windows_mean.png)
- [support_stages_2_blocks_1_window_length_groups_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_1_window_length_groups_mean.png)
- [support_stages_2_blocks_1_window_lengths_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_1_window_lengths_mean.png)
- [support_stages_2_blocks_2_active_per_image_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_2_active_per_image_mean.png)
- [support_stages_2_blocks_2_active_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_2_active_windows_mean.png)
- [support_stages_2_blocks_2_executed_attention_pairs_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_2_executed_attention_pairs_mean.png)
- [support_stages_2_blocks_2_executed_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_2_executed_rows_mean.png)
- [support_stages_2_blocks_2_padding_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_2_padding_rows_mean.png)
- [support_stages_2_blocks_2_total_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_2_total_windows_mean.png)
- [support_stages_2_blocks_2_window_length_groups_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_2_window_length_groups_mean.png)
- [support_stages_2_blocks_2_window_lengths_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_2_window_lengths_mean.png)
- [support_stages_2_blocks_3_active_per_image_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_3_active_per_image_mean.png)
- [support_stages_2_blocks_3_active_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_3_active_windows_mean.png)
- [support_stages_2_blocks_3_executed_attention_pairs_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_3_executed_attention_pairs_mean.png)
- [support_stages_2_blocks_3_executed_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_3_executed_rows_mean.png)
- [support_stages_2_blocks_3_padding_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_3_padding_rows_mean.png)
- [support_stages_2_blocks_3_total_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_3_total_windows_mean.png)
- [support_stages_2_blocks_3_window_length_groups_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_3_window_length_groups_mean.png)
- [support_stages_2_blocks_3_window_lengths_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_3_window_lengths_mean.png)
- [support_stages_2_blocks_4_active_per_image_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_4_active_per_image_mean.png)
- [support_stages_2_blocks_4_active_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_4_active_windows_mean.png)
- [support_stages_2_blocks_4_executed_attention_pairs_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_4_executed_attention_pairs_mean.png)
- [support_stages_2_blocks_4_executed_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_4_executed_rows_mean.png)
- [support_stages_2_blocks_4_padding_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_4_padding_rows_mean.png)
- [support_stages_2_blocks_4_total_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_4_total_windows_mean.png)
- [support_stages_2_blocks_4_window_length_groups_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_4_window_length_groups_mean.png)
- [support_stages_2_blocks_4_window_lengths_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_4_window_lengths_mean.png)
- [support_stages_2_blocks_5_active_per_image_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_5_active_per_image_mean.png)
- [support_stages_2_blocks_5_active_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_5_active_windows_mean.png)
- [support_stages_2_blocks_5_executed_attention_pairs_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_5_executed_attention_pairs_mean.png)
- [support_stages_2_blocks_5_executed_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_5_executed_rows_mean.png)
- [support_stages_2_blocks_5_padding_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_5_padding_rows_mean.png)
- [support_stages_2_blocks_5_total_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_5_total_windows_mean.png)
- [support_stages_2_blocks_5_window_length_groups_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_5_window_length_groups_mean.png)
- [support_stages_2_blocks_5_window_lengths_mean](outputs/figures/mnist/swin_tiny/support_stages_2_blocks_5_window_lengths_mean.png)
- [support_stages_3_blocks_0_active_per_image_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_0_active_per_image_mean.png)
- [support_stages_3_blocks_0_active_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_0_active_windows_mean.png)
- [support_stages_3_blocks_0_executed_attention_pairs_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_0_executed_attention_pairs_mean.png)
- [support_stages_3_blocks_0_executed_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_0_executed_rows_mean.png)
- [support_stages_3_blocks_0_padding_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_0_padding_rows_mean.png)
- [support_stages_3_blocks_0_total_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_0_total_windows_mean.png)
- [support_stages_3_blocks_0_window_length_groups_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_0_window_length_groups_mean.png)
- [support_stages_3_blocks_0_window_lengths_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_0_window_lengths_mean.png)
- [support_stages_3_blocks_1_active_per_image_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_1_active_per_image_mean.png)
- [support_stages_3_blocks_1_active_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_1_active_windows_mean.png)
- [support_stages_3_blocks_1_executed_attention_pairs_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_1_executed_attention_pairs_mean.png)
- [support_stages_3_blocks_1_executed_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_1_executed_rows_mean.png)
- [support_stages_3_blocks_1_padding_rows_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_1_padding_rows_mean.png)
- [support_stages_3_blocks_1_total_windows_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_1_total_windows_mean.png)
- [support_stages_3_blocks_1_window_length_groups_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_1_window_length_groups_mean.png)
- [support_stages_3_blocks_1_window_lengths_mean](outputs/figures/mnist/swin_tiny/support_stages_3_blocks_1_window_lengths_mean.png)
- [latency_7_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/swin_tiny/latency_7_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/swin_tiny/latency_7_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/swin_tiny/latency_7_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/swin_tiny/latency_7_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/swin_tiny/latency_7_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/swin_tiny/latency_7_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/swin_tiny/latency_7_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/swin_tiny/latency_7_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/swin_tiny/latency_7_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/swin_tiny/latency_7_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/swin_tiny/latency_7_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/swin_tiny/latency_7_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/swin_tiny/latency_7_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/swin_tiny/latency_7_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/swin_tiny/latency_7_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/swin_tiny/latency_7_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/swin_tiny/latency_7_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/swin_tiny/latency_7_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/swin_tiny/latency_7_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/swin_tiny/latency_7_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/swin_tiny/latency_7_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/swin_tiny/latency_7_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_7_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/mnist/swin_tiny/latency_7_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_7_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/mnist/swin_tiny/latency_7_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [median_ms_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/median_ms_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_clean_accuracy_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_mCE_raw_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/median_ms_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_clean_accuracy_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_mCE_raw_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_clean_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/median_ms_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_clean_accuracy_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_mCE_raw_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/median_ms_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_clean_accuracy_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_mCE_raw_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_clean_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/median_ms_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_clean_accuracy_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_mCE_raw_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/median_ms_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_clean_accuracy_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_mCE_raw_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/median_ms_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_clean_accuracy_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_mCE_raw_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/median_ms_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_clean_accuracy_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/tradeoff_mCE_raw_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b1_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/stages_profiler_diagnostic_b1_clean_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b8_clean_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/stages_profiler_diagnostic_b8_clean_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/stages_profiler_diagnostic_b1_balanced_corruption_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/mnist/swin_tiny/stages_profiler_diagnostic_b8_balanced_corruption_4ba79ef9f76f.png)
- [clean_accuracy](outputs/figures/cifar10/vit_small/clean_accuracy.png)
- [mean_corruption_error](outputs/figures/cifar10/vit_small/mean_corruption_error.png)
- [per_corruption_error](outputs/figures/cifar10/vit_small/per_corruption_error.png)
- [severity_single_color](outputs/figures/cifar10/vit_small/severity_single_color.png)
- [severity_grayscale](outputs/figures/cifar10/vit_small/severity_grayscale.png)
- [severity_color_opponency](outputs/figures/cifar10/vit_small/severity_color_opponency.png)
- [learning_raw_dense_pNone_loss](outputs/figures/cifar10/vit_small/learning_raw_dense_pNone_loss.png)
- [learning_raw_dense_pNone_accuracy](outputs/figures/cifar10/vit_small/learning_raw_dense_pNone_accuracy.png)
- [learning_single_color_dense_p0_loss](outputs/figures/cifar10/vit_small/learning_single_color_dense_p0_loss.png)
- [learning_single_color_dense_p0_accuracy](outputs/figures/cifar10/vit_small/learning_single_color_dense_p0_accuracy.png)
- [learning_grayscale_dense_p0_loss](outputs/figures/cifar10/vit_small/learning_grayscale_dense_p0_loss.png)
- [learning_grayscale_dense_p0_accuracy](outputs/figures/cifar10/vit_small/learning_grayscale_dense_p0_accuracy.png)
- [learning_color_opponency_dense_p0_loss](outputs/figures/cifar10/vit_small/learning_color_opponency_dense_p0_loss.png)
- [learning_color_opponency_dense_p0_accuracy](outputs/figures/cifar10/vit_small/learning_color_opponency_dense_p0_accuracy.png)
- [learning_single_color_compact_p0_loss](outputs/figures/cifar10/vit_small/learning_single_color_compact_p0_loss.png)
- [learning_single_color_compact_p0_accuracy](outputs/figures/cifar10/vit_small/learning_single_color_compact_p0_accuracy.png)
- [learning_single_color_compact_p20_loss](outputs/figures/cifar10/vit_small/learning_single_color_compact_p20_loss.png)
- [learning_single_color_compact_p20_accuracy](outputs/figures/cifar10/vit_small/learning_single_color_compact_p20_accuracy.png)
- [learning_single_color_compact_p40_loss](outputs/figures/cifar10/vit_small/learning_single_color_compact_p40_loss.png)
- [learning_single_color_compact_p40_accuracy](outputs/figures/cifar10/vit_small/learning_single_color_compact_p40_accuracy.png)
- [learning_single_color_compact_p60_loss](outputs/figures/cifar10/vit_small/learning_single_color_compact_p60_loss.png)
- [learning_single_color_compact_p60_accuracy](outputs/figures/cifar10/vit_small/learning_single_color_compact_p60_accuracy.png)
- [learning_single_color_compact_p80_loss](outputs/figures/cifar10/vit_small/learning_single_color_compact_p80_loss.png)
- [learning_single_color_compact_p80_accuracy](outputs/figures/cifar10/vit_small/learning_single_color_compact_p80_accuracy.png)
- [learning_grayscale_compact_p0_loss](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p0_loss.png)
- [learning_grayscale_compact_p0_accuracy](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p0_accuracy.png)
- [learning_grayscale_compact_p20_loss](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p20_loss.png)
- [learning_grayscale_compact_p20_accuracy](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p20_accuracy.png)
- [learning_grayscale_compact_p40_loss](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p40_loss.png)
- [learning_grayscale_compact_p40_accuracy](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p40_accuracy.png)
- [learning_grayscale_compact_p60_loss](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p60_loss.png)
- [learning_grayscale_compact_p60_accuracy](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p60_accuracy.png)
- [learning_grayscale_compact_p80_loss](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p80_loss.png)
- [learning_grayscale_compact_p80_accuracy](outputs/figures/cifar10/vit_small/learning_grayscale_compact_p80_accuracy.png)
- [learning_color_opponency_compact_p0_loss](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p0_loss.png)
- [learning_color_opponency_compact_p0_accuracy](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p0_accuracy.png)
- [learning_color_opponency_compact_p20_loss](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p20_loss.png)
- [learning_color_opponency_compact_p20_accuracy](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p20_accuracy.png)
- [learning_color_opponency_compact_p40_loss](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p40_loss.png)
- [learning_color_opponency_compact_p40_accuracy](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p40_accuracy.png)
- [learning_color_opponency_compact_p60_loss](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p60_loss.png)
- [learning_color_opponency_compact_p60_accuracy](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p60_accuracy.png)
- [learning_color_opponency_compact_p80_loss](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p80_loss.png)
- [learning_color_opponency_compact_p80_accuracy](outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p80_accuracy.png)
- [support_achieved_zero_fraction_mean](outputs/figures/cifar10/vit_small/support_achieved_zero_fraction_mean.png)
- [support_executed_image_tokens_mean](outputs/figures/cifar10/vit_small/support_executed_image_tokens_mean.png)
- [support_executed_sequence_length_mean](outputs/figures/cifar10/vit_small/support_executed_sequence_length_mean.png)
- [support_natural_zero_fraction_mean](outputs/figures/cifar10/vit_small/support_natural_zero_fraction_mean.png)
- [support_padding_tokens_mean](outputs/figures/cifar10/vit_small/support_padding_tokens_mean.png)
- [support_retained_patches_mean](outputs/figures/cifar10/vit_small/support_retained_patches_mean.png)
- [support_retained_per_image_mean](outputs/figures/cifar10/vit_small/support_retained_per_image_mean.png)
- [support_spatial_zero_fraction_mean](outputs/figures/cifar10/vit_small/support_spatial_zero_fraction_mean.png)
- [latency_19_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/vit_small/latency_19_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/vit_small/latency_19_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/vit_small/latency_19_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/vit_small/latency_19_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/vit_small/latency_19_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/vit_small/latency_19_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/vit_small/latency_19_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/vit_small/latency_19_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/vit_small/latency_19_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/vit_small/latency_19_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/vit_small/latency_19_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/vit_small/latency_19_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/vit_small/latency_19_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/vit_small/latency_19_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/vit_small/latency_19_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/vit_small/latency_19_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/vit_small/latency_19_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/vit_small/latency_19_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/vit_small/latency_19_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/vit_small/latency_19_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/vit_small/latency_19_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/vit_small/latency_19_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/vit_small/latency_19_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/vit_small/latency_19_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_gpu_raw_to_logits_b8_clean_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_median_ms](outputs/figures/cifar10/vit_small/latency_19_gpu_raw_to_logits_b8_clean_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_median_ms.png)
- [latency_19_gpu_raw_to_logits_b8_clean_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_p95_ms](outputs/figures/cifar10/vit_small/latency_19_gpu_raw_to_logits_b8_clean_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_p95_ms.png)
- [latency_19_host_raw_to_logits_b8_clean_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_median_ms](outputs/figures/cifar10/vit_small/latency_19_host_raw_to_logits_b8_clean_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_median_ms.png)
- [latency_19_host_raw_to_logits_b8_clean_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_p95_ms](outputs/figures/cifar10/vit_small/latency_19_host_raw_to_logits_b8_clean_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_p95_ms.png)
- [latency_19_loader_to_cpu_prediction_b8_clean_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_median_ms](outputs/figures/cifar10/vit_small/latency_19_loader_to_cpu_prediction_b8_clean_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_median_ms.png)
- [latency_19_loader_to_cpu_prediction_b8_clean_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_p95_ms](outputs/figures/cifar10/vit_small/latency_19_loader_to_cpu_prediction_b8_clean_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_p95_ms.png)
- [latency_19_gpu_raw_to_logits_b8_balanced_corruption_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_median_ms](outputs/figures/cifar10/vit_small/latency_19_gpu_raw_to_logits_b8_balanced_corruption_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_median_ms.png)
- [latency_19_gpu_raw_to_logits_b8_balanced_corruption_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_p95_ms](outputs/figures/cifar10/vit_small/latency_19_gpu_raw_to_logits_b8_balanced_corruption_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_p95_ms.png)
- [latency_19_host_raw_to_logits_b8_balanced_corruption_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_median_ms](outputs/figures/cifar10/vit_small/latency_19_host_raw_to_logits_b8_balanced_corruption_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_median_ms.png)
- [latency_19_host_raw_to_logits_b8_balanced_corruption_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_p95_ms](outputs/figures/cifar10/vit_small/latency_19_host_raw_to_logits_b8_balanced_corruption_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_p95_ms.png)
- [latency_19_loader_to_cpu_prediction_b8_balanced_corruption_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_median_ms](outputs/figures/cifar10/vit_small/latency_19_loader_to_cpu_prediction_b8_balanced_corruption_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_median_ms.png)
- [latency_19_loader_to_cpu_prediction_b8_balanced_corruption_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_p95_ms](outputs/figures/cifar10/vit_small/latency_19_loader_to_cpu_prediction_b8_balanced_corruption_7df94a2abec9ef0568603c4e7869b4860929e5ea32db55feef4cd1821361a102_p95_ms.png)
- [median_ms_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/median_ms_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_clean_accuracy_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_mCE_raw_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/median_ms_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_clean_accuracy_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_mCE_raw_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_clean_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/median_ms_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_clean_accuracy_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_mCE_raw_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/median_ms_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_clean_accuracy_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_mCE_raw_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_clean_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/median_ms_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_clean_accuracy_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_mCE_raw_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/median_ms_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_clean_accuracy_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_mCE_raw_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/median_ms_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_clean_accuracy_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_mCE_raw_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/median_ms_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_clean_accuracy_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/tradeoff_mCE_raw_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b8_clean_7df94a2abec9](outputs/figures/cifar10/vit_small/median_ms_gpu_raw_to_logits_b8_clean_7df94a2abec9.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b8_clean_7df94a2abec9](outputs/figures/cifar10/vit_small/tradeoff_clean_accuracy_gpu_raw_to_logits_b8_clean_7df94a2abec9.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b8_clean_7df94a2abec9](outputs/figures/cifar10/vit_small/tradeoff_mCE_raw_gpu_raw_to_logits_b8_clean_7df94a2abec9.png)
- [median_ms_host_raw_to_logits_b8_clean_7df94a2abec9](outputs/figures/cifar10/vit_small/median_ms_host_raw_to_logits_b8_clean_7df94a2abec9.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b8_clean_7df94a2abec9](outputs/figures/cifar10/vit_small/tradeoff_clean_accuracy_host_raw_to_logits_b8_clean_7df94a2abec9.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b8_clean_7df94a2abec9](outputs/figures/cifar10/vit_small/tradeoff_mCE_raw_host_raw_to_logits_b8_clean_7df94a2abec9.png)
- [median_ms_gpu_raw_to_logits_b8_balanced_corruption_7df94a2abec9](outputs/figures/cifar10/vit_small/median_ms_gpu_raw_to_logits_b8_balanced_corruption_7df94a2abec9.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b8_balanced_corruption_7df94a2abec9](outputs/figures/cifar10/vit_small/tradeoff_clean_accuracy_gpu_raw_to_logits_b8_balanced_corruption_7df94a2abec9.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b8_balanced_corruption_7df94a2abec9](outputs/figures/cifar10/vit_small/tradeoff_mCE_raw_gpu_raw_to_logits_b8_balanced_corruption_7df94a2abec9.png)
- [median_ms_host_raw_to_logits_b8_balanced_corruption_7df94a2abec9](outputs/figures/cifar10/vit_small/median_ms_host_raw_to_logits_b8_balanced_corruption_7df94a2abec9.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b8_balanced_corruption_7df94a2abec9](outputs/figures/cifar10/vit_small/tradeoff_clean_accuracy_host_raw_to_logits_b8_balanced_corruption_7df94a2abec9.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b8_balanced_corruption_7df94a2abec9](outputs/figures/cifar10/vit_small/tradeoff_mCE_raw_host_raw_to_logits_b8_balanced_corruption_7df94a2abec9.png)
- [stages_profiler_diagnostic_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/stages_profiler_diagnostic_b1_clean_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/vit_small/stages_profiler_diagnostic_b8_clean_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b8_clean_7df94a2abec9](outputs/figures/cifar10/vit_small/stages_profiler_diagnostic_b8_clean_7df94a2abec9.png)
- [stages_profiler_diagnostic_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/stages_profiler_diagnostic_b1_balanced_corruption_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/vit_small/stages_profiler_diagnostic_b8_balanced_corruption_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b8_balanced_corruption_7df94a2abec9](outputs/figures/cifar10/vit_small/stages_profiler_diagnostic_b8_balanced_corruption_7df94a2abec9.png)
- [clean_accuracy](outputs/figures/cifar10/swin_tiny/clean_accuracy.png)
- [mean_corruption_error](outputs/figures/cifar10/swin_tiny/mean_corruption_error.png)
- [per_corruption_error](outputs/figures/cifar10/swin_tiny/per_corruption_error.png)
- [severity_single_color](outputs/figures/cifar10/swin_tiny/severity_single_color.png)
- [severity_grayscale](outputs/figures/cifar10/swin_tiny/severity_grayscale.png)
- [severity_color_opponency](outputs/figures/cifar10/swin_tiny/severity_color_opponency.png)
- [learning_raw_dense_pNone_loss](outputs/figures/cifar10/swin_tiny/learning_raw_dense_pNone_loss.png)
- [learning_raw_dense_pNone_accuracy](outputs/figures/cifar10/swin_tiny/learning_raw_dense_pNone_accuracy.png)
- [learning_single_color_dense_p0_loss](outputs/figures/cifar10/swin_tiny/learning_single_color_dense_p0_loss.png)
- [learning_single_color_dense_p0_accuracy](outputs/figures/cifar10/swin_tiny/learning_single_color_dense_p0_accuracy.png)
- [learning_grayscale_dense_p0_loss](outputs/figures/cifar10/swin_tiny/learning_grayscale_dense_p0_loss.png)
- [learning_grayscale_dense_p0_accuracy](outputs/figures/cifar10/swin_tiny/learning_grayscale_dense_p0_accuracy.png)
- [learning_color_opponency_dense_p0_loss](outputs/figures/cifar10/swin_tiny/learning_color_opponency_dense_p0_loss.png)
- [learning_color_opponency_dense_p0_accuracy](outputs/figures/cifar10/swin_tiny/learning_color_opponency_dense_p0_accuracy.png)
- [learning_single_color_compact_p0_loss](outputs/figures/cifar10/swin_tiny/learning_single_color_compact_p0_loss.png)
- [learning_single_color_compact_p0_accuracy](outputs/figures/cifar10/swin_tiny/learning_single_color_compact_p0_accuracy.png)
- [learning_single_color_compact_p20_loss](outputs/figures/cifar10/swin_tiny/learning_single_color_compact_p20_loss.png)
- [learning_single_color_compact_p20_accuracy](outputs/figures/cifar10/swin_tiny/learning_single_color_compact_p20_accuracy.png)
- [learning_single_color_compact_p40_loss](outputs/figures/cifar10/swin_tiny/learning_single_color_compact_p40_loss.png)
- [learning_single_color_compact_p40_accuracy](outputs/figures/cifar10/swin_tiny/learning_single_color_compact_p40_accuracy.png)
- [learning_single_color_compact_p60_loss](outputs/figures/cifar10/swin_tiny/learning_single_color_compact_p60_loss.png)
- [learning_single_color_compact_p60_accuracy](outputs/figures/cifar10/swin_tiny/learning_single_color_compact_p60_accuracy.png)
- [learning_single_color_compact_p80_loss](outputs/figures/cifar10/swin_tiny/learning_single_color_compact_p80_loss.png)
- [learning_single_color_compact_p80_accuracy](outputs/figures/cifar10/swin_tiny/learning_single_color_compact_p80_accuracy.png)
- [learning_grayscale_compact_p0_loss](outputs/figures/cifar10/swin_tiny/learning_grayscale_compact_p0_loss.png)
- [learning_grayscale_compact_p0_accuracy](outputs/figures/cifar10/swin_tiny/learning_grayscale_compact_p0_accuracy.png)
- [learning_grayscale_compact_p20_loss](outputs/figures/cifar10/swin_tiny/learning_grayscale_compact_p20_loss.png)
- [learning_grayscale_compact_p20_accuracy](outputs/figures/cifar10/swin_tiny/learning_grayscale_compact_p20_accuracy.png)
- [learning_grayscale_compact_p40_loss](outputs/figures/cifar10/swin_tiny/learning_grayscale_compact_p40_loss.png)
- [learning_grayscale_compact_p40_accuracy](outputs/figures/cifar10/swin_tiny/learning_grayscale_compact_p40_accuracy.png)
- [learning_grayscale_compact_p60_loss](outputs/figures/cifar10/swin_tiny/learning_grayscale_compact_p60_loss.png)
- [learning_grayscale_compact_p60_accuracy](outputs/figures/cifar10/swin_tiny/learning_grayscale_compact_p60_accuracy.png)
- [learning_grayscale_compact_p80_loss](outputs/figures/cifar10/swin_tiny/learning_grayscale_compact_p80_loss.png)
- [learning_grayscale_compact_p80_accuracy](outputs/figures/cifar10/swin_tiny/learning_grayscale_compact_p80_accuracy.png)
- [learning_color_opponency_compact_p0_loss](outputs/figures/cifar10/swin_tiny/learning_color_opponency_compact_p0_loss.png)
- [learning_color_opponency_compact_p0_accuracy](outputs/figures/cifar10/swin_tiny/learning_color_opponency_compact_p0_accuracy.png)
- [learning_color_opponency_compact_p20_loss](outputs/figures/cifar10/swin_tiny/learning_color_opponency_compact_p20_loss.png)
- [learning_color_opponency_compact_p20_accuracy](outputs/figures/cifar10/swin_tiny/learning_color_opponency_compact_p20_accuracy.png)
- [learning_color_opponency_compact_p40_loss](outputs/figures/cifar10/swin_tiny/learning_color_opponency_compact_p40_loss.png)
- [learning_color_opponency_compact_p40_accuracy](outputs/figures/cifar10/swin_tiny/learning_color_opponency_compact_p40_accuracy.png)
- [learning_color_opponency_compact_p60_loss](outputs/figures/cifar10/swin_tiny/learning_color_opponency_compact_p60_loss.png)
- [learning_color_opponency_compact_p60_accuracy](outputs/figures/cifar10/swin_tiny/learning_color_opponency_compact_p60_accuracy.png)
- [learning_color_opponency_compact_p80_loss](outputs/figures/cifar10/swin_tiny/learning_color_opponency_compact_p80_loss.png)
- [learning_color_opponency_compact_p80_accuracy](outputs/figures/cifar10/swin_tiny/learning_color_opponency_compact_p80_accuracy.png)
- [support_achieved_zero_fraction_mean](outputs/figures/cifar10/swin_tiny/support_achieved_zero_fraction_mean.png)
- [support_final_retained_per_image_mean](outputs/figures/cifar10/swin_tiny/support_final_retained_per_image_mean.png)
- [support_initial_retained_per_image_mean](outputs/figures/cifar10/swin_tiny/support_initial_retained_per_image_mean.png)
- [support_natural_zero_fraction_mean](outputs/figures/cifar10/swin_tiny/support_natural_zero_fraction_mean.png)
- [support_retained_patches_mean](outputs/figures/cifar10/swin_tiny/support_retained_patches_mean.png)
- [support_spatial_zero_fraction_mean](outputs/figures/cifar10/swin_tiny/support_spatial_zero_fraction_mean.png)
- [support_stages_0_blocks_0_active_per_image_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_0_active_per_image_mean.png)
- [support_stages_0_blocks_0_active_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_0_active_windows_mean.png)
- [support_stages_0_blocks_0_executed_attention_pairs_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_0_executed_attention_pairs_mean.png)
- [support_stages_0_blocks_0_executed_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_0_executed_rows_mean.png)
- [support_stages_0_blocks_0_padding_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_0_padding_rows_mean.png)
- [support_stages_0_blocks_0_total_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_0_total_windows_mean.png)
- [support_stages_0_blocks_0_window_length_groups_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_0_window_length_groups_mean.png)
- [support_stages_0_blocks_0_window_lengths_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_0_window_lengths_mean.png)
- [support_stages_0_blocks_1_active_per_image_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_1_active_per_image_mean.png)
- [support_stages_0_blocks_1_active_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_1_active_windows_mean.png)
- [support_stages_0_blocks_1_executed_attention_pairs_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_1_executed_attention_pairs_mean.png)
- [support_stages_0_blocks_1_executed_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_1_executed_rows_mean.png)
- [support_stages_0_blocks_1_padding_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_1_padding_rows_mean.png)
- [support_stages_0_blocks_1_total_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_1_total_windows_mean.png)
- [support_stages_0_blocks_1_window_length_groups_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_1_window_length_groups_mean.png)
- [support_stages_0_blocks_1_window_lengths_mean](outputs/figures/cifar10/swin_tiny/support_stages_0_blocks_1_window_lengths_mean.png)
- [support_stages_1_blocks_0_active_per_image_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_0_active_per_image_mean.png)
- [support_stages_1_blocks_0_active_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_0_active_windows_mean.png)
- [support_stages_1_blocks_0_executed_attention_pairs_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_0_executed_attention_pairs_mean.png)
- [support_stages_1_blocks_0_executed_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_0_executed_rows_mean.png)
- [support_stages_1_blocks_0_padding_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_0_padding_rows_mean.png)
- [support_stages_1_blocks_0_total_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_0_total_windows_mean.png)
- [support_stages_1_blocks_0_window_length_groups_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_0_window_length_groups_mean.png)
- [support_stages_1_blocks_0_window_lengths_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_0_window_lengths_mean.png)
- [support_stages_1_blocks_1_active_per_image_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_1_active_per_image_mean.png)
- [support_stages_1_blocks_1_active_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_1_active_windows_mean.png)
- [support_stages_1_blocks_1_executed_attention_pairs_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_1_executed_attention_pairs_mean.png)
- [support_stages_1_blocks_1_executed_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_1_executed_rows_mean.png)
- [support_stages_1_blocks_1_padding_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_1_padding_rows_mean.png)
- [support_stages_1_blocks_1_total_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_1_total_windows_mean.png)
- [support_stages_1_blocks_1_window_length_groups_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_1_window_length_groups_mean.png)
- [support_stages_1_blocks_1_window_lengths_mean](outputs/figures/cifar10/swin_tiny/support_stages_1_blocks_1_window_lengths_mean.png)
- [support_stages_2_blocks_0_active_per_image_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_0_active_per_image_mean.png)
- [support_stages_2_blocks_0_active_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_0_active_windows_mean.png)
- [support_stages_2_blocks_0_executed_attention_pairs_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_0_executed_attention_pairs_mean.png)
- [support_stages_2_blocks_0_executed_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_0_executed_rows_mean.png)
- [support_stages_2_blocks_0_padding_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_0_padding_rows_mean.png)
- [support_stages_2_blocks_0_total_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_0_total_windows_mean.png)
- [support_stages_2_blocks_0_window_length_groups_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_0_window_length_groups_mean.png)
- [support_stages_2_blocks_0_window_lengths_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_0_window_lengths_mean.png)
- [support_stages_2_blocks_1_active_per_image_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_1_active_per_image_mean.png)
- [support_stages_2_blocks_1_active_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_1_active_windows_mean.png)
- [support_stages_2_blocks_1_executed_attention_pairs_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_1_executed_attention_pairs_mean.png)
- [support_stages_2_blocks_1_executed_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_1_executed_rows_mean.png)
- [support_stages_2_blocks_1_padding_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_1_padding_rows_mean.png)
- [support_stages_2_blocks_1_total_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_1_total_windows_mean.png)
- [support_stages_2_blocks_1_window_length_groups_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_1_window_length_groups_mean.png)
- [support_stages_2_blocks_1_window_lengths_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_1_window_lengths_mean.png)
- [support_stages_2_blocks_2_active_per_image_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_2_active_per_image_mean.png)
- [support_stages_2_blocks_2_active_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_2_active_windows_mean.png)
- [support_stages_2_blocks_2_executed_attention_pairs_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_2_executed_attention_pairs_mean.png)
- [support_stages_2_blocks_2_executed_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_2_executed_rows_mean.png)
- [support_stages_2_blocks_2_padding_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_2_padding_rows_mean.png)
- [support_stages_2_blocks_2_total_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_2_total_windows_mean.png)
- [support_stages_2_blocks_2_window_length_groups_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_2_window_length_groups_mean.png)
- [support_stages_2_blocks_2_window_lengths_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_2_window_lengths_mean.png)
- [support_stages_2_blocks_3_active_per_image_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_3_active_per_image_mean.png)
- [support_stages_2_blocks_3_active_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_3_active_windows_mean.png)
- [support_stages_2_blocks_3_executed_attention_pairs_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_3_executed_attention_pairs_mean.png)
- [support_stages_2_blocks_3_executed_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_3_executed_rows_mean.png)
- [support_stages_2_blocks_3_padding_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_3_padding_rows_mean.png)
- [support_stages_2_blocks_3_total_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_3_total_windows_mean.png)
- [support_stages_2_blocks_3_window_length_groups_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_3_window_length_groups_mean.png)
- [support_stages_2_blocks_3_window_lengths_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_3_window_lengths_mean.png)
- [support_stages_2_blocks_4_active_per_image_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_4_active_per_image_mean.png)
- [support_stages_2_blocks_4_active_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_4_active_windows_mean.png)
- [support_stages_2_blocks_4_executed_attention_pairs_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_4_executed_attention_pairs_mean.png)
- [support_stages_2_blocks_4_executed_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_4_executed_rows_mean.png)
- [support_stages_2_blocks_4_padding_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_4_padding_rows_mean.png)
- [support_stages_2_blocks_4_total_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_4_total_windows_mean.png)
- [support_stages_2_blocks_4_window_length_groups_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_4_window_length_groups_mean.png)
- [support_stages_2_blocks_4_window_lengths_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_4_window_lengths_mean.png)
- [support_stages_2_blocks_5_active_per_image_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_5_active_per_image_mean.png)
- [support_stages_2_blocks_5_active_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_5_active_windows_mean.png)
- [support_stages_2_blocks_5_executed_attention_pairs_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_5_executed_attention_pairs_mean.png)
- [support_stages_2_blocks_5_executed_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_5_executed_rows_mean.png)
- [support_stages_2_blocks_5_padding_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_5_padding_rows_mean.png)
- [support_stages_2_blocks_5_total_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_5_total_windows_mean.png)
- [support_stages_2_blocks_5_window_length_groups_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_5_window_length_groups_mean.png)
- [support_stages_2_blocks_5_window_lengths_mean](outputs/figures/cifar10/swin_tiny/support_stages_2_blocks_5_window_lengths_mean.png)
- [support_stages_3_blocks_0_active_per_image_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_0_active_per_image_mean.png)
- [support_stages_3_blocks_0_active_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_0_active_windows_mean.png)
- [support_stages_3_blocks_0_executed_attention_pairs_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_0_executed_attention_pairs_mean.png)
- [support_stages_3_blocks_0_executed_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_0_executed_rows_mean.png)
- [support_stages_3_blocks_0_padding_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_0_padding_rows_mean.png)
- [support_stages_3_blocks_0_total_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_0_total_windows_mean.png)
- [support_stages_3_blocks_0_window_length_groups_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_0_window_length_groups_mean.png)
- [support_stages_3_blocks_0_window_lengths_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_0_window_lengths_mean.png)
- [support_stages_3_blocks_1_active_per_image_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_1_active_per_image_mean.png)
- [support_stages_3_blocks_1_active_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_1_active_windows_mean.png)
- [support_stages_3_blocks_1_executed_attention_pairs_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_1_executed_attention_pairs_mean.png)
- [support_stages_3_blocks_1_executed_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_1_executed_rows_mean.png)
- [support_stages_3_blocks_1_padding_rows_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_1_padding_rows_mean.png)
- [support_stages_3_blocks_1_total_windows_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_1_total_windows_mean.png)
- [support_stages_3_blocks_1_window_length_groups_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_1_window_length_groups_mean.png)
- [support_stages_3_blocks_1_window_lengths_mean](outputs/figures/cifar10/swin_tiny/support_stages_3_blocks_1_window_lengths_mean.png)
- [latency_19_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/swin_tiny/latency_19_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/swin_tiny/latency_19_gpu_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/swin_tiny/latency_19_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/swin_tiny/latency_19_host_raw_to_logits_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/swin_tiny/latency_19_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/swin_tiny/latency_19_loader_to_cpu_prediction_b1_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/swin_tiny/latency_19_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/swin_tiny/latency_19_gpu_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/swin_tiny/latency_19_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/swin_tiny/latency_19_host_raw_to_logits_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/swin_tiny/latency_19_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/swin_tiny/latency_19_loader_to_cpu_prediction_b8_clean_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/swin_tiny/latency_19_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/swin_tiny/latency_19_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/swin_tiny/latency_19_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/swin_tiny/latency_19_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/swin_tiny/latency_19_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/swin_tiny/latency_19_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/swin_tiny/latency_19_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/swin_tiny/latency_19_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/swin_tiny/latency_19_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/swin_tiny/latency_19_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [latency_19_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms](outputs/figures/cifar10/swin_tiny/latency_19_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_median_ms.png)
- [latency_19_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms](outputs/figures/cifar10/swin_tiny/latency_19_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76ff06df88f37ba7586b1b5a3ac8b43008b3eed0ca6a744393e6687_p95_ms.png)
- [median_ms_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/median_ms_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_clean_accuracy_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_mCE_raw_gpu_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/median_ms_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_clean_accuracy_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_mCE_raw_host_raw_to_logits_b1_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_clean_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/median_ms_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_clean_accuracy_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_mCE_raw_gpu_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/median_ms_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_clean_accuracy_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_mCE_raw_host_raw_to_logits_b8_clean_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_clean_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/median_ms_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_clean_accuracy_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_mCE_raw_gpu_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/median_ms_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_clean_accuracy_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_mCE_raw_host_raw_to_logits_b1_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b1_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/median_ms_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/paired_seed_ratio_of_median_speedup_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_clean_accuracy_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_mCE_raw_gpu_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [median_ms_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/median_ms_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/paired_seed_ratio_of_median_speedup_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_clean_accuracy_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_clean_accuracy_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [tradeoff_mCE_raw_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/tradeoff_mCE_raw_host_raw_to_logits_b8_balanced_corruption_4ba79ef9f76f.png)
- [paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/paired_seed_ratio_of_median_speedup_loader_to_cpu_prediction_b8_balanced_corruption_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b1_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/stages_profiler_diagnostic_b1_clean_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b8_clean_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/stages_profiler_diagnostic_b8_clean_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b1_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/stages_profiler_diagnostic_b1_balanced_corruption_4ba79ef9f76f.png)
- [stages_profiler_diagnostic_b8_balanced_corruption_4ba79ef9f76f](outputs/figures/cifar10/swin_tiny/stages_profiler_diagnostic_b8_balanced_corruption_4ba79ef9f76f.png)
