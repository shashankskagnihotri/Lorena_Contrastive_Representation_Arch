# Cross-study overviews

All accuracy panels use all three seeds and show mean ± sample SD. Protocol: heldout_val; train N=55,000 MNIST and45,000 CIFAR-10. Each official test cell has10,000 images. mCE means unnormalized macro corruption error.

The original dense result and compact-test0% result use the SAME checkpoint. Naturally empty patches can disappear at0%, changing attention/pooling. Dense0-trained and compact0-trained checkpoint families stay separate. Orange matched-sparsity curves are separately trained parent models. Raw pixel sparsification is separate from native contrast sparsification.

Latency is partial and uses only sealed full workers with all three matching seeds, one frozen A6000 hardware/CPU/thread class, and equal panel hashes. Blank heatmap cells mean incomplete coverage, never zero latency. Raw JSONL timings were not rehashed for these overviews.

[All seed metrics](accuracy_seeds.csv) · [Condition means/SD](accuracy_conditions.csv) · [Paired0/40/60/80 comparisons](cross_study_paired_conditions_0_40_60_80.csv) · [Provenance](provenance.json)

- accuracy_mnist_vit_small_raw: [PNG](accuracy_mnist_vit_small_raw.png) · [PDF](accuracy_mnist_vit_small_raw.pdf) · [SVG](accuracy_mnist_vit_small_raw.svg) · [CSV](accuracy_mnist_vit_small_raw.csv)
- accuracy_mnist_vit_small_grayscale: [PNG](accuracy_mnist_vit_small_grayscale.png) · [PDF](accuracy_mnist_vit_small_grayscale.pdf) · [SVG](accuracy_mnist_vit_small_grayscale.svg) · [CSV](accuracy_mnist_vit_small_grayscale.csv)
- accuracy_mnist_swin_tiny_raw: [PNG](accuracy_mnist_swin_tiny_raw.png) · [PDF](accuracy_mnist_swin_tiny_raw.pdf) · [SVG](accuracy_mnist_swin_tiny_raw.svg) · [CSV](accuracy_mnist_swin_tiny_raw.csv)
- accuracy_mnist_swin_tiny_grayscale: [PNG](accuracy_mnist_swin_tiny_grayscale.png) · [PDF](accuracy_mnist_swin_tiny_grayscale.pdf) · [SVG](accuracy_mnist_swin_tiny_grayscale.svg) · [CSV](accuracy_mnist_swin_tiny_grayscale.csv)
- accuracy_cifar10_vit_small_raw: [PNG](accuracy_cifar10_vit_small_raw.png) · [PDF](accuracy_cifar10_vit_small_raw.pdf) · [SVG](accuracy_cifar10_vit_small_raw.svg) · [CSV](accuracy_cifar10_vit_small_raw.csv)
- accuracy_cifar10_vit_small_single_color: [PNG](accuracy_cifar10_vit_small_single_color.png) · [PDF](accuracy_cifar10_vit_small_single_color.pdf) · [SVG](accuracy_cifar10_vit_small_single_color.svg) · [CSV](accuracy_cifar10_vit_small_single_color.csv)
- accuracy_cifar10_vit_small_grayscale: [PNG](accuracy_cifar10_vit_small_grayscale.png) · [PDF](accuracy_cifar10_vit_small_grayscale.pdf) · [SVG](accuracy_cifar10_vit_small_grayscale.svg) · [CSV](accuracy_cifar10_vit_small_grayscale.csv)
- accuracy_cifar10_vit_small_color_opponency: [PNG](accuracy_cifar10_vit_small_color_opponency.png) · [PDF](accuracy_cifar10_vit_small_color_opponency.pdf) · [SVG](accuracy_cifar10_vit_small_color_opponency.svg) · [CSV](accuracy_cifar10_vit_small_color_opponency.csv)
- accuracy_cifar10_swin_tiny_raw: [PNG](accuracy_cifar10_swin_tiny_raw.png) · [PDF](accuracy_cifar10_swin_tiny_raw.pdf) · [SVG](accuracy_cifar10_swin_tiny_raw.svg) · [CSV](accuracy_cifar10_swin_tiny_raw.csv)
- accuracy_cifar10_swin_tiny_single_color: [PNG](accuracy_cifar10_swin_tiny_single_color.png) · [PDF](accuracy_cifar10_swin_tiny_single_color.pdf) · [SVG](accuracy_cifar10_swin_tiny_single_color.svg) · [CSV](accuracy_cifar10_swin_tiny_single_color.csv)
- accuracy_cifar10_swin_tiny_grayscale: [PNG](accuracy_cifar10_swin_tiny_grayscale.png) · [PDF](accuracy_cifar10_swin_tiny_grayscale.pdf) · [SVG](accuracy_cifar10_swin_tiny_grayscale.svg) · [CSV](accuracy_cifar10_swin_tiny_grayscale.csv)
- accuracy_cifar10_swin_tiny_color_opponency: [PNG](accuracy_cifar10_swin_tiny_color_opponency.png) · [PDF](accuracy_cifar10_swin_tiny_color_opponency.pdf) · [SVG](accuracy_cifar10_swin_tiny_color_opponency.svg) · [CSV](accuracy_cifar10_swin_tiny_color_opponency.csv)
- partial_clean_latency_batch1: [PNG](partial_clean_latency_batch1.png) · [PDF](partial_clean_latency_batch1.pdf) · [SVG](partial_clean_latency_batch1.svg) · [CSV](partial_clean_latency_batch1.csv)
- partial_clean_latency_batch8: [PNG](partial_clean_latency_batch8.png) · [PDF](partial_clean_latency_batch8.pdf) · [SVG](partial_clean_latency_batch8.svg) · [CSV](partial_clean_latency_batch8.csv)
